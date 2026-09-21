"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 
89233218CNA000001 for Los Alamos National Laboratory (LANL), 
which is operated by Triad National Security, LLC for the 
U.S. Department of Energy/National Nuclear Security 
Administration. All rights in the program are reserved by 
Triad National Security, LLC, and the U.S. Department of 
Energy/National Nuclear Security Administration. The 
Government is granted for itself and others acting on its 
behalf a nonexclusive, paid-up, irrevocable worldwide 
license in this material to reproduce, prepare. derivative 
works, distribute copies to the public, perform publicly and 
display publicly, and to permit others to do so.

Tremors Agent : Agent for FDSN Service Checking and Metadata Retrieval

[T]ext-[R]eferenced [E]vent [M]apping & [O]utput [R]enderer for [Seismographs]

Current Authors: Ryley Hill, Richard Alfaro-Diaz, Christopher W. Johnson
Email: rghill@lanl.gov, rad@lanl.gov, cwj@lanl.gov 
tremors_agent.py
================
LangGraph-based agent for querying FDSN seismic datacenters, retrieving
earthquake catalogs, waveforms, and producing publication-quality maps /
timelines.

Architecture
------------
TremorsAgent
 └─ LangGraph StateGraph
      ├─ plan_query               – LLM parses NL query → search_params dict
      ├─ query_cascade            – Fan-out across global+regional DCs, merge & deduplicate
      ├─ retrieve_waveforms       – Per-event waveform download (event mode)
      ├─ retrieve_continuous_waveforms – Bulk continuous download (inventory-driven)
      ├─ plot_results             – Map + timeline figures
      ├─ plot_waveforms           – Per-event waveform figures
      └─ plot_continuous_waveforms – Continuous waveform figure

Supporting classes
------------------
DailyBulkWaveforms  – Builds chunked FDSN bulk-request task list from a
                      stations file or injected request list.
Scheduler           – Multiprocessing fan-out for bulk tasks.
PullWave            – Worker process: fetches one bulk chunk and writes
                      MiniSEED files.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import json
import logging
import multiprocessing
import os
import sys
import traceback
import typing
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Multiprocessing start-method
# ---------------------------------------------------------------------------
# A private context rather than multiprocessing.set_start_method(..., force=True):
# setting it globally mutates the *host* process's configuration, which a library
# has no business doing — a notebook or an application that embeds Tremors would
# silently have its own start method changed. Everything that spawns workers here
# goes through _MP_CTX instead.
#
# "forkserver" over "fork": forking a process that has already imported
# matplotlib/cartopy/obspy and may hold open FDSN sockets or thread state is
# unsafe, and CPython 3.14 warns about forking a multi-threaded process. The
# forkserver pays the import cost once in a clean single-threaded server process
# and forks children from there.
_MP_CTX = multiprocessing.get_context("forkserver")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import obspy
import obspy.clients.fdsn
import pandas as pd
import requests
from matplotlib.patches import Polygon
from obspy import UTCDateTime, Stream, read
from obspy.clients.fdsn import Client
from obspy.core.event import Catalog, Comment
from obspy.core.inventory import Inventory

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles
from pyproj import Geod

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from pydantic import ValidationError
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from typing import Annotated, TypedDict

from deepagents import (
    create_deep_agent,
    DeepAgentState,
    FilesystemMiddleware,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from deepagents.backends import StateBackend

from tremors.utils.geographic import (
    boundingbox,
    boundingradius,
    REGIONAL_RULES,
    add_north_arrow,
    add_scalebar,
)
from tremors.utils.params import SearchParams, verify_search_params
from tremors.utils.schema import catalog_to_kbcore, inventory_to_kbcore


# ---------------------------------------------------------------------------
# Well-known FDSN nodes
# Source: https://github.com/obspy/obspy/blob/main/obspy/clients/fdsn/header.py
#
# The former IRIS DMC is now EarthScope: service.iris.edu has been superseded by
# service.earthscope.org. IRIS and IRISPH5 are kept as aliases for the same
# endpoint (upstream obspy keeps them too), so plans and fallback lists naming
# the old short names keep resolving.
# ---------------------------------------------------------------------------
WELL_KNOWN_NODES: Dict[str, str] = {
    "AUSPASS":    "http://auspass.edu.au",
    "BGR":        "http://eida.bgr.de",
    "EARTHSCOPE": "https://service.earthscope.org",
    "EIDA":       "http://eida-federator.ethz.ch",
    "ETH":        "http://eida.ethz.ch",
    "EMSC":       "http://www.seismicportal.eu",
    "GEONET":     "http://service.geonet.org.nz",
    "GEOFON":     "http://geofon.gfz-potsdam.de",
    "GFZ":        "http://geofon.gfz-potsdam.de",
    "ICGC":       "http://ws.icgc.cat",
    "IESDMC":     "http://batsws.earth.sinica.edu.tw",
    "INGV":       "http://webservices.ingv.it",
    "IPGP":       "http://ws.ipgp.fr",
    "IRIS":       "https://service.earthscope.org",
    "IRISPH5":    "https://service.earthscope.org",
    "ISC":        "http://www.isc.ac.uk",
    "KNMI":       "http://rdsa.knmi.nl",
    "KOERI":      "http://eida.koeri.boun.edu.tr",
    "LMU":        "https://erde.geophysik.uni-muenchen.de",
    "NCEDC":      "https://service.ncedc.org",
    "NIEP":       "http://eida-sc3.infp.ro",
    "NOA":        "http://eida.gein.noa.gr",
    "NRCAN":      "https://earthquakescanada.nrcan.gc.ca",
    "ODC":        "http://www.orfeus-eu.org",
    "ORFEUS":     "http://www.orfeus-eu.org",
    "RESIF":      "http://ws.resif.fr",
    "RESIFPH5":   "http://ph5ws.resif.fr",
    "RASPISHAKE": "https://data.raspberryshake.org",
    "SCEDC":      "http://service.scedc.caltech.edu",
    "TEXNET":     "http://rtserve.beg.utexas.edu",
    "UIB-NORSAR": "http://eida.geo.uib.no",
    "USGS":       "http://earthquake.usgs.gov",
    "USP":        "http://sismo.iag.usp.br",
}

# Ordered fallback DCs for event waveform retrieval
_WAVEFORM_FALLBACK_DCS: List[str] = [
    "IRIS", "GEOFON", "NCEDC", "SCEDC", "RASPISHAKE", "EMSC"
]

# Ordered DCs tried for bulk inventory pulls (continuous mode)
_INVENTORY_DC_PRIORITY: List[str] = [
    "EARTHSCOPE", "IRIS", "GEOFON", "ODC"
]

# Upper bound on concurrent datacenter requests in the event cascade. The DC list
# is small (typically 4-6), and each entry is a different institution's public
# service, so this exists to keep the burst polite rather than to limit work.
_CASCADE_MAX_WORKERS = 6


def _render_param_key_block() -> str:
    """
    Render ``SearchParams`` as the prose key list used in the planner prompt.

    Generated from the model rather than written out by hand so there is a single
    source of truth: the prompt and the JSON schema the provider validates
    against can no longer drift apart as fields are added or re-described.
    """
    type_names = {str: "string", float: "float", int: "int", bool: "boolean"}
    lines: List[str] = []

    for name, field in SearchParams.model_fields.items():
        # Fields are Optional[T]; take T for the human-readable type name.
        args      = [a for a in typing.get_args(field.annotation) if a is not type(None)]
        type_name = type_names.get(args[0] if args else field.annotation, "value")
        lines.append(f"  {name:<24} {type_name:<8} {field.description or ''}".rstrip())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TremorsState(TypedDict):
    """Typed state bag shared across all LangGraph nodes."""

    query:      str   # User's natural-language request
    datacenter: str   # Primary datacenter hint (e.g. "ISC")

    # Per-service availability (populated if a pre-flight check is run)
    service_status: Dict[str, bool]

    # Parsed search parameters produced by the LLM planning node
    search_params: Optional[dict]

    # Output artefacts
    output_dir:                str
    metadata_tables:           Dict[str, str]   # table-name → parquet path
    plots:                     List[str]         # paths to saved map/timeline figures
    waveforms_saved:           List[str]         # paths to event MiniSEED files
    waveform_metadata:         Dict[str, str]    # table-name → parquet path (inventory)
    waveform_plots:            List[str]         # paths to per-event waveform figures

    # Continuous-mode artefacts
    continuous_waveforms_saved:  List[str]
    continuous_waveform_plots:   List[str]

    status:      str
    error:       Optional[str]
    queried_dcs: List[str]   # DCs that returned data


class TremorsDeepState(DeepAgentState, total=False):
    """
    deepagents state schema for the Tremors tool loop.

    Extends :class:`deepagents.DeepAgentState` (which supplies the
    ``messages`` / ``jump_to`` / ``structured_response`` channels used by the
    tool-calling agent and its middleware) with the Tremors pipeline payload
    keys formerly carried by :class:`TremorsState`.

    Every payload key is authored by exactly one tool, so the default
    last-write-wins reducer is correct and no custom reducers are needed.
    ``total=False`` lets the agent be invoked with only ``messages`` present;
    tools populate the remaining keys as the pipeline runs.
    """

    query:      str
    datacenter: str

    service_status: Dict[str, bool]
    search_params:  Optional[dict]

    output_dir:        str
    metadata_tables:   Dict[str, str]
    plots:             List[str]
    waveforms_saved:   List[str]
    waveform_metadata: Dict[str, str]
    waveform_plots:    List[str]

    continuous_waveforms_saved: List[str]
    continuous_waveform_plots:  List[str]

    status:      str
    error:       Optional[str]
    queried_dcs: List[str]

    # Verifier findings from the most recent plan_query, kept as a list rather
    # than only as the joined ``error`` string so a machine-readable caller
    # (``cli.py --json``) can report them individually. Authored solely by
    # plan_query, so last-write-wins stays correct.
    problems: List[str]


# Tremors payload keys projected into every tool's Command(update=...).
# Allowlist (not denylist): the legacy ``_*_node`` methods return ``{**state,
# ...}``, which would otherwise echo the injected ``messages`` channel back and
# double-append via its ``add_messages`` reducer. Projecting to only these keys
# also insulates us from other DeepAgentState reducer channels (``jump_to``,
# ``structured_response``).
_TREMORS_STATE_KEYS: Tuple[str, ...] = (
    "query",
    "datacenter",
    "service_status",
    "search_params",
    "metadata_tables",
    "plots",
    "waveforms_saved",
    "waveform_metadata",
    "waveform_plots",
    "continuous_waveforms_saved",
    "continuous_waveform_plots",
    "status",
    "error",
    "queried_dcs",
    "problems",
)


def _project_state(result: dict) -> dict:
    """Keep only known Tremors payload keys from a node's return dict."""
    return {k: result[k] for k in _TREMORS_STATE_KEYS if k in result}


def _apply_override(state: Any, params_override: Optional[dict]) -> dict:
    """
    Return a plain state dict with *params_override* merged over ``search_params``.

    This is how the "edit" decision at an approval gate takes effect. The gates
    are provided by ``HumanInTheLoopMiddleware``, which edits a tool's *arguments*
    — but the Tremors tools deliberately expose no arguments, reading their inputs
    from state instead. The gated tools therefore carry one otherwise-unused
    ``params_override`` argument for the middleware to write the user's edits
    into, and this merges it back onto the plan the tool actually runs.

    The override is shallow-merged, so an edit touches only the keys the user
    changed and leaves the rest of the verified plan intact.
    """
    merged = dict(state)
    if params_override:
        merged["search_params"] = {**(merged.get("search_params") or {}), **params_override}
        _log("TremorsAgent", f"Applied user override: {params_override}")
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fdsn_client(dc: str, timeout: int = 30) -> Optional[Client]:
    """
    Try to build an obspy FDSN Client for *dc* (short name or base URL).

    Tries the WELL_KNOWN_NODES URL first; falls back to passing the dc string
    directly to Client().  Returns None if both attempts fail.
    """
    url = WELL_KNOWN_NODES.get(dc)
    for target in ([url] if url else []) + [dc]:
        try:
            return Client(base_url=target, timeout=timeout) if "://" in str(target) \
                   else Client(target, timeout=timeout)
        except Exception:
            continue
    return None


def _log(name: str, msg: str) -> None:
    """Uniform prefix logging used throughout the agent.

    Writes to **stderr**, not stdout. ``cli.py --json`` puts a single machine-
    readable document on stdout, so every human/progress line has to leave by the
    other channel or it corrupts the payload. In a terminal both streams still
    land on the tty, so interactive output is unchanged.
    """
    print(f"[{name}] {msg}", file=sys.stderr)


def approve_all(payload: dict) -> dict:
    """
    ``on_interrupt`` callback that approves every approval gate.

    Pass this to :meth:`TremorsAgent.run` for an unattended script that should
    still exercise the gates::

        result = agent.run(query, on_interrupt=approve_all)

    Prefer it over a hand-written ``lambda``: the resume value must carry
    **exactly one decision per pending tool call** (the middleware raises
    ``ValueError`` otherwise), and the two gate kinds take different shapes — a
    literal ``{"decisions": [{"type": "approve"}]}`` would be handed to a
    *clarification* pause as its answer text.

    A clarification is deliberately **not** auto-answerable: approving a plan
    the verifier called unusable is not a decision a default gets to make. This
    raises :class:`ValueError` naming the problems instead, mirroring the CLI's
    refusal to answer one under ``--yes``. Handle
    ``payload["kind"] == "clarification"`` yourself if the caller can supply the
    missing detail.

    Parameters
    ----------
    payload:
        The pending interrupt payload handed to the callback.

    Returns
    -------
    dict
        ``{"decisions": [...]}``, one ``approve`` per pending request.
    """
    if payload.get("kind") == "clarification":
        problems = "; ".join(payload.get("problems") or []) or "unspecified"
        raise ValueError(
            "Tremors needs a clarification, which cannot be auto-approved: "
            f"{problems}. Add the missing detail to the query, or supply an "
            "on_interrupt that answers payload['kind'] == 'clarification'."
        )
    count = len(payload.get("action_requests") or ())
    return {"decisions": [{"type": "approve"} for _ in range(max(count, 1))]}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TremorsAgent:
    """
    deepagents agent for FDSN seismic data retrieval and visualization.

    Rather than a hard-coded LangGraph ``StateGraph``, the Tremors pipeline is
    exposed to an LLM as a set of tools (one per former graph node) and driven
    by :func:`deepagents.create_deep_agent`.  The model orchestrates the
    sequence — planning a query, cascading across datacenters, retrieving
    waveforms, and plotting — guided by :attr:`_SYSTEM_PROMPT` and by the
    explicit next-step hint each tool returns.

    Because the model chooses the order, the guardrails are placed where they do
    not depend on it choosing well:

    * planning uses provider-enforced structured output plus a domain verifier,
      and **fails closed** (see :meth:`_plan_query_node`);
    * three human-in-the-loop gates pause the run before anything expensive or
      irreversible — a clarification pause when the plan fails verification, and
      an approve/edit/reject gate in front of each FDSN entry point.

    Parameters
    ----------
    llm:
        Any LangChain-compatible chat model (e.g. ``ChatOpenAI`` pointed at a
        local Ollama server, or the OpenAI API).
    output_dir:
        Directory where all output files (parquet, plots, MiniSEED) are
        written.  Nodes read ``self.output_dir`` directly, so a query needs to
        supply only its natural-language text.
    checkpointer:
        LangGraph checkpointer passed through to ``create_deep_agent``. Defaults
        to an :class:`~langgraph.checkpoint.memory.InMemorySaver`, which the
        human-in-the-loop gates require; pass a durable saver to resume a paused
        run from another process.
    interrupts:
        Whether the human-in-the-loop gates are active (default ``True``). Set
        ``False`` for unattended runs — the pipeline then proceeds without asking,
        and a plan that fails verification stops with
        ``status="Clarification Required"`` instead of pausing.

    Examples
    --------
    Blocking, with a callback that answers each gate::

        agent  = TremorsAgent(llm=llm, output_dir="./out")
        result = agent.run("M5+ earthquakes near Japan in 2020",
                           on_interrupt=approve_all)

    Non-blocking (notebooks): ``run()`` returns ``status="Awaiting Input"`` with
    the pending request under ``interrupt``, and ``resume(value)`` continues::

        result = agent.run("…")
        while result["status"] == "Awaiting Input":
            print(result["interrupt"])            # inspect what is being asked
            result = agent.resume(approve_all(result["interrupt"]))

    See :func:`approve_all` for why the callback should not be a bare
    ``lambda`` returning a fixed decision list.
    """

    # ------------------------------------------------------------------
    # System prompt — encodes the routing the StateGraph used to hard-code
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT = """\
You are TREMORS, an expert seismology agent. You turn a user's natural-language
request for earthquake catalogs or waveforms into FDSN data retrievals and
plots by calling the tools provided. You do not write prose analyses; you call
tools and let them do the work.

TOOLS (call each AT MOST ONCE, in the order dictated below):
  plan_query                     – parse the request into search parameters. ALWAYS call this FIRST.
  query_cascade                  – fan out across datacenters and build the event catalog tables.
  retrieve_waveforms             – download per-event waveforms.
  plot_results                   – render the event map and timeline.
  plot_waveforms                 – render per-event waveform figures.
  retrieve_continuous_waveforms  – bulk-download continuous waveform streams.
  plot_continuous_waveforms      – render the continuous waveform figure.

ROUTING RULES (follow exactly — each tool's result tells you the next step):
  1. Call `plan_query` first, passing the user's full request as `query`.
  2. If plan_query reports status "Clarification Required" or "Parse Failed",
     STOP immediately and do not call any more tools.
  3. If the plan requested continuous waveforms, call
     `retrieve_continuous_waveforms`, then `plot_continuous_waveforms`, then STOP.
  4. Otherwise call `query_cascade`.
       - If the plan requested event waveforms, next call `retrieve_waveforms`,
         then `plot_results`.
       - Otherwise call `plot_results` directly.
  5. After `plot_results`, if the plan requested waveform plots, call
     `plot_waveforms`; then STOP.

HUMAN REVIEW: some tool calls are shown to the user for approval before they
run. If a call comes back reporting that the user rejected it, read the reason
they gave, then call `plan_query` again with the user's original request plus
that correction appended — do not retry the rejected tool unchanged, and do not
invent a `params_override` argument yourself (it exists only for the user's
edits).

CRITICAL: Call EXACTLY ONE tool per turn and wait for its result before
deciding the next call. Never call multiple tools in a single turn. When the
routing rules say STOP, produce a one-line summary of what was accomplished and
end your turn.
"""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm: Any,
        output_dir: str = "./tremors_output",
        checkpointer: Any = None,
        interrupts: bool = True,
    ):
        self.llm        = llm
        self.name       = "TremorsAgent"
        self.interrupts = interrupts

        # A checkpointer is mandatory for the human-in-the-loop gates: pausing is
        # implemented with interrupt(), which needs somewhere to persist the
        # partial run. Defaulting to an in-process saver keeps the gates working
        # out of the box; callers wanting durable, cross-process resume pass their
        # own (e.g. a SQLite or Postgres saver).
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

        # Set by run()/resume() so a paused run can be continued on the same
        # thread; None between runs.
        self._thread_id: Optional[str] = None

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # self._action is the compiled deepagents CompiledStateGraph — the
        # attribute name is kept so power users can still reach the graph.
        self._action = self._build_deep_agent()

    # ------------------------------------------------------------------
    # deepagents construction
    # ------------------------------------------------------------------

    def _build_tools(self) -> list:
        """
        Wrap each pipeline node as a deepagents tool.

        Each tool reads accumulated pipeline state via ``InjectedState``, runs
        the corresponding ``_*_node`` implementation, and commits only the
        known Tremors payload keys back to state via ``Command(update=...)``
        (see :func:`_project_state`).  The ``ToolMessage`` summary doubles as a
        next-step hint so a weak local model stays on the routing rails.

        Both the injected ``state`` and ``tool_call_id`` parameters are hidden
        from the model's ``tool_call_schema``; only ``plan_query`` exposes a
        real argument (``query``).
        """

        @tool
        def plan_query(
            query: str,
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Parse the user's natural-language request into FDSN search parameters. Call this FIRST, passing the user's full request as `query`."""
            result = self._plan_query_node({**dict(state), "query": query})
            params = result.get("search_params") or {}
            status = result.get("status", "")

            if status in ("Clarification Required", "Parse Failed"):
                hint = f"status={status!r}: {result.get('error', '')} STOP — do not call more tools."
            elif params.get("get_continuous_waveforms"):
                hint = (f"status={status!r}. Continuous-waveform request. "
                        "Next call retrieve_continuous_waveforms.")
            else:
                extras = []
                if params.get("get_waveforms"):
                    extras.append("event waveforms requested")
                if params.get("plot_waveforms"):
                    extras.append("waveform plots requested")
                extra = f" ({'; '.join(extras)})" if extras else ""
                hint = f"status={status!r}{extra}. Next call query_cascade."

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"plan_query: {hint}", tool_call_id=tool_call_id)],
            })

        @tool
        def query_cascade(
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            params_override: Optional[dict] = None,
        ) -> Command:
            """Fan out across global and regional FDSN datacenters, concatenate the event catalogs, and write the KBCore parquet tables. Leave `params_override` unset — it carries the user's edits from the approval gate."""
            result = self._query_cascade_node(_apply_override(state, params_override))
            params = (result.get("search_params") or {})
            tables = result.get("metadata_tables") or {}
            dcs    = result.get("queried_dcs") or []

            if params.get("get_waveforms") or params.get("plot_waveforms"):
                nxt = "Next call retrieve_waveforms."
            else:
                nxt = "Next call plot_results."
            hint = (f"status={result.get('status', '')!r}, {len(tables)} table(s), "
                    f"DCs={dcs}. {nxt}")

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"query_cascade: {hint}", tool_call_id=tool_call_id)],
            })

        @tool
        def retrieve_waveforms(
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Download per-event waveform MiniSEED for the retrieved catalog events. Call after query_cascade when event waveforms were requested."""
            result = self._retrieve_waveforms_node(dict(state))
            n = len(result.get("waveforms_saved") or [])
            hint = f"{n} waveform file(s) saved. Next call plot_results."

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"retrieve_waveforms: {hint}", tool_call_id=tool_call_id)],
            })

        @tool
        def plot_results(
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Render the event map and timeline figures from the retrieved catalog. Call after query_cascade (or retrieve_waveforms)."""
            result = self._plot_results_node(dict(state))
            plots  = result.get("plots") or []
            params = (result.get("search_params") or {})
            if params.get("plot_waveforms"):
                nxt = "Next call plot_waveforms."
            else:
                nxt = "Pipeline complete — STOP and summarize."
            hint = f"status={result.get('status', '')!r}, {len(plots)} plot(s). {nxt}"

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"plot_results: {hint}", tool_call_id=tool_call_id)],
            })

        @tool
        def plot_waveforms(
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Render per-event waveform figures. Call after plot_results when waveform plots were requested. This is the final step."""
            result = self._plot_waveforms_node(dict(state))
            n = len(result.get("waveform_plots") or [])
            hint = f"{n} waveform plot(s) saved. Pipeline complete — STOP and summarize."

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"plot_waveforms: {hint}", tool_call_id=tool_call_id)],
            })

        @tool
        def retrieve_continuous_waveforms(
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            params_override: Optional[dict] = None,
        ) -> Command:
            """Bulk-download continuous waveform streams (inventory-driven). Call after plan_query for continuous-waveform requests. Leave `params_override` unset — it carries the user's edits from the approval gate."""
            result = self._retrieve_continuous_waveforms_node(
                _apply_override(state, params_override)
            )
            n = len(result.get("continuous_waveforms_saved") or [])
            hint = (f"status={result.get('status', '')!r}, {n} trace(s) saved. "
                    "Next call plot_continuous_waveforms.")

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"retrieve_continuous_waveforms: {hint}", tool_call_id=tool_call_id)],
            })

        @tool
        def plot_continuous_waveforms(
            state: Annotated[TremorsDeepState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Render the composite continuous-waveform figure. Call after retrieve_continuous_waveforms. This is the final step."""
            result = self._plot_continuous_waveforms_node(dict(state))
            n = len(result.get("continuous_waveform_plots") or [])
            hint = f"{n} continuous waveform plot(s) saved. Pipeline complete — STOP and summarize."

            return Command(update={
                **_project_state(result),
                "messages": [ToolMessage(f"plot_continuous_waveforms: {hint}", tool_call_id=tool_call_id)],
            })

        return [
            plan_query,
            query_cascade,
            retrieve_waveforms,
            plot_results,
            plot_waveforms,
            retrieve_continuous_waveforms,
            plot_continuous_waveforms,
        ]

    def _build_deep_agent(self):
        """
        Build the compiled deepagents graph that drives the Tremors tools.

        The default deepagents tool surface (a general-purpose ``task``
        subagent plus the full ``ls``/``write_file``/``edit_file``/``glob``/
        ``grep``/``execute`` filesystem suite) is trimmed to the bare minimum
        so a small local model is not tempted into wrong-tool selection for our
        fixed pipeline:

        * harness profiles keyed to the ``openai`` and ``anthropic`` providers
          disable the auto general-purpose subagent, which (with
          ``subagents=None``) drops the ``task`` tool;
        * a custom ``FilesystemMiddleware`` exposes only ``read_file`` (the
          unavoidable floor — the middleware requires at least one tool).

        The result is the seven Tremors tools plus ``read_file``.

        Two of the seven are wrapped in a ``HumanInTheLoopMiddleware`` gate — see
        :meth:`_interrupt_config`.
        """
        # Disable the auto general-purpose subagent. deepagents resolves a
        # model's harness profile from its provider, so each supported backend
        # needs its own registration: the Ollama and OpenAI backends resolve to
        # provider "openai", while the Anthropic backend (ChatAnthropic)
        # resolves to "anthropic" — which ships no built-in provider-level
        # profile, so without this it would fall back to a default that leaves
        # the "task" tool enabled. register_harness_profile is additive/
        # idempotent, and registering a profile for a provider not currently in
        # use is harmless, so both are registered unconditionally on each build.
        for _provider in ("openai", "anthropic"):
            register_harness_profile(
                _provider,
                HarnessProfile(
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                ),
            )

        backend = StateBackend()

        middleware: List[Any] = [FilesystemMiddleware(backend=backend, tools=["read_file"])]
        if self.interrupts:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=self._interrupt_config(),
                    description_prefix="TREMORS is about to retrieve data",
                )
            )

        return create_deep_agent(
            model=self.llm,
            tools=self._build_tools(),
            system_prompt=self._SYSTEM_PROMPT,
            state_schema=TremorsDeepState,
            middleware=middleware,
            backend=backend,
            subagents=None,
            checkpointer=self.checkpointer,
        )

    # ------------------------------------------------------------------
    # Human-in-the-loop gates
    # ------------------------------------------------------------------

    def _interrupt_config(self) -> dict:
        """
        Build the ``interrupt_on`` map for :class:`HumanInTheLoopMiddleware`.

        Gates are keyed by tool name and fire *after* the model has chosen a call
        but *before* it executes, so nothing is replayed and no request has left
        the machine when the user is asked.

        Two tools are gated, because they are the two ways into the FDSN network:

        ``query_cascade``
            Every event query passes through here. This is the cheapest possible
            place to catch a plan that is schema-valid and verifier-clean but
            still not what the user meant — a decade instead of a year, the wrong
            hemisphere, a magnitude floor that will return ten thousand events.

        ``retrieve_continuous_waveforms``
            The bulk continuous path: many processes, potentially many gigabytes,
            and the only step whose cost is unbounded by the catalog size. It is
            gated on the same approve/edit/reject decisions, but its description
            reports the download's *scale* rather than just the plan.

        The remaining five tools are not gated: they either operate on data
        already retrieved (the plotting steps, ``retrieve_waveforms``) or are the
        planning step itself, which carries its own clarification pause.
        """
        decisions = ["approve", "edit", "reject"]
        return {
            "query_cascade": {
                "allowed_decisions": decisions,
                "description":       self._describe_cascade_gate,
            },
            "retrieve_continuous_waveforms": {
                "allowed_decisions": decisions,
                "description":       self._describe_bulk_gate,
            },
        }

    @staticmethod
    def _format_params(params: dict) -> str:
        """Render search parameters as sorted ``key: value`` lines for a gate prompt."""
        if not params:
            return "  (no parameters were resolved)"
        width = max(len(k) for k in params)
        return "\n".join(f"  {k:<{width}} : {params[k]}" for k in sorted(params))

    def _describe_cascade_gate(self, tool_call: dict, state: Any, runtime: Any) -> str:
        """
        Describe the pending event query.

        Reads ``search_params`` from *state*: the gate has to show what the tool
        will actually do, and the tool's own arguments are empty by design.
        """
        params  = (state.get("search_params") or {})
        targets = self._determine_target_dcs(params)
        return (
            "Event catalog query — the following datacenters will be queried:\n"
            f"  {', '.join(targets)}\n\n"
            "Resolved search parameters:\n"
            f"{self._format_params(params)}\n\n"
            "Duplicate reports of the same earthquake from different datacenters "
            "are kept, tagged by their source."
        )

    def _describe_bulk_gate(self, tool_call: dict, state: Any, runtime: Any) -> str:
        """
        Describe the pending bulk continuous download, including its scale.

        Everything reported here is derived from ``search_params`` alone — no
        inventory is fetched, because the point of the gate is to ask *before*
        touching the network.
        """
        params = (state.get("search_params") or {})

        try:
            span = UTCDateTime(params["max_date"]) - UTCDateTime(params["min_date"])
            days = f"{max(1, int(span / 86400))} day(s)"
        except Exception:                             # noqa: BLE001 — display only
            days = "unknown (dates missing or unparseable)"

        selection = " ".join(
            f"{key}={params.get(key) or '*'}" for key in ("net", "sta", "loc", "chan")
        )
        return (
            "Bulk continuous waveform download — this is the expensive path.\n\n"
            f"  time span      : {days}\n"
            f"  channels       : {selection}\n"
            f"  worker procs   : {params.get('parallel', 4)}\n"
            f"  writing to     : {self.output_dir}\n\n"
            "Resolved search parameters:\n"
            f"{self._format_params(params)}\n\n"
            "Days already present on disk are skipped, so approving a re-run is safe."
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        on_interrupt: Optional[Callable[[dict], Any]] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Run one natural-language query through the deepagents tool loop and
        return a legacy-shaped result dict (the same keys ``cli.py`` and the
        notebooks consume).

        Parameters
        ----------
        query:
            The user's natural-language seismic request.
        on_interrupt:
            Optional callback invoked when a human-in-the-loop gate pauses the
            run. It receives the gate's payload and returns the resume value; the
            run then continues, looping for as many gates as fire. When omitted
            (the default), ``run()`` **does not block**: it returns
            ``status="Awaiting Input"`` with the payload under ``interrupt``, and
            :meth:`resume` continues from there.
        thread_id:
            Checkpointer thread to run under. Defaults to a fresh ``uuid4``-based
            id. Supply one — together with a durable ``checkpointer`` — to name a
            session that a **later process** can pick up via :meth:`resume`
            (this is how ``tremors query --session-id`` and ``tremors resume``
            hand a paused run between invocations).

        Returns
        -------
        dict
            Keys: ``status``, ``error``, ``problems``, ``queried_dcs``,
            ``metadata_tables``, ``plots``, ``waveforms_saved``,
            ``continuous_waveforms_saved``, ``waveform_plots``,
            ``continuous_waveform_plots``, plus ``search_params`` and
            ``datacenter`` for downstream inspection, and ``interrupt`` when a
            gate is pending.
        """
        # A fresh thread per run unless the caller names one. The previous fixed
        # "tremors" thread_id meant a second run() on the same agent resumed the
        # first run's *finished* conversation instead of starting a new one.
        self._thread_id = thread_id or f"tremors-{uuid.uuid4().hex[:12]}"

        state = self._action.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=self._config(),
        )
        return self._drive(state, on_interrupt)

    def resume(
        self,
        value: Any,
        *,
        on_interrupt: Optional[Callable[[dict], Any]] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Continue a run that paused at a human-in-the-loop gate.

        Parameters
        ----------
        value:
            The answer to the pending request. For an approval gate this is
            ``{"decisions": [decision]}`` with exactly one decision — one of
            ``{"type": "approve"}``,
            ``{"type": "edit", "edited_action": {"name": …, "args": {"params_override": {…}}}}``,
            or ``{"type": "reject", "message": "why"}``. For a clarification
            pause (``kind == "clarification"``) it is the clarifying text itself.
        on_interrupt:
            As for :meth:`run` — supply it to answer any *subsequent* gates
            without returning to the caller.
        thread_id:
            Session to resume. Required when this agent object did not start the
            run (a fresh process resuming a paused session from a durable
            checkpointer); omit it to continue the run this object is holding.
        """
        if thread_id is not None:
            self._thread_id = thread_id
        if self._thread_id is None:
            raise RuntimeError(
                "resume() called with no run in progress — call run() first, or pass "
                "thread_id= to resume a session held by a durable checkpointer."
            )

        state = self._action.invoke(Command(resume=value), config=self._config())
        return self._drive(state, on_interrupt)

    def plan(self, query: str) -> dict:
        """
        Resolve *query* to verified search parameters **without contacting FDSN**.

        A dry run for callers that want to see what a query becomes — and whether
        the verifier accepts it — before committing to network time. Costs the one
        planner LLM call that :meth:`_plan_query_node` always makes; sends no FDSN
        request and writes no files.

        Verifier findings are returned as data in ``problems`` rather than raising
        a clarification pause, so this is safe to call unattended.

        Returns
        -------
        dict
            ``status``, ``error``, ``problems``, ``search_params``,
            ``datacenter``, and ``target_datacenters`` — the datacenters
            :meth:`_query_cascade_node` *would* fan out to, computed from the
            resolved parameters. ``target_datacenters`` is empty when the plan is
            unusable, since there is nothing to query.
        """
        # The node consults self.interrupts to decide whether to raise a
        # clarification interrupt(). interrupt() outside a graph execution would
        # fail, and a dry run must never block, so suppress gating for the call.
        # Restored in finally: this object stays reusable for a real run().
        saved_interrupts = self.interrupts
        self.interrupts  = False
        try:
            result = self._plan_query_node({"query": query})
        finally:
            self.interrupts = saved_interrupts

        params   = result.get("search_params") or {}
        problems = list(result.get("problems") or [])

        return {
            "status":             result.get("status", "Unknown"),
            "error":              result.get("error"),
            "problems":           problems,
            "search_params":      params,
            "datacenter":         result.get("datacenter"),
            "target_datacenters": (
                self._determine_target_dcs(params) if params and not problems else []
            ),
        }

    def _config(self) -> dict:
        """Invoke-time config: recursion budget plus the current run's thread."""
        return {
            "recursion_limit": 50,
            "configurable":    {"thread_id": self._thread_id},
        }

    def _drive(self, state: dict, on_interrupt: Optional[Callable[[dict], Any]]) -> dict:
        """
        Resolve pending gates, then project the graph state to a result dict.

        With a callback, loops until the graph runs to completion. Without one,
        returns at the first pause so a notebook or a server can decide what to
        do — the run stays resumable via :meth:`resume` because the checkpointer
        holds it.
        """
        while (pending := self._pending_interrupt(state)) is not None:
            if on_interrupt is None:
                result = self._project_result(state)
                result["status"]    = "Awaiting Input"
                result["interrupt"] = pending
                return result
            state = self._action.invoke(
                Command(resume=on_interrupt(pending)),
                config=self._config(),
            )

        self._thread_id = None
        return self._project_result(state)

    @staticmethod
    def _pending_interrupt(state: dict) -> Optional[dict]:
        """Return the payload of the pending interrupt, or None if the run finished."""
        interrupts = (state or {}).get("__interrupt__") or ()
        if not interrupts:
            return None
        payload = getattr(interrupts[0], "value", interrupts[0])
        return payload if isinstance(payload, dict) else {"value": payload}

    @staticmethod
    def _project_result(state: dict) -> dict:
        """Project the final graph state onto the public result dict."""
        state = state or {}
        return {
            "status":                      state.get("status", "Unknown"),
            "error":                       state.get("error"),
            "problems":                    state.get("problems", []),
            "queried_dcs":                 state.get("queried_dcs", []),
            "metadata_tables":             state.get("metadata_tables", {}),
            "plots":                       state.get("plots", []),
            "waveforms_saved":             state.get("waveforms_saved", []),
            "continuous_waveforms_saved":  state.get("continuous_waveforms_saved", []),
            "waveform_plots":              state.get("waveform_plots", []),
            "continuous_waveform_plots":   state.get("continuous_waveform_plots", []),
            "search_params":               state.get("search_params", {}),
            "datacenter":                  state.get("datacenter"),
        }

    # ------------------------------------------------------------------
    # Node: plan_query
    # ------------------------------------------------------------------

    _PLAN_SYSTEM_PROMPT = f"""\
You are an expert seismologist assistant.
Parse the user's natural language query into FDSN search parameters.

Emit ONLY the keys the user actually asked for — omit everything else rather than
guessing. Downstream defaults are applied for anything you leave out.

Available keys:

{_render_param_key_block()}

Return ONLY the JSON object – no prose, no markdown fences."""

    def _extract_search_params(
        self,
        query:         str,
        clarification: Optional[str] = None,
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Ask the LLM for search parameters and validate them against
        :class:`~tremors.utils.params.SearchParams`.

        Two paths, preferred first:

        1. ``with_structured_output(..., method="json_schema")`` — the provider
           enforces the schema, so a hallucinated key or a string where a float
           belongs fails here instead of silently widening an FDSN query.
        2. Free-text JSON, stripped of markdown fences and then validated through
           the *same* schema. Not every Ollama-served model or gateway implements
           ``json_schema``, so this path must stay.

        Which path ran is logged, because "the plan looked fine" means something
        different in each case.

        Returns ``(params, error)`` with exactly one of the two set.
        """
        human = f"User Query: {query}"
        if clarification:
            human += (
                "\n\nA first attempt at these parameters was rejected as invalid. "
                f"The user clarified:\n{clarification}\n"
                "Re-extract the parameters taking that clarification into account."
            )

        # The query goes in a user turn rather than appended to the system prompt:
        # the Anthropic Messages API carries ``system`` as a separate top-level
        # field, so a system-only prompt sends an empty ``messages`` array and
        # 400s ("'messages' must not be empty"). A human turn is also correct for
        # the OpenAI/Ollama backends.
        messages = [SystemMessage(self._PLAN_SYSTEM_PROMPT), HumanMessage(human)]

        try:
            structured = self.llm.with_structured_output(
                SearchParams, method="json_schema", include_raw=True
            )
            result = structured.invoke(messages)
            parsed = result.get("parsed") if isinstance(result, dict) else None
            if parsed is not None:
                _log(self.name, "Planner path: structured output (json_schema).")
                return parsed.to_params(), None
            reason = result.get("parsing_error") if isinstance(result, dict) else "no object returned"
            _log(self.name, f"Structured output produced nothing ({reason}); trying text fallback.")
        except Exception as exc:                      # noqa: BLE001 — provider support varies
            _log(self.name, f"Structured output unavailable ({exc}); trying text fallback.")

        try:
            response = self.llm.invoke(messages)
            content  = response.content
            # Anthropic returns a content *list* when thinking/citation blocks are
            # enabled; concatenate the text parts before parsing.
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            content = (content or "").strip().replace("```json", "").replace("```", "")
            params  = SearchParams.model_validate(json.loads(content)).to_params()
        except json.JSONDecodeError as exc:
            return None, f"the model's reply was not valid JSON ({exc})"
        except ValidationError as exc:
            return None, (
                f"the model returned parameters that failed validation "
                f"({exc.error_count()} problem(s)): {exc.errors()[0].get('msg', '')}"
            )
        except Exception as exc:                      # noqa: BLE001 — surface, don't crash
            return None, f"the planner call failed ({type(exc).__name__}: {exc})"

        _log(self.name, "Planner path: text fallback (json.loads + schema validation).")
        return params, None

    def _plan_query_node(self, state: TremorsState) -> TremorsState:
        """
        Translate the natural-language query into a verified ``search_params`` dict.

        Three layers, in order:

        1. **Structured extraction** — see :meth:`_extract_search_params`.
        2. **Domain verification** — ``verify_search_params`` catches plans that
           are schema-valid but cannot be right (inverted bounding box, magnitude
           window that excludes everything, unknown datacenter, a point with no
           radius). This **fails closed**: any finding stops the run rather than
           letting it reach the datacenters and come back plausibly empty.
        3. **Clarification gate** — when verification fails and interrupts are
           enabled, pause and re-plan once with the user's answer.

        Point queries with a radius are expanded to a bounding box after
        verification, since the un-expanded point form is what the verifier
        checks.

        .. note::
           The clarification pause is a raw ``interrupt()`` rather than a
           middleware gate, because the findings it reports are produced *inside*
           this call and ``HumanInTheLoopMiddleware`` fires before a tool runs.
           ``interrupt()`` replays its node on resume, so the first LLM call is
           paid a second time on a clarified run. That is safe — nothing outside
           the process has happened yet — and bounded: one clarification round,
           then fail closed.
        """
        query = state["query"]
        _log(self.name, f"Planning query: {query}")

        search_params, error = self._extract_search_params(query)
        if error is not None:
            _log(self.name, f"Planning failed: {error}")
            return {
                **state,
                "search_params": {},
                "status": "Parse Failed",
                # No verifier findings exist on this path — extraction never
                # produced params to verify — so ``problems`` stays empty and
                # ``error`` carries the reason.
                "problems": [],
                "error": (
                    f"Could not extract search parameters from your query — {error}. "
                    "Please rephrase and try again."
                ),
            }
        _log(self.name, f"Extracted params: {search_params}")

        # A single location is often emitted as min_lat/min_lon with no maxima.
        # Mirror it into the point form the verifier and the expansion below both
        # recognise, so a legitimate "within 50 km of X" is not rejected as an
        # incomplete bounding box.
        if (
            search_params.get("min_lat") is not None
            and search_params.get("min_lon") is not None
            and search_params.get("max_lat") is None
            and search_params.get("max_lon") is None
        ):
            search_params["max_lat"] = search_params["min_lat"]
            search_params["max_lon"] = search_params["min_lon"]

        problems = verify_search_params(search_params, WELL_KNOWN_NODES.keys())

        if problems and self.interrupts:
            _log(self.name, f"Plan failed verification ({len(problems)} problem(s)); asking user.")
            answer = interrupt({
                "kind":     "clarification",
                "tool":     "plan_query",
                "query":    query,
                "params":   search_params,
                "problems": problems,
                "prompt":   "These search parameters could not be used. "
                            "Clarify the request and the query will be re-planned.",
            })
            revised, error = self._extract_search_params(query, clarification=str(answer))
            if error is None:
                search_params = revised
                problems      = verify_search_params(search_params, WELL_KNOWN_NODES.keys())
                _log(self.name, f"Re-planned params: {search_params}")
            else:
                problems = [f"Re-planning after clarification failed — {error}."]

        if problems:
            _log(self.name, f"Stopping: {len(problems)} unresolved problem(s).")
            return {
                **state,
                "search_params": search_params,
                "status": "Clarification Required",
                "problems": list(problems),
                "error":  " ".join(problems),
            }

        # ── Point-query expansion ──────────────────────────────────────
        # Verification has already established that a single point carries a
        # radius, so this only ever widens a valid point into a real search area.
        min_lat, max_lat = search_params.get("min_lat"), search_params.get("max_lat")
        min_lon, max_lon = search_params.get("min_lon"), search_params.get("max_lon")

        is_point_query = (
            min_lat is not None
            and min_lat == max_lat
            and min_lon is not None
            and min_lon == max_lon
        )

        if is_point_query:
            radius = search_params["radius"]
            unit   = search_params.get("radius_unit", "km")
            exp_min_lat, exp_max_lat, exp_min_lon, exp_max_lon = boundingbox(
                min_lat, min_lon, radius, unit=unit, ellipse="WGS84"
            )
            search_params.update(
                min_lat=exp_min_lat, max_lat=exp_max_lat,
                min_lon=exp_min_lon, max_lon=exp_max_lon,
            )
            _log(
                self.name,
                f"Expanded point ({min_lat}, {min_lon}) → bbox "
                f"[{exp_min_lat:.3f}, {exp_max_lat:.3f}, "
                f"{exp_min_lon:.3f}, {exp_max_lon:.3f}] "
                f"using {radius} {unit}",
            )

        dc = search_params.get("datacenter", "ISC")
        return {
            **state,
            "search_params": search_params,
            "datacenter":    dc,
            "problems":      [],
            "status":        "Query parsed",
        }

    # ------------------------------------------------------------------
    # DC selection helper
    # ------------------------------------------------------------------

    def _determine_target_dcs(self, params: dict) -> List[str]:
        """
        Build the ordered list of FDSN datacenters to query.

        Strategy
        --------
        1. Always start with the four global discovery DCs.
        2. Append any regional DCs whose bounding box overlaps the query area.
        3. If the user explicitly named a DC that isn't already included, append it.
        4. Deduplicate while preserving insertion order.
        """
        seen: dict = {}
        for dc in ("USGS", "EMSC", "GEOFON", "ISC"):
            seen.setdefault(dc, None)

        min_lat = params.get("min_lat")
        max_lat = params.get("max_lat")
        min_lon = params.get("min_lon")
        max_lon = params.get("max_lon")

        if all(x is not None for x in (min_lat, max_lat, min_lon, max_lon)):
            for region, rule in REGIONAL_RULES.items():
                r_min_lat, r_max_lat, r_min_lon, r_max_lon = rule["bbox"]
                lat_overlap = min_lat <= r_max_lat and max_lat >= r_min_lat
                lon_overlap = min_lon <= r_max_lon and max_lon >= r_min_lon
                if lat_overlap and lon_overlap:
                    _log(self.name, f"Region match: {region} → adding {rule['dcs']}")
                    for dc in rule["dcs"]:
                        seen.setdefault(dc, None)

        requested_dc = params.get("datacenter")
        if (
            requested_dc
            and requested_dc != "ISC"
            and requested_dc in WELL_KNOWN_NODES
            and requested_dc not in seen
        ):
            _log(self.name, f"User-requested DC: {requested_dc}")
            seen[requested_dc] = None

        return list(seen.keys())

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_event_dc(event) -> str:
        """
        Read the ``datacenter:<DC>`` comment stamped onto *event* during the
        cascade merge and return the DC name.  Falls back to ``"UNKNOWN"``.
        """
        for comment in event.comments:
            text = getattr(comment, "text", "") or ""
            if text.startswith("datacenter:"):
                return text.split(":", 1)[1].strip()
        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Node: query_cascade
    # ------------------------------------------------------------------

    def _query_one_dc(self, dc: str, query_kwargs: dict) -> Tuple[str, Optional[Catalog], Optional[str]]:
        """
        Query a single datacenter. Never raises.

        Runs on a worker thread of the cascade's pool, so *every* failure path
        must return rather than propagate: an exception escaping here would
        surface out of the pool and abandon the sibling datacenters' results.
        Returns ``(dc, catalog, error)`` with exactly one of ``catalog``/``error``
        set.
        """
        try:
            client = _make_fdsn_client(dc)
            if client is None:
                return dc, None, "client could not be initialised"
            cat = client.get_events(**query_kwargs)
        except Exception as exc:                      # noqa: BLE001 — see docstring
            return dc, None, str(exc)

        # Tag provenance on every event, unconditionally. This used to live inside
        # the (now removed) dedup branch; if it is ever skipped for an event,
        # _extract_event_dc reports "UNKNOWN" and the per-DC parquet grouping
        # collapses into a single mislabelled group.
        for event in cat:
            event.comments.append(Comment(text=f"datacenter:{dc}"))
        return dc, cat, None

    def _query_cascade_node(self, state: TremorsState) -> TremorsState:
        """
        Fan-out across all target DCs and concatenate their catalogs.

        Datacenters are queried **concurrently** — the work is entirely
        network-bound, and querying four-plus institutions in series was the
        dominant cost of an event query. Failures are contained per datacenter
        (see :meth:`_query_one_dc`), matching the previous behaviour of skipping
        an unreachable node and continuing.

        **No deduplication is performed.** Every event returned by every queried
        datacenter reaches the output tables, tagged with a ``datacenter:<DC>``
        Comment. One earthquake reported by USGS, EMSC, GEOFON and ISC therefore
        appears as four rows in ``EVENT.PARQUET`` and four markers on the map,
        distinguished by their ``datacenter`` column. This is deliberate: which
        reports agree, and how their locations differ, is signal — collapsing
        them with a fixed ±10 s / ±0.1° threshold silently discarded it and
        picked a winner by query order.

        Events are sorted by (origin time, datacenter, event id) before
        conversion. Concurrency makes completion order arbitrary, and the sort
        is what keeps the parquet tables byte-identical across runs of the same
        query.
        """
        _log(self.name, "Starting cascade query…")
        params = state.get("search_params", {})

        targets = self._determine_target_dcs(params)
        _log(self.name, f"Target DCs: {targets}")

        now     = UTCDateTime.now().strftime("%Y-%m-%d")
        t_start = UTCDateTime(params.get("min_date", "1970-01-01"))
        t_end   = UTCDateTime(params.get("max_date", now))
        min_mag = params.get("min_mag", 3.0)
        limit   = params.get("limit", 100)

        query_kwargs: dict = {
            "starttime":    t_start,
            "endtime":      t_end,
            "minmagnitude": min_mag,
            "limit":        limit,
        }

        if params.get("min_lat") is not None:
            query_kwargs.update(
                minlatitude=params["min_lat"],
                maxlatitude=params["max_lat"],
                minlongitude=params["min_lon"],
                maxlongitude=params["max_lon"],
            )

        # ── Query every target datacenter concurrently ────────────────
        # executor.map yields in *input* order, so the DC sequence is stable even
        # though completion order is not.
        workers = max(1, min(_CASCADE_MAX_WORKERS, len(targets)))
        _log(self.name, f"Querying {len(targets)} datacenter(s) with {workers} worker(s)…")

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cascade") as pool:
            results = list(pool.map(lambda dc: self._query_one_dc(dc, query_kwargs), targets))

        events: List[Tuple[float, str, str, Any]] = []   # sort key + event
        queried_success: List[str] = []

        for dc, cat, error in results:
            if error is not None:
                _log(self.name, f"{dc}: skipped ({error})")
                continue

            _log(self.name, f"{dc}: {len(cat)} events returned.")
            if len(cat) > 0:
                queried_success.append(dc)

            for event in cat:
                if not event.origins:
                    continue
                origin = event.preferred_origin() or event.origins[0]
                events.append(
                    (float(origin.time.timestamp), dc, str(event.resource_id), event)
                )

        # Deterministic order: origin time, then datacenter, then event id.
        events.sort(key=lambda item: item[:3])

        master_catalog: Catalog = Catalog([item[3] for item in events])

        _log(
            self.name,
            f"Cascade complete. {len(master_catalog)} events from "
            f"{len(queried_success)} datacenter(s) (duplicates retained).",
        )

        if len(master_catalog) == 0:
            return {
                **state,
                "metadata_tables": {},
                "status":          "Success (No Data)",
                "queried_dcs":     queried_success,
            }

        # ── Convert catalog → parquet tables (per-DC provenance) ──────
        dc_groups: Dict[str, Catalog] = defaultdict(Catalog)
        for event in master_catalog:
            dc_label = self._extract_event_dc(event)
            dc_groups[dc_label].append(event)

        _log(
            self.name,
            f"Provenance groups: { {dc: len(cat) for dc, cat in dc_groups.items()} }",
        )

        combined: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        for dc_label, sub_catalog in dc_groups.items():
            for name, df in catalog_to_kbcore(
                sub_catalog, datacenter=dc_label
            ).items():
                if not df.empty:
                    combined[name].append(df)

        saved_files: Dict[str, str] = {}
        for name, frames in combined.items():
            merged_df = pd.concat(frames, ignore_index=True)
            path      = os.path.join(self.output_dir, f"{name}.parquet".upper())
            merged_df.to_parquet(path, index=False)
            saved_files[name] = path
            _log(self.name, f"Saved {name} ({len(merged_df)} rows) → {path}")

        return {
            **state,
            "metadata_tables": saved_files,
            "status":          "Success",
            "queried_dcs":     queried_success,
        }

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _load_and_merge_data(self, state: TremorsState) -> Optional[pd.DataFrame]:
        """
        Load origin / event / netmag parquet files and produce a single
        merged DataFrame ready for plotting.

        Returns ``None`` if the data is missing, empty, or lacks a
        ``magnitude`` column after all merges.
        """
        tables = state.get("metadata_tables", {})

        if "origin" not in tables:
            _log(self.name, "No origin data to plot.")
            return None

        origin_df = pd.read_parquet(tables["origin"])
        if origin_df.empty:
            _log(self.name, "Origin table is empty.")
            return None

        df = origin_df

        if "event" in tables:
            try:
                event_df = pd.read_parquet(tables["event"])
                if not event_df.empty:
                    df = pd.merge(
                        event_df, origin_df,
                        left_on="prefor", right_on="orid",
                        suffixes=("_event", "_origin"),
                    )
                    _log(self.name, f"Merged event+origin: {len(df)} records.")
            except Exception as exc:
                _log(self.name, f"Event merge failed ({exc}). Using origin only.")

        if "netmag" in tables and "prefmag" in df.columns:
            try:
                netmag_df = pd.read_parquet(tables["netmag"])
                if not netmag_df.empty:
                    df["prefmag"]       = df["prefmag"].astype(int)
                    netmag_df["magid"]  = netmag_df["magid"].astype(int)
                    df = pd.merge(
                        df, netmag_df,
                        left_on="prefmag", right_on="magid",
                        suffixes=("", "_netmag"),
                    )
                    _log(self.name, f"Merged netmag: {len(df)} records.")
            except Exception as exc:
                _log(self.name, f"Netmag merge failed: {exc}")

        if "magnitude" not in df.columns:
            _log(self.name, "No 'magnitude' column after merges – skipping plot.")
            return None

        df = df.dropna(subset=["magnitude"])
        if df.empty:
            _log(self.name, "No events with a valid magnitude.")
            return None

        return df

    def _load_stations(self) -> Optional[pd.DataFrame]:
        """
        Load the WAVEFORM_SITE parquet if it exists and join with
        WAVEFORM_SITECHAN to produce a combined station/channel DataFrame.

        Falls back to site-only if sitechan is missing.  Returns None if
        neither file exists or both are empty.
        """
        site_path     = os.path.join(self.output_dir, "WAVEFORM_SITE.PARQUET")
        sitechan_path = os.path.join(self.output_dir, "WAVEFORM_SITECHAN.PARQUET")

        if not os.path.exists(site_path):
            return None

        site_df = pd.read_parquet(site_path)
        if site_df.empty:
            return None

        if os.path.exists(sitechan_path):
            try:
                sitechan_df = pd.read_parquet(sitechan_path)
                if not sitechan_df.empty:
                    merged = pd.merge(
                        site_df,
                        sitechan_df[["sta", "chan", "loc", "net"]].drop_duplicates(),
                        on="sta",
                        how="left",
                    )
                    _log(self.name, f"Loaded site⋈sitechan: {len(merged)} rows.")
                    return merged
            except Exception as exc:
                _log(self.name, f"sitechan merge for plotting failed ({exc}). Using site only.")

        return site_df

    def _setup_map_axes(
        self,
        df_plot: pd.DataFrame,
        stations_df: Optional[pd.DataFrame],
        params: dict,
    ) -> Tuple[plt.Figure, plt.Axes, float]:
        """
        Create a Cartopy figure with basemap imagery, coastlines, borders,
        rivers, and gridlines.

        Returns
        -------
        fig, ax, lon_span
        """
        pad = 3.0

        all_lats: List[float] = df_plot["lat"].tolist()
        all_lons: List[float] = df_plot["lon"].tolist()

        if stations_df is not None:
            all_lats.extend(stations_df["lat"].tolist())
            all_lons.extend(stations_df["lon"].tolist())

        for key in ("min_lat", "max_lat"):
            if key in params:
                all_lats.append(params[key])
        for key in ("min_lon", "max_lon"):
            if key in params:
                all_lons.append(params[key])

        min_lat  = min(all_lats) - pad
        max_lat  = max(all_lats) + pad
        min_lon  = min(all_lons) - pad
        max_lon  = max(all_lons) + pad
        lon_span = max_lon - min_lon

        zoom = (
            10 if lon_span <  1 else
             9 if lon_span <  3 else
             8 if lon_span <  6 else
             7 if lon_span < 12 else 6
        )

        fig = plt.figure(figsize=(12, 8))
        ax  = plt.axes(projection=ccrs.PlateCarree())

        imagery = GoogleTiles(
            url=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )
        )
        ax.add_image(imagery, zoom, alpha=0.7)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND,      facecolor="#f4f4f2", zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor="#ddeeff", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7,       zorder=1)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5,       zorder=1)
        ax.add_feature(cfeature.RIVERS,    linewidth=0.4,       zorder=1)
        ax.add_feature(cfeature.STATES,    edgecolor="#cdd2d6", linewidth=0.8, zorder=2)

        gl = ax.gridlines(
            draw_labels=True, alpha=0.15, linestyle="-", color="black", zorder=2
        )
        gl.top_labels   = False
        gl.right_labels = False

        return fig, ax, lon_span

    def _plot_event_map(
        self,
        df_plot: pd.DataFrame,
        stations_df: Optional[pd.DataFrame],
        params: dict,
        state: TremorsState,
    ) -> str:
        """
        Draw the geographic event map and save it as a JPEG.

        Layers (bottom → top)
        ---------------------
        1. Basemap imagery + coastlines / borders / rivers / states
        2. Search bounding box (semi-transparent grey polygon)
        3. Search-radius circles (dashed red)
        4. Stations: white triangles (possible), dark-green triangles (saved)
        5. Events: scatter coloured by magnitude (inferno_r)
        6. North arrow, legend, title

        Returns the path of the saved JPEG.
        """
        try:
            fig, ax, lon_span = self._setup_map_axes(df_plot, stations_df, params)
            use_cartopy = True
        except ImportError:
            fig = plt.figure(figsize=(10, 6))
            ax  = plt.gca()
            sizes = (10 ** (df_plot["magnitude"] / 2)) * 5
            sc = ax.scatter(
                df_plot["lon"], df_plot["lat"],
                s=sizes, c=df_plot["magnitude"],
                cmap="inferno_r", alpha=0.6, edgecolors="k",
            )
            plt.colorbar(sc, label="Magnitude")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True)
            use_cartopy = False

        transform       = ccrs.PlateCarree() if use_cartopy else None
        scatter_kwargs  = dict(transform=transform) if use_cartopy else {}

        # ── Search-radius circles ──────────────────────────────────────
        plot_radius = params.get("radius", 0.0)
        if plot_radius and plot_radius > 0 and use_cartopy:
            rad_km = (
                plot_radius * 111.32
                if params.get("radius_unit", "") == "deg"
                else float(plot_radius)
            )
            for _, row in df_plot.iterrows():
                row_lat = float(row["lat"])
                row_lon = float(row["lon"])
                points  = boundingradius(
                    row_lat, row_lon, rad_km,
                    unit="km", numpoints=361, ellipse="WGS84",
                )
                coords = list(zip(points[:, 1], points[:, 0]))
                ax.add_patch(Polygon(
                    coords, facecolor="none", alpha=0.5,
                    edgecolor="red", lw=2,
                    transform=ccrs.PlateCarree(), linestyle="--",
                ))
                ax.text(
                    points[180, 1], points[180, 0],
                    f"{rad_km:.0f} km",
                    transform=ccrs.PlateCarree(),
                    ha="center", va="center", fontsize=12, color="red",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                )

        # ── Bounding box ───────────────────────────────────────────────
        if all(k in params for k in ("min_lat", "max_lat", "min_lon", "max_lon")):
            box_coords = [
                (params["min_lon"], params["min_lat"]),
                (params["min_lon"], params["max_lat"]),
                (params["max_lon"], params["max_lat"]),
                (params["max_lon"], params["min_lat"]),
            ]
            patch_kwargs = dict(transform=ccrs.PlateCarree()) if use_cartopy else {}
            ax.add_patch(Polygon(
                box_coords, facecolor="gray", alpha=0.3,
                edgecolor="gray", lw=2, linestyle="--",
                label="Search bounds", **patch_kwargs,
            ))

        # ── Stations ───────────────────────────────────────────────────
        if stations_df is not None and "sta" in stations_df.columns:
            saved_stas = {
                os.path.basename(f).split("_")[2]
                for f in glob.glob(os.path.join(self.output_dir, "*.mseed"))
                if len(os.path.basename(f).split("_")) >= 3
            }
            saved_df    = stations_df[stations_df["sta"].isin(saved_stas)].drop_duplicates("sta")
            possible_df = stations_df[~stations_df["sta"].isin(saved_stas)].drop_duplicates("sta")

            if not possible_df.empty:
                ax.scatter(
                    possible_df["lon"], possible_df["lat"],
                    c="white", s=80, marker="^",
                    edgecolors="k", linewidths=0.6, zorder=11,
                    label="Possible waveform stations",
                    **scatter_kwargs,
                )
            if not saved_df.empty:
                ax.scatter(
                    saved_df["lon"], saved_df["lat"],
                    c="darkgreen", s=140, marker="^",
                    edgecolors="black", linewidths=1.2, zorder=13,
                    label="Saved waveform stations",
                    **scatter_kwargs,
                )
                for _, r in saved_df.iterrows():
                    text_kwargs = dict(transform=ccrs.PlateCarree()) if use_cartopy else {}
                    ax.text(
                        r["lon"] + 0.015, r["lat"] + 0.015, r["sta"],
                        fontsize=8, color="black", zorder=14, fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.1", facecolor="white",
                            alpha=0.7, lw=0.5,
                        ),
                        **text_kwargs,
                    )

        # ── Events ─────────────────────────────────────────────────────
        sc = ax.scatter(
            df_plot["lon"], df_plot["lat"],
            s=60, c=df_plot["magnitude"],
            cmap="inferno_r",
            edgecolors="k", alpha=0.9, zorder=12, label="Events",
            **scatter_kwargs,
        )
        plt.colorbar(sc, label="Magnitude", fraction=0.046, pad=0.04)

        if use_cartopy:
            add_north_arrow(ax, length=0.08, fontsize=16)

        ax.legend(loc="upper right", markerscale=1.0)

        dcs       = state.get("queried_dcs") or [state.get("datacenter", "Unknown")]
        title_str = ", ".join(dcs)
        if len(title_str) > 50:
            title_str = title_str[:47] + "…"
        ax.set_title(f"Seismic events & stations ({title_str})")

        path = os.path.join(self.output_dir, "event_map.jpg")
        plt.savefig(path, bbox_inches="tight", dpi=300, format="jpeg")
        plt.close()
        _log(self.name, f"Saved map → {path}")
        return path

    def _plot_event_timeline(self, df_plot: pd.DataFrame) -> str:
        """
        Plot event origin time vs depth, coloured by magnitude.

        The y-axis is inverted so deeper events appear lower, matching
        geological convention.  Returns the saved JPEG path.
        """
        times = [datetime.fromtimestamp(float(ts)) for ts in df_plot["time"]]

        fig, ax = plt.subplots(figsize=(10, 4))
        sc = ax.scatter(
            times, df_plot["depth"],
            alpha=0.6, c=df_plot["magnitude"],
            cmap="inferno_r", edgecolors="k",
        )
        fig.colorbar(sc, label="Magnitude")

        ax.set_title("Event timeline vs depth")
        ax.set_xlabel("Date")
        ax.set_ylabel("Depth (km)")
        ax.invert_yaxis()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        path = os.path.join(self.output_dir, "event_timeline.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300, format="jpeg")
        plt.close(fig)
        _log(self.name, f"Saved timeline → {path}")
        return path

    # ------------------------------------------------------------------
    # Node: plot_results
    # ------------------------------------------------------------------

    def _plot_results_node(self, state: TremorsState) -> TremorsState:
        """
        Orchestrate data loading and figure generation.

        Produces:
        - ``event_map.jpg``      – geographic scatter map
        - ``event_timeline.jpg`` – depth vs time scatter
        """
        if state.get("status") == "Failed":
            return state

        _log(self.name, "Generating plots…")
        try:
            df_plot = self._load_and_merge_data(state)
            if df_plot is None:
                return {**state, "status": "Success (No Events with Magnitude)"}

            stations_df = self._load_stations()
            params      = state.get("search_params", {})

            plots = [
                self._plot_event_map(df_plot, stations_df, params, state),
                self._plot_event_timeline(df_plot),
            ]
            return {**state, "plots": plots, "status": "Plots Generated"}

        except Exception as exc:
            _log(self.name, f"Plotting failed: {exc}")
            traceback.print_exc()
            return {**state, "error": f"Plotting failed: {exc}"}

    # ------------------------------------------------------------------
    # Node: retrieve_waveforms  (event mode)
    # ------------------------------------------------------------------

    def _retrieve_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Download per-event waveforms for up to 5 events.

        For each event the agent:
        1. Tries the source DC first, then ``_WAVEFORM_FALLBACK_DCS``.
        2. Queries stations within ``radius`` degrees of the epicentre.
        3. Downloads waveforms channel-by-channel (grouped by location code)
           so partial failures don't abort the whole event.
        4. Saves one ``.mseed`` file per trace, named
           ``{evid}_{net}_{sta}_{loc}_{chan}.mseed``.

        Inventory metadata is accumulated across all successful events and
        written to ``WAVEFORM_*.PARQUET`` tables via ``inventory_to_kbcore``.
        """
        if state.get("status") == "Failed":
            return state

        params = state.get("search_params", {})
        if not (params.get("get_waveforms") or params.get("plot_waveforms")):
            _log(self.name, "Waveform retrieval not requested. Skipping.")
            return state

        _log(self.name, "Starting waveform retrieval…")

        tables = state.get("metadata_tables", {})
        if "event" not in tables or "origin" not in tables:
            _log(self.name, "No event/origin tables found. Skipping waveforms.")
            return state

        try:
            events_df  = pd.read_parquet(tables["event"])
            origins_df = pd.read_parquet(tables["origin"])
        except Exception as exc:
            _log(self.name, f"Error reading parquet: {exc}")
            return state

        if events_df.empty:
            return state

        net_param      = params.get("net")
        waveform_limit = params.get("waveform_limit", 10)
        pre_event_sec  = float(params.get("pre_event_sec",  30.0))
        post_event_sec = float(params.get("post_event_sec", 600.0))

        radius_val = float(params.get("radius", 2.0))
        if params.get("radius_unit", "") == "km":
            radius_val /= 111.32

        saved_mseed:             List[str]       = []
        waveform_metadata_files: Dict[str, str]  = {}
        master_inventory = Inventory(networks=[], source="Tremors")

        for _, event_row in events_df.head(5).iterrows():
            evid   = event_row["evid"]
            prefor = event_row["prefor"]
            dc     = event_row["datacenter"]

            origin_row = origins_df[origins_df["orid"] == prefor]
            if origin_row.empty:
                origin_row = origins_df[origins_df["evid"] == evid].head(1)
            if origin_row.empty:
                continue

            origin_row = origin_row.iloc[0]
            ev_lat  = float(origin_row["lat"])
            ev_lon  = float(origin_row["lon"])
            ev_time = UTCDateTime(float(origin_row["time"]))

            _log(self.name, f"Retrieving waveforms for Event {evid} ({dc})…")

            dc_priority    = [dc] + [fb for fb in _WAVEFORM_FALLBACK_DCS if fb != dc]
            waveform_found = False

            for try_dc in dc_priority:
                if waveform_found:
                    break

                client = _make_fdsn_client(try_dc)
                if client is None:
                    continue

                try:
                    station_kwargs: dict = {
                        "latitude":  ev_lat,
                        "longitude": ev_lon,
                        "minradius": 0,
                        "maxradius": radius_val,
                        "channel":   "BH?,HH?,EH?,HN?,EN?,SH?",
                        "level":     "channel",
                        "starttime": ev_time - pre_event_sec,
                        "endtime":   ev_time + post_event_sec,
                    }
                    if net_param:
                        station_kwargs["network"] = net_param

                    inventory = client.get_stations(**station_kwargs)
                    if not inventory:
                        continue

                    n_stations = len(inventory.get_contents()["stations"])
                    _log(self.name, f"{try_dc}: {n_stations} stations found.")

                    dc_trace_count = 0

                    for net in inventory:
                        for sta in net:
                            if dc_trace_count >= waveform_limit:
                                break

                            loc_groups: Dict[str, List[str]] = defaultdict(list)
                            for cha in sta.channels:
                                loc_groups[cha.location_code].append(cha.code)

                            for loc, chans in loc_groups.items():
                                if dc_trace_count >= waveform_limit:
                                    break
                                chan_str = ",".join(sorted(set(chans)))
                                try:
                                    st = client.get_waveforms(
                                        network=net.code,
                                        station=sta.code,
                                        location=loc,
                                        channel=chan_str,
                                        starttime=ev_time - pre_event_sec,
                                        endtime=ev_time + post_event_sec,
                                        attach_response=True,
                                    )
                                    for tr in st:
                                        loc_code = tr.stats.location or "--"
                                        fname    = (
                                            f"{evid}_{tr.stats.network}_"
                                            f"{tr.stats.station}_{loc_code}_"
                                            f"{tr.stats.channel}.mseed"
                                        )
                                        fpath = os.path.join(self.output_dir, fname)
                                        tr.write(fpath, format="MSEED")
                                        if fpath not in saved_mseed:
                                            saved_mseed.append(fpath)
                                        dc_trace_count += 1

                                    waveform_found = True

                                except Exception:
                                    continue

                        if dc_trace_count >= waveform_limit:
                            break

                    if waveform_found:
                        _log(
                            self.name,
                            f"Event {evid}: saved {dc_trace_count} traces from {try_dc}.",
                        )
                        master_inventory.networks.extend(inventory.networks)

                except Exception as exc:
                    err = f"{try_dc} error for event {evid}: {exc}"
                    if "No data available" in str(exc) and net_param:
                        err += f" (network '{net_param}' may not be hosted at {try_dc})"
                    _log(self.name, err)

            if not waveform_found:
                _log(self.name, f"Event {evid}: no waveforms found at any DC.")

        # ── Write inventory metadata tables ───────────────────────────
        if master_inventory.networks:
            _log(self.name, "Generating waveform metadata tables…")
            for name, df in inventory_to_kbcore(
                master_inventory, datacenter=state.get("datacenter", "-"), extended=True
            ).items():
                if not df.empty:
                    path = os.path.join(
                        self.output_dir, f"WAVEFORM_{name}.parquet".upper()
                    )
                    df.to_parquet(path, index=False)
                    waveform_metadata_files[name] = path
                    _log(self.name, f"Saved {name} → {path}")

        return {
            **state,
            "waveforms_saved":   saved_mseed,
            "waveform_metadata": waveform_metadata_files,
        }

    # ------------------------------------------------------------------
    # Node: plot_waveforms
    # ------------------------------------------------------------------

    def _plot_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Plot per-event waveforms using ObsPy's built-in Stream.plot().

        Files are grouped by event ID (first underscore-separated field of
        the MiniSEED filename) so each event gets one multi-trace figure.
        """
        if state.get("status") == "Failed":
            return state

        if not state.get("search_params", {}).get("plot_waveforms"):
            return state

        mseed_files = state.get("waveforms_saved", [])
        if not mseed_files:
            return state

        _log(self.name, "Plotting event waveforms…")

        events_map: Dict[str, List[str]] = defaultdict(list)
        for path in mseed_files:
            evid = os.path.basename(path).split("_")[0]
            events_map[evid].append(path)

        saved_plots: List[str] = []

        for evid, paths in events_map.items():
            try:
                st = Stream()
                for p in paths:
                    st += read(p)

                outfile = os.path.join(self.output_dir, f"waveforms_{evid}.png")
                st.plot(outfile=outfile, number_of_ticks=5)
                saved_plots.append(outfile)
                _log(self.name, f"Saved waveform plot → {outfile}")

            except Exception as exc:
                _log(self.name, f"Error plotting event {evid}: {exc}")

        return {**state, "waveform_plots": saved_plots}

    # ------------------------------------------------------------------
    # Continuous-mode helpers
    # ------------------------------------------------------------------

    def _fetch_inventory_for_continuous(
        self,
        params: dict,
        datacenter: Optional[str] = None,
    ) -> Tuple[Optional[Inventory], Optional[str]]:
        """
        Pull a channel-level inventory from *datacenter* (or the first
        available DC in ``_INVENTORY_DC_PRIORITY``) for the time window
        and spatial/NSLC filters in *params*.

        The inventory is used to drive the bulk continuous download rather
        than relying on a user-supplied stations file.

        Returns
        -------
        (Inventory, str) or (None, None)
            The inventory **and the DC that actually served it**, which is not
            necessarily *datacenter* — the caller must use the returned DC for
            the waveform requests and for provenance. A planner may name a
            catalog-only node (ISC has no station *or* dataselect service): the
            fallback then supplies the inventory, and sending the bulk requests
            to the originally-named DC would download nothing while still
            reporting success, and would stamp the ``WAVEFORM_*`` tables with a
            datacenter that never served them.
        """
        t_start = UTCDateTime(params.get("min_date", "2016-01-01T00:00:00"))
        t_end   = UTCDateTime(params.get("max_date", "2016-01-03T00:00:00"))

        station_query: dict = {
            "starttime": t_start,
            "endtime":   t_end,
            "network":   params.get("net",  "*"),
            "station":   params.get("sta",  "*"),
            "location":  params.get("loc",  "*"),
            "channel":   params.get("chan", "BH?,HH?,EH?,HN?,EN?,SH?"),
            "level":     "channel",
        }
        if "min_lat" in params:
            station_query.update(
                minlatitude=params["min_lat"],
                maxlatitude=params["max_lat"],
                minlongitude=params["min_lon"],
                maxlongitude=params["max_lon"],
            )

        # User-specified DC first, then the standard priority list
        dc_list: List[str] = []
        if datacenter:
            dc_list.append(datacenter)
        dc_list += [dc for dc in _INVENTORY_DC_PRIORITY if dc != datacenter]

        for dc in dc_list:
            _log(self.name, f"Fetching inventory from {dc}…")
            client = _make_fdsn_client(dc)
            if client is None:
                continue
            try:
                inv    = client.get_stations(**station_query)
                n_chan = len(inv.get_contents()["channels"])
                _log(self.name, f"{dc}: {n_chan} channels in inventory.")
                return inv, dc
            except Exception as exc:
                _log(self.name, f"{dc} inventory fetch failed: {exc}")
                continue

        return None, None

    def _inventory_to_station_requests(
        self,
        inv_tables: Dict[str, pd.DataFrame],
        t_start: UTCDateTime,
        t_end: UTCDateTime,
        datacenter: str,
    ) -> List[dict]:
        """
        Join ``sitechan ⋈ site`` on ``sta`` and vectorize into the
        request-dict format expected by ``DailyBulkWaveforms._build_bulk_tasks``.

        Uses the extended columns (``net``, ``loc``, ``datacenter``) that
        ``inventory_to_kbcore(..., extended=True)`` already provides, so no
        manual enrichment is needed.

        Active-channel filtering is applied using the ``ondate``/``offdate``
        jdate integers to avoid requesting data for stations outside the
        query time window.
        """
        site_df     = inv_tables.get("site",     pd.DataFrame())
        sitechan_df = inv_tables.get("sitechan", pd.DataFrame())

        if site_df.empty or sitechan_df.empty:
            _log(self.name, "site or sitechan table is empty – cannot build requests.")
            return []

        # Join: sitechan carries net/sta/chan/loc; site adds lat/lon/elev
        merged = pd.merge(
            sitechan_df,
            site_df[["sta", "lat", "lon", "elev"]],
            on="sta",
        )
        _log(self.name, f"site ⋈ sitechan join: {len(merged)} channel rows.")

        # Filter to channels active during the requested time window.
        # ondate/offdate are YYYYDDD integers; 2286324 is the KBCore
        # sentinel for "still open".
        req_jstart = int(t_start.strftime("%Y%j"))
        req_jend   = int(t_end.strftime("%Y%j"))
        before     = len(merged)
        merged = merged[
            (merged["ondate"]  <= req_jend) &
            (merged["offdate"] >= req_jstart)
        ]
        _log(
            self.name,
            f"Active-channel filter ({req_jstart}–{req_jend}): "
            f"{before} → {len(merged)} rows.",
        )

        if merged.empty:
            _log(self.name, "No active channels in requested time window.")
            return []

        requests: List[dict] = []
        for _, row in merged.iterrows():
            loc_raw = str(row.get("loc", "--"))
            chan    = str(row["chan"]).upper()
            # Use the datacenter stamped on the row when available so that
            # mixed-DC inventories route each request to the correct node.
            row_dc  = str(row.get("datacenter", datacenter))
            req = {
                "net":             str(row["net"]).upper(),
                "sta":             str(row["sta"]).upper(),
                "chan":            chan,
                "loc":             loc_raw,
                "loc_select":      DailyBulkWaveforms._normalize_loc(loc_raw),
                "channel_request": DailyBulkWaveforms._channel_request(chan),
                "channel_select":  DailyBulkWaveforms._channel_select(chan),
                "datacenter":      row_dc,
                "tstart":          t_start,
                "tend":            t_end,
            }
            requests.append(req)

        _log(self.name, f"Built {len(requests)} channel requests from inventory.")
        return requests

    # ------------------------------------------------------------------
    # Node: retrieve_continuous_waveforms
    # ------------------------------------------------------------------

    def _retrieve_continuous_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Retrieve continuous waveform data driven entirely by an FDSN inventory
        pull rather than a user-supplied stations file.

        Pipeline
        --------
        1. Resolve the inventory datacenter (user hint → EARTHSCOPE → fallbacks).
        2. Fetch a channel-level inventory via ``_fetch_inventory_for_continuous``.
        3. Convert to kbcore tables with ``inventory_to_kbcore(..., extended=True)``
           and persist all tables as ``WAVEFORM_*.PARQUET``.
        4. Join ``site ⋈ sitechan`` and apply active-channel filtering to build
           a vectorized request list via ``_inventory_to_station_requests``.
        5. Inject requests into ``DailyBulkWaveforms`` (bypassing file parsing)
           and run the ``Scheduler`` multiprocessing fan-out.

        If a ``stations_file`` is explicitly provided in ``search_params`` and
        the file exists on disk, it is used instead of the inventory pull so
        that power users can override the automatic discovery.
        """
        if state.get("status") == "Failed":
            return state

        params = state.get("search_params", {})
        if not params.get("get_continuous_waveforms"):
            return state

        _log(self.name, "Starting continuous waveform retrieval…")

        try:
            t_start = UTCDateTime(params.get("min_date", "2016-01-01T00:00:00"))
            t_end   = UTCDateTime(params.get("max_date", "2016-01-03T00:00:00"))

            # Resolve DC for inventory pull
            user_dc = params.get("datacenter", "EARTHSCOPE")
            inv_dc  = user_dc if user_dc in WELL_KNOWN_NODES else "EARTHSCOPE"

            # ── Shared DailyBulkWaveforms args ─────────────────────────
            args            = argparse.Namespace()
            args.outdir     = self.output_dir
            args.parallel   = params.get("parallel",   4)
            args.bulk_chunk = params.get("bulk_chunk", 50)
            args.dir_date   = params.get("dir_date",   False)
            args.dir_stat   = params.get("dir_stat",   False)
            args.response   = params.get("response",   False)
            args.download   = True
            args.stations   = None   # signals constructor to skip file parsing

            waveform_metadata_files: Dict[str, str] = {}

            # ── Path A: explicit stations file override ────────────────
            stations_file = params.get("stations_file", "")
            if stations_file and os.path.isfile(stations_file):
                _log(self.name, f"Using explicit stations file: {stations_file}")
                args.stations = stations_file
                wave_list = DailyBulkWaveforms(args)

            # ── Path B: inventory-driven (default) ────────────────────
            else:
                _log(self.name, f"No stations file. Pulling inventory from {inv_dc}…")

                inventory, served_dc = self._fetch_inventory_for_continuous(
                    params, datacenter=inv_dc
                )
                if not inventory or not inventory.networks:
                    return {
                        **state,
                        "error":  (
                            f"Could not retrieve inventory from {inv_dc} "
                            "or fallback DCs."
                        ),
                        "status": "Failed",
                    }

                # Everything downstream must follow the DC that actually served
                # the inventory, not the one originally asked for — see
                # _fetch_inventory_for_continuous.
                if served_dc != inv_dc:
                    _log(
                        self.name,
                        f"Inventory came from {served_dc}, not {inv_dc}; "
                        f"waveform requests and provenance will use {served_dc}.",
                    )
                    inv_dc = served_dc

                # Convert and persist all kbcore tables
                # extended=True gives net/loc in sitechan and datacenter on
                # every row — exactly what _inventory_to_station_requests needs
                inv_tables = inventory_to_kbcore(
                    inventory, datacenter=inv_dc, extended=True
                )
                for name, df in inv_tables.items():
                    if not df.empty:
                        path = os.path.join(
                            self.output_dir, f"WAVEFORM_{name}.parquet".upper()
                        )
                        df.to_parquet(path, index=False)
                        waveform_metadata_files[name] = path
                        _log(self.name, f"Saved {name} ({len(df)} rows) → {path}")

                # Build vectorised request list from site ⋈ sitechan join
                channel_requests = self._inventory_to_station_requests(
                    inv_tables, t_start, t_end, datacenter=inv_dc
                )
                if not channel_requests:
                    return {
                        **state,
                        "error":  (
                            "No active channel requests in the requested "
                            "time window after filtering."
                        ),
                        "status": "Failed",
                    }

                # Inject requests directly — bypass _read_station_file
                wave_list           = DailyBulkWaveforms(args)
                wave_list.requests  = channel_requests
                wave_list.dl_params = wave_list._build_bulk_tasks()

            # ── Run the download ───────────────────────────────────────
            scheduler = Scheduler(wave_list.parallel)
            scheduler.start(iter(wave_list.dl_params))

            # Collect all written MiniSEED files
            saved_mseed: List[str] = []
            for root, _dirs, files in os.walk(self.output_dir):
                saved_mseed.extend(
                    os.path.join(root, f) for f in files if f.endswith(".mseed")
                )

            _log(self.name, f"Download complete. {len(saved_mseed)} traces saved.")
            return {
                **state,
                "continuous_waveforms_saved": saved_mseed,
                "waveform_metadata":          waveform_metadata_files,
                "status":                     "Success",
            }

        except Exception as exc:
            _log(self.name, f"Continuous waveform retrieval failed: {exc}")
            traceback.print_exc()
            return {
                **state,
                "error":  f"Continuous waveform retrieval failed: {exc}",
                "status": "Failed",
            }

    # ------------------------------------------------------------------
    # Node: plot_continuous_waveforms
    # ------------------------------------------------------------------

    def _plot_continuous_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Read all saved continuous MiniSEED files into a single ObsPy Stream
        and produce one composite waveform plot.
        """
        if state.get("status") == "Failed":
            return state

        mseed_files = state.get("continuous_waveforms_saved", [])
        if not mseed_files:
            _log(self.name, "No continuous waveforms to plot.")
            return state

        _log(self.name, "Plotting continuous waveforms…")
        saved_plots: List[str] = []

        try:
            st = Stream()
            for p in mseed_files:
                st += read(p)

            outfile = os.path.join(self.output_dir, "continuous_waveforms.png")
            st.plot(outfile=outfile, number_of_ticks=5, handle=True)
            saved_plots.append(outfile)
            _log(self.name, f"Saved continuous waveform plot → {outfile}")

        except Exception as exc:
            _log(self.name, f"Error plotting continuous waveforms: {exc}")

        return {**state, "continuous_waveform_plots": saved_plots}


# =============================================================================
# Multiprocessing bulk-download pipeline
# =============================================================================

@contextlib.contextmanager
def _detached_main_module():
    """
    Temporarily hide ``__main__``'s identity so workers don't re-import it.

    ``fork`` children inherit the parent's memory and import nothing.
    ``forkserver`` (and ``spawn``) children instead **re-import the parent's
    ``__main__`` module**, which is why the stdlib tells you to guard entry
    points with ``if __name__ == "__main__":``. Without that guard, a plain
    ``python my_script.py`` that kicks off a bulk download has its entire top
    level re-executed inside every worker — a duplicated download, followed by
    ``RuntimeError: An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase`` when the re-run tries
    to spawn workers of its own.

    Requiring the guard would be the usual answer, but it is a footgun to hand a
    library's users, and the workers do not need ``__main__`` at all: ``PullWave``
    lives in this installed module, and the queued payload is plain dicts of
    str/float/``UTCDateTime`` (see ``_make_task``) — no user-defined classes or
    closures cross the process boundary.

    ``multiprocessing.spawn.get_preparation_data`` decides how the child should
    set up its main module by looking at ``__main__.__spec__.name`` and
    ``__main__.__file__``; when neither is available it "leaves it alone
    entirely" — which is exactly how notebook and REPL sessions already behave.
    Clearing both for the duration of process startup puts scripts on that same
    path, then restores them so the caller's module is untouched afterwards.
    """
    main = sys.modules.get("__main__")
    if main is None:                      # pragma: no cover — no main to detach
        yield
        return

    had_file   = hasattr(main, "__file__")
    saved_file = getattr(main, "__file__", None)
    saved_spec = getattr(main, "__spec__", None)
    try:
        if had_file:
            del main.__file__
        main.__spec__ = None
        yield
    finally:
        if had_file:
            main.__file__ = saved_file
        main.__spec__ = saved_spec


class Scheduler:
    """
    Fan-out coordinator for chunked FDSN bulk-waveform downloads.

    Puts all task dicts onto a multiprocessing Queue and spins up
    ``nproc`` ``PullWave`` worker processes.  Each worker consumes
    tasks until it reads the sentinel ``None``.

    Parameters
    ----------
    nproc:
        Number of parallel worker processes.
    """

    def __init__(self, nproc: int):
        self._queue   = _MP_CTX.Queue()
        self._nproc   = nproc
        self._workers: List[PullWave] = [PullWave(self._queue) for _ in range(nproc)]

    def start(self, tasks) -> None:
        """Enqueue all tasks, post the sentinel, start and join workers."""
        queue_count = sum(
            1 for task in tasks if not self._queue.put(copy.deepcopy(task))
        )
        self._queue.put(None)  # sentinel

        _log("Scheduler", f"Queued {queue_count} bulk requests across {self._nproc} threads.")
        logging.info(f"Queued {queue_count} bulk requests / {self._nproc} threads")

        # forkserver children re-import __main__; detach it so an unguarded
        # caller script isn't re-executed inside every worker.  See
        # _detached_main_module.
        with _detached_main_module():
            for w in self._workers:
                w.start()
        for w in self._workers:
            w.join()


class PullWave(_MP_CTX.Process):
    """
    Worker process: consumes bulk-download tasks from a shared Queue.

    Each task contains one FDSN bulk request (a list of
    ``(net, sta, loc, chan, t_start, t_end)`` tuples) for a single
    datacenter.  The worker fetches the stream, then writes one file per
    requested station/channel/day.

    Derives from the ``forkserver`` context's ``Process`` (see ``_MP_CTX``), so
    everything crossing the queue must be picklable — it is: tasks are plain
    dicts of str/float/``UTCDateTime`` plus a ``writer`` dict, and the FDSN
    ``Client`` objects are built lazily *inside* the worker by ``_get_client``.
    """

    def __init__(self, queue):
        super().__init__(name="PullWave")
        self._queue                                   = queue
        self._clients:          Dict[str, Client]     = {}   # DC → Client (reused)
        self._response_written: set                   = set()

    def run(self) -> None:
        while True:
            params = self._queue.get()
            if params is None:
                self._queue.put(None)  # re-post sentinel for sibling workers
                break
            self._pull_data(params)

    def _get_client(self, datacenter: str) -> Client:
        """Return a cached FDSN client, creating one on first access."""
        if datacenter not in self._clients:
            self._clients[datacenter] = obspy.clients.fdsn.Client(datacenter)
        return self._clients[datacenter]

    def _pull_data(self, param: dict) -> None:
        """Fetch one bulk chunk and dispatch each request to the writer."""
        datacenter = param["datacenter"]
        bulk       = param["bulk"]
        requests   = param["requests"]
        client     = self._get_client(datacenter)

        try:
            stream = client.get_waveforms_bulk(bulk)
            msg    = (
                f"Fetched: dc={datacenter} chunk={param['chunk_id']} "
                f"requests={len(requests)}"
            )
        except Exception as exc:
            msg = (
                f"Skipped: dc={datacenter} chunk={param['chunk_id']} "
                f"requests={len(requests)} error={exc}"
            )
            print(msg, file=sys.stderr)
            logging.info(msg)
            return

        print(msg, file=sys.stderr)
        logging.info(msg)

        for request in requests:
            self._write_request(client, stream, request, param["writer"])

    def _write_request(
        self,
        client:  Client,
        stream:  obspy.Stream,
        request: dict,
        writer:  dict,
    ) -> None:
        """
        Slice the bulk stream to one request's NSLC/time window and write
        it to MiniSEED.  Optionally fetches and writes the StationXML
        response file.
        """
        st = stream.select(
            network=request["net"],
            station=request["sta"],
            location=request["loc_select"],
            channel=request["channel_select"],
        ).copy()

        if len(st) == 0:
            msg = (
                f"No data: {request['net']} {request['sta']} {request['loc']} "
                f"{request['channel_request']} {request['tstart']} {request['tend']}"
            )
            print(msg, file=sys.stderr)
            logging.info(msg)
            return

        st.trim(request["tstart"], request["tend"], nearest_sample=False)
        if len(st) == 0:
            return

        if writer.get("response"):
            self._write_response_file(
                client=client, request=request, outdir=writer["outdir"]
            )

        write_daily_per_station_all_channels(
            st=st,
            outdir=writer["outdir"],
            dir_date=writer["dir_date"],
            dir_stat=writer["dir_stat"],
            loc=request["loc"],
            requested_chan=request["chan"],
            overwrite=writer["overwrite"],
        )

    def _write_response_file(
        self, client: Client, request: dict, outdir: str
    ) -> None:
        """
        Download and save the StationXML response for one NSLC if it has
        not already been written during this worker's lifetime.
        """
        resp_key = (
            request["datacenter"], request["net"],
            request["sta"],       request["loc"],  request["chan"],
        )
        if resp_key in self._response_written:
            return

        resp_dir = os.path.join(
            station_output_dir(
                outdir, request["net"], request["sta"], use_station_dir=True
            ),
            "resp",
        )
        does_dir_exist(resp_dir)

        resp_file = os.path.join(
            resp_dir,
            f"RESP_{request['net']}_{request['sta']}_{request['chan']}.xml",
        )
        if os.path.isfile(resp_file) and os.path.getsize(resp_file) > 0:
            self._response_written.add(resp_key)
            return

        try:
            inv = client.get_stations(
                network=request["net"],
                station=request["sta"],
                location=request["loc"],
                channel=request["channel_request"],
                starttime=request["tstart"],
                endtime=request["tend"],
                level="response",
            )
            inv.write(resp_file, format="STATIONXML")
            self._response_written.add(resp_key)
            msg = f"Saved response: {resp_file}"
        except Exception as exc:
            msg = f"Skipped response: {resp_file} error={exc}"

        print(msg, file=sys.stderr)
        logging.info(msg)


class DailyBulkWaveforms:
    """
    Build FDSN bulk-download task chunks from either a station-list text
    file or an injected request list.

    Station file format (one channel per line, ``#`` lines ignored)::

        <net> <sta> <chan> <loc> <datacenter> <tstart> <tend>

    When ``argv.stations`` is ``None`` or points to a non-existent file the
    constructor initialises ``self.requests = []`` and the caller is expected
    to populate it directly before calling ``_build_bulk_tasks()``.

    Tasks are chunked so that no single FDSN bulk request exceeds
    ``bulk_chunk`` station-days.  Requests are skipped if the target output
    file already exists and is larger than 4 KB (i.e. non-empty).

    Parameters
    ----------
    argv:
        ``argparse.Namespace`` (or compatible) with attributes:
        ``outdir``, ``parallel``, ``bulk_chunk``, ``dir_date``, ``dir_stat``,
        ``response``, ``stations``, ``download``.
    """

    def __init__(self, argv: argparse.Namespace):
        self.cwd        = os.getcwd()
        self.outdir     = os.path.join(self.cwd, argv.outdir)
        self.parallel   = argv.parallel
        self.download   = argv.download
        self.dir_date   = argv.dir_date
        self.dir_stat   = argv.dir_stat
        self.bulk_chunk = argv.bulk_chunk
        self.writer: dict = {
            "outdir":    self.outdir,
            "dir_date":  self.dir_date,
            "dir_stat":  self.dir_stat,
            "overwrite": False,
            "response":  argv.response,
        }

        stations_file = getattr(argv, "stations", None)
        if stations_file and os.path.isfile(stations_file):
            self.requests = self._read_station_file(stations_file)
        else:
            # Caller will populate self.requests then call _build_bulk_tasks()
            self.requests = []

        self.dl_params = self._build_bulk_tasks()

    # ── Station-file reader ────────────────────────────────────────────

    def _read_station_file(self, station_file: str) -> List[dict]:
        if not os.path.isfile(station_file):
            raise FileNotFoundError(
                f"--stations must point to a file. Not found: {station_file}"
            )

        requests: List[dict] = []
        with open(station_file) as fobj:
            for lineno, raw in enumerate(fobj, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 7:
                    raise ValueError(
                        "Each line must contain "
                        "<net sta chan loc datacenter tstart tend>. "
                        f"Invalid line {lineno}: {line!r}"
                    )

                net, sta, chan, loc, datacenter, tstart, tend = parts
                req = {
                    "net":             net.upper(),
                    "sta":             sta.upper(),
                    "chan":            chan.upper(),
                    "loc":             loc,
                    "loc_select":      self._normalize_loc(loc),
                    "channel_request": self._channel_request(chan.upper()),
                    "channel_select":  self._channel_select(chan.upper()),
                    "datacenter":      datacenter,
                    "tstart":          obspy.UTCDateTime(tstart),
                    "tend":            obspy.UTCDateTime(tend),
                }
                if req["tend"] <= req["tstart"]:
                    raise ValueError(
                        f"tend must be after tstart on line {lineno}: {line!r}"
                    )
                requests.append(req)

        if not requests:
            raise ValueError("No valid station requests found in file.")
        return requests

    # ── Task builder ───────────────────────────────────────────────────

    def _build_bulk_tasks(self) -> List[dict]:
        """
        Split each request into daily chunks, skip files that already exist,
        group by datacenter, and bundle into bulk tasks of at most
        ``bulk_chunk`` station-days each.
        """
        if not self.requests:
            return []

        grouped: Dict[str, List[dict]] = defaultdict(list)
        for req in self.requests:
            for chunk in self._split_request_into_day_chunks(req):
                if self._request_has_missing_days(chunk):
                    grouped[chunk["datacenter"]].append(chunk)

        tasks: List[dict] = []
        for datacenter, reqs in sorted(grouped.items()):
            reqs = sorted(
                reqs,
                key=lambda x: (x["tstart"], x["tend"], x["net"], x["sta"], x["loc"]),
            )
            chunk_reqs: List[dict] = []
            chunk_days: int        = 0
            chunk_id:   int        = 1

            for req in reqs:
                req_days = self._request_days(req)
                if chunk_reqs and chunk_days + req_days > self.bulk_chunk:
                    tasks.append(self._make_task(datacenter, chunk_id, chunk_reqs))
                    chunk_reqs = []
                    chunk_days = 0
                    chunk_id  += 1
                chunk_reqs.append(req)
                chunk_days += req_days

            if chunk_reqs:
                tasks.append(self._make_task(datacenter, chunk_id, chunk_reqs))

        return tasks

    def _split_request_into_day_chunks(self, request: dict) -> List[dict]:
        """Subdivide a request spanning many days into ``bulk_chunk``-day pieces."""
        chunk_seconds = self.bulk_chunk * 86_400
        chunks: List[dict] = []
        t0 = request["tstart"]
        while t0 < request["tend"]:
            t1 = min(t0 + chunk_seconds, request["tend"])
            c  = copy.deepcopy(request)
            c["tstart"] = t0
            c["tend"]   = t1
            chunks.append(c)
            t0 = t1
        return chunks

    def _make_task(
        self, datacenter: str, chunk_id: int, requests: List[dict]
    ) -> dict:
        return {
            "datacenter": datacenter,
            "chunk_id":   f"{datacenter}:{chunk_id}",
            "bulk": [
                (
                    r["net"], r["sta"], r["loc"],
                    r["channel_request"], r["tstart"], r["tend"],
                )
                for r in requests
            ],
            "requests": requests,
            "writer":   self.writer,
        }

    # ── Utility methods ────────────────────────────────────────────────

    @staticmethod
    def _request_days(request: dict) -> int:
        return max(1, int((request["tend"] - request["tstart"]) / 86_400 + 0.999_999))

    def _request_has_missing_days(self, request: dict) -> bool:
        """Return True if any daily output file for this request is absent or tiny."""
        day  = _day_start_utc(request["tstart"])
        last = _day_start_utc(request["tend"] - 0.000_001)
        while day <= last:
            outfile = build_daily_outfile(
                outdir=self.outdir,
                net=request["net"],
                sta=request["sta"],
                loc=request["loc"],
                chan=request["chan"],
                day_start=day,
                dir_date=self.dir_date,
                dir_stat=self.dir_stat,
            )
            if _should_write_file(outfile):
                return True
            day += 86_400
        return False

    @staticmethod
    def _channel_request(chan: str) -> str:
        return chan + "*" if len(chan) == 2 else chan

    @staticmethod
    def _channel_select(chan: str) -> str:
        return chan + "?" if len(chan) == 2 else chan

    @staticmethod
    def _normalize_loc(loc: str) -> str:
        return "*" if loc.strip() in ("", "--", "**") else loc.strip()


# =============================================================================
# Standalone file-system helpers
# =============================================================================

def does_dir_exist(path: str) -> None:
    """Create *path* (and any missing parents) if it does not yet exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def _day_start_utc(t: obspy.UTCDateTime) -> obspy.UTCDateTime:
    """Return midnight UTC for the calendar day containing *t*."""
    return obspy.UTCDateTime(t.year, t.month, t.day)


def station_output_dir(
    outdir: str, net: str, sta: str, use_station_dir: bool = False
) -> str:
    """Construct the output directory path for a network/station pair."""
    return os.path.join(outdir, net, sta) if use_station_dir else outdir


def build_daily_outfile(
    outdir:    str,
    net:       str,
    sta:       str,
    loc:       str,
    chan:      str,
    day_start: obspy.UTCDateTime,
    dir_date:  bool = False,
    dir_stat:  bool = False,
) -> str:
    """
    Construct the full output file path for one station/channel/day.

    Directory structure (controlled by flags)::

        dir_stat=True  → <outdir>/<net>/<sta>/
        dir_date=True  → <…>/<YYYY>/<DOY>/
        filename       → <net>.<sta>.<loc>.<chan>_<YYYY><DOY>.mseed
    """
    path = station_output_dir(outdir, net, sta, use_station_dir=dir_stat)
    if dir_date:
        year = day_start.datetime.year
        doy  = day_start.datetime.timetuple().tm_yday
        path = os.path.join(path, str(year), f"{doy:03d}")

    year  = day_start.datetime.year
    doy   = day_start.datetime.timetuple().tm_yday
    fname = f"{net}.{sta}.{loc}.{chan}_{year}{doy:03d}.mseed"
    return os.path.join(path, fname)


def _should_write_file(outfile: str) -> bool:
    """
    Return True if *outfile* is absent or suspiciously small (< 4 KB),
    indicating an incomplete or failed previous download.
    """
    if not os.path.isfile(outfile):
        return True
    size = os.path.getsize(outfile)
    if size < 4096:
        print(f"Redownload (too small, {size} B): {os.path.basename(outfile)}", file=sys.stderr)
        return True
    print(f"Exists: {os.path.basename(outfile)}", file=sys.stderr)
    return False


def write_daily_per_station_all_channels(
    st:             obspy.Stream,
    outdir:         str,
    merge:          bool          = True,
    dir_date:       bool          = False,
    dir_stat:       bool          = False,
    loc:            str           = "*",
    requested_chan: Optional[str] = None,
    overwrite:      bool          = False,
) -> None:
    """
    Write one MiniSEED file per station per UTC day, containing all
    requested channels.

    Parameters
    ----------
    st:
        Input ObsPy Stream (may contain multiple stations / channels).
    outdir:
        Root output directory.
    merge:
        If True, merge gapped traces (``method=1, fill_value=None``) before
        writing.
    dir_date:
        Organise output under ``<YYYY>/<DOY>/`` sub-directories.
    dir_stat:
        Organise output under ``<net>/<sta>/`` sub-directories.
    loc:
        Location code used for the output filename (``*`` = wildcard).
    requested_chan:
        Override the channel label in the output filename.
    overwrite:
        If True, re-write files even if they already exist and are healthy.
    """
    if not isinstance(st, obspy.Stream):
        raise TypeError(f"Expected obspy.Stream, got {type(st)}")

    work = st.copy()
    if merge:
        merged = obspy.Stream()
        for tr_id in sorted({tr.id for tr in work}):
            tmp = work.select(id=tr_id).copy()
            tmp.merge(method=1, fill_value=None)
            merged += tmp
        work = merged

    by_station: Dict[Tuple[str, str], obspy.Stream] = defaultdict(obspy.Stream)
    for tr in work:
        by_station[(tr.stats.network, tr.stats.station)] += tr

    for (net, sta), sst in by_station.items():
        if not sst:
            continue

        t0        = min(tr.stats.starttime for tr in sst)
        t1        = max(tr.stats.endtime   for tr in sst)
        day_start = _day_start_utc(t0)
        last_day  = _day_start_utc(t1)

        while day_start <= last_day:
            day_end = day_start + 86_400
            day_st  = sst.slice(day_start, day_end).copy()

            if day_st:
                if merge:
                    day_st.merge(method=1, fill_value=None)

                chan_label = requested_chan or _infer_channel_label(day_st)
                outfile    = build_daily_outfile(
                    outdir=outdir, net=net, sta=sta, loc=loc,
                    chan=chan_label, day_start=day_start,
                    dir_date=dir_date, dir_stat=dir_stat,
                )

                if overwrite or _should_write_file(outfile):
                    does_dir_exist(os.path.dirname(outfile))
                    try:
                        # split() before writing: the merge above uses
                        # fill_value=None, which represents gaps as a masked
                        # array rather than inventing samples — and MiniSEED
                        # writing rejects masked arrays outright ("Masked array
                        # writing is not supported"). Without this, every day
                        # containing a gap is silently skipped while the run
                        # still reports Success. split() turns each masked trace
                        # back into contiguous unmasked traces, which MiniSEED
                        # stores natively as separate records, so the gap is
                        # preserved rather than filled.
                        day_st.split().write(outfile, format="MSEED")
                        msg = f"Saved: {outfile}"
                    except Exception as exc:
                        msg = f"Skipped: {outfile} error={exc}"
                    print(msg, file=sys.stderr)
                    logging.info(msg)

            day_start = day_end


def _infer_channel_label(st: obspy.Stream) -> str:
    """
    Infer a compact channel label from the traces in *st*.

    Priority:
    1. If all traces share the same 2-character band/instrument prefix → return prefix.
    2. If there is only one unique channel code → return that code.
    3. Otherwise → "MULTI".
    """
    prefixes = sorted({tr.stats.channel[:2] for tr in st if tr.stats.channel})
    if len(prefixes) == 1:
        return prefixes[0]
    channels = sorted({tr.stats.channel for tr in st if tr.stats.channel})
    return channels[0] if len(channels) == 1 else "MULTI"