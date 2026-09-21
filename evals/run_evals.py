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

run_evals.py
============
Behavioral eval harness for the TREMORS agent.

Run it directly — there is no pytest in this repo:

    python evals/run_evals.py                 # offline (default), no network, no LLM
    python evals/run_evals.py -k continuous   # only cases whose name matches
    python evals/run_evals.py --live --backend anthropic --model … --base-url …

Why this exists
---------------
Routing is the model's decision (the pipeline is a deepagents tool loop), and
planning is a natural-language translation. Neither can be checked by reading
the code, so every case asserts **two independent things**. Either one alone is
foolable:

1. **Artifact invariants** — constraints *derived from the query*, not a golden
   file: the parquet tables exist, have at least N rows, and every row's origin
   time / location / magnitude falls inside what was asked for. This is what
   catches a plan that reached the datacenters and came back plausibly wrong.
2. **Tool trajectory** — the set of tools that actually executed must contain
   the required ones and must *not* contain the forbidden ones (a continuous
   request must never run ``query_cascade``).

Offline mode
------------
No network and no LLM. Three seams are replaced, and nothing else:

* ``_make_fdsn_client`` returns a fake client that reads the QuakeML fixtures in
  ``evals/fixtures/`` and **applies the query kwargs as a real FDSN service
  would**. That is the point: a bug in the ``search_params`` → FDSN kwargs
  mapping (a dropped bounding box, a magnitude sent under the wrong name) shows
  up as an invariant failure rather than passing silently.
* ``GoogleTiles`` serves a blank tile instead of fetching basemap imagery. The
  rest of the plotting path — extents, features, markers, ``savefig`` — is real.
* ``Scheduler`` records the bulk-download tasks and writes the MiniSEED files
  itself through the real :func:`build_daily_outfile`, instead of fanning out
  worker processes that would each open their own (unpatched) FDSN connection.

The stub also gets a deepagents harness profile registered under its own provider
(:func:`_register_stub_harness_profile`), so the offered tool surface is the same
trimmed set the real backends see rather than deepagents' wider default.

The stub LLM is not a scripted list of tool calls — that would make the
trajectory check a tautology. It **follows the next-step hint** that each tool
returns ("… Next call query_cascade."), so the trajectory assertion tests the
real hint chain in ``_build_tools``. If a hint is missing, misspells a tool, or
points somewhere wrong, the run derails and the case fails.

What offline mode does *not* cover, and needs ``--live``: whether a real model
follows the hints, whether the planner's structured-output path works against a
given provider, and the true multiprocess bulk-download layout.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Must precede any import that pulls in pyplot: these runs are headless.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from obspy import Catalog, Stream, Trace, UTCDateTime, read_events
from obspy.core.event import (
    Comment,
    CreationInfo,
    Event,
    EventDescription,
    Magnitude,
    Origin,
    ResourceIdentifier,
)
from obspy.core.inventory import Channel, Inventory, Network, Site, Station

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field as PydanticField

_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_HERE)
_FIXTURES = os.path.join(_HERE, "fixtures")

# Allow `python evals/run_evals.py` from a source checkout without installing.
sys.path.insert(0, os.path.join(_REPO, "src"))

from tremors.agents import tremors as T                      # noqa: E402
from tremors.utils.params import SearchParams                # noqa: E402


# ===========================================================================
# Fixtures — the synthetic catalog the fake datacenters serve
# ===========================================================================

# Deliberately wider than any case queries. Events outside a case's window are
# what prove the query kwargs are actually applied rather than ignored.
#
# (origin time, lat, lon, depth_km, magnitude)
_JAPAN_EVENTS: List[Tuple[str, float, float, float, float]] = [
    ("2020-02-11T05:12:00", 37.10, 141.20,  35.0, 5.4),
    ("2020-03-24T18:03:30", 38.55, 142.60,  12.0, 6.1),
    ("2020-06-25T04:47:10", 35.20, 140.10,  60.0, 6.8),
    ("2020-09-12T22:15:05", 41.90, 143.30,  45.0, 5.0),
    ("2020-11-02T11:38:20", 33.40, 132.80, 110.0, 7.2),
]

# Rows that must be filtered out by the service, one per filter dimension.
_JAPAN_DECOYS: List[Tuple[str, float, float, float, float]] = [
    ("2019-05-04T09:00:00", 36.00, 140.00,  20.0, 6.5),   # before the window
    ("2021-01-19T09:00:00", 36.00, 140.00,  20.0, 6.5),   # after the window
    ("2020-07-07T09:00:00", 12.00,  92.00,  20.0, 6.5),   # outside the bbox
    ("2020-07-08T09:00:00", 36.00, 140.00,  20.0, 2.1),   # below any min_mag
]

# Which datacenters report what. USGS/EMSC/ISC overlap heavily, which is how the
# no-deduplication behaviour becomes observable; GEOFON is reachable but has
# nothing; RASPISHAKE is down.
_DC_BEHAVIOR: Dict[str, str] = {
    "USGS":       "serve",
    "EMSC":       "serve",
    "ISC":        "serve",
    "GEOFON":     "empty",
    "RASPISHAKE": "raise",
}

# Per-DC slice of _JAPAN_EVENTS. Overlapping but not identical, so a case can
# assert both "the same quake appears once per reporting DC" and "DCs differ".
_DC_EVENT_INDICES: Dict[str, Sequence[int]] = {
    "USGS": (0, 1, 2, 3, 4),
    "EMSC": (1, 2, 4),
    "ISC":  (0, 2, 4),
}


def _build_fixture_catalog(dc: str) -> Catalog:
    """Synthesize the catalog *dc* serves, including the decoy events."""
    rows = [_JAPAN_EVENTS[i] for i in _DC_EVENT_INDICES[dc]] + _JAPAN_DECOYS

    catalog = Catalog()
    for seq, (when, lat, lon, depth_km, mag) in enumerate(rows):
        # A stable, DC-scoped id: real FDSN nodes issue their own ids for the
        # same earthquake, and _extract_id has to cope with that.
        base   = f"smi:{dc.lower()}.org/event/{dc}{seq:03d}"
        origin = Origin(
            resource_id=ResourceIdentifier(f"{base}/origin"),
            time=UTCDateTime(when),
            latitude=lat,
            longitude=lon,
            depth=depth_km * 1000.0,
            creation_info=CreationInfo(agency_id=dc, author=dc),
        )
        magnitude = Magnitude(
            resource_id=ResourceIdentifier(f"{base}/magnitude"),
            mag=mag,
            magnitude_type="mw",
            origin_id=origin.resource_id,
            creation_info=CreationInfo(agency_id=dc, author=dc),
        )
        event = Event(
            resource_id=ResourceIdentifier(base),
            origins=[origin],
            magnitudes=[magnitude],
            event_descriptions=[EventDescription(text=f"{dc} synthetic event {seq}")],
            creation_info=CreationInfo(agency_id=dc, author=dc),
        )
        event.preferred_origin_id    = origin.resource_id
        event.preferred_magnitude_id = magnitude.resource_id
        catalog.append(event)

    return catalog


def _fixture_stamp() -> str:
    """Hash of everything the fixture catalogs are built from."""
    payload = json.dumps(
        [_JAPAN_EVENTS, _JAPAN_DECOYS, _DC_EVENT_INDICES], sort_keys=True, default=list
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def ensure_fixtures(force: bool = False) -> Dict[str, str]:
    """
    Write one QuakeML file per serving datacenter and return {dc: path}.

    The catalogs are materialized to disk rather than held in memory so the fake
    service parses QuakeML exactly as ``obspy.clients.fdsn.Client.get_events``
    does — the resource-id and preferred-origin handling that
    :func:`tremors.utils.schema.catalog_to_kbcore` depends on is then exercised
    for real.

    The files are committed, so they are only rewritten when missing or when the
    tables above have changed. That staleness check matters: without it, editing
    ``_JAPAN_EVENTS`` would silently have no effect on any checkout that already
    has the fixtures on disk, and the suite would keep asserting against the old
    catalog. ``--refresh-fixtures`` forces a rewrite regardless.
    """
    os.makedirs(_FIXTURES, exist_ok=True)

    stamp_path = os.path.join(_FIXTURES, "fixtures.stamp")
    stamp      = _fixture_stamp()
    stale      = True
    if os.path.exists(stamp_path):
        with open(stamp_path) as handle:
            stale = handle.read().strip() != stamp

    paths = {}
    for dc in _DC_EVENT_INDICES:
        path = os.path.join(_FIXTURES, f"catalog_{dc}.xml")
        if force or stale or not os.path.exists(path):
            _build_fixture_catalog(dc).write(path, format="QUAKEML")
        paths[dc] = path

    with open(stamp_path, "w") as handle:
        handle.write(stamp + "\n")
    return paths


# ===========================================================================
# Offline seam 1 — the fake FDSN service
# ===========================================================================

class _FakeFDSNClient:
    """
    A stand-in for ``obspy.clients.fdsn.Client`` that filters like a real node.

    ``get_events`` honours ``starttime``/``endtime``/``minmagnitude``/``limit``
    and the bounding box. Applying the filters rather than replaying a canned
    answer is what makes the artifact invariants meaningful offline: if the
    agent stops sending a bounding box, out-of-area events arrive and the
    invariant fails.
    """

    def __init__(self, dc: str, recorder: "_CallRecorder"):
        self.dc       = dc
        self.recorder = recorder

    # ── Events ────────────────────────────────────────────────────────
    def get_events(self, **kwargs):
        self.recorder.events.append((self.dc, kwargs))
        behavior = _DC_BEHAVIOR.get(self.dc, "empty")

        if behavior == "raise":
            raise RuntimeError(f"{self.dc}: simulated service outage")
        if behavior == "empty" or self.dc not in _DC_EVENT_INDICES:
            return Catalog()

        catalog = read_events(os.path.join(_FIXTURES, f"catalog_{self.dc}.xml"))

        starttime    = kwargs.get("starttime")
        endtime      = kwargs.get("endtime")
        minmagnitude = kwargs.get("minmagnitude")
        limit        = kwargs.get("limit")
        min_lat, max_lat = kwargs.get("minlatitude"),  kwargs.get("maxlatitude")
        min_lon, max_lon = kwargs.get("minlongitude"), kwargs.get("maxlongitude")

        kept = Catalog()
        for event in catalog:
            origin = event.preferred_origin() or event.origins[0]
            mag    = (event.preferred_magnitude() or event.magnitudes[0]).mag

            if starttime is not None and origin.time < starttime:
                continue
            if endtime is not None and origin.time > endtime:
                continue
            if minmagnitude is not None and mag < minmagnitude:
                continue
            if min_lat is not None and not (min_lat <= origin.latitude  <= max_lat):
                continue
            if min_lon is not None and not (min_lon <= origin.longitude <= max_lon):
                continue
            kept.append(event)

        if limit is not None:
            kept = Catalog(kept[: int(limit)])

        if len(kept) == 0:
            # Real FDSN nodes return HTTP 204 as an exception, not an empty
            # catalog; the agent's error handling should see the same shape.
            raise RuntimeError(f"{self.dc}: No data available for request")
        return kept

    # ── Stations ──────────────────────────────────────────────────────
    def get_stations(self, **kwargs):
        self.recorder.stations.append((self.dc, kwargs))
        if _DC_BEHAVIOR.get(self.dc) == "raise":
            raise RuntimeError(f"{self.dc}: simulated service outage")

        requested_net = str(kwargs.get("network") or "*")
        net_code      = "*" if requested_net in ("*", "") else requested_net.split(",")[0]
        if net_code == "*":
            net_code = "IU"

        # Channel codes are drawn from the request so the continuous path's
        # channel selectors are visible in the resulting requests.
        chan_spec = str(kwargs.get("channel") or "BHZ")
        codes     = [c.strip() for c in chan_spec.split(",") if c.strip()]
        chan_codes = []
        for code in codes:
            resolved = code.replace("?", "Z").replace("*", "Z")
            if len(resolved) == 3 and resolved not in chan_codes:
                chan_codes.append(resolved)
        chan_codes = chan_codes[:3] or ["BHZ"]

        loc_spec = str(kwargs.get("location") or "")
        loc_code = "" if loc_spec in ("*", "") else loc_spec.split(",")[0]

        # Two stations near the fixture events, always active over the window.
        stations = []
        for idx, (sta_code, lat, lon) in enumerate(
            (("EVAL1", 36.20, 140.60), ("EVAL2", 38.10, 141.90))
        ):
            channels = [
                Channel(
                    code=code,
                    location_code=loc_code,
                    latitude=lat,
                    longitude=lon,
                    elevation=120.0 + idx,
                    depth=0.0,
                    sample_rate=40.0,
                    start_date=UTCDateTime("2010-01-01"),
                    end_date=UTCDateTime("2035-01-01"),
                )
                for code in chan_codes
            ]
            stations.append(
                Station(
                    code=sta_code,
                    latitude=lat,
                    longitude=lon,
                    elevation=120.0 + idx,
                    channels=channels,
                    site=Site(name=f"Eval station {sta_code}"),
                    start_date=UTCDateTime("2010-01-01"),
                    end_date=UTCDateTime("2035-01-01"),
                )
            )

        return Inventory(
            networks=[
                Network(
                    code=net_code,
                    stations=stations,
                    start_date=UTCDateTime("2010-01-01"),
                )
            ],
            source=f"eval-fixture:{self.dc}",
        )

    # ── Waveforms ─────────────────────────────────────────────────────
    def get_waveforms(self, **kwargs):
        self.recorder.waveforms.append((self.dc, kwargs))
        if _DC_BEHAVIOR.get(self.dc) == "raise":
            raise RuntimeError(f"{self.dc}: simulated service outage")

        starttime = kwargs["starttime"]
        endtime   = kwargs["endtime"]
        rate      = 40.0
        npts      = max(1, int((endtime - starttime) * rate))

        stream = Stream()
        for chan in str(kwargs["channel"]).split(","):
            trace = Trace(data=np.arange(npts, dtype=np.int32) % 1000)
            trace.stats.network       = kwargs["network"]
            trace.stats.station       = kwargs["station"]
            trace.stats.location      = kwargs.get("location") or ""
            trace.stats.channel       = chan.strip()
            trace.stats.starttime     = starttime
            trace.stats.sampling_rate = rate
            stream.append(trace)
        return stream


class _CallRecorder:
    """Collects every fake-service call so a case can assert on them."""

    def __init__(self) -> None:
        self.clients:   List[str]                    = []
        self.events:    List[Tuple[str, dict]]       = []
        self.stations:  List[Tuple[str, dict]]       = []
        self.waveforms: List[Tuple[str, dict]]       = []
        self.bulk_tasks: List[dict]                  = []

    def make_client(self, dc: str, timeout: int = 30):
        self.clients.append(dc)
        return _FakeFDSNClient(dc, self)


# ===========================================================================
# Offline seams 2 and 3 — basemap imagery and the bulk-download fan-out
# ===========================================================================

def _make_blank_tiles_class():
    """A ``GoogleTiles`` subclass that yields a flat grey tile, never a request."""
    from cartopy.io.img_tiles import GoogleTiles

    class _BlankTiles(GoogleTiles):
        def get_image(self, tile):
            img = np.full((16, 16, 3), 210, dtype=np.uint8)
            return img, self.tileextent(tile), "lower"

    return _BlankTiles


class _RecordingScheduler:
    """
    Stands in for the multiprocessing ``Scheduler``.

    The real one forks workers that each build their own FDSN client, which no
    in-process monkeypatch can reach. This drains the same task iterator on the
    calling thread, records every task, and writes **one file per station-day**
    through the production :func:`build_daily_outfile` — the same split
    ``write_daily_per_station_all_channels`` performs — so the resulting on-disk
    naming and day coverage are the real thing.
    """

    recorder: Optional[_CallRecorder] = None

    def __init__(self, nproc: int):
        self.nproc = nproc

    def start(self, tasks) -> None:
        for task in tasks:
            if _RecordingScheduler.recorder is not None:
                _RecordingScheduler.recorder.bulk_tasks.append(task)
            writer = task.get("writer", {})
            for request in task.get("requests", []):
                self._write_stub(request, writer)

    @staticmethod
    def _write_stub(request: dict, writer: dict) -> None:
        day = T._day_start_utc(request["tstart"])
        end = request["tend"]
        while day < end:
            outfile = T.build_daily_outfile(
                outdir=writer.get("outdir", "."),
                net=request["net"],
                sta=request["sta"],
                loc=request["loc"],
                chan=request["chan"],
                day_start=day,
                dir_date=writer.get("dir_date", False),
                dir_stat=writer.get("dir_stat", False),
            )
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            trace = Trace(data=np.arange(400, dtype=np.int32))
            trace.stats.network       = request["net"]
            trace.stats.station       = request["sta"]
            trace.stats.location      = "" if request["loc"] in ("--", None) else request["loc"]
            trace.stats.channel       = request["chan"]
            trace.stats.starttime     = day
            trace.stats.sampling_rate = 40.0
            trace.write(outfile, format="MSEED")

            day += 86_400


# ===========================================================================
# The stub LLM — a hint follower, not a script
# ===========================================================================

_HINT_RE = re.compile(r"Next call (\w+)")


class StubLLM(BaseChatModel):
    """
    Offline stand-in for the chat model, playing two roles.

    Must be a genuine ``BaseChatModel``: ``create_deep_agent`` passes anything
    else to ``init_chat_model`` as a provider string.

    **Planner** (invoked with no tools bound). Answers with the case's ``plan``
    as JSON. If the prompt carries a clarification ("The user clarified:"), it
    answers with ``plan_after_clarification`` instead — keyed on the prompt text
    rather than on a call counter, because ``interrupt()`` replays the tool and
    so asks the planner the same question twice.

    **Orchestrator** (invoked with tools bound). Reads the last ``ToolMessage``
    and calls whatever that message's ``Next call <tool>.`` hint names; on the
    first turn it calls ``plan_query`` with the user's text. It deliberately does
    *not* replay a fixed list of calls — following the hints is what makes the
    trajectory assertion a test of the hint chain in ``_build_tools`` rather than
    a restatement of the expectation.

    ``structured`` selects the planner path: ``True`` serves
    ``with_structured_output``, ``False`` refuses it so the text fallback runs.
    Both paths ship, so the suite covers both.
    """

    plan:       dict
    plan_after_clarification: Optional[dict] = None
    structured: bool           = True

    # Counters live in a dict because pydantic models validate attribute
    # assignment; mutating a container sidesteps that entirely.
    counters:    dict = PydanticField(default_factory=lambda: {"planner": 0, "calls": 0})
    tool_calls:  list = PydanticField(default_factory=list)
    bound_tools: list = PydanticField(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tremors-eval-stub"

    # ── Plan selection ────────────────────────────────────────────────
    def _plan_for(self, messages) -> dict:
        self.counters["planner"] += 1
        text = " ".join(str(getattr(m, "content", "")) for m in messages)
        if self.plan_after_clarification is not None and "The user clarified:" in text:
            return self.plan_after_clarification
        return self.plan

    @property
    def planner_calls(self) -> int:
        return self.counters["planner"]

    # ── Planner: structured path ──────────────────────────────────────
    def with_structured_output(self, schema, **kwargs):
        if not self.structured:
            raise NotImplementedError("eval stub: structured output disabled for this case")

        stub = self

        class _Structured:
            def invoke(self, messages, config=None, **_):
                return {
                    "raw":           None,
                    "parsed":        schema.model_validate(stub._plan_for(messages)),
                    "parsing_error": None,
                }

        return _Structured()

    # ── Tool binding ──────────────────────────────────────────────────
    def bind_tools(self, tools, **kwargs):
        # Binding through Runnable.bind means `tools` arrives in _generate's
        # kwargs, which is how the two roles are told apart.
        formatted = [convert_to_openai_tool(t) for t in tools]
        # Record the surface the agent actually offers, so a case can assert the
        # tool-surface trim still holds (see the `tool_surface` invariant).
        self.bound_tools[:] = [
            f.get("function", {}).get("name") for f in formatted
        ]
        return self.bind(tools=formatted, **kwargs)

    # ── Generation ────────────────────────────────────────────────────
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if kwargs.get("tools"):
            message = self._orchestrate(messages)
        else:
            payload = json.dumps(self._plan_for(messages))
            message = AIMessage(content=f"```json\n{payload}\n```")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _orchestrate(self, messages) -> AIMessage:
        last_tool = next(
            (m for m in reversed(list(messages)) if isinstance(m, ToolMessage)), None
        )

        if last_tool is None:
            query = next(
                (str(m.content) for m in reversed(list(messages))
                 if isinstance(m, HumanMessage)),
                "",
            )
            return self._call("plan_query", {"query": query})

        content = str(last_tool.content)
        match   = _HINT_RE.search(content)
        if match is None:
            # "STOP", a rejection notice, or an unrecognized hint: end the turn.
            return AIMessage(content=f"Done. Last tool result: {content[:200]}")
        return self._call(match.group(1), {})

    def _call(self, name: str, args: dict) -> AIMessage:
        self.counters["calls"] += 1
        self.tool_calls.append(name)
        return AIMessage(
            content="",
            tool_calls=[{
                "name": name,
                "args": args,
                "id":   f"eval-call-{self.counters['calls']}",
            }],
        )


# ===========================================================================
# Cases
# ===========================================================================

_JAPAN_BBOX = {
    "min_lat": 30.0, "max_lat": 46.0,
    "min_lon": 129.0, "max_lon": 146.0,
}
_2020 = {"min_date": "2020-01-01T00:00:00", "max_date": "2020-12-31T23:59:59"}

# A zero-extent box — how the planner expresses "near this one place". Usable
# only once a radius joins it, which is what the verifier insists on.
_TOKYO_POINT = {
    "min_lat": 35.6, "max_lat": 35.6,
    "min_lon": 139.7, "max_lon": 139.7,
}


def _cases() -> List[dict]:
    """
    The eval suite.

    ``plan`` is what the stub planner returns (live mode ignores it and lets the
    real model plan). ``expect_tools`` / ``forbid_tools`` are compared against
    the tools that *executed*. ``invariants`` are checked against the artifacts
    on disk.
    """
    return [
        {
            "name":  "event_catalog",
            "query": "M5+ earthquakes in Japan during 2020",
            "plan":  {**_JAPAN_BBOX, **_2020, "min_mag": 5.0, "limit": 100},
            "structured": True,
            "expect_tools": ["plan_query", "query_cascade", "plot_results"],
            "forbid_tools": ["retrieve_waveforms", "retrieve_continuous_waveforms",
                             "plot_waveforms", "plot_continuous_waveforms"],
            "invariants": {
                "tables":        ["EVENT", "ORIGIN", "NETMAG"],
                "min_rows":      8,
                "in_bbox":       True,
                "in_window":     True,
                "min_mag":       5.0,
                "datacenters":   {"USGS", "EMSC", "ISC"},
                "absent_dcs":    {"GEOFON", "RASPISHAKE", "UNKNOWN"},
                "duplicates_retained": True,
                "plots":         1,
                # The trimmed surface: seven Tremors tools plus read_file, and
                # nothing deepagents adds by default.
                "tool_surface": [
                    "plan_query", "query_cascade", "retrieve_waveforms",
                    "plot_results", "plot_waveforms",
                    "retrieve_continuous_waveforms", "plot_continuous_waveforms",
                    "read_file",
                ],
            },
        },
        {
            # Same query through the *other* planner path: the text fallback.
            # Both paths ship, so both have to be exercised.
            "name":  "event_catalog_text_fallback",
            "query": "M5+ earthquakes in Japan during 2020",
            "plan":  {**_JAPAN_BBOX, **_2020, "min_mag": 5.0, "limit": 100},
            "structured": False,
            "expect_tools": ["plan_query", "query_cascade", "plot_results"],
            "forbid_tools": ["retrieve_waveforms"],
            "invariants": {
                "tables":      ["EVENT", "ORIGIN"],
                "min_rows":    8,
                "in_bbox":     True,
                "in_window":   True,
                "min_mag":     5.0,
                "datacenters": {"USGS", "EMSC", "ISC"},
            },
        },
        {
            "name":  "event_waveforms",
            "query": "M6+ earthquakes in Japan in 2020, get waveforms and plot them",
            "plan":  {**_JAPAN_BBOX, **_2020, "min_mag": 6.0, "radius": 200.0,
                      "radius_unit": "km", "get_waveforms": True,
                      "plot_waveforms": True},
            "structured": True,
            "expect_tools": ["plan_query", "query_cascade", "retrieve_waveforms",
                             "plot_results", "plot_waveforms"],
            "forbid_tools": ["retrieve_continuous_waveforms"],
            "invariants": {
                "tables":    ["EVENT", "ORIGIN"],
                "min_rows":  3,
                "in_bbox":   True,
                "in_window": True,
                "min_mag":   6.0,
                "globs":     {"*.mseed": 1},
            },
        },
        {
            "name":  "continuous",
            "query": "Continuous BHZ data for network CI from 2016-02-01 to 2016-02-04",
            "plan":  {"get_continuous_waveforms": True, "net": "CI", "chan": "BHZ",
                      "min_date": "2016-02-01T00:00:00",
                      "max_date": "2016-02-04T00:00:00", "parallel": 2},
            "structured": True,
            "expect_tools": ["plan_query", "retrieve_continuous_waveforms",
                             "plot_continuous_waveforms"],
            # The whole point of the routing rules: continuous mode must never
            # touch the event cascade.
            "forbid_tools": ["query_cascade", "retrieve_waveforms", "plot_results"],
            "invariants": {
                "tables":     ["WAVEFORM_SITE", "WAVEFORM_SITECHAN"],
                "globs":      {"*.mseed": 1},
                "absent":     ["EVENT.PARQUET"],
                "bulk_days":  3,
            },
        },
        {
            # A single point with no radius: the finding the planner used to
            # hardcode, now the verifier's. Interrupts are off, so fail closed
            # and touch nothing.
            "name":  "verifier_fails_closed",
            "query": "earthquakes near 35.6, 139.7",
            "plan":  {**_TOKYO_POINT, **_2020},
            "structured": True,
            "interrupts": False,
            "expect_tools": ["plan_query"],
            "forbid_tools": ["query_cascade", "retrieve_waveforms",
                             "retrieve_continuous_waveforms"],
            "invariants": {
                "status":     "Clarification Required",
                "absent":     ["EVENT.PARQUET"],
                "no_fdsn":    True,
            },
        },
        {
            # Same bad plan, interrupts on: pause, take the clarification, and
            # complete on the re-plan. The clarified plan keeps the point form,
            # so the boundingbox expansion runs too.
            "name":  "clarification_gate",
            "query": "earthquakes near 35.6, 139.7 in 2020",
            "plan":  {**_TOKYO_POINT, **_2020},
            "plan_after_clarification": {
                **_TOKYO_POINT, "radius": 300.0, "radius_unit": "km",
                "min_mag": 5.0, **_2020,
            },
            "structured": True,
            "interrupts": True,
            "gate_replies": [
                ("clarification", "search within 300 km of that point"),
                ("approve", None),          # the query_cascade gate, still armed
            ],
            "expect_tools": ["plan_query", "query_cascade", "plot_results"],
            "invariants": {
                "status":      "Plots Generated",
                "tables":      ["EVENT", "ORIGIN"],
                "min_rows":    1,
                "in_window":   True,
                "min_mag":     5.0,
                # Both gates: the clarification, then the approval it unblocks.
                "gates_seen":  ["clarification", "query_cascade"],
            },
        },
        {
            # The approval gate's `edit` decision has to reach search_params.
            "name":  "approval_gate_edit",
            "query": "M5+ earthquakes in Japan during 2020",
            "plan":  {**_JAPAN_BBOX, **_2020, "min_mag": 5.0},
            "structured": True,
            "interrupts": True,
            "gate_replies": [("edit", {"min_mag": 6.5})],
            "expect_tools": ["plan_query", "query_cascade", "plot_results"],
            "invariants": {
                "status":     "Plots Generated",
                "tables":     ["EVENT", "ORIGIN"],
                "min_rows":   1,
                "min_mag":    6.5,          # the *edited* floor, not the planned one
                "gates_seen": ["query_cascade"],
            },
        },
        {
            # A rejected gate must block execution outright.
            "name":  "approval_gate_reject",
            "query": "M5+ earthquakes in Japan during 2020",
            "plan":  {**_JAPAN_BBOX, **_2020, "min_mag": 5.0},
            "structured": True,
            "interrupts": True,
            "gate_replies": [("reject", "wrong region, start over")],
            "expect_tools": ["plan_query"],
            "forbid_tools": ["query_cascade"],
            "invariants": {
                "absent":     ["EVENT.PARQUET"],
                "no_fdsn":    True,
                "gates_seen": ["query_cascade"],
            },
        },
    ]


# ===========================================================================
# Gate answering
# ===========================================================================

class _GateAnswerer:
    """
    Answers human-in-the-loop pauses from a case's scripted ``gate_replies``.

    Records what it was asked, so a case can assert *that* a gate fired — a run
    that quietly stopped gating would otherwise look like a pass.
    """

    def __init__(self, replies: Sequence[Tuple[str, Any]], auto_approve: bool = False):
        self.replies      = list(replies)
        self.auto_approve = auto_approve
        self.seen: List[str] = []

    def __call__(self, payload: dict) -> Any:
        kind = payload.get("kind")

        if kind == "clarification":
            self.seen.append("clarification")
            reply = self._next("clarification")
            return reply if reply is not None else "please proceed"

        requests = payload.get("action_requests") or []
        for request in requests:
            self.seen.append(str(request.get("name")))

        kind_wanted, value = self._next_typed()
        if kind_wanted == "edit":
            return {"decisions": [
                {"type": "edit",
                 "edited_action": {
                     "name": r.get("name"),
                     "args": {**(r.get("args") or {}), "params_override": value},
                 }}
                for r in requests
            ]}
        if kind_wanted == "reject":
            return {"decisions": [{"type": "reject", "message": value} for _ in requests]}
        return {"decisions": [{"type": "approve"} for _ in requests]}

    def _next(self, wanted: str) -> Any:
        for i, (kind, value) in enumerate(self.replies):
            if kind == wanted:
                self.replies.pop(i)
                return value
        return None

    def _next_typed(self) -> Tuple[str, Any]:
        for i, (kind, value) in enumerate(self.replies):
            if kind in ("approve", "edit", "reject"):
                self.replies.pop(i)
                return kind, value
        return "approve", None


# ===========================================================================
# Invariant checks
# ===========================================================================

def _table_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f"{name.upper()}.PARQUET")


def _check_invariants(
    case:     dict,
    result:   dict,
    out_dir:  str,
    recorder: _CallRecorder,
    stub:     Optional[StubLLM],
    answerer: _GateAnswerer,
) -> List[str]:
    """Return a list of failure messages; empty means the case's artifacts pass."""
    inv:      dict      = case.get("invariants", {})
    failures: List[str] = []
    plan:     dict      = case.get("plan", {})

    # ── Status and gates ──────────────────────────────────────────────
    if "status" in inv and result.get("status") != inv["status"]:
        failures.append(
            f"status is {result.get('status')!r} (expected {inv['status']!r}); "
            f"error={result.get('error')!r}"
        )

    for expected_gate in inv.get("gates_seen", []):
        if expected_gate not in answerer.seen:
            failures.append(
                f"gate {expected_gate!r} never fired (gates seen: {answerer.seen or 'none'})"
            )

    if inv.get("no_fdsn") and recorder.events:
        failures.append(
            f"FDSN was contacted {len(recorder.events)} time(s) but should not have been: "
            f"{[dc for dc, _ in recorder.events]}"
        )

    # ── Tool surface ──────────────────────────────────────────────────
    # The agent deliberately trims deepagents' defaults (no `task` subagent, no
    # write/edit/glob/grep/execute). A regression that restores them widens what
    # the model may do without changing any result, so nothing else would notice.
    if "tool_surface" in inv and stub is not None:
        offered = set(stub.bound_tools)
        wanted  = set(inv["tool_surface"])
        if offered != wanted:
            extra   = sorted(offered - wanted)
            missing = sorted(wanted - offered)
            failures.append(
                "tool surface drifted"
                + (f"; unexpected: {extra}" if extra else "")
                + (f"; absent: {missing}" if missing else "")
            )

    # ── Files that must not exist ─────────────────────────────────────
    for name in inv.get("absent", []):
        if os.path.exists(os.path.join(out_dir, name)):
            failures.append(f"{name} exists but should not")

    # ── Tables ────────────────────────────────────────────────────────
    for name in inv.get("tables", []):
        if not os.path.exists(_table_path(out_dir, name)):
            failures.append(f"{name}.PARQUET is missing")

    origin_path = _table_path(out_dir, "ORIGIN")
    origins = pd.read_parquet(origin_path) if os.path.exists(origin_path) else None

    if origins is not None and not origins.empty:
        if "min_rows" in inv and len(origins) < inv["min_rows"]:
            failures.append(
                f"ORIGIN has {len(origins)} rows, expected at least {inv['min_rows']}"
            )

        if inv.get("in_window") and plan.get("min_date"):
            lo = UTCDateTime(plan["min_date"]).timestamp
            hi = UTCDateTime(plan["max_date"]).timestamp
            outside = origins[(origins["time"] < lo) | (origins["time"] > hi)]
            if not outside.empty:
                failures.append(
                    f"{len(outside)} origin(s) fall outside "
                    f"[{plan['min_date']}, {plan['max_date']}]"
                )

        if inv.get("in_bbox") and plan.get("min_lat") is not None:
            outside = origins[
                (origins["lat"] < plan["min_lat"]) | (origins["lat"] > plan["max_lat"]) |
                (origins["lon"] < plan["min_lon"]) | (origins["lon"] > plan["max_lon"])
            ]
            if not outside.empty:
                failures.append(
                    f"{len(outside)} origin(s) fall outside the requested bounding box"
                )

        expected_dcs = inv.get("datacenters")
        if expected_dcs:
            present = set(origins["datacenter"].unique())
            missing = expected_dcs - present
            if missing:
                failures.append(f"datacenter(s) {sorted(missing)} absent from ORIGIN")
        forbidden = inv.get("absent_dcs")
        if forbidden:
            leaked = forbidden & set(origins["datacenter"].unique())
            if leaked:
                failures.append(f"datacenter(s) {sorted(leaked)} present in ORIGIN but should not be")

        if inv.get("duplicates_retained"):
            # Dedup was removed on purpose: one earthquake reported by three DCs
            # must survive as three rows. Collapsing them again is a regression.
            grouped = origins.groupby(["time", "lat", "lon"]).size()
            if int(grouped.max()) < 2:
                failures.append(
                    "no origin appears more than once — deduplication seems to have "
                    "returned (expected the same event once per reporting datacenter)"
                )

    elif inv.get("min_rows"):
        failures.append("ORIGIN table missing or empty, so row/range checks could not run")

    # ── Magnitudes ────────────────────────────────────────────────────
    if "min_mag" in inv:
        netmag_path = _table_path(out_dir, "NETMAG")
        if os.path.exists(netmag_path):
            netmag = pd.read_parquet(netmag_path)
            below  = netmag[netmag["magnitude"] < inv["min_mag"]]
            if not below.empty:
                failures.append(
                    f"{len(below)} magnitude(s) below the requested "
                    f"min_mag={inv['min_mag']} (min seen: {netmag['magnitude'].min()})"
                )
        elif inv.get("min_rows"):
            failures.append("NETMAG.PARQUET missing, so the magnitude floor was not checked")

    # ── Files produced ────────────────────────────────────────────────
    for pattern, minimum in (inv.get("globs") or {}).items():
        found = glob.glob(os.path.join(out_dir, "**", pattern), recursive=True)
        if len(found) < minimum:
            failures.append(
                f"expected at least {minimum} file(s) matching {pattern}, found {len(found)}"
            )

    if "plots" in inv:
        plots = result.get("plots") or []
        if len(plots) < inv["plots"]:
            failures.append(f"expected at least {inv['plots']} plot(s), got {len(plots)}")

    # ── Bulk-download day coverage ────────────────────────────────────
    # Read from the filenames the writer produced (…_<YYYY><DOY>.mseed) rather
    # than from the request list: the request that reaches the scheduler spans
    # the whole window, and it is the per-day split on the way to disk that the
    # continuous pipeline is actually judged on.
    if "bulk_days" in inv:
        doys = {
            os.path.basename(path).rsplit("_", 1)[-1].removesuffix(".mseed")
            for path in glob.glob(os.path.join(out_dir, "**", "*.mseed"), recursive=True)
        }
        if len(doys) != inv["bulk_days"]:
            failures.append(
                f"bulk download wrote {len(doys)} distinct day(s) "
                f"({sorted(doys)}), expected {inv['bulk_days']}"
            )

    return failures


def _check_trajectory(case: dict, executed: List[str]) -> List[str]:
    """Compare the tools that ran against the required and forbidden sets."""
    failures: List[str] = []
    ran = set(executed)

    missing = [t for t in case.get("expect_tools", []) if t not in ran]
    if missing:
        failures.append(f"tool(s) never ran: {missing} (ran: {executed})")

    banned = [t for t in case.get("forbid_tools", []) if t in ran]
    if banned:
        failures.append(f"tool(s) ran but must not have: {banned} (ran: {executed})")

    return failures


# ===========================================================================
# Runner
# ===========================================================================

def _executed_tools(agent, config: dict) -> List[str]:
    """
    Tool names that actually executed, read from the final message history.

    Taken from ``ToolMessage`` names rather than from the model's requested
    ``tool_calls``: a call the human rejected produces a message but never runs
    the tool, and the reject case turns on telling those two apart.
    """
    from langchain_core.messages import ToolMessage

    if not config.get("configurable", {}).get("thread_id"):
        return []

    snapshot = agent._action.get_state(config)
    names: List[str] = []
    for message in (snapshot.values or {}).get("messages", []) or []:
        if not isinstance(message, ToolMessage):
            continue
        if message.name is None:
            continue
        # A rejected call comes back as an error ToolMessage; it never ran.
        if getattr(message, "status", None) == "error":
            continue
        names.append(message.name)
    return names


def _capture_thread(agent) -> dict:
    """
    Shadow ``agent._config`` so the harness keeps the run's thread id.

    ``run()`` clears ``_thread_id`` when the graph completes, but the message
    history that proves which tools ran lives in the checkpointer under that id.
    Wrapping the accessor is the least invasive way to keep it — the agent's
    own behaviour is untouched.
    """
    seen: dict = {}
    original   = agent._config

    def _config() -> dict:
        config = original()
        seen.clear()
        seen.update(config)
        return config

    agent._config = _config
    return seen


def _register_stub_harness_profile(stub: StubLLM) -> None:
    """
    Give the stub the same harness profile the real backends get.

    ``_build_deep_agent`` registers a profile per *provider* ("openai",
    "anthropic") to disable the auto general-purpose subagent. deepagents derives
    the provider from the model class, so the stub resolves to its own provider
    and would otherwise fall back to defaults — leaving the ``task`` tool enabled
    and the offline tool surface wider than production's. Registering the same
    profile under the stub's provider keeps the surface faithful; the
    ``tool_surface`` invariant then checks it.
    """
    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        register_harness_profile,
    )

    register_harness_profile(
        type(stub).__name__.lower(),
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )


def _run_case(case: dict, args: argparse.Namespace) -> Tuple[bool, List[str], str]:
    """Run one case in a clean output directory. Returns (passed, failures, note)."""
    out_dir = os.path.join(args.out_dir, case["name"])
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    recorder = _CallRecorder()
    answerer = _GateAnswerer(case.get("gate_replies", []), auto_approve=args.live)

    stub: Optional[StubLLM] = None
    saved: Dict[str, Any] = {}

    if args.live:
        llm = _build_live_llm(args)
    else:
        stub = StubLLM(
            plan=case["plan"],
            plan_after_clarification=case.get("plan_after_clarification"),
            structured=case.get("structured", True),
        )
        llm = stub
        _register_stub_harness_profile(stub)

        # Install the three offline seams, remembering the originals.
        saved = {
            "_make_fdsn_client": T._make_fdsn_client,
            "GoogleTiles":       T.GoogleTiles,
            "Scheduler":         T.Scheduler,
        }
        T._make_fdsn_client = recorder.make_client
        T.GoogleTiles       = _make_blank_tiles_class()
        T.Scheduler         = _RecordingScheduler
        _RecordingScheduler.recorder = recorder

    try:
        agent    = T.TremorsAgent(
            llm=llm,
            output_dir=out_dir,
            interrupts=case.get("interrupts", True),
        )
        config   = _capture_thread(agent)
        result   = agent.run(case["query"], on_interrupt=answerer)
        executed = _executed_tools(agent, config)
    except Exception:                                   # noqa: BLE001 — report, keep going
        return False, [f"raised:\n{traceback.format_exc()}"], ""
    finally:
        for name, original in saved.items():
            setattr(T, name, original)
        _RecordingScheduler.recorder = None

    failures = _check_trajectory(case, executed)
    failures += _check_invariants(case, result, out_dir, recorder, stub, answerer)

    note = (
        f"status={result.get('status')!r} tools={executed} "
        f"dcs={result.get('queried_dcs')}"
    )
    if stub is not None:
        note += f" planner_calls={stub.planner_calls}"
    return (not failures), failures, note


# ===========================================================================
# Unit checks — library-level guards the agent cases cannot reach
# ===========================================================================
#
# The agent cases above drive the whole pipeline, but two failure modes live in
# helpers they never exercise offline, and both fail *silently* — the run reports
# Success while writing nothing, or writes rows attributed to a datacenter that
# never served them. Both were found by live probing, so they get a cheap,
# network-free guard here.

def _unit_writer_keeps_gappy_days(tmp_dir: str) -> List[str]:
    """
    A day whose data has a gap must still be written.

    ``write_daily_per_station_all_channels`` merges with ``fill_value=None``,
    which represents gaps as a masked array rather than inventing samples — and
    MiniSEED writing rejects masked arrays. Without a ``split()`` before writing,
    every day of the channel is skipped (the merge spans the whole request, so
    one gap anywhere masks every day's slice) while the run still says Success.
    """
    failures: List[str] = []
    out = os.path.join(tmp_dir, "gappy")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    rate  = 100.0
    day1  = UTCDateTime("2020-01-01T00:00:00")
    day2  = UTCDateTime("2020-01-02T00:00:00")
    stats = dict(network="XX", station="TEST", location="", channel="BHZ",
                 sampling_rate=rate)

    def _seg(start: UTCDateTime, seconds: int) -> Trace:
        npts = int(seconds * rate)
        return Trace(data=np.arange(npts, dtype=np.int32),
                     header={**stats, "starttime": start})

    # Day 1: one hour, a one-hour gap, then another hour. Day 2: contiguous.
    st = Stream([_seg(day1, 3600), _seg(day1 + 7200, 3600), _seg(day2, 3600)])
    expected_npts = sum(tr.stats.npts for tr in st)

    T.write_daily_per_station_all_channels(
        st, outdir=out, merge=True, dir_date=False, dir_stat=False,
        loc="", requested_chan="BHZ",
    )

    written = sorted(glob.glob(os.path.join(out, "**", "*.mseed"), recursive=True))
    if len(written) != 2:
        failures.append(
            f"expected 2 daily files (2020-01-01, 2020-01-02), got {len(written)}: "
            f"{[os.path.basename(p) for p in written]} — a gap silently dropped the day"
        )
        return failures

    import obspy

    total = 0
    day1_gaps = None
    for path in written:
        got = obspy.read(path)
        if any(np.ma.is_masked(tr.data) for tr in got):
            failures.append(f"{os.path.basename(path)}: masked data survived to disk")
        total += sum(tr.stats.npts for tr in got)
        if "2020001" in os.path.basename(path) or written.index(path) == 0:
            day1_gaps = len(got.get_gaps())

    if day1_gaps is not None and day1_gaps < 1:
        failures.append(
            "day 1 read back with no gap — the one-hour gap was filled rather "
            "than preserved as separate records"
        )
    # Filling the 1 h gap would add 360 000 samples; allow a few boundary samples.
    if not (expected_npts <= total <= expected_npts + 10):
        failures.append(
            f"sample count {total} != input {expected_npts} (±10): samples were "
            "invented or lost"
        )
    return failures


def _unit_inventory_fallback_provenance(tmp_dir: str) -> List[str]:
    """
    When the named DC cannot serve the inventory, the fallback's name must win.

    A planner may name a catalog-only node (ISC has neither a station nor a
    dataselect service). The inventory then comes from the fallback, so both the
    bulk waveform requests and the ``WAVEFORM_*`` provenance column have to
    follow the DC that actually answered — otherwise the download goes to a DC
    that returns nothing and the tables claim a source that never served them.
    """
    failures: List[str] = []

    class _NoStationService:
        def get_stations(self, **kwargs):
            raise ValueError("The current client does not have a station service.")

    class _WorkingService:
        def get_stations(self, **kwargs):
            net = Network(
                code="CI",
                stations=[Station(
                    code="PLM", latitude=33.35, longitude=-116.86, elevation=1000.0,
                    site=Site(name="Palomar"),
                    channels=[Channel(
                        code="BHZ", location_code="", latitude=33.35,
                        longitude=-116.86, elevation=1000.0, depth=0.0,
                        sample_rate=40.0,
                    )],
                )],
            )
            return Inventory(networks=[net], source="unit-check")

    def _fake_client(dc: str):
        return _NoStationService() if dc == "ISC" else _WorkingService()

    agent  = T.TremorsAgent(llm=None, output_dir=tmp_dir, interrupts=False)
    saved  = T._make_fdsn_client
    T._make_fdsn_client = _fake_client
    try:
        inv, served = agent._fetch_inventory_for_continuous(
            {"min_date": "2016-02-01T00:00:00", "max_date": "2016-02-03T00:00:00",
             "net": "CI", "chan": "BHZ"},
            datacenter="ISC",
        )
    finally:
        T._make_fdsn_client = saved

    if inv is None:
        failures.append("inventory fetch gave up instead of falling back off ISC")
        return failures
    if served == "ISC":
        failures.append(
            "reported ISC as the serving DC although its station service failed — "
            "bulk requests and table provenance would both be wrong"
        )
    elif served not in T._INVENTORY_DC_PRIORITY:
        failures.append(f"serving DC {served!r} is not one of the fallbacks")
    return failures


def _unit_cli_outcome_vocabulary(tmp_dir: str) -> List[str]:
    """
    Every status the agent can emit must map into the closed ``outcome`` set,
    and every outcome must have an exit code.

    A harness branches on ``outcome`` and on the exit code, never on the raw
    ``status`` prose. So a new or renamed status that nobody adds to
    ``_OUTCOME_BY_STATUS`` degrades silently to ``"unknown"`` → exit 1, turning a
    successful run into a reported failure. This scrapes the status strings out of
    the agent source and checks each one lands somewhere deliberate.
    """
    import inspect
    import re as _re

    from tremors import cli as C

    failures: List[str] = []

    # Every outcome the mapping can produce needs an exit code, including the
    # two _classify_outcome can synthesize on its own.
    produced = set(C._OUTCOME_BY_STATUS.values()) | {"unknown", "no_data"}
    for outcome in sorted(produced):
        if outcome not in C._EXIT_FOR_OUTCOME:
            failures.append(f"outcome {outcome!r} has no exit code")

    # Statuses the agent actually assigns, harvested from the source rather than
    # duplicated here — a list restating the mapping would never catch drift.
    source   = inspect.getsource(T)
    statuses = set(_re.findall(r'"status":\s*"([^"]+)"', source))
    statuses |= set(_re.findall(r'status="([^"]+)"', source))
    if len(statuses) < 5:
        failures.append(f"only harvested {len(statuses)} status strings; "
                        "the scrape pattern probably broke")

    for status in sorted(statuses):
        outcome = C._classify_outcome(status)
        if outcome == "unknown":
            failures.append(
                f"agent status {status!r} classifies as 'unknown' (exit 1) — "
                "add it to _OUTCOME_BY_STATUS"
            )
        elif outcome not in C._EXIT_FOR_OUTCOME:
            failures.append(f"status {status!r} → outcome {outcome!r} with no exit code")

    # A run that produced nothing must not be reported as an unqualified success.
    if C._classify_outcome("Success (No Data)") != "no_data":
        failures.append("'Success (No Data)' must classify as no_data")
    return failures


def _unit_cli_setting_precedence(tmp_dir: str) -> List[str]:
    """
    Settings resolve flag > env > config file > default, and secrets never echo.

    The CLI reports which source supplied each value (``tremors config``), and a
    harness stores endpoint/model/key in the config file. If precedence inverted,
    a stored stale credential would silently beat the environment — exactly the
    failure that is hardest to diagnose from a 401.
    """
    from tremors import cli as C

    failures: List[str] = []
    config = {"model": "from-config", "api_key": "sk-STORED"}

    cases = [
        # (flag, env dict, expected value, expected source prefix)
        ("from-flag", {"M": "from-env"}, "from-flag",   "flag"),
        (None,        {"M": "from-env"}, "from-env",    "env:"),
        (None,        {},                "from-config", "config"),
    ]
    for flag, env, want_value, want_source in cases:
        saved = {k: os.environ.get(k) for k in ("M",)}
        try:
            os.environ.pop("M", None)
            os.environ.update(env)
            value, source = C._setting(flag, "M", config, "model", "the-default")
        finally:
            for key, previous in saved.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
        if value != want_value:
            failures.append(f"flag={flag!r} env={env}: got {value!r}, want {want_value!r}")
        if not source.startswith(want_source):
            failures.append(f"flag={flag!r} env={env}: source {source!r} "
                            f"should start with {want_source!r}")

    # Default only when nothing else supplies a value.
    value, source = C._setting(None, "ABSENT_VAR_XYZ", {}, "model", "the-default")
    if (value, source) != ("the-default", "default"):
        failures.append(f"empty resolution gave {(value, source)!r}, want ('the-default', 'default')")

    # A setting may live under more than one variable name; both are honoured,
    # in the order given (this is what lets ANTHROPIC_AUTH_TOKEN stand in for
    # ANTHROPIC_API_KEY).
    saved = os.environ.get("SECOND_XYZ")
    try:
        os.environ["SECOND_XYZ"] = "from-second"
        value, source = C._setting(None, ("FIRST_XYZ", "SECOND_XYZ"), {}, "model")
    finally:
        if saved is None:
            os.environ.pop("SECOND_XYZ", None)
        else:
            os.environ["SECOND_XYZ"] = saved
    if value != "from-second" or source != "env:SECOND_XYZ":
        failures.append(f"env fallback tuple gave {(value, source)!r}, "
                        "want ('from-second', 'env:SECOND_XYZ')")

    # The api_key must never be reproduced in anything renderable — including
    # from inside a per-backend section, which is where it actually lives.
    nested = {"backend": "anthropic", "output_dir": "./out",
              "backends": {"anthropic": {"model": "from-config",
                                         "api_key": "sk-STORED"}}}
    for label, subject in (("flat", config), ("nested", nested)):
        redacted = C._redact_config(subject)
        if "sk-STORED" in json.dumps(redacted):
            failures.append(f"_redact_config leaked the stored api_key ({label})")
    if C._redact_config(config).get("model") != "from-config":
        failures.append("_redact_config dropped a non-secret setting")
    if (C._redact_config(nested)["backends"]["anthropic"].get("model")
            != "from-config"):
        failures.append("_redact_config dropped a non-secret setting in a section")

    # Connection settings are scoped per backend: the view for one backend must
    # not offer another's. A flat file handed a gateway model id to ollama, which
    # then failed with a model-not-found from a service that never had it.
    anthropic_view = C._config_view(nested, "anthropic")
    ollama_view    = C._config_view(nested, "ollama")
    if anthropic_view.get("model") != "from-config":
        failures.append(f"_config_view lost the backend's own model: {anthropic_view!r}")
    if "model" in ollama_view or "api_key" in ollama_view:
        failures.append(f"_config_view leaked anthropic settings to ollama: "
                        f"{ollama_view!r}")
    for label, view in (("anthropic", anthropic_view), ("ollama", ollama_view)):
        if view.get("output_dir") != "./out":
            failures.append(f"_config_view dropped the global output_dir ({label})")
    # And the resolution built on that view agrees.
    value, source = C._setting(None, "ABSENT_VAR_XYZ", ollama_view, "model",
                               "ollama-default")
    if (value, source) != ("ollama-default", "default"):
        failures.append(f"ollama resolved model {(value, source)!r}, "
                        "want ('ollama-default', 'default')")
    return failures


def _unit_cli_config_fails_closed(tmp_dir: str) -> List[str]:
    """
    A config file that cannot be trusted stops the run instead of being ignored.

    Silently falling back to defaults on a malformed or misspelled config is the
    dangerous branch: the user believes a private gateway is configured, and the
    request goes somewhere else entirely. Every such case must exit 5.

    Also pins the two shapes the file may take: the nested per-backend form, and
    a legacy flat one, which is migrated at read time into the section for the
    backend the file itself names. Dropping that migration would strip a stored
    gateway from an existing install; dropping the scoping would hand it to
    whichever backend ran next.
    """
    import contextlib
    import io as _io

    from tremors import cli as C

    failures: List[str] = []
    directory = os.path.join(tmp_dir, "cfg")
    os.makedirs(directory, exist_ok=True)

    def exit_code_for(contents: str, label: str) -> None:
        path = os.path.join(directory, f"{label}.json")
        with open(path, "w") as handle:
            handle.write(contents)
        C._config_cache.pop(path, None)
        args = argparse.Namespace(config_file=path)
        try:
            # _die reports to stderr; swallow it so a passing check stays quiet.
            with contextlib.redirect_stderr(_io.StringIO()):
                C._load_user_config(args)
        except SystemExit as exc:
            if exc.code != C._EXIT_CONFIG:
                failures.append(f"{label}: exited {exc.code}, want {C._EXIT_CONFIG}")
        else:
            failures.append(f"{label}: was accepted; it must exit {C._EXIT_CONFIG}")
        finally:
            C._config_cache.pop(path, None)

    exit_code_for("not json{",                 "malformed")
    exit_code_for('["a", "list"]',             "not_an_object")
    exit_code_for('{"base-url": "https://x"}', "misspelled_key")
    # The nested shape has four more ways to be wrong, and none may be shrugged off.
    exit_code_for('{"backends": "nope"}',                       "backends_not_object")
    exit_code_for('{"backends": {"ollama": "nope"}}',           "section_not_object")
    exit_code_for('{"backends": {"anthropics": {"model": "m"}}}', "unknown_backend")
    exit_code_for('{"backends": {"ollama": {"models": "m"}}}',  "misspelled_in_section")

    def loads_as(contents: dict, want: dict, label: str) -> None:
        path = os.path.join(directory, f"{label}.json")
        with open(path, "w") as handle:
            json.dump(contents, handle)
        C._config_cache.pop(path, None)
        loaded = C._load_user_config(argparse.Namespace(config_file=path))
        if loaded != want:
            failures.append(f"{label}: loaded as {loaded!r}, want {want!r}")
        C._config_cache.pop(path, None)

    # A valid nested file loads, and drops nulls rather than storing them.
    loads_as({"backend": "ollama",
              "backends": {"ollama": {"model": "m", "base_url": None}}},
             {"backend": "ollama", "backends": {"ollama": {"model": "m"}}},
             "good_nested")

    # A legacy flat file is attributed to the backend it names...
    loads_as({"backend": "ollama", "model": "m", "base_url": None},
             {"backend": "ollama", "backends": {"ollama": {"model": "m"}}},
             "legacy_named_backend")

    # ...and to the default backend when it names none, which is the shape every
    # file written before per-backend sections existed has.
    loads_as({"model": "gateway-model", "base_url": "https://gw"},
             {"backends": {C._DEFAULT_BACKEND: {"model": "gateway-model",
                                                "base_url": "https://gw"}}},
             "legacy_default_backend")

    # A mixed file is legal, and the explicit section wins over the flat key.
    loads_as({"backend": "anthropic", "model": "flat",
              "backends": {"anthropic": {"model": "sectioned"}}},
             {"backend": "anthropic", "backends": {"anthropic": {"model": "sectioned"}}},
             "legacy_section_wins")
    return failures


def _unit_cli_missing_settings_report(tmp_dir: str) -> List[str]:
    """
    The missing-settings error describes only what is actually missing.

    The anthropic backend needs api_key + base_url + model and reports every
    absent one at once. Two ways that report can lie, both of which shipped:

    * a fixed "all three of these" header above a bullet list filtered to the
      absent settings — one bullet under "all three" reads as a bug, and sends
      the user looking for two problems that do not exist;
    * a fixed remediation command passing all three flags, which tells someone
      whose gateway resolved fine to re-pass ``--base-url https://…`` — running
      it verbatim overwrites a working endpoint with a literal placeholder.

    So the header's count, the bullets and the suggested command must all track
    `missing`. The ``missing`` array itself stays bare setting names, because a
    harness branches on it.
    """
    import contextlib
    import io as _io

    from tremors import cli as C

    failures: List[str] = []
    all_three = ("api_key", "base_url", "model")
    supply    = {"base_url": "https://gw.example", "model": "m-1"}
    # The flag each setting would be stored with; a resolved setting's flag must
    # never appear in the suggested command.
    flag_for  = {"api_key": "--api-key", "base_url": "--base-url", "model": "--model"}
    # Header wording that is only true for that many missing settings.
    count_word = {1: "one setting", 2: "two settings", 3: "all three"}

    # api_key is deliberately unsupplied in every case: it is never stored in the
    # fixture config, so it is always missing and the 1-missing case is reachable.
    for absent in [("api_key",),
                   ("api_key", "model"),
                   all_three]:
        resolved = [s for s in all_three if s not in absent]
        config   = {"backends": {"anthropic": {s: supply[s] for s in resolved}}}
        label    = "+".join(absent)

        err = _io.StringIO()
        try:
            # Env vars would resolve settings the fixture config withholds, so the
            # backend's env names are cleared for the duration of the call.
            saved = {name: os.environ.pop(name, None)
                     for names in C._ENV_FOR_BACKEND.get("anthropic", {}).values()
                     for name in ((names,) if isinstance(names, str) else names)}
            try:
                with contextlib.redirect_stderr(err):
                    C._build_llm("anthropic", None, None, None, config=config)
            finally:
                for name, value in saved.items():
                    if value is not None:
                        os.environ[name] = value
        except SystemExit as exc:
            if exc.code != C._EXIT_CONFIG:
                failures.append(f"{label}: exited {exc.code}, want {C._EXIT_CONFIG}")
        else:
            failures.append(f"{label}: no error raised; want exit {C._EXIT_CONFIG}")

        report = err.getvalue()
        if not report.strip():
            failures.append(f"{label}: reported nothing")
            continue

        # One bullet per missing setting, and no bullet for a resolved one.
        bullets = [line for line in report.splitlines() if line.lstrip().startswith("•")]
        if len(bullets) != len(absent):
            failures.append(f"{label}: {len(bullets)} bullets for "
                            f"{len(absent)} missing setting(s)")
        for name in absent:
            if not any(name in line for line in bullets):
                failures.append(f"{label}: missing setting {name!r} has no bullet")
        for name in resolved:
            if any(name in line for line in bullets):
                failures.append(f"{label}: resolved setting {name!r} was "
                                "reported as missing")

        # The header's count must match. A wrong count is how "all three" over one
        # bullet got shipped.
        want = count_word[len(absent)]
        header = report.splitlines()[0] + " " + (report.splitlines()[1]
                                                if len(absent) == 3 else "")
        if want not in header:
            failures.append(f"{label}: header {header.strip()!r} does not say {want!r}")
        for size, phrase in count_word.items():
            if size != len(absent) and phrase in header:
                failures.append(f"{label}: header claims {phrase!r} with "
                                f"{len(absent)} missing")

        # The suggested `config --save` names the missing settings and nothing else.
        command = next((line for line in report.splitlines()
                        if "config --save" in line), "")
        continuation = report.split(command, 1)[1].splitlines()[1] if command else ""
        command += " " + continuation.strip()
        for name in absent:
            if flag_for[name] not in command:
                failures.append(f"{label}: suggested command omits "
                                f"{flag_for[name]} for missing {name!r}")
        for name in resolved:
            if flag_for[name] in command:
                failures.append(f"{label}: suggested command passes "
                                f"{flag_for[name]} for already-resolved {name!r} "
                                "— running it would overwrite a good value")

    # The machine-readable side: bare setting names, in a stable order, exit 5.
    document: dict = {}
    C._JSON_MODE = True
    try:
        with contextlib.redirect_stdout(_io.StringIO()) as out:
            saved = {name: os.environ.pop(name, None)
                     for names in C._ENV_FOR_BACKEND.get("anthropic", {}).values()
                     for name in ((names,) if isinstance(names, str) else names)}
            try:
                C._build_llm("anthropic", None, None, None, config={})
            finally:
                for name, value in saved.items():
                    if value is not None:
                        os.environ[name] = value
    except SystemExit:
        try:
            document = json.loads(out.getvalue())
        except ValueError as exc:
            failures.append(f"json mode: stdout was not one JSON document ({exc})")
    finally:
        C._JSON_MODE = False

    if document:
        if document.get("missing") != list(all_three):
            failures.append(f"json mode: missing={document.get('missing')!r}, "
                            f"want {list(all_three)!r}")
        if document.get("outcome") != "config_error":
            failures.append(f"json mode: outcome={document.get('outcome')!r}, "
                            "want 'config_error'")
    return failures


def _unit_checks() -> List[Tuple[str, Any]]:
    """Name → callable(tmp_dir) -> list of failure strings."""
    return [
        ("unit:writer_keeps_gappy_days",       _unit_writer_keeps_gappy_days),
        ("unit:inventory_fallback_provenance", _unit_inventory_fallback_provenance),
        ("unit:cli_outcome_vocabulary",        _unit_cli_outcome_vocabulary),
        ("unit:cli_setting_precedence",        _unit_cli_setting_precedence),
        ("unit:cli_config_fails_closed",       _unit_cli_config_fails_closed),
        ("unit:cli_missing_settings_report",   _unit_cli_missing_settings_report),
    ]


def _build_live_llm(args: argparse.Namespace):
    """Build a real chat model for ``--live`` runs, reusing the CLI's factory."""
    from tremors.cli import _build_llm

    return _build_llm(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_evals.py",
        description="Behavioral eval suite for the TREMORS agent.",
    )
    parser.add_argument("-k", "--filter", default=None, metavar="SUBSTR",
                        help="Only run cases whose name contains SUBSTR.")
    parser.add_argument("--out-dir", default=os.path.join(_HERE, "_runs"), metavar="DIR",
                        help="Where each case writes its artifacts. Default: evals/_runs")
    parser.add_argument("--live", action="store_true",
                        help="Query real datacenters with a real LLM backend "
                             "instead of the offline fixtures. Gates auto-approve.")
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "openai", "anthropic"],
                        help="LLM backend for --live. Default: ollama")
    parser.add_argument("--model", default=None, metavar="NAME")
    parser.add_argument("--base-url", dest="base_url", default=None, metavar="URL")
    parser.add_argument("--temperature", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--refresh-fixtures", action="store_true",
                        help="Rewrite evals/fixtures/*.xml from the in-code event pool.")
    parser.add_argument("--list", action="store_true", help="List case names and exit.")
    args = parser.parse_args(argv)

    cases = _cases()
    units = _unit_checks()
    if args.filter:
        cases = [c for c in cases if args.filter in c["name"]]
        units = [u for u in units if args.filter in u[0]]

    if args.list:
        for name in [c["name"] for c in cases] + [u[0] for u in units]:
            print(name)
        return 0

    if not cases and not units:
        print(f"No cases match {args.filter!r}.")
        return 1

    if not args.live:
        ensure_fixtures(force=args.refresh_fixtures)

    mode = "LIVE" if args.live else "offline"
    print(
        f"\nTREMORS evals — {len(cases)} case(s) + {len(units)} unit check(s), "
        f"{mode} mode\n" + "=" * 70
    )

    passed: List[str] = []
    failed: List[Tuple[str, List[str]]] = []

    # Unit checks first: they are fast, offline regardless of --live, and a
    # failure here explains any agent case that fails downstream of them.
    for name, check in units:
        print(f"\n▶ {name}")
        unit_dir = os.path.join(args.out_dir, name.replace(":", "_"))
        os.makedirs(unit_dir, exist_ok=True)
        try:
            unit_failures = check(unit_dir)
        except Exception:
            unit_failures = [traceback.format_exc()]
        if unit_failures:
            failed.append((name, unit_failures))
            print("  FAIL")
            for failure in unit_failures:
                for line in str(failure).splitlines():
                    print(f"        {line}")
        else:
            passed.append(name)
            print("  PASS")

    for case in cases:
        print(f"\n▶ {case['name']}")
        ok, failures, note = _run_case(case, args)
        if ok:
            passed.append(case["name"])
            print(f"  PASS  {note}")
        else:
            failed.append((case["name"], failures))
            print(f"  FAIL  {note}")
            for failure in failures:
                for line in str(failure).splitlines():
                    print(f"        {line}")

    print("\n" + "=" * 70)
    print(f"{len(passed)} passed, {len(failed)} failed")
    for name, failures in failed:
        print(f"  ✗ {name}: {len(failures)} problem(s)")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
