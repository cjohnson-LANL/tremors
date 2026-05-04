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
import copy
import glob
import json
import logging
import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Multiprocessing start-method (must happen before any fork)
# ---------------------------------------------------------------------------
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass  # Already set – harmless on Linux/macOS

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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

from .base import BaseAgent
from tremors.utils.geographic import (
    boundingbox,
    boundingradius,
    REGIONAL_RULES,
    add_north_arrow,
    add_scalebar,
)
from tremors.utils.schema import catalog_to_kbcore, inventory_to_kbcore


# ---------------------------------------------------------------------------
# Well-known FDSN nodes
# Source: https://github.com/obspy/obspy/blob/main/obspy/clients/fdsn/header.py
# ---------------------------------------------------------------------------
WELL_KNOWN_NODES: Dict[str, str] = {
    "AUSPASS":    "http://auspass.edu.au",
    "BGR":        "http://eida.bgr.de",
    "EARTHSCOPE": "http://service.iris.edu",
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
    "IRIS":       "http://service.iris.edu",
    "IRISPH5":    "http://service.iris.edu",
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

# Deduplication thresholds for the catalog merge
_DEDUP_TIME_SEC = 10.0   # seconds
_DEDUP_LAT_DEG  = 0.1    # degrees
_DEDUP_LON_DEG  = 0.1    # degrees


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
    """Uniform prefix logging used throughout the agent."""
    print(f"[{name}] {msg}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TremorsAgent(BaseAgent):
    """
    LangGraph agent for FDSN seismic data retrieval and visualization.

    Parameters
    ----------
    llm:
        Any LangChain-compatible chat model.
    output_dir:
        Directory where all output files (parquet, plots, MiniSEED) are written.
    **kwargs:
        Forwarded to BaseAgent (e.g. ``checkpointer``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, llm: Any, output_dir: str = "./tremors_output", **kwargs):
        try:
            super().__init__(llm, **kwargs)
        except Exception:
            # Graceful fallback if BaseAgent signature differs
            self.llm          = llm
            self.checkpointer = kwargs.get("checkpointer")
            self.name         = "TremorsAgent"

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._action = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(TremorsState)

        # Register nodes
        graph.add_node("plan_query",                    self._plan_query_node)
        graph.add_node("query_cascade",                 self._query_cascade_node)
        graph.add_node("plot_results",                  self._plot_results_node)
        graph.add_node("retrieve_waveforms",            self._retrieve_waveforms_node)
        graph.add_node("plot_waveforms",                self._plot_waveforms_node)
        graph.add_node("retrieve_continuous_waveforms", self._retrieve_continuous_waveforms_node)
        graph.add_node("plot_continuous_waveforms",     self._plot_continuous_waveforms_node)

        graph.set_entry_point("plan_query")

        # plan_query → continuous download OR event cascade
        graph.add_conditional_edges(
            "plan_query",
            lambda s: (
                "retrieve_continuous_waveforms"
                if s.get("search_params", {}).get("get_continuous_waveforms")
                else "query_cascade"
            ),
            {
                "retrieve_continuous_waveforms": "retrieve_continuous_waveforms",
                "query_cascade":                 "query_cascade",
            },
        )

        # query_cascade → waveforms OR straight to plotting
        graph.add_conditional_edges(
            "query_cascade",
            lambda s: (
                "retrieve_waveforms"
                if (
                    s.get("search_params", {}).get("get_waveforms")
                    or s.get("search_params", {}).get("plot_waveforms")
                )
                else "plot_results"
            ),
            {
                "retrieve_waveforms": "retrieve_waveforms",
                "plot_results":       "plot_results",
            },
        )

        graph.add_edge("retrieve_waveforms", "plot_results")

        # plot_results → waveform plots OR done
        graph.add_conditional_edges(
            "plot_results",
            lambda s: (
                "plot_waveforms"
                if s.get("search_params", {}).get("plot_waveforms")
                else END
            ),
            {"plot_waveforms": "plot_waveforms", END: END},
        )

        # Continuous path is linear
        graph.add_edge("retrieve_continuous_waveforms", "plot_continuous_waveforms")

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Node: plan_query
    # ------------------------------------------------------------------

    _PLAN_SYSTEM_PROMPT = """\
You are an expert seismologist assistant.
Parse the user's natural language query into FDSN search parameters and return a
single valid JSON object with these keys (omit any that are not applicable):

  datacenter         string   e.g. "ISC", "USGS", "IRIS". Default: "ISC"
  min_date           string   ISO-8601, e.g. "2010-01-01T00:00:00"
  max_date           string   ISO-8601
  min_lat            float
  max_lat            float
  min_lon            float
  max_lon            float
  min_depth          float    km
  max_depth          float    km
  min_mag            float
  max_mag            float
  radius             float    search radius (used when a single point is given)
  radius_unit        string   "km" or "deg"
  get_waveforms      boolean  true → download event waveforms
  plot_waveforms     boolean  true → also plot event waveforms
  get_continuous_waveforms boolean  true ONLY for continuous-stream requests (no event filter)
  stations_file      string   path to station list file (optional for continuous mode)
  net                string   network filter, e.g. "CI"
  sta                string   station filter, e.g. "ANMO"
  loc                string   location filter, e.g. "00"
  chan               string   channel filter, e.g. "BHZ"
  parallel           int      download threads (default 4)
  bulk_chunk         int      requests per bulk call (default 50)
  dir_date           boolean  organise output by YYYY/DOY (default false)
  dir_stat           boolean  organise output by network/station (default false)
  response           boolean  download StationXML response files (default false)
  pre_event_sec      float    seconds before event origin (default 30)
  post_event_sec     float    seconds after event origin (default 600)

Return ONLY the JSON object – no prose, no markdown fences.
User Query: {query}"""

    def _plan_query_node(self, state: TremorsState) -> TremorsState:
        """
        Invokes the LLM to translate the natural-language query into a
        structured ``search_params`` dict.  Handles point-query expansion
        (lat/lon + radius → bounding box) and emits a ``Clarification Required``
        status when a point is given without a radius.
        """
        _log(self.name, f"Planning query: {state['query']}")

        prompt = ChatPromptTemplate.from_messages(
            [("system", self._PLAN_SYSTEM_PROMPT)]
        )
        chain = prompt | self.llm

        try:
            response = chain.invoke({"query": state["query"]})
            content  = (
                response.content.strip()
                .replace("```json", "")
                .replace("```", "")
            )
            search_params: dict = json.loads(content)
            _log(self.name, f"Extracted params: {search_params}")
        except json.JSONDecodeError as exc:
            _log(self.name, f"Failed to parse LLM response: {exc}")
            return {
                **state,
                "search_params": {},
                "status": "Parse Failed",
                "error": (
                    "Could not extract search parameters from your query. "
                    "Please rephrase and try again."
                ),
            }

        # ── Point-query expansion ──────────────────────────────────────
        min_lat = search_params.get("min_lat")
        max_lat = search_params.get("max_lat")
        min_lon = search_params.get("min_lon")
        max_lon = search_params.get("max_lon")

        is_point_query = (
            min_lat is not None
            and min_lat == max_lat
            and min_lon is not None
            and min_lon == max_lon
        )

        if is_point_query:
            radius = search_params.get("radius")
            if radius:
                unit = search_params.get("radius_unit", "km")
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
            else:
                msg = (
                    f"A single location ({min_lat}, {min_lon}) was provided "
                    "without a search radius. Please specify a radius "
                    "(e.g. 'within 50 km' or 'within 1 degree') to define a "
                    "search area."
                )
                return {
                    **state,
                    "search_params": search_params,
                    "status": "Clarification Required",
                    "error": msg,
                }

        dc = search_params.get("datacenter", "ISC")
        return {
            **state,
            "search_params": search_params,
            "datacenter":    dc,
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

    def _query_cascade_node(self, state: TremorsState) -> TremorsState:
        """
        Fan-out across all target DCs, merge results into a single catalog,
        and deduplicate events using time/location proximity thresholds.

        Deduplication thresholds: ±10 s in time, ±0.1° in lat/lon.
        Events that pass deduplication are tagged with a ``datacenter:<DC>``
        Comment so provenance is preserved downstream.
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

        master_catalog: Catalog = Catalog()
        queried_success: List[str] = []

        for dc in targets:
            _log(self.name, f"Querying {dc}…")
            client = _make_fdsn_client(dc)
            if client is None:
                _log(self.name, f"Could not init client for {dc}. Skipping.")
                continue

            try:
                cat = client.get_events(**query_kwargs)
            except Exception as exc:
                _log(self.name, f"{dc} returned no data: {exc}")
                continue

            _log(self.name, f"{dc}: {len(cat)} events returned.")
            added = 0

            for event in cat:
                if not event.origins:
                    continue

                origin  = event.preferred_origin() or event.origins[0]
                otime   = origin.time
                olat    = origin.latitude
                olon    = origin.longitude

                duplicate = any(
                    (
                        abs(
                            (ex_origin := (ex.preferred_origin() or ex.origins[0])).time
                            - otime
                        ) < _DEDUP_TIME_SEC
                        and abs(ex_origin.latitude  - olat) < _DEDUP_LAT_DEG
                        and abs(ex_origin.longitude - olon) < _DEDUP_LON_DEG
                    )
                    for ex in master_catalog
                    if ex.origins
                )

                if not duplicate:
                    event.comments.append(Comment(text=f"datacenter:{dc}"))
                    master_catalog.append(event)
                    added += 1

            if len(cat) > 0:
                queried_success.append(dc)
            _log(self.name, f"{dc}: added {added} unique events.")

        _log(self.name, f"Cascade complete. Unique events: {len(master_catalog)}")

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
    ) -> Optional[Inventory]:
        """
        Pull a channel-level inventory from *datacenter* (or the first
        available DC in ``_INVENTORY_DC_PRIORITY``) for the time window
        and spatial/NSLC filters in *params*.

        The inventory is used to drive the bulk continuous download rather
        than relying on a user-supplied stations file.

        Returns an ObsPy Inventory or None if all attempts fail.
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
                return inv
            except Exception as exc:
                _log(self.name, f"{dc} inventory fetch failed: {exc}")
                continue

        return None

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

                inventory = self._fetch_inventory_for_continuous(
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
        self._queue   = multiprocessing.Queue()
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

        for w in self._workers:
            w.start()
        for w in self._workers:
            w.join()


class PullWave(multiprocessing.Process):
    """
    Worker process: consumes bulk-download tasks from a shared Queue.

    Each task contains one FDSN bulk request (a list of
    ``(net, sta, loc, chan, t_start, t_end)`` tuples) for a single
    datacenter.  The worker fetches the stream, then writes one file per
    requested station/channel/day.
    """

    def __init__(self, queue: multiprocessing.Queue):
        super().__init__(name="PullWave")
        self._queue:            multiprocessing.Queue = queue
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
            print(msg)
            logging.info(msg)
            return

        print(msg)
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
            print(msg)
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

        print(msg)
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
        print(f"Redownload (too small, {size} B): {os.path.basename(outfile)}")
        return True
    print(f"Exists: {os.path.basename(outfile)}")
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
                        day_st.write(outfile, format="MSEED")
                        msg = f"Saved: {outfile}"
                    except Exception as exc:
                        msg = f"Skipped: {outfile} error={exc}"
                    print(msg)
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