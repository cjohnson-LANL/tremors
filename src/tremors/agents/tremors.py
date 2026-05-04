'''
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
'''
"""
Tremors Agent : Agent for FDSN Service Checking and Metadata Retrieval

[T]ext-[R]eferenced [E]vent [M]apping & [O]utput [R]enderer for [Seismographs]

Current Authors: Ryley Hill, Richard Alfaro-Diaz, Christopher W. Johnson
Email: rghill@lanl.gov, rad@lanl.gov, cwj@lanl.gov 
"""

import os
import sys
import glob
import argparse
import copy
import multiprocessing
import traceback
# do we really need this??
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass
from collections import defaultdict

import json
import logging
import requests
from typing import TypedDict, Any, List, Optional, Mapping, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Obspy
import obspy
import obspy.clients.fdsn
from obspy import UTCDateTime, read, Stream
from obspy.core.inventory import Inventory
from obspy.core.event import Comment
from obspy.clients.fdsn import Client
# from obspy.clients.fdsn.headers import URL_MAPPINGS as WELL_KNOWN_NODES
from obspy.core.event import Catalog


# LangChain / LangGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# import URSA Base Agent
from .base import BaseAgent

# need to double check these RAD
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles
import matplotlib.patheffects as pe
from pyproj import Geod
from matplotlib.patches import Polygon
from cartopy.geodesic import Geodesic



from tremors.utils.geographic import boundingbox, boundingradius, REGIONAL_RULES, add_north_arrow, add_scalebar
from tremors.utils.schema import catalog_to_kbcore, inventory_to_kbcore

# ALL nodes (from Obspy (https://github.com/obspy/obspy/blob/28472163f63efeb4a60de6afca64e16e461275f7/obspy/clients/fdsn/header.py#L102-L136))
WELL_KNOWN_NODES = {
    "AUSPASS": "http://auspass.edu.au",
    "BGR": "http://eida.bgr.de",
    "EARTHSCOPE": "http://service.iris.edu",
    "EIDA": "http://eida-federator.ethz.ch",
    "ETH": "http://eida.ethz.ch",
    "EMSC": "http://www.seismicportal.eu",
    "GEONET": "http://service.geonet.org.nz",
    "GEOFON": "http://geofon.gfz-potsdam.de",
    "GFZ": "http://geofon.gfz-potsdam.de",
    "ICGC": "http://ws.icgc.cat",
    "IESDMC": "http://batsws.earth.sinica.edu.tw",
    "INGV": "http://webservices.ingv.it",
    "IPGP": "http://ws.ipgp.fr",
    "IRIS": "http://service.iris.edu",
    "IRISPH5": "http://service.iris.edu",
    "ISC": "http://www.isc.ac.uk",
    "KNMI": "http://rdsa.knmi.nl",
    "KOERI": "http://eida.koeri.boun.edu.tr",
    "LMU": "https://erde.geophysik.uni-muenchen.de",
    "NCEDC": "https://service.ncedc.org",
    "NIEP": "http://eida-sc3.infp.ro",
    "NOA": "http://eida.gein.noa.gr",
    "NRCAN": "https://earthquakescanada.nrcan.gc.ca",
    "ODC": "http://www.orfeus-eu.org",
    "ORFEUS": "http://www.orfeus-eu.org",
    "RESIF": "http://ws.resif.fr",
    "RESIFPH5": "http://ph5ws.resif.fr",
    "RASPISHAKE": "https://data.raspberryshake.org",
    "SCEDC": "http://service.scedc.caltech.edu",
    "TEXNET": "http://rtserve.beg.utexas.edu",
    "UIB-NORSAR": "http://eida.geo.uib.no",
    "USGS": "http://earthquake.usgs.gov",
    "USP": "http://sismo.iag.usp.br",
}



class TremorsState(TypedDict):
    """State for the TremorsAgent."""
    query: str  # User's natural language request or simple command
    datacenter: str # Target datacenter (e.g. "ISC")
    
    # Check results
    service_status: Dict[str, bool] # {"event": True, "station": False, ...}
    
    # Query parameters
    search_params: Optional[dict]
    
    # Results
    output_dir: str
    metadata_tables: Dict[str, str] # Map table name to filepath
    plots: List[str] # List of paths to generated plots
    waveforms_saved: List[str] # List of paths to saved MSEED files
    waveform_metadata: Dict[str, str] # Map table name to filepath for waveform metadata
    # Continuous Waveforms
    continuous_waveforms_saved: List[str]
    continuous_waveform_plots: List[str]
    
    status: str
    error: Optional[str]
    queried_dcs: List[str]


# --- Agent Class ---

class TremorsAgent(BaseAgent):
    """
    Agent for checking FDSN service availability and retrieving metadata (ISC focused).
    """

    def __init__(
        self,
        llm: Any,
        output_dir: str = "./tremors_output",
        **kwargs
    ):
        # Handle BaseAgent init which might expect llm
        if hasattr(BaseAgent, '__init__'):
             try:
                super().__init__(llm, **kwargs)
             except:
                # Fallback if signature is different
                self.llm = llm
                self.checkpointer = kwargs.get('checkpointer')
                self.name = "TremorsAgent"
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._action = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(TremorsState)
        
        graph.add_node("plan_query", self._plan_query_node)
        # Replaced separate check/query with unified cascade
        graph.add_node("query_cascade", self._query_cascade_node)
        graph.add_node("plot_results", self._plot_results_node)
        graph.add_node("retrieve_waveforms", self._retrieve_waveforms_node)
        graph.add_node("plot_waveforms", self._plot_waveforms_node)
        graph.add_node("retrieve_continuous_waveforms", self._retrieve_continuous_waveforms_node)
        graph.add_node("plot_continuous_waveforms", self._plot_continuous_waveforms_node)
        
        graph.set_entry_point("plan_query")
        
        def check_continuous_waveform_request(state):
            if state.get("search_params", {}).get("get_continuous_waveforms"):
                 return "retrieve_continuous_waveforms"
            return "query_cascade"
            
        graph.add_conditional_edges("plan_query", check_continuous_waveform_request,
                                    {"retrieve_continuous_waveforms": "retrieve_continuous_waveforms", "query_cascade": "query_cascade"})
        
        # Conditional edge for event waveforms
        def check_waveform_request(state):
             params = state.get("search_params", {})
             if params.get("get_waveforms") or params.get("plot_waveforms"):
                  return "retrieve_waveforms"
             return "plot_results"
             
        graph.add_conditional_edges("query_cascade", check_waveform_request, 
                                    {"retrieve_waveforms": "retrieve_waveforms", "plot_results": "plot_results"})

        graph.add_edge("retrieve_waveforms", "plot_results")
        
        def check_after_map(state):
             params = state.get("search_params", {})
             if params.get("plot_waveforms"):
                  return "plot_waveforms"
             return END
             
        graph.add_conditional_edges("plot_results", check_after_map,
                                    {"plot_waveforms": "plot_waveforms", END: END})

        graph.add_edge("retrieve_continuous_waveforms", "plot_continuous_waveforms")
        
        # Finish points could be either of the plot waveform nodes, or plot results if no waveforms
        # LangGraph handles execution ending when there are no more edges
        # We don't strictly need set_finish_point unless it's a single one, but we can leave it implicit or END
        
        return graph.compile(checkpointer=self.checkpointer)

    # --- Nodes ---

    def _plan_query_node(self, state: TremorsState) -> TremorsState:
        """
        Interprets the user's query to extract search parameters using LLM.
        """
        print(f"[{self.name}] Planning query for: {state['query']}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert seismologist assistant.
Your goal is to parse natural language queries into specific database search parameters for the 'origin' and 'arrival' tables.
Return a valid JSON object with the following keys (if applicable):
- "datacenter": string (e.g. "ISC", "USGS", "IRIS"), inferred or explicit. Default to "ISC".
- "min_date": UTCISO string (e.g. 2010-01-01T00:00:00)
- "max_date": UTCISO string
- "min_lat": float
- "max_lat": float
- "min_lon": float
- "max_lon": float
- "min_depth": float (min depth in km)
- "max_depth": float (max depth in km)
- "min_mag": float (min magnitude)
- "max_mag": float (max magnitude)
- "get_waveforms": boolean (true if user asks to get/download waveforms for the discovered events)
- "plot_waveforms": boolean (true if user asks to plot the waveforms for the discovered events)
- "get_continuous_waveforms": boolean (true ONLY if user explicitly asks for continuous data stream for a time period without searching/filtering for specific earthquake events)
- "stations_file": string (path to a test_stations.txt list of waveforms if provided)
- "net": string (network filter, e.g. "CI", "IU")
- "sta": string (station filter, e.g. "ANMO")
- "loc": string (location filter, e.g. "00", "--")
- "chan": string (channel filter, e.g. "BHZ")
- "parallel": int (number of threads for download, default 4)
- "bulk_chunk": int (number of requests per bulk call, default 50)
- "dir_date": boolean (set true if user mentions organizing by date or YYYT/DOY. Default False)
- "dir_stat": boolean (set true if user mentions organizing by station or network/stats. Default False)
- "response": boolean (set true if user asks to download response or xml metadata. Default False)
- "radius": float (radius for search area if user says 'near' or 'within')
- "radius_unit": string (unit for radius, e.g. 'km' or 'deg')
- "pre_event_sec": float (seconds before the event to start downloading waveforms, default 30.0)
- "post_event_sec": float (seconds after the event to end downloading waveforms, default 600.0)

If the user is asking for a specific event or generic data, try to infer reasonable defaults.
User Query: {query}"""),
        ])
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"query": state['query']})
            # Clean possible markdown code fences
            content = response.content.strip().replace("```json", "").replace("```", "")
            search_params = json.loads(content)
            print(f"[{self.name}] Extracted params: {search_params}")
            # Detect Point Query and Check for Radius
            min_lat = search_params.get("min_lat")
            max_lat = search_params.get("max_lat")
            min_lon = search_params.get("min_lon")
            max_lon = search_params.get("max_lon")
            if min_lat is not None and min_lat == max_lat and min_lon is not None and min_lon == max_lon:
                radius = search_params.get("radius")
                if radius:
                    unit = search_params.get("radius_unit", "km")
                    expanded_min_lat, expanded_max_lat, expanded_min_lon, expanded_max_lon = boundingbox(min_lat, min_lon, radius, unit=unit, ellipse='WGS84')
                    search_params.update({
                        "min_lat": expanded_min_lat,
                        "max_lat": expanded_max_lat,
                        "min_lon": expanded_min_lon,
                        "max_lon": expanded_max_lon,
                    })
                    print(f"[{self.name}] Expanded point ({min_lat}, {min_lon}) to bbox using radius {radius} {unit}: {expanded_min_lat}, {expanded_max_lat}, {expanded_min_lon}, {expanded_max_lon}")
                else:
                    # Point query without radius - trigger clarification
                    msg = f"A single location ({min_lat}, {min_lon}) was provided without a search radius. Earthquake locations are rarely exact. Please specify a radius (e.g., 'within 50km' or 'within 1 degree') to define a search area."
                    return {**state, "search_params": search_params, "status": "Clarification Required", "error": msg}
        except json.JSONDecodeError as e:
            print(f"[{self.name}] Failed to parse LLM response: {e}")
            return {
                **state,
                "search_params": {},
                "status": "Parse Failed",
                "error": "Could not extract search parameters from your query. Please rephrase and try again.",
            }
        # Update state with extracted params
        # Also sync 'datacenter' to top-level state key if present
        dc = search_params.get("datacenter", "ISC")
        
        return {**state,
                "search_params": search_params,
                "datacenter": dc,
                "status": "Query parsed"}


    def _determine_target_dcs(self, params: Dict) -> List[str]:
        """
        Determines the list of datacenters to query based on parameters.
        """
        requested_dc = params.get("datacenter")
        
        # 1. Start with Global Discovery
        target_dcs = ["USGS", "EMSC", "GEOFON", "ISC"]
        
        # 2. Check Regional
        min_lat = params.get("min_lat")
        max_lat = params.get("max_lat")
        min_lon = params.get("min_lon")
        max_lon = params.get("max_lon")
        
        if all(x is not None for x in [min_lat, max_lat, min_lon, max_lon]):
            # Check overlap
            for region, rule in REGIONAL_RULES.items():
                r_min_lat, r_max_lat, r_min_lon, r_max_lon = rule["bbox"]
                
                # Overlap logic (simple rect intersection)
                lat_overlap = (min_lat <= r_max_lat) and (max_lat >= r_min_lat)
                lon_overlap = (min_lon <= r_max_lon) and (max_lon >= r_min_lon)
                
                if lat_overlap and lon_overlap:
                    print(f"[{self.name}] Region match: {region}. Adding {rule['dcs']}")
                    target_dcs.extend(rule["dcs"])

        # Deduplicate
        unique_dcs = list(set(target_dcs))
        
        # 3. User Override/Addition
        if requested_dc and requested_dc != "ISC" and requested_dc in WELL_KNOWN_NODES:
             if requested_dc not in unique_dcs:
                 print(f"[{self.name}] User requested specific DC: {requested_dc}. Adding.")
                 unique_dcs.append(requested_dc)
        
        return unique_dcs

    def _query_cascade_node(self, state: TremorsState) -> TremorsState:
        """
        Executes the cascade query logic: Global -> Regional -> Deduplicate.
        """
        print(f"[{self.name}] Starting Cascade Query...")
        params = state.get("search_params", {})
        
        # 1. Determine Targets
        targets = self._determine_target_dcs(params)
        print(f"[{self.name}] Target Datacenters: {targets}")
        
        # 2. Setup Query Args
        now = UTCDateTime.now().strftime("%Y-%m-%d")
        t_start = UTCDateTime(params.get("min_date", "1970-01-01"))
        t_end = UTCDateTime(params.get("max_date", now))
        min_mag = params.get("min_mag", 3.0)
        limit = params.get("limit", 100)
        
        query_kwargs = {
            "starttime": t_start,
            "endtime": t_end,
            "minmagnitude": min_mag,
             # Applying 'limit' per DC to ensure we get candidates, but we truncate later.
             "limit": limit
        }
        
        if params.get("min_lat") is not None:
            query_kwargs["minlatitude"] = params.get("min_lat")
            query_kwargs["maxlatitude"] = params.get("max_lat")
            query_kwargs["minlongitude"] = params.get("min_lon")
            query_kwargs["maxlongitude"] = params.get("max_lon")

        # 3. Execute Loop
        master_catalog = Catalog()
        
        queried_success = []
        
        for dc in targets:
            url = WELL_KNOWN_NODES.get(dc, dc)
            print(f"[{self.name}] Querying {dc}...")
            
            try:
                # JIT Client Init
                try:
                    client = Client(dc)
                except:
                    try:
                        client = Client(base_url=url)
                    except:
                        print(f"[{self.name}] Could not initialize client for {dc}. Skipping.")
                        continue

                # Query
                cat = client.get_events(**query_kwargs)
                print(f"[{self.name}] {dc}: Found {len(cat)} events.")
                
                # Merge & Deduplicate
                added_count = 0
                for event in cat:
                    is_duplicate = False
                    
                    if not event.origins: continue
                    origin = event.preferred_origin() or event.origins[0]
                    
                    # Convert to timestamp/float for comparison
                    # origin.time is UTCDateTime
                    otime = origin.time
                    olat = origin.latitude
                    olon = origin.longitude
                    
                    for existing in master_catalog:
                        if not existing.origins: continue
                        e_origin = existing.preferred_origin() or existing.origins[0]
                        
                        t_diff = abs(e_origin.time - otime)
                        # Quick dist check
                        lat_diff = abs(e_origin.latitude - olat)
                        lon_diff = abs(e_origin.longitude - olon)
                        
                        # Thresholds: 10s and ~0.1 deg
                        if t_diff < 10.0 and lat_diff < 0.1 and lon_diff < 0.1:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # Tag with source datacenter using Comment
                        event.comments.append(Comment(text=f"datacenter:{dc}"))
                        
                        if event.event_descriptions:
                             event.event_descriptions[0].text += f" [{dc}]"
                        else:
                             # event.event_descriptions.append(EventDescription(text=f"[{dc}]"))
                             pass
                        master_catalog.append(event)
                        added_count += 1
                
                if len(cat) > 0:
                    queried_success.append(dc)
                    print(f"[{self.name}] {dc}: Added {added_count} unique events.")

            except Exception as e:
                # print(f"[{self.name}] {dc} query failed/no-data: {e}")
                pass

        # 4. Final Processing
        print(f"[{self.name}] Cascade Complete. Total Unique Events: {len(master_catalog)}")
        
        if len(master_catalog) == 0:
             return {
                **state, 
                "metadata_tables": {}, 
                "status": "Success (No Data)",
                "queried_dcs": queried_success
            }
            
        # Truncate
        # if len(master_catalog) > limit:
        #     master_catalog.events = master_catalog.events[:limit]
        #     print(f"[{self.name}] Truncated to limit {limit}.")

        # Convert to tables
        # Use primary DC as label? Or "Mixed"?
        # catalog_to_kbcore needs to handle mixed authors now.
        #
        tables = catalog_to_kbcore(master_catalog, datacenter="Mixed")
        
        saved_files = {}
        for name, df in tables.items():
            if not df.empty:
                filename = f"MIXED_{name}.parquet".upper()
                path = os.path.join(self.output_dir, filename)
                df.to_parquet(path, index=False)
                saved_files[name] = path
                print(f"[{self.name}] Saved {name} table to {path}")
        
        return {
            **state, 
            "metadata_tables": saved_files, 
            "status": "Success",
            "queried_dcs": queried_success
        }

    # ── plotting helpers ────────────────────────────────────────────────────────────────
    
    def _load_and_merge_data(self, state: TremorsState) -> pd.DataFrame | None:
        """
        Reads origin/event/netmag parquet files and merges them.
        Returns a fully-merged DataFrame, or None if the data is unusable.
        """
        metadata_tables = state.get("metadata_tables", {})
    
        if "origin" not in metadata_tables:
            print(f"[{self.name}] No origin data to plot.")
            return None
    
        origin_df = pd.read_parquet(metadata_tables["origin"])
        if origin_df.empty:
            print(f"[{self.name}] Origin table is empty.")
            return None
    
        df = origin_df
    
        # Merge event → origin
        if "event" in metadata_tables:
            try:
                event_df = pd.read_parquet(metadata_tables["event"])
                if not event_df.empty:
                    df = pd.merge(
                        event_df, origin_df,
                        left_on="prefor", right_on="orid",
                        suffixes=("_event", "_origin"),
                    )
                    print(f"[{self.name}] Merged event+origin: {len(df)} records.")
            except Exception as e:
                print(f"[{self.name}] Failed to merge event data: {e}. Using origin only.")
    
        # Merge netmag (strict – drop rows without a magnitude link)
        if "netmag" in metadata_tables and "prefmag" in df.columns:
            try:
                netmag_df = pd.read_parquet(metadata_tables["netmag"])
                if not netmag_df.empty:
                    df["prefmag"] = df["prefmag"].astype(int)
                    netmag_df["magid"] = netmag_df["magid"].astype(int)
                    df = pd.merge(
                        df, netmag_df,
                        left_on="prefmag", right_on="magid",
                        suffixes=("", "_netmag"),
                    )
                    print(f"[{self.name}] Merged netmag: {len(df)} records.")
            except Exception as e:
                print(f"[{self.name}] Failed to merge netmag data: {e}.")
    
        if "magnitude" not in df.columns:
            print(f"[{self.name}] No 'magnitude' column after merges – skipping plot.")
            return None
    
        df = df.dropna(subset=["magnitude"])
        if df.empty:
            print(f"[{self.name}] No events with a valid magnitude.")
            return None
    
        return df
    
    
    def _load_stations(self) -> pd.DataFrame | None:
        """Loads the WAVEFORM_SITE parquet if it exists."""
        path = os.path.join(self.output_dir, "WAVEFORM_SITE.PARQUET")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        return df if not df.empty else None
    
    
    def _setup_map_axes(
        self,
        df_plot: pd.DataFrame,
        stations_df: pd.DataFrame | None,
        params: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a Cartopy figure, computes smart bounds, and adds all basemap
        features.  Returns (fig, ax) ready for data to be plotted on top.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
    
        # Bounds
        pad = 3.0
        all_lats = df_plot["lat"].tolist()
        all_lons = df_plot["lon"].tolist()
    
        if stations_df is not None:
            all_lats.extend(stations_df["lat"].tolist())
            all_lons.extend(stations_df["lon"].tolist())
    
        for key in ("min_lat", "max_lat"):
            if key in params:
                all_lats.append(params[key])
        for key in ("min_lon", "max_lon"):
            if key in params:
                all_lons.append(params[key])
    
        min_lat, max_lat = min(all_lats) - pad, max(all_lats) + pad
        min_lon, max_lon = min(all_lons) - pad, max(all_lons) + pad
    
        lon_span = max_lon - min_lon
        zoom = (
            10 if lon_span < 1 else
            9  if lon_span < 3 else
            8  if lon_span < 6 else
            7  if lon_span < 12 else 6
        )
    
        imagery = GoogleTiles(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        )
        ax.add_image(imagery, zoom, alpha=0.7)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
        ax.add_feature(cfeature.LAND,      facecolor="#f4f4f2",  zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor="#ddeeff",  zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7,        zorder=1)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5,        zorder=1)
        ax.add_feature(cfeature.RIVERS,    linewidth=0.4,        zorder=1)
        ax.add_feature(cfeature.STATES,    edgecolor="#cdd2d6",
                       linewidth=0.8, zorder=2)
    
        gl = ax.gridlines(draw_labels=True, alpha=0.15, linestyle="-",
                          color="black", zorder=2)
        gl.top_labels = False
        gl.right_labels = False
    
        return fig, ax, lon_span  # lon_span re-used by the caller for scalebar
    
    
    def _plot_event_map(
        self,
        df_plot: pd.DataFrame,
        stations_df: pd.DataFrame | None,
        params: dict,
        state: TremorsState,
    ) -> str:
        """
        Draws the geographic map (stations, bounding shapes, events) and
        saves it as a JPEG.  Returns the saved file path.
        """
        try:
            fig, ax, lon_span = self._setup_map_axes(df_plot, stations_df, params)
        except ImportError:
            # Cartopy unavailable – fall back to a plain scatter
            fig = plt.figure(figsize=(10, 6))
            sizes = (10 ** (df_plot["magnitude"] / 2)) * 5
            plt.scatter(
                df_plot["lon"], df_plot["lat"],
                s=sizes, c=df_plot["magnitude"],
                cmap="inferno_r", alpha=0.6, edgecolors="k",
            )
            plt.colorbar(label="Magnitude")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True)
            lon_span = df_plot["lon"].max() - df_plot["lon"].min()
            ax = plt.gca()
    
        # ── radius circles ──────────────────────────────────────────────────
        # DEBUG - RAD need to talk to RGHILL about this ... a little confused about the logic here APRIL 25 2026
        plot_radius = params.get("radius", 0.0)
        if plot_radius > 0:
            rad_km = (
                plot_radius * 111.32
                if params.get("radius_unit", "") == "deg"
                else plot_radius
            )
            for _, row in df_plot.iterrows():
                # points = boundingradius(row["lat"], row["lon"], rad_km)
                points = boundingradius(lat, lon,rad_km, unit='km', numpoints=361, ellipse='WGS84')
                coords = list(zip(points[:, 1], points[:, 0]))
                polygon = Polygon(
                    coords, facecolor="none", alpha=0.5,
                    edgecolor="red", lw=2,
                    transform=ccrs.PlateCarree(), linestyle="--",
                )
                ax.add_patch(polygon)
                lat_lab, lon_lab = points[180, 0], points[180, 1]
                ax.text(
                    lon_lab, lat_lab, f"{rad_km:.0f} km",
                    transform=ccrs.PlateCarree(),
                    ha="center", va="center", fontsize=12, color="red",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                )
    
        # ── bounding box ────────────────────────────────────────────────────
        if all(k in params for k in ("min_lat", "max_lat", "min_lon", "max_lon")):
            box_coords = [
                (params["min_lon"], params["min_lat"]),
                (params["min_lon"], params["max_lat"]),
                (params["max_lon"], params["max_lat"]),
                (params["max_lon"], params["min_lat"]),
            ]
            ax.add_patch(Polygon(
                box_coords, facecolor="gray", alpha=0.3,
                edgecolor="gray", lw=2,
                transform=ccrs.PlateCarree(), linestyle="--",
                label="Search bounds",
            ))
    
        # ── stations ────────────────────────────────────────────────────────
        if stations_df is not None and "sta" in stations_df.columns:
            saved_stas = {
                os.path.basename(f).split("_")[2]
                for f in glob.glob(os.path.join(self.output_dir, "*.mseed"))
                if len(os.path.basename(f).split("_")) >= 3
            }
            saved_df   = stations_df[stations_df["sta"].isin(saved_stas)].drop_duplicates("sta")
            possible_df = stations_df[~stations_df["sta"].isin(saved_stas)].drop_duplicates("sta")
    
            if not possible_df.empty:
                ax.scatter(
                    possible_df["lon"], possible_df["lat"],
                    c="white", s=80, marker="^",
                    transform=ccrs.PlateCarree(),
                    edgecolors="k", linewidths=0.6, zorder=11,
                    label="Possible waveform stations",
                )
            if not saved_df.empty:
                ax.scatter(
                    saved_df["lon"], saved_df["lat"],
                    c="darkgreen", s=140, marker="^",
                    transform=ccrs.PlateCarree(),
                    edgecolors="black", linewidths=1.2, zorder=13,
                    label="Saved waveform stations",
                )
                for _, r in saved_df.iterrows():
                    ax.text(
                        r["lon"] + 0.015, r["lat"] + 0.015, r["sta"],
                        fontsize=8, transform=ccrs.PlateCarree(),
                        color="black", zorder=14, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor="white", alpha=0.7, lw=0.5),
                    )
    
        # ── events ──────────────────────────────────────────────────────────
        sc = ax.scatter(
            df_plot["lon"], df_plot["lat"],
            s=60, c=df_plot["magnitude"],
            cmap="inferno_r", transform=ccrs.PlateCarree(),
            edgecolors="k", alpha=0.9, label="Events", zorder=12,
        )
        plt.colorbar(sc, label="Magnitude", fraction=0.046, pad=0.04)
    
        add_north_arrow(ax, length=0.08, fontsize=16)

        span_km = lon_span * 111.32 
        sb_len = (
            10  if span_km < 50  else
            50  if span_km < 200 else
            100 if span_km < 500 else 500
        )
        # somethong weird going on here ... RAD 04262026
        # add_scalebar(ax, length_km=sb_len, location=(0.07, 0.05),
        #              linewidth=3, fontsize=14)
    
        ax.legend(loc="upper right", markerscale=1.0)
    
        dcs = state.get("queried_dcs") or [state.get("datacenter", "Unknown")]
        title_str = ", ".join(dcs)
        if len(title_str) > 50:
            title_str = title_str[:47]
        plt.title(f"Seismic events & stations ({title_str})")
    
        path = os.path.join(self.output_dir, "event_map.jpg")
        plt.savefig(path, bbox_inches="tight", dpi=300, format="jpeg")
        plt.close()
        print(f"[{self.name}] Saved map → {path}")
        return path
    
    
    def _plot_event_timeline(self, df_plot: pd.DataFrame) -> str:
        """
        Draws the time-vs-depth scatter and saves it as a JPEG.
        Returns the saved file path.
        """
        times = [datetime.fromtimestamp(ts) for ts in df_plot["time"]]
    
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
        print(f"[{self.name}] Saved timeline → {path}")
        return path
    
    
    # ── orchestrator ────────────────────────────────────────────────────────────
    
    def _plot_results_node(self, state: TremorsState) -> TremorsState:
        """Orchestrates data loading and plot generation."""
        if state.get("status") == "Failed":
            return state
    
        print(f"[{self.name}] Generating plots...")
        try:
            df_plot = self._load_and_merge_data(state)
            if df_plot is None:
                no_mag = "magnitude" not in (state.get("metadata_tables") or {})
                status = (
                    "Success (No Valid Magnitudes to Plot)"
                    if no_mag
                    else "Success (No Events with Magnitude)"
                )
                return {**state, "status": status}
    
            stations_df = self._load_stations()
            params      = state.get("search_params", {})
    
            plots = [
                self._plot_event_map(df_plot, stations_df, params, state),
                self._plot_event_timeline(df_plot),
            ]
            return {**state, "plots": plots, "status": "Plots Generated"}
    
        except Exception as e:
            print(f"[{self.name}] Plotting failed: {e}")
            traceback.print_exc()
            return {**state, "error": f"Plotting failed: {e}"}



    # --- Waveform Retrieval ---

    def _retrieve_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Retrieves waveforms for found events if requested.
        """
        if state.get("status") == "Failed":
             return state

        # Check flag
        params = state.get("search_params", {})
        if not (params.get("get_waveforms") or params.get("plot_waveforms")):
             print(f"[{self.name}] Waveform retrieval not requested. Skipping.")
             return state

        print(f"[{self.name}] Starting Waveform Retrieval...")
        
        # Load necessary data
        # We need events and origins to know what to query
        tables = state.get("metadata_tables", {})
        if "event" not in tables or "origin" not in tables:
             print(f"[{self.name}] No event/origin tables found. Skipping waveforms.")
             return state
             
        try:
             events_df = pd.read_parquet(tables["event"])
             origins_df = pd.read_parquet(tables["origin"])
        except Exception as e:
             print(f"[{self.name}] Error reading parquet: {e}")
             return state

        if events_df.empty:
             return state
             
        # Limit events
        # Retrieve top 5 events? Or strictly follow limit?
        # Waveforms are heavy, let's limit to 5 events max unless specific override?
        # For now, process up to 5 events
        target_events = events_df.head(5)
        
        saved_mseed = []
        waveform_metadata_files = {}
        
        # We need to accumulate inventory data for metadata tables
        master_inventory = Inventory(networks=[], source="Tremors")
        
        # Process loop
        for idx, event_row in target_events.iterrows():
            evid = event_row['evid']
            prefor = event_row['prefor']
            dc = event_row['datacenter']
            
            # Get origin info
            origin_row = origins_df[origins_df['orid'] == prefor]
            if origin_row.empty:
                # Fallback to any origin for this event if prefor mismatch?
                # origins_df['evid'] == evid
                origin_row = origins_df[origins_df['evid'] == evid].head(1)
            
            if origin_row.empty: continue
            
            origin_row = origin_row.iloc[0]
            lat = origin_row['lat']
            lon = origin_row['lon']
            time = UTCDateTime(float(origin_row['time']))
            
            print(f"[{self.name}] recovering waveforms for Event {evid} ({dc})...")
            
            # Prioritized list of datacenters to try
            # Always try the source DC first, then fallbacks
            target_dcs = [dc]
            term_fallbacks = ["IRIS", "GEOFON", "NCEDC", "SCEDC", "RASPISHAKE", "EMSC"]
            for fb in term_fallbacks:
                if fb != dc and fb not in target_dcs:
                    target_dcs.append(fb)

            waveform_found = False
            
            for try_dc in target_dcs:
                if waveform_found: break
                
                # print(f"[{self.name}] Trying DC: {try_dc}...")
                client = None
                try:
                    url = WELL_KNOWN_NODES.get(try_dc, None)
                    if url:
                        client = Client(base_url=url, timeout=30)
                    else:
                        client = Client(try_dc, timeout=30)
                except Exception as e:
                    # print(f"[{self.name}] Could not init client for {try_dc}: {e}")
                    continue
                
                if not client: 
                    continue

                # 1. Get Stations
                try:
                    # Grab search params if provided
                    params = state.get("search_params", {})
                    net_param = params.get("net", None)
                    limit_param = params.get("waveform_limit", 10)
                    
                    # Use parsed radius if available, otherwise default to 2.0. Convert to degrees if necessary
                    radius_val = params.get("radius", 2.0)
                    # If radius was given in km, FDSN expects degrees for maxradius, so approx divide by 111
                    if params.get("radius_unit", "") == "km":
                        radius_val = radius_val / 111.32
                        
                    pre_event_sec = params.get("pre_event_sec", 30.0)
                    post_event_sec = params.get("post_event_sec", 600.0)

                    kwargs = {
                        "latitude": lat, "longitude": lon,
                        "minradius": 0, "maxradius": radius_val,
                        "channel": "BH?,HH?,EH?,HN?,EN?,SH?",
                        "level": "channel", 
                        "starttime": time - pre_event_sec, "endtime": time + post_event_sec
                    }
                    if net_param:
                        kwargs["network"] = net_param
                        
                    inventory = client.get_stations(**kwargs)
                    
                    if not inventory:
                        # print(f"[{self.name}] {try_dc}: No stations found.")
                        continue
                    
                    print(f"[{self.name}] {try_dc}: Found {len(inventory.get_contents()['stations'])} stations.")
                    
                    # 2. Get Waveforms (Granular Station-by-Station)
                    dc_waveform_count = 0
                    
                    for net in inventory:
                        for sta in net:
                            if dc_waveform_count >= limit_param: break
                            
                            # Group by Location Code
                            loc_groups = {}
                            for cha in sta.channels:
                                loc = cha.location_code
                                if loc not in loc_groups:
                                    loc_groups[loc] = []
                                loc_groups[loc].append(cha.code)
                            
                            for loc, chans in loc_groups.items():
                                if dc_waveform_count >= limit_param: break
                                
                                chan_str = ",".join(sorted(list(set(chans))))
                                
                                try:
                                    t_start = time - pre_event_sec
                                    t_end = time + post_event_sec
                                    
                                    # print(f"[{self.name}] Requesting: {net.code}.{sta.code}.{loc}.{chan_str}")
                                    st = client.get_waveforms(
                                        network=net.code, station=sta.code, location=loc, channel=chan_str,
                                        starttime=t_start, endtime=t_end,
                                        attach_response=True
                                    )
                                    
                                    if st:
                                        # Use a consistent filename that includes location to avoid overwrites
                                        # Raspberry Shakes often have multiple location codes / data streams
                                        for tr in st:
                                            loc_code = tr.stats.location if tr.stats.location else "--"
                                            fname = f"{evid}_{tr.stats.network}_{tr.stats.station}_{loc_code}_{tr.stats.channel}.mseed"
                                            path = os.path.join(self.output_dir, fname)
                                            
                                            # Write trace. Note: if multiple segments exist for same NSLC, 
                                            # tr.write might overwrite. For now, including location solves most collisions.
                                            tr.write(path, format="MSEED")
                                            
                                            if path not in saved_mseed:
                                                saved_mseed.append(path)
                                            dc_waveform_count += 1
                                        
                                        waveform_found = True
                                        
                                except Exception as e:
                                    # print(f"[{self.name}] Failed {net.code}.{sta.code}: {e}")
                                    continue
                                    
                            if dc_waveform_count >= limit_param: break
                        if dc_waveform_count >= limit_param: break
                    
                    if waveform_found:
                        print(f"[{self.name}] Success: Saved {dc_waveform_count} traces for Event {evid} from {try_dc}.")
                        # Merge Inventory only heavily on success
                        master_inventory.networks.extend(inventory.networks)
                        break

                except Exception as e:
                    error_msg = f"[{self.name}] Error {try_dc} for {evid}: {e}"
                    params = state.get("search_params", {})
                    net_param = params.get("net", None)
                    if "No data available" in str(e) and net_param:
                        error_msg += f"\nNote: The network '{net_param}' may not be hosted at {try_dc}."
                    print(error_msg)
                    continue
            
            if not waveform_found:
                 print(f"[{self.name}] Failed to retrieve waveforms for Event {evid} from any source.")

        # Generate Metadata Tables from Inventory
        if master_inventory:
             print(f"[{self.name}] Generating waveform metadata tables...")
             wf_tables = inventory_to_kbcore(master_inventory)
             
             for name, df in wf_tables.items():
                if not df.empty:
                    filename = f"WAVEFORM_{name}.parquet".upper()
                    path = os.path.join(self.output_dir, filename)
                    df.to_parquet(path, index=False)
                    waveform_metadata_files[name] = path
                    print(f"[{self.name}] Saved {name} table to {path}")

        return {
            **state,
            "waveforms_saved": saved_mseed,
            "waveform_metadata": waveform_metadata_files
        }

    def _plot_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Plots retrieved waveforms.
        """
        if state.get("status") == "Failed": return state
        
        # Check flag
        params = state.get("search_params", {})
        if not params.get("plot_waveforms"):
             return state
             
        mseed_files = state.get("waveforms_saved", [])
        if not mseed_files: return state
        
        print(f"[{self.name}] Plotting waveforms...")
        
        # Group by Event (prefix of filename)
        # {evid}_{net}_{sta}_{chan}.mseed
        
        events_map = {}
        for path in mseed_files:
            fname = os.path.basename(path)
            parts = fname.split('_')
            evid = parts[0]
            if evid not in events_map: events_map[evid] = []
            events_map[evid].append(path)
            
        saved_plots = []
        
        for evid, paths in events_map.items():
            try:
                st = Stream()
                for p in paths:
                    st += read(p)
                
                # Simple Plot
                # Use matplotlib backend to save file
                title = f"Event {evid} Waveforms"
                outfile = os.path.join(self.output_dir, f"waveforms_{evid}.png")
                
                # Plot
                st.plot(outfile=outfile, number_of_ticks=5) # Basic plot
                saved_plots.append(outfile)
                print(f"[{self.name}] Saved plot to {outfile}")
                
            except Exception as e:
                print(f"[{self.name}] Error plotting event {evid}: {e}")
        
        return {
            **state,
            "waveform_plots": saved_plots
        }

    def _retrieve_continuous_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Retrieves continuous waveforms for a specified network/station/location/channel and time range.
        """
        if state.get("status") == "Failed": return state

        params = state.get("search_params", {})
        if not params.get("get_continuous_waveforms"):
             return state

        print(f"[{self.name}] Starting Continuous Waveform Retrieval...")

        try:
            # Prepare arguments matching the parser
            args = argparse.Namespace()
            args.outdir = self.output_dir
            args.parallel = params.get("parallel", 4)
            args.bulk_chunk = params.get("bulk_chunk", 50)
            args.dir_date = params.get("dir_date", False)
            args.dir_stat = params.get("dir_stat", False)
            args.response = params.get("response", False)
            args.download = True
            args.print = False

            # Check if we should dynamically query for stations
            stations_file_provided = params.get("stations_file", "")
            has_spatial = any(k in params for k in ["min_lat", "max_lat", "min_lon", "max_lon"])
            
            if (not stations_file_provided or not os.path.isfile(stations_file_provided)) and has_spatial:
                print(f"[{self.name}] No valid stations_file found, but regional parameters detected. Discovering stations...")
                t_start = UTCDateTime(params.get("min_date", "2016-01-01T00:00:00"))
                t_end = UTCDateTime(params.get("max_date", "2016-01-03T00:00:00"))
                net_req = params.get("net", "*")
                sta_req = params.get("sta", "*")
                loc_req = params.get("loc", "*")
                chan_req = params.get("chan", "*")
                
                query_kwargs = {
                    "starttime": t_start,
                    "endtime": t_end,
                    "network": net_req,
                    "station": sta_req,
                    "location": loc_req,
                    "channel": chan_req,
                    "level": "channel"
                }

                if "min_lat" in params:
                    query_kwargs["minlatitude"] = params["min_lat"]
                    query_kwargs["maxlatitude"] = params["max_lat"]
                    query_kwargs["minlongitude"] = params["min_lon"]
                    query_kwargs["maxlongitude"] = params["max_lon"]

                targets = self._determine_target_dcs(params)
                print(f"[{self.name}] Querying datacenters for stations: {targets}")
                
                proposed_stations = set() # using a set to deduplicate
                for dc in targets:
                    url = WELL_KNOWN_NODES.get(dc, dc)
                    try:
                        client = Client(base_url=url, timeout=30) if dc in WELL_KNOWN_NODES else Client(dc, timeout=30)
                        inv = client.get_stations(**query_kwargs)
                        # Extract channels
                        for net in inv:
                            for sta in net:
                                for chan in sta:
                                    # Create the row tuple to ensure uniqueness
                                    # Format: <net sta chan loc datacenter tstart tend>
                                    loc_code = chan.location_code if chan.location_code else "--"
                                    row = (net.code, sta.code, chan.code, loc_code, dc, str(t_start), str(t_end))
                                    proposed_stations.add(row)
                        print(f"[{self.name}] Found {len(proposed_stations)} unique channels from {dc} so far.")
                    except Exception as e:
                        print(f"[{self.name}] Could not get stations from {dc}: {e}")
                
                if not proposed_stations:
                    return {**state, "error": "No stations found for the given regional query and time window.", "status": "Failed"}
                
                # Write to proposed file
                proposed_file = os.path.join(self.output_dir, "test_stations_proposed.txt")
                does_dir_exist(self.output_dir)
                with open(proposed_file, "w") as f:
                    f.write("# net sta chan loc datacenter tstart tend\n")
                    for row in sorted(list(proposed_stations)):
                        f.write(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} {row[6]}\n")
                
                args.stations = proposed_file
                print(f"[{self.name}] Generated {proposed_file} with {len(proposed_stations)} entries.")
            else:
                args.stations = stations_file_provided

            if not os.path.isfile(args.stations):
                print(f"[{self.name}] Target stations file not found: {args.stations}")
                return {**state, "error": f"Target stations file not found: {args.stations}", "status": "Failed"}

            print(f"[{self.name}] Requesting continuous waveforms using {args.stations}")
            
            wave_list = DailyBulkWaveforms(args)
            dl_sched = Scheduler(wave_list.parallel)
            dl_sched.start(iter(wave_list.dl_params))
            
            # Find all mseed files that have been written recursively
            saved_mseed = []
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    if file.endswith(".mseed"):
                        saved_mseed.append(os.path.join(root, file))
                    
            print(f"[{self.name}] Bulk request complete. Traces saved: {len(saved_mseed)}.")

            return {
                **state,
                "continuous_waveforms_saved": saved_mseed,
                "status": "Success"
            }

        except Exception as e:
            print(f"[{self.name}] Continuous waveform retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return {**state, "error": f"Continuous waveform retrieval failed: {e}", "status": "Failed"}

    def _plot_continuous_waveforms_node(self, state: TremorsState) -> TremorsState:
        """
        Plots retrieved continuous waveforms.
        """
        if state.get("status") == "Failed": return state
        
        mseed_files = state.get("continuous_waveforms_saved", [])
        if not mseed_files: 
            print(f"[{self.name}] No continuous waveforms to plot.")
            return state
            
        print(f"[{self.name}] Plotting continuous waveforms...")
        
        saved_plots = []
        try:
            st = Stream()
            for p in mseed_files:
                st += read(p)
            
            # Simple Plot
            title = f"Continuous Waveforms"
            outfile = os.path.join(self.output_dir, "continuous_waveforms.png")
            
            # Plot
            st.plot(outfile=outfile, number_of_ticks=5, handle=True)
            saved_plots.append(outfile)
            print(f"[{self.name}] Saved continuous waveform plot to {outfile}")
            
        except Exception as e:
            print(f"[{self.name}] Error plotting continuous waveforms: {e}")
            
        return {
            **state,
            "continuous_waveform_plots": saved_plots
        }


class Scheduler:
    """
    Multithread scheduler for chunked bulk waveform downloads.
    """

    def __init__(self, nproc):
        self._queue = multiprocessing.Queue()
        self._nproc = nproc
        self.__init_workers()

    def __init_workers(self):
        self._workers = []
        for _ in range(self._nproc):
            self._workers.append(PullWave(self._queue))

    def start(self, tasks):
        queue_count = 0
        for task in tasks:
            self._queue.put(copy.deepcopy(task))
            queue_count += 1

        self._queue.put(None)
        print_str = f"\nAdd {queue_count} bulk requests to the queue"
        print(print_str)
        logging.info(print_str)
        print_str = f"Number of threads {self._nproc}"
        print(print_str)
        logging.info(print_str)

        for worker in self._workers:
            worker.start()

        for worker in self._workers:
            worker.join()


class PullWave(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self, name="PullWave")
        self._queue = queue
        self._clients = {}
        self._response_written = set()

    def run(self):
        while True:
            params = self._queue.get()
            if params is None:
                self._queue.put(None)
                break
            self._pull_data(params)

    def _get_client(self, datacenter):
        if datacenter not in self._clients:
            self._clients[datacenter] = obspy.clients.fdsn.Client(datacenter)
        return self._clients[datacenter]

    def _pull_data(self, param):
        datacenter = param["datacenter"]
        bulk = param["bulk"]
        requests = param["requests"]
        client = self._get_client(datacenter)

        try:
            stream = client.get_waveforms_bulk(bulk)
            print_str = (
                f"Fetched bulk request: datacenter={datacenter} "
                f"chunk={param['chunk_id']} requests={len(requests)}"
            )
            print(print_str)
            logging.info(print_str)
        except Exception as exc:
            print_str = (
                f"Skip bulk request: datacenter={datacenter} "
                f"chunk={param['chunk_id']} requests={len(requests)} error={exc}"
            )
            print(print_str)
            logging.info(print_str)
            return

        for request in requests:
            self._write_request(client, stream, request, param["writer"])

    def _write_request(self, client, stream, request, writer):
        st = stream.select(
            network=request["net"],
            station=request["sta"],
            location=request["loc_select"],
            channel=request["channel_select"],
        ).copy()

        if len(st) == 0:
            print_str = (
                f"No data : {request['net']} {request['sta']} {request['loc']} "
                f"{request['channel_request']} {request['tstart']} {request['tend']}"
            )
            print(print_str)
            logging.info(print_str)
            return

        st.trim(request["tstart"], request["tend"], nearest_sample=False)
        if len(st) == 0:
            print_str = (
                f"No data : {request['net']} {request['sta']} {request['loc']} "
                f"{request['channel_request']} {request['tstart']} {request['tend']}"
            )
            print(print_str)
            logging.info(print_str)
            return

        if writer["response"]:
            self._write_response_file(client=client, request=request, outdir=writer["outdir"])

        write_daily_per_station_all_channels(
            st=st,
            outdir=writer["outdir"],
            dir_date=writer["dir_date"],
            dir_stat=writer["dir_stat"],
            loc=request["loc"],
            requested_chan=request["chan"],
            overwrite=writer["overwrite"],
        )

    def _write_response_file(self, client, request, outdir):
        resp_key = (
            request["datacenter"],
            request["net"],
            request["sta"],
            request["loc"],
            request["chan"],
        )
        if resp_key in self._response_written:
            return

        resp_dir = station_output_dir(outdir, request["net"], request["sta"], use_station_dir=True)
        resp_dir = os.path.join(resp_dir, "resp")
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
            print_str = f"Get : {resp_file}"
        except Exception as exc:
            print_str = f"Skip : {resp_file} error={exc}"
        print(print_str)
        logging.info(print_str)


class DailyBulkWaveforms:
    def __init__(self, argv):
        self.cwd = os.getcwd()
        self.outdir = os.path.join(self.cwd, argv.outdir)
        self.parallel = argv.parallel
        self.download = argv.download
        self.dir_date = argv.dir_date
        self.dir_stat = argv.dir_stat
        self.bulk_chunk = argv.bulk_chunk
        self.writer = {
            "outdir": self.outdir,
            "dir_date": self.dir_date,
            "dir_stat": self.dir_stat,
            "overwrite": False,
            "response": argv.response,
        }

        self.requests = self._read_station_file(argv.stations)
        self.dl_params = self._build_bulk_tasks()

    def _read_station_file(self, station_file):
        if not os.path.isfile(station_file):
            raise FileNotFoundError(
                f"--stations must point to a file. Not found: {station_file}"
            )

        requests = []
        with open(station_file, "r") as fobj:
            for iline, line in enumerate(fobj, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                tmp = line.split()
                if len(tmp) != 7:
                    raise ValueError(
                        "Each --stations line must contain "
                        "<net sta chan loc datacenter tstart tend>. "
                        f"Invalid line {iline}: {line}"
                    )

                net, sta, chan, loc, datacenter, tstart, tend = tmp
                request = {
                    "net": net.upper(),
                    "sta": sta.upper(),
                    "chan": chan.upper(),
                    "loc": loc,
                    "loc_select": self._normalize_loc(loc),
                    "channel_request": self._channel_request(chan.upper()),
                    "channel_select": self._channel_select(chan.upper()),
                    "datacenter": datacenter,
                    "tstart": obspy.UTCDateTime(tstart),
                    "tend": obspy.UTCDateTime(tend),
                }
                if request["tend"] <= request["tstart"]:
                    raise ValueError(
                        f"tend must be later than tstart on line {iline}: {line}"
                    )
                requests.append(request)

        if not requests:
            raise ValueError("No station requests found in --stations file")
        return requests

    def _build_bulk_tasks(self):
        grouped = defaultdict(list)
        for request in self.requests:
            for request_chunk in self._split_request_into_day_chunks(request):
                if self._request_has_missing_days(request_chunk):
                    grouped[request_chunk["datacenter"]].append(request_chunk)

        tasks = []
        for datacenter, requests in sorted(grouped.items()):
            requests = sorted(
                requests,
                key=lambda x: (x["tstart"], x["tend"], x["net"], x["sta"], x["loc"]),
            )

            chunk_requests = []
            chunk_days = 0
            chunk_id = 1
            for request in requests:
                request_days = self._request_days(request)
                if chunk_requests and chunk_days + request_days > self.bulk_chunk:
                    tasks.append(
                        self._make_task(datacenter, chunk_id, chunk_requests)
                    )
                    chunk_requests = []
                    chunk_days = 0
                    chunk_id += 1

                chunk_requests.append(request)
                chunk_days += request_days

            if chunk_requests:
                tasks.append(self._make_task(datacenter, chunk_id, chunk_requests))
        return tasks

    def _split_request_into_day_chunks(self, request):
        requests = []
        chunk_seconds = self.bulk_chunk * 86400
        t0 = request["tstart"]
        while t0 < request["tend"]:
            t1 = min(t0 + chunk_seconds, request["tend"])
            request_chunk = copy.deepcopy(request)
            request_chunk["tstart"] = t0
            request_chunk["tend"] = t1
            requests.append(request_chunk)
            t0 = t1
        return requests

    def _make_task(self, datacenter, chunk_id, requests):
        return {
            "datacenter": datacenter,
            "chunk_id": f"{datacenter}:{chunk_id}",
            "bulk": [
                (
                    request["net"],
                    request["sta"],
                    request["loc"],
                    request["channel_request"],
                    request["tstart"],
                    request["tend"],
                )
                for request in requests
            ],
            "requests": requests,
            "writer": self.writer,
        }

    def _request_days(self, request):
        return max(1, int((request["tend"] - request["tstart"]) / 86400 + 0.999999))

    def _request_has_missing_days(self, request):
        day_start = _day_start_utc(request["tstart"])
        last_day = _day_start_utc(request["tend"] - 0.000001)
        while day_start <= last_day:
            outfile = build_daily_outfile(
                outdir=self.outdir,
                net=request["net"],
                sta=request["sta"],
                loc=request["loc"],
                chan=request["chan"],
                day_start=day_start,
                dir_date=self.dir_date,
                dir_stat=self.dir_stat,
            )
            if _should_write_file(outfile):
                return True
            day_start += 86400
        return False

    def _channel_request(self, chan):
        if len(chan) == 2:
            return chan + "*"
        return chan

    def _channel_select(self, chan):
        if len(chan) == 2:
            return chan + "?"
        return chan

    def _normalize_loc(self, loc):
        loc = loc.strip()
        if loc in ["", "--", "**"]:
            return "*"
        return loc


def does_dir_exist(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def _day_start_utc(t):
    return obspy.UTCDateTime(t.year, t.month, t.day)


def station_output_dir(outdir, net, sta, use_station_dir=False):
    path = outdir
    if use_station_dir:
        path = os.path.join(path, net, sta)
    return path


def build_daily_outfile(outdir, net, sta, loc, chan, day_start, dir_date=False, dir_stat=False):
    path = station_output_dir(outdir, net, sta, use_station_dir=dir_stat)
    if dir_date:
        path = os.path.join(
            path,
            str(day_start.datetime.year),
            f"{day_start.datetime.timetuple().tm_yday:03d}",
        )
    fout = (
        f"{net}.{sta}.{loc}.{chan}_"
        f"{day_start.datetime.year}{day_start.datetime.timetuple().tm_yday:03d}.mseed"
    )
    return os.path.join(path, fout)


def _should_write_file(outfile):
    if not os.path.isfile(outfile):
        return True

    print(f"Exist: {os.path.basename(outfile)}")
    if os.path.getsize(outfile) < 4096:
        print("Redownload")
        return True
    return False


def write_daily_per_station_all_channels(
    st,
    outdir,
    merge=True,
    dir_date=False,
    dir_stat=False,
    loc="*",
    requested_chan=None,
    overwrite=False,
):
    """
    Write one MiniSEED file per station per UTC day containing all requested channels.
    """
    if not isinstance(st, obspy.Stream):
        raise TypeError("st must be an obspy.Stream")

    work = st.copy()
    if merge:
        merged = obspy.Stream()
        for tr_id in sorted({tr.id for tr in work}):
            tmp = work.select(id=tr_id).copy()
            tmp.merge(method=1, fill_value=None)
            merged += tmp
        work = merged

    by_station = defaultdict(obspy.Stream)
    for tr in work:
        by_station[(tr.stats.network, tr.stats.station)] += tr

    for (net, sta), sst in by_station.items():
        if len(sst) == 0:
            continue

        t0 = min(tr.stats.starttime for tr in sst)
        t1 = max(tr.stats.endtime for tr in sst)
        day_start = _day_start_utc(t0)
        last_day = _day_start_utc(t1)

        while day_start <= last_day:
            day_end = day_start + 86400
            day_st = sst.slice(day_start, day_end).copy()
            if len(day_st) > 0:
                if merge:
                    day_st.merge(method=1, fill_value=None)

                chan_label = requested_chan or _infer_channel_label(day_st)
                outfile = build_daily_outfile(
                    outdir=outdir,
                    net=net,
                    sta=sta,
                    loc=loc,
                    chan=chan_label,
                    day_start=day_start,
                    dir_date=dir_date,
                    dir_stat=dir_stat,
                )

                if overwrite or _should_write_file(outfile):
                    does_dir_exist(os.path.dirname(outfile))
                    try:
                        day_st.write(outfile, format="MSEED")
                        print_str = f"Get : {outfile}"
                    except Exception as exc:
                        print_str = f"Skip : {outfile} error={exc}"
                    print(print_str)
                    logging.info(print_str)
            day_start = day_end


def _infer_channel_label(st):
    prefixes = sorted(
        {tr.stats.channel[:2] for tr in st if getattr(tr.stats, "channel", "")}
    )
    if len(prefixes) == 1:
        return prefixes[0]

    channels = sorted({tr.stats.channel for tr in st if getattr(tr.stats, "channel", "")})
    if len(channels) == 1:
        return channels[0]
    return "MULTI"





# # --- Entry Point for Testing ---
# if __name__ == "__main__":
#     # Simple test run
#     print("Initializing TremorsAgent...")
#     # Mock LLM
#     mock_llm = type('MockLLM', (), {})() 
    
#     agent = TremorsAgent(mock_llm, output_dir="./test_output")
    
#     initial_state = {
#         "query": "Find big events",
#         "datacenter": "ISC",
#         # "service_status": {},
#         # "search_params": {},
#         # "metadata_tables": {},
#         # "status": "Start",
#         # "error": None
#     }
    
#     print("Running Agent...")
#     result = agent._action.invoke(initial_state)
#     print("Result State:", result)
