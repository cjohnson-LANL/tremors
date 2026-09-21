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

params.py
=========
The FDSN search-parameter schema and its domain verifier.

Two independent layers guard the natural-language → FDSN translation:

``SearchParams``
    A pydantic model giving the planner *structured output*. Field
    descriptions live here rather than in a prompt, so the model is driven by
    a JSON schema instead of prose instructions. This layer catches
    **malformed** plans (wrong types, unknown keys, unparseable output).

``verify_search_params``
    A pure function that catches **well-formed but wrong** plans — an inverted
    bounding box, a magnitude range that excludes everything, a datacenter
    that does not exist. Schema validity says nothing about physical
    plausibility, and every one of these mistakes would otherwise drive real
    FDSN downloads and produce a confidently empty or misleading result.

The verifier is deliberately dependency-free (no ObsPy, no agent imports) and
takes its set of valid datacenters as an argument, so it can be exercised
without a network or an LLM.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Physically meaningful bounds, used by the verifier.
_LAT_MIN, _LAT_MAX = -90.0, 90.0
_LON_MIN, _LON_MAX = -180.0, 180.0

# Deepest recorded earthquakes are ~700 km; allow generous headroom before
# calling a depth implausible.
_DEPTH_MAX_KM = 1000.0

# Moment magnitude has no hard ceiling, but nothing above ~10 has occurred.
_MAG_MIN, _MAG_MAX = -2.0, 10.0

# Tolerance for "in the future": clocks and timezones differ slightly between
# the caller and the datacenters.
_FUTURE_TOLERANCE = timedelta(days=1)


class SearchParams(BaseModel):
    """
    Structured FDSN search parameters extracted from a natural-language query.

    Passed to ``llm.with_structured_output()`` so the planner returns a
    validated object rather than free text that has to be hand-parsed. Every
    field is optional: the planner emits only what the user actually asked
    for, and the pipeline applies its own defaults downstream.

    ``extra="forbid"`` is deliberate — a hallucinated key (``"magnitude"``
    instead of ``"min_mag"``) must be a loud validation error, not a silently
    ignored filter that widens the search.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Where to look ──────────────────────────────────────────────────
    datacenter: Optional[str] = Field(
        None,
        description='FDSN datacenter short name, e.g. "ISC", "USGS", "IRIS". Defaults to "ISC".',
    )

    # ── Time window ────────────────────────────────────────────────────
    min_date: Optional[str] = Field(
        None, description='Start of the time window, ISO-8601, e.g. "2010-01-01T00:00:00".'
    )
    max_date: Optional[str] = Field(
        None, description="End of the time window, ISO-8601."
    )

    # ── Search area ────────────────────────────────────────────────────
    min_lat: Optional[float] = Field(None, description="Southern bound, decimal degrees.")
    max_lat: Optional[float] = Field(None, description="Northern bound, decimal degrees.")
    min_lon: Optional[float] = Field(None, description="Western bound, decimal degrees.")
    max_lon: Optional[float] = Field(None, description="Eastern bound, decimal degrees.")
    radius: Optional[float] = Field(
        None,
        description="Search radius around a single point. REQUIRED when the user "
                    "gives one location rather than a bounding box.",
    )
    radius_unit: Optional[str] = Field(
        None, description='Unit for radius: "km" or "deg". Defaults to "km".'
    )

    # ── Event filters ──────────────────────────────────────────────────
    min_depth: Optional[float] = Field(None, description="Minimum event depth in km.")
    max_depth: Optional[float] = Field(None, description="Maximum event depth in km.")
    min_mag: Optional[float] = Field(None, description="Minimum event magnitude.")
    max_mag: Optional[float] = Field(None, description="Maximum event magnitude.")
    limit: Optional[int] = Field(
        None, description="Maximum number of events to return per datacenter. Defaults to 100."
    )

    # ── What to retrieve ───────────────────────────────────────────────
    get_waveforms: Optional[bool] = Field(
        None, description="True to download per-event waveforms for the matched events."
    )
    plot_waveforms: Optional[bool] = Field(
        None, description="True to also render per-event waveform figures."
    )
    get_continuous_waveforms: Optional[bool] = Field(
        None,
        description="True ONLY for continuous-stream requests, where the user wants a "
                    "time span of data from stations rather than data around events. "
                    "Mutually exclusive with get_waveforms.",
    )

    # ── Station / channel selection ────────────────────────────────────
    stations_file: Optional[str] = Field(
        None, description="Path to an existing station list file (optional, continuous mode)."
    )
    net: Optional[str] = Field(None, description='Network code filter, e.g. "CI".')
    sta: Optional[str] = Field(None, description='Station code filter, e.g. "ANMO".')
    loc: Optional[str] = Field(None, description='Location code filter, e.g. "00".')
    chan: Optional[str] = Field(None, description='Channel code filter, e.g. "BHZ" or "BH*".')

    # ── Download / layout tuning ───────────────────────────────────────
    parallel: Optional[int] = Field(None, description="Download worker processes. Defaults to 4.")
    bulk_chunk: Optional[int] = Field(None, description="Requests per bulk FDSN call. Defaults to 50.")
    dir_date: Optional[bool] = Field(None, description="Organize output by YYYY/DOY. Defaults to false.")
    dir_stat: Optional[bool] = Field(None, description="Organize output by network/station. Defaults to false.")
    response: Optional[bool] = Field(None, description="Also download StationXML response files. Defaults to false.")
    pre_event_sec: Optional[float] = Field(
        None, description="Seconds of waveform before the event origin. Defaults to 30."
    )
    post_event_sec: Optional[float] = Field(
        None, description="Seconds of waveform after the event origin. Defaults to 600."
    )

    def to_params(self) -> dict:
        """Return a plain dict with unset fields dropped (the pipeline's format)."""
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Domain verification
# ---------------------------------------------------------------------------

def _parse_iso(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 string leniently, returning None if unparseable.

    Accepts a trailing ``Z`` and bare dates, both of which models emit
    routinely. Always returns a timezone-aware value (assuming UTC when no
    offset is given) so comparisons never mix naive and aware datetimes.
    """
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _check_range(
    params: dict,
    min_key: str,
    max_key: str,
    label: str,
    problems: List[str],
) -> None:
    """Flag an inverted numeric range (min greater than max)."""
    lo, hi = params.get(min_key), params.get(max_key)
    if lo is not None and hi is not None and lo > hi:
        problems.append(
            f"{label} range is inverted: {min_key}={lo} is greater than {max_key}={hi}. "
            f"No {label.lower()} can satisfy it, so the search would return nothing."
        )


def _check_bounds(
    params: dict, key: str, lo: float, hi: float, unit: str, problems: List[str]
) -> None:
    """Flag a value outside its physically meaningful bounds."""
    value = params.get(key)
    if value is not None and not (lo <= value <= hi):
        problems.append(f"{key}={value} is outside the valid range [{lo}, {hi}] {unit}.")


def verify_search_params(
    params: dict,
    valid_dcs: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Check *params* for plans that are schema-valid but cannot be right.

    Structured output guarantees the shape of a plan, not its sense: a model
    can return a perfectly typed bounding box with the corners swapped, or a
    magnitude window that excludes every earthquake. Each such plan would
    otherwise reach the datacenters and come back plausibly empty.

    Parameters
    ----------
    params:
        The search-parameter dict (as produced by :meth:`SearchParams.to_params`).
    valid_dcs:
        Iterable of acceptable datacenter names. Callers pass
        ``WELL_KNOWN_NODES.keys()``; supplying it as an argument keeps this
        module free of any dependency on the agent.

    Returns
    -------
    list of str
        Human-readable problems, empty when the plan is usable. Each string is
        written to be shown directly to a user, since it becomes the body of a
        clarification request.
    """
    problems: List[str] = []

    # ── Inverted ranges ────────────────────────────────────────────────
    _check_range(params, "min_lat",   "max_lat",   "Latitude",  problems)
    _check_range(params, "min_depth", "max_depth", "Depth",     problems)
    _check_range(params, "min_mag",   "max_mag",   "Magnitude", problems)

    # Longitude is checked separately: min_lon > max_lon is how an
    # antimeridian-crossing box is legitimately expressed, so it is reported as
    # an ambiguity to confirm rather than as an outright error.
    min_lon, max_lon = params.get("min_lon"), params.get("max_lon")
    if min_lon is not None and max_lon is not None and min_lon > max_lon:
        problems.append(
            f"Longitude range min_lon={min_lon} is greater than max_lon={max_lon}. "
            "This reads as a box crossing the antimeridian (180°/-180°), which FDSN "
            "bounding-box queries do not support. Split it into two searches, or "
            "confirm the bounds were meant to be swapped."
        )

    # ── Physical bounds ────────────────────────────────────────────────
    for key in ("min_lat", "max_lat"):
        _check_bounds(params, key, _LAT_MIN, _LAT_MAX, "degrees", problems)
    for key in ("min_lon", "max_lon"):
        _check_bounds(params, key, _LON_MIN, _LON_MAX, "degrees", problems)
    for key in ("min_depth", "max_depth"):
        _check_bounds(params, key, -10.0, _DEPTH_MAX_KM, "km", problems)
    for key in ("min_mag", "max_mag"):
        _check_bounds(params, key, _MAG_MIN, _MAG_MAX, "(magnitude units)", problems)

    # ── Partial bounding box ───────────────────────────────────────────
    bbox_keys = ("min_lat", "max_lat", "min_lon", "max_lon")
    present   = [k for k in bbox_keys if params.get(k) is not None]
    if present and len(present) != len(bbox_keys):
        missing = [k for k in bbox_keys if k not in present]
        problems.append(
            f"The search area is incomplete: {', '.join(present)} given but "
            f"{', '.join(missing)} missing. Specify all four bounds, or a single "
            "location with a radius."
        )

    # ── Single point without a radius ──────────────────────────────────
    # Generalizes the check the planner previously hardcoded: a zero-extent box
    # matches nothing, so a radius is required to turn it into a search area.
    if len(present) == len(bbox_keys):
        same_lat = params["min_lat"] == params["max_lat"]
        same_lon = params["min_lon"] == params["max_lon"]
        if (same_lat or same_lon) and not params.get("radius"):
            problems.append(
                f"A single location ({params['min_lat']}, {params['min_lon']}) was given "
                "without a search radius. Specify a radius (e.g. 'within 50 km' or "
                "'within 1 degree') to define a search area."
            )

    if params.get("radius") is not None and params["radius"] <= 0:
        problems.append(f"radius={params['radius']} must be greater than zero.")

    unit = params.get("radius_unit")
    if unit is not None and str(unit).lower() not in ("km", "deg", "degree", "degrees"):
        problems.append(f'radius_unit={unit!r} is not understood; use "km" or "deg".')

    # ── Time window ───────────────────────────────────────────────────
    now      = datetime.now(timezone.utc)
    parsed   = {}
    for key in ("min_date", "max_date"):
        raw = params.get(key)
        if raw is None:
            continue
        when = _parse_iso(raw)
        if when is None:
            problems.append(f"{key}={raw!r} is not a valid ISO-8601 date/time.")
        else:
            parsed[key] = when
            if when > now + _FUTURE_TOLERANCE:
                problems.append(
                    f"{key}={raw} is in the future; no data exists for it yet."
                )

    if "min_date" in parsed and "max_date" in parsed:
        if parsed["min_date"] > parsed["max_date"]:
            problems.append(
                f"The time window is inverted: min_date={params['min_date']} is after "
                f"max_date={params['max_date']}."
            )
        elif parsed["min_date"] == parsed["max_date"]:
            problems.append(
                f"min_date and max_date are identical ({params['min_date']}), so the "
                "time window has zero length. Give a range."
            )

    # ── Datacenter ────────────────────────────────────────────────────
    dc = params.get("datacenter")
    if dc and valid_dcs is not None:
        known = {str(d).upper() for d in valid_dcs}
        if str(dc).upper() not in known:
            problems.append(
                f"datacenter={dc!r} is not a known FDSN node. Use one of the supported "
                "short names (e.g. ISC, USGS, IRIS, GEOFON, EMSC, NCEDC, SCEDC)."
            )

    # ── Mutually exclusive retrieval modes ────────────────────────────
    if params.get("get_continuous_waveforms") and params.get("get_waveforms"):
        problems.append(
            "get_continuous_waveforms and get_waveforms are both set, but they are "
            "different pipelines: continuous mode retrieves a time span from stations, "
            "event mode retrieves windows around catalog events. Choose one."
        )

    if params.get("get_continuous_waveforms"):
        # Continuous mode never runs the event cascade, so event-only filters
        # would be silently discarded.
        ignored = [k for k in ("min_mag", "max_mag", "min_depth", "max_depth")
                   if params.get(k) is not None]
        if ignored:
            problems.append(
                f"Continuous-waveform mode ignores the event filter(s) "
                f"{', '.join(ignored)} because no event catalog is queried. Drop them, "
                "or ask for event waveforms instead."
            )
        if not any(params.get(k) for k in ("net", "sta", "stations_file")):
            problems.append(
                "Continuous-waveform mode needs a station selection: give a network "
                "(net), a station (sta), or a stations_file. Without one the request "
                "would span every station at the datacenter."
            )
        if not (params.get("min_date") and params.get("max_date")):
            problems.append(
                "Continuous-waveform mode needs both min_date and max_date to bound "
                "the download."
            )

    # ── Referenced files must exist ───────────────────────────────────
    stations_file = params.get("stations_file")
    if stations_file and not os.path.exists(str(stations_file)):
        problems.append(
            f"stations_file={stations_file!r} does not exist on disk."
        )

    # ── Tuning sanity ─────────────────────────────────────────────────
    for key in ("limit", "parallel", "bulk_chunk"):
        value = params.get(key)
        if value is not None and value <= 0:
            problems.append(f"{key}={value} must be greater than zero.")

    for key in ("pre_event_sec", "post_event_sec"):
        value = params.get(key)
        if value is not None and value < 0:
            problems.append(f"{key}={value} cannot be negative.")

    return problems
