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

schema.py
=========
Utility functions for converting ObsPy Inventory and Catalog objects into
KBCore-like pandas DataFrames for local storage, plotting, and archiving.

Tables produced
---------------
inventory_to_kbcore → affiliation, network, site, sitechan, remark
catalog_to_kbcore   → event, origin, origerr, arrival, assoc, netmag,
                       stamag, momenttensor, focalmech, principalaxes

The ``extended=True`` (default) variant retains extra columns such as
``datacenter``, ``net``, and ``loc`` that are not part of standard KBCore
but are useful for provenance tracking and FDSN round-tripping.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict

import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import FlinnEngdahl


# ---------------------------------------------------------------------------
# KBCore lookup tables
# ---------------------------------------------------------------------------

# Simplified from jkmacc/pisces:
# https://github.com/LANL-Seismoacoustics/pisces/blob/master/pisces/catalog.py#L54
KBCORE_EVENT_TYPE: Dict[str, str] = {
    'accidental explosion': 'ex',
    'anthropogenic event':  'ep',
    'atmospheric event':    'xm',
    'blasting levee':       'ec',
    'cavity collapse':      'mc',
    'chemical explosion':   'ec',
    'collapse':             'mc',
    'controlled explosion': 'ec',
    'earthquake':           'qp',
    'experimental explosion': 'ec',
    'explosion':            'ex',
    'ice quake':            'qp',
    'industrial explosion': 'ec',
    'meteorite':            'xm',
    'mine collapse':        'mc',
    'mining explosion':     'me',
    'not existing':         '-',
    'not reported':         '-',
    'nuclear explosion':    'en',
    'other event':          '-',
    'quarry blast':         'ec',
    'rock burst':           'mb',
    'sonic blast':          'xm',
    'sonic boom':           'xm',
    'thunder':              'xm',
}

# Default pick-time uncertainties (seconds) keyed by phase name
DEFDELTIM: Dict[str, float] = {
    'P':  1.0, 'S':  2.0,
    'Pn': 0.75, 'Sn': 1.5,
    'Pg': 1.5,  'Sg': 2.5,
    'Lg': 2.5,
}

FIRSTMOTION: Dict[str, str] = {
    'undecidable': '-',
    'positive':    'c',
    'negative':    'd',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jdate(utcdatetime) -> int:
    """Return a KBCore-style YYYYDDD integer, or -1 for None."""
    if utcdatetime is None:
        return -1
    return int(utcdatetime.strftime('%Y%j'))


def _extract_id(resource_id_obj) -> int:
    """
    Robustly extract an integer ID from an ObsPy ResourceIdentifier.

    Different datacenters format resource IDs in different ways:

    - Standard FDSN ``key=value``:  ``quakeml:us.anss.org/event/evid=12345``
    - URI path (e.g. EMSC):         ``quakeml:www.seismicportal.eu/event/12345``
    - Bare string / UUID:           ``12345`` or ``abc-def-...``

    Strategy
    --------
    1. Try the ``key=value`` format — split on ``=``, take the last token.
    2. Try a trailing integer after a ``/`` (URI path style).
    3. Fall back to a stable positive hash so the ID is always an int and
       never crashes downstream joins, even for exotic formats.

    Parameters
    ----------
    resource_id_obj:
        An ObsPy ResourceIdentifier, a plain string, or None.

    Returns
    -------
    int
        A non-negative integer that uniquely identifies the resource within
        a single catalog (collisions are theoretically possible with the hash
        fallback but are negligible in practice for catalog sizes < ~10 M).
    """
    if not resource_id_obj:
        return -1

    rid_str = (
        resource_id_obj.id
        if hasattr(resource_id_obj, 'id')
        else str(resource_id_obj)
    )

    # 1. Standard key=value format
    if '=' in rid_str:
        base = rid_str.split('=')[-1]
        if base.isdigit():
            return int(base)

    # 2. Trailing integer after a path separator (e.g. EMSC URI style)
    match = re.search(r'/(\d+)$', rid_str)
    if match:
        return int(match.group(1))

    # 3. Whole string is a plain integer
    if rid_str.isdigit():
        return int(rid_str)

    # 4. Stable hash fallback — never crashes, always an int
    return abs(hash(rid_str)) % 10_000_000


# ---------------------------------------------------------------------------
# inventory_to_kbcore
# ---------------------------------------------------------------------------

def inventory_to_kbcore(inventory, datacenter: str = '-', extended: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Convert an ObsPy Inventory to KBCore-like DataFrames.

    Parameters
    ----------
    inventory:
        ObsPy Inventory object (channel-level recommended).
    datacenter:
        Label stamped into the ``datacenter`` extended column on every row.
    extended:
        If True (default), retain non-standard columns (``datacenter``,
        ``net``, ``loc``) that are useful for provenance and FDSN queries.
        If False, drop them to produce standard KBCore-compatible tables.

    Returns
    -------
    dict of {table_name: pd.DataFrame}
        Keys: ``affiliation``, ``network``, ``site``, ``sitechan``, ``remark``.
    """
    affiliation_data = []
    network_data     = []
    site_data        = []
    sitechan_data    = []
    remark_data      = []

    comment_id = 0
    auth   = (inventory.source or inventory.sender or '-').replace('\n', ' ')
    lddate = (inventory.created or UTCDateTime()).strftime('%Y-%m-%d %H%M%S')

    for net in inventory:
        if net.comments:
            comment_id += 1
            for i, comment in enumerate(net.comments):
                remark_data.append({
                    'commid': comment_id,
                    'lineno': i + 1,
                    'remark': comment.value.replace('\n', ' '),
                    'lddate': lddate,
                })

        network_data.append({
            'datacenter': datacenter,
            'net':        net.code,
            'netname':    net.description.replace('\n', ' ') if net.description else '-',
            'nettype':    '-',
            'auth':       auth,
            'commid':     comment_id if net.comments else -1,
            'lddate':     lddate,
        })

        for sta in net:
            affiliation_data.append({
                'datacenter': datacenter,
                'net':        net.code,
                'sta':        sta.code,
                # Use station dates (more precise than network dates) where available;
                # fall back to network dates, then KBCore sentinel values.
                'time':    (sta.start_date or net.start_date).timestamp
                           if (sta.start_date or net.start_date) else -9999999999.999,
                'endtime': (sta.end_date or net.end_date).timestamp
                           if (sta.end_date or net.end_date) else 9999999999.999,
                'lddate':  lddate,
            })

            site_data.append({
                'datacenter': datacenter,
                'net':        net.code,
                'sta':        sta.code,
                'ondate':     jdate(sta.start_date),
                'offdate':    jdate(sta.end_date) if sta.end_date else 2286324,
                'lat':        sta.latitude,
                'lon':        sta.longitude,
                'elev':       sta.elevation / 1000.0 if sta.elevation is not None else 0.0,
                'staname':    (sta.site.name if sta.site else sta.description or '-').replace('\n', ' '),
                'statype':    '-',
                'refsta':     '-',
                'dnorth':     0.0,
                'deast':      0.0,
                'lddate':     lddate,
            })

            for chan in sta:
                sitechan_data.append({
                    'datacenter': datacenter,
                    'net':        net.code,
                    'sta':        sta.code,
                    'chan':       chan.code,
                    'loc':        chan.location_code,
                    'ondate':     jdate(chan.start_date),
                    'offdate':    jdate(chan.end_date) if chan.end_date else 2286324,
                    'chanid':     -1,
                    'ctype':      'b' if 'BEAM' in (chan.types or []) else '-',
                    'edepth':     chan.depth if chan.depth is not None else 0.0,
                    'hang':       chan.azimuth if chan.azimuth is not None else None,
                    'vang':       chan.dip if chan.dip is not None else None,
                    'descrip':    chan.description.replace('\n', ' ') if chan.description else '-',
                    'lddate':     lddate,
                })

    tables = {
        'affiliation': pd.DataFrame(affiliation_data),
        'network':     pd.DataFrame(network_data),
        'site':        pd.DataFrame(site_data),
        'sitechan':    pd.DataFrame(sitechan_data),
        'remark':      pd.DataFrame(remark_data),
    }

    if not extended:
        tables['network']     = tables['network'].drop(columns=['datacenter'])
        tables['affiliation'] = tables['affiliation'].drop(columns=['datacenter'])
        tables['site']        = tables['site'].drop(columns=['datacenter', 'net'])
        tables['sitechan']    = tables['sitechan'].drop(columns=['datacenter', 'net', 'loc'])

    return tables


# ---------------------------------------------------------------------------
# catalog_to_kbcore
# ---------------------------------------------------------------------------

def catalog_to_kbcore(catalog, datacenter: str = '-', extended: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Convert an ObsPy Catalog to KBCore-like DataFrames.

    Parameters
    ----------
    catalog:
        ObsPy Catalog object.
    datacenter:
        Provenance label stamped on every row.
    extended:
        If True (default), retain non-standard columns.  If False, drop them
        to produce standard KBCore-compatible tables.

    Returns
    -------
    dict of {table_name: pd.DataFrame}
        Keys: ``event``, ``origin``, ``origerr``, ``arrival``, ``assoc``,
        ``netmag``, ``stamag``, ``momenttensor``, ``focalmech``,
        ``principalaxes``.
    """
    fe = FlinnEngdahl()

    lddate = datetime.now().strftime('%Y-%m-%d %H%M%S')

    event_data         = []
    origin_data        = []
    origerr_data       = []
    arrival_data       = []
    assoc_data         = []
    netmag_data        = []
    stamag_data        = []
    focalmech_data     = []
    momenttensor_data  = []
    principalaxes_data = []

    for event in catalog:
        evid   = _extract_id(event.resource_id)
        prefor = _extract_id(event.preferred_origin_id)
        prefmag = (
            _extract_id(event.preferred_magnitude_id)
            if event.preferred_magnitude_id is not None else -1
        )
        preffm = (
            _extract_id(event.preferred_focal_mechanism_id)
            if event.preferred_focal_mechanism_id is not None else -1
        )
        etype = KBCORE_EVENT_TYPE.get(event.event_type, '-')

        # FIX 1: guard against None creation_info or missing fields
        ev_ci      = event.creation_info or {}
        ev_agency  = getattr(ev_ci, 'agency_id', '-') or '-'
        ev_auth    = getattr(ev_ci, 'author',    '-') or '-'
        ev_evname  = (
            event.event_descriptions[0].text
            if event.event_descriptions else '-'
        )

        event_data.append({
            'datacenter': datacenter,
            'agency':     ev_agency,
            'evid':       evid,
            'evname':     ev_evname,
            'prefor':     prefor,
            'prefmag':    prefmag,
            'preffm':     preffm,
            'auth':       ev_auth,
            'commid':     -1,
            'lddate':     lddate,
        })

        arids = {}

        for origin in event.origins:
            orid = _extract_id(origin.resource_id)
            grn  = fe.get_number(origin.longitude, origin.latitude)

            m_default   = -999.0
            mid_default = -1

            # FIX 2: guard against None creation_info on origins
            orig_ci     = origin.creation_info or {}
            orig_agency = getattr(orig_ci, 'agency_id', '-') or '-'
            orig_auth   = getattr(orig_ci, 'author',    '-') or '-'

            origin_data.append({
                'datacenter':    datacenter,
                'agency':        orig_agency,
                'lat':           origin.latitude,
                'lon':           origin.longitude,
                'depth':         origin.depth / 1000.0 if origin.depth is not None else -999,
                'time':          origin.time.timestamp,
                'orid':          orid,
                'evid':          evid,
                'jdate':         jdate(origin.time),
                'nass':          origin.quality.associated_phase_count if origin.quality else -1,
                'ndef':          origin.quality.used_phase_count        if origin.quality else -1,
                'ndp':           -1,
                'grn':           grn,
                'srn':           -1,
                'etype':         etype,
                'dtype':         origin.depth_type,
                'mb':            m_default,
                'mbid':          mid_default,
                'ms':            m_default,
                'msid':          mid_default,
                'ml':            m_default,
                'mlid':          mid_default,
                'mw':            m_default,
                'mwid':          mid_default,
                'algorithm':     '-',
                'azimuthal_gap': (
                    origin.quality.azimuthal_gap
                    if origin.quality and origin.quality.azimuthal_gap is not None
                    else -999.0
                ),
                'auth':          orig_auth,
                'commid':        -1,
                'lddate':        lddate,
            })

            origerr_data.append({
                'datacenter': datacenter,
                'orid':       orid,
                'sxx':        -100000000.0,
                'syy':        -100000000.0,
                'szz':        -100000000.0,
                'stt':        -100000000.0,
                'sxy':        -100000000.0,
                'sxz':        -100000000.0,
                'syz':        -100000000.0,
                'stx':        -100000000.0,
                'sty':        -100000000.0,
                'stz':        -100000000.0,
                'sdobs':  (
                    origin.quality.standard_error
                    if origin.quality and origin.quality.standard_error else -1.0
                ),
                'smajax': (
                    origin.origin_uncertainty.confidence_ellipsoid.semi_major_axis_length / 1000.0
                    if origin.origin_uncertainty and origin.origin_uncertainty.confidence_ellipsoid
                    else -1.0
                ),
                'sminax': (
                    origin.origin_uncertainty.confidence_ellipsoid.semi_minor_axis_length / 1000.0
                    if origin.origin_uncertainty and origin.origin_uncertainty.confidence_ellipsoid
                    else -1.0
                ),
                'strike': (
                    origin.origin_uncertainty.confidence_ellipsoid.major_axis_azimuth
                    if origin.origin_uncertainty and origin.origin_uncertainty.confidence_ellipsoid
                    else -1.0
                ),
                'sdepth': (
                    origin.depth_errors.uncertainty / 1000.0
                    if origin.depth_errors and origin.depth_errors.uncertainty else -1.0
                ),
                'stime':  (
                    origin.time_errors.uncertainty
                    if origin.time_errors and origin.time_errors.uncertainty else -1.0
                ),
                'conf':   (
                    origin.origin_uncertainty.confidence_level / 100.0
                    if origin.origin_uncertainty and origin.origin_uncertainty.confidence_level
                    else -1.0
                ),
                'commid': -1,
                'lddate': lddate,
            })

            for arrival in origin.arrivals:
                pickid = arrival.pick_id.id

                esaz  = arrival.azimuth if arrival.azimuth is not None else -1.0
                seaz  = esaz + 180.0    if esaz != -1.0              else -1.0
                if seaz > 360:
                    seaz -= 360.0

                wgt    = arrival.time_weight              if arrival.time_weight              is not None else 0.0
                slowgt = arrival.horizontal_slowness_weight if arrival.horizontal_slowness_weight is not None else 0.0
                azwgt  = arrival.backazimuth_weight        if arrival.backazimuth_weight        is not None else 0.0

                arids[pickid] = {
                    'datacenter': datacenter,
                    'arid':       _extract_id(arrival.pick_id),
                    'orid':       orid,
                    'phase':      arrival.phase,
                    'esaz':       esaz,
                    'seaz':       seaz,
                    'delta':      arrival.distance         if arrival.distance         is not None else -1.0,
                    'tres':       arrival.time_residual    if arrival.time_residual    is not None else -999.0,
                    'wgt':        wgt,
                    'timedef':    'd' if wgt    > 0 else '-',
                    'slores':     arrival.horizontal_slowness_residual if arrival.horizontal_slowness_residual is not None else -999.0,
                    'slodef':     'd' if slowgt > 0 else '-',
                    'azres':      arrival.backazimuth_residual         if arrival.backazimuth_residual         is not None else -999.0,
                    'azdef':      'd' if azwgt  > 0 else '-',
                }

        for pick in event.picks:
            pickid = pick.resource_id.id
            if pickid not in arids:
                continue

            deltim = (
                pick.time_errors.uncertainty
                if pick.time_errors and pick.time_errors.uncertainty else None
            )
            if deltim is None:
                phase = arids[pickid]['phase']
                deltim = DEFDELTIM.get(phase, DEFDELTIM.get(phase[0] if phase else '', -1.0))

            azimuth = pick.backazimuth if pick.backazimuth is not None else -1.0
            delaz   = -1.0
            if azimuth != -1.0:
                delaz = (
                    pick.backazimuth_errors.uncertainty
                    if pick.backazimuth_errors and pick.backazimuth_errors.uncertainty
                    else 10.0
                )

            slow   = pick.horizontal_slowness if pick.horizontal_slowness is not None else -1.0
            delslo = -1.0
            if slow != -1.0:
                delslo = (
                    pick.horizontal_slowness_errors.uncertainty
                    if pick.horizontal_slowness_errors and pick.horizontal_slowness_errors.uncertainty
                    else 5.0
                )

            polarity = pick.polarity if pick.polarity else 'undecidable'
            fm       = FIRSTMOTION.get(polarity, '-')

            arrival_data.append({
                'datacenter': datacenter,
                'net':        pick.waveform_id.network_code,
                'sta':        pick.waveform_id.station_code,
                'time':       pick.time.timestamp,
                'arid':       int(arids[pickid]['arid']),
                'jdate':      jdate(pick.time),
                'stassid':    -1,
                'chanid':     -1,
                'chan':        pick.waveform_id.channel_code,
                'loc':         pick.waveform_id.location_code,
                'iphase':      pick.phase_hint if pick.phase_hint else arids[pickid]['phase'],
                'stype':       '-',
                'deltim':      deltim,
                'azimuth':     azimuth,
                'delaz':       delaz,
                'slow':        slow,
                'delslo':      delslo,
                'ema':         -1.0,
                'rect':        -1.0,
                'amp':         -1.0,
                'per':         -1.0,
                'logat':       -999.0,
                'clip':        '-',
                'fm':          fm,
                'snr':         -1.0,
                'qual':        '-',
                'auth':        datacenter,
                'commid':      -1,
                'lddate':      lddate,
            })

            assoc_data.append({
                'datacenter': datacenter,
                'arid':       int(arids[pickid]['arid']),
                'orid':       int(arids[pickid]['orid']),
                'net':        pick.waveform_id.network_code  or '',
                'sta':        pick.waveform_id.station_code,
                'chan':        pick.waveform_id.channel_code  or '',
                'loc':         pick.waveform_id.location_code or '',
                'phase':       arids[pickid]['phase'],
                'belief':      -1.0,
                'delta':       arids[pickid]['delta'],
                'seaz':        arids[pickid]['seaz'],
                'esaz':        arids[pickid]['esaz'],
                'timeres':     arids[pickid]['tres'],
                'timedef':     arids[pickid]['timedef'],
                'azres':       arids[pickid]['azres'],
                'azdef':       arids[pickid]['azdef'],
                'slores':      arids[pickid]['slores'],
                'slodef':      arids[pickid]['slodef'],
                'emares':      -999.0,
                'wgt':         arids[pickid]['wgt'],
                'vmodel':      '-',
                'commid':      -1,
                'lddate':      lddate,
            })

        for magnitude in event.magnitudes:
            orid    = _extract_id(magnitude.origin_id)
            magid   = _extract_id(magnitude.resource_id)
            magtype = magnitude.magnitude_type
            mag     = magnitude.mag
            nsta    = magnitude.station_count if magnitude.station_count else -1
            magunc  = (
                magnitude.mag_errors.uncertainty
                if magnitude.mag_errors and magnitude.mag_errors.uncertainty else -1.0
            )

            for origin_rec in origin_data:
                if origin_rec['orid'] == orid:
                    if re.match(r'^mb', magtype, re.IGNORECASE):
                        origin_rec['mb']   = mag
                        origin_rec['mbid'] = magid
                    elif re.match(r'^ml', magtype, re.IGNORECASE):
                        origin_rec['ml']   = mag
                        origin_rec['mlid'] = magid
                    elif re.match(r'^ms', magtype, re.IGNORECASE):
                        origin_rec['ms']   = mag
                        origin_rec['msid'] = magid
                    elif re.match(r'^mw', magtype, re.IGNORECASE) or magtype == 'M':
                        # GFZ reports Mw as bare 'M'
                        origin_rec['mw']   = mag
                        origin_rec['mwid'] = magid

            netmag_data.append({
                'datacenter': datacenter,
                'magid':      magid,
                'net':        '-',
                'orid':       orid,
                'evid':       evid,
                'magtype':    magtype,
                'nsta':       nsta,
                'magnitude':  mag,
                'uncertainty': magunc,
                'auth':       datacenter,
                'commid':     -1,
                'lddate':     lddate,
            })

            # FIX 3: station_magnitudes lives on the event, not the magnitude.
            # The original code was nested inside `for magnitude in event.magnitudes`
            # which iterated event.station_magnitudes once per network magnitude —
            # producing duplicates.  Iterate it once at the event level instead.

        for sta_mag in event.station_magnitudes:
            sta_mag_orid  = _extract_id(sta_mag.origin_id)   if sta_mag.origin_id   else -1
            sta_mag_magid = _extract_id(sta_mag.resource_id) if sta_mag.resource_id else -1
            amplitudeid   = _extract_id(sta_mag.amplitude_id) if sta_mag.amplitude_id else -1

            stamag_data.append({
                'datacenter': datacenter,
                'net':        sta_mag.waveform_id.network_code if sta_mag.waveform_id else '-',
                'magid':      sta_mag_magid,
                'ampid':      amplitudeid,
                'stamagid':   sta_mag_magid,
                'sta':        sta_mag.waveform_id.station_code if sta_mag.waveform_id else '-',
                'arid':       -1,
                'orid':       sta_mag_orid,
                'evid':       evid,
                'phase':      '-',
                'delta':      -1.0,
                'magtype':    sta_mag.station_magnitude_type,
                'magnitude':  sta_mag.mag,
                'uncertainty': -1.0,
                'magres':     -999,
                'magdef':     '-',
                'mmodel':     '-',
                'auth':       datacenter,
                'commid':     -1,
                'lddate':     lddate,
            })

        for focal_mech in event.focal_mechanisms:
            fmid    = _extract_id(focal_mech.resource_id)
            fm_auth = datacenter
            mt_orid = -1

            if focal_mech.creation_info is not None:
                fm_auth = focal_mech.creation_info.author or datacenter

            # FIX 4: guard against focal_mech.moment_tensor being None before
            # accessing derived_origin_id — the original code accessed it
            # unconditionally before the `if focal_mech.moment_tensor:` block.
            if focal_mech.moment_tensor:
                if focal_mech.moment_tensor.derived_origin_id:
                    mt_orid = _extract_id(focal_mech.moment_tensor.derived_origin_id)

                scalar_moment = (
                    focal_mech.moment_tensor.scalar_moment
                    if focal_mech.moment_tensor.scalar_moment is not None else -999.0
                )
                duration = (
                    focal_mech.moment_tensor.source_time_function.duration
                    if focal_mech.moment_tensor.source_time_function is not None else -1.0
                )
                fclvd = (
                    focal_mech.moment_tensor.clvd
                    if focal_mech.moment_tensor.clvd is not None else -999.0
                )

                mrr = mtt = mpp = mrt = mtp = mpr = 0.0
                mrr_e = mtt_e = mpp_e = mrt_e = mtp_e = mpr_e = 0.0
                if focal_mech.moment_tensor.tensor:
                    t = focal_mech.moment_tensor.tensor
                    mrr = t.m_rr if t.m_rr is not None else 0.0
                    mtt = t.m_tt if t.m_tt is not None else 0.0
                    mpp = t.m_pp if t.m_pp is not None else 0.0
                    mrt = t.m_rt if t.m_rt is not None else 0.0
                    mtp = t.m_tp if t.m_tp is not None else 0.0
                    mpr = t.m_rp if t.m_rp is not None else 0.0
                    mrr_e = t.m_rr_errors.uncertainty if t.m_rr_errors and t.m_rr_errors.uncertainty is not None else 0.0
                    mtt_e = t.m_tt_errors.uncertainty if t.m_tt_errors and t.m_tt_errors.uncertainty is not None else 0.0
                    mpp_e = t.m_pp_errors.uncertainty if t.m_pp_errors and t.m_pp_errors.uncertainty is not None else 0.0
                    mrt_e = t.m_rt_errors.uncertainty if t.m_rt_errors and t.m_rt_errors.uncertainty is not None else 0.0
                    mtp_e = t.m_tp_errors.uncertainty if t.m_tp_errors and t.m_tp_errors.uncertainty is not None else 0.0
                    mpr_e = t.m_rp_errors.uncertainty if t.m_rp_errors and t.m_rp_errors.uncertainty is not None else 0.0

                momenttensor_data.append({
                    'datacenter':       datacenter,
                    'fmid':             fmid,
                    'evid':             evid,
                    'orid':             mt_orid,
                    'scalar_moment':    scalar_moment,
                    'scale_factor':     -1,
                    'mrr':              mrr,  'mtt':  mtt,  'mpp':  mpp,
                    'mrt':              mrt,  'mtp':  mtp,  'mpr':  mpr,
                    'scalar_moment_err': -999.0,
                    'mrr_err':          mrr_e, 'mtt_err': mtt_e, 'mpp_err': mpp_e,
                    'mrt_err':          mrt_e, 'mtp_err': mtp_e, 'mpr_err': mpr_e,
                    'fclvd':            fclvd,
                    'duration':         duration,
                    'auth':             fm_auth,
                    'commid':           -1,
                    'lddate':           lddate,
                })

            # nodal planes and focalmech record — written regardless of
            # whether moment_tensor was present
            str1 = dip1 = rake1 = str2 = dip2 = rake2 = -999.0
            if focal_mech.nodal_planes is not None:
                np1  = focal_mech.nodal_planes.nodal_plane_1
                np2  = focal_mech.nodal_planes.nodal_plane_2
                str1  = np1.strike if np1 and np1.strike is not None else -999.0
                dip1  = np1.dip    if np1 and np1.dip    is not None else -999.0
                rake1 = np1.rake   if np1 and np1.rake   is not None else -999.0
                str2  = np2.strike if np2 and np2.strike is not None else -999.0
                dip2  = np2.dip    if np2 and np2.dip    is not None else -999.0
                rake2 = np2.rake   if np2 and np2.rake   is not None else -999.0

            focalmech_data.append({
                'datacenter': datacenter,
                'fmid':       fmid,
                'evid':       evid,
                'orid':       mt_orid,
                'strike1':    str1,  'dip1':  dip1,  'rake1': rake1,
                'strike2':    str2,  'dip2':  dip2,  'rake2': rake2,
                'plane_pref': -1,
                'auth':       fm_auth,
                'commid':     -1,
                'lddate':     lddate,
            })

            if focal_mech.principal_axes:
                def _ax(axis, field):
                    return getattr(axis, field) if axis and getattr(axis, field) is not None else -999.0

                pa = focal_mech.principal_axes
                principalaxes_data.append({
                    'datacenter':   datacenter,
                    'fmid':         fmid,
                    'evid':         evid,
                    'orid':         mt_orid,
                    'scale_factor': 0,
                    't_length':     _ax(pa.t_axis, 'length'),
                    't_azimuth':    _ax(pa.t_axis, 'azimuth'),
                    't_plunge':     _ax(pa.t_axis, 'plunge'),
                    'n_length':     _ax(pa.n_axis, 'length'),
                    'n_azimuth':    _ax(pa.n_axis, 'azimuth'),
                    'n_plunge':     _ax(pa.n_axis, 'plunge'),
                    'p_length':     _ax(pa.p_axis, 'length'),
                    'p_azimuth':    _ax(pa.p_axis, 'azimuth'),
                    'p_plunge':     _ax(pa.p_axis, 'plunge'),
                    't_length_err':  -1.0, 't_azimuth_err': -1.0, 't_plunge_err': -1.0,
                    'n_length_err':  -1.0, 'n_azimuth_err': -1.0, 'n_plunge_err': -1.0,
                    'p_length_err':  -1.0, 'p_azimuth_err': -1.0, 'p_plunge_err': -1.0,
                    'auth':         fm_auth,
                    'commid':       -1,
                    'lddate':       lddate,
                })

    tables = {
        'event':         pd.DataFrame(event_data),
        'origin':        pd.DataFrame(origin_data),
        'origerr':       pd.DataFrame(origerr_data),
        'arrival':       pd.DataFrame(arrival_data),
        'assoc':         pd.DataFrame(assoc_data),
        'netmag':        pd.DataFrame(netmag_data),
        'stamag':        pd.DataFrame(stamag_data),
        'momenttensor':  pd.DataFrame(momenttensor_data),
        'focalmech':     pd.DataFrame(focalmech_data),
        'principalaxes': pd.DataFrame(principalaxes_data),
    }

    if not extended:
        tables['event']   = tables['event'].drop(columns=['datacenter', 'agency', 'prefmag', 'preffm'])
        tables['origin']  = tables['origin'].drop(columns=['datacenter', 'agency', 'mw', 'mwid', 'azimuthal_gap'])
        tables['origerr'] = tables['origerr'].drop(columns=['datacenter'])
        tables['arrival'] = tables['arrival'].drop(columns=['datacenter', 'net', 'loc'])
        tables['assoc']   = tables['assoc'].drop(columns=['datacenter', 'net', 'chan', 'loc'])
        tables['netmag']  = tables['netmag'].drop(columns=['datacenter', 'net'])
        tables['stamag']  = tables['stamag'].drop(columns=['datacenter', 'net', 'stamagid'])
        # FIX 5: was referencing undefined local variable `isc_cat_tables` —
        # use `tables` consistently
        tables.pop('momenttensor')
        tables.pop('principalaxes')
        tables.pop('focalmech')

    return tables