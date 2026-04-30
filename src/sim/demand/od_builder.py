"""Commuting OD core assembly from place-level commuting flows."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sim._text import norm_name as _norm_name
from sim._metrics import persons_to_vehicles
from sim.demand.config import DemandBuildCfg, _validate_shares
from sim.demand.naming import _match_zone_id, _zone_index
from sim.demand.gateways import _assign_external_to_gateway_fallback
from sim.demand.seeds import _zero_matrix

logger = logging.getLogger(__name__)


def _resolve_place_candidates_directional(
    name: str,
    *,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    groups: Dict[str, List[Tuple[int, float]]],
    inbound_lookup: Dict[str, List[Tuple[int, float]]],
    outbound_lookup: Dict[str, List[Tuple[int, float]]],
    gateways: Dict[str, List[Tuple[int, float]]],
    allow_legacy_fallback: bool,
) -> Tuple[
    List[Tuple[int, float]], List[Tuple[int, float]], str, str,
]:
    """Resolve place name to zone candidates with separate in/out gateways."""
    zone_id = _match_zone_id(name, primary, stripped)
    if zone_id is not None:
        c = [(zone_id, 1.0)]
        return c, c, "direct", "internal"

    key = _norm_name(name)
    if key in groups:
        c = groups[key]
        return c, c, "group", "internal"

    in_cands = inbound_lookup.get(key)
    out_cands = outbound_lookup.get(key)
    if in_cands or out_cands:
        return in_cands or [], out_cands or [], "lookup", "external"

    if allow_legacy_fallback and gateways:
        cands = _assign_external_to_gateway_fallback(key, gateways)
        if cands:
            return cands, cands, "legacy_fallback", "external"

    return [], [], "missing", "missing"


def _build_od_cores(
    df: pd.DataFrame,
    zone_ids: np.ndarray,
    bcfg: DemandBuildCfg,
    *,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    groups: Dict[str, List[Tuple[int, float]]],
    gateways: Dict[str, List[Tuple[int, float]]],
    external_lookup: Dict[str, List[Tuple[int, float]]],
    external_lookup_out: Optional[Dict[str, List[Tuple[int, float]]]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    _validate_shares(bcfg.periods, bcfg.shares_work, "weekday.work")
    _validate_shares(bcfg.periods, bcfg.shares_school, "weekday.school")

    if external_lookup_out is None:
        external_lookup_out = external_lookup

    z2i = _zone_index(zone_ids)
    zset = set(z2i.keys())
    n = len(zone_ids)

    period_cores = [f"wd_{p}" for p in bcfg.periods]
    core_names = period_cores + ["wd_daily"]
    mats: Dict[str, np.ndarray] = {name: _zero_matrix(n) for name in core_names}

    stats: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "mapped_direct": 0,
        "mapped_group": 0,
        "mapped_external_lookup": 0,
        "mapped_external_legacy_fallback": 0,
        "missing_origin": 0,
        "missing_destination": 0,
        "pairs_used": 0,
        "skipped_external_external_rows": 0,
    }

    origin_col = "op_obec" if "op_obec" in df.columns else None
    dest_col = "doj_obec" if "doj_obec" in df.columns else None
    if origin_col is None or dest_col is None:
        raise RuntimeError(f"Expected columns op_obec/doj_obec, got: {list(df.columns)}")

    origin_names = df[origin_col].fillna("").astype(str).str.strip().values
    dest_names = df[dest_col].fillna("").astype(str).str.strip().values

    all_unique_names: set[str] = set(origin_names) | set(dest_names)
    all_unique_names.discard("")

    _EMPTY_DIR: Tuple[list, list, str, str] = ([], [], "missing", "missing")
    name_cache_dir: Dict[str, Tuple[
        List[Tuple[int, float]], List[Tuple[int, float]], str, str,
    ]] = {}
    for name in all_unique_names:
        name_cache_dir[name] = _resolve_place_candidates_directional(
            name,
            primary=primary,
            stripped=stripped,
            groups=groups,
            inbound_lookup=external_lookup,
            outbound_lookup=external_lookup_out,
            gateways=gateways,
            allow_legacy_fallback=bcfg.external.allow_legacy_fallback,
        )
    logger.info("Pre-cached %d unique place-name resolutions (directional)", len(name_cache_dir))

    work_col_name = "dojizdka_prace"
    school_col_name = "dojizdka_skola"
    work_raw = (
        pd.to_numeric(df[work_col_name], errors="coerce").fillna(0.0).values
        if work_col_name in df.columns else np.zeros(len(df))
    )
    school_raw = (
        pd.to_numeric(df[school_col_name], errors="coerce").fillna(0.0).values
        if school_col_name in df.columns else np.zeros(len(df))
    )

    w_conv = persons_to_vehicles(1.0, car_share=bcfg.conv_work.car_share,
                                occupancy=bcfg.conv_work.occupancy,
                                trips_per_person=bcfg.conv_work.trips_per_person)
    s_conv = persons_to_vehicles(1.0, car_share=bcfg.conv_school.car_share,
                                occupancy=bcfg.conv_school.occupancy,
                                trips_per_person=bcfg.conv_school.trips_per_person)
    work_v_all = np.maximum(work_raw, 0.0) * w_conv
    school_v_all = np.maximum(school_raw, 0.0) * s_conv

    period_list = bcfg.periods
    n_periods = len(period_list)
    work_out = np.array([bcfg.shares_work.outbound.get(p, 0.0) for p in period_list])
    school_out = np.array([bcfg.shares_school.outbound.get(p, 0.0) for p in period_list])
    work_ret = np.array([bcfg.shares_work.return_.get(p, 0.0) for p in period_list])
    school_ret = np.array([bcfg.shares_school.return_.get(p, 0.0) for p in period_list])

    mat_arrays = [mats[f"wd_{p}"] for p in period_list]

    _MODE_STAT = {"direct": "mapped_direct", "group": "mapped_group",
                  "lookup": "mapped_external_lookup",
                  "legacy_fallback": "mapped_external_legacy_fallback"}
    n_rows = len(df)
    use_ext_int = bcfg.external.use_external_internal
    use_int_ext = bcfg.external.use_internal_external
    only_internal = bcfg.only_internal_pairs
    ext_comm_scale = bcfg.external.external_commuting_scale

    for idx in range(n_rows):
        on = origin_names[idx]
        dn = dest_names[idx]

        if not on:
            stats["missing_origin"] += 1
            continue
        if not dn:
            stats["missing_destination"] += 1
            continue

        o_entry = name_cache_dir.get(on, _EMPTY_DIR)
        d_entry = name_cache_dir.get(dn, _EMPTY_DIR)
        o_in_cands, o_out_cands, o_mode, o_kind = o_entry
        d_in_cands, d_out_cands, d_mode, d_kind = d_entry

        if not o_in_cands and not o_out_cands:
            stats["missing_origin"] += 1
            continue
        if not d_in_cands and not d_out_cands:
            stats["missing_destination"] += 1
            continue

        if o_kind == "external" and d_kind == "external":
            stats["skipped_external_external_rows"] += 1
            continue

        if only_internal and (o_kind != "internal" or d_kind != "internal"):
            continue
        if o_kind == "external" and d_kind == "internal" and not use_ext_int:
            continue
        if o_kind == "internal" and d_kind == "external" and not use_int_ext:
            continue

        o_stat = _MODE_STAT.get(o_mode)
        if o_stat:
            stats[o_stat] += 1
        d_stat = _MODE_STAT.get(d_mode)
        if d_stat:
            stats[d_stat] += 1

        wv = work_v_all[idx]
        sv = school_v_all[idx]
        if wv <= 0 and sv <= 0:
            continue

        fwd = wv * work_out + sv * school_out
        ret = wv * work_ret + sv * school_ret

        is_external_pair = (o_kind == "external") or (d_kind == "external")
        if is_external_pair and ext_comm_scale != 1.0:
            fwd = fwd * ext_comm_scale
            ret = ret * ext_comm_scale

        fwd_o_cands = o_in_cands
        fwd_d_cands = d_out_cands

        ret_o_cands = d_in_cands
        ret_d_cands = o_out_cands

        for oz, ow in fwd_o_cands:
            if oz not in zset:
                continue
            oi = z2i[oz]
            for dz, dw in fwd_d_cands:
                if dz not in zset:
                    continue
                di = z2i[dz]
                f = float(ow) * float(dw)
                for pi in range(n_periods):
                    mat_arrays[pi][oi, di] += fwd[pi] * f

        for oz, ow in ret_o_cands:
            if oz not in zset:
                continue
            oi = z2i[oz]
            for dz, dw in ret_d_cands:
                if dz not in zset:
                    continue
                di = z2i[dz]
                f = float(ow) * float(dw)
                for pi in range(n_periods):
                    mat_arrays[pi][oi, di] += ret[pi] * f

        stats["pairs_used"] += 1

    mats["wd_daily"] = sum(mats[name] for name in period_cores)

    summary = {
        **stats,
        "zones": int(len(zone_ids)),
        "cores_sum": {name: round(float(mats[name].sum()), 1) for name in core_names},
        "nonzero_cells": {name: int(np.count_nonzero(mats[name])) for name in core_names},
    }

    return mats, summary
