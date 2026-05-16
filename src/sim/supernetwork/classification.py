"""Classify commuting relations as internal, inbound, outbound, or through-traffic."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sim._metrics import persons_to_vehicles_from_cfg
from sim.supernetwork.config import SuperCfg
from sim.supernetwork.routing import shortest_path_cost_batched


def _boundary_sector(boundary_angle_deg: float) -> int:
    """Compass octant (45°) from gateway ``boundary_angle`` on the model AOI."""
    return int(((float(boundary_angle_deg) + 22.5) % 360.0) // 45) % 8


def classify_relations(
    G: nx.DiGraph,
    commuting_pairs: pd.DataFrame,
    gateway_lookup: pd.DataFrame,
    gateway_pair_costs: pd.DataFrame,
    internal_zone_names: set[str],
    cfg_root: Dict[str, Any],
    cfg: SuperCfg,
    gateway_boundary_angles: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    if "rank_in" not in gateway_lookup.columns:
        if "rank" in gateway_lookup.columns:
            gateway_lookup = gateway_lookup.copy()
            gateway_lookup["rank_in"] = gateway_lookup["rank"]
        else:
            gateway_lookup["rank_in"] = np.nan
    if "rank_out" not in gateway_lookup.columns:
        if "rank" in gateway_lookup.columns:
            gateway_lookup["rank_out"] = gateway_lookup["rank"]
        else:
            gateway_lookup["rank_out"] = np.nan
    if "route_cost_to_gateway_s" not in gateway_lookup.columns:
        gateway_lookup["route_cost_to_gateway_s"] = gateway_lookup.get("route_cost_s", np.nan)
    if "route_cost_from_gateway_s" not in gateway_lookup.columns:
        gateway_lookup["route_cost_from_gateway_s"] = gateway_lookup.get("route_cost_s", np.nan)

    if (
        "unit_id" in gateway_lookup.columns
        and not gateway_lookup.empty
        and gateway_lookup["unit_id"].astype(str).str.strip().ne("").any()
    ):
        id_col = "unit_id"
    else:
        id_col = "place_key"
    best_in = (
        gateway_lookup.dropna(subset=["rank_in"])
        .sort_values([id_col, "rank_in"])
        .drop_duplicates(id_col, keep="first")
    )
    best_out = (
        gateway_lookup.dropna(subset=["rank_out"])
        .sort_values([id_col, "rank_out"])
        .drop_duplicates(id_col, keep="first")
    )
    unit_map_in = {str(r[id_col]).strip(): r for _, r in best_in.iterrows()}
    unit_map_out = {str(r[id_col]).strip(): r for _, r in best_out.iterrows()}
    pair_map = {(str(r["gateway_in"]), str(r["gateway_out"])): float(r["internal_cost_s"]) for _, r in gateway_pair_costs.iterrows()}
    direct_cost_cache: Dict[Tuple[int, int], float] = {}
    direct_cost_source_cache: Dict[int, Dict[int, float]] = {}
    direct_cache_hits = 0
    direct_cache_misses = 0
    source_cache_hits = 0
    source_cache_misses = 0

    rows: List[Dict[str, Any]] = []
    through_rows: List[Dict[str, Any]] = []
    for _, r in commuting_pairs.iterrows():
        o_norm = str(r["origin_place_norm"])
        d_norm = str(r["dest_place_norm"])
        o_key = str(r["origin_place_key"])
        d_key = str(r["dest_place_key"])
        o_uid = str(r.get("origin_unit_id", o_key)).strip()
        d_uid = str(r.get("dest_unit_id", d_key)).strip()
        o_internal = o_norm in internal_zone_names
        d_internal = d_norm in internal_zone_names
        vehicles_daily = persons_to_vehicles_from_cfg(float(r["persons_work"]), float(r["persons_school"]), cfg_root)

        rec: Dict[str, Any] = {
            "origin_place": r["origin_place"],
            "dest_place": r["dest_place"],
            "origin_place_key": o_key,
            "dest_place_key": d_key,
            "origin_unit_id": o_uid,
            "dest_unit_id": d_uid,
            "vehicles_daily": vehicles_daily,
            "classification": None,
            "gateway_in": None,
            "gateway_out": None,
            "direct_cost_s": None,
            "via_model_cost_s": None,
            "detour_ratio": None,
            "extra_minutes": None,
            "rejection_reason": None,
            "accepted": False,
        }

        if o_internal and d_internal:
            rec["classification"] = "internal_internal"
            rows.append(rec)
            continue
        if (not o_internal) and d_internal:
            rec["classification"] = "external_internal"
            if o_uid in unit_map_in:
                rec["gateway_in"] = str(unit_map_in[o_uid]["gateway_name"])
                rec["accepted"] = True
            else:
                rec["rejection_reason"] = "missing_origin_gateway_in"
            rows.append(rec)
            continue
        if o_internal and (not d_internal):
            rec["classification"] = "internal_external"
            if d_uid in unit_map_out:
                rec["gateway_out"] = str(unit_map_out[d_uid]["gateway_name"])
                rec["accepted"] = True
            else:
                rec["rejection_reason"] = "missing_dest_gateway_out"
            rows.append(rec)
            continue

        rec["classification"] = "external_external"
        o_info = unit_map_in.get(o_uid)
        d_info = unit_map_out.get(d_uid)
        if o_info is None or d_info is None:
            rec["rejection_reason"] = "missing_external_gateway"
            rows.append(rec)
            continue

        gw_in = str(o_info["gateway_name"])
        gw_out = str(d_info["gateway_name"])
        rec["gateway_in"] = gw_in
        rec["gateway_out"] = gw_out
        if gw_in == gw_out and not cfg.allow_same_gateway_pair:
            rec["rejection_reason"] = "same_gateway_not_allowed"
            rows.append(rec)
            continue

        if cfg.reject_same_boundary_sector and gateway_boundary_angles:
            ang_in = gateway_boundary_angles.get(gw_in)
            ang_out = gateway_boundary_angles.get(gw_out)
            if ang_in is not None and ang_out is not None:
                if _boundary_sector(ang_in) == _boundary_sector(ang_out):
                    rec["rejection_reason"] = "same_boundary_sector"
                    rows.append(rec)
                    continue

        pair_cost = pair_map.get((gw_in, gw_out), float("nan"))
        o_node = int(o_info["unit_graph_node"])
        d_node = int(d_info["unit_graph_node"])
        direct_cost, pair_hit, source_hit = shortest_path_cost_batched(
            G,
            o_node,
            d_node,
            direct_cost_cache,
            direct_cost_source_cache,
        )
        if pair_hit:
            direct_cache_hits += 1
        else:
            direct_cache_misses += 1
        if source_hit:
            source_cache_hits += 1
        else:
            source_cache_misses += 1
        o_to_gateway = float(o_info["route_cost_to_gateway_s"]) if np.isfinite(o_info.get("route_cost_to_gateway_s", np.nan)) else float("nan")
        gateway_to_d = float(d_info["route_cost_from_gateway_s"]) if np.isfinite(d_info.get("route_cost_from_gateway_s", np.nan)) else float("nan")
        via_cost = (
            o_to_gateway + float(pair_cost) + gateway_to_d
            if np.isfinite(pair_cost) and np.isfinite(o_to_gateway) and np.isfinite(gateway_to_d)
            else float("nan")
        )

        rec["direct_cost_s"] = direct_cost if np.isfinite(direct_cost) else None
        rec["via_model_cost_s"] = via_cost if np.isfinite(via_cost) else None
        if np.isfinite(direct_cost) and np.isfinite(via_cost) and direct_cost > 0:
            rec["detour_ratio"] = via_cost / direct_cost
            extra_minutes = (via_cost - direct_cost) / 60.0
            rec["extra_minutes"] = extra_minutes
            rec["accepted"] = (
                (via_cost / direct_cost) <= cfg.detour_ratio_max
                and extra_minutes <= cfg.max_extra_minutes
            )
            if not rec["accepted"]:
                if (via_cost / direct_cost) > cfg.detour_ratio_max:
                    rec["rejection_reason"] = "detour_ratio_exceeded"
                else:
                    rec["rejection_reason"] = "extra_minutes_exceeded"
        else:
            rec["rejection_reason"] = "missing_costs"

        if rec["accepted"]:
            through_rows.append({
                "gateway_in": gw_in,
                "gateway_out": gw_out,
                "vehicles_daily": vehicles_daily,
                "relations": 1,
            })
        rows.append(rec)

    classified = pd.DataFrame(rows)
    if through_rows:
        through_pairs = pd.DataFrame(through_rows)
        through_pairs = through_pairs.groupby(["gateway_in", "gateway_out"], as_index=False)[
            ["vehicles_daily", "relations"]
        ].sum()
    else:
        through_pairs = pd.DataFrame(columns=["gateway_in", "gateway_out", "vehicles_daily"])
    classified.attrs["profile"] = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "direct_shortest_path_cache_hits": int(direct_cache_hits),
        "direct_shortest_path_cache_misses": int(direct_cache_misses),
        "direct_shortest_path_unique_pairs": int(len(direct_cost_cache)),
        "direct_shortest_path_source_cache_hits": int(source_cache_hits),
        "direct_shortest_path_source_cache_misses": int(source_cache_misses),
        "direct_shortest_path_unique_sources": int(len(direct_cost_source_cache)),
    }
    return classified, through_pairs
