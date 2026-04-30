"""Shortest-path routing: SSSP costs, gateway cost matrices, gateway lookup tables."""
from __future__ import annotations

import heapq
import time
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from sim.supernetwork.config import SuperCfg


def single_source_costs(G: nx.DiGraph, source_node: int) -> Dict[int, float]:
    return nx.single_source_dijkstra_path_length(G, source=source_node, weight="travel_time_s")


def build_gateway_costs(
    G: nx.DiGraph,
    gateways: gpd.GeoDataFrame,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], pd.DataFrame]:
    t0 = time.perf_counter()
    gateway_nodes = {str(r["gateway_name"]): int(r["graph_node"]) for _, r in gateways.iterrows()}
    costs_from_gateway = {name: single_source_costs(G, node) for name, node in gateway_nodes.items()}
    reverse = G.reverse(copy=False)
    costs_to_gateway = {name: single_source_costs(reverse, node) for name, node in gateway_nodes.items()}
    rows = []
    for a, a_node in gateway_nodes.items():
        for b, b_node in gateway_nodes.items():
            if a == b:
                continue
            cost = costs_from_gateway[a].get(b_node)
            rows.append({"gateway_in": a, "gateway_out": b, "internal_cost_s": float(cost) if cost is not None else np.nan})
    if profile is not None:
        profile["gateway_costs"] = {
            "gateway_count": int(len(gateway_nodes)),
            "sssp_runs": int(2 * len(gateway_nodes)),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    return costs_from_gateway, costs_to_gateway, pd.DataFrame(rows)


def build_gateway_lookup(
    units: gpd.GeoDataFrame,
    costs_from_gateway: Dict[str, Dict[int, float]],
    costs_to_gateway: Dict[str, Dict[int, float]],
    cfg: SuperCfg,
    gateway_snap_distances: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    _SNAP_PENALTY_SPEED_MPS = 50.0 * 1000.0 / 3600.0
    snap_penalties: Dict[str, float] = {}
    if gateway_snap_distances:
        for gw, dist in gateway_snap_distances.items():
            snap_penalties[gw] = dist / _SNAP_PENALTY_SPEED_MPS

    columns = [
        "unit_id",
        "place_key",
        "place_name",
        "district_name",
        "admin_level",
        "unit_graph_node",
        "gateway_name",
        "rank_in",
        "rank_out",
        "route_cost_to_gateway_s",
        "route_cost_from_gateway_s",
        "route_cost_s",
    ]
    rows = []
    gateway_names = tuple(sorted(set(costs_from_gateway.keys()) | set(costs_to_gateway.keys())))
    for _, unit in units.iterrows():
        scored_in = []
        scored_out = []
        unit_node = int(unit["graph_node"])
        for gw_name in gateway_names:
            to_cost = costs_to_gateway.get(gw_name, {}).get(unit_node)
            from_cost = costs_from_gateway.get(gw_name, {}).get(unit_node)
            penalty = snap_penalties.get(gw_name, 0.0)
            if to_cost is not None and np.isfinite(to_cost):
                scored_in.append((gw_name, float(to_cost) + penalty))
            if from_cost is not None and np.isfinite(from_cost):
                scored_out.append((gw_name, float(from_cost) + penalty))

        if not scored_in and not scored_out:
            continue

        in_rank = {
            name: rank for rank, (name, _) in enumerate(
                heapq.nsmallest(cfg.max_candidate_gateways, scored_in, key=lambda x: x[1]),
                start=1,
            )
        }
        out_rank = {
            name: rank for rank, (name, _) in enumerate(
                heapq.nsmallest(cfg.max_candidate_gateways, scored_out, key=lambda x: x[1]),
                start=1,
            )
        }

        selected = sorted(set(in_rank.keys()) | set(out_rank.keys()))
        for gw_name in selected:
            to_cost = costs_to_gateway.get(gw_name, {}).get(unit_node)
            from_cost = costs_from_gateway.get(gw_name, {}).get(unit_node)
            preferred = from_cost if from_cost is not None and np.isfinite(from_cost) else to_cost
            rows.append({
                "unit_id": str(unit["unit_id"]).strip(),
                "place_key": unit["place_key"],
                "place_name": unit["place_name"],
                "district_name": unit.get("district_name", ""),
                "admin_level": unit.get("admin_level", ""),
                "unit_graph_node": unit_node,
                "gateway_name": gw_name,
                "rank_in": in_rank.get(gw_name),
                "rank_out": out_rank.get(gw_name),
                "route_cost_to_gateway_s": float(to_cost) if to_cost is not None and np.isfinite(to_cost) else np.nan,
                "route_cost_from_gateway_s": float(from_cost) if from_cost is not None and np.isfinite(from_cost) else np.nan,
                "route_cost_s": float(preferred) if preferred is not None and np.isfinite(preferred) else np.nan,
            })
    out = pd.DataFrame(rows, columns=columns)
    out.attrs["profile"] = {
        "units_total": int(len(units)),
        "gateway_count": int(len(gateway_names)),
        "candidate_evaluations": int(len(units) * len(gateway_names)),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    return out


def shortest_path_cost_batched(
    G: nx.DiGraph,
    source: int,
    target: int,
    pair_cache: Dict[Tuple[int, int], float],
    source_cache: Dict[int, Dict[int, float]],
) -> Tuple[float, bool, bool]:
    """Return shortest path cost with source-level SSSP cache.

    Returns tuple: (value, pair_cache_hit, source_cache_hit).
    """
    key = (int(source), int(target))
    if key in pair_cache:
        return pair_cache[key], True, True
    src = int(source)
    if src in source_cache:
        dist_map = source_cache[src]
        source_hit = True
    else:
        dist_map = nx.single_source_dijkstra_path_length(G, source=src, weight="travel_time_s")
        source_cache[src] = dist_map
        source_hit = False
    value = float(dist_map.get(int(target), float("nan")))
    pair_cache[key] = value
    return value, False, source_hit
