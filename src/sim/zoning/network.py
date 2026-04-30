"""Network access helpers and road-class constants for the zoning pipeline."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd
from aequilibrae import Project
from shapely.geometry import Point

from sim.network.connectivity import _build_digraph
from sim.network.crs import as_gdf, network_links_gdf_with_crs
from sim.network.filtering import DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES

from sim.zoning.geo import force_to_target_crs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Road-class constants
# ---------------------------------------------------------------------------

_EXCLUDED_LINK_TYPES = DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES | frozenset({
    "construction", "elevator", "service", "rest_area",
    "services", "traffic_mirror", "virtual",
})

ROAD_CLASS_WEIGHT: Dict[str, float] = {
    "motorway": 5.0,
    "motorway_link": 4.0,
    "trunk": 4.0,
    "trunk_link": 3.5,
    "primary": 3.0,
    "primary_link": 2.5,
    "secondary": 2.0,
    "secondary_link": 1.8,
    "tertiary": 1.5,
    "tertiary_link": 1.2,
    "unclassified": 0.8,
    "road": 0.7,
    "residential": 0.6,
    "service": 0.2,
    "living_street": 0.05,
}

MERGE_CLASS_GROUPS: Dict[str, str] = {
    "motorway": "A",
    "motorway_link": "A",
    "trunk": "A",
    "trunk_link": "A",
    "primary": "B",
    "primary_link": "B",
    "secondary": "C",
    "secondary_link": "C",
    "tertiary": "C",
    "tertiary_link": "C",
}

MAJOR_ROAD_WEIGHT_THRESHOLD: float = ROAD_CLASS_WEIGHT.get("secondary", 2.0)


def merge_class_group(link_type: str) -> str:
    """Map a link type to its merge-class group letter."""
    return MERGE_CLASS_GROUPS.get(str(link_type), "C")


def pick_best_road_type(a: str, b: str) -> str:
    """Return the higher-class road type between two candidates."""
    if not a:
        return b
    if not b:
        return a
    order = [
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "unclassified", "residential",
        "living_street", "service",
    ]
    ra = order.index(a) if a in order else len(order)
    rb = order.index(b) if b in order else len(order)
    return a if ra <= rb else b


# ---------------------------------------------------------------------------
# CRS-safe network reference
# ---------------------------------------------------------------------------

def network_ref(project: Project, kind: str, target_epsg: int) -> gpd.GeoDataFrame:
    """Return project nodes or links as a GeoDataFrame in *target_epsg*."""
    kind = kind.lower().strip()

    if kind == "nodes":
        df = project.network.nodes.data
        if "geometry" not in df.columns or len(df) == 0:
            raise RuntimeError("Network nodes missing geometry/empty")
        g = as_gdf(df[["node_id", "geometry"]].copy(), crs_hint_epsg=target_epsg)
        g = force_to_target_crs(g, target_epsg, name="network.nodes")
        return g

    if kind == "links":
        try:
            g = network_links_gdf_with_crs(project, crs_epsg_hint=target_epsg)
        except RuntimeError:
            raise
        if "link_type" in g.columns:
            g = g[g["link_type"].astype(str) != "centroid_connector"].copy()
        g = force_to_target_crs(g, target_epsg, name="network.links")
        return g

    raise ValueError("kind must be 'nodes' or 'links'")


# ---------------------------------------------------------------------------
# Eligible road nodes (graph analysis)
# ---------------------------------------------------------------------------

def eligible_road_nodes(
    project: Project,
    *,
    use_directed_scc: bool = True,
) -> Tuple[set[int], Dict[int, float]]:
    """Compute the set of car-reachable network nodes and per-node road-class weights.

    When *use_directed_scc* is True, only nodes in the largest strongly
    connected component (directed) are kept.
    """
    links = project.network.links.data.copy()
    if links.empty:
        return set(), {}

    excluded = set(_EXCLUDED_LINK_TYPES) | {"centroid_connector"}

    car = links[
        links["modes"].astype(str).str.contains("c", na=False)
        & ~links["link_type"].astype(str).isin(excluded)
    ].copy()

    if car.empty:
        return set(), {}

    G_undir = nx.Graph()
    for _, r in car.iterrows():
        G_undir.add_edge(int(r["a_node"]), int(r["b_node"]))

    if G_undir.number_of_nodes() == 0:
        return set(), {}

    largest_undirected = max(nx.connected_components(G_undir), key=len)
    keep_nodes = set(int(x) for x in largest_undirected)

    if use_directed_scc:
        G_dir = _build_digraph(car)
        if G_dir.number_of_nodes() > 0:
            largest_scc = max(nx.strongly_connected_components(G_dir), key=len)
            scc_nodes = set(int(x) for x in largest_scc)
            n_removed = len(keep_nodes - scc_nodes)
            if n_removed > 0:
                logger.debug(
                    "Directed SCC filter: removed %d nodes reachable only one-way "
                    "(SCC=%d, undirected=%d)",
                    n_removed,
                    len(scc_nodes),
                    len(keep_nodes),
                )
            keep_nodes = keep_nodes & scc_nodes

    car = car[
        car["a_node"].astype(int).isin(keep_nodes)
        & car["b_node"].astype(int).isin(keep_nodes)
    ].copy()

    node_weight: Dict[int, float] = {}
    for _, r in car.iterrows():
        lt = str(r.get("link_type", "")).strip()
        w = float(ROAD_CLASS_WEIGHT.get(lt, 0.5))
        a = int(r["a_node"])
        b = int(r["b_node"])
        node_weight[a] = max(node_weight.get(a, 0.0), w)
        node_weight[b] = max(node_weight.get(b, 0.0), w)

    return set(node_weight.keys()), node_weight


# ---------------------------------------------------------------------------
# Boundary-node selection
# ---------------------------------------------------------------------------

def pick_nodes_near_boundary(
    node_ids: List[int],
    node_geom: Dict[int, Point],
    boundary: Any,
    node_weight: Dict[int, float],
    *,
    max_nodes: int,
    min_node_sep_m: float = 25.0,
    scc_nodes: Optional[set] = None,
) -> List[int]:
    """Select up to *max_nodes* road-network nodes nearest to *boundary*."""
    rows = []
    for nid in node_ids:
        geom = node_geom.get(int(nid))
        if geom is None:
            continue
        in_scc = 0 if (scc_nodes is not None and int(nid) in scc_nodes) else 1
        rows.append({
            "node_id": int(nid),
            "in_scc_sort": in_scc,
            "dist_boundary": float(geom.distance(boundary)),
            "road_weight": float(node_weight.get(int(nid), 0.0)),
            "geometry": geom,
        })

    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values(
        ["in_scc_sort", "dist_boundary", "road_weight"],
        ascending=[True, True, False],
    )

    chosen: List[int] = []
    for _, row in df.iterrows():
        nid = int(row["node_id"])
        g = node_geom[nid]
        too_close = False
        for kept in chosen:
            if float(g.distance(node_geom[int(kept)])) < float(min_node_sep_m):
                too_close = True
                break
        if not too_close:
            chosen.append(nid)
        if len(chosen) >= int(max_nodes):
            break

    if not chosen and not df.empty:
        chosen = [int(df.iloc[0]["node_id"])]

    return chosen
