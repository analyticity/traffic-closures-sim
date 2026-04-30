"""NetworkX digraph: build from OSM PBF roads, contract, snap, serialize."""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from sim.datasets.utils import ensure_dir
from sim.supernetwork.config import SuperCfg

logger = logging.getLogger(__name__)

try:
    import pyogrio
except Exception:  # pragma: no cover
    pyogrio = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# --- Speed / travel-time helpers ---

def _parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not m:
        return None
    out = float(m.group(0).replace(",", "."))
    if "mph" in text:
        out *= 1.60934
    return out


def _highway_default_speed_kmh(highway: Any) -> float:
    highway = str(highway or "")
    defaults = {
        "motorway": 130.0,
        "motorway_link": 80.0,
        "trunk": 90.0,
        "trunk_link": 70.0,
        "primary": 70.0,
        "primary_link": 50.0,
        "secondary": 55.0,
        "secondary_link": 45.0,
        "tertiary": 50.0,
        "tertiary_link": 40.0,
    }
    return defaults.get(highway, 60.0)


def _edge_speed_kmh(maxspeed: Any, highway: Any) -> float:
    parsed = _parse_numeric(maxspeed)
    return parsed if parsed and parsed > 0 else _highway_default_speed_kmh(highway)


def _meters_to_seconds(distance_m: float, speed_kmh: float) -> float:
    return float(distance_m) / max(float(speed_kmh) / 3.6, 0.1)


def _iter_lines(geom: Any) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []


# --- OSM extraction ---

def read_osm_lines_from_pbf(pbf_path: Path, highway_types: List[str]) -> gpd.GeoDataFrame:
    wanted = sorted(set(str(x).strip() for x in highway_types if str(x).strip()))
    where = " OR ".join([f"highway = '{h}'" for h in wanted])
    cols = ["osm_id", "name", "ref", "highway", "oneway", "maxspeed", "geometry"]

    if pyogrio is not None:
        try:
            gdf = pyogrio.read_dataframe(pbf_path, layer="lines", columns=cols, where=where)
            if gdf is not None and not gdf.empty:
                return gdf
        except Exception:
            pass

    gdf = gpd.read_file(pbf_path, layer="lines")
    return gdf[gdf["highway"].astype(str).isin(wanted)][[c for c in cols if c in gdf.columns]].copy()


def load_or_extract_major_roads(cfg: SuperCfg) -> gpd.GeoDataFrame:
    ensure_dir(cfg.raw_major_roads_cache.parent)
    if cfg.raw_major_roads_cache.exists():
        roads = gpd.read_parquet(cfg.raw_major_roads_cache)
        if roads.crs is None:
            roads = roads.set_crs(epsg=cfg.metric_epsg, allow_override=True)
        return roads

    roads = read_osm_lines_from_pbf(cfg.pbf_path, cfg.highway_types)
    if roads.crs is None:
        roads = roads.set_crs(epsg=4326, allow_override=True)
    roads = roads.to_crs(epsg=cfg.metric_epsg)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads.to_parquet(cfg.raw_major_roads_cache, index=False)
    return roads


# --- Graph construction ---

def _coord_key(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def add_or_relax_edge(G: nx.DiGraph, u: int, v: int, length_m: float, travel_time_s: float, highway: str, name: str, ref: str) -> None:
    if u == v:
        return
    if G.has_edge(u, v):
        if travel_time_s < float(G[u][v].get("travel_time_s", math.inf)):
            G[u][v].update(length_m=float(length_m), travel_time_s=float(travel_time_s), highway=highway, name=name, ref=ref)
    else:
        G.add_edge(u, v, length_m=float(length_m), travel_time_s=float(travel_time_s), highway=highway, name=name, ref=ref)


def build_graph_from_roads(roads: gpd.GeoDataFrame) -> Tuple[nx.DiGraph, Dict[str, int]]:
    G = nx.DiGraph()
    coord_to_node: Dict[Tuple[float, float], int] = {}
    next_id = 1
    segment_count = 0
    bidir_segments = 0

    def node_id(x: float, y: float) -> int:
        nonlocal next_id
        key = _coord_key(x, y)
        if key not in coord_to_node:
            coord_to_node[key] = next_id
            G.add_node(next_id, x=float(key[0]), y=float(key[1]))
            next_id += 1
        return coord_to_node[key]

    for row in roads.itertuples(index=False):
        geom = getattr(row, "geometry", None)
        highway = str(getattr(row, "highway", "") or "")
        name = str(getattr(row, "name", "") or "")
        ref = str(getattr(row, "ref", "") or "")
        speed_kmh = _edge_speed_kmh(getattr(row, "maxspeed", None), highway)
        oneway = str(getattr(row, "oneway", "") or "").strip().lower()
        reverse_only = oneway == "-1"
        forward_only = oneway in {"yes", "1", "true", "t"} or reverse_only

        for line in _iter_lines(geom):
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                u = node_id(a[0], a[1])
                v = node_id(b[0], b[1])
                seg_len = float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))
                tt = _meters_to_seconds(seg_len, speed_kmh)
                segment_count += 1
                if reverse_only:
                    add_or_relax_edge(G, v, u, seg_len, tt, highway, name, ref)
                elif forward_only:
                    add_or_relax_edge(G, u, v, seg_len, tt, highway, name, ref)
                else:
                    bidir_segments += 1
                    add_or_relax_edge(G, u, v, seg_len, tt, highway, name, ref)
                    add_or_relax_edge(G, v, u, seg_len, tt, highway, name, ref)

    return G, {
        "roads_rows": int(len(roads)),
        "segments_total": int(segment_count),
        "segments_bidirectional": int(bidir_segments),
    }


# --- Snapping ---

def snap_points_to_graph(G: nx.DiGraph, points: gpd.GeoDataFrame, label_col: str) -> pd.DataFrame:
    node_ids = np.array(list(G.nodes()), dtype=np.int64)
    xs = np.array([G.nodes[n]["x"] for n in node_ids], dtype=float)
    ys = np.array([G.nodes[n]["y"] for n in node_ids], dtype=float)
    pts = np.column_stack([points.geometry.x.to_numpy(dtype=float), points.geometry.y.to_numpy(dtype=float)])

    if cKDTree is not None:
        tree = cKDTree(np.column_stack([xs, ys]))
        dists, idxs = tree.query(pts, k=1)
    else:
        idxs, dists = [], []
        for px, py in pts:
            dist2 = (xs - px) ** 2 + (ys - py) ** 2
            idx = int(np.argmin(dist2))
            idxs.append(idx)
            dists.append(float(np.sqrt(dist2[idx])))
        idxs = np.array(idxs, dtype=int)
        dists = np.array(dists, dtype=float)

    rows = []
    for i, (_, row) in enumerate(points.iterrows()):
        nid = int(node_ids[int(idxs[i])])
        rows.append({
            label_col: row[label_col],
            "graph_node": nid,
            "graph_x": float(G.nodes[nid]["x"]),
            "graph_y": float(G.nodes[nid]["y"]),
            "snap_distance_m": float(dists[i]),
        })
    return pd.DataFrame(rows)


# --- Contraction ---

def is_contractible(G: nx.DiGraph, node: int, protected: set[int], degree_threshold: int) -> bool:
    if node in protected or not G.has_node(node):
        return False
    neighbors = (set(G.predecessors(node)) | set(G.successors(node))) - {node}
    return len(neighbors) <= degree_threshold


def contract_graph(G_in: nx.DiGraph, protected: set[int], degree_threshold: int) -> nx.DiGraph:
    G = G_in.copy()
    queue = [n for n in list(G.nodes()) if is_contractible(G, int(n), protected, degree_threshold)]

    while queue:
        n = int(queue.pop())
        if not is_contractible(G, n, protected, degree_threshold):
            continue
        nbrs = list((set(G.predecessors(n)) | set(G.successors(n))) - {n})
        if len(nbrs) != 2:
            continue
        a, b = int(nbrs[0]), int(nbrs[1])

        if G.has_edge(a, n) and G.has_edge(n, b):
            e1, e2 = G[a][n], G[n][b]
            add_or_relax_edge(G, a, b, float(e1["length_m"]) + float(e2["length_m"]), float(e1["travel_time_s"]) + float(e2["travel_time_s"]), str(e1.get("highway", e2.get("highway", ""))), str(e1.get("name", e2.get("name", ""))), str(e1.get("ref", e2.get("ref", ""))))
        if G.has_edge(b, n) and G.has_edge(n, a):
            e1, e2 = G[b][n], G[n][a]
            add_or_relax_edge(G, b, a, float(e1["length_m"]) + float(e2["length_m"]), float(e1["travel_time_s"]) + float(e2["travel_time_s"]), str(e1.get("highway", e2.get("highway", ""))), str(e1.get("name", e2.get("name", ""))), str(e1.get("ref", e2.get("ref", ""))))
        G.remove_node(n)
        for m in (a, b):
            if is_contractible(G, m, protected, degree_threshold):
                queue.append(m)

    return G


# --- Serialization ---

def graph_to_gdfs(G: nx.DiGraph, metric_epsg: int) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = gpd.GeoDataFrame(
        [{"node_id": int(n), "x": float(d["x"]), "y": float(d["y"]), "geometry": Point(float(d["x"]), float(d["y"]))} for n, d in G.nodes(data=True)],
        geometry="geometry",
        crs=f"EPSG:{metric_epsg}",
    )
    edges = gpd.GeoDataFrame(
        [{
            "source": int(u),
            "target": int(v),
            "length_m": float(d["length_m"]),
            "travel_time_s": float(d["travel_time_s"]),
            "highway": str(d.get("highway", "")),
            "name": str(d.get("name", "")),
            "ref": str(d.get("ref", "")),
            "geometry": LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]),
        } for u, v, d in G.edges(data=True)],
        geometry="geometry",
        crs=f"EPSG:{metric_epsg}",
    )
    return nodes, edges


def graph_from_parquets(nodes_path: Path, edges_path: Path) -> nx.DiGraph:
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(int(r["node_id"]), x=float(r["x"]), y=float(r["y"]))
    for _, r in edges.iterrows():
        add_or_relax_edge(G, int(r["source"]), int(r["target"]), float(r["length_m"]), float(r["travel_time_s"]), str(r.get("highway", "")), str(r.get("name", "")), str(r.get("ref", "")))
    return G
