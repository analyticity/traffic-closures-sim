"""
Transport analysis zoning: model AOI from the road network, load zones from modular sources
(see ``sim.zoning_sources`` — OSM or vector files), optional boundary gateways, centroid connectors,
and map/diagnostics exports.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae import Project
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import nearest_points

from sim.io_project import load_config
from sim.zoning_sources import load_zones_from_sources
from sim.zoning_sources.shared import remove_overlaps_by_priority


# ----------------------------
# Helpers
# ----------------------------

def _safe_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _load_cfg(config_or_path: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config_or_path, dict):
        return config_or_path
    return load_config(config_or_path)


def _to_wgs84_point(point, crs_from) -> Any:
    return gpd.GeoSeries([point], crs=crs_from).to_crs("EPSG:4326").iloc[0]


def _guess_crs_from_coords(geoms: gpd.GeoSeries, target_epsg: int) -> str:
    s = geoms.dropna()
    if s.empty:
        return f"EPSG:{target_epsg}"

    s = s.iloc[:200]
    xs: List[float] = []
    ys: List[float] = []

    for g in s:
        try:
            c = g.centroid
            xs.append(float(c.x))
            ys.append(float(c.y))
        except Exception:
            continue

    if not xs or not ys:
        return f"EPSG:{target_epsg}"

    max_abs_x = max(abs(x) for x in xs)
    max_abs_y = max(abs(y) for y in ys)

    if max_abs_x <= 180.0 and max_abs_y <= 90.0:
        return "EPSG:4326"
    return f"EPSG:{target_epsg}"


def _circular_mean_ring_pos(values: List[float], ring_length: float) -> float:
    if not values:
        return 0.0
    if ring_length <= 0:
        return float(np.mean(values))

    ang = np.array(values, dtype=float) / float(ring_length) * (2.0 * np.pi)
    s = np.sin(ang).mean()
    c = np.cos(ang).mean()
    mean_ang = math.atan2(s, c)
    if mean_ang < 0:
        mean_ang += 2.0 * np.pi
    return float(mean_ang / (2.0 * np.pi) * float(ring_length))


def _pick_nodes_near_boundary(
    node_ids: List[int],
    node_geom: Dict[int, Point],
    boundary: Any,
    node_weight: Dict[int, float],
    *,
    max_nodes: int,
    min_node_sep_m: float = 25.0,
) -> List[int]:
    rows = []
    for nid in node_ids:
        geom = node_geom.get(int(nid))
        if geom is None:
            continue
        rows.append({
            "node_id": int(nid),
            "dist_boundary": float(geom.distance(boundary)),
            "road_weight": float(node_weight.get(int(nid), 0.0)),
            "geometry": geom,
        })

    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values(
        ["dist_boundary", "road_weight"],
        ascending=[True, False],
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


def _merge_gateway_candidates_on_boundary(
    gateway_targets: Dict[str, List[int]],
    gateway_meta: Dict[str, Dict[str, Any]],
    node_geom: Dict[int, Point],
    node_weight: Dict[int, float],
    model_area: Any,
    *,
    merge_distance_m: float,
    nodes_per_gateway: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, Any]]]:
    if len(gateway_meta) <= 1:
        return gateway_targets, gateway_meta

    boundary = model_area.boundary
    ring_length = float(boundary.length)

    df = pd.DataFrame(
        [
            {
                "gateway_name": gw_name,
                "boundary_pos": float(meta.get("boundary_pos", 0.0)),
            }
            for gw_name, meta in gateway_meta.items()
        ]
    )

    clusters = _cluster_positions_on_ring(
        df=df,
        pos_col="boundary_pos",
        threshold_m=float(merge_distance_m),
        ring_length=ring_length,
    )

    merged_targets: Dict[str, List[int]] = {}
    merged_meta: Dict[str, Dict[str, Any]] = {}

    for cl in clusters:
        names = cl["gateway_name"].tolist()
        metas = [gateway_meta[n] for n in names]

        rep = sorted(
            metas,
            key=lambda m: (
                int(m.get("whitelist_priority", 9999)),
                float(m.get("dist_boundary_m", 1e9)),
                -float(_ROAD_CLASS_WEIGHT.get(str(m.get("link_type", "")), 0.0)),
            ),
        )[0]

        cluster_boundary_pos = _circular_mean_ring_pos(
            [float(m.get("boundary_pos", 0.0)) for m in metas],
            ring_length,
        )
        cluster_boundary_pt = boundary.interpolate(cluster_boundary_pos)

        union_nodes: List[int] = []
        for m in metas:
            for nid in m.get("target_node_ids", []):
                nid = int(nid)
                if nid in node_geom and nid not in union_nodes:
                    union_nodes.append(nid)

        chosen = _pick_nodes_near_boundary(
            union_nodes,
            node_geom,
            boundary,
            node_weight,
            max_nodes=int(nodes_per_gateway),
            min_node_sep_m=25.0,
        )

        if not chosen:
            anchor_id = int(rep["anchor_node_id"])
            chosen = [anchor_id] if anchor_id in node_geom else []

        if not chosen:
            continue

        anchor_id = int(chosen[0])
        anchor_geom = node_geom[anchor_id]

        new_meta = dict(rep)
        new_meta["target_node_ids"] = chosen
        new_meta["anchor_node_id"] = int(anchor_id)
        new_meta["anchor_x"] = float(anchor_geom.x)
        new_meta["anchor_y"] = float(anchor_geom.y)
        new_meta["boundary_x"] = float(cluster_boundary_pt.x)
        new_meta["boundary_y"] = float(cluster_boundary_pt.y)
        new_meta["boundary_pos"] = float(cluster_boundary_pos)
        new_meta["merged_from"] = "|".join(names)

        merged_targets[new_meta["gateway_name"]] = chosen
        merged_meta[new_meta["gateway_name"]] = new_meta

        if len(names) > 1:
            print(f"  merged boundary-near gateways {names} -> {new_meta['gateway_name']}")

    return merged_targets, merged_meta


def _looks_like_degrees(geoms: gpd.GeoSeries) -> bool:
    s = geoms.dropna()
    if s.empty:
        return False
    s = s.iloc[:300]

    xs: List[float] = []
    ys: List[float] = []

    for g in s:
        try:
            c = g.centroid
            xs.append(float(c.x))
            ys.append(float(c.y))
        except Exception:
            continue

    if not xs or not ys:
        return False

    max_abs_x = max(abs(x) for x in xs)
    max_abs_y = max(abs(y) for y in ys)

    return (max_abs_x <= 180.0) and (max_abs_y <= 90.0)


def _force_to_target_crs(
    gdf: gpd.GeoDataFrame,
    target_epsg: int,
    *,
    name: str = "gdf",
) -> gpd.GeoDataFrame:
    if gdf.empty:
        if gdf.crs is None:
            return gdf.set_crs(epsg=target_epsg, allow_override=True)
        if gdf.crs.to_epsg() != target_epsg:
            return gdf.to_crs(epsg=target_epsg)
        return gdf

    if gdf.crs is None:
        guessed = _guess_crs_from_coords(gdf.geometry, target_epsg=target_epsg)
        gdf = gdf.set_crs(guessed, allow_override=True)

    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None

    if epsg == target_epsg and _looks_like_degrees(gdf.geometry):
        print(
            f"⚠ CRS sanity-fix: {name} looks like degrees but CRS={target_epsg}. "
            "Treating as EPSG:4326 then reprojecting."
        )
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(epsg=target_epsg)
        return gdf

    if epsg != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    return gdf


def _oriented_square_from_points(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    quantile: float = 0.01,
    pad_ratio: float = 0.005,
    make_square: bool = True,
) -> Polygon:
    if xs.size < 3:
        return box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    q = float(quantile)
    q = max(0.0, min(q, 0.49))

    x_lo, x_hi = np.quantile(xs, [q, 1.0 - q])
    y_lo, y_hi = np.quantile(ys, [q, 1.0 - q])
    mask = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)

    pts = np.column_stack([xs, ys])
    pts_core = pts[mask] if mask.sum() >= 50 else pts

    hull = MultiPoint(pts_core).convex_hull
    mrr = hull.minimum_rotated_rectangle

    coords = list(mrr.exterior.coords)
    best_dx, best_dy, best_len2 = 1.0, 0.0, -1.0
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        l2 = dx * dx + dy * dy
        if l2 > best_len2:
            best_len2 = l2
            best_dx, best_dy = dx, dy

    angle = float(np.arctan2(best_dy, best_dx))

    cx, cy = pts_core.mean(axis=0)
    c = float(np.cos(-angle))
    s = float(np.sin(-angle))

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    u = c * dx - s * dy
    v = s * dx + c * dy

    u_lo, u_hi = np.quantile(u, [q, 1.0 - q])
    v_lo, v_hi = np.quantile(v, [q, 1.0 - q])

    w = float(u_hi - u_lo)
    h = float(v_hi - v_lo)
    px = max(w * float(pad_ratio), 0.0)
    py = max(h * float(pad_ratio), 0.0)

    u_lo -= px
    u_hi += px
    v_lo -= py
    v_hi += py

    if make_square:
        side = max(float(u_hi - u_lo), float(v_hi - v_lo))
        cu = 0.5 * (u_lo + u_hi)
        cv = 0.5 * (v_lo + v_hi)
        half = 0.5 * side
        u_lo, u_hi = cu - half, cu + half
        v_lo, v_hi = cv - half, cv + half

    corners_uv = np.array([
        [u_lo, v_lo],
        [u_hi, v_lo],
        [u_hi, v_hi],
        [u_lo, v_hi],
    ])

    c2 = float(np.cos(angle))
    s2 = float(np.sin(angle))
    corners_xy = []
    for uu, vv in corners_uv:
        x = cx + (c2 * uu - s2 * vv)
        y = cy + (s2 * uu + c2 * vv)
        corners_xy.append((x, y))

    return Polygon(corners_xy)


def _safe_line_midpoint(geom: Any) -> Point:
    try:
        if geom is None or geom.is_empty:
            return Point(0, 0)
        if geom.geom_type == "LineString":
            return geom.interpolate(0.5, normalized=True)
        if geom.geom_type == "MultiLineString":
            longest = max(list(geom.geoms), key=lambda g: g.length)
            return longest.interpolate(0.5, normalized=True)
        return geom.representative_point()
    except Exception:
        try:
            return geom.representative_point()
        except Exception:
            return Point(0, 0)


def _first_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return float(value)
        except Exception:
            return default
    if isinstance(value, (list, tuple)) and value:
        return _first_number(value[0], default=default)
    text = str(value).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def _bearing_deg(cx: float, cy: float, x: float, y: float) -> float:
    # 0 = north, 90 = east
    return math.degrees(math.atan2(x - cx, y - cy)) % 360.0


def _normalize_vector(dx: float, dy: float) -> Tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return 1.0, 0.0
    return dx / norm, dy / norm


def _candidate_external_steps() -> List[float]:
    return [0.0, 2.0, 5.0, 10.0, 15.0, 25.0]


def _point_is_too_close(pt: Point, occupied_points: List[Any], min_sep_m: float = 1.0) -> bool:
    for op in occupied_points:
        try:
            if pt.distance(op) < min_sep_m:
                return True
        except Exception:
            continue
    return False


def _row_is_external(row: pd.Series) -> bool:
    val = row.get("is_external", 0)
    if pd.isna(val):
        return False
    return int(val) == 1


def _compass8(angle: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((angle % 360.0) + 22.5) // 45.0) % 8
    return labels[idx]


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(" ", "")
    text = text.replace("/", "")
    text = text.replace("\\", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    return text


def _slug_token(value: Any) -> str:
    text = _norm_text(value)
    return text or "GW"


def _resolve_whitelist(ext_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = ext_cfg.get("whitelist") or []
    return [
        {
            "raw": str(item),
            "norm": _norm_text(item),
            "slug": _slug_token(item),
        }
        for item in raw
        if str(item).strip()
    ]


# ----------------------------
# Network reference (CRS-safe)
# ----------------------------

def _network_ref(project: Project, kind: str, target_epsg: int) -> gpd.GeoDataFrame:
    kind = kind.lower().strip()

    if kind == "nodes":
        df = project.network.nodes.data
        if "geometry" not in df.columns or len(df) == 0:
            raise RuntimeError("Network nodes missing geometry/empty")
        g = gpd.GeoDataFrame(df[["node_id", "geometry"]].copy(), geometry="geometry", crs=getattr(df, "crs", None))
        g = _force_to_target_crs(g, target_epsg, name="network.nodes")
        return g

    if kind == "links":
        df = project.network.links.data
        if "geometry" not in df.columns or len(df) == 0:
            raise RuntimeError("Network links missing geometry/empty")
        g = gpd.GeoDataFrame(df.copy(), geometry="geometry", crs=getattr(df, "crs", None))
        if "link_type" in df.columns:
            g = g[df["link_type"].astype(str) != "centroid_connector"].copy()
        g = _force_to_target_crs(g, target_epsg, name="network.links")
        return g

    raise ValueError("kind must be 'nodes' or 'links'")


# ----------------------------
# AOI (model area) from network – largest component + quantile trim + padding
# ----------------------------

def build_model_area(
    project: Project,
    target_epsg: int,
    *,
    quantile: float = 0.0,
    pad_ratio: float = 0.02,
    extra_margin_m: float = 0.0,
    shift_x_m: float = 0.0,
    shift_y_m: float = 0.0,
) -> Any:
    print("Zoning: building model area (ORIENTED square of largest component)...")

    links = project.network.links.data
    nodes = project.network.nodes.data

    if "geometry" not in nodes.columns or len(nodes) == 0:
        raise RuntimeError("Network nodes missing geometry/empty")
    if len(links) == 0:
        raise RuntimeError("Network links empty")

    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))
    if "link_type" in links_gdf.columns:
        links_gdf = links_gdf[links_gdf["link_type"].astype(str) != "centroid_connector"].copy()
    links_gdf = _force_to_target_crs(links_gdf, target_epsg, name="network.links(for AOI)")

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))
    nodes_gdf = _force_to_target_crs(nodes_gdf, target_epsg, name="network.nodes(for AOI)")

    if "a_node" not in links_gdf.columns or "b_node" not in links_gdf.columns:
        raise RuntimeError("Links are missing a_node/b_node columns")
    if "node_id" not in nodes_gdf.columns:
        raise RuntimeError("Nodes are missing node_id column")

    from collections import defaultdict, deque

    adj = defaultdict(list)
    for a, b in zip(links_gdf["a_node"].astype(int).values, links_gdf["b_node"].astype(int).values):
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    largest_comp = []
    for start in adj.keys():
        if start in visited:
            continue
        qd = deque([start])
        comp = []
        visited.add(start)
        while qd:
            u = qd.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    qd.append(v)
        if len(comp) > len(largest_comp):
            largest_comp = comp

    if not largest_comp:
        minx, miny, maxx, maxy = nodes_gdf.total_bounds
        aoi = box(minx, miny, maxx, maxy)
        print("⚠ could not compute components -> using bbox of ALL nodes (axis-aligned fallback)")
        return aoi

    comp_set = set(largest_comp)
    core_nodes = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(comp_set)].copy()

    xs = core_nodes.geometry.x.to_numpy(dtype=float)
    ys = core_nodes.geometry.y.to_numpy(dtype=float)

    aoi = _oriented_square_from_points(
        xs,
        ys,
        quantile=float(quantile),
        pad_ratio=float(pad_ratio),
        make_square=True,
    )

    if float(extra_margin_m) > 0.0:
        aoi = aoi.buffer(float(extra_margin_m), join_style=2)

    if float(shift_x_m) != 0.0 or float(shift_y_m) != 0.0:
        aoi = translate(aoi, xoff=float(shift_x_m), yoff=float(shift_y_m))

    bminx, bminy, bmaxx, bmaxy = map(float, aoi.bounds)
    print(
        f"✓ AOI ORIENTED (EPSG:{target_epsg}) core_nodes={len(core_nodes)} "
        f"q={float(quantile)} pad={float(pad_ratio)} "
        f"extra_margin_m={float(extra_margin_m)} "
        f"shift=({float(shift_x_m)}, {float(shift_y_m)}): "
        f"bounds(minx={bminx:.2f}, miny={bminy:.2f}, maxx={bmaxx:.2f}, maxy={bmaxy:.2f})"
    )

    return aoi


def filter_zones_centroid_in_bbox(
    zones: gpd.GeoDataFrame,
    model_area_bbox: Any,
    crs_epsg: int,
    *,
    min_intersection_share: float = 0.05,
) -> gpd.GeoDataFrame:
    if zones.empty:
        return zones

    z = zones.copy()
    if z.crs is None:
        z = z.set_crs(epsg=crs_epsg, allow_override=True)
    if z.crs.to_epsg() != crs_epsg:
        z = z.to_crs(epsg=crs_epsg)

    rep_inside = z.geometry.representative_point().within(model_area_bbox)

    inter = z.geometry.intersection(model_area_bbox)
    inter_area = inter.area
    zone_area = z.geometry.area.replace(0, np.nan)
    share = (inter_area / zone_area).fillna(0.0)

    keep = (share >= min_intersection_share) | rep_inside

    n_kept = int(keep.sum())
    n_share_only = int((keep & ~rep_inside).sum())
    print(
        f"Zoning: AOI filter kept={n_kept} dropped={len(z) - n_kept} "
        f"(by intersection share: {n_share_only}, min_share={min_intersection_share})"
    )
    return z.loc[keep].reset_index(drop=True)


# ----------------------------
# Centroids INSIDE polygons
# ----------------------------

def calculate_centroids(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep_cols = [
        c
        for c in [
            "zone_id",
            "name",
            "source_rank",
            "is_external",
            "gateway_name",
            "anchor_node_id",
            "anchor_x",
            "anchor_y",
            "boundary_x",
            "boundary_y",
            "boundary_pos",
            "centroid_x",
            "centroid_y",
            "outward_dx",
            "outward_dy",
            "matched_ref",
            "matched_name",
            "link_type",
            "whitelist_token",
            "merged_from",
        ]
        if c in zones.columns
    ]

    pts = zones[keep_cols].copy()
    rep_points = zones.geometry.representative_point()

    geoms = []
    for idx, row in pts.iterrows():
        is_external_val = row.get("is_external", 0)
        is_external = int(0 if pd.isna(is_external_val) else is_external_val) == 1
        if is_external and "centroid_x" in pts.columns and "centroid_y" in pts.columns:
            cx = row.get("centroid_x")
            cy = row.get("centroid_y")
            if pd.notna(cx) and pd.notna(cy):
                geoms.append(Point(float(cx), float(cy)))
                continue
        geoms.append(rep_points.loc[idx])

    pts["geometry"] = geoms
    return gpd.GeoDataFrame(pts, crs=zones.crs)


# ----------------------------
# Eligible road nodes
# ----------------------------

_EXCLUDED_LINK_TYPES = frozenset({
    "footway", "path", "track", "steps", "cycleway", "pedestrian",
    "corridor", "bridleway", "proposed", "construction", "elevator",
    "service", "rest_area", "services", "traffic_mirror", "virtual",
})

_ROAD_CLASS_WEIGHT = {
    "motorway": 5.0,
    "motorway_link": 4.0,
    "trunk": 4.0,
    "trunk_link": 3.0,
    "primary": 3.0,
    "primary_link": 2.5,
    "secondary": 2.0,
    "secondary_link": 1.5,
    "tertiary": 1.5,
    "tertiary_link": 1.2,
    "unclassified": 0.8,
    "residential": 0.5,
    "living_street": 0.3,
    "road": 0.5,
}


def _eligible_road_nodes(project: Project) -> Tuple[set[int], Dict[int, float]]:
    import networkx as nx

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

    G = nx.Graph()
    for _, r in car.iterrows():
        a = int(r["a_node"])
        b = int(r["b_node"])
        G.add_edge(a, b)

    if G.number_of_nodes() == 0:
        return set(), {}

    largest_component = max(nx.connected_components(G), key=len)
    largest_component = set(int(x) for x in largest_component)

    car = car[
        car["a_node"].astype(int).isin(largest_component)
        & car["b_node"].astype(int).isin(largest_component)
    ].copy()

    road_class_weight = {
        "motorway": 8.0,
        "motorway_link": 7.0,
        "trunk": 7.0,
        "trunk_link": 6.0,
        "primary": 4.0,
        "primary_link": 3.5,
        "secondary": 2.0,
        "secondary_link": 1.8,
        "tertiary": 1.2,
        "tertiary_link": 1.0,
        "unclassified": 0.7,
        "road": 0.7,
        "residential": 0.4,
        "service": 0.2,
        "living_street": 0.05,
    }

    node_weight: Dict[int, float] = {}
    for _, r in car.iterrows():
        lt = str(r.get("link_type", "")).strip()
        w = float(road_class_weight.get(lt, 0.5))
        a = int(r["a_node"])
        b = int(r["b_node"])
        node_weight[a] = max(node_weight.get(a, 0.0), w)
        node_weight[b] = max(node_weight.get(b, 0.0), w)

    return set(node_weight.keys()), node_weight


# ----------------------------
# Boundary-whitelist gateway discovery
# ----------------------------

def _cluster_positions_on_ring(df: pd.DataFrame, pos_col: str, threshold_m: float, ring_length: float) -> List[pd.DataFrame]:
    if df.empty:
        return []

    s = df.sort_values(pos_col).copy()
    idxs = s.index.tolist()
    poss = s[pos_col].astype(float).tolist()

    clusters: List[List[Any]] = [[idxs[0]]]
    prev = poss[0]

    for idx, pos in zip(idxs[1:], poss[1:]):
        if (pos - prev) <= float(threshold_m):
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev = pos

    if len(clusters) > 1 and ring_length > 0:
        first_pos = float(s.loc[clusters[0][0], pos_col])
        last_pos = float(s.loc[clusters[-1][-1], pos_col])
        wrap_gap = ring_length - last_pos + first_pos
        if wrap_gap <= float(threshold_m):
            merged = clusters[-1] + clusters[0]
            clusters = [merged] + clusters[1:-1]

    return [df.loc[c].copy() for c in clusters]


def _select_gateway_target_nodes_boundary_whitelist(
    project: Project,
    target_epsg: int,
    model_area: Any,
    whitelist_specs: List[Dict[str, Any]],
    *,
    nodes_per_gateway: int = 2,
    boundary_buffer_m: float = 600.0,
    min_gateway_separation_m: float = 1800.0,
    allowed_link_types: Optional[List[str]] = None,
    merge_boundary_near_candidates: bool = False,
) -> Tuple[
    Dict[str, List[int]],
    Dict[str, Dict[str, Any]],
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    def _token_variants(raw_token: str, norm_token: str) -> set[str]:
        vals = set()
        for v in (raw_token, norm_token):
            t = _norm_text(v)
            if t:
                vals.add(t)
                if t.startswith("I") and t[1:].isdigit():
                    vals.add(t[1:])
        return vals

    def _match_ref_variants(row: pd.Series, variants: set[str]) -> int:
        for col in ("osm_ref_norm", "osm_ref", "ref"):
            if col in row and pd.notna(row.get(col)):
                rv = _norm_text(row.get(col))
                if rv in variants:
                    return 1
                if rv.startswith("I") and rv[1:].isdigit() and rv[1:] in variants:
                    return 1
        return 0

    road_nids, node_weight = _eligible_road_nodes(project)
    if not road_nids:
        empty_lines = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        empty_points = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        return {}, {}, empty_lines, empty_points

    nodes_gdf = _network_ref(project, "nodes", target_epsg)[["node_id", "geometry"]].copy()
    nodes_gdf = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(road_nids)].copy()
    if nodes_gdf.empty:
        empty_lines = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        empty_points = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        return {}, {}, empty_lines, empty_points

    node_geom: Dict[int, Point] = {
        int(r["node_id"]): r.geometry
        for _, r in nodes_gdf.iterrows()
    }

    center = model_area.representative_point()
    boundary = model_area.boundary
    ring_length = float(boundary.length)
    cx, cy = float(center.x), float(center.y)

    links_gdf = _network_ref(project, "links", target_epsg).copy()
    if links_gdf.empty:
        empty_lines = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        empty_points = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        return {}, {}, empty_lines, empty_points

    links_gdf = links_gdf[
        links_gdf["modes"].astype(str).str.contains("c", na=False)
        & (links_gdf["link_type"].astype(str) != "centroid_connector")
        & links_gdf["a_node"].astype(int).isin(road_nids)
        & links_gdf["b_node"].astype(int).isin(road_nids)
    ].copy()

    if allowed_link_types:
        allowed = {str(x).strip() for x in allowed_link_types}
        if allowed:
            links_gdf = links_gdf[links_gdf["link_type"].astype(str).isin(allowed)].copy()

    if links_gdf.empty:
        empty_lines = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        empty_points = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")
        return {}, {}, empty_lines, empty_points

    def _nearest_boundary_point(geom: Any) -> Point:
        try:
            _, bp = nearest_points(geom, boundary)
            return bp
        except Exception:
            mid = _safe_line_midpoint(geom)
            try:
                _, bp = nearest_points(mid, boundary)
                return bp
            except Exception:
                return mid

    links_gdf["_dist_boundary"] = links_gdf.geometry.distance(boundary)
    links_gdf["_boundary_pt"] = links_gdf.geometry.apply(_nearest_boundary_point)
    links_gdf["_boundary_pos"] = links_gdf["_boundary_pt"].apply(lambda p: float(boundary.project(p)))
    links_gdf["_boundary_angle"] = links_gdf["_boundary_pt"].apply(
        lambda p: _bearing_deg(cx, cy, float(p.x), float(p.y))
    )
    links_gdf["_road_class_weight"] = links_gdf["link_type"].astype(str).map(_ROAD_CLASS_WEIGHT).fillna(0.5)

    gateway_targets: Dict[str, List[int]] = {}
    gateway_meta: Dict[str, Dict[str, Any]] = {}
    used_names: Dict[str, int] = {}

    debug_corridor_parts: List[gpd.GeoDataFrame] = []
    debug_point_rows: List[Dict[str, Any]] = []

    for spec_priority, spec in enumerate(whitelist_specs):
        token_raw = spec["raw"]
        token_norm = spec["norm"]
        token_slug = spec["slug"]

        ref_variants = _token_variants(token_raw, token_norm)

        print(f"=== DISCOVER whitelist token {token_raw} ===")
        print(f"  ref variants: {sorted(ref_variants)}")

        cand = links_gdf.copy()
        cand["_ref_match"] = cand.apply(lambda r: _match_ref_variants(r, ref_variants), axis=1).astype(int)
        matched = cand[cand["_ref_match"] == 1].copy()

        if matched.empty:
            print(f"  [warn] token {token_raw}: no ref match in whole network")
            continue

        print(f"  matched in whole network: {len(matched)}")

        matched["debug_token"] = token_raw
        matched["debug_token_slug"] = token_slug
        debug_corridor_parts.append(matched.copy())

        boundary_near = matched[matched["_dist_boundary"] <= float(boundary_buffer_m)].copy()
        if boundary_near.empty:
            boundary_near = matched.nsmallest(min(20, len(matched)), "_dist_boundary").copy()
            print(
                f"  [warn] token {token_raw}: no boundary-near matched links within "
                f"{boundary_buffer_m:.0f} m, using nearest {len(boundary_near)}"
            )

        clusters = _cluster_positions_on_ring(
            boundary_near,
            pos_col="_boundary_pos",
            threshold_m=float(min_gateway_separation_m),
            ring_length=ring_length,
        )
        print(f"  boundary clusters: {len(clusters)}")

        for cl_i, cl in enumerate(clusters, start=1):
            local_nodes = sorted(
                {
                    int(x)
                    for x in pd.concat([cl["a_node"], cl["b_node"]], ignore_index=True).astype(int).tolist()
                    if int(x) in node_geom
                }
            )
            if not local_nodes:
                continue

            boundary_local_nodes = [
                nid for nid in local_nodes
                if float(node_geom[nid].distance(boundary)) <= float(boundary_buffer_m) * 1.25
            ]
            target_pool = boundary_local_nodes if boundary_local_nodes else local_nodes

            chosen = _pick_nodes_near_boundary(
                target_pool,
                node_geom,
                boundary,
                node_weight,
                max_nodes=int(nodes_per_gateway),
                min_node_sep_m=25.0,
            )

            if not chosen:
                print(f"  [warn] token {token_raw} cluster {cl_i}: no chosen boundary nodes")
                continue

            anchor_id = int(chosen[0])
            anchor_geom = node_geom[anchor_id]

            cluster_boundary_pos = _circular_mean_ring_pos(
                cl["_boundary_pos"].astype(float).tolist(),
                ring_length,
            )
            cluster_boundary_pt = boundary.interpolate(cluster_boundary_pos)

            ux, uy = _normalize_vector(
                float(cluster_boundary_pt.x) - cx,
                float(cluster_boundary_pt.y) - cy,
            )
            cluster_angle = _bearing_deg(cx, cy, float(cluster_boundary_pt.x), float(cluster_boundary_pt.y))
            compass = _compass8(cluster_angle)

            gw_name = f"{token_slug}_{compass}"
            if gw_name in used_names:
                used_names[gw_name] += 1
                gw_name = f"{gw_name}_{used_names[gw_name]}"
            else:
                used_names[gw_name] = 1

            best_link = cl.sort_values(
                ["_road_class_weight", "_dist_boundary"],
                ascending=[False, True],
            ).iloc[0]

            matched_ref = str(
                best_link.get("osm_ref_norm", "")
                or best_link.get("osm_ref", "")
                or best_link.get("ref", "")
                or ""
            )
            matched_name = str(
                best_link.get("osm_name_raw", "")
                or best_link.get("name", "")
                or ""
            )

            gateway_targets[gw_name] = chosen
            gateway_meta[gw_name] = {
                "gateway_name": gw_name,
                "whitelist_token": token_raw,
                "whitelist_priority": int(spec_priority),
                "cluster_index": cl_i,
                "anchor_node_id": int(anchor_id),
                "anchor_x": float(anchor_geom.x),
                "anchor_y": float(anchor_geom.y),
                "boundary_x": float(cluster_boundary_pt.x),
                "boundary_y": float(cluster_boundary_pt.y),
                "boundary_pos": float(cluster_boundary_pos),
                "outward_dx": float(ux),
                "outward_dy": float(uy),
                "target_node_ids": chosen,
                "matched_ref": matched_ref,
                "matched_name": matched_name,
                "link_type": str(best_link.get("link_type", "") or ""),
                "boundary_angle": float(cluster_angle),
                "dist_boundary_m": float(anchor_geom.distance(boundary)),
                "merged_from": "",
            }

            print(
                f"  gateway {gw_name}: "
                f"boundary=({float(cluster_boundary_pt.x):.1f}, {float(cluster_boundary_pt.y):.1f}) "
                f"anchor=({float(anchor_geom.x):.1f}, {float(anchor_geom.y):.1f}) "
                f"targets={chosen} "
                f"matched_ref='{matched_ref}' "
                f"matched_name='{matched_name}' "
                f"type={gateway_meta[gw_name]['link_type']}"
            )

            debug_point_rows.append({
                "kind": "anchor",
                "token": token_raw,
                "cluster": cl_i,
                "node_id": int(anchor_id),
                "geometry": anchor_geom,
            })
            for nid in chosen:
                debug_point_rows.append({
                    "kind": "target",
                    "token": token_raw,
                    "cluster": cl_i,
                    "node_id": int(nid),
                    "geometry": node_geom[int(nid)],
                })

    if merge_boundary_near_candidates:
        gateway_targets, gateway_meta = _merge_gateway_candidates_on_boundary(
            gateway_targets=gateway_targets,
            gateway_meta=gateway_meta,
            node_geom=node_geom,
            node_weight=node_weight,
            model_area=model_area,
            merge_distance_m=float(min_gateway_separation_m),
            nodes_per_gateway=int(nodes_per_gateway),
        )
    else:
        print("  boundary-near merge disabled by config")

    debug_corridors_gdf = (
        gpd.GeoDataFrame(pd.concat(debug_corridor_parts, ignore_index=True), crs=links_gdf.crs)
        if debug_corridor_parts
        else gpd.GeoDataFrame({"geometry": []}, crs=links_gdf.crs)
    )

    debug_points_gdf = (
        gpd.GeoDataFrame(debug_point_rows, geometry="geometry", crs=nodes_gdf.crs)
        if debug_point_rows
        else gpd.GeoDataFrame({"geometry": []}, crs=nodes_gdf.crs)
    )

    return gateway_targets, gateway_meta, debug_corridors_gdf, debug_points_gdf


def _build_external_gateway_zones_from_boundary_meta(
    gateway_meta: Dict[str, Dict[str, Any]],
    target_epsg: int,
    *,
    zone_offset_m: float = 10.0,
    zone_size_m: float = 40.0,
    start_id: int = 8_000_000_000,
) -> gpd.GeoDataFrame:
    """
    Build external gateway zones from REAL AOI boundary points.
    Zone centroid is placed just outside the AOI:
        centroid = boundary_point + outward * (half_size + zone_offset_m)
    """
    if not gateway_meta:
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")

    rows: List[Dict[str, Any]] = []
    half = float(zone_size_m) / 2.0

    for i, gw_name in enumerate(sorted(gateway_meta.keys())):
        meta = gateway_meta[gw_name]

        bx = float(meta["boundary_x"])
        by = float(meta["boundary_y"])
        ux = float(meta["outward_dx"])
        uy = float(meta["outward_dy"])

        centroid_dist = half + float(zone_offset_m)
        cx = bx + ux * centroid_dist
        cy = by + uy * centroid_dist

        geom = translate(
            box(-half, -half, half, half),
            xoff=cx,
            yoff=cy,
        )

        rows.append({
            "zone_id": start_id + i,
            "name": f"EXT_{gw_name}",
            "source_rank": 999,
            "is_external": 1,
            "gateway_name": gw_name,
            "anchor_node_id": int(meta["anchor_node_id"]),
            "anchor_x": float(meta["anchor_x"]),
            "anchor_y": float(meta["anchor_y"]),
            "boundary_x": bx,
            "boundary_y": by,
            "boundary_pos": float(meta.get("boundary_pos", 0.0)),
            "centroid_x": cx,
            "centroid_y": cy,
            "outward_dx": ux,
            "outward_dy": uy,
            "matched_ref": str(meta.get("matched_ref", "")),
            "matched_name": str(meta.get("matched_name", "")),
            "link_type": str(meta.get("link_type", "")),
            "whitelist_token": str(meta.get("whitelist_token", "")),
            "merged_from": str(meta.get("merged_from", "")),
            "geometry": geom,
        })

        print(
            f"  gateway zone {gw_name}: "
            f"boundary=({bx:.1f}, {by:.1f}) "
            f"centroid=({cx:.1f}, {cy:.1f}) "
            f"anchor_node={int(meta['anchor_node_id'])}"
        )

    return gpd.GeoDataFrame(rows, crs=f"EPSG:{target_epsg}")


# ----------------------------
# Map export (zones over network + AOI highlighted)
# ----------------------------

def export_map_png(
    project: Project,
    zones: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    model_area_bbox: Any,
    output_dir: Path,
    filename: str = "zones_map.png",
    title_suffix: str = "",
    dpi: int = 250,
    debug_corridors: Optional[gpd.GeoDataFrame] = None,
    debug_points: Optional[gpd.GeoDataFrame] = None,
    *,
    crs_epsg: Optional[int] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("⚠ matplotlib not installed — skipping map export")
        return

    if zones.empty:
        print("⚠ no zones — skipping map export")
        return

    if zones.crs is not None and zones.crs.to_epsg() is not None:
        target_epsg = int(zones.crs.to_epsg())
    elif crs_epsg is not None:
        target_epsg = int(crs_epsg)
    else:
        raise ValueError("export_map_png: zones have no CRS and crs_epsg was not provided")

    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        print("⚠ links have no geometry — skipping map export")
        return

    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))
    if "link_type" in links_gdf.columns:
        links_gdf = links_gdf[links_gdf["link_type"].astype(str) != "centroid_connector"].copy()

    links_gdf = _force_to_target_crs(links_gdf, target_epsg, name="network.links(for map)")

    z = zones.copy()
    c = centroids.copy()

    if "is_external" not in z.columns:
        z["is_external"] = 0
    z["is_external"] = pd.to_numeric(z["is_external"], errors="coerce").fillna(0).astype(int)

    if "is_external" not in c.columns:
        c["is_external"] = 0
    c["is_external"] = pd.to_numeric(c["is_external"], errors="coerce").fillna(0).astype(int)

    internal_zones = z[z["is_external"] == 0].copy()
    external_zones = z[z["is_external"] == 1].copy()

    internal_centroids = c[c["is_external"] == 0].copy()
    external_centroids = c[c["is_external"] == 1].copy()

    sx, sy, sx2, sy2 = links_gdf.total_bounds
    zx, zy, zx2, zy2 = z.total_bounds

    minx0 = min(float(sx), float(zx))
    miny0 = min(float(sy), float(zy))
    maxx0 = max(float(sx2), float(zx2))
    maxy0 = max(float(sy2), float(zy2))

    margin_ratio = 0.02
    w = maxx0 - minx0
    h = maxy0 - miny0
    margin_x = max(w * margin_ratio, 50.0)
    margin_y = max(h * margin_ratio, 50.0)

    minx = minx0 - margin_x
    maxx = maxx0 + margin_x
    miny = miny0 - margin_y
    maxy = maxy0 + margin_y

    aoi_gdf = gpd.GeoDataFrame({"geometry": [model_area_bbox]}, crs=f"EPSG:{target_epsg}")

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))

    links_gdf.plot(ax=ax, color="dimgray", linewidth=0.8, alpha=0.9, zorder=1)

    aoi_gdf.plot(ax=ax, color="gold", alpha=0.08, zorder=2)
    aoi_gdf.boundary.plot(ax=ax, color="goldenrod", linewidth=3.0, zorder=10)

    if not internal_zones.empty:
        internal_zones.plot(
            ax=ax,
            facecolor="lightblue",
            edgecolor="navy",
            alpha=0.35,
            linewidth=1.0,
            zorder=3,
        )

    if not external_zones.empty:
        external_zones.plot(
            ax=ax,
            facecolor="mistyrose",
            edgecolor="darkred",
            alpha=0.55,
            linewidth=1.2,
            zorder=4,
        )

    if not internal_centroids.empty:
        internal_centroids.plot(
            ax=ax,
            color="navy",
            markersize=20,
            marker="o",
            zorder=5,
        )

    if not external_centroids.empty:
        external_centroids.plot(
            ax=ax,
            color="red",
            markersize=45,
            marker="X",
            zorder=6,
        )

    if debug_corridors is not None and not debug_corridors.empty:
        dbg = debug_corridors.copy()
        if dbg.crs is None:
            dbg = dbg.set_crs(epsg=target_epsg, allow_override=True)
        if dbg.crs.to_epsg() != target_epsg:
            dbg = dbg.to_crs(epsg=target_epsg)

        dbg.plot(
            ax=ax,
            color="red",
            linewidth=2.5,
            alpha=0.85,
            zorder=7,
        )

    if debug_points is not None and not debug_points.empty:
        dbg_pts = debug_points.copy()
        if dbg_pts.crs is None:
            dbg_pts = dbg_pts.set_crs(epsg=target_epsg, allow_override=True)
        if dbg_pts.crs.to_epsg() != target_epsg:
            dbg_pts = dbg_pts.to_crs(epsg=target_epsg)

        terminals = dbg_pts[dbg_pts["kind"] == "terminal"].copy() if "kind" in dbg_pts.columns else dbg_pts.iloc[0:0]
        anchors = dbg_pts[dbg_pts["kind"] == "anchor"].copy() if "kind" in dbg_pts.columns else dbg_pts.iloc[0:0]

        if not terminals.empty:
            terminals.plot(
                ax=ax,
                color="yellow",
                markersize=35,
                marker="o",
                zorder=8,
            )

        if not anchors.empty:
            anchors.plot(
                ax=ax,
                color="red",
                markersize=90,
                marker="X",
                zorder=9,
            )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Network + TAZ zones ({len(zones)}) + AOI (oriented){title_suffix}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    legend_handles = [
        Line2D([0], [0], color="dimgray", linewidth=2.0, label="Network"),
        Patch(facecolor="lightblue", edgecolor="navy", alpha=0.35, label=f"Internal zones ({len(internal_zones)})"),
        Patch(facecolor="mistyrose", edgecolor="darkred", alpha=0.55, label=f"Exit zones ({len(external_zones)})"),
        Line2D([0], [0], marker="o", color="navy", linestyle="None", markersize=8, label=f"Internal centroids ({len(internal_centroids)})"),
        Line2D([0], [0], marker="X", color="red", linestyle="None", markersize=9, label=f"Exit centroids ({len(external_centroids)})"),
        Patch(facecolor="gold", edgecolor="goldenrod", alpha=0.08, label="AOI"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)

    ax.set_axis_off()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / filename
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ map: {png_path}")


# ----------------------------
# Centroid node + connectors
# ----------------------------

def _delete_all_connectors_and_reset_centroids(project: Project) -> int:
    db = str(project.project_base_path) + "/project_database.sqlite"
    conn = sqlite3.connect(db)
    try:
        before = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='centroid_connector'"
        ).fetchone()[0]
        conn.execute("DELETE FROM links WHERE link_type='centroid_connector'")
        conn.execute("UPDATE nodes SET is_centroid=0 WHERE is_centroid=1")

        orphans = conn.execute("""
            SELECT n.node_id FROM nodes n
            WHERE NOT EXISTS (SELECT 1 FROM links l WHERE l.a_node=n.node_id OR l.b_node=n.node_id)
        """).fetchall()
        if orphans:
            ids = [r[0] for r in orphans]
            conn.executemany("DELETE FROM nodes WHERE node_id=?", [(i,) for i in ids])
            print(f"  Removed {len(ids)} orphan nodes")

        conn.commit()
        print(f"  Cleaned: {before} old connectors deleted, is_centroid flags cleared")
        return before
    finally:
        conn.close()


def _select_diverse_connectors(
    eligible: gpd.GeoDataFrame,
    centroid_pt: Point,
    max_connectors: int,
    max_distance_m: float,
    pool_size: int = 40,
) -> gpd.GeoDataFrame:
    if eligible.empty or max_connectors <= 0:
        return eligible.head(0).copy()

    cand = eligible.copy()
    cand["_dist"] = cand.geometry.distance(centroid_pt)

    nearby = cand[cand["_dist"] <= float(max_distance_m)].copy()

    if nearby.empty:
        nearby = cand.nsmallest(min(pool_size, len(cand)), "_dist").copy()
    else:
        nearby = nearby.nsmallest(min(pool_size, len(nearby)), "_dist").copy()

    if nearby.empty:
        return nearby

    nearby["_dist"] = nearby["_dist"].clip(lower=1.0)

    cx, cy = centroid_pt.x, centroid_pt.y
    nearby["_angle"] = np.degrees(
        np.arctan2(nearby.geometry.y - cy, nearby.geometry.x - cx)
    ) % 360.0

    nearby["_score"] = nearby["road_weight"] / np.power(nearby["_dist"], 0.75)

    n_sectors = min(max_connectors, 8)
    sector_size = 360.0 / n_sectors
    nearby["_sector"] = (nearby["_angle"] // sector_size).astype(int)

    chosen_idx: List[int] = []

    for sec in range(n_sectors):
        sec_df = nearby[nearby["_sector"] == sec]
        if sec_df.empty:
            continue
        best_idx = sec_df["_score"].idxmax()
        chosen_idx.append(best_idx)
        if len(chosen_idx) >= max_connectors:
            break

    if len(chosen_idx) < max_connectors:
        remaining = nearby.drop(index=chosen_idx, errors="ignore").nlargest(
            max_connectors - len(chosen_idx), "_score"
        )
        chosen_idx.extend(remaining.index.tolist())

    selected = nearby.loc[chosen_idx[:max_connectors]].copy()
    selected = selected.sort_values(
        ["_score", "road_weight", "_dist"],
        ascending=[False, False, True],
    ).copy()
    selected["dist"] = selected["_dist"]

    return selected


def create_centroid_connectors(
    project: Project,
    centroids: gpd.GeoDataFrame,
    max_connectors: int,
    max_distance_m: float,
    speed_kmh: float = 30.0,
    capacity_vph: float = 5000.0,
    lanes: int = 4,
    access_penalty_s: float = 120.0,
    gateway_targets: Optional[Dict[str, List[int]]] = None,
    external_speed_kmh: Optional[float] = None,
    external_access_penalty_s: Optional[float] = None,
) -> Dict[int, int]:
    """
    For external zones:
    - centroid is placed outside model according to stored centroid_x / centroid_y
    - if that still collides, it is moved only further OUT along the same outward axis
    - connectors go directly to the configured gateway target nodes
    """
    print("\n=== CREATE centroid connectors ===")

    gateway_targets = gateway_targets or {}
    nodes = project.network.nodes.data
    if "geometry" not in nodes.columns or len(nodes) == 0:
        raise RuntimeError("Network nodes missing geometry/empty")

    road_nids, node_weight = _eligible_road_nodes(project)
    existing_nids = set(int(x) for x in nodes["node_id"].values)

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))
    nodes_gdf = _force_to_target_crs(
        nodes_gdf,
        int(centroids.crs.to_epsg()),
        name="network.nodes(for connector candidates)",
    )

    eligible = nodes_gdf[nodes_gdf["node_id"].isin(road_nids)].copy()
    eligible["road_weight"] = eligible["node_id"].map(node_weight).fillna(0.5)
    print(
        f"  Eligible road-network nodes: {len(eligible)} (from {len(nodes_gdf)} total, "
        f"excluded {len(nodes_gdf) - len(eligible)} non-car types)"
    )

    occupied_points = list(nodes_gdf.geometry.dropna())

    if eligible.empty:
        print("  WARNING: no eligible road-network nodes found")
        return {}

    next_id = 1
    zone_to_centroid: Dict[int, int] = {}
    for zid in sorted(centroids["zone_id"].astype(int)):
        while next_id in existing_nids:
            next_id += 1
        zone_to_centroid[zid] = next_id
        existing_nids.add(next_id)
        next_id += 1

    print(
        f"  Centroid IDs: {min(zone_to_centroid.values())}-{max(zone_to_centroid.values())} "
        f"for {len(zone_to_centroid)} zones"
    )

    nodes_wgs84 = project.network.nodes.data[["node_id", "geometry"]].copy()
    wgs84_lookup = {int(r["node_id"]): r["geometry"] for _, r in nodes_wgs84.iterrows()}

    created = 0
    for _, c in centroids.iterrows():
        zone_id = int(c["zone_id"])
        centroid_node_id = zone_to_centroid[zone_id]

        is_external = bool(int(c.get("is_external", 0))) if pd.notna(c.get("is_external", 0)) else False
        gateway_name = str(c.get("gateway_name", "") or "").strip()

        if is_external and pd.notna(c.get("centroid_x")) and pd.notna(c.get("centroid_y")):
            base_pt = Point(float(c["centroid_x"]), float(c["centroid_y"]))
            ux, uy = _normalize_vector(float(c.get("outward_dx", 1.0)), float(c.get("outward_dy", 0.0)))
        else:
            base_pt = c.geometry
            ux, uy = 1.0, 0.0

        saved_centroid_pt: Optional[Point] = None
        saved_centroid_wgs84 = None

        if is_external:
            candidate_steps = _candidate_external_steps()
        else:
            candidate_steps = [0.0]

        for step in candidate_steps:
            candidate_pt = translate(base_pt, xoff=ux * step, yoff=uy * step)

            if _point_is_too_close(candidate_pt, occupied_points, min_sep_m=1.0):
                continue

            candidate_wgs84 = _to_wgs84_point(candidate_pt, centroids.crs)

            try:
                new_node = (
                    project.network.nodes.new_centroid(centroid_node_id)
                    if hasattr(project.network.nodes, "new_centroid")
                    else project.network.nodes.new()
                )
                if not hasattr(project.network.nodes, "new_centroid"):
                    new_node.__dict__["node_id"] = centroid_node_id

                new_node.is_centroid = 1
                new_node.geometry = candidate_wgs84
                new_node.save()

                saved_centroid_pt = candidate_pt
                saved_centroid_wgs84 = candidate_wgs84
                occupied_points.append(candidate_pt)
                break

            except sqlite3.IntegrityError:
                continue

        if saved_centroid_pt is None or saved_centroid_wgs84 is None:
            raise RuntimeError(
                f"Could not place centroid node for zone_id={zone_id} "
                f"without overlapping an existing node"
            )

        centroid_pt = saved_centroid_pt
        centroid_wgs84 = saved_centroid_wgs84

        if is_external and gateway_name in gateway_targets:
            target_ids = [int(x) for x in gateway_targets[gateway_name]]
            cand = eligible[eligible["node_id"].astype(int).isin(target_ids)].copy()
            cand["dist"] = cand.geometry.distance(centroid_pt)
            cand = cand.sort_values(["dist", "road_weight"], ascending=[True, False]).head(len(target_ids))
        else:
            cand = _select_diverse_connectors(
                eligible,
                centroid_pt,
                max_connectors,
                max_distance_m,
            )

        for _, n in cand.iterrows():
            target_node_id = int(n["node_id"])
            dist_m = float(n["dist"])

            target_geom = wgs84_lookup.get(target_node_id)
            if target_geom is None:
                continue

            link = project.network.links.new()
            link.__dict__["a_node"] = centroid_node_id
            link.__dict__["b_node"] = target_node_id
            link.direction = 0
            link.modes = "c"
            link.link_type = "centroid_connector"

            link_speed = (
                float(external_speed_kmh)
                if (is_external and external_speed_kmh is not None)
                else float(speed_kmh)
            )
            link_penalty = (
                float(external_access_penalty_s)
                if (is_external and external_access_penalty_s is not None)
                else float(access_penalty_s)
            )

            link.distance = dist_m
            link.speed_ab = link_speed
            link.speed_ba = link_speed
            link.capacity_ab = capacity_vph
            link.capacity_ba = capacity_vph
            link.lanes_ab = lanes
            link.lanes_ba = lanes

            time_s = (dist_m / 1000.0) / max(link_speed, 1e-6) * 3600.0 + link_penalty
            if hasattr(link, "travel_time_ab"):
                link.travel_time_ab = time_s
                link.travel_time_ba = time_s

            link.geometry = LineString([
                (centroid_wgs84.x, centroid_wgs84.y),
                (target_geom.x, target_geom.y),
            ])

            link.save()
            created += 1

    print(f"  connectors created: {created}  ({len(zone_to_centroid)} zones, {max_connectors} per zone)")
    return zone_to_centroid


def export_connector_diagnostics(
    project: Project,
    zone_to_centroid: Dict[int, int],
    output_dir: Path,
) -> None:
    db = str(project.project_base_path) + "/project_database.sqlite"
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT link_id, a_node, b_node, distance, speed_ab, travel_time_ab "
        "FROM links WHERE link_type='centroid_connector'"
    ).fetchall()
    conn.close()

    centroid_to_zone = {v: k for k, v in zone_to_centroid.items()}
    records = []
    for lid, a, b, dist, speed, tt in rows:
        zid = centroid_to_zone.get(a, centroid_to_zone.get(b))
        road_node = b if a in centroid_to_zone.values() else a
        travel_km = (dist or 0) / 1000.0
        penalty_s = (tt or 0) - (travel_km / max(speed or 30, 1) * 3600) if tt and speed else 0
        records.append({
            "zone_id": zid,
            "centroid_node": a,
            "road_node": road_node,
            "link_id": lid,
            "distance_m": round(dist or 0, 1),
            "speed_kmh": speed,
            "travel_time_s": round(tt or 0, 1),
            "access_penalty_s": round(max(penalty_s, 0), 1),
        })

    if records:
        df = pd.DataFrame(records)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "connector_diagnostics.csv"
        df.to_csv(out, index=False)
        print(f"  Connector diagnostics: {out} ({len(df)} connectors)")


def export_gateway_diagnostics(
    gateway_meta: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    if not gateway_meta:
        return

    rows = []
    for gw_name, meta in gateway_meta.items():
        rows.append({
            "gateway_name": gw_name,
            "whitelist_token": meta.get("whitelist_token"),
            "whitelist_priority": meta.get("whitelist_priority"),
            "cluster_index": meta.get("cluster_index"),
            "anchor_node_id": meta.get("anchor_node_id"),
            "boundary_x": meta.get("boundary_x"),
            "boundary_y": meta.get("boundary_y"),
            "boundary_pos": meta.get("boundary_pos"),
            "anchor_x": meta.get("anchor_x"),
            "anchor_y": meta.get("anchor_y"),
            "outward_dx": meta.get("outward_dx"),
            "outward_dy": meta.get("outward_dy"),
            "matched_ref": meta.get("matched_ref"),
            "matched_name": meta.get("matched_name"),
            "link_type": meta.get("link_type"),
            "boundary_angle": meta.get("boundary_angle"),
            "dist_boundary_m": meta.get("dist_boundary_m"),
            "target_node_ids": ",".join(str(x) for x in meta.get("target_node_ids", [])),
            "merged_from": meta.get("merged_from", ""),
        })

    if rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "gateway_diagnostics.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Gateway diagnostics: {out} ({len(rows)} gateways)")


def export_gateway_seed_lookup(
    gateway_meta: Dict[str, Dict[str, Any]],
    output_path: Path,
    *,
    crs_epsg: int,
) -> None:
    """
    Export stable gateway seed lookup for build-supernetwork.

    Writes:
    - parquet (if suffix .parquet)
    - or csv/geojson according to suffix
    """
    if not gateway_meta:
        return

    rows = []
    for gw_name, meta in gateway_meta.items():
        rows.append({
            "gateway_name": gw_name,
            "whitelist_token": str(meta.get("whitelist_token", "")),
            "boundary_x": float(meta.get("boundary_x", 0.0)),
            "boundary_y": float(meta.get("boundary_y", 0.0)),
            "boundary_pos": float(meta.get("boundary_pos", 0.0)),
            "anchor_node_id": int(meta.get("anchor_node_id", 0)),
            "anchor_x": float(meta.get("anchor_x", 0.0)),
            "anchor_y": float(meta.get("anchor_y", 0.0)),
            "outward_dx": float(meta.get("outward_dx", 0.0)),
            "outward_dy": float(meta.get("outward_dy", 0.0)),
            "matched_ref": str(meta.get("matched_ref", "")),
            "matched_name": str(meta.get("matched_name", "")),
            "link_type": str(meta.get("link_type", "")),
            "geometry": Point(float(meta.get("boundary_x", 0.0)), float(meta.get("boundary_y", 0.0))),
        })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{crs_epsg}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(output_path, index=False)
    elif suffix == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    elif suffix == ".csv":
        pd.DataFrame(gdf.drop(columns="geometry")).to_csv(output_path, index=False)
    else:
        # default: parquet
        out = output_path.with_suffix(".parquet")
        gdf.to_parquet(out, index=False)
        output_path = out

    print(f"  Gateway seed lookup: {output_path}")


# ----------------------------
# Main
# ----------------------------

def build_zones_and_connectors(config_path: str | Path | Dict[str, Any] = "config/sim.yaml") -> None:
    print("Zoning: loading config...")
    cfg = _load_cfg(config_path)

    project_dir = Path(cfg["project_path"])

    if cfg.get("crs_epsg") is None:
        raise ValueError("config.crs_epsg is required for zoning")
    crs_epsg = int(cfg["crs_epsg"])
    zoning_cfg = cfg.get("zoning", {}) or {}

    sources = zoning_cfg.get("sources")
    if not sources:
        raise ValueError("config.zoning.sources must be set")

    area_filter_cfg = zoning_cfg.get("zone_area_filter", {}) or {}
    cache_file = zoning_cfg.get("cache_file")
    output_dir = Path(zoning_cfg.get("output_dir", "outputs/baseline/zones"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_connectors = int(zoning_cfg.get("max_connectors", 6))
    max_distance_m = float(zoning_cfg.get("max_distance_m", 3000.0))
    connector_speed = float(zoning_cfg.get("connector_speed_kmh", 20.0))
    connector_capacity = float(zoning_cfg.get("connector_capacity_vph", 2000.0))
    connector_lanes = int(zoning_cfg.get("connector_lanes", 1))
    connector_penalty = float(zoning_cfg.get("connector_access_penalty_s", 300.0))

    ext_cfg = zoning_cfg.get("external_gateways", {}) or {}
    merge_boundary_near_candidates = bool(ext_cfg.get("merge_boundary_near_candidates", False))
    export_lookup = bool(ext_cfg.get("export_lookup", False))
    export_lookup_path = _safe_path(ext_cfg.get("export_lookup_path", "data/cache/gateway_lookup_seed.parquet"))

    print("Zoning: opening project...")
    project = Project()
    project.open(str(project_dir))

    try:
        aoi_cfg = zoning_cfg.get("aoi", {}) or {}
        model_area = build_model_area(
            project,
            crs_epsg,
            quantile=float(aoi_cfg.get("quantile", 0.003)),
            pad_ratio=float(aoi_cfg.get("pad_ratio", 0.008)),
            extra_margin_m=float(aoi_cfg.get("extra_margin_m", 200.0)),
            shift_x_m=float(aoi_cfg.get("shift_x_m", 0.0)),
            shift_y_m=float(aoi_cfg.get("shift_y_m", 0.0)),
        )

        print("Zoning: loading zones (cache or sources)...")
        zones = load_zones_from_sources(
            sources=sources,
            crs_epsg=crs_epsg,
            cache_file=Path(cache_file) if cache_file else None,
            area_filter_cfg=area_filter_cfg,
        )

        zones = filter_zones_centroid_in_bbox(zones, model_area, crs_epsg)
        print(f"✓ zones with representative_point in AOI: {len(zones)}")

        zones = remove_overlaps_by_priority(zones, rank_col="source_rank", min_area_m2=25.0)
        print(f"✓ zones after priority de-overlap: {len(zones)}")

        gateway_targets: Dict[str, List[int]] = {}
        gateway_meta: Dict[str, Dict[str, Any]] = {}
        debug_corridors = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")
        debug_points = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

        if bool(ext_cfg.get("enabled", False)):
            whitelist_specs = _resolve_whitelist(ext_cfg)

            gateway_targets, gateway_meta, debug_corridors, debug_points = _select_gateway_target_nodes_boundary_whitelist(
                project=project,
                target_epsg=crs_epsg,
                model_area=model_area,
                whitelist_specs=whitelist_specs,
                nodes_per_gateway=int(ext_cfg.get("connectors_per_gateway", 2)),
                boundary_buffer_m=float(ext_cfg.get("boundary_buffer_m", 600.0)),
                min_gateway_separation_m=float(ext_cfg.get("min_gateway_separation_m", 1800.0)),
                allowed_link_types=ext_cfg.get(
                    "allowed_link_types",
                    ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"],
                ),
                merge_boundary_near_candidates=merge_boundary_near_candidates,
            )

            if debug_corridors is not None and not debug_corridors.empty:
                dbg = debug_corridors.copy()

                extra_geom_cols = [
                    c for c in dbg.columns if c != dbg.geometry.name and dbg[c].dtype == "geometry"
                ]
                if extra_geom_cols:
                    dbg = dbg.drop(columns=extra_geom_cols)

                dbg.to_file(output_dir / "gateway_corridors_debug.geojson", driver="GeoJSON")
                print(f"  gateway corridor debug: {output_dir / 'gateway_corridors_debug.geojson'}")

            if debug_points is not None and not debug_points.empty:
                debug_points.to_file(output_dir / "gateway_points_debug.geojson", driver="GeoJSON")
                print(f"  gateway points debug: {output_dir / 'gateway_points_debug.geojson'}")

            ext_zones = _build_external_gateway_zones_from_boundary_meta(
                gateway_meta=gateway_meta,
                target_epsg=crs_epsg,
                zone_offset_m=float(ext_cfg.get("zone_offset_m", 10.0)),
                zone_size_m=float(ext_cfg.get("zone_size_m", 40.0)),
                start_id=int(ext_cfg.get("synthetic_zone_id_start", 8_000_000_000)),
            )

            if not ext_zones.empty:
                zones = gpd.GeoDataFrame(
                    pd.concat([zones, ext_zones], ignore_index=True),
                    crs=zones.crs,
                )
                print(f"✓ synthetic external gateway zones added: {len(ext_zones)}")

            export_gateway_diagnostics(gateway_meta, output_dir)

            if export_lookup:
                export_gateway_seed_lookup(
                    gateway_meta,
                    export_lookup_path,
                    crs_epsg=crs_epsg,
                )

        if zones.crs is None:
            zones = zones.set_crs(epsg=crs_epsg, allow_override=True)
        if zones.crs.to_epsg() != crs_epsg:
            zones = zones.to_crs(epsg=crs_epsg)

        centroids = calculate_centroids(zones)

        export_map_png(
            project=project,
            zones=zones,
            centroids=centroids,
            model_area_bbox=model_area,
            output_dir=output_dir,
            filename="zones_map_pre_connectors.png",
            title_suffix=" (pre-connectors)",
            dpi=int(zoning_cfg.get("map_dpi", 250)),
            debug_corridors=debug_corridors,
            debug_points=debug_points,
            crs_epsg=crs_epsg,
        )

        _delete_all_connectors_and_reset_centroids(project)
        project.close()

        project = Project()
        project.open(str(project_dir))

        zone_to_centroid = create_centroid_connectors(
            project=project,
            centroids=centroids,
            max_connectors=max_connectors,
            max_distance_m=max_distance_m,
            speed_kmh=connector_speed,
            capacity_vph=connector_capacity,
            lanes=connector_lanes,
            access_penalty_s=connector_penalty,
            gateway_targets=gateway_targets,
            external_speed_kmh=float(ext_cfg.get("external_connector_speed_kmh", 90.0)),
            external_access_penalty_s=float(ext_cfg.get("external_connector_access_penalty_s", 5.0)),
        )

        export_connector_diagnostics(project, zone_to_centroid, output_dir)

        centroids["centroid_node_id"] = centroids["zone_id"].map(
            lambda z: zone_to_centroid.get(int(z), int(z))
        ).astype(int)

        pop_path = Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")) / "zone_population.parquet"
        if pop_path.exists():
            pop_df = pd.read_parquet(pop_path)
            pop_map = dict(zip(pop_df["zone_id"], pop_df["population"]))

            centroids["population"] = centroids.apply(
                lambda r: 0 if _row_is_external(r) else int(pop_map.get(int(r["zone_id"]), 0)),
                axis=1,
            ).astype(int)

            zones["population"] = zones.apply(
                lambda r: 0 if _row_is_external(r) else int(pop_map.get(int(r["zone_id"]), 0)),
                axis=1,
            ).astype(int)

            print(f"  population data loaded: {len(pop_map)} zones, total={sum(pop_map.values()):,}")
            print("  external zones exported with population=0")
        else:
            print(f"  [warn] {pop_path} not found — population not added to centroids")

        output_dir.mkdir(parents=True, exist_ok=True)
        zones.to_file(output_dir / "zones.geojson", driver="GeoJSON")
        centroids.to_file(output_dir / "centroids.geojson", driver="GeoJSON")

        mapping_path = output_dir / "zone_centroid_mapping.json"
        mapping_path.write_text(
            json.dumps({str(k): v for k, v in zone_to_centroid.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"  zone->centroid mapping: {mapping_path}")

        gpd.GeoDataFrame({"geometry": [model_area]}, crs=f"EPSG:{crs_epsg}").to_file(
            output_dir / "model_area.geojson", driver="GeoJSON"
        )

        print(f"✓ exported: {output_dir} (zones={len(zones)})")

        export_map_png(
            project=project,
            zones=zones,
            centroids=centroids,
            model_area_bbox=model_area,
            output_dir=output_dir,
            filename="zones_map.png",
            title_suffix=" (final)",
            dpi=int(zoning_cfg.get("map_dpi", 250)),
            crs_epsg=crs_epsg,
        )

    finally:
        project.close()


if __name__ == "__main__":
    build_zones_and_connectors()
