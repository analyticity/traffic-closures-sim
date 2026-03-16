"""
Generic TAZ zoning (priority + AOI-from-network + optional prefilter + connectors + map).

CO JE OPRAVENO:
- AOI už není axis-aligned bbox z min/max X,Y (který v EPSG:5514 často působí „nahnutě“ vůči síti),
  ale počítá se jako ORIENTOVANÝ (rotovaný) čtverec podle hlavní osy sítě (PCA / principal axis).
- Minima/maxima se tedy berou v otočeném souřadném systému (u,v), a výsledný čtverec se vrátí zpět
  do EPSG:5514 jako Polygon (rotovaný rámeček).

POZNÁMKA:
- Funkce filter_zones_centroid_in_bbox() i nadále funguje, protože testuje within() nad geometrií AOI.
  Jen AOI už není "box", ale Polygon (rotovaný).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae import Project
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, box, MultiPoint
from shapely.ops import unary_union

from sim.io_project import load_config


# ----------------------------
# Helpers
# ----------------------------

def _safe_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _fix_polygons(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = g[g.geometry.notna()].copy()
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if g.empty:
        return g
    invalid = ~g.geometry.is_valid
    if invalid.any():
        g.loc[invalid, "geometry"] = g.loc[invalid, "geometry"].buffer(0)
    g = g[~g.geometry.is_empty].copy()
    return g


def _normalize_osmid(v: Any) -> int:
    if isinstance(v, (tuple, list)) and v:
        v = v[0]
    try:
        return int(v)
    except Exception:
        digest = hashlib.sha256(str(v).encode()).hexdigest()
        return int(digest[:15], 16) % 2_000_000_000


def _to_wgs84_point(point, crs_from) -> Any:
    return gpd.GeoSeries([point], crs=crs_from).to_crs("EPSG:4326").iloc[0]


def _guess_crs_from_coords(geoms: gpd.GeoSeries, target_epsg: int) -> str:
    """
    Heuristic:
    - If coordinates look like degrees (|x| <= 180 and |y| <= 90-ish) -> EPSG:4326
    - Otherwise assume target_epsg (e.g., 5514)
    """
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


def _looks_like_degrees(geoms: gpd.GeoSeries) -> bool:
    """
    Stronger "looks like WGS84 degrees" check. We only need to detect the common failure mode:
    lon/lat coordinates incorrectly labeled as EPSG:5514.
    """
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


def _force_to_target_crs(gdf: gpd.GeoDataFrame, target_epsg: int, *, name: str = "gdf") -> gpd.GeoDataFrame:
    """
    CRS-safe conversion into target_epsg with a guard against:
    - CRS missing -> guess
    - CRS says target_epsg but coords look like degrees -> override to EPSG:4326 then convert
    """
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
            f"Overriding to EPSG:4326 then converting."
        )
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(epsg=target_epsg)
        return gdf

    if epsg != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    return gdf


def _cache_meta_path(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".meta.json")


def _write_cache(zones: gpd.GeoDataFrame, cache_file: Path, crs_epsg: int) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.suffix.lower() == ".gpkg":
        zones.to_file(cache_file, layer="zones", driver="GPKG")
    else:
        zones.to_file(cache_file, driver="GeoJSON")

    meta = {"crs_epsg": int(crs_epsg)}
    _cache_meta_path(cache_file).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✓ zones cache saved: {cache_file} (+ meta) ({len(zones)})")


def _read_cache(cache_file: Path, crs_epsg: int) -> Optional[gpd.GeoDataFrame]:
    if not cache_file.exists():
        return None

    if cache_file.suffix.lower() == ".gpkg":
        gdf = gpd.read_file(cache_file, layer="zones")
    else:
        gdf = gpd.read_file(cache_file)

    meta_path = _cache_meta_path(cache_file)
    if gdf.crs is None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        epsg = int(meta.get("crs_epsg", crs_epsg))
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=crs_epsg, allow_override=True)

    if gdf.crs.to_epsg() != crs_epsg:
        gdf = gdf.to_crs(epsg=crs_epsg)

    print(f"✓ zones loaded from cache: {cache_file} ({len(gdf)}) CRS={gdf.crs}")
    return gdf


def _oriented_square_from_points(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    quantile: float = 0.01,
    pad_ratio: float = 0.005,
    make_square: bool = True,
) -> Polygon:
    """
    Oriented AOI based on MINIMUM ROTATED RECTANGLE (MRR) of convex hull
    (more stable than PCA for "city-shaped" networks).

    Steps:
    1) pre-trim XY by quantile to drop long spikes
    2) convex hull -> minimum_rotated_rectangle -> get dominant edge angle
    3) rotate points into that frame, trim by quantile in (u,v)
    4) pad + optional square, rotate back -> Polygon
    """
    if xs.size < 3:
        return box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    q = float(quantile)
    q = max(0.0, min(q, 0.49))

    # --- pre-trim in XY (reduces effect of long branches) ---
    x_lo, x_hi = np.quantile(xs, [q, 1.0 - q])
    y_lo, y_hi = np.quantile(ys, [q, 1.0 - q])
    mask = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)

    pts = np.column_stack([xs, ys])
    pts_core = pts[mask] if mask.sum() >= 50 else pts

    # --- MRR angle from convex hull ---
    hull = MultiPoint(pts_core).convex_hull
    mrr = hull.minimum_rotated_rectangle

    coords = list(mrr.exterior.coords)
    # coords: 5 points (closed ring). Find longest edge.
    best_dx, best_dy, best_len2 = 1.0, 0.0, -1.0
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        l2 = dx * dx + dy * dy
        if l2 > best_len2:
            best_len2 = l2
            best_dx, best_dy = dx, dy

    angle = float(np.arctan2(best_dy, best_dx))  # radians

    # --- rotate all points into (u,v) frame around center ---
    cx, cy = pts_core.mean(axis=0)
    c = float(np.cos(-angle))
    s = float(np.sin(-angle))

    # rotate: [u] = [ c -s ] [x-cx]
    #         [v]   [ s  c ] [y-cy]
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    u = c * dx - s * dy
    v = s * dx + c * dy

    # quantile trim in rotated frame
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

    # rotate back by +angle
    c2 = float(np.cos(angle))
    s2 = float(np.sin(angle))
    corners_xy = []
    for uu, vv in corners_uv:
        x = cx + (c2 * uu - s2 * vv)
        y = cy + (s2 * uu + c2 * vv)
        corners_xy.append((x, y))

    return Polygon(corners_xy)


# ----------------------------
# Drop huge zones per source
# ----------------------------

def _drop_huge_zones_per_source(
    g: gpd.GeoDataFrame,
    *,
    crs_epsg: int,
    enabled: bool,
    rel_factor: float,
    abs_max_km2: Optional[float],
) -> gpd.GeoDataFrame:
    if not enabled or g.empty:
        return g

    g_area = g if (g.crs is not None and g.crs.to_epsg() == crs_epsg) else g.to_crs(epsg=crs_epsg)

    areas = g_area.geometry.area
    med = float(areas.median()) if len(areas) else 0.0
    if med <= 0:
        return g

    thr_rel = med * float(rel_factor)
    keep = areas <= thr_rel

    if abs_max_km2 is not None:
        thr_abs = float(abs_max_km2) * 1_000_000.0
        keep = keep & (areas <= thr_abs)

    removed = int((~keep).sum())
    if removed > 0:
        msg = f"  - drop huge zones: removed={removed} (median={med:.0f} m², rel_thr={thr_rel:.0f} m²"
        if abs_max_km2 is not None:
            msg += f", abs_thr={float(abs_max_km2):.1f} km²"
        msg += ")"
        print(msg)

    return g.loc[keep.values].copy()


# ----------------------------
# Priority de-overlap
# ----------------------------

def remove_overlaps_by_priority(
    zones: gpd.GeoDataFrame,
    *,
    rank_col: str = "source_rank",
    min_area_m2: float = 25.0,
) -> gpd.GeoDataFrame:
    if zones.empty:
        return zones

    zones = zones.sort_values([rank_col, "zone_id"]).copy()

    out_rows: List[Dict[str, Any]] = []
    accepted_geoms: List[Any] = []
    accepted_union = None

    for _, row in zones.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if accepted_union is not None and not accepted_union.is_empty:
            geom = geom.difference(accepted_union)

        if geom is None or geom.is_empty:
            continue
        if float(getattr(geom, "area", 0.0)) < min_area_m2:
            continue

        new_row = dict(row)
        new_row["geometry"] = geom
        out_rows.append(new_row)

        accepted_geoms.append(geom)
        accepted_union = accepted_geoms[0] if len(accepted_geoms) == 1 else unary_union(accepted_geoms)

    out = gpd.GeoDataFrame(out_rows, crs=zones.crs)

    try:
        out = out.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        pass

    return _fix_polygons(out)

def _angle_in_range(angle: float, lo: float, hi: float) -> bool:
    angle = angle % 360.0
    lo = lo % 360.0
    hi = hi % 360.0
    if lo <= hi:
        return lo <= angle <= hi
    return angle >= lo or angle <= hi


def _angular_distance_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _bearing_deg(cx: float, cy: float, x: float, y: float) -> float:
    # 0 = north, 90 = east  (stejná konvence jako v build-demand.py)
    return math.degrees(math.atan2(x - cx, y - cy)) % 360.0


def _build_external_gateway_zones(
    model_area: Any,
    crs_epsg: int,
    gateways_cfg: List[Dict[str, Any]],
    *,
    zone_offset_m: float = 4000.0,
    zone_size_m: float = 800.0,
    start_id: int = 8_000_000_000,
) -> gpd.GeoDataFrame:
    if not gateways_cfg:
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

    center = model_area.representative_point()
    cx, cy = float(center.x), float(center.y)

    boundary_coords = list(model_area.exterior.coords)
    max_r = max(math.hypot(x - cx, y - cy) for x, y in boundary_coords)

    half = zone_size_m / 2.0
    rows = []

    for i, gw in enumerate(gateways_cfg):
        gw_name = str(gw.get("name", f"gw_{i}")).strip()
        ang = math.radians(float(gw.get("direction_deg", 0.0)) % 360.0)

        # 0=north, 90=east
        x = cx + math.sin(ang) * (max_r + zone_offset_m)
        y = cy + math.cos(ang) * (max_r + zone_offset_m)

        geom = translate(
            box(-half, -half, half, half),
            xoff=x,
            yoff=y,
        )

        rows.append({
            "zone_id": start_id + i,
            "name": f"EXT_{gw_name}",
            "source_rank": 999,
            "is_external": 1,
            "gateway_name": gw_name,
            "geometry": geom,
        })

    return gpd.GeoDataFrame(rows, crs=f"EPSG:{crs_epsg}")


def _select_gateway_target_nodes(
    project: Project,
    target_epsg: int,
    model_area: Any,
    gateways_cfg: List[Dict[str, Any]],
    *,
    nodes_per_gateway: int = 2,
    outer_quantile: float = 0.90,
) -> Dict[str, List[int]]:
    road_nids, node_weight = _eligible_road_nodes(project)
    if not road_nids:
        return {}

    nodes_gdf = _network_ref(project, "nodes", target_epsg)
    eligible = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(road_nids)].copy()
    if eligible.empty:
        return {}

    eligible["road_weight"] = eligible["node_id"].map(node_weight).fillna(0.5)

    center = model_area.representative_point()
    cx, cy = float(center.x), float(center.y)

    eligible["_dist_center"] = np.sqrt(
        (eligible.geometry.x - cx) ** 2 + (eligible.geometry.y - cy) ** 2
    )
    eligible["_angle"] = eligible.geometry.apply(
        lambda g: _bearing_deg(cx, cy, float(g.x), float(g.y))
    )

    outer_thr = float(eligible["_dist_center"].quantile(outer_quantile))
    outer = eligible[eligible["_dist_center"] >= outer_thr].copy()
    if outer.empty:
        outer = eligible.copy()

    out: Dict[str, List[int]] = {}

    for gw in gateways_cfg:
        gw_name = str(gw.get("name", "")).strip()
        lo, hi = gw.get("angle_range", [0, 360])
        dir_deg = float(gw.get("direction_deg", 0.0)) % 360.0

        cand = outer[
            outer["_angle"].apply(lambda a: _angle_in_range(float(a), float(lo), float(hi)))
        ].copy()

        if cand.empty:
            cand = outer.copy()

        cand["_ang_err"] = cand["_angle"].apply(
            lambda a: _angular_distance_deg(float(a), dir_deg)
        )

        cand = cand.sort_values(
            ["_ang_err", "road_weight", "_dist_center"],
            ascending=[True, False, False],
        )

        chosen = cand["node_id"].astype(int).head(nodes_per_gateway).tolist()
        out[gw_name] = chosen
        print(f"  gateway targets {gw_name}: {chosen}")

    return out

# ----------------------------
# Download zones (OSM, generic)
# ----------------------------

def download_zones_from_sources(
    sources: List[Dict[str, Any]],
    crs_epsg: int,
    cache_file: Optional[Path],
    area_filter_cfg: Dict[str, Any],
) -> gpd.GeoDataFrame:
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("Chybí osmnx. Doinstaluj: pip install osmnx") from e

    if cache_file is not None:
        cache_file = _safe_path(cache_file)
        cached = _read_cache(cache_file, crs_epsg)
        if cached is not None:
            return cached

    enabled = bool(area_filter_cfg.get("enabled", True))
    rel_factor = float(area_filter_cfg.get("rel_factor", 10.0))
    abs_max_km2 = area_filter_cfg.get("abs_max_km2", None)
    abs_max_km2 = None if abs_max_km2 in ("", "null", "None") else abs_max_km2
    abs_max_km2 = float(abs_max_km2) if abs_max_km2 is not None else None

    parts: List[gpd.GeoDataFrame] = []

    for rank, src in enumerate(sources):
        place = src["place"]
        admin_level = str(src.get("admin_level", "")).strip()
        tags = src.get("tags") or {"boundary": "administrative", "admin_level": admin_level}

        print(f"=== DOWNLOAD zones[{rank}] place={place} tags={tags} ===")
        g = ox.features.features_from_place(place, tags)

        if g is None or len(g) == 0:
            print("⚠ 0 features")
            continue

        g = g.reset_index()
        g = _fix_polygons(g)
        if g.empty:
            print("⚠ no polygonal features after fix")
            continue

        osmid_col = next((c for c in ("osmid", "osm_id", "id") if c in g.columns), None)
        zone_id = g[osmid_col].map(_normalize_osmid) if osmid_col else range(1, len(g) + 1)
        name = g["name"].fillna("zone") if "name" in g.columns else "zone"

        out = gpd.GeoDataFrame(
            {"zone_id": zone_id, "name": name, "source_rank": rank, "geometry": g.geometry},
            crs="EPSG:4326",
        )

        if crs_epsg != 4326:
            out = out.to_crs(epsg=crs_epsg)

        out = out.drop_duplicates(subset=["zone_id"]).copy()

        out = _drop_huge_zones_per_source(
            out,
            crs_epsg=crs_epsg,
            enabled=enabled,
            rel_factor=rel_factor,
            abs_max_km2=abs_max_km2,
        )

        parts.append(out)
        print(f"✓ zones[{rank}] kept: {len(out)}")

    if not parts:
        raise RuntimeError("Nepodařilo se stáhnout žádné zóny z žádného zdroje (sources).")

    zones = gpd.GeoDataFrame(gpd.pd.concat(parts, ignore_index=True), crs=f"EPSG:{crs_epsg}")

    print("=== REMOVE overlaps by priority ===")
    before = len(zones)
    zones = remove_overlaps_by_priority(zones, rank_col="source_rank", min_area_m2=25.0)
    after = len(zones)
    print(f"zones after de-overlap: {before} -> {after}")

    if cache_file is not None:
        _write_cache(zones, cache_file, crs_epsg)

    return zones


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
        g = gpd.GeoDataFrame(df[["geometry"]].copy(), geometry="geometry", crs=getattr(df, "crs", None))
        if "link_type" in df.columns:
            g = g[df["link_type"].astype(str) != "centroid_connector"].copy()
        g = _force_to_target_crs(g, target_epsg, name="network.links")
        return g

    raise ValueError("kind must be 'nodes' or 'links'")


# ----------------------------
# AOI (model area) from network – largest component + quantile trim + padding
# NOW: oriented square (PCA) instead of axis-aligned bbox
# ----------------------------

def build_model_area(
    project: Project,
    target_epsg: int,
    *,
    quantile: float = 0.01,
    pad_ratio: float = 0.005,
) -> Any:
    """
    AOI from the *largest connected component* of the network:
    - robust trimming (quantile) to reduce effect of long branches
    - PCA to find dominant orientation
    - oriented square (rotated Polygon) in target CRS
    """
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
        make_square=True,  # změň na False pokud chceš orientovaný obdélník místo čtverce
    )

    bminx, bminy, bmaxx, bmaxy = map(float, aoi.bounds)
    print(
        f"✓ AOI ORIENTED (EPSG:{target_epsg}) core_nodes={len(core_nodes)} "
        f"q={float(quantile)} pad={float(pad_ratio)}: "
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
    """Keep zones that significantly overlap the AOI.

    A zone is kept if:
    - its intersection area with AOI >= ``min_intersection_share`` of the
      zone's own area, OR
    - its representative_point lies within AOI (backward compatibility).

    This prevents dropping zones whose polygon straddles the AOI boundary
    but whose representative_point falls just outside.
    """
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
    print(f"Zoning: AOI filter kept={n_kept} dropped={len(z) - n_kept} "
          f"(by intersection share: {n_share_only}, min_share={min_intersection_share})")
    return z.loc[keep].reset_index(drop=True)


# ----------------------------
# Centroids INSIDE polygons
# ----------------------------

def calculate_centroids(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep_cols = [
        c for c in ["zone_id", "name", "source_rank", "is_external", "gateway_name"]
        if c in zones.columns
    ]
    pts = zones[keep_cols].copy()
    pts["geometry"] = zones.geometry.representative_point()
    return gpd.GeoDataFrame(pts, crs=zones.crs)


# ----------------------------
# Optional pre-filter (DEBUG)
# ----------------------------

def fast_prefilter_zones_by_centroid_near_network(
    zones: gpd.GeoDataFrame,
    project: Project,
    target_epsg: int,
    reference: str,
    max_distance_m: float,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    print("\n=== FAST pre-filter (representative_point distance in meters) ===")

    if zones.empty:
        c = calculate_centroids(zones)
        return zones, c

    if zones.crs is None:
        zones = zones.set_crs(epsg=target_epsg, allow_override=True)
    if zones.crs.to_epsg() != target_epsg:
        zones = zones.to_crs(epsg=target_epsg)

    centroids = calculate_centroids(zones)

    ref_mode = reference.lower().strip()
    if ref_mode not in {"nodes", "links", "both"}:
        raise ValueError("zoning.pre_filter.reference must be: nodes | links | both")

    def dist_to(kind: str) -> gpd.GeoSeries:
        ref = _network_ref(project, kind, target_epsg)
        u = ref.geometry.unary_union
        return centroids.geometry.distance(u)

    if ref_mode == "nodes":
        dist = dist_to("nodes")
    elif ref_mode == "links":
        dist = dist_to("links")
    else:
        dist = gpd.pd.concat([dist_to("nodes"), dist_to("links")], axis=1).min(axis=1)

    centroids = centroids.copy()
    centroids["dist_m"] = dist

    keep = centroids["dist_m"].notna() & (centroids["dist_m"] <= float(max_distance_m))
    kept_ids = centroids.loc[keep, "zone_id"].tolist()

    zones_f = zones[zones["zone_id"].isin(kept_ids)].copy()
    centroids_f = centroids[centroids["zone_id"].isin(kept_ids)].copy()

    print(f"zones: {len(zones)} -> {len(zones_f)} (ref={reference}, max_distance_m={max_distance_m})")
    return zones_f, centroids_f


# ----------------------------
# Gap-fill fallback zones (guarantee coverage)
# ----------------------------

def add_fallback_zones_for_gaps(
    zones: gpd.GeoDataFrame,
    model_area: Any,
    *,
    crs_epsg: int,
    enabled: bool,
    min_area_m2: float,
    start_id: int = 9_000_000_000,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gaps_out = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

    if not enabled:
        return zones, gaps_out

    if zones.crs is None:
        zones = zones.set_crs(epsg=crs_epsg, allow_override=True)
    if zones.crs.to_epsg() != crs_epsg:
        zones = zones.to_crs(epsg=crs_epsg)

    covered = unary_union(zones.geometry.values.tolist()) if len(zones) else None
    gaps = model_area if (covered is None or covered.is_empty) else model_area.difference(covered)

    if gaps is None or gaps.is_empty:
        print("✓ no gaps in coverage")
        return zones, gaps_out

    gaps_gdf = gpd.GeoDataFrame({"geometry": [gaps]}, crs=f"EPSG:{crs_epsg}")
    try:
        gaps_gdf = gaps_gdf.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        pass
    gaps_gdf = _fix_polygons(gaps_gdf)

    if gaps_gdf.empty:
        print("✓ gaps resolved to empty after fix")
        return zones, gaps_out

    areas = gaps_gdf.geometry.area
    keep = areas >= float(min_area_m2)
    gaps_gdf = gaps_gdf.loc[keep].copy()

    if gaps_gdf.empty:
        print("✓ gaps only tiny slivers -> ignored")
        return zones, gaps_out

    max_existing = int(zones["zone_id"].max()) if ("zone_id" in zones.columns and len(zones)) else 0
    base = max(max_existing + 1, start_id)

    fallback_rank = int(zones["source_rank"].max() + 1) if ("source_rank" in zones.columns and len(zones)) else 999

    fallback = gpd.GeoDataFrame(
        {
            "zone_id": [base + i for i in range(len(gaps_gdf))],
            "name": [f"fallback_gap_{i}" for i in range(len(gaps_gdf))],
            "source_rank": fallback_rank,
            "geometry": gaps_gdf.geometry.values,
        },
        crs=zones.crs,
    )

    out = gpd.GeoDataFrame(gpd.pd.concat([zones, fallback], ignore_index=True), crs=zones.crs)
    out = _fix_polygons(out)

    print(f"✓ added fallback zones for gaps: {len(fallback)}")
    return out, gaps_gdf


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
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib není nainstalovaný -> mapa přeskočena")
        return

    if zones.empty:
        print("⚠ no zones -> mapa přeskočena")
        return

    target_epsg = int(zones.crs.to_epsg()) if zones.crs is not None else 5514

    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        print("⚠ links bez geometry -> mapa přeskočena")
        return

    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))
    if "link_type" in links_gdf.columns:
        links_gdf = links_gdf[links_gdf["link_type"].astype(str) != "centroid_connector"].copy()

    links_gdf = _force_to_target_crs(links_gdf, target_epsg, name="network.links(for map)")

    # Map extent = network extent (align view to network)
    sx, sy, sx2, sy2 = links_gdf.total_bounds
    zx, zy, zx2, zy2 = zones.total_bounds

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

    links_gdf.plot(ax=ax, linewidth=0.8, alpha=1.0, zorder=1)

    # AOI (Polygon/box) - visible
    aoi_gdf.plot(ax=ax, alpha=0.08, zorder=2)
    aoi_gdf.boundary.plot(ax=ax, linewidth=6.0, zorder=10)

    zones.plot(
        ax=ax,
        facecolor="lightblue",
        edgecolor="navy",
        alpha=0.35,
        linewidth=2.0,
        label=f"Zones ({len(zones)})",
        zorder=3,
    )
    centroids.plot(
        ax=ax,
        markersize=30,
        marker="o",
        linewidth=1.0,
        label=f"Centroids ({len(centroids)})",
        zorder=4,
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
    ax.legend(loc="upper right", framealpha=0.95)
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
    """Delete ALL centroid_connector links, clear is_centroid flags,
    and remove orphan centroid nodes that only had connector links."""
    import sqlite3
    db = str(project.project_base_path) + "/project_database.sqlite"
    conn = sqlite3.connect(db)
    try:
        before = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='centroid_connector'"
        ).fetchone()[0]
        conn.execute("DELETE FROM links WHERE link_type='centroid_connector'")
        conn.execute("UPDATE nodes SET is_centroid=0 WHERE is_centroid=1")

        # Delete orphan nodes (nodes with zero links after connector removal)
        # so AequilibraE cannot snap new connector endpoints to them.
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


_EXCLUDED_LINK_TYPES = frozenset({
    "footway", "path", "track", "steps", "cycleway", "pedestrian",
    "corridor", "bridleway", "proposed", "construction", "elevator",
    "service", "rest_area", "services", "traffic_mirror", "virtual",
})

_ROAD_CLASS_WEIGHT = {
    "motorway": 5.0, "motorway_link": 4.0,
    "trunk": 4.0, "trunk_link": 3.0,
    "primary": 3.0, "primary_link": 2.5,
    "secondary": 2.0, "secondary_link": 1.5,
    "tertiary": 1.5, "tertiary_link": 1.2,
    "unclassified": 0.8,
    "residential": 0.5,
    "living_street": 0.3,
    "road": 0.5,
}


def _eligible_road_nodes(project: Project) -> Tuple[set[int], Dict[int, float]]:
    """Return (node_ids, node_weight) for nodes on the main connected car network.

    Rules:
    - exclude clearly non-driveable / irrelevant connector targets
    - require car mode in link.modes
    - keep only nodes from the largest connected car component
    - assign node weight from the highest-class road touching the node
    """
    import networkx as nx

    links = project.network.links.data.copy()
    if links.empty:
        return set(), {}

    excluded = {
        "footway", "path", "track", "steps", "cycleway", "pedestrian",
        "corridor", "bridleway", "proposed", "construction", "elevator",
        "rest_area", "services", "traffic_mirror", "virtual",
        "centroid_connector",
    }

    car = links[
        links["modes"].astype(str).str.contains("c", na=False)
        & ~links["link_type"].astype(str).isin(excluded)
    ].copy()

    if car.empty:
        return set(), {}

    # Largest connected component of the drivable graph
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


def _select_diverse_connectors(
    eligible: gpd.GeoDataFrame,
    centroid_pt,
    max_connectors: int,
    max_distance_m: float,
    pool_size: int = 40,
) -> gpd.GeoDataFrame:
    """Pick connector targets that are nearby and directionally diverse.

    Strategy:
    1. Compute distance to all eligible nodes.
    2. Keep only nodes within max_distance_m.
    3. Score by road importance and proximity.
    4. Prefer angular diversity across sectors.
    5. If nothing is within max_distance_m, fall back to nearest nodes.
    """
    if eligible.empty or max_connectors <= 0:
        return eligible.head(0).copy()

    cand = eligible.copy()
    cand["_dist"] = cand.geometry.distance(centroid_pt)

    # Hard distance filter
    nearby = cand[cand["_dist"] <= float(max_distance_m)].copy()

    # Fallback if nothing is within the threshold
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

    # Prefer important roads, but still strongly prefer closer nodes
    nearby["_score"] = nearby["road_weight"] / np.power(nearby["_dist"], 0.75)

    n_sectors = min(max_connectors, 8)
    sector_size = 360.0 / n_sectors
    nearby["_sector"] = (nearby["_angle"] // sector_size).astype(int)

    chosen_idx: List[int] = []

    # First pass: best candidate per sector
    for sec in range(n_sectors):
        sec_df = nearby[nearby["_sector"] == sec]
        if sec_df.empty:
            continue
        best_idx = sec_df["_score"].idxmax()
        chosen_idx.append(best_idx)
        if len(chosen_idx) >= max_connectors:
            break

    # Second pass: fill remaining slots globally by score
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
) -> Dict[int, int]:
    """
    For each zone, create a small-ID centroid node and connect it to the
    nearest road-network nodes.

    AequilibraE's compressed-graph builder uses node IDs as array indices,
    so centroid IDs must be small (< graph node count).  We allocate
    sequential IDs 1, 2, 3, … that are guaranteed to fit.

    Returns ``{zone_id: centroid_node_id}`` mapping.
    """
    print("\n=== CREATE centroid connectors ===")

    gateway_targets = gateway_targets or {}
    nodes = project.network.nodes.data
    if "geometry" not in nodes.columns or len(nodes) == 0:
        raise RuntimeError("Network nodes missing geometry/empty")

    road_nids, node_weight = _eligible_road_nodes(project)
    existing_nids = set(int(x) for x in nodes["node_id"].values)

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))
    nodes_gdf = _force_to_target_crs(nodes_gdf, int(centroids.crs.to_epsg()),
                                     name="network.nodes(for connector candidates)")

    eligible = nodes_gdf[nodes_gdf["node_id"].isin(road_nids)].copy()
    eligible["road_weight"] = eligible["node_id"].map(node_weight).fillna(0.5)
    print(f"  Eligible road-network nodes: {len(eligible)} (from {len(nodes_gdf)} total, "
          f"excluded {len(nodes_gdf) - len(eligible)} non-car types)")

    if eligible.empty:
        print("  WARNING: no eligible road-network nodes found")
        return {}

    # Allocate small sequential centroid IDs (1, 2, 3, …) avoiding collisions
    next_id = 1
    zone_to_centroid: Dict[int, int] = {}
    for zid in sorted(centroids["zone_id"].astype(int)):
        while next_id in existing_nids:
            next_id += 1
        zone_to_centroid[zid] = next_id
        existing_nids.add(next_id)
        next_id += 1

    print(f"  Centroid IDs: {min(zone_to_centroid.values())}-{max(zone_to_centroid.values())} "
          f"for {len(zone_to_centroid)} zones")

    # Load original WGS84 node geometries (to avoid CRS precision issues)
    nodes_wgs84 = project.network.nodes.data[["node_id", "geometry"]].copy()
    wgs84_lookup = {int(r["node_id"]): r["geometry"] for _, r in nodes_wgs84.iterrows()}

    created = 0
    for _, c in centroids.iterrows():
        zone_id = int(c["zone_id"])
        centroid_pt = c.geometry
        centroid_node_id = zone_to_centroid[zone_id]

        centroid_wgs84 = _to_wgs84_point(centroid_pt, centroids.crs)

        new_node = (project.network.nodes.new_centroid(centroid_node_id)
                    if hasattr(project.network.nodes, "new_centroid")
                    else project.network.nodes.new())
        if not hasattr(project.network.nodes, "new_centroid"):
            new_node.__dict__["node_id"] = centroid_node_id
        new_node.is_centroid = 1
        new_node.geometry = centroid_wgs84
        new_node.save()

        is_external = (
            bool(int(c.get("is_external", 0))) if pd.notna(c.get("is_external", 0)) else False
        )
        gateway_name = str(c.get("gateway_name", "") or "").strip()

        if is_external and gateway_name in gateway_targets:
            target_ids = set(int(x) for x in gateway_targets[gateway_name])
            cand = eligible[eligible["node_id"].astype(int).isin(target_ids)].copy()
            cand["dist"] = cand.geometry.distance(centroid_pt)
            cand = cand.sort_values(["dist", "road_weight"], ascending=[True, False]).head(
                len(target_ids)
            )
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

            # Use the target node's EXACT WGS84 geometry from the DB
            # (no CRS conversion — avoids precision loss that causes
            # AequilibraE to create a new node instead of matching the target)
            target_geom = wgs84_lookup.get(target_node_id)
            if target_geom is None:
                continue

            link = project.network.links.new()
            link.__dict__["a_node"] = centroid_node_id
            link.__dict__["b_node"] = target_node_id
            link.direction = 0
            link.modes = "c"
            link.link_type = "centroid_connector"

            link.distance = dist_m
            link.speed_ab = speed_kmh
            link.speed_ba = speed_kmh
            link.capacity_ab = capacity_vph
            link.capacity_ba = capacity_vph
            link.lanes_ab = lanes
            link.lanes_ba = lanes

            time_s = (dist_m / 1000.0) / speed_kmh * 3600.0 + access_penalty_s
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
    """Export CSV with per-connector info: zone, node, distance, link_type, penalty."""
    import sqlite3
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


# ----------------------------
# Main
# ----------------------------

def build_zones_and_connectors(config_path: str | Path = "config/sim.yaml") -> None:
    print("Zoning: loading config...")
    cfg = load_config(config_path)

    project_dir = Path(cfg["project_path"])

    crs_epsg = int(cfg.get("crs_epsg", 5514))
    zoning_cfg = cfg.get("zoning", {}) or {}

    sources = zoning_cfg.get("sources")
    if not sources:
        raise ValueError("config.zoning.sources must be set")

    area_filter_cfg = zoning_cfg.get("zone_area_filter", {}) or {}
    cache_file = zoning_cfg.get("cache_file")
    output_dir = Path(zoning_cfg.get("output_dir", "outputs/baseline/zones"))

    max_connectors = int(zoning_cfg.get("max_connectors", 4))
    max_distance_m = float(zoning_cfg.get("max_distance_m", 5000.0))
    connector_speed = float(zoning_cfg.get("connector_speed_kmh", 30.0))
    connector_capacity = float(zoning_cfg.get("connector_capacity_vph", 5000.0))
    connector_lanes = int(zoning_cfg.get("connector_lanes", 4))
    connector_penalty = float(zoning_cfg.get("connector_access_penalty_s", 120.0))

    print("Zoning: opening project...")
    project = Project()
    project.open(str(project_dir))

    try:
        # 0) AOI from network (ORIENTED, CRS-safe)
        model_area = build_model_area(project, crs_epsg)

        # 1) zones from OSM (cached)
        print("Zoning: loading zones (cache or download)...")
        zones = download_zones_from_sources(
            sources=sources,
            crs_epsg=crs_epsg,
            cache_file=Path(cache_file) if cache_file else None,
            area_filter_cfg=area_filter_cfg,
        )

        # 2) filter by AOI
        zones = filter_zones_centroid_in_bbox(zones, model_area, crs_epsg)
        print(f"✓ zones with representative_point in AOI: {len(zones)}")

        # 3) de-overlap again
        zones = remove_overlaps_by_priority(zones, rank_col="source_rank", min_area_m2=25.0)
        print(f"✓ zones after priority de-overlap: {len(zones)}")

        gateway_targets: Dict[str, List[int]] = {}

        ext_cfg = zoning_cfg.get("external_gateways", {}) or {}
        if bool(ext_cfg.get("enabled", False)):
            gw_list = cfg.get("demand", {}).get("gateways", []) or []

            ext_zones = _build_external_gateway_zones(
                model_area,
                crs_epsg,
                gw_list,
                zone_offset_m=float(ext_cfg.get("zone_offset_m", 4000.0)),
                zone_size_m=float(ext_cfg.get("zone_size_m", 800.0)),
            )

            if not ext_zones.empty:
                zones = gpd.GeoDataFrame(
                    pd.concat([zones, ext_zones], ignore_index=True),
                    crs=zones.crs,
                )
                print(f"✓ synthetic external gateway zones added: {len(ext_zones)}")

                gateway_targets = _select_gateway_target_nodes(
                    project,
                    crs_epsg,
                    model_area,
                    gw_list,
                    nodes_per_gateway=int(ext_cfg.get("connectors_per_gateway", 2)),
                    outer_quantile=float(ext_cfg.get("candidate_outer_quantile", 0.90)),
                )

        # 4) centroids inside
        if zones.crs is None:
            zones = zones.set_crs(epsg=crs_epsg, allow_override=True)
        if zones.crs.to_epsg() != crs_epsg:
            zones = zones.to_crs(epsg=crs_epsg)

        centroids = calculate_centroids(zones)

        # 6.5) EARLY DEBUG MAP (before connectors)
        export_map_png(
            project=project,
            zones=zones,
            centroids=centroids,
            model_area_bbox=model_area,
            output_dir=output_dir,
            filename="zones_map_pre_connectors.png",
            title_suffix=" (pre-connectors)",
            dpi=int(zoning_cfg.get("map_dpi", 250)),
        )

        # 7) connectors
        #    If running on a fresh project (after `clean`), there are no old
        #    connectors.  If re-running, we close+reopen to flush AequilibraE's
        #    in-memory cache so link.save() doesn't snap to stale nodes.
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
        )

        # Connector diagnostics
        export_connector_diagnostics(project, zone_to_centroid, output_dir)

        # Add centroid_node_id to centroids GeoDataFrame
        centroids["centroid_node_id"] = centroids["zone_id"].map(
            lambda z: zone_to_centroid.get(int(z), int(z))
        ).astype(int)

        # Add population data if available
        pop_path = Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")) / "zone_population.parquet"
        if pop_path.exists():
            pop_df = pd.read_parquet(pop_path)
            pop_map = dict(zip(pop_df["zone_id"], pop_df["population"]))
            centroids["population"] = centroids["zone_id"].map(
                lambda z: pop_map.get(int(z), 0)
            ).astype(int)
            zones["population"] = zones["zone_id"].map(
                lambda z: pop_map.get(int(z), 0)
            ).astype(int)
            print(f"  population data loaded: {len(pop_map)} zones, total={sum(pop_map.values()):,}")
        else:
            print(f"  [warn] {pop_path} not found — population not added to centroids")

        # 8) export final
        output_dir.mkdir(parents=True, exist_ok=True)
        zones.to_file(output_dir / "zones.geojson", driver="GeoJSON")
        centroids.to_file(output_dir / "centroids.geojson", driver="GeoJSON")

        import json as _json
        mapping_path = output_dir / "zone_centroid_mapping.json"
        mapping_path.write_text(_json.dumps(
            {str(k): v for k, v in zone_to_centroid.items()}, indent=2
        ), encoding="utf-8")
        print(f"  zone->centroid mapping: {mapping_path}")

        # AOI and gaps
        gpd.GeoDataFrame({"geometry": [model_area]}, crs=f"EPSG:{crs_epsg}").to_file(
            output_dir / "model_area.geojson", driver="GeoJSON"
        )

        print(f"✓ exported: {output_dir} (zones={len(zones)})")

        # final map
        export_map_png(
            project=project,
            zones=zones,
            centroids=centroids,
            model_area_bbox=model_area,
            output_dir=output_dir,
            filename="zones_map.png",
            title_suffix=" (final)",
            dpi=int(zoning_cfg.get("map_dpi", 250)),
        )

    finally:
        project.close()


if __name__ == "__main__":
    build_zones_and_connectors()
