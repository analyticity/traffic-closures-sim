"""CRS detection, bbox computation, and coordinate-system conversion helpers."""
from __future__ import annotations

import logging
from typing import Iterable, Tuple

import geopandas as gpd
from aequilibrae import Project
from shapely.geometry import box

logger = logging.getLogger(__name__)

from sim._metrics import MAJOR_ROAD_TYPES as _CORRIDOR_LINK_TYPES  # same set, shared


# ---------------------------------------------------------------------------
# CRS heuristics
# ---------------------------------------------------------------------------

def guess_crs_from_coords(geoms: gpd.GeoSeries, fallback_epsg: int = 5514) -> str:
    """Heuristic: WGS-84 when coordinates look like degrees, else *fallback_epsg*."""
    s = geoms.dropna()
    if s.empty:
        return f"EPSG:{fallback_epsg}"

    s = s.iloc[:200]
    xs: list[float] = []
    ys: list[float] = []
    for g in s:
        try:
            c = g.centroid
            xs.append(float(c.x))
            ys.append(float(c.y))
        except Exception:
            continue

    if not xs or not ys:
        return f"EPSG:{fallback_epsg}"

    if max(abs(x) for x in xs) <= 180.0 and max(abs(y) for y in ys) <= 90.0:
        return "EPSG:4326"
    return f"EPSG:{fallback_epsg}"


def as_gdf(df: gpd.GeoDataFrame, crs_hint_epsg: int) -> gpd.GeoDataFrame:
    """Ensure *df* is a GeoDataFrame with a CRS (guessed if missing)."""
    g = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None))
    if g.crs is None:
        guessed = guess_crs_from_coords(g.geometry, fallback_epsg=crs_hint_epsg)
        g = g.set_crs(guessed, allow_override=True)
    return g


# ---------------------------------------------------------------------------
# Project-level bbox helpers
# ---------------------------------------------------------------------------

def network_links_gdf_with_crs(project: Project, crs_epsg_hint: int) -> gpd.GeoDataFrame:
    """Return project links as a CRS-aware GeoDataFrame."""
    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        raise RuntimeError("Network links missing geometry/empty")
    return as_gdf(links, crs_epsg_hint)


def network_bbox_wgs84_from_project(
    project: Project,
    crs_epsg_hint: int,
    *,
    pad_ratio: float = 0.005,
    min_pad_deg: float = 0.0015,
) -> Tuple[float, float, float, float]:
    """Current network extent in WGS-84, with small padding."""
    gdf = network_links_gdf_with_crs(project, crs_epsg_hint)
    if gdf.crs is None:
        raise RuntimeError("Could not determine CRS of project links")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    minx, miny, maxx, maxy = map(float, gdf.total_bounds)
    dx = max(maxx - minx, 0.0)
    dy = max(maxy - miny, 0.0)
    padx = max(dx * pad_ratio, min_pad_deg)
    pady = max(dy * pad_ratio, min_pad_deg)
    return (minx - padx, miny - pady, maxx + padx, maxy + pady)


def native_bbox_from_wgs84_bbox(
    project: Project,
    bbox_wgs84: Iterable[float],
    crs_epsg_hint: int,
) -> Tuple[float, float, float, float]:
    """Convert a WGS-84 bbox to the project's native CRS."""
    west, south, east, north = [float(x) for x in bbox_wgs84]
    bbox_geom = box(west, south, east, north)

    links_gdf = network_links_gdf_with_crs(project, crs_epsg_hint)
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs="EPSG:4326")

    if links_gdf.crs is not None and links_gdf.crs.to_epsg() != 4326:
        bbox_gdf = bbox_gdf.to_crs(links_gdf.crs)

    minx, miny, maxx, maxy = map(float, bbox_gdf.total_bounds)
    if not (minx < maxx and miny < maxy):
        raise RuntimeError(f"Invalid converted native bbox: {(minx, miny, maxx, maxy)}")
    return minx, miny, maxx, maxy


def compute_bbox_from_links_raw(project: Project) -> Tuple[float, float, float, float]:
    """Bbox from link geometries in their native (DB) coordinates."""
    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        raise RuntimeError("Network links missing geometry/empty")

    minx, miny, maxx, maxy = map(float, links.total_bounds)
    if not (minx < maxx and miny < maxy):
        raise RuntimeError(f"Invalid link bounds: {(minx, miny, maxx, maxy)}")
    return minx, miny, maxx, maxy


def compute_urban_trim_bbox(
    project: Project,
    *,
    pad_ratio: float = 0.02,
    quantile: float = 0.01,
) -> Tuple[float, float, float, float]:
    """Bbox of the "urban core" -- nodes touching at least one non-corridor link.

    Long highway-only segments are excluded so they don't inflate the area.
    A quantile trim + padding is applied.
    """
    import numpy as np

    links = project.network.links.data
    nodes = project.network.nodes.data
    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))

    if "link_type" in links_gdf.columns:
        local = links_gdf[
            ~links_gdf["link_type"].astype(str).str.strip().str.lower().isin(_CORRIDOR_LINK_TYPES)
        ]
        urban_ids = set(local["a_node"].astype(int)) | set(local["b_node"].astype(int))
    else:
        urban_ids = set(nodes_gdf["node_id"].astype(int))

    urban = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(urban_ids)]
    if len(urban) < 50:
        urban = nodes_gdf

    xs = urban.geometry.x.to_numpy(dtype=float)
    ys = urban.geometry.y.to_numpy(dtype=float)

    if quantile > 0:
        lx, hx = np.quantile(xs, quantile), np.quantile(xs, 1 - quantile)
        ly, hy = np.quantile(ys, quantile), np.quantile(ys, 1 - quantile)
    else:
        lx, hx = xs.min(), xs.max()
        ly, hy = ys.min(), ys.max()

    dx = (hx - lx) * pad_ratio
    dy = (hy - ly) * pad_ratio
    return (lx - dx, ly - dy, hx + dx, hy + dy)
