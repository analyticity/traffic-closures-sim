"""Geometry and CRS helpers for the zoning pipeline."""
from __future__ import annotations

import logging
import math
from typing import Any, List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, box

from sim.network.crs import guess_crs_from_coords

logger = logging.getLogger(__name__)


def to_wgs84_point(point: Any, crs_from: Any) -> Any:
    """Reproject a single point to WGS-84."""
    return gpd.GeoSeries([point], crs=crs_from).to_crs("EPSG:4326").iloc[0]


def force_to_target_crs(
    gdf: gpd.GeoDataFrame,
    target_epsg: int,
    *,
    name: str = "gdf",
) -> gpd.GeoDataFrame:
    """Ensure *gdf* is in *target_epsg*, guessing / fixing CRS when needed.

    Handles three edge-cases that the simpler ``sim.network.crs.as_gdf``
    does not cover:

    * Empty GeoDataFrames (set CRS without reprojection).
    * Missing CRS (heuristic guess via ``guess_crs_from_coords``).
    * CRS claims to be *target_epsg* but coordinates look like geographic
      degrees -- re-set to EPSG:4326 and reproject.
    """
    if gdf.empty:
        if gdf.crs is None:
            return gdf.set_crs(epsg=target_epsg, allow_override=True)
        if gdf.crs.to_epsg() != target_epsg:
            return gdf.to_crs(epsg=target_epsg)
        return gdf

    if gdf.crs is None:
        guessed = guess_crs_from_coords(gdf.geometry, fallback_epsg=target_epsg)
        gdf = gdf.set_crs(guessed, allow_override=True)

    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None

    if epsg == target_epsg and guess_crs_from_coords(gdf.geometry, fallback_epsg=target_epsg) == "EPSG:4326":
        logger.warning(
            "CRS sanity-fix: %s looks like degrees but CRS=%s. "
            "Treating as EPSG:4326 then reprojecting.",
            name,
            target_epsg,
        )
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(epsg=target_epsg)
        return gdf

    if epsg != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    return gdf


def axis_aligned_square_from_points(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    quantile: float = 0.01,
    pad_ratio: float = 0.005,
    make_square: bool = True,
) -> Polygon:
    """Axis-aligned bounding square/rectangle from point cloud with quantile trim."""
    if xs.size == 0:
        raise ValueError("Empty point array for AOI computation")

    q = float(max(0.0, min(float(quantile), 0.49)))

    x_lo, x_hi = float(np.quantile(xs, q)), float(np.quantile(xs, 1.0 - q))
    y_lo, y_hi = float(np.quantile(ys, q)), float(np.quantile(ys, 1.0 - q))

    w = x_hi - x_lo
    h = y_hi - y_lo
    px = max(w * float(pad_ratio), 0.0)
    py = max(h * float(pad_ratio), 0.0)

    x_lo -= px
    x_hi += px
    y_lo -= py
    y_hi += py

    if make_square:
        side = max(x_hi - x_lo, y_hi - y_lo)
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        half = 0.5 * side
        x_lo, x_hi = cx - half, cx + half
        y_lo, y_hi = cy - half, cy + half

    return box(x_lo, y_lo, x_hi, y_hi)


def safe_line_midpoint(geom: Any) -> Point:
    """Representative midpoint of a LineString / MultiLineString / any geometry."""
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


from sim._geo import bearing_deg  # noqa: F401  # canonical shared implementation


def normalize_vector(dx: float, dy: float) -> Tuple[float, float]:
    """Return unit vector; defaults to (1, 0) for zero-length input."""
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return 1.0, 0.0
    return dx / norm, dy / norm


def point_is_too_close(pt: Point, occupied_points: List[Any], min_sep_m: float = 1.0) -> bool:
    """True when *pt* is within *min_sep_m* of any point in *occupied_points*."""
    for op in occupied_points:
        try:
            if pt.distance(op) < min_sep_m:
                return True
        except Exception:
            continue
    return False


def circular_mean_ring_pos(values: List[float], ring_length: float) -> float:
    """Circular mean of positions on a closed ring of given length."""
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
