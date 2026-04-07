"""Shared helpers for zone geometry and overlap handling (used by OSM and file loaders)."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


def fix_polygons(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = g[g.geometry.notna()].copy()
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if g.empty:
        return g
    invalid = ~g.geometry.is_valid
    if invalid.any():
        g.loc[invalid, "geometry"] = g.loc[invalid, "geometry"].buffer(0)
    g = g[~g.geometry.is_empty].copy()
    return g


def normalize_osmid(v: Any) -> int:
    if isinstance(v, (tuple, list)) and v:
        v = v[0]
    try:
        return int(v)
    except Exception:
        digest = hashlib.sha256(str(v).encode()).hexdigest()
        return int(digest[:15], 16) % 2_000_000_000


def drop_huge_zones_per_source(
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

    return fix_polygons(out)
