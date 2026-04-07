"""Load TAZ polygons from GeoPackage, GeoJSON, or other formats readable by geopandas."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd

from sim.zoning_sources.shared import drop_huge_zones_per_source, fix_polygons, normalize_osmid


def _resolve_path(src: Dict[str, Any]) -> Path:
    p = src.get("path") or src.get("file")
    if not p:
        raise ValueError("file zone source requires 'path' or 'file'")
    return Path(p)


def load_zones_file_source(
    src: Dict[str, Any],
    rank: int,
    crs_epsg: int,
    area_filter_cfg: Dict[str, Any],
) -> gpd.GeoDataFrame:
    path = _resolve_path(src)
    layer = src.get("layer")
    id_col = str(src.get("id_column", "") or "").strip() or None
    name_col = str(src.get("name_column", "") or "").strip() or None

    enabled = bool(area_filter_cfg.get("enabled", True))
    rel_factor = float(area_filter_cfg.get("rel_factor", 10.0))
    abs_max_km2 = area_filter_cfg.get("abs_max_km2", None)
    abs_max_km2 = None if abs_max_km2 in ("", "null", "None") else abs_max_km2
    abs_max_km2 = float(abs_max_km2) if abs_max_km2 is not None else None

    print(f"=== FILE zones[{rank}] path={path} layer={layer!r} ===")
    if not path.is_file():
        raise FileNotFoundError(f"Zone file source not found: {path}")

    kwargs: Dict[str, Any] = {}
    if layer:
        kwargs["layer"] = layer

    g = gpd.read_file(path, **kwargs)
    if g is None or g.empty:
        print("⚠ 0 rows")
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

    g = fix_polygons(g)
    if g.empty:
        print("⚠ no polygonal features after fix")
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

    id_actual: Optional[str] = id_col
    if not id_actual:
        id_actual = next((c for c in ("zone_id", "osmid", "osm_id", "id") if c in g.columns), None)

    if id_actual and id_actual in g.columns:
        zone_id = g[id_actual].map(normalize_osmid)
    else:
        zone_id = range(1, len(g) + 1)

    name_actual: Optional[str] = name_col
    if not name_actual:
        name_actual = "name" if "name" in g.columns else None

    if name_actual and name_actual in g.columns:
        name = g[name_actual].fillna("zone")
    else:
        name = pd.Series(["zone"] * len(g), index=g.index)

    out = gpd.GeoDataFrame(
        {"zone_id": zone_id, "name": name, "source_rank": rank, "geometry": g.geometry},
        crs=g.crs,
    )

    if out.crs is None:
        out = out.set_crs(epsg=4326, allow_override=True)

    epsg = out.crs.to_epsg() if out.crs is not None else None
    if epsg is None or epsg != crs_epsg:
        out = out.to_crs(epsg=crs_epsg)

    out = out.drop_duplicates(subset=["zone_id"]).copy()

    out = drop_huge_zones_per_source(
        out,
        crs_epsg=crs_epsg,
        enabled=enabled,
        rel_factor=rel_factor,
        abs_max_km2=abs_max_km2,
    )

    print(f"✓ zones[{rank}] file kept: {len(out)}")
    return out
