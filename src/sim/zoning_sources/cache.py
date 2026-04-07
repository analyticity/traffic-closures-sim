"""Optional on-disk cache for merged zones (after all sources + de-overlap)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd


def safe_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def cache_meta_path(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".meta.json")


def write_zones_cache(zones: gpd.GeoDataFrame, cache_file: Path, crs_epsg: int) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.suffix.lower() == ".gpkg":
        zones.to_file(cache_file, layer="zones", driver="GPKG")
    else:
        zones.to_file(cache_file, driver="GeoJSON")

    meta = {"crs_epsg": int(crs_epsg)}
    cache_meta_path(cache_file).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✓ zones cache saved: {cache_file} (+ meta) ({len(zones)})")


def read_zones_cache(cache_file: Path, crs_epsg: int) -> Optional[gpd.GeoDataFrame]:
    if not cache_file.exists():
        return None

    if cache_file.suffix.lower() == ".gpkg":
        gdf = gpd.read_file(cache_file, layer="zones")
    else:
        gdf = gpd.read_file(cache_file)

    meta_path = cache_meta_path(cache_file)
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
