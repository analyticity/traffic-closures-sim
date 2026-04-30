"""
Modular zone sources: OSM (osmnx) and local vector files.

Orchestrator merges all entries in ``zoning.sources``, de-overlaps by ``source_rank``,
and optionally writes ``zoning.cache_file``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd

from sim.zoning.sources.cache import read_zones_cache, safe_path, write_zones_cache
from sim.zoning.sources.file import load_zones_file_source
from sim.zoning.sources.osm import load_zones_osm_source
from sim.zoning.sources.shared import remove_overlaps_by_priority

logger = logging.getLogger(__name__)


def _loader_kind(src: Dict[str, Any]) -> str:
    t = str(src.get("type", "")).strip().lower()
    if t == "file":
        return "file"
    if t == "osm":
        return "osm"
    if src.get("path") or src.get("file"):
        return "file"
    if src.get("place"):
        return "osm"
    raise ValueError(
        "Each zoning.sources entry needs type: osm|file, or keys 'place' (OSM), "
        "or 'path'/'file' (vector dataset)."
    )


def load_zones_from_sources(
    sources: List[Dict[str, Any]],
    crs_epsg: int,
    cache_file: Optional[Path],
    area_filter_cfg: Dict[str, Any],
) -> gpd.GeoDataFrame:
    if cache_file is not None:
        cache_file = safe_path(cache_file)
        cached = read_zones_cache(cache_file, crs_epsg)
        if cached is not None:
            return cached

    parts: List[gpd.GeoDataFrame] = []

    for rank, src in enumerate(sources):
        if not isinstance(src, dict):
            raise TypeError(f"zoning.sources[{rank}] must be a mapping, got {type(src).__name__}")
        kind = _loader_kind(src)
        if kind == "osm":
            part = load_zones_osm_source(src, rank, crs_epsg, area_filter_cfg)
        else:
            part = load_zones_file_source(src, rank, crs_epsg, area_filter_cfg)
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError("No zones loaded from any zoning.sources entry.")

    zones = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=f"EPSG:{crs_epsg}")

    logger.info("Remove overlaps by priority")
    before = len(zones)
    zones = remove_overlaps_by_priority(zones, rank_col="source_rank", min_area_m2=25.0)
    after = len(zones)
    logger.info("zones after de-overlap: %d -> %d", before, after)

    if cache_file is not None:
        write_zones_cache(zones, cache_file, crs_epsg)

    return zones
