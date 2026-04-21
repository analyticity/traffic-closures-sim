"""Load TAZ polygons from OpenStreetMap via osmnx."""
from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd

from sim.zoning_sources.shared import (
    drop_huge_zones_per_source,
    fix_polygons,
    normalize_osmid,
)


def load_zones_osm_source(
    src: Dict[str, Any],
    rank: int,
    crs_epsg: int,
    area_filter_cfg: Dict[str, Any],
) -> gpd.GeoDataFrame:
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("osmnx is required for OSM zone sources: pip install osmnx") from e

    place = src.get("place")
    if not place:
        raise ValueError("OSM zone source requires 'place'")

    admin_level = str(src.get("admin_level", "")).strip()
    tags = src.get("tags") or {"boundary": "administrative", "admin_level": admin_level}

    enabled = bool(area_filter_cfg.get("enabled", True))
    rel_factor = float(area_filter_cfg.get("rel_factor", 10.0))
    abs_max_km2 = area_filter_cfg.get("abs_max_km2", None)
    abs_max_km2 = None if abs_max_km2 in ("", "null", "None") else abs_max_km2
    abs_max_km2 = float(abs_max_km2) if abs_max_km2 is not None else None

    print(f"=== OSM zones[{rank}] place={place} tags={tags} ===")
    g = ox.features.features_from_place(place, tags)

    if g is None or len(g) == 0:
        print("⚠ 0 features")
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

    g = g.reset_index()
    g = fix_polygons(g)
    if g.empty:
        print("⚠ no polygonal features after fix")
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

    osmid_col = next((c for c in ("osmid", "osm_id", "id") if c in g.columns), None)
    zone_id = g[osmid_col].map(normalize_osmid) if osmid_col else range(1, len(g) + 1)
    name = g["name"].fillna("zone") if "name" in g.columns else "zone"

    out = gpd.GeoDataFrame(
        {
            "zone_id": zone_id,
            "name": name,
            "source_rank": rank,
            "source_place": place,
            "geometry": g.geometry,
        },
        crs="EPSG:4326",
    )

    if crs_epsg != 4326:
        out = out.to_crs(epsg=crs_epsg)

    out = out.drop_duplicates(subset=["zone_id"]).copy()

    out = drop_huge_zones_per_source(
        out,
        crs_epsg=crs_epsg,
        enabled=enabled,
        rel_factor=rel_factor,
        abs_max_km2=abs_max_km2,
    )

    print(f"✓ zones[{rank}] OSM kept: {len(out)}")
    return out
