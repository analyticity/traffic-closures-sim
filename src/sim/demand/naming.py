"""Zone name matching and indexing helpers."""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import geopandas as gpd

from sim._text import norm_name as _norm_name

_GEO_SUFFIXES = re.compile(r"\s+(u|nad|pod|na|ve|pri|při)\s+\S+$", re.IGNORECASE)


def _strip_geo_suffix(name_norm: str) -> str:
    return _GEO_SUFFIXES.sub("", name_norm).strip()


def _build_zone_name_index(
    zones_gdf: gpd.GeoDataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    primary: Dict[str, int] = {}
    stripped: Dict[str, int] = {}

    for _, row in zones_gdf.iterrows():
        zone_id = int(row["zone_id"])
        name_norm = _norm_name(row.get("name", ""))
        if not name_norm:
            continue

        primary.setdefault(name_norm, zone_id)

        stripped_name = _strip_geo_suffix(name_norm)
        if stripped_name != name_norm:
            stripped.setdefault(stripped_name, zone_id)

    return primary, stripped


def _match_zone_id(
    name: str,
    primary: Dict[str, int],
    stripped: Dict[str, int],
) -> Optional[int]:
    import difflib

    name_norm = _norm_name(name)
    if not name_norm:
        return None

    if name_norm in primary:
        return primary[name_norm]

    stripped_name = _strip_geo_suffix(name_norm)
    if stripped_name in primary:
        return primary[stripped_name]

    if name_norm in stripped:
        return stripped[name_norm]

    if stripped_name in stripped:
        return stripped[stripped_name]

    for prefix in ("mesto ", "obec ", "mestys "):
        if name_norm.startswith(prefix):
            rest = name_norm[len(prefix):]
            if rest in primary:
                return primary[rest]

    all_names = list(primary.keys()) + list(stripped.keys())
    for candidate in (name_norm, stripped_name):
        close = difflib.get_close_matches(candidate, all_names, n=1, cutoff=0.85)
        if close:
            match_name = close[0]
            return primary.get(match_name) or stripped.get(match_name)

    return None


def _zone_index(zone_ids) -> Dict[int, int]:
    return {int(z): i for i, z in enumerate(zone_ids)}
