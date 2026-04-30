"""Core-city hub group builder."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim._text import norm_name as _norm_name
from sim.demand.naming import _match_zone_id

logger = logging.getLogger(__name__)


def _build_hub_group(
    zones_gdf,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    csv_place_names: set[str],
    *,
    hub_name: str,
    hub_source_rank: Optional[int],
    zone_population: Optional[Dict[int, int]] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    """Build a synthetic 'hub' group for the main multi-zone core city."""
    hub_norm = _norm_name(hub_name)
    if not hub_norm:
        return {}

    csv_norm_names = {_norm_name(x) for x in csv_place_names}
    if hub_norm not in csv_norm_names:
        return {}

    claimed: set[int] = set()
    for place in csv_place_names:
        if _norm_name(place) == hub_norm:
            continue
        zid = _match_zone_id(place, primary, stripped)
        if zid is not None:
            claimed.add(zid)

    internal = zones_gdf[zones_gdf.get("is_external", 0).fillna(0).astype(int) == 0].copy()
    if internal.empty:
        return {}

    if hub_source_rank is None:
        hub_source_rank = int(internal["source_rank"].min()) if "source_rank" in internal.columns else 0

    use_population = bool(zone_population)
    members: list[Tuple[int, float]] = []

    for _, row in internal.iterrows():
        zone_id = int(row["zone_id"])
        if zone_id in claimed:
            continue
        if int(row.get("source_rank", -1)) != int(hub_source_rank):
            continue

        if use_population:
            weight = max(float(zone_population.get(zone_id, 1)), 1.0)
        else:
            weight = max(float(row.geometry.area), 1.0)

        members.append((zone_id, weight))

    if len(members) < 2:
        return {}

    ids, weights = zip(*members)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    method = "population" if use_population else "area"
    logger.info("Hub group %r: %d zones, weighted by %s", hub_name, len(members), method)

    return {hub_norm: [(zid, float(wi)) for zid, wi in zip(ids, w)]}
