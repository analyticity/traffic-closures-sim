"""OD matrix seed builders: gravity, external-local, external-through."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sim.io_project import pairwise_euclidean
from sim.demand.config import _normalize_named_weights

logger = logging.getLogger(__name__)


def _zero_matrix(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def _build_gravity_seed(
    zone_ids: np.ndarray,
    population: Dict[int, int],
    *,
    trip_rate: float,
    car_share: float,
    occupancy: float,
    beta: float,
    centroids_gdf=None,
    excluded_zone_ids: Optional[set[int]] = None,
    metric_epsg: int,
) -> np.ndarray:
    excluded_zone_ids = excluded_zone_ids or set()
    n = len(zone_ids)

    pop = np.array(
        [
            0.0 if int(z) in excluded_zone_ids else max(population.get(int(z), 0), 0)
            for z in zone_ids
        ],
        dtype=np.float64,
    )

    productions = pop * trip_rate * car_share / max(occupancy, 0.01)
    attractions = pop.copy()

    if centroids_gdf is not None and len(centroids_gdf) >= n:
        imp = pairwise_euclidean(centroids_gdf, zone_ids, metric_epsg=metric_epsg)
    else:
        imp = np.ones((n, n), dtype=np.float64)

    imp = np.maximum(imp, 100.0)
    deterrence = np.exp(-float(beta) * imp)
    np.fill_diagonal(deterrence, 0.0)

    od = productions[:, None] * attractions[None, :] * deterrence
    total_prod = productions.sum()
    total_od = od.sum()

    if total_od > 0 and total_prod > 0:
        od *= total_prod / total_od

    return od


def _build_external_local_seed(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    population: Dict[int, int],
    *,
    total_daily_trips: float,
    corridor_weights: Optional[Dict[str, Any]] = None,
    gateway_link_types: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = _zero_matrix(n)

    if not gateways or total_daily_trips <= 0:
        return od

    gateway_zone_ids = {
        int(zid)
        for gateway_zones in gateways.values()
        for zid, _ in gateway_zones
        if int(zid) in z2i
    }

    internal_zone_ids = [int(z) for z in zone_ids if int(z) not in gateway_zone_ids]
    if not internal_zone_ids:
        return od

    internal_pop = np.array(
        [max(population.get(z, 0), 0) for z in internal_zone_ids],
        dtype=np.float64,
    )
    if internal_pop.sum() <= 0:
        return od
    internal_pop = internal_pop / internal_pop.sum()

    corridor_names = sorted(gateways.keys())
    corridor_weights_norm = _normalize_named_weights(
        corridor_names, corridor_weights, gateway_link_types=gateway_link_types,
    )

    for gateway_name in corridor_names:
        gateway_zones = [(zid, w) for zid, w in gateways[gateway_name] if int(zid) in z2i]
        if not gateway_zones:
            continue

        corridor_total = total_daily_trips * corridor_weights_norm.get(gateway_name, 0.0)
        if corridor_total <= 0:
            continue

        inbound_total = corridor_total * 0.5
        outbound_total = corridor_total * 0.5

        for gateway_zone_id, gateway_weight in gateway_zones:
            gi = z2i[int(gateway_zone_id)]

            for internal_zone_id, share in zip(internal_zone_ids, internal_pop):
                ii = z2i[int(internal_zone_id)]
                if gi == ii:
                    continue

                od[gi, ii] += outbound_total * float(gateway_weight) * float(share)
                od[ii, gi] += inbound_total * float(gateway_weight) * float(share)

    return od


def _estimate_total_daily_trips_from_csd(
    cfg: Dict[str, Any],
    gw_diag_path: Path,
) -> Optional[float]:
    """Estimate external-local daily trips by summing CSD AADT on gateway roads."""
    if not gw_diag_path.exists():
        return None

    try:
        gw_diag = pd.read_csv(gw_diag_path)
    except Exception:
        return None

    if "whitelist_token" not in gw_diag.columns and "matched_ref" not in gw_diag.columns:
        return None

    try:
        from sim.calibration import load_csd
        csd = load_csd(cfg)
    except Exception:
        logger.debug("CSD data unavailable for trip estimation", exc_info=True)
        return None

    if csd.empty or "sil" not in csd.columns or "sv" not in csd.columns:
        return None

    csd["sil"] = csd["sil"].astype(str).str.strip().str.upper()
    csd["sv"] = pd.to_numeric(csd["sv"], errors="coerce").fillna(0)

    def _norm_road(ref: Any) -> str:
        s = str(ref or "").strip().upper()
        s = s.replace("/", "").replace(" ", "").replace("\\", "")
        return s

    gateway_refs: Dict[str, str] = {}
    for _, row in gw_diag.iterrows():
        gw_name = str(row.get("gateway_name", ""))
        token = _norm_road(row.get("whitelist_token") or row.get("matched_ref", ""))
        if token:
            gateway_refs[gw_name] = token

    if not gateway_refs:
        return None

    unique_roads = set(gateway_refs.values())
    total = 0.0
    matched_gateways = 0

    for gw_name, road_code in gateway_refs.items():
        csd_match = csd[csd["sil"] == road_code]
        if csd_match.empty:
            csd_match = csd[csd["sil"].str.replace("M", "", regex=False) == road_code.replace("M", "")]
        if csd_match.empty:
            continue
        mean_aadt = float(csd_match["sv"].mean())
        if mean_aadt > 0:
            total += mean_aadt
            matched_gateways += 1

    if matched_gateways == 0:
        return None

    coverage = matched_gateways / max(len(gateway_refs), 1)
    if coverage < 1.0 and matched_gateways > 0:
        total = total / coverage

    logger.info(
        "Auto-estimated total_daily_trips=%.0f from CSD "
        "(%d/%d gateways matched, %d unique roads)",
        total, matched_gateways, len(gateway_refs), len(unique_roads),
    )
    return total


def _build_external_through_seed(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    *,
    total_daily_trips: float,
    pair_weights: Optional[Dict[tuple, float]] = None,
) -> np.ndarray:
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = _zero_matrix(n)

    if not gateways or total_daily_trips <= 0:
        return od

    gateway_names = sorted(gateways.keys())
    if not gateway_names:
        return od

    if not pair_weights:
        pair_weights = {
            (a, b): 1.0
            for a in gateway_names
            for b in gateway_names
            if a != b
        }

    valid_pairs = {
        (a, b): max(float(w), 0.0)
        for (a, b), w in pair_weights.items()
        if a in gateways and b in gateways and a != b and float(w) > 0
    }

    total_weight = sum(valid_pairs.values())
    if total_weight <= 0:
        return od

    for (a_name, b_name), pair_weight in valid_pairs.items():
        pair_total = total_daily_trips * float(pair_weight) / total_weight

        a_zones = [(zid, w) for zid, w in gateways[a_name] if int(zid) in z2i]
        b_zones = [(zid, w) for zid, w in gateways[b_name] if int(zid) in z2i]
        if not a_zones or not b_zones:
            continue

        for a_zid, a_w in a_zones:
            ai = z2i[int(a_zid)]
            for b_zid, b_w in b_zones:
                bi = z2i[int(b_zid)]
                if ai == bi:
                    continue
                od[ai, bi] += pair_total * float(a_w) * float(b_w)

    return od


def _build_external_through_from_pairs(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    pairs_df: pd.DataFrame,
) -> np.ndarray:
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = _zero_matrix(n)

    if pairs_df.empty or not gateways:
        return od

    for _, row in pairs_df.iterrows():
        gateway_in = str(row["gateway_in"]).strip()
        gateway_out = str(row["gateway_out"]).strip()
        total = float(row.get("vehicles_daily", 0.0))

        if total <= 0 or gateway_in == gateway_out:
            continue
        if gateway_in not in gateways or gateway_out not in gateways:
            continue

        in_members = [(zid, w) for zid, w in gateways[gateway_in] if int(zid) in z2i]
        out_members = [(zid, w) for zid, w in gateways[gateway_out] if int(zid) in z2i]
        if not in_members or not out_members:
            continue

        for in_zid, in_w in in_members:
            ii = z2i[int(in_zid)]
            for out_zid, out_w in out_members:
                oi = z2i[int(out_zid)]
                if ii == oi:
                    continue
                od[ii, oi] += total * float(in_w) * float(out_w)

    return od
