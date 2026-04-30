"""Gateway zone loading, external lookup, and through-pair helpers."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sim._text import norm_name as _norm_name

if TYPE_CHECKING:
    import geopandas as gpd
    from sim.demand.config import DemandBuildCfg

logger = logging.getLogger(__name__)


def _load_gateways(zones_gdf: gpd.GeoDataFrame) -> Dict[str, List[Tuple[int, float]]]:
    """Load gateway corridors from synthetic external zones created by zoning."""
    if "is_external" not in zones_gdf.columns:
        return {}

    external = zones_gdf[zones_gdf["is_external"].fillna(0).astype(int) == 1].copy()
    if external.empty:
        return {}

    grouped: Dict[str, List[int]] = {}
    for _, row in external.iterrows():
        gateway_name = str(row.get("gateway_name", "")).strip()
        if not gateway_name:
            name = str(row.get("name", "")).strip()
            gateway_name = name.replace("EXT_", "", 1) if name.startswith("EXT_") else name

        if not gateway_name:
            continue

        grouped.setdefault(gateway_name, []).append(int(row["zone_id"]))

    gateways: Dict[str, List[Tuple[int, float]]] = {}
    for gateway_name, zone_ids in grouped.items():
        if not zone_ids:
            continue
        weight = 1.0 / len(zone_ids)
        gateways[gateway_name] = [(zid, weight) for zid in zone_ids]

    if gateways:
        logger.info("Using synthetic external gateway zones: %d corridors", len(gateways))
        for name, members in gateways.items():
            logger.debug("%s: %s", name, [z for z, _ in members])

    return gateways


def _assign_external_to_gateway_fallback(
    place_name: str,
    gateways: Dict[str, List[Tuple[int, float]]],
) -> List[Tuple[int, float]]:
    """Legacy deterministic fallback when no supernetwork lookup exists."""
    if not gateways:
        return []

    gateway_names = sorted(gateways.keys())
    stable = hashlib.blake2b(place_name.encode("utf-8"), digest_size=8).hexdigest()
    idx = int(stable, 16) % len(gateway_names)
    return gateways[gateway_names[idx]]


def _preflight_external_inputs(
    bcfg: DemandBuildCfg,
    gateways: Dict[str, List[Tuple[int, float]]],
) -> None:
    if not bcfg.external.enabled:
        return

    if not gateways:
        raise RuntimeError(
            "External processing is enabled, but no gateway zones were found. "
            "Run build-zones and verify external gateways configuration."
        )

    lookup_path = bcfg.external.gateway_lookup_path
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Missing external gateway lookup: {lookup_path}. "
            "Run build-supernetwork before build-demand."
        )

    lookup_df = pd.read_parquet(lookup_path)
    if lookup_df.empty:
        raise RuntimeError(
            f"External gateway lookup is empty: {lookup_path}. "
            "Rebuild supernetwork and check centroid/gateway mapping coverage."
        )

    required_lookup = {"gateway_name"}
    missing_lookup = required_lookup - set(lookup_df.columns)
    has_place = ("place_name" in lookup_df.columns) or ("place_name_norm" in lookup_df.columns)
    if missing_lookup or not has_place:
        raise RuntimeError(
            f"External gateway lookup schema mismatch at {lookup_path}. "
            f"Missing required columns: {sorted(missing_lookup)}; "
            "expected place_name or place_name_norm."
        )

    usable_lookup = lookup_df[lookup_df["gateway_name"].astype(str).isin(set(gateways.keys()))]
    if usable_lookup.empty:
        raise RuntimeError(
            f"External gateway lookup at {lookup_path} has no rows matching current gateway names. "
            "Re-run build-zones and build-supernetwork in the same pipeline run."
        )

    if bcfg.external.use_through_traffic:
        through_path = bcfg.external.through_pairs_path
        if not through_path.exists():
            raise FileNotFoundError(
                f"Missing through gateway pairs: {through_path}. "
                "Run build-supernetwork before build-demand."
            )
        through_df = pd.read_parquet(through_path)
        required_through = {"gateway_in", "gateway_out", "vehicles_daily"}
        missing_through = required_through - set(through_df.columns)
        if missing_through:
            raise RuntimeError(
                f"through_gateway_pairs schema mismatch at {through_path}. "
                f"Missing required columns: {sorted(missing_through)}."
            )


def _load_external_gateway_lookup_directional(
    path: Path,
    gateways: Dict[str, List[Tuple[int, float]]],
) -> Tuple[
    Dict[str, List[Tuple[int, float]]],
    Dict[str, List[Tuple[int, float]]],
    Dict[str, str],
    Dict[str, str],
]:
    """Build **directional** gateway lookup for external places.

    Returns (inbound_members, outbound_members, inbound_gateway, outbound_gateway).
    """
    _empty: Tuple[dict, dict, dict, dict] = ({}, {}, {}, {})

    if not path.exists():
        logger.warning("External gateway lookup not found: %s", path)
        return _empty

    df = pd.read_parquet(path)
    if df.empty:
        logger.warning("External gateway lookup is empty: %s", path)
        return _empty

    if "place_name_norm" not in df.columns:
        if "place_name" in df.columns:
            df["place_name_norm"] = df["place_name"].astype(str).map(_norm_name)
        else:
            raise RuntimeError(
                f"External gateway lookup missing place_name/place_name_norm: {path}"
            )

    if "gateway_name" not in df.columns:
        raise RuntimeError(f"External gateway lookup missing gateway_name: {path}")

    df["place_name_norm"] = df["place_name_norm"].astype(str).map(_norm_name)
    df["gateway_name"] = df["gateway_name"].astype(str).str.strip()
    df = df[df["place_name_norm"] != ""].copy()
    df = df[df["gateway_name"].isin(set(gateways.keys()))].copy()
    if df.empty:
        logger.warning("External gateway lookup has no usable rows after filtering: %s", path)
        return _empty

    has_directional = (
        "rank_in" in df.columns and "rank_out" in df.columns
    )

    if has_directional:
        in_cost = "route_cost_to_gateway_s" if "route_cost_to_gateway_s" in df.columns else "route_cost_s"
        out_cost = "route_cost_from_gateway_s" if "route_cost_from_gateway_s" in df.columns else "route_cost_s"
    else:
        in_cost = "route_cost_s" if "route_cost_s" in df.columns else None
        out_cost = in_cost

    def _build_weighted_members(
        df_src: pd.DataFrame,
        cost_col: Optional[str],
    ) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[str, str]]:
        members: Dict[str, List[Tuple[int, float]]] = {}
        primary_gw: Dict[str, str] = {}

        if cost_col is None or cost_col not in df_src.columns:
            return members, primary_gw

        for place, grp in df_src.groupby("place_name_norm"):
            pn = str(place).strip()
            if not pn:
                continue

            costs = pd.to_numeric(grp[cost_col], errors="coerce").values
            gw_names = grp["gateway_name"].values

            valid_mask = np.isfinite(costs) & (costs > 0)
            if not valid_mask.any():
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                inv_costs = np.where(valid_mask, 1.0 / costs, 0.0)
            total_inv = inv_costs.sum()
            if total_inv <= 0:
                continue
            weights = inv_costs / total_inv

            zone_list: List[Tuple[int, float]] = []
            best_gw: Optional[str] = None
            best_w = -1.0

            for gw_name_raw, w in zip(gw_names, weights):
                if w <= 0:
                    continue
                gw_name = str(gw_name_raw).strip()
                if gw_name not in gateways:
                    continue
                if best_gw is None or w > best_w:
                    best_gw = gw_name
                    best_w = w
                for zone_id, zone_w in gateways[gw_name]:
                    zone_list.append((zone_id, zone_w * w))

            if zone_list:
                members[pn] = zone_list
                primary_gw[pn] = best_gw or ""

        return members, primary_gw

    inbound_members, inbound_gateway = _build_weighted_members(df, in_cost)
    outbound_members, outbound_gateway = _build_weighted_members(df, out_cost)

    n_differ = sum(
        1 for pn in inbound_gateway
        if pn in outbound_gateway and inbound_gateway[pn] != outbound_gateway[pn]
    )

    gw_counts_in: Dict[str, int] = {}
    for pn, zlist in inbound_members.items():
        for gw_name in set(str(g) for g in [inbound_gateway.get(pn, "")]):
            gw_counts_in[gw_name] = gw_counts_in.get(gw_name, 0) + 1
    top_in = sorted(gw_counts_in.items(), key=lambda x: -x[1])
    gw_summary = ", ".join(f"{g}={c}" for g, c in top_in[:5])

    logger.info(
        "External gateway lookup loaded (multi-candidate weighted): %d inbound, %d outbound places "
        "(%d differ in/out); primary gw: %s",
        len(inbound_members),
        len(outbound_members),
        n_differ,
        gw_summary,
    )
    return inbound_members, outbound_members, inbound_gateway, outbound_gateway


def _load_through_gateway_pairs(path: Path) -> pd.DataFrame:
    """Load aggregated external->external flows from the supernetwork."""
    if not path.exists():
        logger.warning("Through gateway pairs not found: %s", path)
        return pd.DataFrame(columns=["gateway_in", "gateway_out", "vehicles_daily"])

    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["gateway_in", "gateway_out", "vehicles_daily"])

    required = {"gateway_in", "gateway_out", "vehicles_daily"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"through_gateway_pairs missing required columns {sorted(missing)}: {path}"
        )

    df = df.copy()
    df["gateway_in"] = df["gateway_in"].astype(str).str.strip()
    df["gateway_out"] = df["gateway_out"].astype(str).str.strip()
    df["vehicles_daily"] = pd.to_numeric(df["vehicles_daily"], errors="coerce").fillna(0.0)
    df = df[(df["gateway_in"] != "") & (df["gateway_out"] != "") & (df["vehicles_daily"] > 0)].copy()
    return df
