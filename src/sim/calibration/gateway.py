"""Gateway calibration helpers: screenline↔gateway mapping, observed AADT, OD scaling."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sim.io_project import get_metric_epsg, get_nested

logger = logging.getLogger(__name__)

import warnings as _warnings

# Deprecated: use ``auto_gw_`` prefix screenlines instead.
# Will be removed in a future version.
_SCREENLINE_TO_GATEWAY_LEGACY = {
    "D1_west": "D1_NW",
    "D1_east": "D1_E",
    "D2_south": "D2_S",
    "I43_north": "I43_N",
    "I52_south": "I52_S",
}


def _build_screenline_gateway_map(
    screenlines: list,
    gateway_zone_names: Optional[set] = None,
) -> Dict[str, str]:
    """Auto-derive screenline_name -> gateway_zone_name mapping.

    For ``auto_gw_*`` screenlines the gateway name is the suffix after the
    prefix, validated against *gateway_zone_names* when provided.
    Legacy manual mappings are kept as fallback.
    """
    mapping: Dict[str, str] = {}
    gw_set = gateway_zone_names or set()

    for sl in screenlines:
        name = sl.name if hasattr(sl, "name") else str(sl)
        if name.startswith("auto_gw_"):
            gw_candidate = name[len("auto_gw_"):]
            if not gw_set or gw_candidate in gw_set:
                mapping[name] = gw_candidate

    for k, v in _SCREENLINE_TO_GATEWAY_LEGACY.items():
        if k not in mapping:
            mapping[k] = v

    return mapping


from sim._metrics import MAJOR_ROAD_TYPES as _MAJOR_ROAD_TYPES  # noqa: F401
from sim._metrics import MAJOR_ROAD_TYPES_STRICT as _MAJOR_ROAD_TYPES_STRICT  # noqa: F401


def _load_gateway_zone_map(
    cfg: Dict[str, Any],
    mat_index: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build gateway_name -> array of matrix row/col indices for external zones.

    Reads ``zones.geojson`` from the zoning output directory, selects rows
    where ``is_external == 1``, and maps each ``gateway_name`` to the
    corresponding matrix indices (positions in ``mat_index``).

    The matrix uses centroid IDs (1..N) while zones.geojson uses zone_ids
    (which may be large synthetic values like 8000000020 for external zones).
    A zone_centroid_mapping.json bridges the two ID spaces.
    """
    zoning_dir = Path(get_nested(cfg, ["zoning", "output_dir"], "outputs/baseline/zones"))
    zones_path = zoning_dir / "zones.geojson"
    if not zones_path.exists():
        return {}

    zones_gdf = gpd.read_file(zones_path)
    if "is_external" not in zones_gdf.columns or "zone_id" not in zones_gdf.columns:
        return {}

    external = zones_gdf[zones_gdf["is_external"].fillna(0).astype(int) == 1].copy()
    if external.empty:
        return {}

    mapping_path = zoning_dir / "zone_centroid_mapping.json"
    zone_to_centroid: Dict[int, int] = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            raw = json.load(f)
        zone_to_centroid = {int(k): int(v) for k, v in raw.items()}

    idx_lookup = {int(v): i for i, v in enumerate(mat_index)}
    result: Dict[str, np.ndarray] = {}

    for _, row in external.iterrows():
        gw = str(row.get("gateway_name", "")).strip()
        if not gw:
            name = str(row.get("name", "")).strip()
            gw = name.replace("EXT_", "", 1) if name.startswith("EXT_") else name
        if not gw:
            continue
        zid = int(row["zone_id"])
        centroid_id = zone_to_centroid.get(zid, zid)
        mat_idx = idx_lookup.get(centroid_id)
        if mat_idx is not None:
            result.setdefault(gw, []).append(mat_idx)

    return {gw: np.array(indices, dtype=int) for gw, indices in result.items()}


def _load_gateway_observed(
    cfg: Dict[str, Any],
    screenlines_list: Optional[list] = None,
    gateway_zone_names: Optional[set] = None,
) -> Dict[str, float]:
    """Load observed AADT per gateway for gateway calibration.

    Tries the parquet at ``calibration.gateway_calibration.observed_path``
    first (columns: ``gateway_name``, ``observed_aadt``).  If not found,
    falls back to observed values from screenlines (including auto-generated
    ones) using the dynamic screenline-to-gateway mapping.
    """
    gw_cfg = get_nested(cfg, ["calibration", "gateway_calibration"], {}) or {}
    _cache = str(Path(get_nested(cfg, ["datasets", "cache_dir"], "data/cache")))
    obs_path = Path(gw_cfg.get("observed_path", f"{_cache}/gateway_counts_2025.parquet"))

    if obs_path.exists():
        try:
            df = pd.read_parquet(obs_path)
            if "gateway_name" in df.columns and "observed_aadt" in df.columns:
                return {
                    str(r["gateway_name"]).strip(): float(r["observed_aadt"])
                    for _, r in df.iterrows()
                    if float(r["observed_aadt"]) > 0
                }
        except Exception:
            logger.debug(
                "Failed to load gateway observed counts from parquet; falling back to screenlines",
                exc_info=True,
            )

    if screenlines_list is None:
        from sim.calibration.observed import load_csd, load_csd_unfiltered
        from sim.calibration.screenlines import load_screenlines_with_auto

        try:
            _csd = None
            _csd_full = None
            try:
                _csd = load_csd(cfg)
            except Exception:
                pass
            try:
                _csd_full = load_csd_unfiltered(cfg)
            except Exception:
                pass
            screenlines_list = load_screenlines_with_auto(
                cfg,
                csd_df=_csd,
                csd_df_full=_csd_full,
            )
        except Exception:
            from sim.calibration.screenlines import load_screenlines

            config_dir = cfg.get("_meta", {}).get("base_dir", "")
            default_sl = str(Path(config_dir) / "screenlines.yaml") if config_dir else ""
            sl_path = str(get_nested(cfg, ["calibration", "screenlines_path"], default_sl))
            screenlines_list = load_screenlines(sl_path) if sl_path else []

    sl_gw_map = _build_screenline_gateway_map(screenlines_list, gateway_zone_names)

    result: Dict[str, float] = {}
    for sl in screenlines_list:
        gw = sl_gw_map.get(sl.name)
        aadt = getattr(sl, "observed_aadt_all", None) or getattr(sl, "observed_aadt_cars", None)
        if gw and aadt and float(aadt) > 0:
            result[gw] = float(aadt)

    return result


def _apply_gateway_calibration(
    demand: np.ndarray,
    gateway_zone_map: Dict[str, np.ndarray],
    gateway_observed: Dict[str, float],
    gateway_modeled: Dict[str, float],
    *,
    damping: float = 0.08,
    min_factor: float = 0.90,
    max_factor: float = 1.10,
    seed_lower: Optional[np.ndarray] = None,
    seed_upper: Optional[np.ndarray] = None,
) -> List[str]:
    """Scale OD rows/columns for each gateway's external zones toward observed AADT.

    For each gateway with both observed and modeled totals, compute
    ``factor = 1 + damping * (obs/mod - 1)`` clipped to ``[min_factor, max_factor]``
    and multiply all OD cells where either origin or destination belongs
    to that gateway's external zones.

    Returns a list of log strings describing corrections applied.
    """
    corrections: List[str] = []
    n = demand.shape[0]

    for gw_name, ext_indices in gateway_zone_map.items():
        obs = gateway_observed.get(gw_name, 0.0)
        mod = gateway_modeled.get(gw_name, 0.0)
        if obs <= 0 or mod <= 0:
            continue

        ratio = obs / mod
        if ratio > 10.0 or ratio < 0.1:
            continue

        factor = 1.0 + damping * (ratio - 1.0)
        factor = float(np.clip(factor, min_factor, max_factor))

        if abs(factor - 1.0) < 0.001:
            continue

        mask = np.zeros(n, dtype=bool)
        mask[ext_indices] = True

        demand[mask, :] *= factor
        demand[:, mask] *= factor
        # Undo double-scaling of ext-ext cells within same gateway
        demand[np.ix_(mask, mask)] /= factor

        if seed_lower is not None and seed_upper is not None:
            rows_to_clip = np.where(mask)[0]
            for ri in rows_to_clip:
                np.clip(demand[ri, :], seed_lower[ri, :], seed_upper[ri, :], out=demand[ri, :])
                np.clip(demand[:, ri], seed_lower[:, ri], seed_upper[:, ri], out=demand[:, ri])

        corrections.append(
            f"{gw_name}: obs={obs:,.0f} mod={mod:,.0f} "
            f"ratio={ratio:.2f} factor={factor:.4f}"
        )

    return corrections


def _compute_gateway_modeled_volumes(
    vol_df: pd.DataFrame,
    screenlines: list,
    vol_col: str,
    sl_gw_map: Optional[Dict[str, str]] = None,
    *,
    corridor_link_specs: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Dict[str, float]:
    """Sum modeled volume on each screenline and map to gateway names.

    Optional *corridor_link_specs* overrides modeled totals for selected gateways
    by summing assignment volumes on explicit ``(link_id, direction)`` tuples
    (direction ``0`` = screenline-style total column, ``1`` / ``-1`` = AB / BA).
    """
    from sim.calibration.screenlines import _get_link_volume

    result: Dict[str, float] = {}
    if not vol_col:
        return result

    if sl_gw_map is None:
        sl_gw_map = _build_screenline_gateway_map(screenlines)

    if screenlines:
        for sl in screenlines:
            gw = sl_gw_map.get(sl.name)
            if not gw or not sl.links:
                continue

            total = 0.0
            for link_id, direction in sl.links:
                total += _get_link_volume(vol_df, link_id, direction, vol_col)
            result[gw] = total

    for gw, pairs in (corridor_link_specs or {}).items():
        if not pairs:
            continue
        total = 0.0
        for link_id, direction in pairs:
            total += _get_link_volume(vol_df, int(link_id), int(direction), vol_col)
        result[str(gw).strip()] = total

    return result
