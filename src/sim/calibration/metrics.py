"""Link calibration statistics and diagnostics metrics."""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from sim._metrics import compute_geh  # re-export for backward compat

logger = logging.getLogger(__name__)


def compute_stats(
    modeled: np.ndarray,
    observed: np.ndarray,
    *,
    daily_capacity_factor: float = 1.0,
) -> Dict[str, Any]:
    """Compute link-level fit statistics.

    When *daily_capacity_factor* > 1 the model operates on daily aggregates;
    an adjusted GEH threshold (``5 * sqrt(K)``) is used alongside the
    standard hourly GEH<5 so that daily-model convergence criteria are
    meaningful (per FHWA/DMRB, GEH<5 targets apply to hourly flows).
    """
    m = np.asarray(modeled, dtype=float)
    c = np.asarray(observed, dtype=float)
    valid = np.isfinite(m) & np.isfinite(c) & (c > 0)
    m, c = m[valid], c[valid]
    n = len(m)
    if n == 0:
        return {"n": 0}

    geh = compute_geh(m, c)
    rmse = float(np.sqrt(np.mean((m - c) ** 2)))
    mean_obs = float(np.mean(c))
    pct_rmse = rmse / mean_obs * 100 if mean_obs > 0 else float("nan")

    ss_res = float(np.sum((m - c) ** 2))
    ss_tot = float(np.sum((c - np.mean(c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # OLS regression slope / intercept  (model = slope * observed + intercept)
    if n >= 2:
        coeffs = np.polyfit(c, m, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
    else:
        slope = float("nan")
        intercept = float("nan")

    # MAPE: mean(|M-O|/O) * 100 — only for positive observed
    mape = float(np.mean(np.abs(m - c) / c) * 100.0)

    # Overall volume bias
    sum_m, sum_c = float(m.sum()), float(c.sum())
    bias_pct = (sum_m - sum_c) / sum_c * 100.0 if sum_c > 0 else float("nan")

    # Daily-adjusted GEH threshold: GEH scales as ~sqrt(K) on daily
    # volumes relative to hourly, so the "GEH<5" criterion becomes
    # "GEH < 5*sqrt(K)" for an equivalent daily acceptance band.
    k = max(daily_capacity_factor, 1.0)
    daily_geh_thr = 5.0 * math.sqrt(k)
    daily_geh_lt_adj_pct = float(np.nanmean(geh < daily_geh_thr) * 100.0)

    return {
        "n": int(n),
        "r2": round(r2, 4) if np.isfinite(r2) else None,
        "slope": round(slope, 4) if np.isfinite(slope) else None,
        "intercept": round(intercept, 1) if np.isfinite(intercept) else None,
        "rmse": round(rmse, 1),
        "pct_rmse": round(pct_rmse, 1) if np.isfinite(pct_rmse) else None,
        "mape_pct": round(mape, 1) if np.isfinite(mape) else None,
        "bias_pct": round(bias_pct, 2) if np.isfinite(bias_pct) else None,
        "geh_mean": round(float(np.nanmean(geh)), 2),
        "geh_median": round(float(np.nanmedian(geh)), 2),
        "geh_lt5_pct": round(float(np.nanmean(geh < 5) * 100), 1),
        "geh_lt10_pct": round(float(np.nanmean(geh < 10) * 100), 1),
        "daily_geh_threshold": round(daily_geh_thr, 1),
        "daily_geh_lt_adj_pct": round(daily_geh_lt_adj_pct, 1),
        "sum_modeled": round(sum_m, 0),
        "sum_observed": round(sum_c, 0),
    }


def _coarse_road_class(link_type: object) -> str:
    """Map OSM link_type to a small set of buckets for bias / MAE reporting."""
    s = str(link_type).lower()
    for rc in ("motorway", "trunk", "primary", "secondary", "tertiary"):
        if rc in s:
            return rc
    return "other"


def _compute_class_residuals(
    valid: pd.DataFrame,
    obs_col: str,
    compare_col: str,
    *,
    min_counts: int = 3,
) -> Dict[str, float]:
    """Per-road-class obs/mod ratio for class-specific OD correction.

    Returns a dict mapping coarse road class names to their aggregate
    observed/modeled ratio.  Classes with fewer than *min_counts* matched
    count posts are omitted.
    """
    if "link_type" not in valid.columns:
        return {}
    coarse = valid["link_type"].map(_coarse_road_class)
    ratios: Dict[str, float] = {}
    for rc in sorted(coarse.unique()):
        mask = coarse == rc
        n = int(mask.sum())
        if n < min_counts:
            continue
        obs_sum = float(valid.loc[mask, obs_col].sum())
        mod_sum = float(valid.loc[mask, compare_col].sum())
        if mod_sum > 0 and obs_sum > 0:
            ratios[rc] = obs_sum / mod_sum
    return ratios


def _supplement_class_ratios_from_screenlines(
    sl_results: Dict[str, Any],
    links_gdf: "gpd.GeoDataFrame",
    pentlogram_ratios: Dict[str, float],
) -> Dict[str, float]:
    """Derive per-road-class obs/mod ratios from screenline evaluation results.

    Screenlines carry aggregate observed/modeled totals and resolve to network
    links with known ``link_type``.  This supplements the pentlogram-based
    ``class_ratios`` with classes that have no pentlogram coverage (typically
    motorway/trunk on boundary gateways).

    For classes already present in *pentlogram_ratios* the pentlogram value
    is kept (higher spatial resolution).  New classes are added from
    screenline evidence.
    """
    if not sl_results or "link_type" not in links_gdf.columns:
        return dict(pentlogram_ratios)

    lt_series = links_gdf.set_index("link_id")["link_type"]

    # Accumulate per-class obs/mod sums across screenlines
    class_obs: Dict[str, float] = {}
    class_mod: Dict[str, float] = {}
    from sim.calibration.screenlines import screenline_excluded_from_benchmark

    for sl_name, sr in sl_results.items():
        sr_d = sr if isinstance(sr, dict) else {}
        if screenline_excluded_from_benchmark(sl_name, sr_d):
            continue
        obs_total = float(sr_d.get("observed_total") or 0)
        mod_total = float(sr_d.get("modeled_total") or 0)
        if obs_total <= 0 or mod_total <= 0:
            continue
        ratio = obs_total / mod_total

        per_link = sr_d.get("per_link", [])
        if not per_link and hasattr(sr, "per_link"):
            per_link = sr.per_link if sr.per_link else []

        link_ids = []
        if isinstance(per_link, list):
            link_ids = [
                pl.get("link_id") if isinstance(pl, dict)
                else getattr(pl, "link_id", None)
                for pl in per_link
            ]
        elif isinstance(per_link, dict):
            link_ids = list(per_link.keys())

        if not link_ids:
            continue

        types = [
            _coarse_road_class(lt_series.get(lid, "other"))
            for lid in link_ids if lid is not None
        ]
        if not types:
            continue
        dominant = Counter(types).most_common(1)[0][0]

        class_obs[dominant] = class_obs.get(dominant, 0.0) + obs_total
        class_mod[dominant] = class_mod.get(dominant, 0.0) + mod_total

    merged = dict(pentlogram_ratios)
    for rc in class_obs:
        if rc in merged:
            continue
        obs_sum = class_obs[rc]
        mod_sum = class_mod[rc]
        if mod_sum > 0 and obs_sum > 0:
            merged[rc] = max(0.5, min(2.0, obs_sum / mod_sum))

    return merged


def _clamp_class_ratios(
    class_ratios: Dict[str, float],
    odme_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Apply optional per-class floors/caps before class-residual OD correction."""
    if not class_ratios:
        return {}
    out = dict(class_ratios)
    for rc, cap in (odme_cfg.get("class_residual_ratio_caps") or {}).items():
        if rc in out:
            out[rc] = min(out[rc], float(cap))
    for rc, floor in (odme_cfg.get("class_residual_ratio_floors") or {}).items():
        if rc in out:
            out[rc] = max(out[rc], float(floor))
    return out


def _apply_class_residual_correction(
    demand: np.ndarray,
    vol_df: pd.DataFrame,
    links_gdf: "gpd.GeoDataFrame",
    class_ratios: Dict[str, float],
    sl_matrices: Optional[Dict[str, Any]],
    sl_results: Dict[str, Any],
    *,
    damping: float = 0.15,
    clip_min: float = 0.5,
    clip_max: float = 2.0,
    reference_total: Optional[float] = None,
) -> List[str]:
    """Apply per-road-class OD correction using screenline select-link shares.

    For each road class with a non-unity ratio, identifies which screenlines
    belong to that class (via their matched link types) and applies a damped
    multiplicative correction using those screenlines' select-link OD shares.

    Falls back to a uniform scalar when no select-link data is available for
    a class.

    *reference_total*: when supplied, the volume-neutral rescale targets this
    value instead of the pre-correction demand total.  This allows callers to
    preserve gains from prior steps (e.g. gateway calibration).

    Returns a list of log messages describing corrections applied.
    """
    if not class_ratios:
        return []

    total_before = reference_total if reference_total is not None else float(demand.sum())

    # Map screenlines to their dominant road class
    sl_class_map: Dict[str, str] = {}
    from sim.calibration.screenlines import screenline_excluded_from_benchmark

    if sl_results:
        for sl_name, sr in sl_results.items():
            sr_d = sr if isinstance(sr, dict) else {}
            if screenline_excluded_from_benchmark(sl_name, sr_d):
                continue
            per_link = sr_d.get("per_link", [])
            if not per_link and hasattr(sr, "per_link"):
                per_link = sr.per_link if sr.per_link else []
            link_ids = []
            if isinstance(per_link, list):
                link_ids = [pl.get("link_id") if isinstance(pl, dict) else getattr(pl, "link_id", None) for pl in per_link]
            elif isinstance(per_link, dict):
                link_ids = list(per_link.keys())

            if link_ids and "link_type" in links_gdf.columns:
                lt_series = links_gdf.set_index("link_id")["link_type"]
                types = [_coarse_road_class(lt_series.get(lid, "other")) for lid in link_ids if lid is not None]
                if types:
                    sl_class_map[sl_name] = Counter(types).most_common(1)[0][0]

    corrections: List[str] = []

    for rc, ratio in class_ratios.items():
        if abs(ratio - 1.0) < 0.02:
            continue

        # Try select-link approach: use screenlines belonging to this class
        applied_via_sl = False
        if sl_matrices and sl_class_map:
            class_sl_names = [sn for sn, sc in sl_class_map.items() if sc == rc and sn in sl_matrices]
            if class_sl_names:
                class_factor = 1.0 + damping * (ratio - 1.0)
                class_factor = float(np.clip(class_factor, clip_min, clip_max))
                for sl_name in class_sl_names:
                    sl_od_raw = sl_matrices[sl_name]
                    sl_od = np.asarray(sl_od_raw, dtype=np.float64).reshape(demand.shape)
                    proportion = np.where(
                        demand > 0,
                        np.clip(sl_od / np.maximum(demand, 1e-9), 0.0, 1.0),
                        0.0,
                    )
                    adjustment = 1.0 + damping * (ratio - 1.0) * proportion
                    np.clip(adjustment, clip_min, clip_max, out=adjustment)
                    demand *= adjustment
                applied_via_sl = True
                corrections.append(
                    f"{rc}: ratio={ratio:.3f} factor={class_factor:.4f} "
                    f"via {len(class_sl_names)} screenline(s)"
                )

        if not applied_via_sl:
            # Fallback: build road-class weight from link volumes
            if "link_id" in vol_df.columns and "link_type" in links_gdf.columns:
                lt_map = links_gdf.set_index("link_id")["link_type"]
                merged_lt = vol_df["link_id"].map(lt_map).map(_coarse_road_class)
                vol_col_name = None
                for c in vol_df.columns:
                    if c.endswith("_tot") or c == "tot":
                        vol_col_name = c
                        break
                if vol_col_name is None:
                    vol_col_name = [c for c in vol_df.columns if c not in ("link_id",) and vol_df[c].dtype in (np.float64, np.float32, np.int64)]
                    vol_col_name = vol_col_name[0] if vol_col_name else None

                if vol_col_name is not None:
                    class_mask = merged_lt == rc
                    class_vol = float(vol_df.loc[class_mask, vol_col_name].sum()) if class_mask.any() else 0.0
                    total_vol = float(vol_df[vol_col_name].sum())
                    if total_vol > 0 and class_vol > 0:
                        weight = class_vol / total_vol
                        effective_factor = 1.0 + damping * weight * (ratio - 1.0)
                        effective_factor = float(np.clip(effective_factor, clip_min, clip_max))
                        if abs(effective_factor - 1.0) > 0.002:
                            demand *= effective_factor
                            corrections.append(
                                f"{rc}: ratio={ratio:.3f} vol_share={weight:.2f} "
                                f"factor={effective_factor:.4f} (fallback)"
                            )

    # Volume-neutral rescale: class correction only redistributes, total
    # volume adjustment is handled by global_residual separately.
    total_after = float(demand.sum())
    if total_before > 0 and total_after > 0 and abs(total_after - total_before) > 1.0:
        demand *= total_before / total_after
        corrections.append(
            f"volume-neutral rescale: {total_after:,.0f} → {total_before:,.0f}"
        )

    return corrections


def compute_extended_link_metrics(
    matched: pd.DataFrame,
    model_col: str,
    obs_col: str,
) -> Dict[str, Any]:
    """Extra fit diagnostics for metric validity experiments (wMAPE, class MAE, Spearman).

    *matched* should be rows with positive observed counts and finite modeled volumes.
    """
    if model_col not in matched.columns or obs_col not in matched.columns:
        return {}

    sub = matched[[model_col, obs_col]].dropna()
    sub = sub[(sub[obs_col] > 0) & np.isfinite(sub[model_col])].copy()
    if sub.empty:
        return {}

    m = sub[model_col].astype(float).values
    o = sub[obs_col].astype(float).values

    wmape = float(np.sum(np.abs(m - o)) / max(np.sum(o), 1e-9) * 100.0)
    denom_smape = np.abs(m) + np.abs(o)
    mask = denom_smape > 0
    smape = float(
        np.mean(2.0 * np.abs(m[mask] - o[mask]) / denom_smape[mask]) * 100.0
    ) if mask.any() else float("nan")

    srs = pd.Series(m)
    srs_o = pd.Series(o)
    rho = srs.corr(srs_o, method="spearman")
    spearman_rho = round(float(rho), 4) if rho is not None and np.isfinite(rho) else None

    sum_m, sum_o = float(m.sum()), float(o.sum())
    sum_ratio = round(sum_m / max(sum_o, 1e-9), 4) if sum_o > 0 else None

    mae_by_class: Dict[str, float] = {}
    bias_pct_by_class: Dict[str, float] = {}
    n_by_class: Dict[str, int] = {}

    if "link_type" in matched.columns:
        lt_sub = matched.loc[sub.index]
        coarse = lt_sub["link_type"].map(_coarse_road_class)
        for rc in coarse.unique():
            idx = coarse == rc
            if not idx.any():
                continue
            take = idx.to_numpy()
            mc = m[take]
            oc = o[take]
            mae_by_class[str(rc)] = round(float(np.mean(np.abs(mc - oc))), 1)
            bias_pct_by_class[str(rc)] = round(
                float(np.mean((mc - oc) / np.maximum(oc, 1e-9)) * 100.0), 2
            )
            n_by_class[str(rc)] = int(idx.sum())

    biases = list(bias_pct_by_class.values()) if bias_pct_by_class else []
    class_bias_max_abs = round(float(max(abs(b) for b in biases)), 2) if biases else None

    return {
        "wmape_pct": round(wmape, 2),
        "smape_pct": round(smape, 2) if np.isfinite(smape) else None,
        "spearman_rho": spearman_rho,
        "sum_ratio": sum_ratio,
        "mae_by_class": mae_by_class,
        "bias_pct_by_class": bias_pct_by_class,
        "n_by_class": n_by_class,
        "class_bias_max_abs_pct": class_bias_max_abs,
    }


def compute_class_volume_breakdown(
    vol_df: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """Per-road-class volume breakdown split by traffic class (local vs through).

    Joins assignment results with network link_type and groups total assigned
    volume by coarse road class.  When multi-class columns are present
    (e.g. ``local_tot``, ``through_tot``), reports each class separately so
    callers can see how much local vs through traffic uses motorway vs trunk.
    """
    if vol_df.empty:
        return {}

    merged = vol_df.copy()
    if "link_type" not in merged.columns and "link_id" in merged.columns:
        lt_map = links_gdf.set_index("link_id")["link_type"] if "link_type" in links_gdf.columns else None
        if lt_map is not None:
            merged["link_type"] = merged["link_id"].map(lt_map)

    if "link_type" not in merged.columns:
        return {}

    merged["_road_class"] = merged["link_type"].map(_coarse_road_class)

    tot_cols = [c for c in merged.columns if c.endswith("_tot")]
    if not tot_cols:
        return {}

    breakdown: Dict[str, Any] = {}
    for col in tot_cols:
        grp = merged.groupby("_road_class")[col].agg(["sum", "count"])
        breakdown[col] = {
            rc: {"total_volume": round(float(row["sum"]), 0), "n_links": int(row["count"])}
            for rc, row in grp.iterrows()
        }

    total_col = tot_cols[0]
    if len(tot_cols) > 1:
        for rc in ("motorway", "trunk"):
            rc_mask = merged["_road_class"] == rc
            if not rc_mask.any():
                continue
            parts = {col: round(float(merged.loc[rc_mask, col].sum()), 0) for col in tot_cols}
            total = sum(parts.values())
            shares = {col: round(v / max(total, 1), 3) for col, v in parts.items()}
            breakdown.setdefault("_class_shares", {})[rc] = {
                "volumes": parts, "shares": shares,
            }

    return breakdown
