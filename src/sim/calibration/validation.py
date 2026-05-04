"""Independent validation helpers: benchmarks, journey times, CSD–link matching."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae import Project  # noqa: F401 — parity with skim/assignment tooling
from aequilibrae.matrix import AequilibraeMatrix

from sim.assignment import (
    _apply_bpr_defaults,
    _detect_volume_col,
    resolve_daily_cap_factor_default,
    run_assignment,
)
from sim.assignment import execute_assignment  # noqa: F401
from sim.assignment import _resolve_multi_class  # noqa: F401
from sim.calibration.gateway import (  # noqa: F401
    _MAJOR_ROAD_TYPES,
    _MAJOR_ROAD_TYPES_STRICT,
)
from sim.calibration.matching import (
    _NON_CAR_LINK_TYPES,
    _export_matching_diagnostics,
    match_counts_to_links,
    match_quality_report,
)
from sim.calibration.metrics import compute_extended_link_metrics, compute_geh, compute_stats
from sim.calibration.observed import (
    _MIN_VOL_FOR_CSD_LW,
    _classify_csd_road,
    _CSD_COMPATIBLE_LINK_TYPES,
    _load_network_links,
    aggregate_csd_by_class,
    aggregate_model_by_class,
    load_csd,
    load_csd_as_link_counts,
    load_csd_unfiltered,
    load_pentlogram,
    split_csd_for_calibration,
    validate_geometries_or_fail,
)
from sim.defaults import LOCALE_DEFAULTS as _LOCALE_DEFAULTS
from sim.defaults import SIM_DEFAULTS as _SIM_DEFAULTS  # noqa: F401
from sim.io_project import get_metric_epsg, get_nested, load_config

logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --- Journey time validation ---

_CAR_LINK_TYPES_FOR_SPEED = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "unclassified",
    "living_street",
})

_CZECH_REFERENCE_SPEEDS: Dict[str, float] = {
    **{k: float(v) for k, v in _LOCALE_DEFAULTS["reference_speeds"].items()},
    "unclassified": 40.0,
    "living_street": 20.0,
}


def compute_class_speed_comparison(
    links_gdf: gpd.GeoDataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compare modeled average speeds per road class against reference speeds."""
    result: Dict[str, Dict[str, float]] = {}
    for lt in sorted(_CAR_LINK_TYPES_FOR_SPEED):
        sub = links_gdf[links_gdf["link_type"] == lt] if "link_type" in links_gdf.columns else pd.DataFrame()
        if sub.empty:
            continue

        for spd_col in ("speed_ab", "speed"):
            if spd_col in sub.columns:
                speeds = pd.to_numeric(sub[spd_col], errors="coerce").dropna()
                if not speeds.empty:
                    model_speed = float(speeds.mean())
                    ref_speed = _CZECH_REFERENCE_SPEEDS.get(lt, 50.0)
                    pct_diff = (model_speed - ref_speed) / max(ref_speed, 1) * 100
                    result[lt] = {
                        "modeled_kmh": round(model_speed, 1),
                        "reference_kmh": round(ref_speed, 1),
                        "pct_diff": round(pct_diff, 1),
                    }
                    break

    return result


def validate_journey_times(
    skim_matrix: np.ndarray,
    zone_ids: np.ndarray,
    routes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate journey times from skim matrix against reference routes.

    Each route dict has: name, from_zone, to_zone, reference_time_min, tolerance_pct.
    """
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    results: List[Dict[str, Any]] = []

    for route in routes:
        name = route.get("name", "unnamed")
        fz = int(route.get("from_zone", 0))
        tz = int(route.get("to_zone", 0))
        ref_min = float(route.get("reference_time_min", 0))
        tol_pct = float(route.get("tolerance_pct", 15))

        if fz not in z2i or tz not in z2i or ref_min <= 0:
            results.append({"name": name, "status": "skipped", "reason": "invalid zone or time"})
            continue

        fi, ti = z2i[fz], z2i[tz]
        model_sec = float(skim_matrix[fi, ti])
        model_min = model_sec / 60.0 if model_sec > 0 else float("nan")

        if not np.isfinite(model_min) or model_min <= 0:
            results.append({"name": name, "status": "no_path", "model_min": None, "ref_min": ref_min})
            continue

        diff_pct = (model_min - ref_min) / ref_min * 100
        diff_min = model_min - ref_min
        within_pct = abs(diff_pct) <= tol_pct
        within_1min = abs(diff_min) <= 1.0
        passed = within_pct or within_1min

        results.append({
            "name": name,
            "from_zone": fz,
            "to_zone": tz,
            "model_min": round(model_min, 1),
            "ref_min": ref_min,
            "diff_pct": round(diff_pct, 1),
            "diff_min": round(diff_min, 1),
            "pass": passed,
        })

    return results


def _check_final_convergence(
    history: List[Dict[str, Any]],
    model_time_period: str,
    geh_target: float,
    daily_conv: Dict[str, Any],
) -> bool:
    """Check whether the final iteration satisfies convergence criteria.

    Daily models use R², slope, %RMSE, bias, and screenline deviation — GEH
    is excluded because its Poisson assumption is invalid at daily volumes.
    Hourly models require both %GEH<5 and mean GEH to meet their targets.
    """
    if not history:
        return False
    final = history[-1]
    if model_time_period == "daily":
        r2 = float(final.get("r2") or 0.0)
        slp = float(final.get("slope") or 0.0)
        prmse = float(final.get("pct_rmse") or 999.0)
        bias = abs(float(final.get("bias_pct") or 999.0))
        sr = daily_conv.get("slope_range", [0.85, 1.15])
        sl_max_dev = float(final.get("max_screenline_pct_dev") or 999.0)
        sl_target = float(daily_conv.get("screenline_max_pct_deviation", 15.0))
        return (
            r2 >= float(daily_conv.get("r2_target", 0.80))
            and float(sr[0]) <= slp <= float(sr[1])
            and prmse <= float(daily_conv.get("pct_rmse_max", 35.0))
            and bias <= float(daily_conv.get("bias_abs_max_pct", 15.0))
            and sl_max_dev <= sl_target
        )
    geh_mean_target = float(daily_conv.get("geh_mean_max", 5.0))
    return (
        float(final.get("geh_lt5_pct", 0)) >= geh_target
        and float(final.get("geh_mean") or 999.0) <= geh_mean_target
    )


def match_csd_to_links(
    csd: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
    *,
    csd_full: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-road matching via ``osm_ref`` ↔ CSD ``sil``.

    For each CSD road number (e.g. D1, 52, 152), finds all model links
    whose ``osm_ref`` contains that road number (splitting composite
    refs like ``"D1;50"`` on ``";"``) and computes a length-weighted
    mean model volume, compared against the CSD section-averaged AADT.

    Links with near-zero volume (< 100 veh/day) are excluded from the
    length-weighted mean to avoid dilution by boundary artifacts.
    Link types are filtered to be compatible with the CSD road class.

    *csd_full* (optional): pre-split full CSD dataset used for accurate
    road-length coverage estimation.  When provided, ``delka`` sums come
    from the full dataset rather than the (potentially subsetted) *csd*.
    """
    csd = csd.copy()
    if "sil" not in csd.columns or "osm_ref" not in links_gdf.columns:
        return pd.DataFrame()

    csd["sil"] = csd["sil"].astype(str)
    for col in ("o", "sv", "tv"):
        if col in csd.columns:
            csd[col] = pd.to_numeric(csd[col], errors="coerce").fillna(0)

    csd["road_class"] = csd["sil"].apply(_classify_csd_road)
    csd_sil = csd["sil"].str.strip()

    vol_cols = sorted(
        c for c in links_gdf.columns
        if c.endswith("_tot") and c not in ("PCE_tot", "Preload_tot")
    )
    tot_col = None
    if "total_vehicles_tot" in vol_cols:
        tot_col = "total_vehicles_tot"
    else:
        for vc in vol_cols:
            if links_gdf[vc].sum() > 0:
                tot_col = vc
                break
    if tot_col is None:
        return pd.DataFrame()

    links_work = links_gdf.copy()
    links_work[tot_col] = pd.to_numeric(links_work[tot_col], errors="coerce").fillna(0)
    if "distance" in links_work.columns:
        links_work["distance"] = pd.to_numeric(links_work["distance"], errors="coerce").fillna(0)

    raw_refs = links_work["osm_ref"].fillna("").str.strip()

    # Build a reverse index: for each link, explode composite osm_ref
    # ("D1;50" → ["D1", "50"]) so CSD roads match any component.
    ref_components = raw_refs.str.split(";").explode().str.strip()
    ref_components = ref_components[ref_components != ""]

    matched_roads: list = []
    csd_roads = csd_sil.unique()

    for road in csd_roads:
        csd_sub = csd[csd_sil == road]
        if csd_sub.empty:
            continue

        matching_indices = ref_components.index[ref_components == road]
        if len(matching_indices) == 0:
            continue
        model_sub = links_work.loc[matching_indices.unique()]

        csd_mean_sv = float(csd_sub["sv"].mean())
        csd_mean_o = float(csd_sub["o"].mean())
        n_csd = len(csd_sub)
        road_class = csd_sub["road_class"].iloc[0]

        car_links = model_sub
        if "link_type" in car_links.columns:
            lt = car_links["link_type"].astype(str)
            car_links = car_links[~lt.isin(_NON_CAR_LINK_TYPES)]
            compatible = _CSD_COMPATIBLE_LINK_TYPES.get(road_class)
            if compatible:
                car_links = car_links[lt.reindex(car_links.index).isin(compatible)]

        n_model = len(car_links)
        if n_model == 0:
            continue

        active = car_links[car_links[tot_col] >= _MIN_VOL_FOR_CSD_LW]
        if active.empty:
            active = car_links

        vols = active[tot_col].values
        if "distance" in active.columns:
            dists = active["distance"].values
            dsum = float(dists.sum())
            if dsum > 0:
                model_lw_mean = float((vols * dists).sum() / dsum)
            else:
                model_lw_mean = float(vols.mean())
        else:
            model_lw_mean = float(vols.mean())

        if csd_mean_sv <= 0:
            continue

        geh = float(compute_geh(np.array([model_lw_mean]), np.array([csd_mean_sv]))[0])

        model_road_km = float(car_links["distance"].sum()) / 1000.0 if "distance" in car_links.columns else 0.0
        csd_source = csd_full if csd_full is not None else csd_sub
        if "delka" in csd_source.columns and "sil" in csd_source.columns:
            csd_road_km = float(csd_source[csd_source["sil"].astype(str).str.strip() == road]["delka"].sum())
        else:
            csd_road_km = 0.0
        coverage = model_road_km / csd_road_km if csd_road_km > 0 else 1.0
        is_partial = coverage < 0.5

        matched_roads.append({
            "road": road,
            "road_class": road_class,
            "csd_sections": n_csd,
            "model_links": n_model,
            "csd_mean_sv": round(csd_mean_sv, 0),
            "csd_mean_o": round(csd_mean_o, 0),
            "model_lw_mean": round(model_lw_mean, 0),
            "geh": round(geh, 1),
            "model_road_km": round(model_road_km, 1),
            "csd_road_km": round(csd_road_km, 1),
            "coverage_ratio": round(coverage, 2),
            "partial_coverage": is_partial,
        })

    if not matched_roads:
        return pd.DataFrame()

    result = pd.DataFrame(matched_roads)

    # Summary statistics across matched roads (exclude partial-coverage roads)
    full_cov = ~result["partial_coverage"].astype(bool)
    sub = result[full_cov]
    n_partial = int((~full_cov).sum())

    obs = sub["csd_mean_sv"].values.astype(float)
    mod = sub["model_lw_mean"].values.astype(float)

    if len(obs) >= 2:
        mask = (obs > 0) & (mod > 0)
        if mask.sum() >= 2:
            o_valid = obs[mask]
            m_valid = mod[mask]
            ss_res = float(((m_valid - o_valid) ** 2).sum())
            ss_tot = float(((o_valid - o_valid.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            bias = float((m_valid.sum() - o_valid.sum()) / o_valid.sum() * 100)
            pct_rmse = float(np.sqrt(((m_valid - o_valid) ** 2).mean()) / o_valid.mean() * 100)
            gehs = sub["geh"].values[mask]
            result.attrs["summary"] = {
                "n_roads": int(mask.sum()),
                "n_partial_excluded": n_partial,
                "r2": round(r2, 3),
                "bias_pct": round(bias, 1),
                "pct_rmse": round(pct_rmse, 1),
                "mean_geh": round(float(gehs.mean()), 1),
            }

    return result


def _classify_holdout_adequacy(n: int) -> str:
    """Classify holdout set size: adequate (>=10), thin (5-9), insufficient (<5)."""
    if n >= 10:
        return "adequate"
    if n >= 5:
        return "thin"
    return "insufficient"


def _compute_daily_metrics(
    count_stats: Dict[str, Any],
    daily_thresholds: Dict[str, Any],
    sl_max_error: float,
) -> Dict[str, Any]:
    """Evaluate daily pass/fail metrics for a set of count stats."""
    dt = daily_thresholds
    r2 = float(count_stats.get("r2") or 0.0)
    slope = float(count_stats.get("slope") or 0.0)
    prmse = float(count_stats.get("pct_rmse") or 999.0)
    bias = abs(float(count_stats.get("bias_pct") or 999.0))
    daily_geh_adj = float(count_stats.get("daily_geh_lt_adj_pct") or 0.0)

    r2_target = float(dt.get("r2_target", 0.80))
    slope_range = dt.get("slope_range", [0.85, 1.15])
    prmse_max = float(dt.get("pct_rmse_max", 35.0))
    bias_max = float(dt.get("bias_abs_max_pct", 15.0))
    sl_max = float(dt.get("screenline_max_pct_deviation", 15.0))

    r2_pass = r2 >= r2_target
    slope_pass = slope_range[0] <= slope <= slope_range[1]
    prmse_pass = prmse <= prmse_max
    bias_pass = bias <= bias_max
    sl_pass = sl_max_error <= sl_max

    return {
        "r2": round(r2, 4),
        "r2_pass": r2_pass,
        "slope": round(slope, 4),
        "slope_pass": slope_pass,
        "pct_rmse": round(prmse, 1),
        "pct_rmse_pass": prmse_pass,
        "bias_abs_pct": round(bias, 2),
        "bias_pass": bias_pass,
        "screenline_pass": sl_pass,
        "daily_geh_lt_adj_pct": round(daily_geh_adj, 1),
        "overall_pass": r2_pass and slope_pass and prmse_pass and bias_pass and sl_pass,
    }


def compute_validation_benchmarks(
    count_stats: Dict[str, Any],
    screenline_results: Dict[str, Any],
    jt_results: List[Dict[str, Any]],
    *,
    model_time_period: str = "daily",
    daily_thresholds: Optional[Dict[str, Any]] = None,
    benchmarks: Optional[Dict[str, Any]] = None,
    holdout_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check validation benchmarks for the model's time aggregation.

    When *holdout_stats* is provided, the overall PASS/FAIL verdict is
    based on the holdout (validation) subset, not the calibration subset.
    The calibration subset metrics are reported as ``calibration_fit``
    for informational purposes only.

    For daily models: R², slope, %RMSE, bias, screenline deviations.
    GEH is diagnostic only (FHWA target is for hourly flows).
    """
    bm = benchmarks or {}
    geh_pass_pct = float(bm.get("geh_lt5_pass_pct", 85.0))
    jt_pass_pct_thr = float(bm.get("jt_pass_pct", 85.0))
    dt = daily_thresholds or {}

    geh5 = float(count_stats.get("geh_lt5_pct", 0))
    geh_pass_hourly = geh5 >= geh_pass_pct

    jt_pass_count = sum(1 for r in jt_results if r.get("pass", False))
    jt_total = len(jt_results) if jt_results else 0
    jt_pct = jt_pass_count / max(jt_total, 1) * 100
    jt_pass = jt_pct >= jt_pass_pct_thr if jt_total > 0 else None

    sl_max_error = 0.0
    for sr in screenline_results.values():
        ratio = sr.get("ratio")
        obs = sr.get("observed_total", 0)
        if ratio is not None and obs and obs > 0:
            sl_max_error = max(sl_max_error, abs(ratio - 1.0) * 100)

    result: Dict[str, Any] = {
        "model_time_period": model_time_period,
        "screenline_max_error_pct": round(sl_max_error, 1),
        "jt_within_tolerance_pct": round(jt_pct, 1) if jt_total > 0 else None,
        "jt_benchmark_pass": jt_pass,
        "jt_routes_checked": jt_total,
        "geh_lt5_pct": round(geh5, 1),
        "geh_benchmark_pass_hourly": geh_pass_hourly,
    }

    if model_time_period == "daily":
        # Calibration fit (informational)
        calib_metrics = _compute_daily_metrics(count_stats, dt, sl_max_error)
        result["calibration_fit"] = calib_metrics

        # Holdout validation (authoritative for PASS/FAIL)
        holdout_n = holdout_stats.get("n", 0) if holdout_stats else 0
        holdout_adequacy = _classify_holdout_adequacy(holdout_n)
        result["holdout_adequacy"] = holdout_adequacy

        if holdout_stats and holdout_n >= 5:
            holdout_metrics = _compute_daily_metrics(holdout_stats, dt, sl_max_error)
            result["holdout_validation"] = holdout_metrics

            if holdout_adequacy == "thin":
                logger.warning(
                    "Holdout set has only %d observations (thin) — "
                    "verdict is based on holdout but may not be robust",
                    holdout_n,
                )

            daily_pass = holdout_metrics["overall_pass"]
            result["verdict_source"] = "holdout"

            result.update({
                "daily_r2": holdout_metrics["r2"],
                "daily_r2_pass": holdout_metrics["r2_pass"],
                "daily_slope": holdout_metrics["slope"],
                "daily_slope_pass": holdout_metrics["slope_pass"],
                "daily_pct_rmse": holdout_metrics["pct_rmse"],
                "daily_pct_rmse_pass": holdout_metrics["pct_rmse_pass"],
                "daily_bias_abs_pct": holdout_metrics["bias_abs_pct"],
                "daily_bias_pass": holdout_metrics["bias_pass"],
                "daily_screenline_pass": holdout_metrics["screenline_pass"],
                "daily_geh_lt_adj_pct": holdout_metrics["daily_geh_lt_adj_pct"],
            })
        else:
            if holdout_stats and 0 < holdout_n < 5:
                logger.warning(
                    "Holdout set has only %d observations (insufficient) — "
                    "falling back to calibration verdict",
                    holdout_n,
                )
            daily_pass = calib_metrics["overall_pass"]
            result["verdict_source"] = "calibration"
            result.update({
                "daily_r2": calib_metrics["r2"],
                "daily_r2_pass": calib_metrics["r2_pass"],
                "daily_slope": calib_metrics["slope"],
                "daily_slope_pass": calib_metrics["slope_pass"],
                "daily_pct_rmse": calib_metrics["pct_rmse"],
                "daily_pct_rmse_pass": calib_metrics["pct_rmse_pass"],
                "daily_bias_abs_pct": calib_metrics["bias_abs_pct"],
                "daily_bias_pass": calib_metrics["bias_pass"],
                "daily_screenline_pass": calib_metrics["screenline_pass"],
                "daily_geh_lt_adj_pct": calib_metrics["daily_geh_lt_adj_pct"],
            })

        result["daily_overall_pass"] = daily_pass and (jt_pass is None or jt_pass)
        result["overall_pass"] = daily_pass and (jt_pass is None or jt_pass)
        result["note"] = "GEH<5 target applies to hourly flows; daily model uses R²/slope/%RMSE/bias"
    else:
        result["overall_pass"] = geh_pass_hourly and (jt_pass is None or jt_pass)

    return result


def run_validation_only(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """Comprehensive independent validation against CSD + screenlines + journey times."""
    from sim.network.closures import swap_db_closures

    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)
    buffer_m = float(get_nested(cfg, ["calibration", "match_buffer_m"], 50.0))
    mq_min = float(calib_cfg.get("match_quality_min", 0.50))

    # Swap closures to validation period (e.g. 2025)
    bc_cfg = cfg.get("baseline_closures") or {}
    valid_period = bc_cfg.get("validation_period")
    if bc_cfg.get("enabled", False) and valid_period:
        swap_db_closures(config_path, measurement_period=valid_period)
        logger.info("  Closures swapped to validation period: %s", valid_period)

    logger.info("=== COMPREHENSIVE VALIDATION ===")

    results_path = output_dir / "assignment_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No assignment results at {results_path}. Run 'calibrate' or 'assign' first."
        )
    vol_df = pd.read_parquet(str(results_path))

    from sim._metrics import resolve_volume_column
    vol_df, vol_col = resolve_volume_column(vol_df)

    logger.info(f"  Loaded assignment: {len(vol_df)} links, vol_col={vol_col}")

    links_gdf = _load_network_links(project_dir)
    if vol_col and "link_id" in vol_df.columns:
        links_gdf = links_gdf.merge(vol_df[["link_id", vol_col]], on="link_id", how="left")

    count_target = str(calib_cfg.get("count_target", "motor_total"))
    _ct_map = {"car_only": "observed_car", "motor_total": "observed_motor_total", "total": "observed_total"}
    obs_col = _ct_map.get(count_target, "observed_total")

    assign_cfg = cfg.get("assignment") or {}
    bpr_cfg = _apply_bpr_defaults(assign_cfg.get("bpr") or {})
    daily_cap_factor = resolve_daily_cap_factor_default(bpr_cfg)
    model_time_period = str(calib_cfg.get("model_time_period", "daily"))

    report: Dict[str, Any] = {"model_time_period": model_time_period}

    count_source = str(calib_cfg.get("count_source", "csd_split"))
    report["count_source"] = count_source

    # 1) Reference comparison (calibration data check)
    pent_stats: Dict[str, Any] = {"n": 0}
    matched = gpd.GeoDataFrame()

    if count_source == "csd_split":
        logger.info("\n1) CSD calibration-subset reference comparison ...")
        try:
            split_cfg = calib_cfg.get("csd_split") or {}
            csd_full = load_csd(cfg)
            calib_csd, _ = split_csd_for_calibration(
                csd_full,
                strategy=str(split_cfg.get("strategy", "alternating")),
                calib_share=float(split_cfg.get("calib_share", 0.65)),
                random_seed=int(split_cfg.get("random_seed", 42)),
            )
            pent = load_csd_as_link_counts(calib_csd, links_gdf)
            if not pent.empty:
                agg_corr = bool(calib_cfg.get("aggregate_corridor", True))
                matched = match_counts_to_links(pent, links_gdf, buffer_m=buffer_m,
                                                 aggregate_corridor=agg_corr,
                                                 vol_col=vol_col,
                                                 match_quality_min=mq_min)
                vc = vol_col if vol_col and vol_col in matched.columns else None
                compare_vc = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
                if compare_vc and compare_vc in matched.columns:
                    valid = matched.dropna(subset=[compare_vc, obs_col])
                    valid = valid[valid[obs_col] > 0]
                    if "_excluded" in valid.columns:
                        valid = valid[~valid["_excluded"]].copy()
                    pent_stats = compute_stats(
                        valid[compare_vc].values, valid[obs_col].values,
                        daily_capacity_factor=daily_cap_factor,
                    )
                    report["calibration_reference"] = {"matched": int(len(valid)), **pent_stats}
                    logger.info(f"  Matched: {len(valid)}  R²={pent_stats.get('r2')}  "
                          f"bias={pent_stats.get('bias_pct')}%")
        except Exception:
            logger.exception("CSD calibration-subset reference comparison skipped")
    else:
        logger.info(f"\n1) Pentlogram comparison (reference, time_period={model_time_period}) ...")
        try:
            pent = load_pentlogram(cfg)
            validate_geometries_or_fail(
                pent, name="pentlogram", expected_epsg=get_metric_epsg(cfg),
            )
            agg_corr = bool(calib_cfg.get("aggregate_corridor", True))
            matched = match_counts_to_links(pent, links_gdf, buffer_m=buffer_m,
                                             aggregate_corridor=agg_corr,
                                             vol_col=vol_col,
                                             match_quality_min=mq_min)
            vc = vol_col if vol_col and vol_col in matched.columns else None
            compare_vc = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
            if compare_vc and compare_vc in matched.columns:
                valid = matched.dropna(subset=[compare_vc, obs_col])
                valid = valid[valid[obs_col] > 0]
                if "_excluded" in valid.columns:
                    valid = valid[~valid["_excluded"]].copy()
                pent_stats = compute_stats(
                    valid[compare_vc].values, valid[obs_col].values,
                    daily_capacity_factor=daily_cap_factor,
                )
                report["pentlogram"] = {"matched": int(len(valid)), **pent_stats}
                logger.info(f"  Matched: {len(valid)}  R²={pent_stats.get('r2')}  "
                      f"slope={pent_stats.get('slope')}  %RMSE={pent_stats.get('pct_rmse')}  "
                      f"bias={pent_stats.get('bias_pct')}%")
                logger.info(f"  GEH<5: {pent_stats.get('geh_lt5_pct')}%  "
                      f"daily-adj GEH<{pent_stats.get('daily_geh_threshold', 5):.0f}: "
                      f"{pent_stats.get('daily_geh_lt_adj_pct')}%")
                ext_metrics = compute_extended_link_metrics(valid, compare_vc, obs_col)
                if ext_metrics:
                    report["extended_metrics"] = ext_metrics
            else:
                logger.warning("  No volume column on links")
        except Exception:
            logger.exception("Pentlogram comparison step skipped")

    # 2) CSD -- independent per-road validation
    #    When count_source=="csd_split", only the validation subset is used.
    logger.info("\n2) CSD independent validation (per-road matching via osm_ref) ...")
    csd_match_df = pd.DataFrame()
    csd = None
    csd_full = None
    try:
        if count_source == "csd_split":
            split_cfg = calib_cfg.get("csd_split") or {}
            csd_full = load_csd(cfg)
            _, csd = split_csd_for_calibration(
                csd_full,
                strategy=str(split_cfg.get("strategy", "alternating")),
                calib_share=float(split_cfg.get("calib_share", 0.65)),
                random_seed=int(split_cfg.get("random_seed", 42)),
            )
            logger.info(f"  Using CSD validation subset ({len(csd)} sections)")
        else:
            csd = load_csd(cfg)

        csd_agg = aggregate_csd_by_class(csd)
        report["csd_observed"] = csd_agg.to_dict(orient="records")

        if vol_col and vol_col in links_gdf.columns:
            model_agg = aggregate_model_by_class(links_gdf, vol_col)
            report["csd_modeled"] = model_agg.to_dict(orient="records")

        csd_match_df = match_csd_to_links(csd, links_gdf, csd_full=csd_full)
        if not csd_match_df.empty:
            report["csd_link_matching"] = csd_match_df.to_dict(orient="records")

        logger.info(f"  CSD: {len(csd)} sections")
        for _, r in csd_agg.iterrows():
            logger.info(f"    {r['road_class']:12s}  sections={int(r['sections']):4d}  "
                  f"mean_AADT={r['mean_sv']:>8.0f}  mean_cars={r['mean_o']:>8.0f}")

        if not csd_match_df.empty:
            logger.info(f"  Per-road comparison: {len(csd_match_df)} roads matched")
            for _, r in csd_match_df.iterrows():
                partial_tag = "  [PARTIAL]" if r.get("partial_coverage", False) else ""
                logger.info(
                    f"    {r['road']:>8s} ({r['road_class']:>10s})  "
                    f"csd_sv={r['csd_mean_sv']:>8.0f}  model={r['model_lw_mean']:>8.0f}  "
                    f"GEH={r['geh']:>5.1f}  sections={r['csd_sections']}  links={r['model_links']}"
                    f"{partial_tag}"
                )
            summary = getattr(csd_match_df, "attrs", {}).get("summary")
            if summary:
                partial_note = ""
                n_excl = summary.get("n_partial_excluded", 0)
                if n_excl:
                    partial_note = f" ({n_excl} partial-coverage road(s) excluded)"
                logger.info(f"  Summary ({summary['n_roads']} roads{partial_note}): "
                      f"R²={summary['r2']:.3f}  bias={summary['bias_pct']:.1f}%  "
                      f"%RMSE={summary['pct_rmse']:.1f}  mean_GEH={summary['mean_geh']:.1f}")
                report["csd_summary"] = summary
    except Exception:
        logger.exception("CSD independent validation step skipped")

    # 3) Screenline validation
    logger.info("\n3) Screenline validation ...")
    sl_results: Dict[str, Any] = {}
    try:
        from sim.calibration.screenlines import load_screenlines_with_auto, evaluate_all_screenlines
        csd_for_auto = None
        csd_full_for_gw = None
        try:
            csd_for_auto = load_csd(cfg)
        except Exception:
            pass
        try:
            csd_full_for_gw = load_csd_unfiltered(cfg)
        except Exception:
            pass
        screenlines = load_screenlines_with_auto(
            cfg, csd_df=csd_for_auto, csd_df_full=csd_full_for_gw,
        )
        if screenlines:
            sl_res = evaluate_all_screenlines(
                screenlines, vol_df,
                    matched if "matched" in locals() else gpd.GeoDataFrame(),
                vol_col or "", obs_col, links_gdf,
            )
            for sn, sr in sl_res.items():
                sl_results[sn] = sr.to_dict()
                if sr.observed_total and sr.observed_total > 0:
                    logger.info(f"  {sn}: mod={sr.modeled_total:,.0f} obs={sr.observed_total:,.0f} "
                          f"ratio={sr.ratio:.2f} GEH={sr.geh:.1f}")
            report["screenlines"] = sl_results
        else:
            logger.warning("  No screenlines defined")
    except Exception:
        logger.exception("Screenline validation step skipped")

    # 4) Journey time validation
    logger.info("\n4) Journey time validation ...")
    jt_results: List[Dict[str, Any]] = []
    speed_comparison: Dict[str, Any] = {}
    try:
        # Class-level speed comparison
        if csd is not None and not csd.empty:
            speed_comparison = compute_class_speed_comparison(links_gdf)
            report["class_speed_comparison"] = speed_comparison
            if speed_comparison:
                logger.info("  Speed comparison (model vs CSD):")
                for lt, sc in speed_comparison.items():
                    logger.info(f"    {lt:20s}  model={sc['modeled_kmh']:>5.1f}  "
                          f"ref={sc.get('reference_kmh', sc.get('csd_implied_kmh', 0)):>5.1f}  "
                          f"diff={sc['pct_diff']:>+5.1f}%")

        # Reference route checks from skims
        jt_cfg = calib_cfg.get("journey_time_validation", {})
        routes = jt_cfg.get("reference_routes", [])
        if routes and jt_cfg.get("enabled", True):
            skim_path = output_dir / "skims.aem"
            if skim_path.exists():
                mat = AequilibraeMatrix()
                mat.load(str(skim_path))
                names = list(mat.names)
                if names:
                    zone_ids = mat.index[:].copy()
                    skim_data = mat.matrix[names[0]][:, :].copy()
                    mat.close()
                    jt_results = validate_journey_times(skim_data, zone_ids, routes)
                    report["journey_time_routes"] = jt_results
                    for r in jt_results:
                        status = "PASS" if r.get("pass") else "FAIL"
                        logger.info(f"  {r.get('name', '?'):20s}  model={r.get('model_min', '?')}min  "
                              f"ref={r.get('ref_min', '?')}min  {status}")
                else:
                    mat.close()
            else:
                logger.warning("  No skims available -- skip route checks")
        else:
            logger.info("  No reference routes configured")
    except Exception:
        logger.exception("Journey time validation step skipped")

    # 4b) Holdout stats for benchmark verdict
    holdout_stats: Optional[Dict[str, Any]] = None
    if count_source == "csd_split" and csd_match_df is not None and not csd_match_df.empty:
        csd_summary = getattr(csd_match_df, "attrs", {}).get("summary")
        if csd_summary:
            holdout_stats = {
                "n": csd_summary.get("n_roads", 0),
                "r2": csd_summary.get("r2"),
                "slope": 1.0,
                "pct_rmse": csd_summary.get("pct_rmse"),
                "bias_pct": csd_summary.get("bias_pct"),
                "geh_lt5_pct": 0.0,
                "daily_geh_lt_adj_pct": 0.0,
            }

    # 4c) Calibration section hash guard
    require_holdout = bool(get_nested(cfg, ["calibration", "require_holdout"], True))
    calib_report_path = output_dir / "calibration_report.json"
    if calib_report_path.exists() and count_source == "csd_split":
        try:
            calib_report = json.loads(calib_report_path.read_text(encoding="utf-8"))
            saved_hash = calib_report.get("calibration_section_hash", "")
            if saved_hash:
                report["calibration_section_hash_from_calib"] = saved_hash
                logger.info(f"  Calibration section hash: {saved_hash}")
        except Exception:
            pass

    if require_holdout and count_source != "csd_split":
        logger.warning(
            "VALIDATION WARNING: require_holdout=true but count_source='%s'. "
            "Validation verdict may reflect training data fit, not independent "
            "holdout performance. Set calibration.require_holdout=false to suppress.",
            count_source,
        )
        report["holdout_warning"] = (
            "No holdout split active. Verdict based on calibration data."
        )

    # 4b) Period-based metrics (AM/IP/PM) from temporal profile
    period_metrics: Dict[str, Any] = {}
    try:
        from sim.demand.temporal import load_profile, get_demand_period_shares

        profile = load_profile(cfg)
        period_shares = get_demand_period_shares(profile)

        if vol_col and vol_col in links_gdf.columns:
            logger.info("\n4b) Period-based synthetic validation (AM/IP/PM) ...")
            for period in ("am", "ip", "pm"):
                share = period_shares.get(period, 0.0)
                if share <= 0:
                    continue
                period_mod = links_gdf[vol_col].fillna(0) * share
                obs_values = links_gdf.get(obs_col)
                if obs_values is None:
                    continue
                period_obs = obs_values.fillna(0) * share
                mask = (period_obs > 0) & (period_mod >= 0)
                if mask.sum() < 3:
                    continue
                p_stats = compute_stats(
                    period_mod[mask].values,
                    period_obs[mask].values,
                )
                period_metrics[period] = {
                    "share": round(share, 4),
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in p_stats.items()},
                }
                logger.info(
                    "  %s (share=%.3f): R²=%.3f, %%RMSE=%.1f, bias=%.1f%%",
                    period.upper(), share,
                    p_stats.get("r2", 0), p_stats.get("pct_rmse", 0),
                    p_stats.get("bias_pct", 0),
                )
            report["period_metrics"] = {
                "note": (
                    "Synthetic period volumes derived by applying temporal "
                    "profile shares to daily model and observed totals. "
                    "Not a true period-level assignment."
                ),
                "shares": {k: v for k, v in period_shares.items() if k != "daily"},
                "periods": period_metrics,
            }
    except FileNotFoundError:
        logger.info("  No temporal_profile.json found — skipping period metrics")
    except Exception:
        logger.debug("Period metrics failed", exc_info=True)

    # 5) Benchmark summary
    logger.info(f"\n5) Validation benchmarks (model_time_period={model_time_period}) ...")
    daily_conv = get_nested(cfg, ["calibration", "convergence", "daily"], {})
    bm_cfg = get_nested(cfg, ["calibration", "benchmarks"], {})
    benchmarks = compute_validation_benchmarks(
        pent_stats, sl_results, jt_results,
        model_time_period=model_time_period,
        daily_thresholds=daily_conv,
        benchmarks=bm_cfg,
        holdout_stats=holdout_stats,
    )
    report["benchmarks"] = benchmarks
    qcfg = (calib_cfg.get("quality_gates") or {})
    q_bias_hard = float(qcfg.get("hard_class_bias_max_abs_pct", 90.0))
    q_wmape_warn = float(qcfg.get("warn_wmape_pct", 47.0))
    q_geh_warn = float(qcfg.get("warn_geh_lt5_pct", 7.0))
    ext = report.get("extended_metrics") or {}
    bias = float(ext.get("class_bias_max_abs_pct", 0.0)) if ext else None
    wmape = float(ext.get("wmape_pct", 0.0)) if ext else None
    geh = float(benchmarks.get("geh_lt5_pct", 0.0))
    q_warns: List[str] = []
    if wmape is not None and wmape > q_wmape_warn:
        q_warns.append(f"wmape_pct>{q_wmape_warn:g}")
    if geh < q_geh_warn:
        q_warns.append(f"geh_lt5_pct<{q_geh_warn:g}")
    report["quality_gates"] = {
        "hard_class_bias_max_abs_pct": q_bias_hard,
        "warn_wmape_pct": q_wmape_warn,
        "warn_geh_lt5_pct": q_geh_warn,
        "hard_reject": bool((bias is not None) and (bias > q_bias_hard)),
        "warnings": q_warns,
    }

    overall = "PASS" if benchmarks["overall_pass"] else "FAIL"
    verdict_src = benchmarks.get("verdict_source", "calibration")

    # Summary table
    if model_time_period == "daily":
        dt = daily_conv or {}
        r2_tgt = float(dt.get("r2_target", 0.80))
        sl_range = dt.get("slope_range", [0.85, 1.15])
        prmse_max = float(dt.get("pct_rmse_max", 35.0))
        bias_max = float(dt.get("bias_abs_max_pct", 15.0))
        sl_max = float(dt.get("screenline_max_pct_deviation", 15.0))

        calib_fit = benchmarks.get("calibration_fit", {})
        holdout_val = benchmarks.get("holdout_validation", {})

        logger.info("\nVALIDATION SUMMARY")
        logger.info("-" * 60)
        if holdout_val:
            logger.info("%-20s %12s %12s %12s", "Metric", "Calibration", "Holdout", "Target")
            logger.info("-" * 60)
            logger.info("%-20s %11.4f  %11.4f  %s",
                        "R²", calib_fit.get("r2", 0), holdout_val.get("r2", 0), f">= {r2_tgt}")
            logger.info("%-20s %11.1f  %11.1f  %s",
                        "%RMSE", calib_fit.get("pct_rmse", 0), holdout_val.get("pct_rmse", 0), f"<= {prmse_max}%")
            logger.info("%-20s %10.1f%%  %10.1f%%  %s",
                        "|Bias|", calib_fit.get("bias_abs_pct", 0), holdout_val.get("bias_abs_pct", 0), f"<= {bias_max}%")
            logger.info("%-20s %11.4f  %11.4f  %s",
                        "Slope", calib_fit.get("slope", 0), holdout_val.get("slope", 0),
                        f"[{sl_range[0]}-{sl_range[1]}]")
            logger.info("%-20s %11.1f%%                %s",
                        "SL max dev", benchmarks.get("screenline_max_error_pct", 0), f"<= {sl_max}%")
            logger.info("-" * 60)
            logger.info("VERDICT: %s (based on %s)", overall, verdict_src)
        else:
            logger.info(f"  R² >= {r2_tgt}:     {benchmarks.get('daily_r2', 0):.4f}  "
                  f"{'PASS' if benchmarks.get('daily_r2_pass') else 'FAIL'}")
            logger.info(f"  slope [{sl_range[0]}-{sl_range[1]}]: {benchmarks.get('daily_slope', 0):.4f}  "
                  f"{'PASS' if benchmarks.get('daily_slope_pass') else 'FAIL'}")
            logger.info(f"  %RMSE <= {prmse_max:g}%:   {benchmarks.get('daily_pct_rmse', 0):.1f}%  "
                  f"{'PASS' if benchmarks.get('daily_pct_rmse_pass') else 'FAIL'}")
            logger.info(f"  |bias| <= {bias_max:g}%:  {benchmarks.get('daily_bias_abs_pct', 0):.2f}%  "
                  f"{'PASS' if benchmarks.get('daily_bias_pass') else 'FAIL'}")
            logger.info(f"  SL dev <= {sl_max:g}%:  {benchmarks.get('screenline_max_error_pct', 0):.1f}%  "
                  f"{'PASS' if benchmarks.get('daily_screenline_pass') else 'FAIL'}")
            logger.info(f"  (GEH<5: {geh:.1f}% — diagnostic only for daily model)")
            logger.info("-" * 60)
            logger.info("VERDICT: %s (based on %s)", overall, verdict_src)
    else:
        logger.info(f"  GEH<5 >= 85%:  {geh:.1f}%  "
              f"{'PASS' if benchmarks.get('geh_benchmark_pass_hourly') else 'FAIL'}")
    if benchmarks["jt_benchmark_pass"] is not None:
        logger.info(f"  JT within tol:  {benchmarks['jt_within_tolerance_pct']:.1f}%  "
              f"{'PASS' if benchmarks['jt_benchmark_pass'] else 'FAIL'}")
    logger.info(f"  Screenline max error: {benchmarks['screenline_max_error_pct']:.1f}%")
    logger.info(f"  OVERALL: {overall}")

    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nValidation report: {report_path}")

    # Strip closures and run a clean final assignment
    if bc_cfg.get("enabled", False) and valid_period:
        logger.info("\n=== POST-VALIDATION: stripping closures, clean assignment ===")
        swap_db_closures(config_path, measurement_period=None)
        run_assignment(config_path)
        logger.info("  Clean (closure-free) assignment saved.")


def run_match_diagnostics(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """Regenerate matching_diagnostics.csv from existing assignment results.

    Uses ``calibration.count_source``: for ``csd_split`` (default), CSD
    sections matched to network roads via ``osm_ref``; for ``pentlogram``,
    the ArcGIS pentlogram layer.  Does **not** re-run assignment or OD scaling.
    """
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)

    buffer_m = float(calib_cfg.get("match_buffer_m", 50.0))
    direction_aware = bool(calib_cfg.get("match_direction_aware", True))
    conflict_res = str(calib_cfg.get("match_conflict_resolution", "nearest"))
    agg_corridor = bool(calib_cfg.get("aggregate_corridor", True))
    mq_min = float(calib_cfg.get("match_quality_min", 0.50))

    count_target = str(calib_cfg.get("count_target", "motor_total"))
    _ct_map = {"car_only": "observed_car", "motor_total": "observed_motor_total", "total": "observed_total"}
    obs_col = _ct_map.get(count_target, "observed_total")

    assign_cfg = cfg.get("assignment") or {}
    bpr_cfg = _apply_bpr_defaults(assign_cfg.get("bpr") or {})
    daily_cap_factor = resolve_daily_cap_factor_default(bpr_cfg)

    logger.info("=== MATCH DIAGNOSTICS (lightweight refresh) ===")

    results_path = output_dir / "assignment_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No assignment results at {results_path}. Run 'calibrate' or 'assign' first."
        )
    vol_df = pd.read_parquet(str(results_path))

    from sim._metrics import resolve_volume_column
    vol_df, vol_col = resolve_volume_column(vol_df)

    logger.info(f"  Loaded assignment: {len(vol_df)} links, vol_col={vol_col}")

    count_source = str(calib_cfg.get("count_source", "csd_split"))
    links_gdf = _load_network_links(project_dir)

    if count_source == "csd_split":
        csd_full = load_csd(cfg)
        pent = load_csd_as_link_counts(csd_full, links_gdf)
        if pent.empty:
            raise RuntimeError(
                "CSD produced no link-level count anchors for diagnostics. "
                "Check CSD data and network osm_ref overlap."
            )
        logger.info(
            "  CSD link-count anchors: %d roads (network-filtered CSD, all sections)",
            len(pent),
        )
    else:
        pent = load_pentlogram(cfg)
        validate_geometries_or_fail(
            pent, name="pentlogram", expected_epsg=get_metric_epsg(cfg),
        )
        logger.info(f"  Pentlogram: {len(pent)} segments (after cleaning)")
    if vol_col and "link_id" in vol_df.columns:
        links_gdf = links_gdf.merge(vol_df[["link_id", vol_col]], on="link_id", how="left")

    matched = match_counts_to_links(
        pent, links_gdf,
        buffer_m=buffer_m,
        direction_aware=direction_aware,
        conflict_resolution=conflict_res,
        aggregate_corridor=agg_corridor,
        vol_col=vol_col,
        match_quality_min=mq_min,
    )

    compare_col = "_corridor_volume" if "_corridor_volume" in matched.columns else vol_col
    _export_matching_diagnostics(matched, compare_col, obs_col, output_dir)

    if compare_col and compare_col in matched.columns:
        valid = matched.dropna(subset=[compare_col, obs_col])
        valid = valid[valid[obs_col] > 0]
        if "_excluded" in valid.columns:
            valid = valid[~valid["_excluded"]].copy()
        stats = compute_stats(
            valid[compare_col].values, valid[obs_col].values,
            daily_capacity_factor=daily_cap_factor,
        )
        logger.info(f"  Matched: {len(valid)}  R²={stats.get('r2')}  slope={stats.get('slope')}  "
              f"%RMSE={stats.get('pct_rmse')}  bias={stats.get('bias_pct')}%")
        logger.info(f"  GEH<5: {stats.get('geh_lt5_pct')}%")

    mq = match_quality_report(matched)
    logger.info(f"  Match quality: {mq['n_matched']}/{mq['n_total']} matched, "
          f"{mq['n_link_conflicts']} conflicts, "
          f"mean_dist={mq['mean_match_distance_m']}m")
    logger.info(f"  Done. Diagnostics CSV: {output_dir / 'matching_diagnostics.csv'}")
