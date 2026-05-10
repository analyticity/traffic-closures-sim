"""Legacy FSM iterative calibration and multi-stage pipeline.

``run_calibration`` is the assign → compare → scale loop.  ``run_multistage_calibration``
chains gravity/IPF, gateway pre-calib, ODME, and screenline fine-tuning.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml as _yaml
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix

from sim.assignment import execute_assignment, _detect_volume_col
from sim.calibration.context import (
    _CalibrationContext,
    _ensure_dir,
    _sr_val,
)
from sim.calibration.state import CalibrationState
from sim.calibration.matching import match_counts_to_links, match_quality_report
from sim.calibration.metrics import (
    compute_stats,
    compute_extended_link_metrics,
    compute_class_volume_breakdown,
    _coarse_road_class,
    _compute_class_residuals,
    _supplement_class_ratios_from_screenlines,
)
from sim.calibration.gateway import _apply_gateway_calibration
from sim.calibration.validation import _check_final_convergence
from sim.calibration.odme import run_odme_calibration, _spiess_update_step
from sim.io_project import load_config
from sim.calibration.screenlines import evaluate_all_screenlines

logger = logging.getLogger(__name__)


def run_multistage_calibration(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """Multi-stage pipeline: gravity recalib → gateway pre-calib → ODME → screenline fine-tune.

    Stage 1: Re-calibrate gravity model + IPF using skims to improve seed.
    Stage 2: Gateway pre-calibration with higher damping to fix external demand.
    Stage 3: ODME with tighter elasticity bounds (seed is already improved).
    Stage 4: Screenline-only fine-tuning pass with low damping.
    """
    cfg = load_config(config_path)
    calib_cfg = cfg.get("calibration") or {}
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))
    ms_cfg = calib_cfg.get("multistage") or {}
    max_total_change_pct = float(ms_cfg.get("max_total_change_pct", 50.0))

    stage_reports: List[Dict[str, Any]] = []

    logger.info("=== MULTI-STAGE PIPELINE CALIBRATION ===")

    # ---- Stage 1: Gravity recalibration + IPF ----
    logger.info("\n--- Stage 1: Gravity recalibration + IPF ---")
    skim_path = output_dir / "skims.aem"
    stage1_improved = False
    if skim_path.exists():
        try:
            from sim.distribution import calibrate_gravity_simple, run_ipf

            mat = AequilibraeMatrix()
            mat.load(str(matrix_path))
            mat.computational_view([core_name])
            seed = mat.matrix[core_name][:, :].copy().astype(np.float64)
            seed_total = float(seed.sum())

            skim_mat = AequilibraeMatrix()
            skim_mat.load(str(skim_path))
            skim_names = list(skim_mat.names)
            impedance = skim_mat.matrix[skim_names[0]][:, :].copy() if skim_names else None
            skim_mat.close()

            if impedance is not None:
                params = calibrate_gravity_simple(seed, impedance)
                logger.info(f"  Gravity params: {params}")

                beta = params.get("beta", 0.0001)
                gravity = np.exp(-beta * impedance)
                np.fill_diagonal(gravity, 0)

                row_targets = seed.sum(axis=1)
                col_targets = seed.sum(axis=0)
                valid_rc = (row_targets > 0) & (col_targets > 0)
                if valid_rc.any():
                    ipf_result = run_ipf(gravity, row_targets, col_targets)
                    blended = 0.7 * ipf_result + 0.3 * seed
                    blended_total = float(blended.sum())
                    if blended_total > 0:
                        blended *= seed_total / blended_total
                    # Guard-rail: cap total demand change
                    final_total = float(blended.sum())
                    change_pct = abs(final_total - seed_total) / max(seed_total, 1) * 100
                    if change_pct > max_total_change_pct:
                        logger.warning(
                            f"  Stage 1: demand change {change_pct:.1f}% exceeds "
                            f"cap {max_total_change_pct:.0f}%, rescaling to seed total"
                        )
                        blended *= seed_total / max(final_total, 1.0)
                    mat.matrix[core_name][:, :] = blended
                    mat.save()
                    stage1_improved = True
                    logger.info(
                        f"  Stage 1 complete: demand total {float(blended.sum()):,.0f} "
                        f"(seed was {seed_total:,.0f})"
                    )

            mat.close()
        except Exception:
            logger.exception("Stage 1 (gravity recalibration) failed, continuing with seed")

    stage_reports.append({
        "stage": 1,
        "name": "gravity_recalibration",
        "applied": stage1_improved,
    })
    if not stage1_improved:
        logger.info("  Stage 1 skipped (no skims or error)")

    # ---- Stage 2: Gateway pre-calibration with higher damping ----
    logger.info("\n--- Stage 2: Gateway pre-calibration ---")
    stage2_applied = False
    gw_cal_cfg = calib_cfg.get("gateway_calibration") or {}
    if gw_cal_cfg.get("enabled", True):
        try:
            ctx_tmp = _CalibrationContext(config_path)

            if ctx_tmp.gw_cal_enabled and ctx_tmp.gw_zone_map and ctx_tmp.gw_observed:
                data = ctx_tmp.mat.matrix[core_name]
                demand = data[:, :].copy().astype(np.float64)

                vol_df_gw, _, _ = ctx_tmp.run_assignment()
                vol_col_gw = _detect_volume_col(vol_df_gw)

                if vol_col_gw:
                    gw_modeled = _compute_gateway_modeled_volumes(
                        vol_df_gw, ctx_tmp.screenlines, vol_col_gw,
                        sl_gw_map=ctx_tmp.sl_gw_map,
                    )
                    pre_damping = float(gw_cal_cfg.get("pre_damping", 0.4))
                    gw_corrections = _apply_gateway_calibration(
                        demand, ctx_tmp.gw_zone_map, ctx_tmp.gw_observed, gw_modeled,
                        damping=pre_damping,
                        min_factor=0.5,
                        max_factor=2.0,
                        seed_lower=ctx_tmp.seed_lower,
                        seed_upper=ctx_tmp.seed_upper,
                    )
                    if gw_corrections:
                        data[:, :] = demand
                        ctx_tmp.mat.save()
                        stage2_applied = True
                        for gc_line in gw_corrections:
                            logger.info(f"    {gc_line}")

            ctx_tmp.mat.close()
            ctx_tmp.project.close()
        except Exception:
            logger.exception("Stage 2 (gateway pre-calibration) failed")

    stage_reports.append({
        "stage": 2,
        "name": "gateway_pre_calibration",
        "applied": stage2_applied,
    })

    # ---- Stage 3: ODME with tighter bounds ----
    logger.info("\n--- Stage 3: ODME (tight bounds) ---")
    pre_odme_total = 0.0
    try:
        _pre_mat = AequilibraeMatrix()
        _pre_mat.load(str(matrix_path))
        _pre_mat.computational_view([core_name])
        pre_odme_total = float(_pre_mat.matrix[core_name][:, :].sum())
        _pre_mat.close()
    except Exception:
        pass

    raw_yaml_path = Path(config_path).expanduser().resolve()
    raw = _yaml.safe_load(raw_yaml_path.read_text(encoding="utf-8")) or {}
    raw_calib = raw.get("calibration") or {}
    raw_calib.setdefault("odme", {})
    ms_max_dev = float(ms_cfg.get("multistage_max_deviation", 3.0))
    raw_calib["odme"]["max_deviation"] = ms_max_dev
    raw_calib["method"] = "odme"
    raw["calibration"] = raw_calib

    tmp_cfg = raw_yaml_path.parent / "_tmp_multistage_odme.yaml"
    tmp_cfg.write_text(
        _yaml.dump(raw, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    try:
        run_odme_calibration(str(tmp_cfg))
        report_path = output_dir / "calibration_report.json"
        stage3_report = {}
        if report_path.exists():
            stage3_report = json.loads(report_path.read_text(encoding="utf-8"))
        stage_reports.append({
            "stage": 3,
            "name": "odme_tight_bounds",
            "applied": True,
            "final": stage3_report.get("final", {}),
            "iterations": stage3_report.get("iterations", 0),
        })
    except Exception:
        logger.exception("Stage 3 (ODME) failed")
        stage_reports.append({"stage": 3, "name": "odme_tight_bounds", "applied": False})
    finally:
        tmp_cfg.unlink(missing_ok=True)

    if pre_odme_total > 0:
        try:
            _post_mat = AequilibraeMatrix()
            _post_mat.load(str(matrix_path))
            _post_mat.computational_view([core_name])
            _post_data = _post_mat.matrix[core_name]
            post_total = float(_post_data[:, :].sum())
            pct_change = (post_total - pre_odme_total) / pre_odme_total * 100
            if abs(pct_change) > max_total_change_pct:
                logger.warning(
                    f"  Post-ODME demand change {pct_change:+.1f}% exceeds cap "
                    f"{max_total_change_pct:.0f}%, rescaling to pre-ODME total"
                )
                _post_data[:, :] = (
                    _post_data[:, :].astype(np.float64)
                    * (pre_odme_total / max(post_total, 1.0))
                )
                _post_mat.save()
            else:
                logger.info(f"  Post-ODME demand change: {pct_change:+.1f}% (within cap)")
            _post_mat.close()
        except Exception:
            logger.exception("Post-ODME demand cap check failed")

    # ---- Stage 4: Screenline fine-tuning ----
    logger.info("\n--- Stage 4: Screenline fine-tuning ---")
    stage4_applied = False
    try:
        ctx_final = _CalibrationContext(config_path)
        mat_f = ctx_final.mat
        data_f = mat_f.matrix[core_name]
        demand_f = data_f[:, :].copy().astype(np.float64)

        vol_df_f, _, sl_matrices_f = ctx_final.run_assignment()
        vol_col_f = _detect_volume_col(vol_df_f)

        if vol_col_f and ctx_final.screenlines and sl_matrices_f:
            links_with_vol_f = ctx_final.links_gdf.copy()
            links_with_vol_f = links_with_vol_f.merge(
                vol_df_f[["link_id", vol_col_f]], on="link_id", how="left",
            )
            matched_f = match_counts_to_links(
                ctx_final.pent, links_with_vol_f,
                buffer_m=ctx_final.buffer_m,
                direction_aware=ctx_final.direction_aware,
                vol_col=vol_col_f,
                match_quality_min=ctx_final.match_quality_min,
            )
            sl_res_f = evaluate_all_screenlines(
                ctx_final.screenlines, vol_df_f, matched_f,
                vol_col_f, ctx_final.obs_col, ctx_final.links_gdf,
            )
            sl_results_f = {sn: sr.to_dict() for sn, sr in sl_res_f.items()}

            fine_damping = 0.1
            n_sl_active_f = sum(
                1 for sn in sl_matrices_f
                if _sr_val(sl_results_f.get(sn, {}), "observed_total") > 0
                and _sr_val(sl_results_f.get(sn, {}), "modeled_total") > 0
            )
            sl_damp_f = fine_damping / max(np.sqrt(n_sl_active_f), 1.0)

            for sl_name, sl_od_raw in sl_matrices_f.items():
                obs_sl = _sr_val(sl_results_f.get(sl_name, {}), "observed_total")
                mod_sl = _sr_val(sl_results_f.get(sl_name, {}), "modeled_total")
                if obs_sl <= 0 or mod_sl <= 0:
                    continue
                ratio = obs_sl / mod_sl
                if ratio > ctx_final.sl_ratio_max or ratio < ctx_final.sl_ratio_min:
                    continue

                sl_od = np.asarray(sl_od_raw, dtype=np.float64).reshape(demand_f.shape)
                proportion = np.where(
                    demand_f > 0,
                    np.clip(sl_od / np.maximum(demand_f, 1e-9), 0.0, 1.0),
                    0.0,
                )
                adjustment = 1.0 + sl_damp_f * (ratio - 1.0) * proportion
                np.clip(adjustment, ctx_final.sl_clip_min, ctx_final.sl_clip_max, out=adjustment)
                demand_f *= adjustment

            np.clip(demand_f, ctx_final.seed_lower, ctx_final.seed_upper, out=demand_f)
            np.maximum(demand_f, 0.0, out=demand_f)
            data_f[:, :] = demand_f
            mat_f.save()
            stage4_applied = True
            logger.info(f"  Stage 4 complete: demand total={demand_f.sum():,.0f}")

        mat_f.close()
        ctx_final.project.close()
    except Exception:
        logger.exception("Stage 4 (screenline fine-tuning) failed")

    stage_reports.append({
        "stage": 4,
        "name": "screenline_fine_tuning",
        "applied": stage4_applied,
    })

    report = {
        "method": "multistage",
        "stages": stage_reports,
    }
    report_path = output_dir / "calibration_report.json"
    if report_path.exists():
        try:
            odme_report = json.loads(report_path.read_text(encoding="utf-8"))
            report["final"] = odme_report.get("final", {})
            report["history"] = odme_report.get("history", [])
            report["iterations"] = odme_report.get("iterations", 0)
            report["converged"] = odme_report.get("converged", False)
            report["screenlines"] = odme_report.get("screenlines", {})
        except Exception:
            pass
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nMulti-stage report: {report_path}")


def run_calibration(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """FSM iterative calibration: assign → compare → scale → repeat."""
    ctx = _CalibrationContext(config_path)
    calib_cfg = ctx.calib_cfg
    mat = ctx.mat
    cached_graph = ctx.cached_graph
    links_gdf = ctx.links_gdf
    screenlines = ctx.screenlines
    output_dir = ctx.output_dir
    project_dir = ctx.project_dir
    matrix_path = ctx.matrix_path
    core_name = ctx.core_name
    algorithm = ctx.algorithm
    max_iter_assign = ctx.max_iter_assign
    rgap = ctx.rgap
    gc_field = ctx.gc_field
    gc_mult = ctx.gc_mult
    gc_vot = ctx.gc_vot
    cfg_bpr = ctx.cfg_bpr
    cfg_multi = ctx.cfg_multi
    daily_cap_factor = ctx.daily_cap_factor
    cores = ctx.cores
    buffer_m = ctx.buffer_m
    direction_aware = ctx.direction_aware
    agg_corridor = ctx.agg_corridor
    mq_min = ctx.match_quality_min
    obs_col = ctx.obs_col
    count_target = ctx.count_target
    daily_conv = ctx.daily_conv
    model_time_period = str(calib_cfg.get("model_time_period", "daily"))
    weight_method = str(
        calib_cfg.get("weight_function")
        or (calib_cfg.get("odme") or {}).get("weight_function", "inverse_sqrt")
    )

    max_iterations = int(calib_cfg.get("max_iterations", 10))
    conv_cfg = calib_cfg.get("convergence") or {}
    geh_target = float(conv_cfg.get("geh_lt5_target_pct", 85.0))
    geh_mean_target = float(conv_cfg.get("geh_mean_target", 5.0))
    min_improvement = float(conv_cfg.get("min_improvement_pct", -5.0))

    daily_r2_target = float(daily_conv.get("r2_target", 0.80))
    daily_slope_range = daily_conv.get("slope_range", [0.85, 1.15])
    daily_slope_lo = float(daily_slope_range[0])
    daily_slope_hi = float(daily_slope_range[1])
    daily_pct_rmse_max = float(daily_conv.get("pct_rmse_max", 35.0))
    daily_sl_max_dev = float(daily_conv.get("screenline_max_pct_deviation", 15.0))
    daily_bias_max = float(daily_conv.get("bias_abs_max_pct", 15.0))

    scale_cfg = calib_cfg.get("scaling") or {}
    scale_method = str(scale_cfg.get("method", "select_link"))
    scale_enabled = bool(scale_cfg.get("enabled", True))
    damping = float(scale_cfg.get("damping", 0.40))
    _unused_scale_params = {
        k: scale_cfg[k]
        for k in ("min_factor", "max_factor", "adaptive_data_driven")
        if k in scale_cfg
    }
    if _unused_scale_params:
        logger.debug(
            "Scaling config params present but unused (elasticity uses max_deviation): %s",
            _unused_scale_params,
        )
    odme_cfg_for_class = calib_cfg.get("odme") or {}
    class_res_enabled = bool(odme_cfg_for_class.get("class_residual_enabled", True))
    class_res_damping = float(odme_cfg_for_class.get("class_residual_damping", 0.15))
    class_res_min_counts = int(odme_cfg_for_class.get("class_residual_min_counts", 3))

    quality_cfg = calib_cfg.get("quality_gates") or {}
    q_bias_hard = float(quality_cfg.get("hard_class_bias_max_abs_pct", 90.0))
    q_wmape_warn = float(quality_cfg.get("warn_wmape_pct", 47.0))
    q_geh_warn = float(quality_cfg.get("warn_geh_lt5_pct", 7.0))
    q_obj_patience = int(quality_cfg.get("objective_patience", 10))
    q_obj_weights = quality_cfg.get("objective_weights") or {}
    w_rho = float(q_obj_weights.get("spearman", 120.0))
    w_r2 = float(q_obj_weights.get("r2", 80.0))
    w_slope = float(q_obj_weights.get("slope_penalty", 50.0))
    w_geh = float(q_obj_weights.get("geh_lt5", 1.8))
    w_geh_mean = float(q_obj_weights.get("geh_mean", 3.0))
    w_wmape = float(q_obj_weights.get("wmape_pct", 1.0))
    w_bias = float(q_obj_weights.get("class_bias_max_abs_pct", 0.35))
    w_rmse = float(q_obj_weights.get("pct_rmse", 0.35))

    logger.info("=== FSM ITERATIVE CALIBRATION ===")
    logger.info(f"  model_time_period={model_time_period}, max_iterations={max_iterations}")
    if model_time_period == "daily":
        logger.info(
            f"  Daily convergence: R²>={daily_r2_target}, slope∈[{daily_slope_lo},{daily_slope_hi}], "
            f"%RMSE<={daily_pct_rmse_max}, SL_dev<={daily_sl_max_dev}%, |bias|<={daily_bias_max}%"
        )
        logger.info(
            f"  (GEH<5 >= {geh_target}% kept as diagnostic; daily_capacity_factor={daily_cap_factor})"
        )
    else:
        logger.info(f"  Hourly convergence: target GEH<5 >= {geh_target}%")
    logger.info(f"  scaling: enabled={scale_enabled}, method={scale_method}, damping={damping}")
    logger.info(f"  count_target: {count_target} (comparing against '{obs_col}')")

    sl_query = ctx.sl_query
    history = ctx.history
    prev_geh5 = 0.0
    mq: Dict[str, Any] = {}
    last_extended: Dict[str, Any] = {}
    obj_best = float("-inf")
    obj_non_improve = 0
    sl_results: Dict[str, Any] = {}

    try:
        for it in range(1, max_iterations + 1):
            logger.info(f"\n── Iteration {it}/{max_iterations} ──")

            total_demand = float(mat.matrix_view.sum())
            logger.info(f"  Demand total: {total_demand:,.0f}")

            save_skims_now = bool(calib_cfg.get("save_skims", False)) and it == 1
            vol_df, skims, sl_matrices, vol_col, total_vol = ctx.run_iteration_assignment(
                save_skims=save_skims_now,
            )
            if skims is not None:
                skim_path = output_dir / "skims.aem"
                try:
                    skims.export(str(skim_path))
                    logger.info(f"  Skims saved: {skim_path}")
                except Exception as exc:
                    warnings.warn(f"Skim export failed ({skim_path}): {exc}", RuntimeWarning, stacklevel=1)

            logger.info(f"  Assigned volume: {total_vol:,.0f}  (col={vol_col})")
            if total_vol > 0 and total_demand > 0:
                logger.info(f"  Route amplification: {total_vol / total_demand:.1f} links/trip")

            if it == 1:
                try:
                    bkdn = compute_class_volume_breakdown(vol_df, links_gdf)
                    if bkdn:
                        logger.info("  Volume breakdown by road class / traffic class:")
                        for col_name, by_rc in bkdn.items():
                            if col_name.startswith("_"):
                                continue
                            for rc in ("motorway", "trunk", "primary", "secondary", "tertiary", "other"):
                                info = by_rc.get(rc)
                                if info:
                                    logger.info(
                                        f"    {col_name:20s}  {rc:12s}  "
                                        f"vol={info['total_volume']:>12,.0f}  links={info['n_links']}"
                                    )
                        shares = bkdn.get("_class_shares", {})
                        for rc, detail in shares.items():
                            logger.info(f"    {rc} class shares: {detail['shares']}")
                except Exception as ex:
                    logger.warning(f"  WARNING: volume breakdown failed: {ex}")

            matched, valid, compare_col = ctx.match_and_filter(vol_df, vol_col, iteration=it)

            Z_current = float("inf")
            stats: Dict[str, Any]
            if not valid.empty and compare_col and compare_col in valid.columns:
                Z_current, stats = ctx.compute_objective_and_stats(valid, compare_col, weight_method)
                ctx.update_best_state(Z_current, it)
            else:
                stats = {"n": 0}

            geh5 = float(stats.get("geh_lt5_pct", 0))
            geh10 = float(stats.get("geh_lt10_pct", 0))
            geh_mean = float(stats.get("geh_mean") or 0.0)
            r2 = stats.get("r2")
            slope = stats.get("slope")
            pct_rmse = stats.get("pct_rmse")
            bias_pct = stats.get("bias_pct")
            daily_geh_adj = float(stats.get("daily_geh_lt_adj_pct") or 0)

            logger.info(f"  R²={r2}  slope={slope}  %RMSE={pct_rmse}  bias={bias_pct}%  GEH_mean={geh_mean:.2f}")
            logger.info(
                f"  GEH<5: {geh5:.1f}%  GEH<10: {geh10:.1f}%  "
                f"daily-adj GEH<{stats.get('daily_geh_threshold', 5):.0f}: {daily_geh_adj:.1f}%"
            )
            logger.info(f"  Z={Z_current:,.0f}  (best={ctx.best_Z:,.0f} at it={ctx.best_iteration})")

            if not valid.empty and "link_id" in valid.columns and "direction" in links_gdf.columns:
                _dir_lookup = links_gdf[["link_id", "direction"]].drop_duplicates("link_id")
                _dir_lookup = _dir_lookup.rename(columns={"direction": "_link_dir"})
                vdir = valid.merge(_dir_lookup, on="link_id", how="left")
                for dval, label in [(0, "bidir"), (1, "oneway")]:
                    mask = vdir["_link_dir"] == dval if dval == 0 else vdir["_link_dir"] != 0
                    sub = vdir.loc[mask]
                    if len(sub) >= 2:
                        s_stats = compute_stats(
                            sub[compare_col].values, sub[obs_col].values,
                            daily_capacity_factor=daily_cap_factor,
                        )
                        logger.info(
                            f"    {label} (n={len(sub)}): slope={s_stats.get('slope')}  "
                            f"bias={s_stats.get('bias_pct')}%  R²={s_stats.get('r2')}"
                        )

            if it == 1 and not valid.empty and "link_type" in valid.columns and compare_col:
                _rc = valid["link_type"].map(_coarse_road_class)
                for rc in ("motorway", "trunk", "primary", "secondary", "tertiary", "other"):
                    rc_rows = valid[_rc == rc]
                    if len(rc_rows) < 2:
                        continue
                    _m = rc_rows[compare_col].values.astype(float)
                    _o = rc_rows[obs_col].values.astype(float)
                    _ratio = float(_m.sum() / max(_o.sum(), 1))
                    logger.info(
                        f"    {rc:12s} (n={len(rc_rows):4d}): "
                        f"sum_mod={_m.sum():>12,.0f}  sum_obs={_o.sum():>12,.0f}  "
                        f"ratio={_ratio:.3f}"
                    )

            last_extended = {}
            if not valid.empty and compare_col and compare_col in valid.columns:
                try:
                    last_extended = compute_extended_link_metrics(
                        valid, compare_col, obs_col,
                    )
                except Exception as ex:
                    logger.warning(f"  WARNING: extended link metrics failed: {ex}")

            mq = match_quality_report(matched)
            if it == 1:
                logger.info(
                    f"  Match quality: {mq['n_matched']}/{mq['n_total']} matched, "
                    f"{mq['n_link_conflicts']} conflicts, "
                    f"mean_dist={mq['mean_match_distance_m']}m"
                )

                if compare_col in valid.columns and "link_type" in valid.columns:
                    try:
                        diag = valid[[compare_col, obs_col, "link_type"]].copy()
                        diag["_bias"] = diag[compare_col] - diag[obs_col]
                        diag["_road_class"] = diag["link_type"].map(_coarse_road_class)
                        for rc in ("motorway", "trunk"):
                            rc_rows = diag[diag["_road_class"] == rc].copy()
                            if rc_rows.empty:
                                continue
                            worst = rc_rows.reindex(rc_rows["_bias"].abs().nlargest(5).index)
                            logger.info(f"  Top-5 {rc} links by |bias|:")
                            for _, row in worst.iterrows():
                                logger.info(
                                    f"    mod={row[compare_col]:>8,.0f}  obs={row[obs_col]:>8,.0f}  "
                                    f"bias={row['_bias']:>+8,.0f}  type={row['link_type']}"
                                )
                    except Exception:
                        logger.debug(
                            "Top-5 bias by road class diagnostic failed",
                            exc_info=True,
                        )

            sl_results = {}
            max_sl_pct_dev: float = 0.0
            if screenlines and vol_col:
                sl_res = evaluate_all_screenlines(
                    screenlines, vol_df, matched, vol_col, obs_col, links_gdf,
                )
                for sn, sr in sl_res.items():
                    sl_results[sn] = sr.to_dict()
                    if sr.observed_total > 0 and np.isfinite(sr.ratio):
                        dev = abs(sr.ratio - 1.0) * 100.0
                        max_sl_pct_dev = max(max_sl_pct_dev, dev)
                        if it == 1 or it == max_iterations:
                            logger.info(
                                f"  Screenline '{sn}': mod={sr.modeled_total:,.0f} "
                                f"obs={sr.observed_total:,.0f} ratio={sr.ratio:.2f} GEH={sr.geh:.1f}"
                            )
                if sl_results:
                    logger.info(f"  Screenline max %deviation: {max_sl_pct_dev:.1f}%")

            iter_record: Dict[str, Any] = {
                "iteration": it,
                "demand_total": round(total_demand, 0),
                "assigned_total": round(total_vol, 0),
                **stats,
                "max_screenline_pct_dev": round(max_sl_pct_dev, 1),
            }
            if last_extended:
                iter_record["wmape_pct"] = last_extended.get("wmape_pct")
                iter_record["class_bias_max_abs_pct"] = last_extended.get("class_bias_max_abs_pct")
                iter_record["spearman_rho"] = last_extended.get("spearman_rho")

            if last_extended:
                try:
                    rho = float(last_extended.get("spearman_rho") or 0.0)
                    wmape = float(last_extended.get("wmape_pct") or 0.0)
                    bias_abs = float(last_extended.get("class_bias_max_abs_pct") or 0.0)
                    cur_r2 = float(stats.get("r2") or 0.0)
                    cur_slope = float(stats.get("slope") or 1.0)
                    slope_dev = abs(cur_slope - 1.0)
                    cur_pct_rmse = float(stats.get("pct_rmse") or 0.0)
                    obj_q = (
                        w_rho * rho
                        + w_r2 * cur_r2
                        - w_slope * slope_dev
                        + w_geh * geh5
                        - w_geh_mean * geh_mean
                        - w_wmape * wmape
                        - w_bias * bias_abs
                        - w_rmse * cur_pct_rmse
                    )
                    iter_record["quality_objective"] = round(obj_q, 4)
                    if obj_q > obj_best + 1e-9:
                        obj_best = obj_q
                        obj_non_improve = 0
                    else:
                        obj_non_improve += 1
                except Exception:
                    logger.debug("Quality objective computation failed", exc_info=True)

            if class_res_enabled and len(valid) > 0 and compare_col:
                cr = _compute_class_residuals(
                    valid, obs_col, compare_col,
                    min_counts=class_res_min_counts,
                )
                cr = _supplement_class_ratios_from_screenlines(
                    sl_results, links_gdf, cr,
                )
                if cr:
                    iter_record["class_ratios"] = {k: round(v, 3) for k, v in cr.items()}
                    logger.info(
                        f"  Per-class obs/mod: "
                        + ", ".join(f"{k}={v:.3f}" for k, v in cr.items())
                    )
            history.append(iter_record)

            converged = False
            if model_time_period == "daily":
                cur_r2 = float(stats.get("r2") or 0.0)
                cur_slope = float(stats.get("slope") or 0.0)
                cur_prmse = float(stats.get("pct_rmse") or 999.0)
                cur_bias = abs(float(stats.get("bias_pct") or 999.0))
                checks: Dict[str, bool] = {
                    f"R²>={daily_r2_target}": cur_r2 >= daily_r2_target,
                    f"slope∈[{daily_slope_lo},{daily_slope_hi}]": daily_slope_lo <= cur_slope <= daily_slope_hi,
                    f"%RMSE<={daily_pct_rmse_max}": cur_prmse <= daily_pct_rmse_max,
                    f"|bias|<={daily_bias_max}%": cur_bias <= daily_bias_max,
                }
                if sl_results:
                    checks[f"SL_dev<={daily_sl_max_dev}%"] = max_sl_pct_dev <= daily_sl_max_dev

                passed = [k for k, v in checks.items() if v]
                failed = [k for k, v in checks.items() if not v]
                logger.info(f"  Daily convergence: {len(passed)}/{len(checks)} criteria met")
                if failed:
                    logger.info(f"    PASS: {', '.join(passed) if passed else 'none'}")
                    logger.info(f"    FAIL: {', '.join(failed)}")

                converged = ctx.check_convergence(stats, max_sl_pct_dev, sl_results)
                if converged:
                    logger.info("  CONVERGED (daily): all criteria met")
            else:
                if geh5 >= geh_target and geh_mean <= geh_mean_target:
                    logger.info(
                        f"  CONVERGED: GEH<5 = {geh5:.1f}% >= {geh_target}% "
                        f"and GEH_mean = {geh_mean:.2f} <= {geh_mean_target}"
                    )
                    converged = True
                elif geh5 >= geh_target:
                    logger.info(
                        f"  GEH<5 = {geh5:.1f}% >= {geh_target}% OK, "
                        f"but GEH_mean = {geh_mean:.2f} > {geh_mean_target} — continuing"
                    )

            if converged:
                break

            if it - ctx.best_iteration >= q_obj_patience + 1:
                logger.info(f"  Z-STALL: no Z improvement since iteration {ctx.best_iteration}")
                break
            if model_time_period == "daily":
                if q_obj_patience > 0 and obj_non_improve >= q_obj_patience:
                    logger.info(
                        f"  QUALITY STOP: objective non-improving for "
                        f"{obj_non_improve} iterations"
                    )
                    break
            else:
                improvement = geh5 - prev_geh5
                if it > 1 and improvement < min_improvement:
                    logger.info(
                        f"  STALLED: GEH improvement {improvement:.2f}% "
                        f"< {min_improvement}%"
                    )
                    break
                if q_obj_patience > 0 and obj_non_improve >= q_obj_patience:
                    logger.info(
                        f"  QUALITY STOP: objective non-improving for "
                        f"{obj_non_improve} iterations"
                    )
                    break

            prev_geh5 = geh5

            if scale_enabled and not valid.empty and compare_col and total_vol > 0:
                data = mat.matrix[core_name]
                demand = data[:, :].copy().astype(np.float64)

                if sl_matrices and sl_results:
                    n_sl_active = sum(
                        1 for sn in sl_matrices
                        if _sr_val(sl_results.get(sn, {}), "observed_total") > 0
                        and _sr_val(sl_results.get(sn, {}), "modeled_total") > 0
                    )
                    sl_eff_damping = damping / max(np.sqrt(n_sl_active), 1.0)
                    corrections_pairs = _spiess_update_step(
                        demand,
                        sl_matrices,
                        sl_results,
                        sl_damping=sl_eff_damping,
                        sl_ratio_max=ctx.sl_ratio_max,
                        sl_ratio_min=ctx.sl_ratio_min,
                        sl_clip_min=ctx.sl_clip_min,
                        sl_clip_max=ctx.sl_clip_max,
                        seed_lower=ctx.seed_lower,
                        seed_upper=ctx.seed_upper,
                        gd_inner=1,
                    )
                    if corrections_pairs:
                        logger.info(
                            "  Spiess SL corrections "
                            f"({len(corrections_pairs)}): "
                            + ", ".join(f"{n}={r:.2f}" for n, r in corrections_pairs)
                        )

                ctx.apply_global_residual(demand, valid, compare_col, damping=damping)

                if class_res_enabled and compare_col:
                    ctx.apply_class_residual(
                        demand,
                        vol_df,
                        valid,
                        compare_col,
                        sl_matrices,
                        sl_results,
                        damping=class_res_damping,
                    )

                ctx.apply_gateway_and_rebase(demand, vol_df, vol_col, it)

                np.clip(demand, ctx.seed_lower, ctx.seed_upper, out=demand)
                np.maximum(demand, 0.0, out=demand)
                data[:, :] = demand
                logger.info(f"  Demand after update: {demand.sum():,.0f}")

            elif not scale_enabled:
                logger.info("  Scaling disabled (principle: frozen / diagnostic run)")
            else:
                logger.warning("  Cannot scale — no valid matched volumes")

            ctx._transition(CalibrationState.PERSIST_CHECKPOINT)
            mat.save()

    finally:
        ctx.restore_best_and_close()

    best_vol_df = ctx.finalize_best_state()
    if best_vol_df is not None:
        vol_df = best_vol_df

    period_stats: Dict[str, Any] = {}
    period_cfg = calib_cfg.get("period_calibration") or {}
    if period_cfg.get("enabled", False) and history:
        try:
            from sim.demand.temporal import load_profile, get_demand_period_shares

            profile = load_profile(ctx.cfg)
            period_shares = get_demand_period_shares(profile)
            cal_periods = period_cfg.get("periods", ["am", "pm", "daily"])

            logger.info("\n=== PER-PERIOD VALIDATION ===")

            mat_p = AequilibraeMatrix()
            mat_p.load(str(matrix_path))
            project_p = Project()
            project_p.open(str(project_dir))

            try:
                for period in cal_periods:
                    p_core = f"wd_{period}" if period != "daily" else "wd_daily"
                    if p_core not in mat_p.names:
                        logger.warning(f"  Skipping {period}: core '{p_core}' not in matrix")
                        continue

                    p_share = period_shares.get(period, 1.0)
                    logger.info(f"\n  Period: {period} (share={p_share:.3f}, core={p_core})")

                    mat_p.computational_view([p_core])
                    p_demand = float(mat_p.matrix_view.sum())
                    logger.info(f"    Demand: {p_demand:,.0f}")

                    vol_df_p, _, _sl_p, _conv_p = execute_assignment(
                        project_p,
                        mat_p,
                        algorithm=algorithm,
                        max_iter=max_iter_assign,
                        rgap_target=rgap,
                        save_skims=False,
                        select_links=sl_query,
                        fixed_cost_field=gc_field,
                        fixed_cost_multiplier=gc_mult,
                        vot=gc_vot,
                        bpr_parameters=cfg_bpr,
                        multi_class=cfg_multi,
                        graph=cached_graph,
                        cores=cores,
                    )

                    vc_p = _detect_volume_col(vol_df_p)
                    if not vc_p:
                        continue

                    lv = links_gdf.copy()
                    lv = lv.merge(vol_df_p[["link_id", vc_p]], on="link_id", how="left")
                    m_p = match_counts_to_links(
                        ctx.pent, lv, buffer_m=buffer_m, vol_col=vc_p,
                        match_quality_min=mq_min,
                    )

                    obs_period_col = f"_obs_{period}"
                    if period == "daily":
                        m_p[obs_period_col] = m_p[obs_col]
                    else:
                        m_p[obs_period_col] = m_p[obs_col] * p_share

                    vp = m_p.dropna(subset=[vc_p, obs_period_col])
                    vp = vp[vp[obs_period_col] > 0].copy()
                    if "_excluded" in vp.columns:
                        vp = vp[~vp["_excluded"]].copy()
                    if len(vp) > 0:
                        p_dcf = daily_cap_factor if period == "daily" else 1.0
                        ps = compute_stats(
                            vp[vc_p].values, vp[obs_period_col].values,
                            daily_capacity_factor=p_dcf,
                        )
                        period_stats[period] = ps
                        logger.info(
                            f"    R²={ps.get('r2')}  slope={ps.get('slope')}  "
                            f"%RMSE={ps.get('pct_rmse')}  n={ps.get('n')}"
                        )
                    else:
                        logger.warning(f"    No valid matched volumes for {period}")
            finally:
                mat_p.close()
                project_p.close()
        except Exception:
            logger.exception("Period calibration validation failed")

    best_iteration = ctx.best_iteration
    best_final = (
        history[best_iteration - 1]
        if history and 0 < best_iteration <= len(history)
        else (history[-1] if history else {})
    )

    report = {
        "iterations": len(history),
        "model_time_period": model_time_period,
        "converged": _check_final_convergence(history, model_time_period, geh_target, daily_conv),
        "history": history,
        "final": best_final,
        "config": {
            "max_iterations": max_iterations,
            "model_time_period": model_time_period,
            "daily_capacity_factor": daily_cap_factor,
            "geh_target": geh_target,
            "daily_convergence": daily_conv if model_time_period == "daily" else None,
            "scale_method": scale_method,
            "scale_enabled": scale_enabled,
            "damping": damping,
            "count_target": count_target,
            "obs_col": obs_col,
            "aggregate_corridor": agg_corridor,
            "matching": calib_cfg.get("matching") or {},
        },
        "extended_metrics": last_extended,
        "quality_gates": {
            "hard_class_bias_max_abs_pct": q_bias_hard,
            "warn_wmape_pct": q_wmape_warn,
            "warn_geh_lt5_pct": q_geh_warn,
            "objective_patience": q_obj_patience,
            "objective_weights": {
                "spearman": w_rho,
                "r2": w_r2,
                "slope_penalty": w_slope,
                "geh_lt5": w_geh,
                "geh_mean": w_geh_mean,
                "wmape_pct": w_wmape,
                "class_bias_max_abs_pct": w_bias,
                "pct_rmse": w_rmse,
            },
            "hard_reject": bool((last_extended or {}).get("class_bias_max_abs_pct", 0.0) > q_bias_hard)
            if last_extended
            else False,
            "warnings": [
                w for w in [
                    (
                        f"wmape_pct>{q_wmape_warn}"
                        if last_extended and (last_extended.get("wmape_pct") or 0.0) > q_wmape_warn
                        else None
                    ),
                    (
                        f"geh_lt5_pct<{q_geh_warn}"
                        if history and (history[-1].get("geh_lt5_pct", 0.0) < q_geh_warn)
                        else None
                    ),
                ] if w is not None
            ],
        },
        "observed_summary": {
            "total_motor": round(float(ctx.pent["observed_motor_total"].sum()), 0),
            "total_truck": round(float(ctx.pent["observed_truck"].sum()), 0),
            "total_all": round(float(ctx.pent["observed_total"].sum()), 0),
            "n_count_stations": len(ctx.pent),
        },
        "match_quality": mq,
        "period_stats": period_stats,
        "screenlines": sl_results,
    }
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nCalibration report: {report_path}")

    if history:
        out_path = output_dir / "assignment_results.parquet"
        vol_df.to_parquet(str(out_path), index=False)
        logger.info(f"Final assignment: {out_path}")
