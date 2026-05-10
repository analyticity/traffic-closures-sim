"""ODME calibration: Spiess gradient and entropy-maximization methods.

Both share the same bi-level loop (lower: equilibrium assignment,
upper: OD adjustment). Only the update step differs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sim.calibration.context import _CalibrationContext, _sr_val
from sim.calibration.metrics import (
    _compute_class_residuals,
    _supplement_class_ratios_from_screenlines,
)
from sim.calibration.validation import _check_final_convergence

logger = logging.getLogger(__name__)


def _spiess_update_step(
    demand: np.ndarray,
    sl_matrices: Dict[str, Any],
    sl_results: Dict[str, Any],
    *,
    sl_damping: float,
    sl_ratio_max: float,
    sl_ratio_min: float,
    sl_clip_min: float,
    sl_clip_max: float,
    seed_lower: np.ndarray,
    seed_upper: np.ndarray,
    gd_inner: int,
) -> List[Tuple[str, float]]:
    """Spiess relative-gradient multiplicative update across all screenlines."""
    corrections_applied: List[Tuple[str, float]] = []
    for gd_it in range(1, gd_inner + 1):
        corrections_applied = []
        for sl_name, sl_od_raw in sl_matrices.items():
            obs_sl = _sr_val(sl_results.get(sl_name, {}), "observed_total")
            mod_sl = _sr_val(sl_results.get(sl_name, {}), "modeled_total")
            if obs_sl <= 0 or mod_sl <= 0:
                continue
            ratio = obs_sl / mod_sl
            if ratio > sl_ratio_max or ratio < sl_ratio_min:
                continue
            sl_od = np.asarray(sl_od_raw, dtype=np.float64).reshape(demand.shape)
            proportion = np.where(
                demand > 0,
                np.clip(sl_od / np.maximum(demand, 1e-9), 0.0, 1.0),
                0.0,
            )
            adjustment = 1.0 + sl_damping * (ratio - 1.0) * proportion
            np.clip(adjustment, sl_clip_min, sl_clip_max, out=adjustment)
            demand *= adjustment
            corrections_applied.append((sl_name, round(ratio, 3)))
        np.clip(demand, seed_lower, seed_upper, out=demand)
        np.maximum(demand, 0.0, out=demand)
        logger.info(
            f"    GD inner {gd_it}: {len(corrections_applied)} SLs applied, "
            f"demand_total={demand.sum():,.0f}"
        )
    return corrections_applied


def _entropy_update_step(
    demand: np.ndarray,
    sl_matrices: Dict[str, Any],
    sl_results: Dict[str, Any],
    *,
    step_size: float = 0.3,
    ratio_max: float = 5.0,
    ratio_min: float = 0.2,
    clip_min: float = 0.5,
    clip_max: float = 2.0,
) -> int:
    """Single entropy-maximization multiplicative update across all screenlines."""
    n_applied = 0
    for sl_name, sl_od_raw in sl_matrices.items():
        obs_sl = _sr_val(sl_results.get(sl_name, {}), "observed_total")
        mod_sl = _sr_val(sl_results.get(sl_name, {}), "modeled_total")
        if obs_sl <= 0 or mod_sl <= 0:
            continue
        ratio = obs_sl / mod_sl
        if ratio > ratio_max or ratio < ratio_min:
            continue
        sl_od = np.asarray(sl_od_raw, dtype=np.float64).reshape(demand.shape)
        proportion = np.where(
            demand > 0,
            np.clip(sl_od / np.maximum(demand, 1e-9), 0.0, 1.0),
            0.0,
        )
        log_ratio = np.log(max(ratio, 1e-12))
        adjustment = np.exp(step_size * proportion * log_ratio)
        np.clip(adjustment, clip_min, clip_max, out=adjustment)
        demand *= adjustment
        n_applied += 1
    return n_applied


def run_odme_calibration(
    config_path: str | Path = "config/brno/sim.yaml",
    *,
    method: str = "spiess",
) -> None:
    """Unified ODME calibration. method: 'spiess' | 'entropy'."""
    if method not in ("spiess", "entropy"):
        raise ValueError(f"method must be 'spiess' or 'entropy', got {method!r}")

    ctx = _CalibrationContext(config_path)
    calib_cfg = ctx.calib_cfg
    odme_cfg = ctx.odme_cfg

    # --- Stage-aware parameter resolution ---
    stages_cfg = odme_cfg.get("stages")
    if stages_cfg and isinstance(stages_cfg, list) and len(stages_cfg) > 0:
        stages: List[Dict[str, Any]] = stages_cfg
    else:
        stages = [{}]

    max_outer = int(odme_cfg.get("max_outer_iterations", 25))
    gd_inner = int(odme_cfg.get("gradient_descent_iterations", 5))
    weight_method = str(odme_cfg.get("weight_function", "inverse_sqrt"))
    conv_tol = float(odme_cfg.get("convergence_tol", 0.001))
    global_residual_damping = float(odme_cfg.get("global_residual_damping", 0.25))
    class_res_enabled = bool(odme_cfg.get("class_residual_enabled", True))
    class_res_damping = float(odme_cfg.get("class_residual_damping", 0.15))
    max_iter_change_pct = float(odme_cfg.get("max_iter_change_pct", 15.0))
    entropy_step_size = float(odme_cfg.get("entropy_step_size", 0.3))
    stall_patience = int(odme_cfg.get("stall_patience", 8))

    method_name = "entropy_odme" if method == "entropy" else "odme_spiess_gradient"
    banner = (
        "ENTROPY-MAXIMIZATION ODME"
        if method == "entropy"
        else "ODME GRADIENT CALIBRATION (Spiess method)"
    )
    logger.info(f"=== {banner} ===")
    logger.info(
        f"  method={method}, max_outer={max_outer}, gd_inner={gd_inner}, "
        f"max_deviation={ctx.max_deviation}"
    )
    if len(stages) > 1:
        logger.info(f"  Multi-stage ODME: {len(stages)} stages configured")
    if method == "entropy":
        logger.info(
            f"  entropy_step_size={entropy_step_size}, conv_tol={conv_tol}, "
            f"max_iter_change={max_iter_change_pct}%"
        )
        logger.info(
            f"  class_residual: enabled={class_res_enabled}, damping={class_res_damping}"
        )
    else:
        logger.info(f"  weight_function={weight_method}, convergence_tol={conv_tol}")
        logger.info(
            f"  global_residual_damping={global_residual_damping}, "
            f"max_iter_change={max_iter_change_pct}%"
        )
        logger.info(
            f"  class_residual: enabled={class_res_enabled}, damping={class_res_damping}"
        )

    prev_Z = float("inf")
    effective_global_damping = global_residual_damping
    consecutive_improvements = 0
    consecutive_deteriorations = 0
    sl_results: Dict[str, Any] = {}
    global_iter = 0
    current_stage = 0
    stage_iters_done = 0

    def _stage_param(key: str, default: Any) -> Any:
        """Resolve a parameter from the current stage override or global odme_cfg."""
        if current_stage < len(stages):
            stage = stages[current_stage]
            if key in stage:
                return stage[key]
        return odme_cfg.get(key, default)

    try:
        for outer_it in range(1, max_outer + 1):
            stage_max = int(_stage_param("max_iterations", max_outer))
            if stage_iters_done >= stage_max and current_stage < len(stages) - 1:
                current_stage += 1
                stage_iters_done = 0
                prev_Z = float("inf")
                consecutive_improvements = 0
                consecutive_deteriorations = 0

                new_max_dev = float(_stage_param("max_deviation", ctx.max_deviation))
                if new_max_dev != ctx.max_deviation:
                    seed = ctx.mat.matrix[ctx.core_name][:, :].copy().astype(np.float64)
                    ctx.max_deviation = new_max_dev
                    ctx.seed_lower = seed / new_max_dev
                    ctx.seed_upper = seed * new_max_dev
                    ctx.seed_lower[seed <= 0] = 0.0
                    if ctx.epsilon_support > 0:
                        ctx.seed_upper[seed <= 0] = ctx.epsilon_support
                    else:
                        ctx.seed_upper[seed <= 0] = 0.0

                ctx.gw_rebase_seed_bounds = bool(
                    _stage_param("rebase_seed_bounds", ctx.gw_rebase_seed_bounds)
                )
                class_res_enabled = bool(_stage_param("class_residual_enabled", class_res_enabled))
                global_residual_damping = float(
                    _stage_param("global_residual_damping", global_residual_damping)
                )
                effective_global_damping = global_residual_damping
                stall_patience = int(_stage_param("stall_patience", stall_patience))

                logger.info(
                    f"\n{'#' * 60}\n"
                    f"  STAGE {current_stage + 1}/{len(stages)} — "
                    f"max_deviation={ctx.max_deviation}, "
                    f"class_residual={class_res_enabled}, "
                    f"rebase={ctx.gw_rebase_seed_bounds}\n"
                    f"{'#' * 60}"
                )

            stage_iters_done += 1
            global_iter += 1

            logger.info(f"\n{'=' * 60}")
            logger.info(f"  {banner} Iteration {outer_it}/{max_outer} (stage {current_stage + 1})")
            logger.info(f"{'=' * 60}")

            total_demand = float(ctx.mat.matrix_view.sum())
            logger.info(f"  Demand total: {total_demand:,.0f}")

            save_skims_now = bool(calib_cfg.get("save_skims", False)) and outer_it == 1
            vol_df, skims, sl_matrices, vol_col, total_vol = ctx.run_iteration_assignment(
                save_skims=save_skims_now
            )
            if skims is not None and outer_it == 1:
                try:
                    skims.export(str(ctx.output_dir / "skims.aem"))
                except Exception:
                    logger.debug("Skim export failed", exc_info=True)
            logger.info(f"  Assigned volume: {total_vol:,.0f}  (col={vol_col})")

            matched, valid, compare_col = ctx.match_and_filter(
                vol_df, vol_col, iteration=outer_it
            )
            if valid.empty:
                logger.warning("  WARNING: no valid matched counts — cannot compute gradient")
                continue

            Z_current, stats = ctx.compute_objective_and_stats(
                valid, compare_col, weight_method
            )
            ctx.update_best_state(Z_current, outer_it)

            if outer_it > 1 and Z_current > prev_Z:
                effective_global_damping = max(effective_global_damping * 0.7, 0.05)
                consecutive_improvements = 0
                consecutive_deteriorations += 1
                if consecutive_deteriorations >= 3 and ctx.best_demand is not None:
                    ctx.mat.matrix[ctx.core_name][:, :] = ctx.best_demand
                    ctx.mat.save()
                    logger.info(
                        "  REVERT: Z deteriorated "
                        f"{consecutive_deteriorations}x → restoring best demand "
                        f"from iteration {ctx.best_iteration}"
                    )
                    consecutive_deteriorations = 0
                else:
                    logger.warning(
                        "  WARNING: Z increased — reducing global_damping to "
                        f"{effective_global_damping:.3f}"
                    )
            elif outer_it > 1 and Z_current < prev_Z:
                consecutive_improvements += 1
                consecutive_deteriorations = 0
                if (
                    consecutive_improvements >= 2
                    and effective_global_damping < global_residual_damping
                ):
                    effective_global_damping = global_residual_damping
                    consecutive_improvements = 0
                    logger.info(
                        "  Damping reset to "
                        f"{effective_global_damping:.3f} after 2 consecutive improvements"
                    )

            geh5 = float(stats.get("geh_lt5_pct", 0))
            r2 = stats.get("r2")
            slope = stats.get("slope")
            pct_rmse = stats.get("pct_rmse")
            bias_pct = stats.get("bias_pct")
            logger.info(
                f"  Z={Z_current:,.1f}  (prev={prev_Z:,.1f}  "
                f"delta={Z_current - prev_Z:+,.1f})"
            )
            logger.info(
                f"  R²={r2}  slope={slope}  %RMSE={pct_rmse}  bias={bias_pct}%"
            )
            logger.info(f"  GEH<5: {geh5:.1f}%  n_counts={len(valid)}")

            sl_results, max_sl_pct_dev = ctx.evaluate_and_log_screenlines(
                vol_df, matched, vol_col
            )

            n_active_screenlines = sum(
                1
                for sn in (sl_results or {})
                if _sr_val(sl_results.get(sn, {}), "observed_total") > 0
                and _sr_val(sl_results.get(sn, {}), "modeled_total") > 0
            )

            if sl_matrices and sl_results:
                dead_sls = [
                    sn for sn in list(sl_matrices)
                    if _sr_val(sl_results.get(sn, {}), "observed_total") > 0
                    and _sr_val(sl_results.get(sn, {}), "modeled_total") <= 0
                ]
                for sn in dead_sls:
                    del sl_matrices[sn]
                    logger.warning(
                        "  Deactivating screenline '%s': observed>0 but modeled=0 "
                        "(dead-end or disconnected link)",
                        sn,
                    )

            iter_record = {
                "iteration": outer_it,
                "demand_total": round(total_demand, 0),
                "assigned_total": round(total_vol, 0),
                "Z_objective": round(Z_current, 1),
                **stats,
                "max_screenline_pct_dev": round(max_sl_pct_dev, 1),
                "n_count_posts": len(valid),
                "n_active_screenlines": n_active_screenlines,
            }
            if class_res_enabled and len(valid) > 0 and compare_col:
                cr = _compute_class_residuals(
                    valid,
                    ctx.obs_col,
                    compare_col,
                    min_counts=int(
                        odme_cfg.get("class_residual_min_counts", 3)
                    ),
                )
                cr = _supplement_class_ratios_from_screenlines(
                    sl_results, ctx.links_gdf, cr
                )
                if cr:
                    iter_record["class_ratios"] = {
                        k: round(v, 3) for k, v in cr.items()
                    }
                    logger.info(
                        "  Per-class obs/mod: "
                        + ", ".join(f"{k}={v:.3f}" for k, v in cr.items())
                    )
            ctx.history.append(iter_record)

            if outer_it > 1:
                rel_change = abs(Z_current - prev_Z) / max(prev_Z, 1.0)
                if rel_change < conv_tol:
                    logger.info(
                        f"  CONVERGED: |delta Z|/Z = {rel_change:.6f} < {conv_tol}"
                    )
                    break
            if outer_it - ctx.best_iteration >= stall_patience:
                logger.info(
                    "  STALLED: no Z improvement since iteration "
                    f"{ctx.best_iteration}"
                )
                break
            if ctx.check_convergence(stats, max_sl_pct_dev, sl_results):
                logger.info("  CONVERGED: all daily criteria met")
                break

            prev_Z = Z_current

            data = ctx.mat.matrix[ctx.core_name]
            demand = data[:, :].copy().astype(np.float64)

            if sl_matrices:
                if method == "entropy":
                    for gd_it in range(1, gd_inner + 1):
                        n_applied = _entropy_update_step(
                            demand,
                            sl_matrices,
                            sl_results,
                            step_size=entropy_step_size,
                            ratio_max=ctx.sl_ratio_max,
                            ratio_min=ctx.sl_ratio_min,
                            clip_min=ctx.sl_clip_min,
                            clip_max=ctx.sl_clip_max,
                        )
                        np.clip(demand, ctx.seed_lower, ctx.seed_upper, out=demand)
                        np.maximum(demand, 0.0, out=demand)
                        logger.info(
                            f"    Entropy inner {gd_it}: {n_applied} SLs, "
                            f"demand_total={demand.sum():,.0f}"
                        )
                else:
                    n_sl_active = sum(
                        1
                        for sn in sl_matrices
                        if _sr_val(sl_results.get(sn, {}), "observed_total") > 0
                        and _sr_val(sl_results.get(sn, {}), "modeled_total") > 0
                    )
                    sl_damping = 1.0 / max(np.sqrt(n_sl_active), 1.0)
                    corrections = _spiess_update_step(
                        demand,
                        sl_matrices,
                        sl_results,
                        sl_damping=sl_damping,
                        sl_ratio_max=ctx.sl_ratio_max,
                        sl_ratio_min=ctx.sl_ratio_min,
                        sl_clip_min=ctx.sl_clip_min,
                        sl_clip_max=ctx.sl_clip_max,
                        seed_lower=ctx.seed_lower,
                        seed_upper=ctx.seed_upper,
                        gd_inner=gd_inner,
                    )
                    if corrections:
                        logger.info(
                            "  SL ratios: "
                            + ", ".join(f"{n}={r}" for n, r in corrections)
                        )
                    np.clip(demand, ctx.seed_lower, ctx.seed_upper, out=demand)
                    np.maximum(demand, 0.0, out=demand)

            if len(valid) > 0 and total_vol > 0:
                ctx.apply_global_residual(
                    demand,
                    valid,
                    compare_col,
                    damping=effective_global_damping,
                )

            if class_res_enabled and len(valid) > 0 and compare_col:
                ctx.apply_class_residual(
                    demand,
                    vol_df,
                    valid,
                    compare_col,
                    sl_matrices,
                    sl_results,
                    damping=class_res_damping,
                )

            ctx.apply_gateway_and_rebase(demand, vol_df, vol_col, outer_it)

            ctx.apply_demand_cap_and_save(demand, total_demand, max_change_pct=max_iter_change_pct)

    finally:
        ctx.restore_best_and_close()

    best_vol_df = ctx.finalize_best_state()
    if best_vol_df is not None:
        vol_df = best_vol_df

    history = ctx.history
    model_time_period = str(calib_cfg.get("model_time_period", "daily"))
    best_final = (
        history[ctx.best_iteration - 1]
        if history and 0 < ctx.best_iteration <= len(history)
        else (history[-1] if history else {})
    )
    sl_for_conv = (
        sl_results if sl_results else ({"_": {}} if ctx.screenlines else {})
    )
    converged_at_best = False
    if best_final:
        max_sl_best = float(best_final.get("max_screenline_pct_dev") or 0.0)
        converged_at_best = ctx.check_convergence(
            best_final, max_sl_best, sl_for_conv
        )

    # P0-3: Seed-deviation analysis
    seed_deviation_report: Dict[str, Any] = {}
    if ctx.best_demand is not None:
        try:
            backup_path = ctx.matrix_path.with_suffix(".aem.orig")
            if backup_path.exists():
                from aequilibrae.matrix import AequilibraeMatrix as _AEM
                seed_mat = _AEM()
                seed_mat.load(str(backup_path))
                seed_data = seed_mat.matrix[ctx.core_name][:, :].copy().astype(np.float64)
                seed_mat.close()

                final_data = ctx.best_demand.astype(np.float64)
                nonzero = seed_data > 1.0
                if nonzero.any():
                    ratios = np.where(nonzero, final_data / seed_data, 1.0)
                    max_ratio = float(np.max(ratios[nonzero]))
                    min_ratio = float(np.min(ratios[nonzero]))
                    n_suspicious = int(np.sum(
                        (ratios > ctx.max_deviation) | (ratios < 1.0 / ctx.max_deviation)
                    ))

                    total_seed = float(seed_data.sum())
                    total_final = float(final_data.sum())
                    prior_drift_pct = (
                        (total_final - total_seed) / max(total_seed, 1.0) * 100
                    )

                    diffs = np.abs(final_data - seed_data)
                    top_k = 10
                    flat_idx = np.argsort(diffs.ravel())[-top_k:][::-1]
                    zone_ids = ctx.mat.index[:]
                    top_k_cells = []
                    for idx in flat_idx:
                        i, j = divmod(int(idx), diffs.shape[1])
                        d = float(diffs.ravel()[idx])
                        if d < 1.0:
                            break
                        top_k_cells.append({
                            "origin": int(zone_ids[i]),
                            "destination": int(zone_ids[j]),
                            "seed": round(float(seed_data[i, j]), 1),
                            "final": round(float(final_data[i, j]), 1),
                            "abs_diff": round(d, 1),
                        })

                    seed_deviation_report = {
                        "max_cell_ratio": round(max_ratio, 3),
                        "min_cell_ratio": round(min_ratio, 3),
                        "max_deviation_threshold": ctx.max_deviation,
                        "n_cells_exceeding_threshold": n_suspicious,
                        "total_seed": round(total_seed, 0),
                        "total_final": round(total_final, 0),
                        "prior_drift_pct": round(prior_drift_pct, 2),
                        "top_changed_cells": top_k_cells,
                    }
                    if n_suspicious > 0:
                        logger.warning(
                            "  ODME seed deviation: %d OD cells exceed %.1fx threshold "
                            "(max_ratio=%.2f, min_ratio=%.2f, prior_drift=%.1f%%)",
                            n_suspicious, ctx.max_deviation, max_ratio, min_ratio,
                            prior_drift_pct,
                        )
        except Exception:
            logger.debug("Seed deviation analysis failed", exc_info=True)

    # Write calibration section hashes for validation holdout guard
    calib_section_hash = ""
    if hasattr(ctx, "pent") and not ctx.pent.empty and "objectid" in ctx.pent.columns:
        ids_str = ",".join(str(x) for x in sorted(ctx.pent["objectid"].dropna().astype(int)))
        calib_section_hash = hashlib.sha256(ids_str.encode()).hexdigest()[:16]

    report = {
        "method": method_name,
        "iterations": len(history),
        "converged": len(history) > 0
        and (
            (
                len(history) >= 2
                and abs(
                    history[-1].get("Z_objective", 0)
                    - history[-2].get("Z_objective", 1)
                )
                / max(abs(history[-2].get("Z_objective", 1)), 1)
                < conv_tol
            )
            or _check_final_convergence(
                history, model_time_period, 85.0, ctx.daily_conv
            )
            or converged_at_best
        ),
        "history": history,
        "final": best_final,
        "config": {
            "method": method,
            "max_outer_iterations": max_outer,
            "gradient_descent_iterations": gd_inner,
            "max_deviation": ctx.max_deviation,
            "weight_function": weight_method,
            "convergence_tol": conv_tol,
            "global_residual_damping": global_residual_damping,
            "algorithm": ctx.algorithm,
            "count_target": ctx.count_target,
            "obs_col": ctx.obs_col,
            "aggregate_corridor": ctx.agg_corridor,
        },
        "screenlines": sl_results,
        "seed_deviation": seed_deviation_report,
        "calibration_section_hash": calib_section_hash,
    }
    if method == "entropy":
        report["config"]["entropy_step_size"] = entropy_step_size
    report_path = ctx.output_dir / "calibration_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"\n{banner} report: {report_path}")

    if history:
        z_start = history[0].get("Z_objective", 0)
        z_end = (
            ctx.best_Z
            if ctx.best_Z < float("inf")
            else history[-1].get("Z_objective", 0)
        )
        logger.info(
            f"  Z: {z_start:,.1f} → {z_end:,.1f}  "
            f"(reduction: {(1 - z_end / max(z_start, 1)) * 100:.1f}%)  "
            f"best at iteration {ctx.best_iteration}"
        )
        out_path = ctx.output_dir / "assignment_results.parquet"
        vol_df.to_parquet(str(out_path), index=False)
        logger.info(f"  Final assignment: {out_path}")


def run_entropy_odme(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """Backward-compatible entry point for entropy ODME."""
    run_odme_calibration(config_path, method="entropy")
