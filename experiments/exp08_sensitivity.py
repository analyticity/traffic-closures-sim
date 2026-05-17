#!/usr/bin/env python3
"""Experiment 08: Sensitivity analysis.

Three sub-experiments testing model robustness:
 a) Demand scaling — OD matrix × {0.8, 0.9, 1.0, 1.1, 1.2}, report ΔVHT and
    overloaded link count for a test closure scenario.
 b) Severity variation — for one closure, sweep residual capacity from 0%
    (full closure) through 25%, 50%, 75% to 100% (no closure).
 c) Convergence — vary RGAP target {1e-3, 1e-4, 1e-5} to verify result
    stability vs compute cost trade-off.
"""
from __future__ import annotations

import copy
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    _cache_dir,
    compute_scenario_kpis,
    compute_vht,
    init_experiment,
    load_assignment_results,
    load_baseline_links,
    load_closures,
    match_links_near_point,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp08_sensitivity"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scaled_assignment(
    cfg: Dict[str, Any],
    scenario_links: List[Dict[str, Any]],
    demand_factor: float = 1.0,
    *,
    rgap: float | None = None,
) -> pd.DataFrame:
    """Like run_scenario_assignment but with OD demand scaling + custom rgap."""
    from aequilibrae import Project
    from aequilibrae.matrix import AequilibraeMatrix
    from sim.assignment import build_graph, execute_assignment, fix_node_ids
    from sim.scenarios.engine import apply_scenario_to_graph
    from sim._metrics import aggregate_daily_volumes

    calib_cfg = cfg.get("calibration") or {}
    assign_cfg = cfg.get("assignment") or {}
    demand_cfg = cfg.get("demand") or {}

    _algorithm = str(calib_cfg.get("algorithm", "bfw"))
    _max_iter = int(assign_cfg.get("scenario_max_iter", calib_cfg.get("max_iter", 100)))
    _rgap = rgap or float(assign_cfg.get("scenario_rgap", calib_cfg.get("rgap_target", 0.001)))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))
    bpr_params = dict(assign_cfg.get("bpr") or {}) or None

    mc_cfg = assign_cfg.get("multi_class") or {}
    multi_classes = (
        list(mc_cfg["classes"])
        if mc_cfg.get("enabled") and "classes" in mc_cfg
        else None
    )
    gc_cfg = assign_cfg.get("generalized_cost") or {}
    gc_enabled = bool(gc_cfg.get("enabled", False))
    gc_field = str(gc_cfg["fixed_cost_field"]) if gc_enabled and "fixed_cost_field" in gc_cfg else None
    gc_mult = float(gc_cfg.get("fixed_cost_multiplier", 0.0)) if gc_enabled else 0.0
    gc_vot = float(gc_cfg.get("vot", 1.0))

    project_path = Path(cfg["project_path"])
    fix_node_ids(project_path)

    orig_matrix_path = Path(demand_cfg.get("matrix_path"))
    tmp_matrix = None
    try:
        if demand_factor != 1.0:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".aem")
            import os; os.close(tmp_fd)
            tmp_matrix = Path(tmp_path)
            shutil.copy2(orig_matrix_path, tmp_matrix)
            mat = AequilibraeMatrix()
            mat.load(str(tmp_matrix))
        else:
            mat = AequilibraeMatrix()
            mat.load(str(orig_matrix_path))
        mat.computational_view([core_name])

        if demand_factor != 1.0:
            mat.matrix_view[:] = mat.matrix_view[:] * demand_factor
            logger.info("Demand scaled by %.2f (total trips: %.0f)", demand_factor, float(mat.matrix_view.sum()))

        project = Project()
        project.open(str(project_path))
        try:
            graph = build_graph(project, mat, bpr_parameters=bpr_params, assignment_cfg=assign_cfg)
            if scenario_links:
                apply_scenario_to_graph(graph, scenario_links)

            df, _skims, _sl, _conv = execute_assignment(
                project, mat,
                algorithm=_algorithm,
                max_iter=_max_iter,
                rgap_target=_rgap,
                bpr_parameters=bpr_params,
                multi_class=multi_classes,
                fixed_cost_field=gc_field,
                fixed_cost_multiplier=gc_mult,
                vot=gc_vot,
                graph=graph,
            )
        finally:
            project.close()
            mat.close()
    finally:
        if tmp_matrix and tmp_matrix.exists():
            tmp_matrix.unlink()

    aggregate_daily_volumes(df)
    return df


def _pick_test_closure(
    closures: pd.DataFrame,
    links: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Pick one representative closure and convert to scenario_links."""
    if "quality_score" in closures.columns:
        closures = closures.assign(
            _qs=pd.to_numeric(closures["quality_score"], errors="coerce").fillna(0)
        ).sort_values("_qs", ascending=False).drop(columns=["_qs"])

    sev_col = "pg_severity" if "pg_severity" in closures.columns else "severity"
    full = closures[closures[sev_col].str.contains("full|closure", case=False, na=False)]
    chosen = full.iloc[0] if not full.empty else closures.iloc[0]

    lat, lon = chosen.get("lat"), chosen.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        return []

    nearby = match_links_near_point(float(lat), float(lon), links, max_dist_m=200)

    result = []
    for _, link in nearby.iterrows():
        lanes = max(int(link.get("lanes", 2)), 1)
        result.append({
            "link_id": int(link["link_id"]),
            "direction": "both",
            "closure_type": "full",
            "lanes": lanes,
            "lanes_remaining": 0,
        })
    return result


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    links = load_baseline_links(cfg)
    baseline = load_assignment_results(cfg)
    closures = load_closures(cfg)

    from sim._metrics import aggregate_daily_volumes
    aggregate_daily_volumes(baseline)

    baseline_merged = links.copy()
    for c in baseline.columns:
        if c != "link_id" and c not in baseline_merged.columns:
            baseline_merged = baseline_merged.merge(baseline[["link_id", c]], on="link_id", how="left")
    aggregate_daily_volumes(baseline_merged)

    test_scenario = _pick_test_closure(closures, links)
    if not test_scenario:
        print(f"[{NAME}] Could not build test scenario.")
        return

    all_results: Dict[str, Any] = {}

    # ==================================================================
    # A) Demand scaling
    # ==================================================================
    logger.info("=== A) Demand scaling ===")
    demand_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    demand_rows = []

    for factor in demand_factors:
        label = f"demand_{factor:.1f}"
        logger.info("Running demand factor %.1f ...", factor)
        try:
            df = _run_scaled_assignment(cfg, test_scenario, demand_factor=factor)
            kpis = compute_scenario_kpis(baseline_merged, df)
            kpis["demand_factor"] = factor
            demand_rows.append(kpis)
        except Exception as e:
            logger.error("Demand %.1f failed: %s", factor, e)

    if demand_rows:
        demand_df = pd.DataFrame(demand_rows)
        save_csv(demand_df, out_dir / "demand_scaling.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(demand_df["demand_factor"], demand_df["delta_vht"], "o-", markersize=8, color="steelblue")
        ax1.set_xlabel("Faktor poptávky")
        ax1.set_ylabel("ΔVHT (voz·hod)")
        ax1.set_title("Citlivost ΔVHT na škálování poptávky")
        ax1.grid(True, alpha=0.3)

        ax2.plot(demand_df["demand_factor"], demand_df["scenario_overloaded"], "s-",
                 markersize=8, color="coral")
        ax2.set_xlabel("Faktor poptávky")
        ax2.set_ylabel("Počet přetížených hran")
        ax2.set_title("Přetížené hrany vs. poptávka")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Citlivost na objem poptávky", fontsize=13)
        fig.tight_layout()
        save_figure(fig, out_dir / "demand_sensitivity.png")
        plt.close(fig)

        all_results["demand_scaling"] = demand_rows

    # ==================================================================
    # B) Severity variation
    # ==================================================================
    logger.info("=== B) Severity variation ===")
    capacity_factors = [0.0, 0.25, 0.5, 0.75, 1.0]
    severity_rows = []

    for cap_frac in capacity_factors:
        label = f"capacity_{cap_frac}"
        logger.info("Running lane-open fraction %.2f ...", cap_frac)

        if cap_frac >= 1.0:
            # No modifications — baseline-equivalent (ΔVHT ≈ 0)
            variant = []
        else:
            variant = []
            for sl in test_scenario:
                sl_copy = dict(sl)
                lanes = max(int(sl_copy.get("lanes", 1)), 1)
                if cap_frac <= 0.0:
                    sl_copy["closure_type"] = "full"
                    sl_copy["lanes_remaining"] = 0
                else:
                    sl_copy["closure_type"] = "lanes"
                    sl_copy["lanes_remaining"] = max(
                        0, min(lanes, int(round(lanes * float(cap_frac)))),
                    )
                variant.append(sl_copy)

        try:
            df = _run_scaled_assignment(cfg, variant, demand_factor=1.0)
            kpis = compute_scenario_kpis(baseline_merged, df)
            kpis["capacity_factor"] = cap_frac
            severity_rows.append(kpis)
        except Exception as e:
            logger.error("Capacity %.1f failed: %s", cap_frac, e)

    if severity_rows:
        sev_df = pd.DataFrame(severity_rows)
        save_csv(sev_df, out_dir / "severity_variation.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(sev_df["capacity_factor"], sev_df["delta_vht"], "s-", color="coral", markersize=8)
        ax1.set_xlabel("Zbytková kapacita (podíl)")
        ax1.set_ylabel("ΔVHT (voz·hod)")
        ax1.set_title("ΔVHT vs. závažnost uzavírky")
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()
        for _, row in sev_df.iterrows():
            ax1.annotate(f"{row['delta_vht']:.0f}", (row["capacity_factor"], row["delta_vht"]),
                         textcoords="offset points", xytext=(0, 10), fontsize=8, ha="center")

        ax2.plot(sev_df["capacity_factor"], sev_df["delta_overloaded"], "D-",
                 color="darkred", markersize=8)
        ax2.set_xlabel("Zbytková kapacita (podíl)")
        ax2.set_ylabel("Δ přetížených hran")
        ax2.set_title("Přetížené hrany vs. závažnost")
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()

        fig.suptitle("Citlivost na závažnost uzavírky", fontsize=13)
        fig.tight_layout()
        save_figure(fig, out_dir / "severity_sensitivity.png")
        plt.close(fig)

        all_results["severity_variation"] = severity_rows

    # ==================================================================
    # C) Convergence sensitivity
    # ==================================================================
    logger.info("=== C) Convergence sensitivity ===")
    rgap_values = [1e-3, 1e-4, 1e-5]
    conv_rows = []

    for rgap_val in rgap_values:
        label = f"rgap_{rgap_val}"
        logger.info("Running RGAP = %s ...", rgap_val)
        t0 = time.time()
        try:
            df = _run_scaled_assignment(cfg, test_scenario, demand_factor=1.0, rgap=rgap_val)
            elapsed = time.time() - t0
            kpis = compute_scenario_kpis(baseline_merged, df)
            kpis["rgap"] = rgap_val
            kpis["runtime_s"] = round(elapsed, 1)
            conv_rows.append(kpis)
        except Exception as e:
            logger.error("RGAP %s failed: %s", rgap_val, e)

    if conv_rows:
        conv_df = pd.DataFrame(conv_rows)
        save_csv(conv_df, out_dir / "convergence.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.semilogx(conv_df["rgap"], conv_df["delta_vht"], "D-", color="mediumseagreen", markersize=8)
        ax1.set_xlabel("RGAP cíl")
        ax1.set_ylabel("ΔVHT (voz·hod)")
        ax1.set_title("Stabilita ΔVHT")
        ax1.grid(True, alpha=0.3)
        ax1.invert_xaxis()

        ax2.semilogx(conv_df["rgap"], conv_df["runtime_s"], "D-", color="steelblue", markersize=8)
        ax2.set_xlabel("RGAP cíl")
        ax2.set_ylabel("Doba výpočtu (s)")
        ax2.set_title("Čas konvergence")
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()

        fig.suptitle("Citlivost na konvergenční kritérium", fontsize=13)
        fig.tight_layout()
        save_figure(fig, out_dir / "convergence_sensitivity.png")
        plt.close(fig)

        all_results["convergence"] = conv_rows

    save_json(all_results, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
