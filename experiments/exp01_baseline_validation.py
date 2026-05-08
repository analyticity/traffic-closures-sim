#!/usr/bin/env python3
"""Experiment 01: Baseline model validation against CSD counts.

Reads the existing validation report and assignment results, then produces:
- Scatter plot: observed vs modelled volumes (holdout set)
- GEH histogram
- Screenline bar chart
- Summary JSON with R², %RMSE, GEH pass rate, MAPE, bias
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    init_experiment,
    load_assignment_results,
    load_network_links,
    save_figure,
    save_json,
)

NAME = "exp01_baseline_validation"


def main() -> None:
    cfg, out_dir = init_experiment(NAME)
    demand_dir = Path(cfg["demand"]["output_dir"])

    # ------------------------------------------------------------------
    # 1. Load existing validation report
    # ------------------------------------------------------------------
    vr_path = demand_dir / "validation_report.json"
    if vr_path.exists():
        vr = json.loads(vr_path.read_text(encoding="utf-8"))
    else:
        vr = {}

    # ------------------------------------------------------------------
    # 2. Load assignment + CSD match data
    # ------------------------------------------------------------------
    assign_df = load_assignment_results(cfg)
    links = load_network_links(cfg)

    merged = links[["link_id", "link_type"]].merge(assign_df, on="link_id", how="inner")

    # Try loading CSD matched diagnostics
    diag_path = demand_dir / "matching_diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
    else:
        diag = pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Compute statistics from matched CSD data
    # ------------------------------------------------------------------
    from sim.calibration.metrics import compute_stats
    from sim._metrics import compute_geh

    summary = dict(vr)  # start from existing report

    if not diag.empty and "observed" in diag.columns and "modeled" in diag.columns:
        obs = diag["observed"].values
        mod = diag["modeled"].values
        valid = np.isfinite(obs) & np.isfinite(mod) & (obs > 0)
        obs_v, mod_v = obs[valid], mod[valid]

        stats = compute_stats(mod_v, obs_v)
        summary["holdout_stats"] = stats

        # --- Scatter plot ---
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(obs_v, mod_v, alpha=0.5, s=15, edgecolors="none")
        lim = max(obs_v.max(), mod_v.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
        if stats.get("slope") and np.isfinite(stats["slope"]):
            xs = np.array([0, lim])
            ax.plot(xs, stats["slope"] * xs + stats["intercept"], "r-", lw=0.8,
                    label=f"regrese: y = {stats['slope']:.2f}x + {stats['intercept']:.0f}")
        ax.set_xlabel("Pozorované AADT (CSD)")
        ax.set_ylabel("Modelované AADT")
        ax.set_title(f"Baseline validace  (n={stats['n']},  R²={stats.get('r2', 0):.3f})")
        ax.legend()
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        save_figure(fig, out_dir / "scatter.png")
        plt.close(fig)

        # --- GEH histogram ---
        geh = compute_geh(mod_v, obs_v)
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.arange(0, max(geh.max(), 20) + 1, 1)
        ax.hist(geh, bins=bins, edgecolor="white", alpha=0.8)
        pct_lt5 = float(np.mean(geh < 5) * 100)
        ax.axvline(5, color="red", ls="--", label=f"GEH=5  ({pct_lt5:.1f}% profilů < 5)")
        ax.set_xlabel("GEH")
        ax.set_ylabel("Počet profilů")
        ax.set_title("Distribuce GEH statistiky")
        ax.legend()
        save_figure(fig, out_dir / "geh_histogram.png")
        plt.close(fig)

        # --- Residual histogram ---
        residuals = mod_v - obs_v
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(residuals, bins=40, edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", ls="--")
        ax.set_xlabel("Reziduum (model − pozorované)")
        ax.set_ylabel("Počet profilů")
        ax.set_title(f"Histogram reziduí  (bias = {np.mean(residuals):.0f} voz/den)")
        save_figure(fig, out_dir / "residuals.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Screenline results (if present in validation report)
    # ------------------------------------------------------------------
    sl_results = vr.get("screenlines") or vr.get("screenline_results")
    if isinstance(sl_results, list) and sl_results:
        names = [s.get("name", f"SL{i}") for i, s in enumerate(sl_results)]
        obs_vals = [s.get("observed", 0) for s in sl_results]
        mod_vals = [s.get("modeled", 0) for s in sl_results]

        x = np.arange(len(names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w / 2, obs_vals, w, label="Pozorované")
        ax.bar(x + w / 2, mod_vals, w, label="Model")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Objem (voz/den)")
        ax.set_title("Screenline analýza")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out_dir / "screenlines.png")
        plt.close(fig)

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
