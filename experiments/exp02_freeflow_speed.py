#!/usr/bin/env python3
"""Experiment 02: Validate model free-flow speeds against Waze data.

Compares model ``speed_ab`` / ``speed_ba`` with the median
``speed_normal_kmh`` reported by Waze in the ``traffic_jams`` table
(aggregated per road segment in ``jams_segment_stats.parquet``).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    init_experiment,
    load_jams_stats,
    load_network_links,
    load_segment_map,
    save_csv,
    save_figure,
    save_json,
)

NAME = "exp02_freeflow_speed"


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    links = load_network_links(cfg)
    jams_stats = load_jams_stats(cfg)
    seg_map = load_segment_map(cfg, links)

    # Map jams stats segment_id -> link_id
    speed_col = None
    for candidate in ("median_speed_normal_kmh", "speed_normal_kmh", "mean_speed_normal_kmh", "speed_normal_median"):
        if candidate in jams_stats.columns:
            speed_col = candidate
            break
    if speed_col is None:
        raise ValueError(f"No speed column found in jams_segment_stats. Columns: {list(jams_stats.columns)}")

    jams_stats["link_id"] = jams_stats["segment_id"].map(seg_map)
    matched = jams_stats.dropna(subset=["link_id", speed_col]).copy()
    matched["link_id"] = matched["link_id"].astype(int)

    # Average model speed per link (take max of ab/ba as representative free-flow).
    # Prefer pre-closure speeds when available so the comparison reflects the
    # model's free-flow calibration, not closure penalties.
    model_speeds = links[["link_id", "link_type"]].copy()
    spd_ab_col = "_preclosure_speed_ab" if "_preclosure_speed_ab" in links.columns else "speed_ab"
    spd_ba_col = "_preclosure_speed_ba" if "_preclosure_speed_ba" in links.columns else "speed_ba"
    if spd_ab_col in links.columns and spd_ba_col in links.columns:
        ab = pd.to_numeric(links[spd_ab_col], errors="coerce")
        ba = pd.to_numeric(links[spd_ba_col], errors="coerce")
        model_speeds["model_speed"] = pd.concat([ab, ba], axis=1).max(axis=1)
    elif "speed" in links.columns:
        model_speeds["model_speed"] = links["speed"]
    else:
        raise ValueError("No speed columns found in network links")

    df = matched.merge(model_speeds, on="link_id", how="inner")
    df = df.rename(columns={speed_col: "waze_speed"})
    df = df[df["waze_speed"] > 0].copy()

    if df.empty:
        print(f"[{NAME}] No matched records — check segment map.")
        return

    # Outlier filter: Waze speed_normal_kmh can contain physically
    # impossible values (>1000 km/h); cap at 200 km/h.
    _MAX_PLAUSIBLE_SPEED = 200.0
    n_before = len(df)
    df = df[df["waze_speed"] <= _MAX_PLAUSIBLE_SPEED].copy()
    n_outliers = n_before - len(df)
    if n_outliers > 0:
        print(f"[{NAME}] Filtered {n_outliers} Waze outliers (>{_MAX_PLAUSIBLE_SPEED} km/h)")

    if df.empty:
        print(f"[{NAME}] No records left after outlier filtering.")
        return

    # ------------------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------------------
    waze = df["waze_speed"].values
    model = df["model_speed"].values

    ss_res = float(np.sum((model - waze) ** 2))
    ss_tot = float(np.sum((waze - np.mean(waze)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((model - waze) ** 2)))
    mae = float(np.mean(np.abs(model - waze)))
    bias = float(np.mean(model - waze))

    summary = {
        "n_links": len(df),
        "r2": round(r2, 4),
        "rmse_kmh": round(rmse, 2),
        "mae_kmh": round(mae, 2),
        "bias_kmh": round(bias, 2),
    }

    # ------------------------------------------------------------------
    # Per road-class breakdown
    # ------------------------------------------------------------------
    class_rows = []
    for lt, grp in df.groupby("link_type"):
        w, m = grp["waze_speed"].values, grp["model_speed"].values
        if len(w) < 3:
            continue
        ss_r = float(np.sum((m - w) ** 2))
        ss_t = float(np.sum((w - np.mean(w)) ** 2))
        class_rows.append({
            "link_type": lt,
            "n": len(grp),
            "waze_mean": round(float(np.mean(w)), 1),
            "model_mean": round(float(np.mean(m)), 1),
            "bias": round(float(np.mean(m - w)), 1),
            "rmse": round(float(np.sqrt(np.mean((m - w) ** 2))), 1),
            "r2": round(1 - ss_r / ss_t if ss_t > 0 else float("nan"), 3),
        })
    class_df = pd.DataFrame(class_rows).sort_values("n", ascending=False)
    summary["per_class"] = class_rows
    save_csv(class_df, out_dir / "class_comparison.csv")

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(waze, model, alpha=0.3, s=8, edgecolors="none")
    lim = max(waze.max(), model.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
    slope, intercept, *_ = sp_stats.linregress(waze, model)
    xs = np.array([0, lim])
    ax.plot(xs, slope * xs + intercept, "r-", lw=0.8,
            label=f"regrese: y = {slope:.2f}x + {intercept:.1f}")
    ax.set_xlabel("Waze free-flow rychlost (km/h)")
    ax.set_ylabel("Modelová free-flow rychlost (km/h)")
    ax.set_title(f"Validace rychlostí  (n={len(df)},  R²={r2:.3f},  RMSE={rmse:.1f} km/h)")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    save_figure(fig, out_dir / "scatter.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Per-class bar chart
    # ------------------------------------------------------------------
    if not class_df.empty:
        top = class_df.head(10)
        x = np.arange(len(top))
        w_bar = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w_bar / 2, top["waze_mean"], w_bar, label="Waze")
        ax.bar(x + w_bar / 2, top["model_mean"], w_bar, label="Model")
        ax.set_xticks(x)
        ax.set_xticklabels(top["link_type"], rotation=45, ha="right")
        ax.set_ylabel("Průměrná rychlost (km/h)")
        ax.set_title("Porovnání rychlostí podle třídy komunikace")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out_dir / "scatter_by_class.png")
        plt.close(fig)

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
