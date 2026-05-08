#!/usr/bin/env python3
"""Experiment 03: Spatial congestion patterns — model V/C vs real jam frequency.

Compares the baseline model's Volume/Capacity ratio per link with the
number of Waze-reported traffic jams on that segment.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    init_experiment,
    load_assignment_results,
    load_jams_stats,
    load_network_links,
    load_segment_map,
    save_csv,
    save_figure,
    save_json,
)

NAME = "exp03_congestion_patterns"


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    links = load_network_links(cfg)
    assign = load_assignment_results(cfg)
    jams_stats = load_jams_stats(cfg)
    seg_map = load_segment_map(cfg, links)

    # Merge assignment volumes onto links
    vol_cols = [c for c in assign.columns if c != "link_id" and c not in links.columns]
    net = links[["link_id", "link_type"]].merge(
        assign[["link_id"] + vol_cols], on="link_id", how="left",
    )

    # V/C ratio
    if "VOC_max" in net.columns:
        net["vc"] = net["VOC_max"].fillna(0)
    elif "capacity_ab" in links.columns:
        cap = links.set_index("link_id")[["capacity_ab", "capacity_ba"]].max(axis=1)
        vol = net.set_index("link_id")["wd_daily_tot"].fillna(0) if "wd_daily_tot" in net.columns else 0
        net["vc"] = (vol / cap.clip(lower=1)).reindex(net.index).fillna(0).values
    else:
        net["vc"] = 0

    # Map jams to links
    jam_col = None
    for c in ("jam_count", "count", "n_jams"):
        if c in jams_stats.columns:
            jam_col = c
            break
    if jam_col is None:
        raise ValueError(f"No jam count column in jams_segment_stats. Cols: {list(jams_stats.columns)}")

    jams_stats["link_id"] = jams_stats["segment_id"].map(seg_map)
    jams_agg = jams_stats.dropna(subset=["link_id"]).groupby("link_id").agg(
        jam_count=(jam_col, "sum"),
        mean_delay=("mean_delay_seconds", "mean") if "mean_delay_seconds" in jams_stats.columns else (jam_col, "count"),
    ).reset_index()
    jams_agg["link_id"] = jams_agg["link_id"].astype(int)

    df = net.merge(jams_agg, on="link_id", how="inner")
    df = df[(df["vc"] > 0) & (df["jam_count"] > 0)].copy()

    if df.empty:
        print(f"[{NAME}] No overlapping data — check segment mapping.")
        return

    # ------------------------------------------------------------------
    # Spearman correlation
    # ------------------------------------------------------------------
    rho, p_value = sp_stats.spearmanr(df["vc"], df["jam_count"])

    summary = {
        "n_links": len(df),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_value),
    }

    # ------------------------------------------------------------------
    # Scatter: V/C vs jam_count
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["vc"], df["jam_count"], alpha=0.3, s=8, edgecolors="none")
    ax.set_xlabel("Modelový V/C poměr")
    ax.set_ylabel("Počet zácp (Waze)")
    ax.set_title(f"V/C vs. frekvence zácp  (ρ = {rho:.3f},  n = {len(df)})")
    save_figure(fig, out_dir / "correlation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Box plot: jam_count by V/C quartile
    # ------------------------------------------------------------------
    df["vc_quartile"] = pd.qcut(df["vc"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                 duplicates="drop")
    if df["vc_quartile"].nunique() >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column="jam_count", by="vc_quartile", ax=ax, showfliers=False)
        ax.set_xlabel("V/C kvartil")
        ax.set_ylabel("Počet zácp")
        ax.set_title("Distribuce zácp podle V/C kvartilů")
        fig.suptitle("")
        save_figure(fig, out_dir / "boxplot_quartiles.png")
        plt.close(fig)

        quartile_stats = df.groupby("vc_quartile", observed=True)["jam_count"].describe()
        save_csv(quartile_stats.reset_index(), out_dir / "quartile_stats.csv")

    # ------------------------------------------------------------------
    # Per-class correlation
    # ------------------------------------------------------------------
    class_rows = []
    for lt, grp in df.groupby("link_type"):
        if len(grp) < 10:
            continue
        r, p = sp_stats.spearmanr(grp["vc"], grp["jam_count"])
        class_rows.append({
            "link_type": lt, "n": len(grp),
            "spearman_rho": round(float(r), 3), "p_value": float(p),
        })
    if class_rows:
        summary["per_class"] = class_rows
        save_csv(pd.DataFrame(class_rows), out_dir / "class_correlation.csv")

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
