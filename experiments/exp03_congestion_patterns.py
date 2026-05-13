#!/usr/bin/env python3
"""Experiment 03: Congestion patterns — directional checks vs Waze (daily model).

No strong global correlation expectation. Instead:
- V/C quartiles vs jam counts (medians, Q4/Q1 ratio, Mann–Whitney)
- Top-V/C vs top-jam overlap (directional sanity)
- Per link_type Spearman (where n is sufficient)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    compute_quartile_comparison,
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

VC_TOP_FRAC = 0.10
JAM_TOP_FRAC = 0.30
MIN_PER_CLASS = 30


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    links = load_network_links(cfg)
    assign = load_assignment_results(cfg)
    jams_stats = load_jams_stats(cfg)
    seg_map = load_segment_map(cfg, links)

    vol_cols = [c for c in assign.columns if c != "link_id" and c not in links.columns]
    net = links[["link_id", "link_type"]].merge(
        assign[["link_id"] + vol_cols], on="link_id", how="left",
    )

    if "VOC_max" in net.columns:
        net["vc"] = net["VOC_max"].fillna(0).astype(float)
    elif "capacity_ab" in links.columns:
        cap = links.set_index("link_id")[["capacity_ab", "capacity_ba"]].max(axis=1)
        vol = net.set_index("link_id")["wd_daily_tot"].fillna(0) if "wd_daily_tot" in net.columns else 0
        net["vc"] = (vol / cap.clip(lower=1)).reindex(net.index).fillna(0).values
    else:
        net["vc"] = 0.0

    jam_col = None
    for c in ("jam_count", "count", "n_jams"):
        if c in jams_stats.columns:
            jam_col = c
            break
    if jam_col is None:
        raise ValueError(f"No jam count column in jams_segment_stats. Cols: {list(jams_stats.columns)}")

    jams_stats = jams_stats.copy()
    jams_stats["link_id"] = jams_stats["segment_id"].map(seg_map)
    jams_agg = jams_stats.dropna(subset=["link_id"]).groupby("link_id").agg(
        jam_count=(jam_col, "sum"),
    ).reset_index()
    jams_agg["link_id"] = jams_agg["link_id"].astype(int)

    df = net.merge(jams_agg, on="link_id", how="inner")
    df = df[(df["vc"] >= 0) & (df["jam_count"] >= 0)].copy()

    if df.empty:
        print(f"[{NAME}] No overlapping data — check segment mapping.")
        return

    summary: dict = {
        "n_links": len(df),
        "vc_top_frac": VC_TOP_FRAC,
        "jam_top_frac": JAM_TOP_FRAC,
    }

    rho, p_value = sp_stats.spearmanr(df["vc"], df["jam_count"])
    summary["spearman_global"] = {
        "rho": round(float(rho), 4),
        "p": float(p_value),
    }

    # --- Quartile comparison + Mann–Whitney (via helper) ---
    qcomp = compute_quartile_comparison(df, "vc", "jam_count")
    summary["quartiles"] = qcomp

    # --- Top V/C vs top jam overlap ---
    n = len(df)
    k_vc = max(int(np.ceil(n * VC_TOP_FRAC)), 10)
    k_jam = max(int(np.ceil(n * JAM_TOP_FRAC)), 10)
    hi_vc = set(df.nlargest(k_vc, "vc")["link_id"].astype(int))
    hi_jam = set(df.nlargest(k_jam, "jam_count")["link_id"].astype(int))
    inter = hi_vc & hi_jam
    summary["top_overlap"] = {
        "n_top_vc": len(hi_vc),
        "n_top_jam": len(hi_jam),
        "n_intersection": len(inter),
        "pct_of_top_vc_also_top_jam": round(100.0 * len(inter) / max(len(hi_vc), 1), 2),
    }

    # --- Boxplot by quartile ---
    try:
        df["vc_quartile"] = pd.qcut(df["vc"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        df["vc_quartile"] = None

    if df["vc_quartile"] is not None and df["vc_quartile"].notna().any():
        quartile_stats = df.groupby("vc_quartile", observed=True)["jam_count"].describe()
        save_csv(quartile_stats.reset_index(), out_dir / "quartile_stats.csv")

        fig, ax = plt.subplots(figsize=(8, 5))
        df.boxplot(column="jam_count", by="vc_quartile", ax=ax, showfliers=False)
        ax.set_xlabel("Kvartil modelového V/C")
        ax.set_ylabel("Počet hlášených zácp (Waze)")
        ax.set_title("Zácpy podle kvartilu V/C")
        fig.suptitle("")
        fig.tight_layout()
        save_figure(fig, out_dir / "quartile_boxplot.png")
        plt.close(fig)

    # --- Overlap bar ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Průnik"],
        [summary["top_overlap"]["pct_of_top_vc_also_top_jam"]],
        color="steelblue",
    )
    ax.set_ylabel("% top V/C také v top jam")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Top {int(VC_TOP_FRAC*100)}% V/C vs top {int(JAM_TOP_FRAC*100)}% jam "
        f"(n={summary['n_links']})"
    )
    fig.tight_layout()
    save_figure(fig, out_dir / "topn_overlap.png")
    plt.close(fig)

    # --- Global scatter (illustrative) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["vc"], df["jam_count"], alpha=0.25, s=10, edgecolors="none")
    ax.set_xlabel("Model V/C")
    ax.set_ylabel("Počet zácp (Waze)")
    ax.set_title(f"V/C vs zácpy (globální ρ = {rho:.3f})")
    save_figure(fig, out_dir / "correlation.png")
    plt.close(fig)

    # --- Per-class Spearman ---
    class_rows = []
    for lt, grp in df.groupby("link_type"):
        if len(grp) < MIN_PER_CLASS:
            continue
        r, p = sp_stats.spearmanr(grp["vc"], grp["jam_count"])
        class_rows.append({
            "link_type": str(lt),
            "n": len(grp),
            "spearman_rho": round(float(r), 4),
            "p_value": float(p),
        })
    if class_rows:
        summary["per_class_spearman"] = class_rows
        save_csv(pd.DataFrame(class_rows), out_dir / "class_correlation.csv")

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
