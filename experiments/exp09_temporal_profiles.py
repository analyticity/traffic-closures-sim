#!/usr/bin/env python3
"""Experiment 09: Temporal congestion profiles — model V/C vs hourly jam patterns.

The static AequilibraE model produces daily-average V/C.  This experiment
checks whether that daily V/C correlates more strongly with *peak-hour*
jam frequency (7–9, 15–17) than with off-peak periods, and visualises
the hourly jam profile per road class.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    _cache_dir,
    init_experiment,
    load_assignment_results,
    load_network_links,
    load_segment_map,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp09_temporal_profiles"

TIME_BANDS = {
    "ráno (6–9)":   (6, 9),
    "dopoledne (9–12)": (9, 12),
    "odpoledne (12–15)": (12, 15),
    "odpolední špička (15–18)": (15, 18),
    "večer (18–22)": (18, 22),
}

PEAK_HOURS = {7, 8, 15, 16}
WORKDAY_RANGE = range(0, 5)  # Mon–Fri


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    cache = _cache_dir(cfg)
    jams_path = cache / "jams.parquet"
    if not jams_path.exists():
        print(f"[{NAME}] jams.parquet not found — run fetch-data first.")
        return

    jams = pd.read_parquet(jams_path)
    links = load_network_links(cfg)
    assign = load_assignment_results(cfg)
    seg_map = load_segment_map(cfg, links)

    from sim._metrics import aggregate_daily_volumes
    aggregate_daily_volumes(assign)

    # Ensure hour/weekday columns
    if "hour" not in jams.columns and "first_seen" in jams.columns:
        jams["first_seen"] = pd.to_datetime(jams["first_seen"], errors="coerce", utc=True)
        jams["hour"] = jams["first_seen"].dt.hour
    if "weekday" not in jams.columns and "first_seen" in jams.columns:
        jams["weekday"] = jams["first_seen"].dt.weekday

    # Filter to workdays only
    jams_wd = jams[jams["weekday"].isin(WORKDAY_RANGE)].copy()
    logger.info("Workday jams: %d / %d total", len(jams_wd), len(jams))

    # Map to link_id
    jams_wd["link_id"] = jams_wd["segment_id"].map(seg_map)
    jams_wd = jams_wd.dropna(subset=["link_id"])
    jams_wd["link_id"] = jams_wd["link_id"].astype(int)

    # Model V/C per link
    net = links[["link_id", "link_type"]].merge(assign, on="link_id", how="left")
    aggregate_daily_volumes(net)
    if "VOC_max" in net.columns:
        net["vc"] = net["VOC_max"].fillna(0)
    elif "capacity_ab" in links.columns:
        cap = links.set_index("link_id")[["capacity_ab", "capacity_ba"]].max(axis=1)
        vol = net.set_index("link_id").get("wd_daily_tot", pd.Series(dtype=float)).fillna(0)
        net["vc"] = (vol / cap.clip(lower=1)).values
    else:
        net["vc"] = 0

    vc_map = net.set_index("link_id")["vc"].to_dict()
    jams_wd["vc"] = jams_wd["link_id"].map(vc_map)
    jams_wd = jams_wd.dropna(subset=["vc"])

    # ------------------------------------------------------------------
    # 1. Hourly jam profile (overall)
    # ------------------------------------------------------------------
    hourly = jams_wd.groupby("hour").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["coral" if h in PEAK_HOURS else "steelblue" for h in range(24)]
    ax.bar(range(24), hourly.values, color=colors, edgecolor="white")
    ax.set_xlabel("Hodina dne")
    ax.set_ylabel("Počet zácp (Waze)")
    ax.set_title("Hodinový profil zácp (pracovní dny)")
    ax.set_xticks(range(24))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="coral", label="Špička"), Patch(color="steelblue", label="Mimo špičku")])
    save_figure(fig, out_dir / "hourly_profile.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Per time-band jam count per segment → correlate with V/C
    # ------------------------------------------------------------------
    band_correlations = []
    for band_name, (h_start, h_end) in TIME_BANDS.items():
        band_jams = jams_wd[(jams_wd["hour"] >= h_start) & (jams_wd["hour"] < h_end)]
        band_counts = band_jams.groupby("link_id").size().reset_index(name="jam_count")
        band_counts["vc"] = band_counts["link_id"].map(vc_map)
        band_counts = band_counts.dropna(subset=["vc"])
        band_counts = band_counts[(band_counts["vc"] > 0) & (band_counts["jam_count"] > 0)]

        if len(band_counts) >= 10:
            rho, p = sp_stats.spearmanr(band_counts["vc"], band_counts["jam_count"])
        else:
            rho, p = float("nan"), float("nan")

        band_correlations.append({
            "time_band": band_name,
            "h_start": h_start,
            "h_end": h_end,
            "n_jams": len(band_jams),
            "n_links_matched": len(band_counts),
            "spearman_rho": round(float(rho), 4),
            "p_value": float(p),
        })

    band_df = pd.DataFrame(band_correlations)
    save_csv(band_df, out_dir / "time_band_correlations.csv")

    # ------------------------------------------------------------------
    # 3. Peak vs off-peak correlation comparison
    # ------------------------------------------------------------------
    peak_jams = jams_wd[jams_wd["hour"].isin(PEAK_HOURS)]
    offpeak_jams = jams_wd[~jams_wd["hour"].isin(PEAK_HOURS)]

    peak_counts = peak_jams.groupby("link_id").size().reset_index(name="jam_count_peak")
    offpeak_counts = offpeak_jams.groupby("link_id").size().reset_index(name="jam_count_offpeak")

    comparison = net[["link_id", "vc"]].merge(peak_counts, on="link_id", how="left")
    comparison = comparison.merge(offpeak_counts, on="link_id", how="left")
    comparison["jam_count_peak"] = comparison["jam_count_peak"].fillna(0)
    comparison["jam_count_offpeak"] = comparison["jam_count_offpeak"].fillna(0)

    has_data = comparison[(comparison["vc"] > 0) & (
        (comparison["jam_count_peak"] > 0) | (comparison["jam_count_offpeak"] > 0)
    )]

    peak_valid = has_data[has_data["jam_count_peak"] > 0]
    offpeak_valid = has_data[has_data["jam_count_offpeak"] > 0]

    rho_peak = rho_offpeak = float("nan")
    p_peak = p_offpeak = float("nan")
    if len(peak_valid) >= 10:
        rho_peak, p_peak = sp_stats.spearmanr(peak_valid["vc"], peak_valid["jam_count_peak"])
    if len(offpeak_valid) >= 10:
        rho_offpeak, p_offpeak = sp_stats.spearmanr(offpeak_valid["vc"], offpeak_valid["jam_count_offpeak"])

    summary = {
        "total_jams_workday": len(jams_wd),
        "jams_peak_hours": len(peak_jams),
        "jams_offpeak": len(offpeak_jams),
        "peak_share_pct": round(len(peak_jams) / max(len(jams_wd), 1) * 100, 1),
        "spearman_peak": round(float(rho_peak), 4),
        "p_peak": float(p_peak),
        "spearman_offpeak": round(float(rho_offpeak), 4),
        "p_offpeak": float(p_offpeak),
        "time_band_correlations": band_correlations,
    }

    # ------------------------------------------------------------------
    # 4. Dual scatter: peak vs off-peak
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    if len(peak_valid) >= 5:
        ax1.scatter(peak_valid["vc"], peak_valid["jam_count_peak"], alpha=0.3, s=8, edgecolors="none")
        ax1.set_title(f"Špička (7–9, 15–17)  ρ = {rho_peak:.3f}")
    else:
        ax1.set_title("Špička — nedostatek dat")
    ax1.set_xlabel("Modelový V/C")
    ax1.set_ylabel("Počet zácp (Waze)")

    if len(offpeak_valid) >= 5:
        ax2.scatter(offpeak_valid["vc"], offpeak_valid["jam_count_offpeak"], alpha=0.3, s=8, edgecolors="none")
        ax2.set_title(f"Mimo špičku  ρ = {rho_offpeak:.3f}")
    else:
        ax2.set_title("Mimo špičku — nedostatek dat")
    ax2.set_xlabel("Modelový V/C")
    ax2.set_ylabel("Počet zácp (Waze)")

    fig.suptitle("V/C korelace: špička vs. mimo špičku", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir / "peak_vs_offpeak.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Time-band correlation bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(band_df))
    rhos = band_df["spearman_rho"].values
    colors = ["coral" if r > 0 else "steelblue" for r in rhos]
    ax.bar(x, rhos, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(band_df["time_band"], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ (V/C vs. počet zácp)")
    ax.set_title("Korelace V/C s frekvencí zácp podle časového pásma")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "band_correlations.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Hourly profile by road class (top 5)
    # ------------------------------------------------------------------
    type_map = net.set_index("link_id")["link_type"].to_dict() if "link_type" in net.columns else {}
    jams_wd["link_type"] = jams_wd["link_id"].map(type_map)
    top_types = jams_wd["link_type"].value_counts().head(5).index.tolist()

    if top_types:
        fig, ax = plt.subplots(figsize=(10, 5))
        for lt in top_types:
            sub = jams_wd[jams_wd["link_type"] == lt]
            hourly_lt = sub.groupby("hour").size().reindex(range(24), fill_value=0)
            ax.plot(range(24), hourly_lt.values, marker="o", markersize=3, label=str(lt))
        ax.set_xlabel("Hodina dne")
        ax.set_ylabel("Počet zácp")
        ax.set_title("Hodinové profily zácp podle třídy komunikace")
        ax.set_xticks(range(24))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save_figure(fig, out_dir / "hourly_by_class.png")
        plt.close(fig)

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
