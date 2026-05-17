#!/usr/bin/env python3
"""Experiment 03: Congestion pattern validation — model V/C vs Waze jams.

Validates whether the static daily model captures congestion *patterns*
(not magnitudes) using Waze crowdsourced jam reports as an independent proxy.

Sub-analyses:
 A) **Spatial**: Global & per-class Spearman of daily V/C vs jam frequency
 B) **Quartile analysis**: V/C quartiles vs jam counts + Mann–Whitney
 C) **Top-overlap**: High-V/C links vs high-jam links intersection
 D) **Temporal bands**: Does daily V/C correlate more with peak-hour jams?
    (merged from former exp09_temporal_profiles)
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats as sp_stats

from _common import (
    _cache_dir,
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

logger = logging.getLogger(__name__)
NAME = "exp03_congestion_patterns"

VC_TOP_FRAC = 0.10
JAM_TOP_FRAC = 0.30
MIN_PER_CLASS = 30

TIME_BANDS = {
    "ráno (6–9)": (6, 9),
    "dopoledne (9–12)": (9, 12),
    "odpoledne (12–15)": (12, 15),
    "odpolední špička (15–18)": (15, 18),
    "večer (18–22)": (18, 22),
}
PEAK_HOURS = {7, 8, 9, 15, 16, 17}
WORKDAY_RANGE = range(0, 5)


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

    # ==================================================================
    # D) Temporal band analysis (merged from exp09)
    # ==================================================================
    cache = _cache_dir(cfg)
    jams_path = cache / "jams.parquet"
    if jams_path.exists():
        try:
            jams_raw = pd.read_parquet(jams_path)

            if "hour" not in jams_raw.columns and "first_seen" in jams_raw.columns:
                jams_raw["first_seen"] = pd.to_datetime(jams_raw["first_seen"], errors="coerce", utc=True)
                jams_raw["hour"] = jams_raw["first_seen"].dt.hour
            if "weekday" not in jams_raw.columns and "first_seen" in jams_raw.columns:
                jams_raw["weekday"] = jams_raw["first_seen"].dt.weekday

            jams_wd = jams_raw[jams_raw["weekday"].isin(WORKDAY_RANGE)].copy()
            jams_wd["link_id"] = jams_wd["segment_id"].map(seg_map)
            jams_wd = jams_wd.dropna(subset=["link_id"])
            jams_wd["link_id"] = jams_wd["link_id"].astype(int)

            vc_map = df.set_index("link_id")["vc"].to_dict()

            # Hourly profile
            hourly = jams_wd.groupby("hour").size().reindex(range(24), fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["coral" if h in PEAK_HOURS else "steelblue" for h in range(24)]
            ax.bar(range(24), hourly.values, color=colors, edgecolor="white")
            ax.set_xlabel("Hodina dne")
            ax.set_ylabel("Počet zácp (Waze)")
            ax.set_title("Hodinový profil zácp (pracovní dny)")
            ax.set_xticks(range(24))
            ax.legend(handles=[Patch(color="coral", label="Špička"),
                                Patch(color="steelblue", label="Mimo špičku")])
            save_figure(fig, out_dir / "hourly_profile.png")
            plt.close(fig)

            # Per time-band correlation
            band_correlations = []
            for band_name, (h_start, h_end) in TIME_BANDS.items():
                band_jams = jams_wd[(jams_wd["hour"] >= h_start) & (jams_wd["hour"] < h_end)]
                band_counts = band_jams.groupby("link_id").size().reset_index(name="jam_count")
                band_counts["vc"] = band_counts["link_id"].map(vc_map)
                band_counts = band_counts.dropna(subset=["vc"])
                band_counts = band_counts[(band_counts["vc"] > 0) & (band_counts["jam_count"] > 0)]

                if len(band_counts) >= 10:
                    rho_b, p_b = sp_stats.spearmanr(band_counts["vc"], band_counts["jam_count"])
                else:
                    rho_b, p_b = float("nan"), float("nan")

                band_correlations.append({
                    "time_band": band_name,
                    "h_start": h_start, "h_end": h_end,
                    "n_jams": len(band_jams),
                    "n_links_matched": len(band_counts),
                    "spearman_rho": round(float(rho_b), 4),
                    "p_value": float(p_b),
                })

            band_df = pd.DataFrame(band_correlations)
            save_csv(band_df, out_dir / "time_band_correlations.csv")
            summary["temporal_band_correlations"] = band_correlations

            # Time-band bar chart
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(band_df))
            rhos = band_df["spearman_rho"].values
            bar_colors = ["coral" if r > 0 else "steelblue" for r in rhos]
            ax.bar(x, rhos, color=bar_colors, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(band_df["time_band"], rotation=30, ha="right")
            ax.set_ylabel("Spearman ρ (V/C vs. počet zácp)")
            ax.set_title("Korelace V/C s frekvencí zácp podle časového pásma")
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            save_figure(fig, out_dir / "band_correlations.png")
            plt.close(fig)

            # Peak vs off-peak comparison
            peak_jams = jams_wd[jams_wd["hour"].isin(PEAK_HOURS)]
            offpeak_jams = jams_wd[~jams_wd["hour"].isin(PEAK_HOURS)]
            peak_counts = peak_jams.groupby("link_id").size().reset_index(name="jc_peak")
            offpeak_counts = offpeak_jams.groupby("link_id").size().reset_index(name="jc_offpeak")

            comp = net[["link_id", "vc"]].merge(peak_counts, on="link_id", how="left")
            comp = comp.merge(offpeak_counts, on="link_id", how="left")
            comp["jc_peak"] = comp["jc_peak"].fillna(0)
            comp["jc_offpeak"] = comp["jc_offpeak"].fillna(0)

            pv = comp[(comp["vc"] > 0) & (comp["jc_peak"] > 0)]
            ov = comp[(comp["vc"] > 0) & (comp["jc_offpeak"] > 0)]

            rho_peak = rho_off = float("nan")
            if len(pv) >= 10:
                rho_peak, _ = sp_stats.spearmanr(pv["vc"], pv["jc_peak"])
            if len(ov) >= 10:
                rho_off, _ = sp_stats.spearmanr(ov["vc"], ov["jc_offpeak"])

            summary["temporal_peak_vs_offpeak"] = {
                "spearman_peak": round(float(rho_peak), 4),
                "spearman_offpeak": round(float(rho_off), 4),
                "peak_share_pct": round(len(peak_jams) / max(len(jams_wd), 1) * 100, 1),
            }

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            if len(pv) >= 5:
                ax1.scatter(pv["vc"], pv["jc_peak"], alpha=0.3, s=8, edgecolors="none")
                ax1.set_title(f"Špička (7–9, 15–17)  ρ = {rho_peak:.3f}")
            else:
                ax1.set_title("Špička — nedostatek dat")
            ax1.set_xlabel("Modelový V/C")
            ax1.set_ylabel("Počet zácp (Waze)")

            if len(ov) >= 5:
                ax2.scatter(ov["vc"], ov["jc_offpeak"], alpha=0.3, s=8, edgecolors="none")
                ax2.set_title(f"Mimo špičku  ρ = {rho_off:.3f}")
            else:
                ax2.set_title("Mimo špičku — nedostatek dat")
            ax2.set_xlabel("Modelový V/C")
            ax2.set_ylabel("Počet zácp (Waze)")

            fig.suptitle("V/C korelace: špička vs. mimo špičku", fontsize=13)
            fig.tight_layout()
            save_figure(fig, out_dir / "peak_vs_offpeak.png")
            plt.close(fig)

        except Exception as e:
            logger.warning("Temporal analysis skipped: %s", e)
            summary["temporal_analysis_warning"] = str(e)
    else:
        summary["temporal_analysis_note"] = "jams.parquet not found — temporal sub-analysis skipped"

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
