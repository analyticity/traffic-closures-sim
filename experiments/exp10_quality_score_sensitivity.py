#!/usr/bin/env python3
"""Experiment 10: Sensitivity to quality_score filtering of closures.

Compares scenario results when including ALL closures (min_quality_score=0)
versus only high-quality closures (min_quality_score ≥ 70).  Determines
whether low-quality records add noise without changing aggregate KPIs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    compute_scenario_kpis,
    init_experiment,
    load_assignment_results,
    load_baseline_links,
    load_closures,
    match_links_near_point,
    run_scenario_assignment,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp10_quality_score_sensitivity"

QUALITY_THRESHOLDS = [0, 30, 50, 70, 90]


def _closures_to_scenario_links(
    closures_subset: pd.DataFrame,
    links: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Convert a set of closures into scenario_links for a single day."""
    result: List[Dict[str, Any]] = []
    seen_link_ids: set = set()

    for _, cl in closures_subset.iterrows():
        lat, lon = cl.get("lat"), cl.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue

        nearby = match_links_near_point(float(lat), float(lon), links, max_dist_m=150)

        sev = str(cl.get("pg_severity", cl.get("severity", "")))
        is_full = "full" in sev.lower() or "closure" in sev.lower()

        direction = "both"
        raw_dir = str(cl.get("closure_direction", "")).strip().lower()
        if raw_dir == "aligned":
            direction = "ab"
        elif raw_dir == "opposite":
            direction = "ba"

        for _, link in nearby.iterrows():
            lid = int(link["link_id"])
            if lid in seen_link_ids:
                continue
            seen_link_ids.add(lid)
            lanes = max(int(link.get("lanes", 2)), 1)
            result.append({
                "link_id": lid,
                "direction": direction,
                "closure_type": "full" if is_full else "lanes",
                "lanes": lanes,
                "lanes_remaining": 0 if is_full else max(lanes - 1, 1),
            })

    return result


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    closures = load_closures(cfg)
    links = load_baseline_links(cfg)
    baseline = load_assignment_results(cfg)

    from sim._metrics import aggregate_daily_volumes
    aggregate_daily_volumes(baseline)

    baseline_merged = links.copy()
    for c in baseline.columns:
        if c != "link_id" and c not in baseline_merged.columns:
            baseline_merged = baseline_merged.merge(baseline[["link_id", c]], on="link_id", how="left")
    aggregate_daily_volumes(baseline_merged)

    # Ensure quality_score is numeric
    if "quality_score" in closures.columns:
        closures["quality_score"] = pd.to_numeric(closures["quality_score"], errors="coerce").fillna(0)
    else:
        closures["quality_score"] = 50
        logger.warning("No quality_score in closures — using default 50 for all")

    # Pick a representative date with many active closures
    if "valid_from" in closures.columns and "valid_to" in closures.columns:
        closures["valid_from"] = pd.to_datetime(closures["valid_from"], errors="coerce")
        closures["valid_to"] = pd.to_datetime(closures["valid_to"], errors="coerce")
        # Use the date with most active closures
        valid = closures.dropna(subset=["valid_from", "valid_to"])
        if not valid.empty:
            all_dates = pd.date_range(valid["valid_from"].min(), valid["valid_to"].max(), freq="D")
            best_date = None
            best_count = 0
            for d in all_dates[:365]:
                active = valid[(valid["valid_from"] <= d) & (valid["valid_to"] >= d)]
                if len(active) > best_count:
                    best_count = len(active)
                    best_date = d
            if best_date is not None:
                closures = closures[
                    (closures["valid_from"] <= best_date) & (closures["valid_to"] >= best_date)
                ].copy()
                logger.info("Using date %s with %d active closures", best_date.date(), len(closures))
    else:
        # Use all closures if no date columns
        closures = closures.head(50)

    if closures.empty:
        print(f"[{NAME}] No closures found for the selected date.")
        return

    # ------------------------------------------------------------------
    # Run scenario for each quality threshold
    # ------------------------------------------------------------------
    results = []
    for threshold in QUALITY_THRESHOLDS:
        subset = closures[closures["quality_score"] >= threshold]
        n_closures = len(subset)
        logger.info("Threshold %d: %d closures", threshold, n_closures)

        if subset.empty:
            results.append({
                "min_quality_score": threshold,
                "n_closures": 0,
                "n_affected_links": 0,
                "delta_vht": 0,
                "delta_vht_pct": 0,
                "delta_overloaded": 0,
            })
            continue

        scenario_links = _closures_to_scenario_links(subset, links)
        if not scenario_links:
            results.append({
                "min_quality_score": threshold,
                "n_closures": n_closures,
                "n_affected_links": 0,
                "delta_vht": 0,
                "delta_vht_pct": 0,
                "delta_overloaded": 0,
            })
            continue

        try:
            scenario_df = run_scenario_assignment(cfg, scenario_links)
            kpis = compute_scenario_kpis(baseline_merged, scenario_df)
        except Exception as e:
            logger.error("Threshold %d failed: %s", threshold, e)
            continue

        row = {
            "min_quality_score": threshold,
            "n_closures": n_closures,
            "n_affected_links": len(scenario_links),
            **kpis,
        }
        results.append(row)

    if not results:
        print(f"[{NAME}] No scenarios completed.")
        return

    results_df = pd.DataFrame(results)
    save_csv(results_df, out_dir / "quality_score_sensitivity.csv")

    # ------------------------------------------------------------------
    # Quality score distribution of closures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(closures["quality_score"], bins=20, edgecolor="white", alpha=0.8)
    for t in QUALITY_THRESHOLDS[1:]:
        ax.axvline(t, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Počet uzavírek")
    ax.set_title("Distribuce quality_score aktivních uzavírek")
    save_figure(fig, out_dir / "quality_distribution.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # ΔVHT vs quality threshold
    # ------------------------------------------------------------------
    valid_results = results_df.dropna(subset=["delta_vht"])
    if len(valid_results) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(valid_results["min_quality_score"], valid_results["delta_vht"],
                 "o-", color="steelblue", markersize=8)
        ax1.set_xlabel("Minimální quality_score")
        ax1.set_ylabel("ΔVHT (voz·hod)")
        ax1.set_title("Dopad na VHT podle kvality uzavírek")
        ax1.grid(True, alpha=0.3)

        ax2.plot(valid_results["min_quality_score"], valid_results["n_closures"],
                 "s-", color="coral", markersize=8)
        ax2.set_xlabel("Minimální quality_score")
        ax2.set_ylabel("Počet uzavírek")
        ax2.set_title("Počet zahrnutých uzavírek")
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Citlivost na filtraci quality_score", fontsize=13)
        fig.tight_layout()
        save_figure(fig, out_dir / "sensitivity_curves.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "total_closures": len(closures),
        "quality_score_stats": {
            "mean": round(float(closures["quality_score"].mean()), 1),
            "median": round(float(closures["quality_score"].median()), 1),
            "null_pct": round(float((closures["quality_score"] == 0).mean() * 100), 1),
        },
        "threshold_results": results,
    }
    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
