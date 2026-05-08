#!/usr/bin/env python3
"""Experiment 06: Validate model predictions against real Waze jams (event_links).

The most novel validation — for closures with known jam impacts (via
``event_links`` + ``restriction_impact``), check whether the model
predicts congestion on the same links where real jams occurred.

Computes precision/recall of model congestion prediction and scatter of
model ΔVHT vs real total_delay_s.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    _cache_dir,
    compute_scenario_kpis,
    compute_vht,
    init_experiment,
    load_assignment_results,
    load_baseline_links,
    load_closures,
    load_segment_map,
    run_scenario_assignment,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp06_event_links"

VC_DELTA_THRESHOLD = 0.15


def _closure_to_scenario_links(
    closure: pd.Series,
    links: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Simple spatial matching for a closure."""
    lat, lon = closure.get("lat"), closure.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        return []

    import geopandas as _gpd
    from shapely.geometry import Point

    pt = Point(float(lon), float(lat))
    if links.crs and links.crs.to_epsg() != 4326:
        pt_gdf = _gpd.GeoDataFrame(geometry=[pt], crs="EPSG:4326").to_crs(links.crs)
        pt = pt_gdf.geometry.iloc[0]

    dists = links.geometry.distance(pt)
    nearby = links[dists < 200]
    if nearby.empty:
        nearby = links.loc[[dists.idxmin()]]

    sev = str(closure.get("pg_severity", closure.get("severity", "")))
    is_full = "full" in sev.lower() or "closure" in sev.lower()

    direction = "both"
    raw_dir = str(closure.get("closure_direction", "")).strip().lower()
    if raw_dir == "aligned":
        direction = "ab"
    elif raw_dir == "opposite":
        direction = "ba"

    result = []
    for _, link in nearby.iterrows():
        lanes = max(int(link.get("lanes", 2)), 1)
        result.append({
            "link_id": int(link["link_id"]),
            "direction": direction,
            "closure_type": "full" if is_full else "lanes",
            "lanes": lanes,
            "lanes_remaining": 0 if is_full else max(lanes - 1, 1),
        })
    return result


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    cache = _cache_dir(cfg)
    ri_path = cache / "restriction_impact.parquet"
    el_path = cache / "event_links.parquet"

    if not ri_path.exists():
        print(f"[{NAME}] restriction_impact.parquet not found — run fetch-data first.")
        return

    ri = pd.read_parquet(ri_path)
    el = pd.read_parquet(el_path) if el_path.exists() else pd.DataFrame()

    links = load_baseline_links(cfg)
    baseline = load_assignment_results(cfg)
    closures = load_closures(cfg)
    seg_map = load_segment_map(cfg, links)

    from sim._metrics import aggregate_daily_volumes
    aggregate_daily_volumes(baseline)

    baseline_merged = links.copy()
    for c in baseline.columns:
        if c != "link_id" and c not in baseline_merged.columns:
            baseline_merged = baseline_merged.merge(baseline[["link_id", c]], on="link_id", how="left")
    aggregate_daily_volumes(baseline_merged)

    # Invert seg_map for segment lookup
    link_to_seg = {v: k for k, v in seg_map.items()}

    # Focus on closures that have real jam data
    closures_with_impact = ri[ri.get("jam_count", pd.Series(dtype=int)).fillna(0) > 0].copy()
    if closures_with_impact.empty:
        closures_with_impact = ri.head(10)

    test = closures_with_impact.head(15)
    logger.info("Evaluating %d closures with known jam impacts", len(test))

    # Map event_links jams to link_ids
    if not el.empty and "segment_id" in el.columns:
        el["link_id"] = el["segment_id"].map(seg_map)
    else:
        el = pd.DataFrame(columns=["restriction_id", "link_id"])

    results = []
    for _, ri_row in test.iterrows():
        rid = int(ri_row.get("restriction_id", ri_row.get("id", 0)))
        cl_rows = closures[closures["id"] == rid]
        if cl_rows.empty:
            continue

        cl = cl_rows.iloc[0]
        scenario_links = _closure_to_scenario_links(cl, links)
        if not scenario_links:
            continue

        # Real jam link IDs (from event_links)
        real_jam_links = set()
        if not el.empty:
            cl_events = el[el["restriction_id"] == rid]
            real_jam_links = set(cl_events["link_id"].dropna().astype(int).tolist())

        try:
            scenario_df = run_scenario_assignment(cfg, scenario_links)
        except Exception as e:
            logger.error("Scenario for restriction %d failed: %s", rid, e)
            continue

        # Find model-predicted congested links: links where V/C increased significantly
        merged = baseline_merged[["link_id"]].copy()
        merged["baseline_voc"] = baseline_merged.get("VOC_max", pd.Series(dtype=float)).fillna(0).values

        scenario_voc = scenario_df.set_index("link_id").get("VOC_max", pd.Series(dtype=float)).fillna(0)
        merged["scenario_voc"] = merged["link_id"].map(scenario_voc).fillna(0)
        merged["delta_voc"] = merged["scenario_voc"] - merged["baseline_voc"]
        model_congested = set(merged[merged["delta_voc"] > VC_DELTA_THRESHOLD]["link_id"].astype(int))

        # Precision / recall
        tp = len(model_congested & real_jam_links)
        fp = len(model_congested - real_jam_links)
        fn = len(real_jam_links - model_congested)

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

        kpis = compute_scenario_kpis(baseline_merged, scenario_df)
        real_delay = float(ri_row.get("total_delay_s", 0))

        row = {
            "restriction_id": rid,
            "description": str(cl.get("description_cs", ""))[:50],
            "real_jam_links": len(real_jam_links),
            "model_congested_links": len(model_congested),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "model_dvht": kpis["delta_vht"],
            "real_total_delay_s": real_delay,
        }
        results.append(row)

    if not results:
        print(f"[{NAME}] No closures could be evaluated.")
        return

    results_df = pd.DataFrame(results)
    save_csv(results_df, out_dir / "precision_recall.csv")

    summary = {
        "n_closures": len(results),
        "mean_precision": round(float(results_df["precision"].mean()), 3),
        "mean_recall": round(float(results_df["recall"].mean()), 3),
        "mean_f1": round(float(results_df["f1"].mean()), 3),
    }

    # ------------------------------------------------------------------
    # Scatter: model ΔVHT vs real total delay
    # ------------------------------------------------------------------
    valid = results_df.dropna(subset=["model_dvht", "real_total_delay_s"])
    valid = valid[(valid["model_dvht"] > 0) & (valid["real_total_delay_s"] > 0)]
    if len(valid) >= 3:
        rho, p = sp_stats.spearmanr(valid["model_dvht"], valid["real_total_delay_s"])
        summary["dvht_vs_delay_spearman"] = round(float(rho), 3)
        summary["dvht_vs_delay_p"] = float(p)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(valid["real_total_delay_s"] / 3600, valid["model_dvht"], s=40, alpha=0.7)
        ax.set_xlabel("Reálné zpoždění (Waze, hod)")
        ax.set_ylabel("Modelový ΔVHT (voz·hod)")
        ax.set_title(f"ΔVHT vs reálné zpoždění  (ρ = {rho:.3f})")
        for _, r in valid.iterrows():
            ax.annotate(str(r["restriction_id"]),
                        (r["real_total_delay_s"] / 3600, r["model_dvht"]),
                        fontsize=7, alpha=0.7)
        save_figure(fig, out_dir / "scatter_delay.png")
        plt.close(fig)

    # Precision/recall bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    w = 0.3
    ax.bar(x - w, results_df["precision"], w, label="Precision", color="steelblue")
    ax.bar(x, results_df["recall"], w, label="Recall", color="coral")
    ax.bar(x + w, results_df["f1"], w, label="F1", color="mediumseagreen")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in results_df["restriction_id"]], rotation=45, ha="right")
    ax.set_ylabel("Skóre")
    ax.set_title("Precision / Recall validace kongesce")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, out_dir / "precision_recall.png")
    plt.close(fig)

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done — {len(results)} closures → {out_dir}")


if __name__ == "__main__":
    main()
