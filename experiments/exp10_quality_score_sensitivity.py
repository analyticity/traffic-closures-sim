#!/usr/bin/env python3
"""Experiment 10: Sensitivity of closure scenarios to ``quality_score`` filtering.

Unlike a single-closure sweep, this builds **one combined scenario per threshold**
that applies *all* active closures whose ``quality_score`` meets the minimum.
That way lowering the threshold monotonically adds links and should move ΔVHT,
instead of trivial flat curves when only one closure exists in the pool.
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

THRESHOLDS = [0, 30, 50, 70, 90]
MAX_CLOSURES_PER_THRESHOLD = 40
MATCH_RADIUS_M = 200.0


def _closures_meeting_qs(closures: pd.DataFrame, min_qs: float) -> pd.DataFrame:
    c = closures.copy()
    if "quality_score" not in c.columns:
        c["quality_score"] = 100.0
    c["_qs"] = pd.to_numeric(c["quality_score"], errors="coerce").fillna(0)
    c = c[c["_qs"] >= float(min_qs)]
    c = c[c["lat"].notna() & c["lon"].notna()]
    sev_col = "pg_severity" if "pg_severity" in c.columns else "severity"
    if sev_col in c.columns:
        sev = c[sev_col].astype(str)
        major = c[sev.str.contains("full|serious|standstill|closure", case=False, na=False)]
        if len(major) >= 3:
            c = major
    return c.sort_values("_qs", ascending=False)


def _build_union_scenario(
    closures_subset: pd.DataFrame,
    links,
) -> List[Dict[str, Any]]:
    """Merge unique link hits from many closures into one scenario list."""
    scenario: Dict[int, Dict[str, Any]] = {}
    for _, row in closures_subset.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        nearby = match_links_near_point(lat, lon, links, max_dist_m=MATCH_RADIUS_M)
        for _, link in nearby.iterrows():
            lid = int(link["link_id"])
            if lid in scenario:
                continue
            lanes = max(int(link.get("lanes", 2)), 1)
            scenario[lid] = {
                "link_id": lid,
                "direction": "both",
                "closure_type": "full",
                "lanes": lanes,
                "lanes_remaining": 0,
            }
    return list(scenario.values())


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

    qs_vals = pd.to_numeric(closures.get("quality_score", pd.Series(dtype=float)), errors="coerce")

    summary: Dict[str, Any] = {
        "total_closures": int(len(closures)),
        "quality_score_stats": {
            "mean": round(float(qs_vals.mean()), 3) if len(qs_vals) else None,
            "median": round(float(qs_vals.median()), 3) if len(qs_vals) else None,
            "null_pct": round(float(qs_vals.isna().mean() * 100), 2) if len(qs_vals) else None,
        },
        "threshold_results": [],
    }

    rows = []
    for thr in THRESHOLDS:
        pool = _closures_meeting_qs(closures, thr).head(MAX_CLOSURES_PER_THRESHOLD)
        scen = _build_union_scenario(pool, links)
        if not scen:
            rows.append({
                "min_quality_score": thr,
                "n_closures": int(len(pool)),
                "n_affected_links": 0,
                "skipped": True,
            })
            continue
        try:
            df = run_scenario_assignment(cfg, scen)
            kpis = compute_scenario_kpis(baseline_merged, df)
        except Exception as exc:
            logger.error("Threshold %s failed: %s", thr, exc)
            rows.append({"min_quality_score": thr, "error": str(exc)})
            continue
        row = {
            "min_quality_score": thr,
            "n_closures": int(len(pool)),
            "n_affected_links": len(scen),
            **kpis,
        }
        rows.append(row)
        summary["threshold_results"].append(row)

    if rows:
        save_csv(pd.DataFrame(rows), out_dir / "quality_score_sensitivity.csv")

    # Histogram of quality scores (closures with coords)
    valid_qs = closures.dropna(subset=["lat", "lon"])
    if "quality_score" in valid_qs.columns and len(valid_qs):
        fig0, ax0 = plt.subplots(figsize=(7, 4))
        qs = pd.to_numeric(valid_qs["quality_score"], errors="coerce").dropna()
        if len(qs):
            ax0.hist(qs, bins=min(20, max(5, int(len(qs) ** 0.5))), color="steelblue", edgecolor="white")
        for thr in THRESHOLDS[1:]:
            ax0.axvline(thr, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax0.set_xlabel("Quality score")
        ax0.set_ylabel("Počet uzavírek")
        ax0.set_title("Distribuce quality_score aktivních uzavírek")
        fig0.tight_layout()
        save_figure(fig0, out_dir / "quality_distribution.png")
        plt.close(fig0)

    # Sensitivity curves
    df_r = pd.DataFrame([r for r in rows if "delta_vht" in r and not r.get("skipped")])
    if not df_r.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(df_r["min_quality_score"], df_r["delta_vht"], "o-", color="steelblue")
        ax1.set_xlabel("Minimální quality_score")
        ax1.set_ylabel("ΔVHT (voz·hod)")
        ax1.set_title("Dopad na VHT podle kvality uzavírek")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df_r["min_quality_score"], df_r["n_affected_links"], "s-", color="darkorange")
        ax2.set_xlabel("Minimální quality_score")
        ax2.set_ylabel("Počet zasažených hran (unie)")
        ax2.set_title("Počet zahrnutých hran ve scénáři")
        ax2.grid(True, alpha=0.3)
        fig.suptitle("Citlivost na filtraci quality_score (kombinované scénáře)", fontsize=12)
        fig.tight_layout()
        save_figure(fig, out_dir / "sensitivity_curves.png")
        plt.close(fig)

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
