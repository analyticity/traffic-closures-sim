#!/usr/bin/env python3
"""Experiment 04: Closure scenario experiments.

Selects representative closures from the database, runs each as a
scenario against the calibrated baseline, and computes delta KPIs
(ΔVHT, ΔVKT, overloaded links).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    _cache_dir,
    city_display_name,
    compute_scenario_kpis,
    compute_vht,
    init_experiment,
    load_assignment_results,
    load_baseline_links,
    load_closures,
    load_network_links,
    match_links_near_point,
    run_scenario_assignment,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp04_closure_scenarios"


def _select_representative_closures(closures: pd.DataFrame, city_name: str = "") -> pd.DataFrame:
    """Pick up to 5 diverse closures for experiments."""
    candidates = closures.copy()

    # Prefer closures with line geometry and higher quality
    if "line_wkt" in candidates.columns:
        candidates["has_line"] = candidates["line_wkt"].notna() & (candidates["line_wkt"] != "")
    else:
        candidates["has_line"] = False
    if "quality_score" in candidates.columns:
        candidates["quality_score"] = pd.to_numeric(candidates["quality_score"], errors="coerce").fillna(50)
    else:
        candidates["quality_score"] = 50

    selected = []
    sev_col = "pg_severity" if "pg_severity" in candidates.columns else "severity"

    # a) Full closure on major road
    full = candidates[
        (candidates[sev_col].str.contains("full|closure", case=False, na=False)) &
        (candidates.get("road_type_code", pd.Series(dtype=str)).isin(
            ["motorway", "trunk", "primary", "secondary"]
        ) | candidates.get("road_number", pd.Series(dtype=str)).notna())
    ].sort_values("quality_score", ascending=False)
    if not full.empty:
        selected.append(full.iloc[0])

    # b) Lane reduction
    lane = candidates[
        candidates[sev_col].str.contains("lane|reduc|partial", case=False, na=False)
    ].sort_values("quality_score", ascending=False)
    if not lane.empty:
        selected.append(lane.iloc[0])

    # c) City center closure (if city column exists)
    if "city" in candidates.columns and city_name:
        center = candidates[
            candidates["city"].str.contains(city_name, case=False, na=False)
        ].sort_values("quality_score", ascending=False)
        if len(center) > len(selected):
            for _, row in center.iterrows():
                if row["id"] not in [s["id"] for s in selected]:
                    selected.append(row)
                    break

    # d) High-quality with line geometry
    with_line = candidates[candidates["has_line"]].sort_values("quality_score", ascending=False)
    for _, row in with_line.iterrows():
        if len(selected) >= 5:
            break
        if row["id"] not in [s["id"] for s in selected]:
            selected.append(row)

    # Fill remaining slots
    remaining = candidates.sort_values("quality_score", ascending=False)
    for _, row in remaining.iterrows():
        if len(selected) >= 5:
            break
        if row["id"] not in [s["id"] for s in selected]:
            selected.append(row)

    return pd.DataFrame(selected)


def _closure_to_scenario_links(
    closure: pd.Series,
    links: pd.DataFrame,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a single closure record into scenario_links format."""
    sev = str(closure.get("pg_severity", closure.get("severity", "")))
    is_full = "full" in sev.lower() or "closure" in sev.lower()

    lat, lon = closure.get("lat"), closure.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        return []

    nearby = match_links_near_point(float(lat), float(lon), links, max_dist_m=200)

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

    closures = load_closures(cfg)
    links = load_baseline_links(cfg)
    baseline_assign = load_assignment_results(cfg)

    from sim._metrics import aggregate_daily_volumes
    aggregate_daily_volumes(baseline_assign)

    selected = _select_representative_closures(closures, city_name=city_display_name(cfg))
    logger.info("Selected %d closures for scenarios", len(selected))

    if selected.empty:
        print(f"[{NAME}] No suitable closures found.")
        return

    # Merge assignment onto links for baseline KPIs
    baseline_merged = links.copy()
    for c in baseline_assign.columns:
        if c != "link_id" and c not in baseline_merged.columns:
            baseline_merged = baseline_merged.merge(
                baseline_assign[["link_id", c]], on="link_id", how="left",
            )
    aggregate_daily_volumes(baseline_merged)

    results = []
    for idx, (_, closure) in enumerate(selected.iterrows()):
        label = f"S{idx+1}"
        desc = str(closure.get("description_cs", closure.get("road_number", f"closure_{closure['id']}")))[:60]
        logger.info("Running scenario %s: %s", label, desc)

        scenario_links = _closure_to_scenario_links(closure, links, cfg)
        if not scenario_links:
            logger.warning("Skipping %s — no matching links", label)
            continue

        try:
            scenario_df = run_scenario_assignment(cfg, scenario_links)
        except Exception as e:
            logger.error("Scenario %s failed: %s", label, e)
            continue

        kpis = compute_scenario_kpis(baseline_merged, scenario_df)
        kpis["label"] = label
        kpis["description"] = desc
        kpis["closure_id"] = int(closure["id"])
        kpis["severity"] = str(closure.get("pg_severity", closure.get("severity", "")))
        kpis["n_affected_links"] = len(scenario_links)
        results.append(kpis)

        save_json(kpis, out_dir / f"scenario_{label}.json")

    if not results:
        print(f"[{NAME}] No scenarios completed.")
        return

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(results)
    save_csv(summary_df, out_dir / "summary.csv")

    # ------------------------------------------------------------------
    # KPI comparison bar chart
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    labels = summary_df["label"]
    x = np.arange(len(labels))

    axes[0].bar(x, summary_df["delta_vht"], color="steelblue")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("ΔVHT (voz·hod)")
    axes[0].set_title("Nárůst VHT")

    axes[1].bar(x, summary_df["delta_vht_pct"], color="coral")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("ΔVHT (%)")
    axes[1].set_title("Relativní nárůst VHT")

    axes[2].bar(x, summary_df["delta_overloaded"], color="goldenrod")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Δ přetížených hran")
    axes[2].set_title("Nárůst přetížených hran (V/C>1)")

    fig.suptitle("Porovnání scénářů uzavírek", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir / "kpi_comparison.png")
    plt.close(fig)

    save_json(results, out_dir / "summary.json")
    print(f"[{NAME}] Done — {len(results)} scenarios → {out_dir}")


if __name__ == "__main__":
    main()
