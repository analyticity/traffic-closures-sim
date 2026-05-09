#!/usr/bin/env python3
"""Experiment 05: Point-based vs line-based closure matching.

For closures that have ``line_wkt`` geometry, compares:
 a) point matching  — nearest link to (lat, lon)
 b) line matching   — buffer the line, intersect with network links
 c) direction-aware — apply only to the specified direction (ab/ba)
 d) undirected      — apply to both directions

Runs assignment for each variant and compares ΔVHT and affected link count.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

from _common import (
    compute_scenario_kpis,
    ensure_metric_links,
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
NAME = "exp05_line_vs_point"

BUFFER_M = 50


def _match_point(lat: float, lon: float, links: gpd.GeoDataFrame, max_dist: float = 200) -> List[int]:
    """Match by nearest link to a point (distance in meters)."""
    nearby = match_links_near_point(lat, lon, links, max_dist_m=max_dist)
    return nearby["link_id"].astype(int).tolist()


def _match_line(line_wkt_str: str, links: gpd.GeoDataFrame) -> List[int]:
    """Match by buffering the line geometry (BUFFER_M meters) and intersecting."""
    try:
        geom = wkt.loads(line_wkt_str)
    except Exception:
        return []
    metric_links = ensure_metric_links(links)
    line_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(metric_links.crs)
    buffered = line_gdf.geometry.iloc[0].buffer(BUFFER_M)
    mask = metric_links.geometry.intersects(buffered)
    matched = links.loc[mask, "link_id"].astype(int).tolist()
    if len(matched) > 200:
        logger.warning("Line buffer matched %d links — capping at 200", len(matched))
        matched = matched[:200]
    return matched


def _build_scenario_links(
    link_ids: List[int],
    direction: str,
    links: gpd.GeoDataFrame,
    closure_type: str = "full",
) -> List[Dict[str, Any]]:
    result = []
    for lid in link_ids:
        row = links[links["link_id"] == lid]
        lanes = max(int(row.iloc[0].get("lanes", 2)), 1) if not row.empty else 2
        result.append({
            "link_id": lid,
            "direction": direction,
            "closure_type": closure_type,
            "lanes": lanes,
            "lanes_remaining": 0,
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

    # Select closures that have line geometry
    has_line = closures[
        closures["line_wkt"].notna() & (closures["line_wkt"] != "")
    ].copy()

    if "quality_score" in has_line.columns:
        has_line["quality_score"] = pd.to_numeric(has_line["quality_score"], errors="coerce").fillna(0)
        has_line = has_line.sort_values("quality_score", ascending=False)

    test_closures = has_line.head(5)
    if test_closures.empty:
        print(f"[{NAME}] No closures with line geometry found.")
        return

    logger.info("Testing %d closures with line geometry", len(test_closures))
    results = []

    for _, cl in test_closures.iterrows():
        cid = int(cl["id"])
        desc = str(cl.get("description_cs", cl.get("road_number", "")))[:50]
        lat, lon = cl.get("lat"), cl.get("lon")
        line_wkt_str = str(cl["line_wkt"])

        raw_dir = str(cl.get("closure_direction", "")).strip().lower()
        directed = "ab" if raw_dir == "aligned" else ("ba" if raw_dir == "opposite" else "both")

        # --- Point matching ---
        point_ids = _match_point(float(lat), float(lon), links) if pd.notna(lat) else []
        # --- Line matching ---
        line_ids = _match_line(line_wkt_str, links)

        row = {
            "closure_id": cid,
            "description": desc,
            "point_links": len(point_ids),
            "line_links": len(line_ids),
            "direction": directed,
        }

        # Run 4 variants if we have links for both
        variants = {}
        if point_ids:
            variants["point_both"] = _build_scenario_links(point_ids, "both", links)
        if line_ids:
            variants["line_both"] = _build_scenario_links(line_ids, "both", links)
        if line_ids and directed != "both":
            variants["line_directed"] = _build_scenario_links(line_ids, directed, links)
            variants["line_undirected"] = _build_scenario_links(line_ids, "both", links)

        for variant_name, sl in variants.items():
            try:
                scenario_df = run_scenario_assignment(cfg, sl)
                kpis = compute_scenario_kpis(baseline_merged, scenario_df)
                row[f"{variant_name}_dvht"] = kpis["delta_vht"]
                row[f"{variant_name}_dvht_pct"] = kpis["delta_vht_pct"]
                row[f"{variant_name}_n_links"] = len(sl)
            except Exception as e:
                logger.error("Variant %s for closure %d failed: %s", variant_name, cid, e)
                row[f"{variant_name}_dvht"] = None

        results.append(row)

    summary_df = pd.DataFrame(results)
    save_csv(summary_df, out_dir / "comparison.csv")
    save_json(results, out_dir / "summary.json")

    # ------------------------------------------------------------------
    # Bar chart: point vs line affected links
    # ------------------------------------------------------------------
    if not summary_df.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(summary_df))
        w = 0.35
        ax.bar(x - w/2, summary_df["point_links"], w, label="Bodový matching")
        ax.bar(x + w/2, summary_df["line_links"], w, label="Liniový matching")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{r['closure_id']}" for r in results], rotation=45, ha="right")
        ax.set_ylabel("Počet zasažených hran")
        ax.set_title("Porovnání bodového a liniového matchingu")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out_dir / "matching_comparison.png")
        plt.close(fig)

    print(f"[{NAME}] Done — {len(results)} closures → {out_dir}")


if __name__ == "__main__":
    main()
