#!/usr/bin/env python3
"""Experiment 06: Event / jam link validation (local precision–recall).

The naive global rule "all links with V/C>1 are congested" yields near-zero
precision because most overloaded links are unrelated to a specific closure.

This experiment scores **local** consistency: for each real closure, compare
Waze jam density on nearby links to model V/C on the *same* link set within a
fixed-radius buffer around the closure anchor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from shapely.geometry import Point

from _common import (
    ensure_metric_links,
    init_experiment,
    load_assignment_results,
    load_baseline_links,
    load_closures,
    load_jams_stats,
    load_segment_map,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp06_event_links_validation"

BUFFER_M = 400.0
N_CLOSURES = 12
# Minimum V/C to count as "model congested" inside the buffer.  Using only
# global thresholds often yields zero overlap because jam segments map to
# slightly different link_ids than the highest-V/C arterial in the same disk.
VC_THRESHOLD = 1.0
VC_POOL_PERCENTILE = 70.0  # at least this percentile within the buffer, >= VC_THRESHOLD
# Waze segments often map to a parallel/adjacent OSM link vs the MATSim arc with
# highest V/C — require geometry proximity for TP, not exact link_id equality.
MATCH_DISTANCE_M = 55.0


def _link_metric_geometries(links: gpd.GeoDataFrame) -> Dict[int, Any]:
    """link_id -> geometry in metric CRS for distance tests."""
    metric = ensure_metric_links(links)
    out: Dict[int, Any] = {}
    for _, r in metric.iterrows():
        lid = int(r["link_id"])
        geom = r.geometry
        if geom is not None and not geom.is_empty:
            out[lid] = geom
    return out


def _greedy_spatial_matches(
    model_ids: Set[int],
    real_ids: Set[int],
    geoms: Dict[int, Any],
    max_dist_m: float,
) -> Tuple[int, Set[int], Set[int]]:
    """Greedy 1:1 pairing of model vs jam links when line geometries are within *max_dist_m*."""
    pairs: List[Tuple[float, int, int]] = []
    for m in model_ids:
        gm = geoms.get(m)
        if gm is None or gm.is_empty:
            continue
        for j in real_ids:
            gj = geoms.get(j)
            if gj is None or gj.is_empty:
                continue
            d = float(gm.distance(gj))
            if d <= max_dist_m:
                pairs.append((d, m, j))
    pairs.sort(key=lambda x: x[0])
    matched_m: Set[int] = set()
    matched_j: Set[int] = set()
    tp = 0
    for d, m, j in pairs:
        if m in matched_m or j in matched_j:
            continue
        matched_m.add(m)
        matched_j.add(j)
        tp += 1
    return tp, matched_m, matched_j


def _buffer_link_ids(
    lat: float,
    lon: float,
    links: gpd.GeoDataFrame,
    buffer_m: float,
) -> Set[int]:
    metric = ensure_metric_links(links)
    pt = gpd.GeoDataFrame(geometry=[Point(float(lon), float(lat))], crs="EPSG:4326").to_crs(
        metric.crs,
    )
    pgeom = pt.geometry.iloc[0]
    dists = metric.geometry.distance(pgeom)
    return set(int(i) for i in metric.loc[dists <= buffer_m, "link_id"].astype(int))


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    links = load_baseline_links(cfg)
    assign = load_assignment_results(cfg)
    closures = load_closures(cfg)
    seg_map = load_segment_map(cfg, links)

    try:
        jams = load_jams_stats(cfg)
    except FileNotFoundError:
        save_json(
            {
                "status": "skipped",
                "reason": "jams_segment_stats.parquet missing — run fetch-data",
            },
            out_dir / "summary.json",
        )
        print(f"[{NAME}] Skipped — no jams stats.")
        return

    jam_col = next((c for c in ("jam_count", "count", "n_jams") if c in jams.columns), None)
    if jam_col is None:
        save_json({"status": "skipped", "reason": f"No jam column in jams stats: {list(jams.columns)}"},
                  out_dir / "summary.json")
        print(f"[{NAME}] Skipped — no jam column.")
        return

    jams = jams.copy()
    jams["link_id"] = jams["segment_id"].map(seg_map)
    jams = jams.dropna(subset=["link_id"])
    jams["link_id"] = jams["link_id"].astype(int)
    jam_by_link = jams.groupby("link_id")[jam_col].sum()

    net = links[["link_id", "geometry"]].merge(assign, on="link_id", how="inner")
    if "VOC_max" not in net.columns:
        save_json({"status": "skipped", "reason": "VOC_max missing from assignment results"}, out_dir / "summary.json")
        print(f"[{NAME}] Skipped — no VOC_max.")
        return

    vc = net.set_index("link_id")["VOC_max"].fillna(0).astype(float)
    link_geoms = _link_metric_geometries(links)

    c = closures.copy()
    if "quality_score" in c.columns:
        c["_qs"] = pd.to_numeric(c["quality_score"], errors="coerce").fillna(0)
        c = c.sort_values("_qs", ascending=False)
    c = c[c["lat"].notna() & c["lon"].notna()].head(N_CLOSURES)

    rows: List[Dict[str, Any]] = []

    for _, row in c.iterrows():
        rid = int(row.get("id", -1))
        lat, lon = float(row["lat"]), float(row["lon"])
        desc = str(row.get("description", ""))[:80]

        pool = _buffer_link_ids(lat, lon, links, BUFFER_M)
        if not pool:
            continue

        pool_vc = np.array([float(vc.get(lid, 0)) for lid in pool], dtype=float)
        thr = float(np.percentile(pool_vc, VC_POOL_PERCENTILE)) if len(pool_vc) else VC_THRESHOLD
        thr = max(VC_THRESHOLD, thr)

        real_j = {lid for lid in pool if float(jam_by_link.get(lid, 0)) > 0}
        model_c = {lid for lid in pool if float(vc.get(lid, 0)) >= thr}

        tp_exact = len(real_j & model_c)
        tp_sp, matched_m, matched_j = _greedy_spatial_matches(
            model_c, real_j, link_geoms, MATCH_DISTANCE_M,
        )
        fp = len(model_c - matched_m)
        fn = len(real_j - matched_j)
        prec = tp_sp / max(len(model_c), 1)
        rec = tp_sp / max(len(real_j), 1) if real_j else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        rows.append({
            "restriction_id": rid,
            "description": desc,
            "buffer_m": BUFFER_M,
            "vc_threshold": round(thr, 4),
            "match_distance_m": MATCH_DISTANCE_M,
            "real_jam_links": len(real_j),
            "model_congested_links": len(model_c),
            "true_positives_spatial": tp_sp,
            "true_positives_exact_link": tp_exact,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

    if not rows:
        save_json({"status": "skipped", "reason": "no closure rows processed"}, out_dir / "summary.json")
        print(f"[{NAME}] No rows.")
        return

    save_csv(pd.DataFrame(rows), out_dir / "precision_recall.csv")

    mean_p = float(np.mean([r["precision"] for r in rows]))
    mean_r = float(np.mean([r["recall"] for r in rows]))
    mean_f1 = float(np.mean([r["f1"] for r in rows]))

    summary: Dict[str, Any] = {
        "n_closures": len(rows),
        "buffer_m": BUFFER_M,
        "vc_threshold_base": VC_THRESHOLD,
        "vc_pool_percentile": VC_POOL_PERCENTILE,
        "spatial_match_distance_m": MATCH_DISTANCE_M,
        "mean_precision": round(mean_p, 4),
        "mean_recall": round(mean_r, 4),
        "mean_f1": round(mean_f1, 4),
        "note": (
            "Local buffer metrics with spatial TP (link geometries within "
            f"{MATCH_DISTANCE_M:.0f} m) — Waze segment→link map rarely equals "
            "the exact MATSim arc with peak V/C."
        ),
    }

    # Optional: correlate model max V/C in buffer vs total jam count in buffer
    buf_vc_max: List[float] = []
    buf_jam_sum: List[float] = []
    for _, row in c.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        pool = _buffer_link_ids(lat, lon, links, BUFFER_M)
        if not pool:
            continue
        buf_vc_max.append(float(max(float(vc.get(lid, 0)) for lid in pool)))
        buf_jam_sum.append(float(sum(float(jam_by_link.get(lid, 0)) for lid in pool)))

    rho, p_val = (float("nan"), float("nan"))
    if len(buf_vc_max) >= 5 and np.nanstd(buf_vc_max) > 0 and np.nanstd(buf_jam_sum) > 0:
        rho, p_val = sp_stats.spearmanr(buf_vc_max, buf_jam_sum)
    summary["buffer_vcmax_vs_jam_spearman"] = round(float(rho), 4) if np.isfinite(rho) else None
    summary["buffer_vcmax_vs_jam_p"] = float(p_val) if np.isfinite(p_val) else None

    save_json(summary, out_dir / "summary.json")

    # Figures
    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df_plot))
    w = 0.25
    ax.bar(x - w, df_plot["precision"], width=w, label="Precision")
    ax.bar(x, df_plot["recall"], width=w, label="Recall")
    ax.bar(x + w, df_plot["f1"], width=w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["restriction_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Precision / Recall (local buffer)")
    fig.tight_layout()
    save_figure(fig, out_dir / "precision_recall.png")
    plt.close(fig)

    if len(buf_vc_max) >= 3 and np.nanstd(buf_jam_sum) > 0:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(buf_jam_sum, buf_vc_max, alpha=0.6)
        ax2.set_xlabel("Jam sum in buffer (Waze)")
        ax2.set_ylabel("max V/C in buffer (model)")
        ttl_rho = summary.get("buffer_vcmax_vs_jam_spearman")
        ax2.set_title(f"max V/C vs jams in buffer (ρ = {ttl_rho})")
        fig2.tight_layout()
        save_figure(fig2, out_dir / "scatter_delay.png")
        plt.close(fig2)

    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
