#!/usr/bin/env python3
"""Experiment 11: Model speeds vs Waze-derived free-flow (segment stats).

Joins ``jams_segment_stats.parquet`` (median ``speed_normal_kmh`` per DB segment)
to network links via ``road_segments`` OSM IDs, merges latest assignment
(``Congested_Time_Max``, ``VOC_max``, ``wd_daily_tot``), and writes:

* ``waze_vs_model_speed_by_link_type.csv`` — one row per link with a Waze match
* ``waze_vs_model_speed_summary.json`` — volume-weighted aggregates by
  ``link_type`` and coarse road class

With ``--ff-only`` (no assignment required):

* Same CSV/summary but **unit weights** (every matched link weight = 1) so metrics
  reflect **free-flow only**, not OD volumes or congested times.
* ``ff_speed_histogram.json`` — global and per-``link_type`` histogram of
  ``model_ff_kmh`` for quick checks after ``normalize-network``.

Prerequisites: ``fetch-data`` (PostgreSQL jams + road_segments),
``normalize-network`` (``network_links`` with ``osm_id``). Full mode also needs
``assign``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from _common import (
    init_experiment,
    load_assignment_results,
    load_jams_stats,
    load_network_links,
    load_segment_map,
    save_csv,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp11_waze_speeds"

# Metadata key kept for parity with earlier thesis runs (filtering threshold
# for optional downstream tables — all link types with matches are listed).
_MIN_LINKS_PER_ROW = 25

_COARSE_ORDER = ("other", "tertiary", "primary_secondary", "trunk", "motorway")


def _coarse_class(link_type: Any) -> str:
    lt = str(link_type or "")
    if lt in ("motorway", "motorway_link"):
        return "motorway"
    if lt in ("trunk", "trunk_link"):
        return "trunk"
    if lt in ("primary", "primary_link", "secondary", "secondary_link"):
        return "primary_secondary"
    if lt in ("tertiary", "tertiary_link"):
        return "tertiary"
    return "other"


def _strip_geometry(links: pd.DataFrame) -> pd.DataFrame:
    if hasattr(links, "geometry") and "geometry" in links.columns:
        return pd.DataFrame(links.drop(columns="geometry"))
    return pd.DataFrame(links)


def _waze_by_link(stats: pd.DataFrame, seg_to_link: Dict[int, int]) -> pd.DataFrame:
    """Aggregate jam segment stats onto ``link_id`` (multiple segments → one link)."""
    s = stats.copy()
    if "segment_id" not in s.columns:
        raise ValueError("jams_segment_stats.parquet must contain segment_id")
    s["link_id"] = s["segment_id"].map(lambda x: seg_to_link.get(int(x)) if pd.notna(x) else None)
    s = s[s["link_id"].notna()].copy()
    s["link_id"] = s["link_id"].astype(int)

    s["jam_count"] = pd.to_numeric(s.get("jam_count"), errors="coerce").fillna(0)
    s["speed_normal_median"] = pd.to_numeric(s.get("speed_normal_median"), errors="coerce")

    rows: List[Dict[str, Any]] = []
    for lid, g in s.groupby("link_id"):
        w = g["jam_count"].to_numpy(dtype=float)
        sp = g["speed_normal_median"].to_numpy(dtype=float)
        valid = np.isfinite(sp) & (sp > 0)
        w_eff = np.where(valid, w, 0.0)
        sp_eff = np.where(valid, sp, np.nan)
        wsum = float(np.sum(w_eff))
        jam_total = int(float(np.nansum(g["jam_count"].to_numpy(dtype=float))))
        if wsum > 0:
            waze_ff = float(np.nansum(sp_eff * w_eff) / wsum)
        else:
            waze_ff = float(np.nanmedian(sp_eff)) if np.any(np.isfinite(sp_eff)) else float("nan")
        rows.append({"link_id": int(lid), "waze_ff_kmh": waze_ff, "jam_count": jam_total})

    return pd.DataFrame(rows)


def _vol_weighted_mean(df: pd.DataFrame, col: str, wcol: str) -> float:
    w = pd.to_numeric(df[wcol], errors="coerce").fillna(0).to_numpy()
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    finite = np.isfinite(x)
    w_eff = np.where(finite, w, 0.0)
    s = float(np.sum(w_eff))
    if s <= 0:
        return float("nan")
    return float(np.nansum(x * w_eff) / s)


def _vol_weighted_mean_abs_pct_ff_vs_waze(df: pd.DataFrame, wcol: str) -> float:
    w = pd.to_numeric(df[wcol], errors="coerce").fillna(0).to_numpy()
    ff = pd.to_numeric(df["model_ff_kmh"], errors="coerce").to_numpy()
    wz = pd.to_numeric(df["waze_ff_kmh"], errors="coerce").to_numpy()
    pct = np.where(
        np.isfinite(ff) & np.isfinite(wz) & (wz > 0),
        np.abs((ff - wz) / wz * 100.0),
        np.nan,
    )
    finite = np.isfinite(pct)
    pct_f = np.where(finite, pct, 0.0)
    w_eff = np.where(finite, w, 0.0)
    s = float(np.sum(w_eff))
    if s <= 0:
        return float("nan")
    return float(np.sum(pct_f * w_eff) / s)


def _mean_round(series: pd.Series, ndigits: int = 3) -> Optional[float]:
    m = float(pd.to_numeric(series, errors="coerce").mean())
    return round(m, ndigits) if np.isfinite(m) else None


def _build_summary(
    matched: pd.DataFrame,
    vol_col: str,
    *,
    volume_source_col: str,
) -> Dict[str, Any]:
    rows_lt: List[Dict[str, Any]] = []
    for lt, g in matched.groupby("link_type", dropna=False):
        if len(g) == 0:
            continue
        row_d: Dict[str, Any] = {
            "link_type": str(lt),
            "n_links": int(len(g)),
            "vol_weighted_model_ff_kmh": round(
                _vol_weighted_mean(g, "model_ff_kmh", vol_col), 2
            ),
            "vol_weighted_waze_ff_kmh": round(
                _vol_weighted_mean(g, "waze_ff_kmh", vol_col), 2
            ),
            "mean_abs_pct_diff_model_ff_vs_waze": round(
                _vol_weighted_mean_abs_pct_ff_vs_waze(g, vol_col), 2
            ),
        }
        if "model_cong_kmh" in g.columns and g["model_cong_kmh"].notna().any():
            row_d["vol_weighted_model_cong_kmh"] = round(
                _vol_weighted_mean(g, "model_cong_kmh", vol_col), 2
            )
            row_d["vol_weighted_bias_cong_vs_waze_pct"] = round(
                _vol_weighted_mean(g, "pct_cong_vs_waze", vol_col), 2
            )
        if "vc_max" in g.columns:
            row_d["mean_vc_max"] = _mean_round(g["vc_max"])
        rows_lt.append(row_d)
    rows_lt.sort(key=lambda r: (-r["n_links"], r["link_type"]))

    matched = matched.copy()
    matched["_coarse"] = matched["link_type"].map(_coarse_class)
    rows_cc: List[Dict[str, Any]] = []
    for cc in _COARSE_ORDER:
        g = matched[matched["_coarse"] == cc]
        if len(g) == 0:
            continue
        row_c: Dict[str, Any] = {
            "coarse_class": cc,
            "n_links": int(len(g)),
            "vol_weighted_waze_ff_kmh": round(
                _vol_weighted_mean(g, "waze_ff_kmh", vol_col), 2
            ),
        }
        if "model_cong_kmh" in g.columns and g["model_cong_kmh"].notna().any():
            row_c["vol_weighted_model_cong_kmh"] = round(
                _vol_weighted_mean(g, "model_cong_kmh", vol_col), 2
            )
            row_c["vol_weighted_bias_cong_vs_waze_pct"] = round(
                _vol_weighted_mean(g, "pct_cong_vs_waze", vol_col), 2
            )
        rows_cc.append(row_c)

    return {
        "volume_column": volume_source_col,
        "min_links_per_row": _MIN_LINKS_PER_ROW,
        "n_links_with_waze_match": int(len(matched)),
        "by_link_type": rows_lt,
        "by_coarse_class": rows_cc,
    }


def _histogram_speeds_kmh(values: np.ndarray) -> Dict[str, Any]:
    s = values.astype(float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return {"n": 0, "bin_edges_kmh": [], "counts": [], "mean": None, "median": None}
    bins = np.arange(0.0, 135.0, 5.0)
    counts, edges = np.histogram(s, bins=bins)
    return {
        "n": int(s.size),
        "bin_edges_kmh": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "mean": round(float(np.mean(s)), 3),
        "median": round(float(np.median(s)), 3),
    }


def _build_ff_histogram_report(merged: pd.DataFrame) -> Dict[str, Any]:
    ff = pd.to_numeric(merged["model_ff_kmh"], errors="coerce").to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "global_model_ff_kmh": _histogram_speeds_kmh(ff),
        "by_link_type": {},
    }
    if "link_type" not in merged.columns:
        return out
    for lt, g in merged.groupby("link_type", dropna=False):
        arr = pd.to_numeric(g["model_ff_kmh"], errors="coerce").to_numpy(dtype=float)
        out["by_link_type"][str(lt)] = _histogram_speeds_kmh(arr)
    return out


def main() -> None:
    cfg, out_dir = init_experiment(NAME)
    ff_only = bool(cfg.get("__exp11_ff_only__"))

    links = load_network_links(cfg)
    net = _strip_geometry(links)
    base_cols = ["link_id", "link_type", "distance"]
    for c in ("speed_ab", "osm_id"):
        if c in net.columns:
            base_cols.append(c)
    base_cols = [c for c in base_cols if c in net.columns]

    if "speed_ab" not in net.columns:
        raise ValueError("network_links missing speed_ab (free-flow speed)")
    if "distance" not in net.columns:
        raise ValueError("network_links missing distance (meters)")

    vol_col: str
    volume_source_col: str

    if ff_only:
        merged = net[base_cols].copy()
        merged["model_ff_kmh"] = pd.to_numeric(merged["speed_ab"], errors="coerce")
        merged["model_cong_kmh"] = np.nan
        merged["Congested_Time_Max"] = np.nan
        merged["vc_max"] = np.nan
        merged["model_volume_tot"] = 1.0
        vol_col = "model_volume_tot"
        volume_source_col = "unit_weight_ff_only"
        logger.info("FF-only mode: skipping assignment merge (uniform link weights)")
    else:
        assign = load_assignment_results(cfg)
        vol_col = "wd_daily_tot" if "wd_daily_tot" in assign.columns else ""
        if not vol_col:
            for c in assign.columns:
                if c.startswith("wd_daily") and c.endswith("tot"):
                    vol_col = c
                    break
        if not vol_col:
            raise ValueError("No wd_daily_tot (or similar) volume column in assignment results")

        merged = net[base_cols].merge(assign, on="link_id", how="inner")
        if "Congested_Time_Max" not in merged.columns:
            raise ValueError("assignment results missing Congested_Time_Max")

        merged["model_ff_kmh"] = pd.to_numeric(merged["speed_ab"], errors="coerce")
        dist = pd.to_numeric(merged["distance"], errors="coerce")
        ct = pd.to_numeric(merged["Congested_Time_Max"], errors="coerce")
        merged["model_cong_kmh"] = np.where(
            ct.notna() & (ct > 0) & dist.notna(),
            dist * 3.6 / ct,
            np.nan,
        )

        voc_col: Optional[str] = "VOC_max" if "VOC_max" in merged.columns else None
        if voc_col:
            merged["vc_max"] = pd.to_numeric(merged[voc_col], errors="coerce")
        else:
            merged["vc_max"] = np.nan
            logger.warning("VOC_max not found; vc_max column will be empty")

        merged["model_volume_tot"] = pd.to_numeric(merged[vol_col], errors="coerce").fillna(0)
        volume_source_col = vol_col

    stats = load_jams_stats(cfg)
    seg_map = load_segment_map(cfg, links)
    if not seg_map:
        raise RuntimeError(
            "Empty segment→link map. Ensure road_segments.parquet exists and "
            "network_links contain osm_id (run fetch-data + normalize-network)."
        )

    waze_link = _waze_by_link(stats, seg_map)
    merged = merged.merge(waze_link, on="link_id", how="inner")

    wz = pd.to_numeric(merged["waze_ff_kmh"], errors="coerce")
    mc = pd.to_numeric(merged.get("model_cong_kmh"), errors="coerce") if "model_cong_kmh" in merged.columns else pd.Series(np.nan, index=merged.index)
    merged["pct_cong_vs_waze"] = np.where(
        wz.notna() & (wz > 0) & mc.notna(),
        (mc - wz) / wz * 100.0,
        np.nan,
    )

    csv_cols = [
        "link_id",
        "link_type",
        "model_volume_tot",
        "speed_ab",
        "distance",
        "model_ff_kmh",
        "waze_ff_kmh",
        "jam_count",
    ]
    if "Congested_Time_Max" in merged.columns:
        csv_cols.insert(5, "Congested_Time_Max")
    if "model_cong_kmh" in merged.columns:
        csv_cols.append("model_cong_kmh")
    if "vc_max" in merged.columns:
        csv_cols.append("vc_max")
    csv_cols.append("pct_cong_vs_waze")
    csv_cols = [c for c in csv_cols if c in merged.columns]
    out_csv = merged[csv_cols].copy()

    save_csv(out_csv, out_dir / "waze_vs_model_speed_by_link_type.csv")
    summary = _build_summary(merged, vol_col, volume_source_col=volume_source_col)
    if ff_only:
        summary["mode"] = "ff_only"
    save_json(summary, out_dir / "waze_vs_model_speed_summary.json")

    if ff_only:
        hist = _build_ff_histogram_report(merged)
        hist["note"] = (
            "Histogram of model free-flow (speed_ab) on Waze-matched links; "
            "no assignment — use after normalize-network to isolate FF changes."
        )
        save_json(hist, out_dir / "ff_speed_histogram.json")

    logger.info(
        "Waze speed comparison: %d links with segment match, summary -> %s",
        len(out_csv),
        out_dir,
    )


if __name__ == "__main__":
    # Allow `python experiments/exp11_waze_speeds.py --help` without importing NAME twice
    main()
