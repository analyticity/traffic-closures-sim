"""Supply-side audit: basic diagnostics on VDF parameters, capacities, and speeds.

Produces ``supply_audit.json`` that satisfies the ODME prerequisite guard.
This step is required before calibration unless ``calibration.require_supply_audit``
is set to ``false``, or ``tune-supply`` has already produced a report.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from sim.defaults import NETWORK_NORM_DEFAULTS
from sim.io_project import load_config

logger = logging.getLogger(__name__)

_SPEED_MATCH_TOLERANCE = 0.5  # km/h

_EXPECTED_SPEED_RANGES: Dict[str, tuple] = {
    "motorway": (100, 160),
    "motorway_link": (40, 100),
    "trunk": (60, 110),
    "trunk_link": (30, 90),
    "primary": (50, 110),
    "primary_link": (25, 70),
    "secondary": (40, 90),
    "secondary_link": (20, 60),
    "tertiary": (30, 70),
    "tertiary_link": (15, 50),
    "residential": (10, 50),
    "living_street": (5, 30),
    "service": (5, 30),
    "unclassified": (15, 60),
}

_EXPECTED_CAPACITY_RANGES: Dict[str, tuple] = {
    "motorway": (3000, 15000),
    "motorway_link": (1000, 6000),
    "trunk": (2000, 10000),
    "trunk_link": (800, 5000),
    "primary": (1000, 6000),
    "primary_link": (500, 3000),
    "secondary": (400, 3000),
    "secondary_link": (300, 2500),
    "tertiary": (200, 2000),
    "tertiary_link": (150, 1500),
    "residential": (100, 1500),
    "service": (50, 800),
    "living_street": (30, 500),
    "unclassified": (100, 2000),
}


def run_supply_audit(config_path: str | Path = "config/brno/sim.yaml") -> Dict[str, Any]:
    """Run basic supply-side diagnostics and write ``supply_audit.json``."""
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = project_dir / "project_database.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Project database not found: {db_path}. Run build-network first."
        )

    logger.info("=== SUPPLY AUDIT ===")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT link_type, "
            "COUNT(*) as n, "
            "AVG(speed_ab) as avg_speed, "
            "MIN(speed_ab) as min_speed, "
            "MAX(speed_ab) as max_speed, "
            "AVG(capacity_ab) as avg_capacity, "
            "MIN(capacity_ab) as min_capacity, "
            "MAX(capacity_ab) as max_capacity, "
            "AVG(lanes_ab) as avg_lanes "
            "FROM links "
            "WHERE link_type != 'centroid_connector' "
            "GROUP BY link_type "
            "ORDER BY n DESC"
        ).fetchall()
    finally:
        conn.close()

    diagnostics: List[Dict[str, Any]] = []
    warnings_list: List[str] = []
    all_ok = True

    for row in rows:
        lt = row["link_type"]
        n = row["n"]
        avg_spd = row["avg_speed"] or 0
        min_spd = row["min_speed"] or 0
        max_spd = row["max_speed"] or 0
        avg_cap = row["avg_capacity"] or 0
        min_cap = row["min_capacity"] or 0
        max_cap = row["max_capacity"] or 0
        avg_lanes = row["avg_lanes"] or 0

        speed_range = _EXPECTED_SPEED_RANGES.get(lt, (5, 160))
        cap_range = _EXPECTED_CAPACITY_RANGES.get(lt, (50, 15000))

        speed_ok = speed_range[0] <= avg_spd <= speed_range[1]
        cap_ok = cap_range[0] <= avg_cap <= cap_range[1]

        if not speed_ok:
            w = f"{lt}: avg speed {avg_spd:.1f} km/h outside expected [{speed_range[0]}-{speed_range[1]}]"
            warnings_list.append(w)
            all_ok = False
        if not cap_ok:
            w = f"{lt}: avg capacity {avg_cap:.0f} veh/h/dir outside expected [{cap_range[0]}-{cap_range[1]}]"
            warnings_list.append(w)
            all_ok = False

        diag = {
            "link_type": lt,
            "n_links": n,
            "avg_speed_kmh": round(avg_spd, 1),
            "min_speed_kmh": round(min_spd, 1),
            "max_speed_kmh": round(max_spd, 1),
            "speed_ok": speed_ok,
            "avg_capacity_vph": round(avg_cap, 0),
            "min_capacity_vph": round(min_cap, 0),
            "max_capacity_vph": round(max_cap, 0),
            "capacity_ok": cap_ok,
            "avg_lanes": round(avg_lanes, 1),
        }
        diagnostics.append(diag)

        status = "OK" if (speed_ok and cap_ok) else "WARNING"
        logger.info(
            "  %-20s n=%5d  speed=[%4.0f-%4.0f] avg=%5.1f %s  "
            "cap=[%5.0f-%6.0f] avg=%6.0f %s  lanes=%.1f  [%s]",
            lt, n, min_spd, max_spd, avg_spd,
            "OK" if speed_ok else "!!",
            min_cap, max_cap, avg_cap,
            "OK" if cap_ok else "!!",
            avg_lanes, status,
        )

    # BPR parameters from config
    assign_cfg = cfg.get("assignment") or {}
    bpr_cfg = assign_cfg.get("bpr") or {}
    bpr_by_type = bpr_cfg.get("by_link_type", {})
    bpr_audit: Dict[str, Any] = {
        "vdf_function": bpr_cfg.get("vdf_function", "BPR"),
        "alpha_default": bpr_cfg.get("alpha_default", 0.55),
        "beta_default": bpr_cfg.get("beta_default", 4.0),
        "per_link": bpr_cfg.get("per_link", True),
        "n_type_specific": len(bpr_by_type),
    }

    # Estimated vs empirical coverage
    _IMPUTATION_WARN_THRESHOLD = 70  # percent
    estimated_coverage = _compute_estimated_coverage(db_path)
    if estimated_coverage["pct_speed_estimated"] > _IMPUTATION_WARN_THRESHOLD:
        w = (
            f"Speed: {estimated_coverage['pct_speed_estimated']:.0f}% of links use "
            f"default/imputed values — supply is not empirically calibrated"
        )
        warnings_list.append(w)
        all_ok = False
    if estimated_coverage["pct_capacity_estimated"] > _IMPUTATION_WARN_THRESHOLD:
        w = (
            f"Capacity: {estimated_coverage['pct_capacity_estimated']:.0f}% of links use "
            f"default/imputed values — supply is not empirically calibrated"
        )
        warnings_list.append(w)
        all_ok = False
    logger.info(
        "  Estimated coverage: speed=%.0f%% imputed, capacity=%.0f%% imputed",
        estimated_coverage["pct_speed_estimated"],
        estimated_coverage["pct_capacity_estimated"],
    )

    if warnings_list:
        logger.warning("  Supply audit warnings:")
        for w in warnings_list:
            logger.warning("    %s", w)

    verdict = "PASS" if all_ok else "PASS_WITH_WARNINGS"
    logger.info("  Supply audit verdict: %s", verdict)

    # V/C ratio diagnostics (if assignment results exist)
    vc_diagnostics = _compute_vc_diagnostics(output_dir, db_path, assign_cfg)

    report = {
        "verdict": verdict,
        "all_ok": all_ok,
        "warnings": warnings_list,
        "diagnostics": diagnostics,
        "estimated_coverage": estimated_coverage,
        "bpr": bpr_audit,
        "vc_diagnostics": vc_diagnostics,
    }

    report_path = output_dir / "supply_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("  Supply audit report: %s", report_path)
    return report


def _compute_vc_diagnostics(
    output_dir: Path,
    db_path: Path,
    assign_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute V/C ratio distribution from assignment results if available."""
    results_path = output_dir / "assignment_results.parquet"
    if not results_path.exists():
        return {"available": False, "reason": "no_assignment_results"}

    try:
        import pandas as pd
        vol_df = pd.read_parquet(str(results_path))
    except Exception:
        return {"available": False, "reason": "cannot_read_results"}

    vol_col = None
    for candidate in ("PCE_tot", "tot_flow_ab", "matrix_ab", "PCE_tot_ab", "volume_ab",
                      "wd_daily_local_tot", "Preload_tot"):
        if candidate in vol_df.columns:
            vol_col = candidate
            break
    if vol_col is None:
        return {"available": False, "reason": "no_volume_column"}

    conn = sqlite3.connect(str(db_path))
    try:
        cap_rows = conn.execute(
            "SELECT link_id, link_type, capacity_ab FROM links "
            "WHERE link_type != 'centroid_connector'"
        ).fetchall()
    finally:
        conn.close()

    cap_df = pd.DataFrame(cap_rows, columns=["link_id", "link_type", "capacity_ab"])
    cap_df["link_id"] = cap_df["link_id"].astype(int)
    vol_df["link_id"] = vol_df["link_id"].astype(int) if "link_id" in vol_df.columns else vol_df.index

    merged = vol_df.merge(cap_df, on="link_id", how="inner")
    merged["volume"] = pd.to_numeric(merged[vol_col], errors="coerce").fillna(0)
    merged["capacity"] = pd.to_numeric(merged["capacity_ab"], errors="coerce").fillna(1)

    bpr_cfg = assign_cfg.get("bpr") or {}
    dcf_raw = bpr_cfg.get("daily_capacity_factor", {})
    dcf_default = float(dcf_raw) if isinstance(dcf_raw, (int, float)) else float(
        dcf_raw.get("default", 9.0) if isinstance(dcf_raw, dict) else 9.0
    )
    dcf_by_lt = dcf_raw.get("by_link_type", {}) if isinstance(dcf_raw, dict) else {}

    def _get_dcf(lt: str) -> float:
        return float(dcf_by_lt.get(lt, dcf_default))

    merged["daily_capacity"] = merged.apply(
        lambda r: r["capacity"] * _get_dcf(str(r.get("link_type", ""))), axis=1,
    )
    merged["vc_ratio"] = merged["volume"] / merged["daily_capacity"].clip(lower=1)

    total = len(merged)
    if total == 0:
        return {"available": False, "reason": "no_merged_links"}

    vc_over_1 = int((merged["vc_ratio"] > 1.0).sum())
    vc_over_08 = int((merged["vc_ratio"] > 0.8).sum())
    vc_over_05 = int((merged["vc_ratio"] > 0.5).sum())

    by_type = {}
    for lt, grp in merged.groupby("link_type"):
        by_type[str(lt)] = {
            "n_links": len(grp),
            "mean_vc": round(float(grp["vc_ratio"].mean()), 3),
            "max_vc": round(float(grp["vc_ratio"].max()), 3),
            "pct_over_1": round(float((grp["vc_ratio"] > 1.0).sum()) / max(len(grp), 1) * 100, 1),
        }

    result = {
        "available": True,
        "total_links": total,
        "pct_vc_over_1": round(vc_over_1 / total * 100, 1),
        "pct_vc_over_08": round(vc_over_08 / total * 100, 1),
        "pct_vc_over_05": round(vc_over_05 / total * 100, 1),
        "by_link_type": by_type,
    }

    if vc_over_1 / max(total, 1) > 0.05:
        logger.warning(
            "  V/C diagnostics: %.1f%% of links have V/C > 1.0 — "
            "capacity may be under-estimated or demand over-estimated",
            vc_over_1 / total * 100,
        )
    else:
        logger.info(
            "  V/C diagnostics: %.1f%% over capacity (OK, <5%% threshold)",
            vc_over_1 / total * 100,
        )

    return result


def _compute_estimated_coverage(db_path: Path) -> Dict[str, Any]:
    """Count links whose speed/capacity match normalization defaults (imputed)."""
    default_speeds = (
        NETWORK_NORM_DEFAULTS.get("normalization", {})
        .get("defaults", {})
        .get("speed_by_link_type", {})
    )
    default_cap_per_lane = (
        NETWORK_NORM_DEFAULTS.get("normalization", {})
        .get("defaults", {})
        .get("capacity_per_lane_by_link_type", {})
    )
    default_lanes = (
        NETWORK_NORM_DEFAULTS.get("normalization", {})
        .get("defaults", {})
        .get("lanes_by_link_type", {})
    )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT link_type, speed_ab, capacity_ab, lanes_ab "
            "FROM links WHERE link_type != 'centroid_connector'"
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    speed_imputed = 0
    cap_imputed = 0

    for row in rows:
        lt = str(row["link_type"] or "")
        spd = float(row["speed_ab"] or 0)
        cap = float(row["capacity_ab"] or 0)
        lanes = float(row["lanes_ab"] or 1)

        ref_spd = default_speeds.get(lt)
        if ref_spd is not None and abs(spd - ref_spd) <= _SPEED_MATCH_TOLERANCE:
            speed_imputed += 1

        ref_cap_per_lane = default_cap_per_lane.get(lt)
        ref_lanes = default_lanes.get(lt, 1)
        if ref_cap_per_lane is not None:
            ref_cap = ref_cap_per_lane * ref_lanes
            if abs(cap - ref_cap) <= 1.0 or (cap > 0 and abs(cap - ref_cap) / cap < 0.01):
                cap_imputed += 1

    pct_spd = (speed_imputed / max(total, 1)) * 100
    pct_cap = (cap_imputed / max(total, 1)) * 100

    return {
        "total_links": total,
        "speed_imputed": speed_imputed,
        "pct_speed_estimated": round(pct_spd, 1),
        "capacity_imputed": cap_imputed,
        "pct_capacity_estimated": round(pct_cap, 1),
    }
