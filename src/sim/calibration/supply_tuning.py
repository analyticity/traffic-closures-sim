"""Outer-loop coordinate descent tuning of network speed/capacity factors."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import load_config

logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class SupplyParams:
    speed_factors: Dict[str, float] = field(default_factory=lambda: {})
    capacity_factors: Dict[str, float] = field(default_factory=lambda: {})


def _save_base_values(project_dir: Path) -> None:
    """Copy current speed/capacity to _base columns in SQLite (runs once)."""
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(links)").fetchall()]
        if "_base_speed_ab" in cols:
            conn.close()
            return

        for base, src in [
            ("_base_speed_ab", "speed_ab"), ("_base_speed_ba", "speed_ba"),
            ("_base_capacity_ab", "capacity_ab"), ("_base_capacity_ba", "capacity_ba"),
        ]:
            conn.execute(f"ALTER TABLE links ADD COLUMN {base} REAL")
            conn.execute(f"UPDATE links SET {base} = {src}")
        conn.commit()
    finally:
        conn.close()


def apply_supply_params(project_dir: Path, params: SupplyParams) -> int:
    """Apply class-specific speed/capacity factors to the network DB.

    Reads from _base_* columns (preserved originals) and writes
    factored values to speed_ab/speed_ba/capacity_ab/capacity_ba.
    """
    _save_base_values(project_dir)
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    updated = 0
    try:
        links = conn.execute(
            "SELECT link_id, link_type, _base_speed_ab, _base_speed_ba, "
            "_base_capacity_ab, _base_capacity_ba FROM links"
        ).fetchall()

        for lid, lt, bsab, bsba, bcab, bcba in links:
            lt_str = str(lt or "").lower()
            sf = params.speed_factors.get(lt_str, 1.0)
            cf = params.capacity_factors.get(lt_str, 1.0)

            for road_class, factor_s, factor_c in [
                ("motorway", sf, cf), ("trunk", sf, cf), ("primary", sf, cf),
                ("secondary", sf, cf), ("tertiary", sf, cf), ("residential", sf, cf),
            ]:
                if road_class in lt_str:
                    sf = params.speed_factors.get(road_class, sf)
                    cf = params.capacity_factors.get(road_class, cf)
                    break

            new_sab = (bsab or 50) * sf
            new_sba = (bsba or 50) * sf
            new_cab = (bcab or 900) * cf
            new_cba = (bcba or 900) * cf

            tt_ab = 0
            tt_ba = 0
            dist = conn.execute(
                "SELECT distance FROM links WHERE link_id=?", (lid,)
            ).fetchone()
            if dist and dist[0] and new_sab > 0:
                tt_ab = dist[0] * 3.6 / new_sab
            if dist and dist[0] and new_sba > 0:
                tt_ba = dist[0] * 3.6 / new_sba

            conn.execute(
                "UPDATE links SET speed_ab=?, speed_ba=?, capacity_ab=?, capacity_ba=?, "
                "travel_time_ab=?, travel_time_ba=? WHERE link_id=?",
                (new_sab, new_sba, new_cab, new_cba, tt_ab, tt_ba, lid),
            )
            updated += 1

        conn.commit()
    finally:
        conn.close()
    return updated


def compute_objective(
    count_stats: Dict[str, Any],
    screenline_results: Dict[str, Any],
    jt_results: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> float:
    """Weighted composite objective for supply calibration (lower is better).

    Includes R², slope, and %RMSE terms so that daily-model supply tuning
    optimises for regression fit rather than hourly-specific GEH.
    """
    w = weights
    geh_term = (100.0 - float(count_stats.get("geh_lt5_pct", 0))) * w.get("geh", 1.0)

    r2 = float(count_stats.get("r2") or 0.0)
    r2_term = (1.0 - r2) * 100.0 * w.get("r2", 1.0)

    slope = float(count_stats.get("slope") or 1.0)
    slope_term = abs(slope - 1.0) * 100.0 * w.get("slope", 1.0)

    prmse = float(count_stats.get("pct_rmse") or 0.0)
    rmse_term = prmse * w.get("pct_rmse", 0.5)

    sl_term = 0.0
    for sr in screenline_results.values():
        ratio = sr.get("ratio", 1.0)
        if ratio is not None:
            sl_term += abs(ratio - 1.0)
    sl_term *= w.get("screenline", 2.0)

    jt_fail = sum(1 for r in jt_results if not r.get("pass", True))
    jt_term = jt_fail / max(len(jt_results), 1) * 100 * w.get("jt", 1.0) if jt_results else 0.0

    return geh_term + r2_term + slope_term + rmse_term + sl_term + jt_term


def run_supply_tuning(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """Outer-loop supply parameter tuning via coordinate descent."""
    cfg = load_config(config_path)
    calib_cfg = cfg.get("calibration") or {}
    tuning_cfg = calib_cfg.get("supply_tuning") or {}

    if not tuning_cfg.get("enabled", False):
        logger.info("Supply tuning disabled in config.")
        return

    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)

    road_classes = tuning_cfg.get("road_classes", ["motorway", "trunk", "primary", "secondary", "tertiary"])
    speed_range = tuning_cfg.get("speed_factor_range", [0.8, 0.9, 1.0, 1.1, 1.2])
    cap_range = tuning_cfg.get("capacity_factor_range", [0.8, 0.9, 1.0, 1.1, 1.2])
    inner_max_iter = int(tuning_cfg.get("inner_max_iterations", 3))
    obj_weights = {"geh": 1.0, "screenline": 2.0, "jt": 1.0}

    logger.info("=== SUPPLY PARAMETER TUNING ===")
    logger.info("  Road classes: %s", road_classes)
    logger.info("  Speed range: %s", speed_range)
    logger.info("  Capacity range: %s", cap_range)

    _save_base_values(project_dir)

    best_params = SupplyParams(
        speed_factors={rc: 1.0 for rc in road_classes},
        capacity_factors={rc: 1.0 for rc in road_classes},
    )
    best_obj = float("inf")

    orig_max_iter = calib_cfg.get("max_iterations", 30)

    def _evaluate(params: SupplyParams) -> float:
        """Apply params, run short calibration, return objective."""
        from sim.calibration.legacy import run_calibration

        apply_supply_params(project_dir, params)
        calib_cfg["max_iterations"] = inner_max_iter

        try:
            run_calibration(config_path)
        except Exception:
            logger.exception("Calibration failed during supply tuning evaluation")
            return float("inf")

        report_path = output_dir / "calibration_report.json"
        if not report_path.exists():
            return float("inf")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        count_stats = report.get("final", {})
        sl_results = report.get("screenlines", {})

        jt_cfg = calib_cfg.get("journey_time_validation", {})
        jt_results: List[Dict[str, Any]] = []
        if jt_cfg.get("reference_routes"):
            skim_path = output_dir / "skims.aem"
            if skim_path.exists():
                try:
                    from sim.calibration import validate_journey_times

                    mat = AequilibraeMatrix()
                    mat.load(str(skim_path))
                    names = list(mat.names)
                    if names:
                        zone_ids = mat.index[:].copy()
                        skim_data: np.ndarray = mat.matrix[names[0]][:, :].copy()
                        mat.close()
                        jt_results = validate_journey_times(
                            skim_data, zone_ids, jt_cfg["reference_routes"],
                        )
                except Exception:
                    logger.debug(
                        "Journey time validation from skim matrix failed",
                        exc_info=True,
                    )

        return compute_objective(count_stats, sl_results, jt_results, obj_weights)

    best_obj = _evaluate(best_params)
    logger.info("\n  Baseline objective: %.2f", best_obj)

    for rc in road_classes:
        logger.info("\n  --- Tuning %s ---", rc)

        # Speed factor
        best_sf = best_params.speed_factors.get(rc, 1.0)
        for sf in speed_range:
            if sf == best_sf:
                continue
            trial = SupplyParams(
                speed_factors={**best_params.speed_factors, rc: sf},
                capacity_factors=dict(best_params.capacity_factors),
            )
            obj = _evaluate(trial)
            logger.info("    speed_factor[%s]=%s ... obj=%.2f", rc, sf, obj)
            if obj < best_obj:
                best_obj = obj
                best_params = trial
                best_sf = sf
        best_params.speed_factors[rc] = best_sf

        # Capacity factor
        best_cf = best_params.capacity_factors.get(rc, 1.0)
        for cf in cap_range:
            if cf == best_cf:
                continue
            trial = SupplyParams(
                speed_factors=dict(best_params.speed_factors),
                capacity_factors={**best_params.capacity_factors, rc: cf},
            )
            obj = _evaluate(trial)
            logger.info("    capacity_factor[%s]=%s ... obj=%.2f", rc, cf, obj)
            if obj < best_obj:
                best_obj = obj
                best_params = trial
                best_cf = cf
        best_params.capacity_factors[rc] = best_cf

    # Apply best params and run final full calibration
    calib_cfg["max_iterations"] = orig_max_iter
    logger.info(
        "\n  Best params: speed=%s, capacity=%s, obj=%.2f",
        best_params.speed_factors,
        best_params.capacity_factors,
        best_obj,
    )
    apply_supply_params(project_dir, best_params)

    tuning_report = {
        "best_objective": round(best_obj, 3),
        "speed_factors": best_params.speed_factors,
        "capacity_factors": best_params.capacity_factors,
        "config": {
            "road_classes": road_classes,
            "speed_range": speed_range,
            "capacity_range": cap_range,
            "inner_max_iterations": inner_max_iter,
            "objective_weights": obj_weights,
        },
    }
    report_path = output_dir / "supply_tuning_report.json"
    report_path.write_text(json.dumps(tuning_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("\n  Supply tuning report: %s", report_path)
    logger.info("  Running final calibration with best parameters ...")
    from sim.calibration.legacy import run_calibration

    run_calibration(config_path)
