"""Distribution pipeline orchestrator: gravity calibration + IPF."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from aequilibrae.matrix import AequilibraeMatrix

from sim.defaults import SIM_DEFAULTS
from sim.io_project import get_nested, load_config, load_zone_population, load_zones
from sim.distribution.pa_vectors import build_pa_vectors
from sim.distribution.impedance import _euclidean_impedance, _load_impedance, _load_zone_employment
from sim.distribution.gravity import calibrate_gravity_simple, run_ipf

logger = logging.getLogger(__name__)


def run_distribution(config_path: str | Path = "config/brno/sim.yaml", cfg: dict | None = None) -> None:
    """Run the distribution step: gravity calibration + IPF on the seed OD."""
    if cfg is None:
        cfg = load_config(config_path)
    demand_cfg = cfg.get("demand") or {}
    dist_cfg = get_nested(cfg, ["demand", "distribution"], {})

    if not dist_cfg.get("enabled", True):
        logger.info("Distribution step disabled in config, skipping")
        return

    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)
    core_name = str(cfg.get("calibration", {}).get("core_name", "wd_daily"))

    logger.info("Trip distribution: gravity and IPF")

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}. Run build-demand first.")

    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])
    seed = mat.matrix[core_name][:, :].copy().astype(np.float64)
    zone_index = mat.index[:].copy()
    n = len(zone_index)
    logger.info("Seed: %d zones, total=%s", n, f"{float(seed.sum()):,.0f}")

    zones_gdf = load_zones(cfg, normalize_columns=False)
    zone_ids = np.array(sorted(zones_gdf["zone_id"].astype(int).unique()), dtype=np.int64)
    population = load_zone_population(
        Path(get_nested(cfg, ["datasets", "cache_dir"], "data/cache"))
    )

    imp_mode = str(dist_cfg.get("impedance", "auto")).lower().strip()
    if imp_mode not in ("auto", "skim"):
        raise ValueError(
            f"demand.distribution.impedance must be 'auto' or 'skim', got {imp_mode!r}"
        )
    skim_path = output_dir / "skims.aem"

    if imp_mode == "skim":
        if not skim_path.exists():
            raise FileNotFoundError(
                f"No skim matrix at {skim_path}. Run assign (with calibration.save_skims=true) "
                "or assign-warm-skims, then distribute again."
            )
        impedance = _load_impedance(output_dir, zone_index)
        if impedance is None:
            raise RuntimeError(
                f"demand.distribution.impedance=skim but could not load a valid matrix from {skim_path} "
                f"(check zone count matches matrix, n={n})."
            )
        logger.info("Impedance: loaded from skims.aem (required mode)")
        imp_source = "skim"
    else:
        impedance = _load_impedance(output_dir, zone_index)
        if impedance is not None:
            logger.info("Impedance: loaded from skims.aem")
            imp_source = "skim"
        else:
            logger.info("Impedance: Euclidean distance (no skims available)")
            impedance = _euclidean_impedance(zones_gdf, zone_ids)
            if impedance.shape[0] != n:
                logger.warning("Impedance shape mismatch (%d vs %d), using uniform", impedance.shape[0], n)
                _fallback = float(SIM_DEFAULTS["demand"]["distribution"]["uniform_impedance_fallback"])
                impedance = np.ones((n, n), dtype=np.float64) * _fallback
            imp_source = "euclidean"

    deterrence = str(dist_cfg.get("deterrence_function", "EXPO"))
    logger.info("Calibrating gravity model (%s)", deterrence)
    params = calibrate_gravity_simple(seed, impedance, function=deterrence)
    logger.info("Gravity params: %s", params)

    pa_trip_rate = float(dist_cfg.get("pa_trip_rate", 2.5))
    pa_car_share = float(dist_cfg.get("pa_car_share", 0.50))
    pa_occupancy = float(dist_cfg.get("pa_occupancy", 1.3))

    employment: Optional[Dict[int, int]] = None
    emp_source = str(dist_cfg.get("employment_source", "auto")).lower().strip()
    if emp_source == "auto":
        cache_dir = Path(get_nested(cfg, ["datasets", "cache_dir"], "data/cache"))
        employment = _load_zone_employment(cache_dir)
        if employment:
            logger.info("Loaded employment data for %d zones (asymmetric P/A)", len(employment))
        else:
            logger.info("No employment data found, using symmetric population-based P/A")

    pa = build_pa_vectors(zone_ids, population, pa_trip_rate, pa_car_share, pa_occupancy,
                          employment=employment)
    logger.info("P/A: total_production=%s, total_attraction=%s",
                f"{float(pa['production'].sum()):,.0f}",
                f"{float(pa['attraction'].sum()):,.0f}")

    target_rows = pa["production"].values
    target_cols = pa["attraction"].values

    ipf_max_iter = int(dist_cfg.get("ipf_max_iter", 200))
    ipf_tol = float(dist_cfg.get("ipf_tolerance", 0.001))
    logger.info("Running IPF (max_iter=%d)", ipf_max_iter)
    adjusted = run_ipf(seed, target_rows, target_cols,
                       max_iter=ipf_max_iter, tolerance=ipf_tol)

    alpha = float(dist_cfg.get("blend_alpha", 0.7))
    blended = alpha * adjusted + (1.0 - alpha) * seed
    logger.info(
        "Blended (alpha=%s): total=%s (seed=%s, ipf=%s)",
        alpha,
        f"{float(blended.sum()):,.0f}",
        f"{float(seed.sum()):,.0f}",
        f"{float(adjusted.sum()):,.0f}",
    )

    mat.matrix[core_name][:, :] = blended
    mat.save()
    mat.close()
    logger.info("Updated matrix: %s", matrix_path)

    report = {
        "impedance_mode": imp_mode,
        "impedance_source": imp_source,
        "gravity_params": params,
        "pa_config": {
            "trip_rate": pa_trip_rate,
            "car_share": pa_car_share,
            "occupancy": pa_occupancy,
            "employment_source": emp_source,
            "employment_zones": len(employment) if employment else 0,
        },
        "totals": {
            "seed": round(float(seed.sum()), 0),
            "ipf_adjusted": round(float(adjusted.sum()), 0),
            "blended": round(float(blended.sum()), 0),
            "target_productions": round(float(target_rows.sum()), 0),
            "target_attractions": round(float(target_cols.sum()), 0),
        },
        "ipf": {"max_iter": ipf_max_iter, "tolerance": ipf_tol},
        "blend_alpha": alpha,
    }
    report_path = output_dir / "distribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report: %s", report_path)
