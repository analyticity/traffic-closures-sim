"""Distribution pipeline orchestrator: gravity calibration + IPF.

The pipeline distributes **selected demand segments** (by default only
``other``) rather than the combined ``wd_daily`` core.  Segments not
listed in ``distribution.segments`` are left untouched; after
distribution the ``wd_daily`` core is reconstructed as the sum of all
segment cores.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from aequilibrae.matrix import AequilibraeMatrix

from sim.defaults import SIM_DEFAULTS
from sim.io_project import get_nested, load_config, load_zone_population, load_zones
from sim.distribution.pa_vectors import build_pa_vectors
from sim.distribution.impedance import (
    _euclidean_impedance,
    _load_impedance,
    _load_zone_employment,
    _validate_skim_convergence,
)
from sim.distribution.gravity import calibrate_gravity_simple, run_ipf

logger = logging.getLogger(__name__)

_ALL_DEMAND_SEGMENTS = ("commuting", "other", "external_local", "external_through")


def _segment_core_name(segment: str) -> str:
    return f"wd_daily_{segment}"


def run_distribution(config_path: str | Path = "config/brno/sim.yaml", cfg: dict | None = None) -> None:
    """Run the distribution step: gravity calibration + IPF on selected segments."""
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

    segments_to_distribute: List[str] = list(
        dist_cfg.get("segments", SIM_DEFAULTS["demand"]["distribution"]["segments"])
    )

    logger.info("Trip distribution: gravity and IPF")
    logger.info("Segments to distribute: %s", segments_to_distribute)
    logger.info(
        "Segments preserved (no distribution): %s",
        [s for s in _ALL_DEMAND_SEGMENTS if s not in segments_to_distribute],
    )

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}. Run build-demand first.")

    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    available_cores = set(mat.names)

    for seg in segments_to_distribute:
        cn = _segment_core_name(seg)
        if cn not in available_cores:
            raise ValueError(
                f"Segment core '{cn}' not found in matrix. "
                f"Available cores: {sorted(available_cores)}. "
                f"Check demand.distribution.segments config."
            )

    zone_index = mat.index[:].copy()
    n = len(zone_index)

    zones_gdf = load_zones(cfg, normalize_columns=False)
    zone_ids = np.array(sorted(zones_gdf["zone_id"].astype(int).unique()), dtype=np.int64)
    population = load_zone_population(
        Path(get_nested(cfg, ["datasets", "cache_dir"], "data/cache"))
    )

    # --- impedance --------------------------------------------------------
    imp_mode = str(dist_cfg.get("impedance", "skim")).lower().strip()
    allow_euclidean = bool(dist_cfg.get("allow_euclidean_fallback", False))

    if imp_mode == "auto":
        logger.warning(
            "impedance='auto' is deprecated and will be removed. "
            "Treating as 'skim'. Set impedance='euclidean' with "
            "allow_euclidean_fallback=true for explicit Euclidean mode."
        )
        imp_mode = "skim"

    if imp_mode not in ("skim", "euclidean"):
        raise ValueError(
            f"demand.distribution.impedance must be 'skim' or 'euclidean', got {imp_mode!r}"
        )

    skim_path = output_dir / "skims.aem"

    if imp_mode == "euclidean":
        if not allow_euclidean:
            raise ValueError(
                "demand.distribution.impedance='euclidean' requires "
                "demand.distribution.allow_euclidean_fallback=true. "
                "Euclidean impedance destroys spatial friction and produces "
                "unreliable trip distribution. Use skims from a converged "
                "assignment instead (run assign-warm-skims first)."
            )
        logger.warning(
            "*** EUCLIDEAN IMPEDANCE ACTIVE *** "
            "This is acceptable only for prototyping. "
            "Production models MUST use network skims."
        )
        impedance = _euclidean_impedance(zones_gdf, zone_ids)
        if impedance.shape[0] != n:
            logger.warning("Impedance shape mismatch (%d vs %d), using uniform", impedance.shape[0], n)
            _fallback = float(SIM_DEFAULTS["demand"]["distribution"]["uniform_impedance_fallback"])
            impedance = np.ones((n, n), dtype=np.float64) * _fallback
        imp_source = "euclidean"
    else:
        if not skim_path.exists():
            raise FileNotFoundError(
                f"No skim matrix at {skim_path}. "
                f"Run assign-warm-skims (or assign with calibration.save_skims=true) "
                f"before distribute. Distribution requires network-based impedance."
            )
        impedance = _load_impedance(output_dir, zone_index)
        if impedance is None:
            raise RuntimeError(
                f"Could not load a valid skim matrix from {skim_path} "
                f"(check zone count matches matrix, n={n}). "
                f"Re-run assign-warm-skims to regenerate skims."
            )
        allow_unconverged = bool(dist_cfg.get("allow_unconverged_skims", False))
        _validate_skim_convergence(output_dir, allow_unconverged=allow_unconverged)
        logger.info("Impedance: loaded from skims.aem")
        imp_source = "skim"

    # --- P/A vectors ------------------------------------------------------
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
            require_emp = bool(dist_cfg.get("require_employment", False))
            if require_emp:
                raise RuntimeError(
                    "No employment data found (zone_employment.parquet) but "
                    "demand.distribution.require_employment=true. "
                    "P/A attractions without employment are purely population-"
                    "based and structurally incorrect."
                )
            logger.warning(
                "No employment data found — attractions will be symmetric "
                "copies of production (population-only). This weakens spatial "
                "differentiation. Provide zone_employment.parquet or set "
                "demand.distribution.require_employment=true to enforce."
            )

    pa = build_pa_vectors(zone_ids, population, pa_trip_rate, pa_car_share, pa_occupancy,
                          employment=employment)
    logger.info("P/A: total_production=%s, total_attraction=%s",
                f"{float(pa['production'].sum()):,.0f}",
                f"{float(pa['attraction'].sum()):,.0f}")

    target_rows = pa["production"].values
    target_cols = pa["attraction"].values

    deterrence = str(dist_cfg.get("deterrence_function", "EXPO"))
    ipf_max_iter = int(dist_cfg.get("ipf_max_iter", 200))
    ipf_tol = float(dist_cfg.get("ipf_tolerance", 0.001))
    alpha = float(dist_cfg.get("blend_alpha", 0.7))
    min_beta = float(dist_cfg.get("min_gravity_beta", 0.0001))

    # --- distribute each selected segment ---------------------------------
    segment_reports: Dict[str, Any] = {}

    for seg in segments_to_distribute:
        cn = _segment_core_name(seg)
        mat.computational_view([cn])
        seed = mat.matrix[cn][:, :].copy().astype(np.float64)
        seed_total = float(seed.sum())
        logger.info("--- Segment '%s' (%s): seed total=%s ---", seg, cn, f"{seed_total:,.0f}")

        if seed_total <= 0:
            logger.warning("Segment '%s' has zero/negative seed total, skipping distribution", seg)
            segment_reports[seg] = {"skipped": True, "reason": "zero_seed"}
            continue

        logger.info("Calibrating gravity model for '%s' (%s)", seg, deterrence)
        params = calibrate_gravity_simple(seed, impedance, function=deterrence)
        logger.info("Gravity params for '%s': %s", seg, params)

        beta_val = params.get("beta", 0.0)
        if beta_val < min_beta:
            logger.warning(
                "Segment '%s': gravity beta=%.2e is below min_gravity_beta=%.2e. "
                "Impedance has negligible spatial effect — skipping IPF for this "
                "segment (seed preserved unchanged).",
                seg, beta_val, min_beta,
            )
            segment_reports[seg] = {
                "skipped": True,
                "reason": "beta_below_threshold",
                "gravity_params": params,
                "seed_total": round(seed_total, 0),
            }
            continue

        logger.info("Running IPF for '%s' (max_iter=%d)", seg, ipf_max_iter)
        adjusted = run_ipf(seed, target_rows, target_cols,
                           max_iter=ipf_max_iter, tolerance=ipf_tol)

        blended = alpha * adjusted + (1.0 - alpha) * seed
        logger.info(
            "Segment '%s' blended (alpha=%s): total=%s (seed=%s, ipf=%s)",
            seg, alpha,
            f"{float(blended.sum()):,.0f}",
            f"{seed_total:,.0f}",
            f"{float(adjusted.sum()):,.0f}",
        )

        mat.matrix[cn][:, :] = blended

        segment_reports[seg] = {
            "skipped": False,
            "gravity_params": params,
            "seed_total": round(seed_total, 0),
            "ipf_adjusted_total": round(float(adjusted.sum()), 0),
            "blended_total": round(float(blended.sum()), 0),
        }

    # --- reconstruct wd_daily from all segments ---------------------------
    if "wd_daily" in available_cores:
        combined = np.zeros((n, n), dtype=np.float64)
        for seg in _ALL_DEMAND_SEGMENTS:
            cn = _segment_core_name(seg)
            if cn in available_cores:
                mat.computational_view([cn])
                combined += mat.matrix[cn][:, :].astype(np.float64)
        mat.computational_view(["wd_daily"])
        old_total = float(mat.matrix["wd_daily"][:, :].sum())
        mat.matrix["wd_daily"][:, :] = combined
        new_total = float(combined.sum())
        logger.info(
            "Reconstructed wd_daily from segments: %s -> %s (delta=%s)",
            f"{old_total:,.0f}", f"{new_total:,.0f}",
            f"{new_total - old_total:+,.0f}",
        )

    mat.save()
    mat.close()
    logger.info("Updated matrix: %s", matrix_path)

    report: Dict[str, Any] = {
        "impedance_mode": imp_mode,
        "impedance_source": imp_source,
        "segments_distributed": segments_to_distribute,
        "segments_preserved": [s for s in _ALL_DEMAND_SEGMENTS if s not in segments_to_distribute],
        "pa_config": {
            "trip_rate": pa_trip_rate,
            "car_share": pa_car_share,
            "occupancy": pa_occupancy,
            "employment_source": emp_source,
            "employment_zones": len(employment) if employment else 0,
        },
        "segment_results": segment_reports,
        "ipf": {"max_iter": ipf_max_iter, "tolerance": ipf_tol},
        "blend_alpha": alpha,
        "min_gravity_beta": min_beta,
    }
    report_path = output_dir / "distribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report: %s", report_path)
