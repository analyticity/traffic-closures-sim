"""CLI entry-points: run_assignment, run_warm_skim_assignment."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import load_config
from sim.assignment.config import (
    _apply_bpr_defaults,
    _resolve_multi_class,
    multiclass_matrix_core_status,
)
from sim.assignment.preflight import _check_connectors, fix_node_ids
from sim.assignment.executor import _detect_volume_col, execute_assignment

logger = logging.getLogger(__name__)


def _aem_index_len(matrix_path: Path) -> int | None:
    """Return zone count from an ``.aem`` index column, or ``None`` on failure."""
    try:
        mat = AequilibraeMatrix()
        mat.load(str(matrix_path))
        try:
            return int(len(mat.index[:]))
        finally:
            mat.close()
    except Exception:
        logger.debug("Failed to read matrix index from %s", matrix_path, exc_info=True)
        return None


def _aem_first_core_leading_dim(matrix_path: Path) -> int | None:
    """Return the first dimension of the first core (must match OD index for skims)."""
    try:
        mat = AequilibraeMatrix()
        mat.load(str(matrix_path))
        try:
            core_names = list(mat.names) if hasattr(mat, "names") and mat.names else []
            if not core_names:
                return None
            data = mat.matrix[core_names[0]][:, :]
            return int(data.shape[0])
        finally:
            mat.close()
    except Exception:
        logger.debug("Failed to read skim leading dim from %s", matrix_path, exc_info=True)
        return None


def warm_skims_regeneration_reason(cfg: dict, skim_path: Path) -> str | None:
    """Return why warm skims should be recomputed, or ``None`` if the file looks reusable.

    Mirrors ``distribute`` skim validation (leading matrix dimension vs OD index)
    plus simple mtime checks against the OD matrix and project database.

    ``SIM_FORCE_SKIMS`` is handled by the caller; this function ignores it.
    """
    demand_cfg = cfg.get("demand") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    project_db = Path(cfg["project_path"]) / "project_database.sqlite"

    if not skim_path.exists():
        return "skims.aem is missing"

    od_n = _aem_index_len(matrix_path)
    if od_n is None:
        return f"cannot read OD matrix index ({matrix_path}) to validate skims"

    skim_n = _aem_first_core_leading_dim(skim_path)
    if skim_n is None:
        return "skims.aem is unreadable or has no matrix cores"
    if skim_n != od_n:
        return f"skim matrix size ({skim_n}) does not match OD matrix zones ({od_n})"

    try:
        skim_mtime = skim_path.stat().st_mtime
    except OSError as exc:
        return f"cannot stat skims.aem ({exc})"

    if matrix_path.exists() and skim_mtime < matrix_path.stat().st_mtime:
        return "skims.aem is older than the OD matrix"

    if project_db.exists() and skim_mtime < project_db.stat().st_mtime:
        return "skims.aem is older than the project database"

    return None


def _run_assignment_pass(
    config_path: str | Path,
    *,
    algorithm: str,
    max_iter: int,
    rgap: float,
    save_skims: bool,
    banner: str,
    log_volume_note: str = "",
    skim_method: str = "blended",
    cfg: dict | None = None,
    allow_aon: bool = False,
    strict_convergence: bool = False,
    convergence_filename: str = "assignment_convergence.json",
) -> None:
    """Shared body for ``assign`` and ``assign-warm-skims``."""
    if cfg is None:
        cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    assign_cfg = cfg.get("assignment") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)
    core_name = str(calib_cfg.get("core_name", "wd_daily"))
    cores = int(assign_cfg.get("cores", 0))

    logger.info(banner)

    logger.info("\n1) Pre-flight checks ...")
    renamed = fix_node_ids(project_dir)
    if renamed:
        logger.info(f"   Fixed {renamed} overflow node IDs")
    conn_info = _check_connectors(project_dir)
    logger.info(f"   Connectors: {conn_info['connectors']}, "
          f"centroids: {conn_info['centroids']}, "
          f"connected: {conn_info['centroids_connected']}")
    if not conn_info["ok"]:
        logger.warning("   WARNING: not all centroids reach the road network!")

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}")

    logger.info(f"\n2) Loading matrix: {matrix_path}")
    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat_core_names = [str(x) for x in (list(mat.names) if hasattr(mat, "names") else [])]
    bpr_params = _apply_bpr_defaults(assign_cfg.get("bpr") or {})
    mc_cfg = assign_cfg.get("multi_class") or {}
    multi_classes = _resolve_multi_class(mc_cfg)
    mc_ok, mc_req, mc_miss = multiclass_matrix_core_status(mat_core_names, multi_classes)
    logger.info(
        "   Matrix stores %d OD core name(s); multi-class prerequisites ok=%s (required %s)",
        len(mat_core_names),
        mc_ok,
        mc_req or "—",
    )
    if multi_classes and not mc_ok:
        logger.warning("   Missing multi-class cores on matrix file: %s", mc_miss)

    mat.computational_view([core_name])
    total_demand = float(mat.matrix_view.sum())
    logger.info(f"   Core '{core_name}': {mat.zones} zones, demand={total_demand:,.0f}")

    gc_cfg = assign_cfg.get("generalized_cost") or {}
    gc_enabled = bool(gc_cfg.get("enabled", True))
    gc_field = str(gc_cfg.get("fixed_cost_field", "distance")) if gc_enabled else None
    gc_mult = float(gc_cfg.get("fixed_cost_multiplier", 0.006)) if gc_enabled else 0.0
    gc_vot = float(gc_cfg.get("vot", 1.0))

    if gc_enabled and gc_mult > 0:
        penalty_per_km = gc_mult * 1000.0 / max(gc_vot, 1e-9)
        logger.info(
            "  Generalized cost: field=%s, multiplier=%.4f, vot=%.2f "
            "=> effective %.1f sec/km distance penalty",
            gc_field, gc_mult, gc_vot, penalty_per_km,
        )

    logger.info(f"\n3) Running {algorithm.upper()} ...")
    project = Project()
    project.open(str(project_dir))
    try:
        df, skims, _sl, convergence_meta = execute_assignment(
            project,
            mat,
            algorithm=algorithm,
            max_iter=max_iter,
            rgap_target=rgap,
            save_skims=save_skims,
            fixed_cost_field=gc_field,
            fixed_cost_multiplier=gc_mult,
            vot=gc_vot,
            bpr_parameters=bpr_params,
            multi_class=multi_classes,
            cores=cores,
            skim_method=skim_method,
            allow_aon=allow_aon,
            strict_convergence=strict_convergence,
            assignment_cfg=assign_cfg,
        )
    finally:
        project.close()
        mat.close()

    tot_col = _detect_volume_col(df)
    total_vol = float(df[tot_col].sum()) if tot_col else 0.0
    note = log_volume_note or "Results"
    logger.info(f"\n4) {note}: {len(df)} links, vol_col={tot_col}, total={total_vol:,.0f}")

    if total_vol <= 0:
        import sqlite3 as _sq
        logger.warning("\n   *** ZERO VOLUME — diagnostics ***")
        logger.info(f"   Matrix demand: {total_demand:,.0f}")
        logger.info(f"   Connectors OK: {conn_info['ok']}  "
              f"(connected={conn_info['centroids_connected']}/{conn_info['centroids']})")
        db = str(project_dir / "project_database.sqlite")
        cn = _sq.connect(db)
        modes = cn.execute(
            "SELECT modes, COUNT(*) FROM links WHERE link_type='centroid_connector' GROUP BY modes"
        ).fetchall()
        logger.info(f"   Connector modes: {modes}")
        bad_tt = cn.execute(
            "SELECT SUM(CASE WHEN travel_time_ab<=0 OR travel_time_ab IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM links"
        ).fetchone()
        logger.info(f"   Links with bad travel_time: {bad_tt[0]}/{bad_tt[1]}")
        cn.close()
        logger.info("   Likely causes: connectors not reaching road network, or centroid IDs mismatch.")

    out_path = output_dir / "assignment_results.parquet"
    df.to_parquet(str(out_path), index=False)
    logger.info(f"   Saved: {out_path}")

    # Write convergence metadata for downstream guards
    conv_path = output_dir / convergence_filename
    conv_path.write_text(
        json.dumps(convergence_meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info(f"   Convergence metadata: {conv_path}")

    if skims is not None:
        skim_path = output_dir / "skims.aem"
        try:
            skims.export(str(skim_path))
            logger.info(f"   Skims saved: {skim_path}")
            # Write skim metadata sidecar for distribution convergence check
            skims_meta = {
                "algorithm": convergence_meta.get("algorithm"),
                "final_rgap": convergence_meta.get("final_rgap"),
                "converged": convergence_meta.get("converged"),
                "n_iterations": convergence_meta.get("n_iterations"),
                "skim_method": skim_method,
            }
            meta_path = output_dir / "skims_meta.json"
            meta_path.write_text(
                json.dumps(skims_meta, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        except Exception as exc:
            warnings.warn(
                f"Skim export failed ({skim_path}): {exc}. "
                "Trip distribution will require skims from a prior run.",
                RuntimeWarning,
                stacklevel=2,
            )


def run_assignment(config_path: str | Path = "config/brno/sim.yaml", cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config(config_path)
    calib_cfg = cfg.get("calibration") or {}
    algorithm = str(calib_cfg.get("algorithm", "bfw"))
    max_iter = int(calib_cfg.get("max_iter", 100))
    rgap = float(calib_cfg.get("rgap_target", 0.001))
    save_skims = bool(calib_cfg.get("save_skims", False))
    allow_aon = bool(calib_cfg.get("allow_aon", False))
    strict = bool(calib_cfg.get("strict_convergence", True))
    _run_assignment_pass(
        config_path,
        algorithm=algorithm,
        max_iter=max_iter,
        rgap=rgap,
        save_skims=save_skims,
        banner="=== TRAFFIC ASSIGNMENT ===",
        cfg=cfg,
        allow_aon=allow_aon,
        strict_convergence=strict,
        convergence_filename="pre_odme_convergence.json",
    )


def run_warm_skim_assignment(config_path: str | Path = "config/brno/sim.yaml", cfg: dict | None = None) -> None:
    """Shorter assignment pass with skims always saved, for impedance before ``distribute``."""
    if cfg is None:
        cfg = load_config(config_path)
    warm = (cfg.get("assignment") or {}).get("warm_skim_pass") or {}
    calib_cfg = cfg.get("calibration") or {}
    algorithm = str(warm.get("algorithm") or calib_cfg.get("algorithm", "bfw"))
    max_iter = int(warm.get("max_iter", 30))
    rgap = float(warm.get("rgap_target", 0.01))
    with warnings.catch_warnings():
        warnings.filterwarnings("default", message="Assignment did NOT converge", category=RuntimeWarning)
        warnings.simplefilter("always", RuntimeWarning)

        old_showwarning = warnings.showwarning

        def _warn_to_print(message, category, filename, lineno, file=None, line=None):
            if "Assignment did NOT converge" in str(message):
                logger.info(f"  [info] Warm skim: {message} (acceptable for preliminary pass)")
            else:
                # Must call the *saved* handler: ``warnings.showwarning`` is
                # rebound to this function below, so calling it here recurses.
                old_showwarning(message, category, filename, lineno, file, line)

        warnings.showwarning = _warn_to_print
        try:
            _run_assignment_pass(
                config_path,
                algorithm=algorithm,
                max_iter=max_iter,
                rgap=rgap,
                save_skims=True,
                banner="=== WARM SKIM ASSIGNMENT (for trip distribution) ===",
                log_volume_note="Warm-pass results (intermediate; re-run assign after distribute if needed)",
                skim_method="final",
                cfg=cfg,
                convergence_filename="warm_assignment_convergence.json",
            )
        finally:
            warnings.showwarning = old_showwarning
