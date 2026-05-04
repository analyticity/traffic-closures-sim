#!/usr/bin/env python3
"""
Simulation pipeline runner.

Workflow:
  0) clean              – delete generated data, start fresh
  1) check              – verify AequilibraE project bootstrap
  2) build-network      – import OSM network into AequilibraE
  3) fetch-data         – download & preprocess external datasets (incl. closures)
  4) normalize-network  – clean / normalise link attributes (+ apply closures if enabled)
  5) build-zones        – create TAZ zones + centroid connectors (auto-remaps population)
  6) build-supernetwork – national coarse net → gateway lookups & through pairs
  7) build-demand        – seed OD matrix (SLDB, gateways, synthetic segments)
  8) assign-warm-skims  – optional short assign; always saves skims.aem for distribute
  9) distribute         – gravity + IPF (see skim vs Euclidean below)
 10) assign              – full traffic assignment (user equilibrium; AoN blocked by default)
 11) audit-supply        – basic supply-side diagnostics (speeds, capacities, VDF); prerequisite for ODME
 12) calibrate           – Spiess gradient ODME (default); set calibration.method for others
 13) calibrate-odme      – explicit alias for Spiess ODME
 14) tune-supply         – optional outer loop on supply parameters
 15) validate            – match diagnostics + independent validation (CSD)
 16) learn-profile       – day-type factors from CSD
 17) strip-closures      – remove baseline closures → clean network for scenarios
 18) serve               – REST API (read-only results)

  Skim-driven distribute: after build-demand, run assign-warm-skims (or assign with
  calibration.save_skims=true) so distribute can use network times; then distribute, then assign.
  Distribution requires skims by default (demand.distribution.impedance=skim).

  Prerequisite validation: each step declares required input files (project DB,
  OD matrix, assignment results, etc.) in _STEP_PREREQUISITES.  The runner
  checks existence before execution and warns about stale outputs via
  _STALENESS_CHECKS.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
import sys
import argparse

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sim.calibration.telemetry import setup_logging
from sim.io_project import load_config, resolve_project_database_path
from sim.network import build_network_from_osm
from sim.network import normalize_and_export_network, strip_closures
from sim.zoning import build_zones_and_connectors
from sim.datasets import resolved_csd2025_validation_parquet_path, run_fetch_datasets
from sim.demand import assert_build_demand_prerequisites, load_or_build_od_matrix
from sim.assignment import run_assignment, run_warm_skim_assignment
from sim.calibration import (
    run_calibration,
    run_odme_calibration,
    run_entropy_odme,
    run_multistage_calibration,
    run_validation_only,
    run_match_diagnostics,
)
from sim.demand.temporal import run_learn_profile
from sim.supernetwork import run_build_supernetwork

logger = logging.getLogger(__name__)

STEPS = [
    "init-city",
    "clean",
    "check",
    "build-network",
    "normalize-network",
    "build-zones",
    "fetch-data",
    "build-supernetwork",
    "build-demand",
    "assign-warm-skims",
    "distribute",
    "assign",
    "audit-supply",
    "calibrate",
    "calibrate-odme",  # explicit alias; calibrate also defaults to ODME
    "tune-supply",
    "validate",
    "sensitivity",
    "learn-profile",
    "strip-closures",
    "serve",
]


def _require_abs_paths(cfg: dict, step: str, paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{step}: missing required inputs: {missing}. "
            f"Run prerequisite steps first."
        )


def _require_assign_inputs(cfg: dict, step: str) -> None:
    mp = Path(cfg["demand"]["matrix_path"])
    if not mp.exists():
        raise FileNotFoundError(f"{step}: OD matrix not found: {mp}. Run build-demand first.")
    pdb = Path(cfg["project_path"]) / "project_database.sqlite"
    if not pdb.exists():
        raise FileNotFoundError(
            f"{step}: project database not found: {pdb}. Run build-network first."
        )


# ---------------------------------------------------------------------------
#  Step dependency DAG – declarative prerequisite & staleness checks
# ---------------------------------------------------------------------------

def _resolve_cfg_path(cfg: dict, spec: str) -> Path:
    """Resolve a prerequisite path spec against the loaded config.

    Supported specs:
      - ``project_db``  → ``{project_path}/project_database.sqlite``
      - ``matrix``      → ``demand.matrix_path``
      - ``results``     → ``{demand.output_dir}/assignment_results.parquet``
      - ``skims``       → ``{demand.output_dir}/skims.aem``
      - ``zones``       → ``{zoning.output_dir}/zones.geojson``
      - ``zone_map``    → ``{zoning.output_dir}/zone_centroid_mapping.json``
    """
    if spec == "project_db":
        return Path(cfg["project_path"]) / "project_database.sqlite"
    if spec == "matrix":
        return Path(cfg["demand"]["matrix_path"])
    demand_out = Path(
        cfg.get("demand", {}).get("output_dir", "outputs/baseline/demand")
    )
    if spec == "results":
        return demand_out / "assignment_results.parquet"
    if spec == "skims":
        return demand_out / "skims.aem"
    zoning_out = Path(
        cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones")
    )
    if spec == "zones":
        return zoning_out / "zones.geojson"
    if spec == "zone_map":
        return zoning_out / "zone_centroid_mapping.json"
    raise ValueError(f"Unknown prerequisite spec: {spec!r}")


# (spec, human hint for the error message)
_STEP_PREREQUISITES: dict[str, list[tuple[str, str]]] = {
    "normalize-network": [
        ("project_db", "Run build-network first"),
    ],
    "build-zones": [
        ("project_db", "Run build-network first"),
    ],
    "assign-warm-skims": [
        ("project_db", "Run build-network first"),
        ("matrix", "Run build-demand first"),
    ],
    "distribute": [
        ("project_db", "Run build-network first"),
        ("matrix", "Run build-demand first"),
    ],
    "assign": [
        ("project_db", "Run build-network first"),
        ("matrix", "Run build-demand first"),
    ],
    "audit-supply": [
        ("project_db", "Run build-network first"),
    ],
    "calibrate": [
        ("project_db", "Run build-network first"),
        ("matrix", "Run build-demand first"),
    ],
    "calibrate-odme": [
        ("project_db", "Run build-network first"),
        ("matrix", "Run build-demand first"),
    ],
    "tune-supply": [
        ("project_db", "Run build-network first"),
        ("matrix", "Run build-demand first"),
    ],
    "validate": [
        ("matrix", "Run build-demand first"),
        ("results", "Run assign or calibrate first"),
    ],
    "strip-closures": [
        ("project_db", "Run build-network first"),
    ],
}

# Pairs of (newer_spec, older_spec, warning_message) for staleness detection.
# Warns when the *older* file has a newer mtime than the *newer* file,
# suggesting a prerequisite was re-run without updating downstream outputs.
_STALENESS_CHECKS: dict[str, list[tuple[str, str, str]]] = {
    "distribute": [
        (
            "skims",
            "project_db",
            "skims.aem is older than project database — network may have "
            "changed since last skim generation. Consider re-running "
            "assign-warm-skims.",
        ),
    ],
    "validate": [
        (
            "results",
            "matrix",
            "assignment_results.parquet is older than the OD matrix — "
            "results may be stale. Consider re-running assign or calibrate.",
        ),
    ],
    "calibrate": [
        (
            "results",
            "project_db",
            "assignment_results.parquet is older than project database — "
            "consider re-running assign before calibrate.",
        ),
    ],
    "calibrate-odme": [
        (
            "results",
            "project_db",
            "assignment_results.parquet is older than project database — "
            "consider re-running assign before calibrate.",
        ),
    ],
}


def _check_step_prerequisites(cfg: dict, step: str) -> None:
    """Validate file-existence prerequisites and warn about stale inputs."""
    prereqs = _STEP_PREREQUISITES.get(step)
    if prereqs:
        missing: list[str] = []
        for spec, hint in prereqs:
            p = _resolve_cfg_path(cfg, spec)
            if not p.exists():
                missing.append(f"{p} ({hint})")
        if missing:
            raise FileNotFoundError(
                f"{step}: missing required inputs:\n  "
                + "\n  ".join(missing)
            )

    staleness = _STALENESS_CHECKS.get(step)
    if staleness:
        for newer_spec, older_spec, msg in staleness:
            try:
                newer = _resolve_cfg_path(cfg, newer_spec)
                older = _resolve_cfg_path(cfg, older_spec)
            except (KeyError, ValueError):
                continue
            if newer.exists() and older.exists():
                if newer.stat().st_mtime < older.stat().st_mtime:
                    logger.warning("Staleness warning for %s: %s", step, msg)


def run_clean(config_path: str, *, force: bool = False) -> None:
    """Delete city-specific generated data so the next run starts fresh.

    Shared national data in ``data/sources/`` is preserved unless
    ``force=True``, which also removes downloaded source files.
    """
    cfg = load_config(config_path)
    city_slug = cfg["_meta"].get("city_slug", "default")
    project_root = Path(cfg["_meta"]["project_root"])

    dirs_to_remove = [
        Path(cfg["project_path"]),
        project_root / "outputs" / city_slug,
        project_root / "data" / city_slug,
    ]
    if force:
        dirs_to_remove.append(project_root / "data" / "sources")

    for d in dirs_to_remove:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed: {d}")
        else:
            print(f"  Already clean: {d}")

    if force:
        print(f"Full clean done for '{city_slug}' (shared sources removed too).")
    else:
        print(f"Clean done for '{city_slug}'. Shared data in data/sources/ preserved.")
        print("  Use --force to also remove shared downloaded sources.")
    print("Re-run pipeline from build-network.")


def run_check(config_path: str) -> None:
    """Lightweight sanity check of configuration and project bootstrap paths."""
    cfg = load_config(config_path)
    project_root = Path(cfg["_meta"]["project_root"])
    project_path = Path(cfg["project_path"])
    print("=== CHECK ===")
    print(f"  config: {cfg['_meta']['config_path']}")
    print(f"  project_root: {project_root}")
    print(f"  project_path: {project_path}")
    if not project_root.exists():
        raise FileNotFoundError(f"project root missing: {project_root}")
    # Project path may not exist before build-network, but parent must be writable.
    parent = project_path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    test_file = parent / ".check_write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink(missing_ok=True)
    print("  check: OK")


def main() -> None:
    setup_logging()

    ap = argparse.ArgumentParser(
        description="Simulation pipeline – run individual steps or the full chain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config",
                     default=os.environ.get("SIM_CONFIG", "config/brno/sim.yaml"),
                     help="path to sim.yaml (default: $SIM_CONFIG or config/brno/sim.yaml)")
    ap.add_argument("--force", action="store_true",
                     help="for 'clean': also remove shared downloaded sources in data/sources/")
    ap.add_argument("step", choices=STEPS, help="pipeline step to execute")
    args = ap.parse_args()

    cfg = args.config
    step = args.step

    # --- Preflight: unified prerequisite & staleness checks ---
    _no_preflight = {"init-city", "clean", "check", "build-network", "fetch-data", "serve"}
    if step not in _no_preflight:
        _pre_cfg = load_config(cfg)
        _check_step_prerequisites(_pre_cfg, step)

    if step == "init-city":
        from scripts.generate_city_config import main as gen_main
        gen_main()
        return
    elif step == "clean":
        run_clean(cfg, force=args.force)
    elif step == "check":
        run_check(cfg)
    elif step == "build-network":
        build_network_from_osm(cfg)
        print("NETWORK BUILD DONE")
    elif step == "normalize-network":
        normalize_and_export_network(cfg)
    elif step == "build-zones":
        build_zones_and_connectors(cfg)
    elif step == "fetch-data":
        run_fetch_datasets(config_path=cfg)
    elif step == "build-demand":
        _cfg = load_config(cfg)
        assert_build_demand_prerequisites(_cfg)
        load_or_build_od_matrix(cfg, cfg=_cfg)
    elif step == "assign-warm-skims":
        _cfg = load_config(cfg)
        run_warm_skim_assignment(cfg, cfg=_cfg)
    elif step == "distribute":
        from sim.distribution import run_distribution
        run_distribution(cfg)
    elif step == "build-supernetwork":
        _sn_cfg = load_config(cfg)
        _zoning_out = Path(_sn_cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))
        _require_abs_paths(_sn_cfg, step, [
            _zoning_out / "model_area.geojson",
            _zoning_out / "zones.geojson",
        ])
        run_build_supernetwork(cfg)
    elif step == "assign":
        _cfg = load_config(cfg)
        run_assignment(cfg, cfg=_cfg)
    elif step == "audit-supply":
        from sim.calibration.supply_audit import run_supply_audit
        run_supply_audit(cfg)
    elif step in ("calibrate", "calibrate-odme"):
        method = (_pre_cfg.get("calibration") or {}).get("method", "odme")
        if step == "calibrate-odme" or method == "odme":
            run_odme_calibration(cfg)
        elif method == "entropy_odme":
            run_entropy_odme(cfg)
        elif method == "multistage":
            run_multistage_calibration(cfg)
        else:
            run_calibration(cfg)
    elif step == "tune-supply":
        from sim.calibration import run_supply_tuning
        run_supply_tuning(cfg)
    elif step == "validate":
        run_match_diagnostics(cfg)
        run_validation_only(cfg)
    elif step == "sensitivity":
        from sim.sensitivity import run_sensitivity
        run_sensitivity(cfg)
    elif step == "learn-profile":
        _c = load_config(cfg)
        _require_abs_paths(_c, step, [resolved_csd2025_validation_parquet_path(_c)])
        run_learn_profile(cfg)
    elif step == "strip-closures":
        strip_closures(cfg)
    elif step == "serve":
        from sim.api import start_server
        start_server(cfg)


if __name__ == "__main__":
    main()
