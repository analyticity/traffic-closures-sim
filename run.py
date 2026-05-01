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
 10) assign              – full traffic assignment (AoN / equilibrium)
 11) calibrate           – Spiess gradient ODME (default); set calibration.method for others
 12) calibrate-odme      – explicit alias for Spiess ODME
 13) tune-supply         – optional outer loop on supply parameters
 14) validate            – match diagnostics + independent validation (CSD)
 15) learn-profile       – day-type factors from CSD
 16) strip-closures      – remove baseline closures → clean network for scenarios
 17) serve               – REST API (read-only results)

  Skim-driven distribute: after build-demand, run assign-warm-skims (or assign with
  calibration.save_skims=true) so distribute can use network times; then distribute, then assign.
  If distribute runs first without skims, impedance defaults to Euclidean (demand.distribution.impedance=auto).
"""
from __future__ import annotations

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
    "calibrate",
    "calibrate-odme",  # explicit alias; calibrate also defaults to ODME
    "tune-supply",
    "validate",
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
    ap.add_argument("--config", default="config/brno/sim.yaml", help="path to sim.yaml")
    ap.add_argument("--force", action="store_true",
                     help="for 'clean': also remove shared downloaded sources in data/sources/")
    ap.add_argument("step", choices=STEPS, help="pipeline step to execute")
    args = ap.parse_args()

    cfg = args.config
    step = args.step

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
        _require_assign_inputs(_cfg, step)
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
        _require_assign_inputs(_cfg, step)
        run_assignment(cfg, cfg=_cfg)
    elif step in ("calibrate", "calibrate-odme"):
        _c = load_config(cfg)
        _require_abs_paths(_c, step, [
            Path(_c["demand"]["matrix_path"]),
            resolve_project_database_path(_c),
        ])
        method = (_c.get("calibration") or {}).get("method", "odme")
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
        _c = load_config(cfg)
        _require_abs_paths(_c, step, [
            Path(_c["demand"]["matrix_path"]),
        ])
        run_match_diagnostics(cfg)
        run_validation_only(cfg)
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
