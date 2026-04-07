#!/usr/bin/env python3
"""
Simulation pipeline runner.

Workflow:
  0) clean              – delete generated data, start fresh
  1) check              – verify AequilibraE project bootstrap
  2) build-network      – import OSM network into AequilibraE
  3) normalize-network  – clean / normalise link attributes
  4) build-zones        – create TAZ zones + centroid connectors
  5) fetch-data         – download & preprocess external datasets
  6) build-supernetwork – national coarse net → gateway lookups & through pairs
  7) build-demand        – seed OD matrix (SLDB, gateways, synthetic segments)
  8) assign-warm-skims  – optional short assign; always saves skims.aem for distribute
  9) distribute         – gravity + IPF (see skim vs Euclidean below)
 10) assign              – full traffic assignment (AoN / equilibrium)
 11) calibrate           – iterative: assign → compare counts → scale OD
 12) tune-supply         – optional outer loop on supply parameters
 13) validate            – independent validation (CSD)
 14) learn-profile       – day-type factors from CSD
 15) serve               – REST API (read-only results)

  Skim-driven distribute: after build-demand, run assign-warm-skims (or assign with
  calibration.save_skims=true) so distribute can use network times; then distribute, then assign.
  If distribute runs first without skims, impedance defaults to Euclidean (demand.distribution.impedance=auto).
"""
from __future__ import annotations

import shutil
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sim.io_project import load_config
from sim.network_pipeline import build_network_from_osm
from sim.network_normalization import normalize_and_export_network
from sim.zoning import build_zones_and_connectors
from sim.fetch_datasets import run_fetch_datasets
from sim.demand import assert_build_demand_prerequisites, load_or_build_od_matrix
from sim.assignment import run_assignment, run_warm_skim_assignment
from sim.calibration import run_calibration, run_validation_only
from sim.temporal import run_learn_profile
from sim.supernetwork import run_build_supernetwork

STEPS = [
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
    "tune-supply",
    "validate",
    "learn-profile",
    "serve",
]


def _require_paths(cfg: dict, step: str, rel_paths: list[str]) -> None:
    root = Path(cfg["_meta"]["project_root"])
    missing = [rp for rp in rel_paths if not (root / rp).exists()]
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


def run_clean(config_path: str) -> None:
    """Delete all generated data so the next run starts fresh.
    Use --keep-sources to preserve downloaded files (SLDB, pentlogram, CSD2020)."""
    cfg = load_config(config_path)
    project_root = Path(cfg["_meta"]["project_root"])

    dirs_to_remove = [
        Path(cfg["project_path"]),
        project_root / "outputs",
        project_root / "data" / "demand",
        project_root / "data" / "cache",
        project_root / "data" / "zones",
        project_root / "data" / "sources",
    ]

    for d in dirs_to_remove:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed: {d}")
        else:
            print(f"  Already clean: {d}")

    print("Clean done.  Re-run pipeline from build-network.")


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
    ap = argparse.ArgumentParser(
        description="Simulation pipeline – run individual steps or the full chain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="config/sim.yaml", help="path to sim.yaml")
    ap.add_argument("step", choices=STEPS, help="pipeline step to execute")
    args = ap.parse_args()

    cfg = args.config
    step = args.step

    if step == "clean":
        run_clean(cfg)
    elif step == "check":
        run_check(cfg)
    elif step == "build-network":
        build_network_from_osm(cfg)
    elif step == "normalize-network":
        normalize_and_export_network(cfg)
    elif step == "build-zones":
        build_zones_and_connectors(cfg)
    elif step == "fetch-data":
        run_fetch_datasets(config_path=cfg)
    elif step == "build-demand":
        assert_build_demand_prerequisites(load_config(cfg))
        load_or_build_od_matrix(cfg)
    elif step == "assign-warm-skims":
        _require_assign_inputs(load_config(cfg), step)
        run_warm_skim_assignment(cfg)
    elif step == "distribute":
        from sim.distribution import run_distribution
        run_distribution(cfg)
    elif step == "build-supernetwork":
        _require_paths(cfg=load_config(cfg), step=step, rel_paths=[
            "outputs/baseline/zones/model_area.geojson",
            "outputs/baseline/zones/zones.geojson",
        ])
        run_build_supernetwork(cfg)
    elif step == "assign":
        _require_assign_inputs(load_config(cfg), step)
        run_assignment(cfg)
    elif step == "calibrate":
        _require_paths(cfg=load_config(cfg), step=step, rel_paths=[
            "data/demand/od_matrix.aem",
            "project/brno_aeq/project_database.sqlite",
            "data/sources/brno/intensity/intenzita_dopravy_pentlogram_2024.geojson",
        ])
        run_calibration(cfg)
    elif step == "tune-supply":
        from sim.calibration import run_supply_tuning
        run_supply_tuning(cfg)
    elif step == "validate":
        _require_paths(cfg=load_config(cfg), step=step, rel_paths=[
            "data/demand/od_matrix.aem",
            "data/cache/v2_csd2025.parquet",
        ])
        run_validation_only(cfg)
    elif step == "learn-profile":
        _require_paths(cfg=load_config(cfg), step=step, rel_paths=[
            "data/cache/v2_csd2025.parquet",
        ])
        run_learn_profile(cfg)
    elif step == "serve":
        from sim.api import start_server
        start_server(cfg)


if __name__ == "__main__":
    main()
