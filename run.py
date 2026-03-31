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
  6) build-supernetwork – build simplified national gateway supernetwork
  7) build-demand       – build OD matrix from SLDB commuting data
  8) distribute         – gravity calibration + IPF on seed OD
  8) assign             – traffic assignment (shortest path / equilibrium)
  9) calibrate          – iterative: assign → compare → scale → repeat
 10) tune-supply        – outer-loop supply parameter optimization
 11) validate           – independent validation on CSD2020
 12) learn-profile      – learn day-type factors from CSD2020
 13) serve              – start REST API server (read-only results)
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
from sim.demand import load_or_build_od_matrix
from sim.assignment import run_assignment
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
    "distribute",
    "assign",
    "calibrate",
    "tune-supply",
    "validate",
    "learn-profile",
    "serve",
]


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
    elif step == "build-network":
        build_network_from_osm(cfg)
    elif step == "normalize-network":
        normalize_and_export_network(cfg)
    elif step == "build-zones":
        build_zones_and_connectors(cfg)
    elif step == "fetch-data":
        run_fetch_datasets(config_path=cfg)
    elif step == "build-demand":
        load_or_build_od_matrix(cfg)
    elif step == "distribute":
        from sim.distribution import run_distribution
        run_distribution(cfg)
    elif step == "build-supernetwork":
        run_build_supernetwork(cfg)
    elif step == "assign":
        run_assignment(cfg)
    elif step == "calibrate":
        run_calibration(cfg)
    elif step == "tune-supply":
        from sim.calibration import run_supply_tuning
        run_supply_tuning(cfg)
    elif step == "validate":
        run_validation_only(cfg)
    elif step == "learn-profile":
        run_learn_profile(cfg)
    elif step == "serve":
        from sim.api import start_server
        start_server(cfg)


if __name__ == "__main__":
    main()
