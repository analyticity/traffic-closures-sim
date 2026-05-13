#!/usr/bin/env python3
"""Write a temporary ``sim.yaml`` fork for external-demand sensitivity runs.

Deep-copies ``config/olomouc/sim.yaml`` and overrides
``demand.segments.external_local.total_daily_trips`` and/or
``demand.sldb.external_processing.through_traffic_scale``. The output path is
meant for ``SIM_CONFIG`` / ``run.py --config`` (outputs dir is gitignored).

Example::

    python scripts/write_olomouc_sensitivity_sim_yaml.py \\
        --out outputs/olomouc/sensitivity/sim_through_0p22.yaml \\
        --through-scale 0.22

    SIM_CONFIG=outputs/olomouc/sensitivity/sim_through_0p22.yaml \\
        python run.py build-demand
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("config/olomouc/sim.yaml"),
        help="Source sim.yaml (default: config/olomouc/sim.yaml)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--external-local-trips",
        type=int,
        default=None,
        help="Override demand.segments.external_local.total_daily_trips",
    )
    ap.add_argument(
        "--through-scale",
        type=float,
        default=None,
        help="Override demand.sldb.external_processing.through_traffic_scale",
    )
    args = ap.parse_args()

    data = yaml.safe_load(args.base.read_text(encoding="utf-8")) or {}
    cfg = copy.deepcopy(data)

    if args.external_local_trips is not None:
        cfg.setdefault("demand", {}).setdefault("segments", {}).setdefault(
            "external_local", {}
        )["total_daily_trips"] = int(args.external_local_trips)

    if args.through_scale is not None:
        cfg.setdefault("demand", {}).setdefault("sldb", {}).setdefault(
            "external_processing", {}
        )["through_traffic_scale"] = float(args.through_scale)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
