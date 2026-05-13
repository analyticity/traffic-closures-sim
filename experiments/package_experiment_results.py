#!/usr/bin/env python3
"""Zip all experiment outputs for a city, including every expected experiment folder.

The thesis archive previously missed ``exp07_accident_correlation`` when the
packaging step used a glob that did not match the folder name, or when an
experiment had not been run yet.  This script walks **known** experiment names
(from ``run_all.ALL_EXPERIMENTS``) so each directory is included if it exists,
and always adds ``experiment_report.json`` when present.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from _common import DEFAULT_CONFIG, EXPERIMENTS_OUTPUT, city_from_config  # noqa: E402
from run_all import ALL_EXPERIMENTS  # noqa: E402


def _add_tree(zf: zipfile.ZipFile, base: Path, arc_prefix: str) -> int:
    n = 0
    if not base.exists():
        return 0
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base)
        arc = f"{arc_prefix}/{rel.as_posix()}"
        if p.is_file():
            zf.write(p, arcname=arc)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip experiment outputs for thesis export")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="City sim.yaml")
    parser.add_argument(
        "-o", "--output",
        default="",
        help="Output .zip path (default: outputs/experiments/<city>_experiments.zip)",
    )
    args = parser.parse_args()

    city = city_from_config(args.config)
    root = EXPERIMENTS_OUTPUT / city
    out_zip = Path(args.output) if args.output else (_REPO / "outputs" / "experiments" / f"{city}_experiments.zip")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    files_written = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        report = root / "experiment_report.json"
        if report.is_file():
            zf.write(report, arcname=f"{city}/experiment_report.json")
            files_written += 1

        for exp in ALL_EXPERIMENTS:
            exp_dir = root / exp
            files_written += _add_tree(zf, exp_dir, f"{city}/{exp}")

    print(f"Wrote {out_zip} ({files_written} files under {city}/)")


if __name__ == "__main__":
    main()
