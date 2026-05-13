#!/usr/bin/env python3
"""Export CSV: assignment volumes + link supply + implied BPR for preset parallel corridors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from sim.diagnostics.parallel_corridor import export_parallel_corridor_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="config/brno/sim.yaml",
        help="City sim.yaml (project_path, demand.output_dir, assignment.bpr).",
    )
    ap.add_argument(
        "--presets",
        default="config/brno/parallel_corridor_presets.yaml",
        help="YAML with 'corridors' list (id, link_ids, validation_screenline, description).",
    )
    ap.add_argument(
        "--assignment-parquet",
        default=None,
        help="Override path to assignment_results.parquet (default: <output_dir>/assignment_results.parquet).",
    )
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: <output_dir>/parallel_corridor_diagnostics.csv).",
    )
    ap.add_argument(
        "--corridor",
        default=None,
        help="If set, export only this corridor preset id.",
    )
    args = ap.parse_args()
    cfg_path = (_REPO / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    presets_path = (_REPO / args.presets).resolve() if not Path(args.presets).is_absolute() else Path(args.presets)
    pq = Path(args.assignment_parquet).resolve() if args.assignment_parquet else None
    out = Path(args.output_csv).resolve() if args.output_csv else None
    out_path = export_parallel_corridor_csv(
        cfg_path,
        presets_path=presets_path,
        assignment_parquet=pq,
        output_csv=out,
        corridor_filter=args.corridor,
    )
    print(out_path)


if __name__ == "__main__":
    main()
