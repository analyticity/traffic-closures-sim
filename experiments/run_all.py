#!/usr/bin/env python3
"""Run all experiment scripts for a given city config.

Usage:
    python experiments/run_all.py --config config/most/sim.yaml
    python experiments/run_all.py --config config/brno/sim.yaml --experiments exp01 exp03
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from _common import DEFAULT_CONFIG, EXPERIMENTS_OUTPUT, city_from_config

logger = logging.getLogger(__name__)

ALL_EXPERIMENTS = [
    "exp01_baseline_plausibility",
    "exp02_closure_response",
    "exp03_congestion_patterns",
    "exp04_portability",
    "exp05_line_vs_point",
    "exp08_sensitivity",
    "exp11_waze_speeds",
]


def _run_one(module_name: str, config_path: str) -> Dict[str, Any]:
    """Import and run a single experiment, return status record."""
    t0 = time.time()
    record: Dict[str, Any] = {"experiment": module_name, "config": config_path}

    saved_argv = sys.argv[:]
    try:
        sys.argv = [module_name, "--config", config_path]
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        mod.main()
        record["status"] = "ok"
    except FileNotFoundError as exc:
        record["status"] = "skipped"
        record["reason"] = str(exc)
        logger.warning("SKIP %s: %s", module_name, exc)
    except Exception as exc:
        record["status"] = "error"
        record["reason"] = str(exc)
        record["traceback"] = traceback.format_exc()
        logger.error("FAIL %s: %s", module_name, exc)
    finally:
        sys.argv = saved_argv
        record["elapsed_s"] = round(time.time() - t0, 1)

    # Detect experiments that handled missing data internally (wrote status=skipped).
    if record["status"] == "ok":
        city = city_from_config(config_path)
        summary_path = EXPERIMENTS_OUTPUT / city / module_name / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as _f:
                    exp_summary = json.load(_f)
                if isinstance(exp_summary, dict) and exp_summary.get("status") == "skipped":
                    record["status"] = "skipped"
                    record["reason"] = exp_summary.get("reason", "internal skip")
                    logger.warning("SKIP %s (internal): %s", module_name, record["reason"])
            except (json.JSONDecodeError, OSError):
                pass

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment suite for a city")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="City sim.yaml config path (default: %(default)s)",
    )
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Subset of experiments to run (e.g. exp01 exp03). Default: all.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    city = city_from_config(args.config)

    if args.experiments:
        names = []
        for short in args.experiments:
            matches = [e for e in ALL_EXPERIMENTS if e.startswith(short)]
            if matches:
                names.extend(matches)
            else:
                logger.warning("Unknown experiment prefix: %s", short)
        experiments = names or ALL_EXPERIMENTS
    else:
        experiments = ALL_EXPERIMENTS

    logger.info("=== Experiment suite for %s (%d experiments) ===", city, len(experiments))

    results: List[Dict[str, Any]] = []
    t_total = time.time()

    for exp_name in experiments:
        logger.info("--- %s ---", exp_name)
        rec = _run_one(exp_name, args.config)
        results.append(rec)
        status_icon = {"ok": "OK", "skipped": "SKIP", "error": "FAIL"}.get(rec["status"], "?")
        logger.info("  %s  (%.1fs)", status_icon, rec["elapsed_s"])

    elapsed = round(time.time() - t_total, 1)

    summary = {
        "city": city,
        "config": args.config,
        "total_elapsed_s": elapsed,
        "counts": {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "error": sum(1 for r in results if r["status"] == "error"),
        },
        "experiments": results,
    }

    out_dir = EXPERIMENTS_OUTPUT / city
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    logger.info("=== Done: %d ok, %d skipped, %d error (%.1fs) ===",
                summary["counts"]["ok"], summary["counts"]["skipped"],
                summary["counts"]["error"], elapsed)
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
