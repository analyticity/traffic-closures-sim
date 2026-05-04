"""Parameter sensitivity runner.

Sweeps one or more config parameters, runs a quick assignment for each
value, and collects key metrics into ``sensitivity_report.json``.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from sim.io_project import load_config, get_nested

logger = logging.getLogger(__name__)


def _set_nested(cfg: dict, dotpath: str, value: Any) -> None:
    """Set a value deep inside a nested dict using a dotted path."""
    keys = dotpath.split(".")
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _get_nested_dot(cfg: dict, dotpath: str) -> Any:
    """Get a value from a nested dict using a dotted path."""
    return get_nested(cfg, dotpath.split("."))


def run_sensitivity(config_path: Union[str, Path] = "config/brno/sim.yaml") -> Dict[str, Any]:
    """Run parameter sensitivity sweeps and write a report."""
    cfg = load_config(config_path)
    sens_cfg = cfg.get("sensitivity") or {}

    if not sens_cfg.get("enabled", False):
        logger.info("Sensitivity step disabled in config, skipping")
        return {"skipped": True}

    parameters: List[Dict[str, Any]] = sens_cfg.get("parameters", [])
    if not parameters:
        logger.info("No sensitivity parameters configured, skipping")
        return {"skipped": True, "reason": "no_parameters"}

    max_iter = int(sens_cfg.get("max_iter", 30))
    rgap = float(sens_cfg.get("rgap_target", 0.05))

    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from sim.assignment import run_assignment

    results: List[Dict[str, Any]] = []

    for param_spec in parameters:
        dotpath = param_spec.get("path", "")
        values = param_spec.get("values", [])
        if not dotpath or not values:
            continue

        baseline_value = _get_nested_dot(cfg, dotpath)
        logger.info(
            "=== Sensitivity sweep: %s (baseline=%s, values=%s) ===",
            dotpath, baseline_value, values,
        )

        for val in values:
            patched_cfg = copy.deepcopy(cfg)
            _set_nested(patched_cfg, dotpath, val)

            patched_cfg.setdefault("calibration", {})
            patched_cfg["calibration"]["max_iter"] = max_iter
            patched_cfg["calibration"]["rgap_target"] = rgap
            patched_cfg["calibration"]["save_skims"] = False

            logger.info("  Running assignment with %s=%s ...", dotpath, val)
            try:
                run_assignment(config_path, cfg=patched_cfg)

                result_path = output_dir / "assignment_results.parquet"
                total_assigned = None
                if result_path.exists():
                    import pandas as pd
                    res_df = pd.read_parquet(result_path)
                    vol_cols = [c for c in res_df.columns if c.startswith("volume")]
                    if vol_cols:
                        total_assigned = float(res_df[vol_cols[0]].sum())

                entry = {
                    "parameter": dotpath,
                    "value": val,
                    "total_assigned": round(total_assigned, 0) if total_assigned is not None else None,
                    "status": "ok",
                }
            except Exception as exc:
                logger.warning("  Assignment failed for %s=%s: %s", dotpath, val, exc)
                entry = {
                    "parameter": dotpath,
                    "value": val,
                    "status": "error",
                    "error": str(exc),
                }

            results.append(entry)
            logger.info("  Result: %s", entry)

    report = {
        "parameters_swept": [p.get("path") for p in parameters],
        "max_iter": max_iter,
        "rgap_target": rgap,
        "results": results,
    }

    report_path = output_dir / "sensitivity_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Sensitivity report: %s", report_path)
    return report
