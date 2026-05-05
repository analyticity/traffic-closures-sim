"""Pre-ODME seed matrix diagnostics.

Checks seed quality before running calibration and logs warnings
for potential issues that would prevent convergence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import load_config

logger = logging.getLogger(__name__)


def run_seed_diagnostics(
    config_path: str | Path = "config/brno/sim.yaml",
) -> Dict[str, Any]:
    """Analyze OD seed matrix quality and report potential ODME issues."""
    cfg = load_config(config_path) if isinstance(config_path, (str, Path)) else config_path
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))

    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    if not matrix_path.exists():
        logger.warning("Seed diagnostics: demand matrix not found at %s", matrix_path)
        return {"error": "matrix_not_found"}

    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    if core_name not in mat.names:
        core_name = mat.names[0]
    mat.computational_view([core_name])

    data = mat.matrix[core_name][:, :].copy().astype(np.float64)
    zone_ids = mat.index[:].copy()
    n_zones = len(zone_ids)
    mat.close()

    total = float(data.sum())
    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)

    zero_production_zones = int((row_sums == 0).sum())
    zero_attraction_zones = int((col_sums == 0).sum())
    n_nonzero_cells = int((data > 0).sum())
    sparsity = 1.0 - n_nonzero_cells / max(n_zones * n_zones, 1)

    max_cell = float(data.max())
    mean_nonzero = float(data[data > 0].mean()) if n_nonzero_cells > 0 else 0.0
    cv = float(data[data > 0].std() / mean_nonzero) if mean_nonzero > 0 else 0.0

    row_balance = float(row_sums.sum())
    col_balance = float(col_sums.sum())
    balance_diff_pct = abs(row_balance - col_balance) / max(row_balance, 1) * 100

    report: Dict[str, Any] = {
        "n_zones": n_zones,
        "total_demand": round(total, 0),
        "n_nonzero_cells": n_nonzero_cells,
        "sparsity_pct": round(sparsity * 100, 1),
        "zero_production_zones": zero_production_zones,
        "zero_attraction_zones": zero_attraction_zones,
        "max_cell_value": round(max_cell, 1),
        "mean_nonzero_cell": round(mean_nonzero, 2),
        "coefficient_of_variation": round(cv, 2),
        "production_attraction_balance_diff_pct": round(balance_diff_pct, 2),
    }

    warnings = []
    if zero_production_zones > n_zones * 0.1:
        warnings.append(
            f"{zero_production_zones}/{n_zones} zones have zero production "
            f"({zero_production_zones/n_zones*100:.0f}%) — seed may be too sparse"
        )
    if zero_attraction_zones > n_zones * 0.1:
        warnings.append(
            f"{zero_attraction_zones}/{n_zones} zones have zero attraction "
            f"({zero_attraction_zones/n_zones*100:.0f}%) — seed may be too sparse"
        )
    if sparsity > 0.95:
        warnings.append(
            f"Matrix is {sparsity*100:.0f}% sparse — ODME may struggle to converge"
        )
    if balance_diff_pct > 5.0:
        warnings.append(
            f"Production/attraction imbalance: {balance_diff_pct:.1f}% — "
            "consider scaling before ODME"
        )
    if cv > 10.0:
        warnings.append(
            f"High coefficient of variation ({cv:.1f}) — "
            "a few dominant cells may distort calibration"
        )
    if total < 1000:
        warnings.append(f"Very low total demand ({total:.0f}) — check seed generation")

    report["warnings"] = warnings
    report["quality"] = "good" if not warnings else ("fair" if len(warnings) <= 2 else "poor")

    for w in warnings:
        logger.warning("Seed diagnostics: %s", w)

    logger.info(
        "Seed diagnostics: %d zones, %.0f total demand, %.1f%% sparse, "
        "quality=%s (%d warnings)",
        n_zones, total, sparsity * 100, report["quality"], len(warnings),
    )

    report_path = output_dir / "seed_diagnostics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
