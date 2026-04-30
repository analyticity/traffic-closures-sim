"""Gravity calibration and IPF (iterative proportional fitting)."""
from __future__ import annotations

from typing import Dict

import numpy as np


def calibrate_gravity_simple(
    seed: np.ndarray,
    impedance: np.ndarray,
    function: str = "EXPO",
) -> Dict[str, float]:
    """Calibrate a simple deterrence function from seed OD + impedance.

    Uses AequilibraE's GravityCalibration when available, otherwise
    falls back to a least-squares fit of the deterrence parameter.
    """
    try:
        from aequilibrae.distribution import GravityCalibration
        from aequilibrae.matrix import AequilibraeMatrix as AEM

        n = seed.shape[0]
        seed_mat = AEM()
        seed_mat.create_empty(zones=n, matrix_names=["seed"], memory_only=True)
        seed_mat.index[:] = np.arange(1, n + 1, dtype=np.int32)
        seed_mat.matrix["seed"][:, :] = seed
        seed_mat.computational_view(["seed"])

        imp_mat = AEM()
        imp_mat.create_empty(zones=n, matrix_names=["cost"], memory_only=True)
        imp_mat.index[:] = np.arange(1, n + 1, dtype=np.int32)
        imp_mat.matrix["cost"][:, :] = impedance
        imp_mat.computational_view(["cost"])

        gc = GravityCalibration(matrix=seed_mat, impedance=imp_mat, function=function)
        gc.execute()
        params = {"function": function}
        if hasattr(gc, "model") and gc.model is not None:
            for attr in ("alpha", "beta", "gamma"):
                if hasattr(gc.model, attr):
                    params[attr] = float(getattr(gc.model, attr))
        seed_mat.close()
        imp_mat.close()
        return params
    except Exception:
        pass

    flat_t = seed.ravel()
    flat_c = impedance.ravel()
    mask = (flat_t > 0) & (flat_c > 0) & np.isfinite(flat_t) & np.isfinite(flat_c)
    if mask.sum() < 10:
        return {"function": "EXPO", "beta": 0.0001}

    log_t = np.log(flat_t[mask])
    c = flat_c[mask]
    beta = -float(np.polyfit(c, log_t, 1)[0])
    beta = max(beta, 1e-6)
    return {"function": "EXPO", "beta": round(beta, 6)}


def run_ipf(
    seed: np.ndarray,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    max_iter: int = 200,
    tolerance: float = 0.001,
) -> np.ndarray:
    """IPF to adjust OD row/column totals to production/attraction targets.

    Uses AequilibraE's Ipf when available, otherwise a pure-numpy fallback.
    """
    try:
        from aequilibrae.distribution import Ipf
        from aequilibrae.matrix import AequilibraeMatrix as AEM

        n = seed.shape[0]
        seed_mat = AEM()
        seed_mat.create_empty(zones=n, matrix_names=["seed"], memory_only=True)
        seed_mat.index[:] = np.arange(1, n + 1, dtype=np.int32)
        seed_mat.matrix["seed"][:, :] = seed
        seed_mat.computational_view(["seed"])

        ipf = Ipf(matrix=seed_mat, rows=target_rows, columns=target_cols)
        ipf.max_iterations = max_iter
        ipf.tolerance = tolerance
        ipf.execute()

        result = ipf.output.matrix_view[:, :].copy()
        seed_mat.close()
        ipf.output.close()
        return result
    except Exception:
        pass

    # Numpy fallback (Furness method)
    mat = seed.copy().astype(np.float64)
    mat = np.maximum(mat, 1e-12)
    for _ in range(max_iter):
        row_sums = mat.sum(axis=1)
        row_factors = np.where(row_sums > 0, target_rows / row_sums, 1.0)
        mat *= row_factors[:, None]

        col_sums = mat.sum(axis=0)
        col_factors = np.where(col_sums > 0, target_cols / col_sums, 1.0)
        mat *= col_factors[None, :]

        row_err = np.max(np.abs(mat.sum(axis=1) - target_rows))
        col_err = np.max(np.abs(mat.sum(axis=0) - target_cols))
        if max(row_err, col_err) < tolerance:
            break

    return mat
