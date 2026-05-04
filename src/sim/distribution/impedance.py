"""Impedance loading: skim matrices and Euclidean fallback."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import pairwise_euclidean

logger = logging.getLogger(__name__)

_SKIM_CONVERGENCE_WARN_THRESHOLD = 0.01


def load_skim_metadata(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load skim sidecar metadata written by the assignment step."""
    meta_path = output_dir / "skims_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read skims_meta.json", exc_info=True)
        return None


def _validate_skim_convergence(
    output_dir: Path,
    *,
    allow_unconverged: bool = False,
) -> None:
    """Check that skims come from a sufficiently converged assignment.

    Raises ``RuntimeError`` when skims are non-converged and
    *allow_unconverged* is ``False`` (the default).
    """
    meta = load_skim_metadata(output_dir)
    if meta is None:
        logger.warning(
            "No skims_meta.json found alongside skims.aem. "
            "Cannot verify assignment convergence for distribution impedance. "
            "Re-run assign-warm-skims or assign to generate metadata."
        )
        return
    rgap = meta.get("final_rgap")
    converged = meta.get("converged", False)
    if not converged:
        msg = (
            f"Skims come from a NON-CONVERGED assignment (rgap={rgap}). "
            "Distribution on unconverged impedance produces unreliable "
            "trip tables. Re-run assign-warm-skims with a tighter rgap target."
        )
        if allow_unconverged:
            logger.warning(msg + " (proceeding because allow_unconverged_skims=true)")
        else:
            raise RuntimeError(
                msg + " Set demand.distribution.allow_unconverged_skims=true "
                "to override (prototyping only)."
            )
    elif rgap is not None and rgap > _SKIM_CONVERGENCE_WARN_THRESHOLD:
        logger.warning(
            "Skims come from an assignment with rgap=%.6f, "
            "above the recommended threshold (%.0e). "
            "Consider tightening convergence for better impedance quality.",
            rgap, _SKIM_CONVERGENCE_WARN_THRESHOLD,
        )


def _load_impedance(
    output_dir: Path,
    zone_ids: np.ndarray,
) -> Optional[np.ndarray]:
    """Load skim matrix from previous assignment as impedance."""
    skim_path = output_dir / "skims.aem"
    if not skim_path.exists():
        return None
    try:
        mat = AequilibraeMatrix()
        mat.load(str(skim_path))
        core_names = list(mat.names)
        if not core_names:
            mat.close()
            return None
        data = mat.matrix[core_names[0]][:, :]
        mat.close()
        if data.shape[0] == len(zone_ids):
            return data.astype(np.float64)
        return None
    except Exception:
        return None


def _euclidean_impedance(zones_gdf: gpd.GeoDataFrame, zone_ids: np.ndarray) -> np.ndarray:
    """Fallback impedance: Euclidean distance between zone centroids."""
    return pairwise_euclidean(zones_gdf, zone_ids, metric_epsg=5514)


def _load_zone_employment(cache_dir: Path) -> Optional[Dict[int, int]]:
    """Load zone employment counts from ``zone_employment.parquet`` if available."""
    path = cache_dir / "zone_employment.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "zone_id" not in df.columns:
            return None
        emp_col = next((c for c in ("employment", "jobs", "employees") if c in df.columns), None)
        if emp_col is None:
            return None
        return dict(zip(df["zone_id"].astype(int), df[emp_col].astype(int)))
    except Exception:
        logger.debug("Failed to load zone employment data", exc_info=True)
        return None
