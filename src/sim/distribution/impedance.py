"""Impedance loading: skim matrices and Euclidean fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import pairwise_euclidean

logger = logging.getLogger(__name__)


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
