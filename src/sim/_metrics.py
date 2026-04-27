"""Shared traffic model metrics and volume helpers."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_geh(modeled: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Vectorized GEH statistic.  Returns NaN where both inputs are zero."""
    m = np.asarray(modeled, dtype=float)
    c = np.asarray(observed, dtype=float)
    denom = m + c
    mask = denom > 0
    geh = np.full_like(m, np.nan)
    geh[mask] = np.sqrt(2.0 * (m[mask] - c[mask]) ** 2 / denom[mask])
    return geh


def persons_to_vehicles(
    persons: float,
    *,
    car_share: float,
    occupancy: float,
    trips_per_person: float,
) -> float:
    """Convert commuting persons to vehicle trips.

    ``vehicles = persons * trips_per_person * car_share / occupancy``
    """
    return max(float(persons), 0.0) * trips_per_person * car_share / max(occupancy, 0.01)


def persons_to_vehicles_from_cfg(
    work: float,
    school: float,
    cfg_root: Dict[str, Any],
) -> float:
    """Sum work + school vehicle trips using ``demand.conversion`` config."""
    from sim.defaults import SIM_DEFAULTS
    _DEFAULTS = SIM_DEFAULTS["demand"]["conversion"]

    def _conv(p: float, branch: str) -> float:
        conv_cfg = (cfg_root.get("demand") or {}).get("conversion", {}).get(branch, {}) or {}
        defaults = _DEFAULTS.get(branch, _DEFAULTS["work"])
        return persons_to_vehicles(
            p,
            car_share=float(conv_cfg.get("car_share", defaults["car_share"])),
            occupancy=float(conv_cfg.get("occupancy", defaults["occupancy"])),
            trips_per_person=float(conv_cfg.get("trips_per_person", defaults["trips_per_person"])),
        )
    return _conv(work, "work") + _conv(school, "school")


def aggregate_daily_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``wd_daily_{ab,ba,tot}`` from local + external-through components.

    Operates in-place on *df* and returns it for convenience.
    """
    for suffix in ("ab", "ba", "tot"):
        target = f"wd_daily_{suffix}"
        local = f"wd_daily_local_{suffix}"
        ext = f"wd_daily_external_through_{suffix}"
        if target not in df.columns and local in df.columns:
            df[target] = df[local].fillna(0)
            if ext in df.columns:
                df[target] = df[target] + df[ext].fillna(0)
    return df
