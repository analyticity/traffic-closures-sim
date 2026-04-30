"""Production/attraction vector construction from population and employment."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def build_pa_vectors(
    zone_ids: np.ndarray,
    population: Dict[int, int],
    trip_rate: float = 2.5,
    car_share: float = 0.50,
    occupancy: float = 1.3,
    employment: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """Build production/attraction vectors from population and optional employment.

    When *employment* is provided, attractions are derived from employment
    counts (asymmetric P/A), which better reflects the spatial distribution
    of trip destinations.  Without employment data, P_i = A_i (symmetric
    proxy based on population only).
    """
    pa = pd.DataFrame({"zone_id": zone_ids.astype(int)})
    pa["population"] = pa["zone_id"].map(lambda z: population.get(int(z), 0))
    daily_prod = pa["population"] * trip_rate * car_share / max(occupancy, 0.01)
    pa["production"] = daily_prod

    if employment:
        pa["employment"] = pa["zone_id"].map(lambda z: employment.get(int(z), 0))
        emp_total = pa["employment"].sum()
        if emp_total > 0:
            pa["attraction"] = pa["employment"] * (daily_prod.sum() / emp_total)
        else:
            pa["attraction"] = daily_prod
    else:
        pa["attraction"] = daily_prod

    return pa
