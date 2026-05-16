"""Ratio-based exclusion of unreliable CSD link matches."""
import geopandas as gpd
import pandas as pd

from sim.calibration.matching import _apply_ratio_exclusions


def _joined(observed, modeled, *, matched=True, excluded=False):
    return gpd.GeoDataFrame({
        "objectid": [8000019, 8000006],
        "observed_car": [float(observed[0]), float(observed[1])],
        "_corridor_volume": [float(modeled[0]), float(modeled[1])],
        "_matched": [matched, matched],
        "_excluded": [excluded, excluded],
        "geometry": [None, None],
    })


def test_ratio_exclusion_below_and_above():
    df = _joined([1775, 3002], [314, 15092])
    out = _apply_ratio_exclusions(
        df, "_corridor_volume", ratio_below=0.25, ratio_above=5.0,
    )
    assert out["_excluded"].tolist() == [True, True]


def test_ratio_exclusion_skips_already_excluded():
    df = _joined([1775, 3002], [314, 15092], excluded=True)
    out = _apply_ratio_exclusions(
        df, "_corridor_volume", ratio_below=0.25, ratio_above=5.0,
    )
    assert out["_excluded"].all()
