"""Benchmark metrics must ignore excluded counts and screenlines."""

import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from sim.calibration.matching import filter_matched_counts_for_benchmark
from sim.calibration.metrics import compute_stats, compute_stats_from_matched
from sim.calibration.screenlines import (
    filter_screenline_results_for_benchmark,
    max_screenline_pct_deviation,
)


def test_compute_stats_from_matched_drops_excluded_rows():
    df = pd.DataFrame({
        "_excluded": [False, True, False, False],
        "mod": [100.0, 999.0, 110.0, 90.0],
        "obs": [100.0, 10.0, 100.0, 100.0],
    })
    full = compute_stats(df["mod"].values, df["obs"].values)
    filt = compute_stats_from_matched(df, "mod", "obs")
    assert full["n"] == 4
    assert filt["n"] == 3
    assert filt["bias_pct"] == pytest.approx(0.0, abs=0.1)


def test_max_screenline_pct_dev_ignores_excluded():
    sl = {
        "good": {"modeled_total": 900, "observed_total": 1000, "ratio": 0.9},
        "bad": {"modeled_total": 100, "observed_total": 1000, "ratio": 0.1},
    }
    assert max_screenline_pct_deviation(sl) == pytest.approx(10.0)
    assert len(filter_screenline_results_for_benchmark(sl)) == 1


def test_filter_matched_counts_for_benchmark():
    gdf = gpd.GeoDataFrame({
        "_excluded": [False, True],
        "geometry": [Point(0, 0), Point(1, 1)],
    }, crs="EPSG:4326")
    out = filter_matched_counts_for_benchmark(gdf)
    assert len(out) == 1
