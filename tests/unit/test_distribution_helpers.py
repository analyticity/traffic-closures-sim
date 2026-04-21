"""Unit tests for pure helpers in sim.distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Point

from sim.distribution import (
    _euclidean_impedance,
    _get_nested,
    build_pa_vectors,
    calibrate_gravity_simple,
    run_ipf,
)


# ---------------------------------------------------------------------------
# build_pa_vectors
# ---------------------------------------------------------------------------
class TestBuildPaVectors:
    def test_known_population(self):
        zone_ids = np.array([1, 2, 3])
        pop = {1: 1000, 2: 500, 3: 200}
        pa = build_pa_vectors(zone_ids, pop, trip_rate=2.0, car_share=0.5, occupancy=1.0)
        assert list(pa["zone_id"]) == [1, 2, 3]
        assert pa.loc[0, "production"] == pytest.approx(1000 * 2.0 * 0.5 / 1.0)
        assert pa.loc[1, "production"] == pytest.approx(500 * 2.0 * 0.5 / 1.0)

    def test_symmetric(self):
        pa = build_pa_vectors(np.array([10]), {10: 100})
        assert pa["production"].iloc[0] == pa["attraction"].iloc[0]

    def test_missing_zone_gets_zero(self):
        pa = build_pa_vectors(np.array([1, 2]), {1: 100})
        assert pa.loc[1, "production"] == 0.0

    def test_near_zero_occupancy_clamped(self):
        pa = build_pa_vectors(np.array([1]), {1: 100}, occupancy=0.0)
        assert pa["production"].iloc[0] > 0


# ---------------------------------------------------------------------------
# _euclidean_impedance
# ---------------------------------------------------------------------------
class TestEuclideanImpedance:
    def test_two_zones_distance(self):
        gdf = gpd.GeoDataFrame({
            "zone_id": [1, 2],
            "geometry": [Point(0, 0), Point(3, 4)],
        }, crs="EPSG:5514")
        zone_ids = np.array([1, 2])
        imp = _euclidean_impedance(gdf, zone_ids)
        assert imp.shape == (2, 2)
        assert imp[0, 0] == pytest.approx(0.0)
        assert imp[1, 1] == pytest.approx(0.0)
        assert imp[0, 1] == pytest.approx(5.0)
        assert imp[1, 0] == pytest.approx(5.0)

    def test_missing_zone_stays_zero(self):
        gdf = gpd.GeoDataFrame({
            "zone_id": [1],
            "geometry": [Point(100, 200)],
        }, crs="EPSG:5514")
        zone_ids = np.array([1, 2])
        imp = _euclidean_impedance(gdf, zone_ids)
        assert imp[1, 0] > 0
        assert imp[1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _get_nested
# ---------------------------------------------------------------------------
class TestGetNested:
    def test_happy_path(self):
        cfg = {"a": {"b": {"c": 42}}}
        assert _get_nested(cfg, ["a", "b", "c"]) == 42

    def test_missing_key(self):
        cfg = {"a": {"b": 1}}
        assert _get_nested(cfg, ["a", "x"], default="N/A") == "N/A"

    def test_non_dict_in_path(self):
        cfg = {"a": 5}
        assert _get_nested(cfg, ["a", "b"], default=None) is None

    def test_empty_path(self):
        cfg = {"key": "val"}
        assert _get_nested(cfg, []) == cfg


# ---------------------------------------------------------------------------
# run_ipf (numpy fallback)
# ---------------------------------------------------------------------------
class TestRunIpf:
    @staticmethod
    def _force_numpy_fallback(seed, rows, cols, **kwargs):
        """Force the numpy fallback by monkeypatching the AequilibraE import to fail."""
        import sim.distribution as mod
        import importlib
        import sys

        _saved = sys.modules.get("aequilibrae.distribution")
        sys.modules["aequilibrae.distribution"] = None  # type: ignore[assignment]
        try:
            importlib.reload(mod)
            return mod.run_ipf(seed, rows, cols, **kwargs)
        finally:
            if _saved is None:
                sys.modules.pop("aequilibrae.distribution", None)
            else:
                sys.modules["aequilibrae.distribution"] = _saved
            importlib.reload(mod)

    def test_balanced_targets_converge(self):
        seed = np.array([[10, 5], [3, 8]], dtype=np.float64)
        target_rows = np.array([20.0, 30.0])
        target_cols = np.array([25.0, 25.0])
        result = self._force_numpy_fallback(seed, target_rows, target_cols)
        assert result.sum(axis=1) == pytest.approx(target_rows, rel=0.01)
        assert result.sum(axis=0) == pytest.approx(target_cols, rel=0.01)

    def test_already_matching_seed(self):
        seed = np.array([[5, 5], [5, 5]], dtype=np.float64)
        target_rows = np.array([10.0, 10.0])
        target_cols = np.array([10.0, 10.0])
        result = self._force_numpy_fallback(seed, target_rows, target_cols)
        assert result == pytest.approx(seed, rel=0.01)

    def test_zero_cells_become_positive(self):
        """IPF Furness fallback sets zeros to 1e-12, so output cells are non-zero."""
        seed = np.array([[10, 0], [0, 10]], dtype=np.float64)
        target_rows = np.array([10.0, 10.0])
        target_cols = np.array([10.0, 10.0])
        result = self._force_numpy_fallback(seed, target_rows, target_cols)
        assert result[0, 1] > 0
        assert result[1, 0] > 0


# ---------------------------------------------------------------------------
# calibrate_gravity_simple (fallback path)
# ---------------------------------------------------------------------------
class TestCalibrateGravityFallback:
    @staticmethod
    def _force_fallback(seed, impedance, **kwargs):
        import sim.distribution as mod
        import importlib
        import sys

        _saved = sys.modules.get("aequilibrae.distribution")
        sys.modules["aequilibrae.distribution"] = None  # type: ignore[assignment]
        try:
            importlib.reload(mod)
            return mod.calibrate_gravity_simple(seed, impedance, **kwargs)
        finally:
            if _saved is None:
                sys.modules.pop("aequilibrae.distribution", None)
            else:
                sys.modules["aequilibrae.distribution"] = _saved
            importlib.reload(mod)

    def test_exponential_data_fits_beta(self):
        n = 20
        beta_true = 0.01
        impedance = np.random.default_rng(42).uniform(100, 5000, (n, n))
        seed = np.exp(-beta_true * impedance)
        np.fill_diagonal(seed, 0)

        result = self._force_fallback(seed, impedance)
        assert result["function"] == "EXPO"
        assert abs(result["beta"] - beta_true) < 0.005

    def test_too_few_points_returns_default(self):
        seed = np.zeros((3, 3))
        imp = np.ones((3, 3))
        result = self._force_fallback(seed, imp)
        assert result["beta"] == pytest.approx(0.0001)
