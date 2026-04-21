"""Unit tests for pure helpers in sim.calibration."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from sim.calibration import (
    _bearing_diff,
    _bearing_from_geom,
    _check_final_convergence,
    _coarse_road_class,
    _compute_count_weights,
    _odme_objective,
    _sr_val,
    compute_objective,
    compute_stats,
)


# ---------------------------------------------------------------------------
# _odme_objective
# ---------------------------------------------------------------------------
class TestOdmeObjective:
    def test_known_residuals(self):
        mod = np.array([10.0, 20.0])
        obs = np.array([12.0, 18.0])
        w = np.array([1.0, 1.0])
        result = _odme_objective(mod, obs, w)
        expected = (10 - 12) ** 2 + (20 - 18) ** 2
        assert result == pytest.approx(expected)

    def test_weights_applied(self):
        mod = np.array([10.0])
        obs = np.array([20.0])
        w = np.array([0.5])
        assert _odme_objective(mod, obs, w) == pytest.approx(0.5 * 100)

    def test_zero_residuals(self):
        v = np.array([5.0, 10.0])
        assert _odme_objective(v, v, np.ones(2)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_count_weights
# ---------------------------------------------------------------------------
class TestComputeCountWeights:
    def test_inverse_sqrt(self):
        obs = np.array([400.0, 100.0])
        w = _compute_count_weights(obs, "inverse_sqrt")
        assert w[0] == pytest.approx(1.0 / 20.0)
        assert w[1] == pytest.approx(1.0 / 10.0)

    def test_inverse(self):
        obs = np.array([200.0])
        w = _compute_count_weights(obs, "inverse")
        assert w[0] == pytest.approx(1.0 / 200.0)

    def test_uniform(self):
        w = _compute_count_weights(np.array([500.0, 1000.0]), "uniform")
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(1.0)

    def test_small_obs_clamped_to_100(self):
        w = _compute_count_weights(np.array([10.0]), "inverse_sqrt")
        assert w[0] == pytest.approx(1.0 / 10.0)


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------
class TestComputeStats:
    def test_empty_input(self):
        result = compute_stats(np.array([]), np.array([]))
        assert result == {"n": 0}

    def test_perfect_fit(self):
        obs = np.array([100.0, 200.0, 300.0])
        result = compute_stats(obs, obs)
        assert result["r2"] == pytest.approx(1.0, abs=0.001)
        assert result["slope"] == pytest.approx(1.0, abs=0.001)
        assert result["rmse"] == pytest.approx(0.0, abs=0.1)
        assert result["geh_lt5_pct"] == pytest.approx(100.0)

    def test_known_regression(self):
        obs = np.array([100.0, 200.0, 300.0, 400.0])
        mod = np.array([110.0, 190.0, 310.0, 390.0])
        result = compute_stats(mod, obs)
        assert result["n"] == 4
        assert result["r2"] is not None and result["r2"] > 0.9
        assert result["slope"] is not None

    def test_zero_obs_excluded(self):
        mod = np.array([10.0, 20.0, 5.0])
        obs = np.array([0.0, 20.0, 5.0])
        result = compute_stats(mod, obs)
        assert result["n"] == 2

    def test_daily_capacity_factor_scales_geh_threshold(self):
        obs = np.array([1000.0, 2000.0])
        mod = np.array([1050.0, 2050.0])
        r1 = compute_stats(mod, obs, daily_capacity_factor=1.0)
        r10 = compute_stats(mod, obs, daily_capacity_factor=10.0)
        assert r10["daily_geh_threshold"] > r1["daily_geh_threshold"]


# ---------------------------------------------------------------------------
# _coarse_road_class
# ---------------------------------------------------------------------------
class TestCoarseRoadClass:
    def test_motorway_link(self):
        assert _coarse_road_class("motorway_link") == "motorway"

    def test_secondary(self):
        assert _coarse_road_class("secondary") == "secondary"

    def test_trunk(self):
        assert _coarse_road_class("trunk_link") == "trunk"

    def test_unknown(self):
        assert _coarse_road_class("residential") == "other"

    def test_none(self):
        assert _coarse_road_class(None) == "other"


# ---------------------------------------------------------------------------
# _check_final_convergence
# ---------------------------------------------------------------------------
class TestCheckFinalConvergence:
    @staticmethod
    def _daily_conv():
        return {
            "r2_target": 0.80,
            "slope_range": [0.85, 1.15],
            "pct_rmse_max": 35.0,
            "bias_abs_max_pct": 15.0,
            "screenline_max_pct_deviation": 15.0,
        }

    def test_empty_history(self):
        assert _check_final_convergence([], "daily", 85.0, self._daily_conv()) is False

    def test_daily_all_pass(self):
        history = [{
            "r2": 0.90,
            "slope": 1.0,
            "pct_rmse": 25.0,
            "bias_pct": 5.0,
            "max_screenline_pct_dev": 10.0,
        }]
        assert _check_final_convergence(history, "daily", 85.0, self._daily_conv()) is True

    def test_daily_fail_r2(self):
        history = [{
            "r2": 0.50,
            "slope": 1.0,
            "pct_rmse": 25.0,
            "bias_pct": 5.0,
            "max_screenline_pct_dev": 10.0,
        }]
        assert _check_final_convergence(history, "daily", 85.0, self._daily_conv()) is False

    def test_daily_fail_screenline(self):
        history = [{
            "r2": 0.90,
            "slope": 1.0,
            "pct_rmse": 25.0,
            "bias_pct": 5.0,
            "max_screenline_pct_dev": 20.0,
        }]
        assert _check_final_convergence(history, "daily", 85.0, self._daily_conv()) is False

    def test_hourly_pass(self):
        history = [{"geh_lt5_pct": 90.0}]
        assert _check_final_convergence(history, "hourly", 85.0, {}) is True

    def test_hourly_fail(self):
        history = [{"geh_lt5_pct": 70.0}]
        assert _check_final_convergence(history, "hourly", 85.0, {}) is False


# ---------------------------------------------------------------------------
# _sr_val
# ---------------------------------------------------------------------------
class TestSrVal:
    def test_dict_input(self):
        assert _sr_val({"observed_total": 500}, "observed_total") == 500.0

    def test_object_input(self):
        obj = SimpleNamespace(observed_total=300)
        assert _sr_val(obj, "observed_total") == 300.0

    def test_missing_key_returns_zero(self):
        assert _sr_val({}, "observed_total") == 0.0

    def test_none_value_returns_zero(self):
        assert _sr_val({"observed_total": None}, "observed_total") == 0.0


# ---------------------------------------------------------------------------
# _bearing_diff
# ---------------------------------------------------------------------------
class TestBearingDiff:
    def test_opposite(self):
        assert _bearing_diff(0, 180) == pytest.approx(180.0)

    def test_wrap_around(self):
        assert _bearing_diff(10, 350) == pytest.approx(20.0)

    def test_same(self):
        assert _bearing_diff(90, 90) == pytest.approx(0.0)

    def test_270_vs_90(self):
        assert _bearing_diff(270, 90) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# _bearing_from_geom
# ---------------------------------------------------------------------------
class TestBearingFromGeom:
    def test_north(self):
        from shapely.geometry import LineString
        geom = LineString([(0, 0), (0, 1)])
        bearing = _bearing_from_geom(geom)
        assert bearing == pytest.approx(0.0, abs=0.1)

    def test_east(self):
        from shapely.geometry import LineString
        geom = LineString([(0, 0), (1, 0)])
        bearing = _bearing_from_geom(geom)
        assert bearing == pytest.approx(90.0, abs=0.1)

    def test_empty_linestring_returns_none(self):
        from shapely.geometry import LineString
        geom = LineString()
        assert _bearing_from_geom(geom) is None

    def test_none_returns_none(self):
        assert _bearing_from_geom(None) is None


# ---------------------------------------------------------------------------
# compute_objective
# ---------------------------------------------------------------------------
class TestComputeObjective:
    def test_perfect_stats(self):
        stats = {"geh_lt5_pct": 100, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        weights = {"geh": 1.0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0.5, "screenline": 2.0, "jt": 1.0}
        result = compute_objective(stats, {}, [], weights)
        assert result == pytest.approx(0.0)

    def test_screenline_penalty(self):
        stats = {"geh_lt5_pct": 100, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        sl = {"SL1": {"ratio": 1.2}, "SL2": {"ratio": 0.9}}
        weights = {"geh": 1.0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0.5, "screenline": 2.0, "jt": 1.0}
        result = compute_objective(stats, sl, [], weights)
        expected_sl = (0.2 + 0.1) * 2.0
        assert result == pytest.approx(expected_sl)

    def test_jt_failures(self):
        stats = {"geh_lt5_pct": 100, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        jt = [{"pass": False}, {"pass": True}]
        weights = {"geh": 1.0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0.5, "screenline": 2.0, "jt": 1.0}
        result = compute_objective(stats, {}, jt, weights)
        expected_jt = 1 / 2 * 100 * 1.0
        assert result == pytest.approx(expected_jt)
