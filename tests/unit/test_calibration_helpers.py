"""Unit tests for pure helpers in sim.calibration."""
from __future__ import annotations

import math
from pathlib import Path
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
    _entropy_update_step,
    _odme_objective,
    _sr_val,
    compute_objective,
    compute_stats,
    compute_validation_benchmarks,
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

    def test_daily_ignores_geh(self):
        """Daily convergence should not depend on GEH at all."""
        history = [{
            "r2": 0.90,
            "slope": 1.0,
            "pct_rmse": 25.0,
            "bias_pct": 5.0,
            "geh_mean": 99.0,
            "geh_lt5_pct": 0.0,
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
        history = [{"geh_lt5_pct": 90.0, "geh_mean": 3.0}]
        assert _check_final_convergence(history, "hourly", 85.0, {"geh_mean_max": 5.0}) is True

    def test_hourly_fail_geh_lt5(self):
        history = [{"geh_lt5_pct": 70.0, "geh_mean": 3.0}]
        assert _check_final_convergence(history, "hourly", 85.0, {"geh_mean_max": 5.0}) is False

    def test_hourly_fail_geh_mean(self):
        history = [{"geh_lt5_pct": 90.0, "geh_mean": 8.0}]
        assert _check_final_convergence(history, "hourly", 85.0, {"geh_mean_max": 5.0}) is False


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
    _weights = {"geh": 1.0, "geh_mean": 2.0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0.5, "screenline": 2.0, "jt": 1.0}

    def test_perfect_stats_hourly(self):
        stats = {"geh_lt5_pct": 100, "geh_mean": 0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        result = compute_objective(stats, {}, [], self._weights, model_time_period="hourly")
        assert result == pytest.approx(0.0)

    def test_geh_mean_penalty_hourly(self):
        stats = {"geh_lt5_pct": 100, "geh_mean": 4.0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        result = compute_objective(stats, {}, [], self._weights, model_time_period="hourly")
        assert result == pytest.approx(4.0 * 2.0)

    def test_daily_ignores_geh(self):
        """Daily objective should be zero for perfect R²/slope/RMSE regardless of GEH."""
        stats = {"geh_lt5_pct": 0, "geh_mean": 99.0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        result = compute_objective(stats, {}, [], self._weights, model_time_period="daily")
        assert result == pytest.approx(0.0)

    def test_screenline_penalty(self):
        stats = {"geh_lt5_pct": 100, "geh_mean": 0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        sl = {"SL1": {"ratio": 1.2}, "SL2": {"ratio": 0.9}}
        result = compute_objective(stats, sl, [], self._weights, model_time_period="hourly")
        expected_sl = (0.2 + 0.1) * 2.0
        assert result == pytest.approx(expected_sl)

    def test_jt_failures(self):
        stats = {"geh_lt5_pct": 100, "geh_mean": 0, "r2": 1.0, "slope": 1.0, "pct_rmse": 0}
        jt = [{"pass": False}, {"pass": True}]
        result = compute_objective(stats, {}, jt, self._weights, model_time_period="hourly")
        expected_jt = 1 / 2 * 100 * 1.0
        assert result == pytest.approx(expected_jt)


# ---------------------------------------------------------------------------
# _entropy_update_step
# ---------------------------------------------------------------------------
class TestEntropyUpdateStep:
    @staticmethod
    def _make_demand_and_sl(n: int = 4):
        """Helper: 4x4 demand matrix, one screenline OD."""
        demand = np.ones((n, n), dtype=np.float64) * 100.0
        np.fill_diagonal(demand, 0.0)
        sl_od = np.zeros((n, n), dtype=np.float64)
        sl_od[0, 1] = 60.0
        sl_od[0, 2] = 40.0
        return demand, sl_od

    def test_ratio_above_one_increases_demand(self):
        demand, sl_od = self._make_demand_and_sl()
        demand_before = demand.copy()
        sl_matrices = {"SL_test": sl_od}
        sl_results = {"SL_test": {"observed_total": 200, "modeled_total": 100}}
        n = _entropy_update_step(demand, sl_matrices, sl_results, step_size=0.5)
        assert n == 1
        # Cells using the screenline should increase
        assert demand[0, 1] > demand_before[0, 1]
        assert demand[0, 2] > demand_before[0, 2]
        # Cells not using the screenline stay the same
        assert demand[2, 3] == pytest.approx(demand_before[2, 3])

    def test_ratio_below_one_decreases_demand(self):
        demand, sl_od = self._make_demand_and_sl()
        demand_before = demand.copy()
        sl_matrices = {"SL_test": sl_od}
        sl_results = {"SL_test": {"observed_total": 50, "modeled_total": 100}}
        _entropy_update_step(demand, sl_matrices, sl_results, step_size=0.5)
        assert demand[0, 1] < demand_before[0, 1]

    def test_extreme_ratio_skipped(self):
        demand, sl_od = self._make_demand_and_sl()
        demand_before = demand.copy()
        sl_matrices = {"SL_test": sl_od}
        sl_results = {"SL_test": {"observed_total": 1000, "modeled_total": 10}}
        n = _entropy_update_step(demand, sl_matrices, sl_results, step_size=0.5, ratio_max=5.0)
        assert n == 0
        np.testing.assert_array_equal(demand, demand_before)

    def test_zero_obs_skipped(self):
        demand, sl_od = self._make_demand_and_sl()
        demand_before = demand.copy()
        sl_matrices = {"SL_test": sl_od}
        sl_results = {"SL_test": {"observed_total": 0, "modeled_total": 100}}
        n = _entropy_update_step(demand, sl_matrices, sl_results)
        assert n == 0
        np.testing.assert_array_equal(demand, demand_before)

    def test_step_size_zero_no_change(self):
        demand, sl_od = self._make_demand_and_sl()
        demand_before = demand.copy()
        sl_matrices = {"SL_test": sl_od}
        sl_results = {"SL_test": {"observed_total": 200, "modeled_total": 100}}
        _entropy_update_step(demand, sl_matrices, sl_results, step_size=0.0)
        np.testing.assert_array_almost_equal(demand, demand_before)

    def test_adjustment_clipped(self):
        demand, sl_od = self._make_demand_and_sl()
        sl_matrices = {"SL_test": sl_od}
        sl_results = {"SL_test": {"observed_total": 400, "modeled_total": 100}}
        _entropy_update_step(
            demand, sl_matrices, sl_results,
            step_size=1.0, clip_min=0.8, clip_max=1.2,
        )
        # Even with large step size, clipping prevents extreme changes
        assert demand.max() <= 120.1  # 100 * 1.2 + tolerance


# ---------------------------------------------------------------------------
# compute_validation_benchmarks — config-driven thresholds
# ---------------------------------------------------------------------------
class TestValidationBenchmarksConfigDriven:
    def test_default_geh_threshold_85(self):
        stats = {"geh_lt5_pct": 86.0}
        result = compute_validation_benchmarks(stats, {}, [])
        assert result["geh_benchmark_pass_hourly"] is True

    def test_custom_geh_threshold_90(self):
        stats = {"geh_lt5_pct": 86.0}
        result = compute_validation_benchmarks(
            stats, {}, [], benchmarks={"geh_lt5_pass_pct": 90.0},
        )
        assert result["geh_benchmark_pass_hourly"] is False

    def test_default_jt_threshold_85(self):
        jt = [{"pass": True}] * 84 + [{"pass": False}] * 16
        stats = {"geh_lt5_pct": 90.0}
        result = compute_validation_benchmarks(stats, {}, jt)
        assert result["jt_benchmark_pass"] is False

    def test_custom_jt_threshold_80(self):
        jt = [{"pass": True}] * 84 + [{"pass": False}] * 16
        stats = {"geh_lt5_pct": 90.0}
        result = compute_validation_benchmarks(
            stats, {}, jt, benchmarks={"jt_pass_pct": 80.0},
        )
        assert result["jt_benchmark_pass"] is True


# ---------------------------------------------------------------------------
# Screenline factors config wiring (smoke test)
# ---------------------------------------------------------------------------
class TestScreenlineFactorsConfig:
    """Verify that _CalibrationContext reads screenline_factors from config."""

    def test_defaults_match_sim_defaults(self):
        from sim.defaults import SIM_DEFAULTS
        sf = SIM_DEFAULTS["calibration"]["screenline_factors"]
        assert sf["ratio_max"] == 5.0
        assert sf["ratio_min"] == 0.2
        assert sf["clip_min"] == 0.5
        assert sf["clip_max"] == 2.0
        assert sf["global_min"] == 0.70
        assert sf["global_max"] == 1.50

    def test_gateway_defaults_match_sim_defaults(self):
        from sim.defaults import SIM_DEFAULTS
        gw = SIM_DEFAULTS["calibration"]["gateway_calibration"]
        assert gw["damping"] == 0.30
        assert gw["min_factor"] == 0.50
        assert gw["max_factor"] == 2.00

    def test_auto_screenlines_defaults(self):
        from sim.defaults import SIM_DEFAULTS
        asl = SIM_DEFAULTS["calibration"]["auto_screenlines"]
        assert asl["gateway_screenlines"] is True
        assert asl["csd_screenlines"] is True


# ---------------------------------------------------------------------------
# Dynamic screenline-gateway mapping
# ---------------------------------------------------------------------------
class TestBuildScreenlineGatewayMap:
    def test_auto_gw_screenlines(self):
        from sim.calibration import _build_screenline_gateway_map
        from sim.calibration.screenlines import ScreenlineDef

        sls = [
            ScreenlineDef(name="auto_gw_D35_N"),
            ScreenlineDef(name="auto_gw_D46_E"),
            ScreenlineDef(name="auto_gw_I46_S"),
        ]
        gw_names = {"D35_N", "D46_E", "I46_S"}
        mapping = _build_screenline_gateway_map(sls, gw_names)

        assert mapping["auto_gw_D35_N"] == "D35_N"
        assert mapping["auto_gw_D46_E"] == "D46_E"
        assert mapping["auto_gw_I46_S"] == "I46_S"

    def test_legacy_fallback(self):
        from sim.calibration import _build_screenline_gateway_map
        from sim.calibration.screenlines import ScreenlineDef

        sls = [ScreenlineDef(name="D1_west")]
        mapping = _build_screenline_gateway_map(sls, set())
        assert mapping["D1_west"] == "D1_NW"

    def test_auto_overrides_legacy(self):
        from sim.calibration import _build_screenline_gateway_map
        from sim.calibration.screenlines import ScreenlineDef

        sls = [
            ScreenlineDef(name="auto_gw_D1_NW"),
            ScreenlineDef(name="D1_west"),
        ]
        mapping = _build_screenline_gateway_map(sls, {"D1_NW"})
        assert mapping["auto_gw_D1_NW"] == "D1_NW"
        assert mapping["D1_west"] == "D1_NW"

    def test_no_gw_names_accepts_any(self):
        from sim.calibration import _build_screenline_gateway_map
        from sim.calibration.screenlines import ScreenlineDef

        sls = [ScreenlineDef(name="auto_gw_ANYTHING")]
        mapping = _build_screenline_gateway_map(sls)
        assert mapping["auto_gw_ANYTHING"] == "ANYTHING"


# ---------------------------------------------------------------------------
# Benchmark save/restore
# ---------------------------------------------------------------------------
class TestBenchmarkSaveRestore:
    def test_save_creates_file(self, tmp_path):
        """save_seed copies the OD matrix to the benchmark directory."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

        # Create a fake matrix file
        demand_dir = tmp_path / "data" / "brno" / "demand"
        demand_dir.mkdir(parents=True)
        matrix_file = demand_dir / "od_matrix.aem"
        matrix_file.write_bytes(b"fake-aem-content")

        # Create a minimal config
        import yaml
        cfg_dir = tmp_path / "config" / "brno"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "sim.yaml"
        cfg_path.write_text(yaml.dump({
            "project_path": str(tmp_path / "project" / "brno_aeq"),
            "demand": {
                "matrix_path": str(matrix_file),
                "output_dir": str(tmp_path / "outputs" / "brno" / "baseline" / "demand"),
            },
        }), encoding="utf-8")

        from calibration_benchmark import save_seed
        snapshot = save_seed(str(cfg_path))
        assert Path(snapshot).exists()
        assert Path(snapshot).read_bytes() == b"fake-aem-content"

    def test_restore_overwrites(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

        demand_dir = tmp_path / "data" / "brno" / "demand"
        demand_dir.mkdir(parents=True)
        matrix_file = demand_dir / "od_matrix.aem"
        matrix_file.write_bytes(b"modified-content")

        snap_file = tmp_path / "snapshot.aem"
        snap_file.write_bytes(b"original-content")

        import yaml
        cfg_dir = tmp_path / "config" / "brno"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "sim.yaml"
        cfg_path.write_text(yaml.dump({
            "project_path": str(tmp_path / "project" / "brno_aeq"),
            "demand": {
                "matrix_path": str(matrix_file),
                "output_dir": str(tmp_path / "outputs" / "brno" / "baseline" / "demand"),
            },
        }), encoding="utf-8")

        from calibration_benchmark import restore_seed
        restore_seed(str(cfg_path), str(snap_file))
        assert matrix_file.read_bytes() == b"original-content"
        orig = matrix_file.with_suffix(".aem.orig")
        assert orig.read_bytes() == b"original-content"
