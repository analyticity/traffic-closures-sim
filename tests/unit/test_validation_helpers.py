"""Unit tests for validation helpers in sim.calibration."""
from __future__ import annotations

import numpy as np
import pytest

from sim.calibration import (
    _classify_csd_road,
    compute_validation_benchmarks,
    validate_journey_times,
)


# ---------------------------------------------------------------------------
# validate_journey_times
# ---------------------------------------------------------------------------
class TestValidateJourneyTimes:
    @staticmethod
    def _skim(n=3, val=600.0):
        mat = np.full((n, n), val)
        np.fill_diagonal(mat, 0)
        return mat

    def test_valid_route_pass(self):
        skim = self._skim(3, val=600.0)  # 10 min
        zones = np.array([1, 2, 3])
        routes = [{"name": "A-B", "from_zone": 1, "to_zone": 2,
                    "reference_time_min": 10, "tolerance_pct": 15}]
        results = validate_journey_times(skim, zones, routes)
        assert len(results) == 1
        assert results[0]["pass"] is True
        assert results[0]["model_min"] == pytest.approx(10.0)

    def test_valid_route_fail(self):
        skim = self._skim(3, val=1200.0)  # 20 min
        zones = np.array([1, 2, 3])
        routes = [{"name": "A-B", "from_zone": 1, "to_zone": 2,
                    "reference_time_min": 10, "tolerance_pct": 15}]
        results = validate_journey_times(skim, zones, routes)
        assert results[0]["pass"] is False

    def test_unknown_zone_skipped(self):
        skim = self._skim(2)
        zones = np.array([1, 2])
        routes = [{"name": "bad", "from_zone": 1, "to_zone": 99,
                    "reference_time_min": 10}]
        results = validate_journey_times(skim, zones, routes)
        assert results[0]["status"] == "skipped"

    def test_zero_ref_min_skipped(self):
        skim = self._skim(2)
        zones = np.array([1, 2])
        routes = [{"name": "zero", "from_zone": 1, "to_zone": 2,
                    "reference_time_min": 0}]
        results = validate_journey_times(skim, zones, routes)
        assert results[0]["status"] == "skipped"

    def test_zero_model_sec_no_path(self):
        skim = np.zeros((2, 2))
        zones = np.array([1, 2])
        routes = [{"name": "zero", "from_zone": 1, "to_zone": 2,
                    "reference_time_min": 10}]
        results = validate_journey_times(skim, zones, routes)
        assert results[0]["status"] == "no_path"

    def test_within_1min_overrides_pct(self):
        skim = self._skim(2, val=11.5 * 60)  # 11.5 min
        zones = np.array([1, 2])
        routes = [{"name": "close", "from_zone": 1, "to_zone": 2,
                    "reference_time_min": 10.6, "tolerance_pct": 1}]
        results = validate_journey_times(skim, zones, routes)
        assert results[0]["pass"] is True


# ---------------------------------------------------------------------------
# compute_validation_benchmarks
# ---------------------------------------------------------------------------
class TestComputeValidationBenchmarks:
    @staticmethod
    def _good_stats():
        return {
            "r2": 0.92, "slope": 1.02, "pct_rmse": 25.0,
            "bias_pct": 5.0, "geh_lt5_pct": 70.0, "daily_geh_lt_adj_pct": 85.0,
        }

    def test_daily_all_pass(self):
        result = compute_validation_benchmarks(
            self._good_stats(), {}, [],
            model_time_period="daily",
        )
        assert result["daily_overall_pass"] is True
        assert result["overall_pass"] is True

    def test_daily_fail_r2(self):
        stats = self._good_stats()
        stats["r2"] = 0.50
        result = compute_validation_benchmarks(stats, {}, [], model_time_period="daily")
        assert result["daily_r2_pass"] is False
        assert result["overall_pass"] is False

    def test_hourly_geh_pass(self):
        stats = {"geh_lt5_pct": 90.0}
        result = compute_validation_benchmarks(stats, {}, [], model_time_period="hourly")
        assert result["overall_pass"] is True

    def test_hourly_geh_fail(self):
        stats = {"geh_lt5_pct": 70.0}
        result = compute_validation_benchmarks(stats, {}, [], model_time_period="hourly")
        assert result["overall_pass"] is False

    def test_jt_empty_is_none(self):
        result = compute_validation_benchmarks(
            self._good_stats(), {}, [],
            model_time_period="daily",
        )
        assert result["jt_benchmark_pass"] is None

    def test_jt_partial_pass(self):
        jt = [{"pass": True}, {"pass": False}, {"pass": True}]
        result = compute_validation_benchmarks(
            self._good_stats(), {}, jt,
            model_time_period="daily",
        )
        assert result["jt_within_tolerance_pct"] == pytest.approx(200 / 3, rel=0.01)

    def test_screenline_max_error(self):
        sl = {
            "SL1": {"ratio": 1.20, "modeled_total": 1200, "observed_total": 1000},
            "SL2": {"ratio": 0.85, "modeled_total": 1700, "observed_total": 2000},
        }
        result = compute_validation_benchmarks(
            self._good_stats(), sl, [],
            model_time_period="daily",
        )
        assert result["screenline_max_error_pct"] == pytest.approx(20.0)
        assert result["daily_screenline_pass"] is False

    def test_custom_thresholds(self):
        stats = self._good_stats()
        stats["r2"] = 0.75
        result = compute_validation_benchmarks(
            stats, {}, [],
            model_time_period="daily",
            daily_thresholds={"r2_target": 0.70},
        )
        assert result["daily_r2_pass"] is True

    def test_null_r2_propagated(self):
        """When r2/slope are None (n<2), benchmarks must propagate None, not 0.0."""
        stats = self._good_stats()
        stats["r2"] = None
        stats["slope"] = None
        result = compute_validation_benchmarks(
            stats, {}, [], model_time_period="daily",
        )
        assert result["daily_r2"] is None
        assert result["daily_r2_pass"] is None
        assert result["daily_slope"] is None
        assert result["daily_slope_pass"] is None
        assert result["overall_pass"] is False

    def test_null_r2_calibration_fallback(self):
        """Null R² via calibration fallback (no holdout) still propagates None."""
        stats = {"r2": None, "slope": None, "pct_rmse": 20.0,
                 "bias_pct": 3.0, "geh_lt5_pct": 90.0,
                 "daily_geh_lt_adj_pct": 90.0}
        result = compute_validation_benchmarks(
            stats, {}, [], model_time_period="daily",
        )
        assert result["verdict_source"] == "calibration"
        assert result["calibration_fit"]["r2"] is None
        assert result["calibration_fit"]["r2_pass"] is None
        assert result["calibration_fit"]["overall_pass"] is False

    def test_holdout_slope_propagated(self):
        """Holdout slope from CSD summary must flow through to the verdict."""
        calib_stats = self._good_stats()
        holdout = {
            "n": 12,
            "r2": 0.85,
            "slope": 0.72,
            "pct_rmse": 30.0,
            "bias_pct": 8.0,
            "geh_lt5_pct": 0.0,
            "daily_geh_lt_adj_pct": 0.0,
        }
        result = compute_validation_benchmarks(
            calib_stats, {}, [],
            model_time_period="daily",
            holdout_stats=holdout,
        )
        assert result["verdict_source"] == "holdout"
        assert result["daily_slope"] == pytest.approx(0.72, abs=0.01)
        assert result["daily_slope_pass"] is False

    def test_holdout_slope_pass(self):
        """Holdout slope within range should pass."""
        calib_stats = self._good_stats()
        holdout = {
            "n": 15,
            "r2": 0.90,
            "slope": 1.05,
            "pct_rmse": 25.0,
            "bias_pct": 5.0,
            "geh_lt5_pct": 0.0,
            "daily_geh_lt_adj_pct": 0.0,
        }
        result = compute_validation_benchmarks(
            calib_stats, {}, [],
            model_time_period="daily",
            holdout_stats=holdout,
        )
        assert result["verdict_source"] == "holdout"
        assert result["daily_slope"] == pytest.approx(1.05, abs=0.01)
        assert result["daily_slope_pass"] is True


# ---------------------------------------------------------------------------
# _classify_csd_road (calibration version)
# ---------------------------------------------------------------------------
class TestClassifyCsdRoadCalibration:
    def test_motorway(self):
        assert _classify_csd_road("D1") == "motorway"

    def test_trunk(self):
        assert _classify_csd_road("M55") == "trunk"

    def test_secondary(self):
        assert _classify_csd_road("M250") == "secondary"

    def test_tertiary(self):
        assert _classify_csd_road("M1500") == "tertiary"

    def test_m_only(self):
        assert _classify_csd_road("M") == "trunk"

    def test_garbage(self):
        assert _classify_csd_road("xyz") == "other"
