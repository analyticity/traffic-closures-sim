"""Unit tests for sim._metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim._metrics import aggregate_daily_volumes, compute_geh, persons_to_vehicles


# ---------------------------------------------------------------------------
# compute_geh
# ---------------------------------------------------------------------------
class TestComputeGeh:
    def test_known_values(self):
        m = np.array([100.0, 200.0])
        c = np.array([110.0, 190.0])
        geh = compute_geh(m, c)
        expected_0 = np.sqrt(2 * (100 - 110) ** 2 / (100 + 110))
        expected_1 = np.sqrt(2 * (200 - 190) ** 2 / (200 + 190))
        assert geh[0] == pytest.approx(expected_0, rel=1e-6)
        assert geh[1] == pytest.approx(expected_1, rel=1e-6)

    def test_both_zero_is_nan(self):
        geh = compute_geh(np.array([0.0]), np.array([0.0]))
        assert np.isnan(geh[0])

    def test_equal_is_zero(self):
        geh = compute_geh(np.array([500.0]), np.array([500.0]))
        assert geh[0] == pytest.approx(0.0)

    def test_negative_inputs(self):
        geh = compute_geh(np.array([-10.0]), np.array([10.0]))
        assert np.isfinite(geh[0]) or np.isnan(geh[0])

    def test_vectorized(self):
        m = np.array([100.0, 0.0, 50.0])
        c = np.array([100.0, 0.0, 60.0])
        geh = compute_geh(m, c)
        assert geh[0] == pytest.approx(0.0)
        assert np.isnan(geh[1])
        assert geh[2] > 0


# ---------------------------------------------------------------------------
# persons_to_vehicles
# ---------------------------------------------------------------------------
class TestPersonsToVehicles:
    def test_known_conversion(self):
        result = persons_to_vehicles(
            1000, car_share=0.5, occupancy=1.25, trips_per_person=2.0
        )
        assert result == pytest.approx(1000 * 2.0 * 0.5 / 1.25)

    def test_zero_occupancy_clamped(self):
        result = persons_to_vehicles(
            100, car_share=1.0, occupancy=0.0, trips_per_person=1.0
        )
        assert result == pytest.approx(100 / 0.01)

    def test_negative_persons(self):
        result = persons_to_vehicles(
            -50, car_share=1.0, occupancy=1.0, trips_per_person=1.0
        )
        assert result == 0.0


# ---------------------------------------------------------------------------
# aggregate_daily_volumes
# ---------------------------------------------------------------------------
class TestAggregateDailyVolumes:
    def test_local_plus_external(self):
        df = pd.DataFrame({
            "wd_daily_local_ab": [100, 200],
            "wd_daily_external_through_ab": [10, 20],
            "wd_daily_local_ba": [50, 60],
            "wd_daily_external_through_ba": [5, 6],
            "wd_daily_local_tot": [150, 260],
            "wd_daily_external_through_tot": [15, 26],
        })
        result = aggregate_daily_volumes(df)
        assert list(result["wd_daily_ab"]) == [110, 220]
        assert list(result["wd_daily_ba"]) == [55, 66]
        assert list(result["wd_daily_tot"]) == [165, 286]

    def test_no_external_column(self):
        df = pd.DataFrame({
            "wd_daily_local_ab": [100],
            "wd_daily_local_ba": [50],
            "wd_daily_local_tot": [150],
        })
        result = aggregate_daily_volumes(df)
        assert result["wd_daily_ab"].iloc[0] == 100
        assert result["wd_daily_tot"].iloc[0] == 150

    def test_already_has_target_noop(self):
        df = pd.DataFrame({
            "wd_daily_ab": [999],
            "wd_daily_local_ab": [100],
        })
        result = aggregate_daily_volumes(df)
        assert result["wd_daily_ab"].iloc[0] == 999

    def test_missing_all_columns_noop(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = aggregate_daily_volumes(df)
        assert "wd_daily_ab" not in result.columns
