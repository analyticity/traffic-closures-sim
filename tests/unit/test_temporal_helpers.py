"""Unit tests for pure helpers in sim.temporal."""
from __future__ import annotations

from datetime import date

import pytest

from sim.temporal import (
    _classify_csd_road,
    classify_day,
    get_combined_factor,
    get_day_factor,
    get_demand_period_shares,
    get_period_share,
)


# ---------------------------------------------------------------------------
# _classify_csd_road (now aligned with calibration.py)
# ---------------------------------------------------------------------------
class TestClassifyCsdRoad:
    def test_motorway(self):
        assert _classify_csd_road("D1") == "motorway"

    def test_motorway_lowercase(self):
        assert _classify_csd_road("d5") == "motorway"

    def test_trunk(self):
        assert _classify_csd_road("M55") == "trunk"

    def test_secondary(self):
        assert _classify_csd_road("M250") == "secondary"

    def test_tertiary(self):
        assert _classify_csd_road("M1500") == "tertiary"

    def test_non_numeric_with_m(self):
        assert _classify_csd_road("M") == "trunk"

    def test_garbage(self):
        assert _classify_csd_road("xyz") == "other"

    def test_empty(self):
        assert _classify_csd_road("") == "other"


# ---------------------------------------------------------------------------
# classify_day
# ---------------------------------------------------------------------------
class TestClassifyDay:
    def test_workday(self):
        assert classify_day(date(2026, 4, 20)) == "workday"  # Monday

    def test_saturday(self):
        assert classify_day(date(2026, 4, 18)) == "saturday"

    def test_sunday(self):
        assert classify_day(date(2026, 4, 19)) == "sunday"

    def test_holiday_jan1(self):
        assert classify_day(date(2026, 1, 1)) == "holiday"

    def test_holiday_christmas(self):
        assert classify_day(date(2026, 12, 25)) == "holiday"

    def test_string_date(self):
        assert classify_day("2026-04-20") == "workday"


# ---------------------------------------------------------------------------
# get_period_share
# ---------------------------------------------------------------------------
class TestGetPeriodShare:
    @staticmethod
    def _profile():
        return {
            "day_period_shares": {"day": 0.785, "evening": 0.137, "night": 0.078},
            "day_factors": {"workday": 1.07, "saturday": 0.90, "sunday": 0.75, "holiday": 0.70},
        }

    def test_daily(self):
        assert get_period_share("daily", self._profile()) == 1.0

    def test_known_period(self):
        assert get_period_share("day", self._profile()) == pytest.approx(0.785)

    def test_unknown_period_returns_one(self):
        assert get_period_share("nonexistent", self._profile()) == 1.0


# ---------------------------------------------------------------------------
# get_day_factor
# ---------------------------------------------------------------------------
class TestGetDayFactor:
    @staticmethod
    def _profile():
        return {
            "day_period_shares": {"day": 0.785, "evening": 0.137, "night": 0.078},
            "day_factors": {"workday": 1.07, "saturday": 0.90, "sunday": 0.75, "holiday": 0.70},
        }

    def test_workday_factor(self):
        assert get_day_factor(date(2026, 4, 20), self._profile()) == pytest.approx(1.07)

    def test_holiday_factor(self):
        assert get_day_factor(date(2026, 1, 1), self._profile()) == pytest.approx(0.70)

    def test_missing_day_type_defaults(self):
        profile = {
            "day_period_shares": {},
            "day_factors": {},
        }
        assert get_day_factor(date(2026, 4, 20), profile) == 1.0


# ---------------------------------------------------------------------------
# get_combined_factor
# ---------------------------------------------------------------------------
class TestGetCombinedFactor:
    @staticmethod
    def _profile():
        return {
            "day_period_shares": {"day": 0.785, "evening": 0.137, "night": 0.078},
            "day_factors": {"workday": 1.07, "saturday": 0.90, "sunday": 0.75, "holiday": 0.70},
        }

    def test_product(self):
        result = get_combined_factor(date(2026, 4, 20), "day", self._profile())
        expected = 1.07 * 0.785
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_demand_period_shares
# ---------------------------------------------------------------------------
class TestGetDemandPeriodShares:
    def test_default_profile_sums_to_one(self):
        profile = {
            "day_period_shares": {"day": 0.785, "evening": 0.137, "night": 0.078},
        }
        shares = get_demand_period_shares(profile)
        total = shares["am"] + shares["ip"] + shares["pm"] + shares["ev"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_custom_policy(self):
        profile = {
            "day_period_shares": {"day": 0.8, "evening": 0.12, "night": 0.08},
            "demand_period_split_policy": {"am": 1.0, "ip": 0.0, "pm": 0.0},
        }
        shares = get_demand_period_shares(profile)
        assert shares["am"] == pytest.approx(0.8)
        assert shares["ip"] == pytest.approx(0.0)
        assert shares["pm"] == pytest.approx(0.0)

    def test_ev_is_evening_plus_night(self):
        profile = {
            "day_period_shares": {"day": 0.7, "evening": 0.2, "night": 0.1},
        }
        shares = get_demand_period_shares(profile)
        assert shares["ev"] == pytest.approx(0.3, abs=0.001)

    def test_daily_is_one(self):
        profile = {"day_period_shares": {"day": 0.8, "evening": 0.1, "night": 0.1}}
        shares = get_demand_period_shares(profile)
        assert shares["daily"] == 1.0
