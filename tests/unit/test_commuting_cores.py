"""Unit tests for the commuting cores built from the SLDB 2021 census.

Covers the conversion of census persons to vehicle trips (two legs a day, not
four), the census workplace categories that count as commuting, and the split
of the core city's trip ends among its zones (home ends by population, work
ends by employment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim.defaults import SIM_DEFAULTS
from sim.demand.commuting_io import _filter_commuting
from sim.demand.config import _build_cfg
from sim.demand.od_builder import _build_od_cores

# Zones 1 and 2 are municipalities, 10 and 11 the two districts of the core city.
ZONES = np.array([1, 2, 10, 11], dtype=np.int64)
PRIMARY = {"kurim": 1, "sokolnice": 2}
HOME = {"brno": [(10, 0.5), (11, 0.5)]}
WORK = {"brno": [(10, 0.9), (11, 0.1)]}
KURIM, SOKOLNICE, DISTRICT_A, DISTRICT_B = range(4)


def _cfg(tmp_path) -> dict:
    return {
        "datasets": {"cache_dir": str(tmp_path)},
        "demand": {
            "conversion": {
                "work": {"car_share": 0.48, "occupancy": 1.2, "trips_per_person": 2.0},
                "school": {"car_share": 0.25, "occupancy": 1.3, "trips_per_person": 2.0},
            },
        },
    }


def _rows(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["lokalizace", "op_obec", "doj_obec", "dojizdka_prace", "dojizdka_skola"],
    )


def _daily(df, bcfg, groups_work=None) -> np.ndarray:
    mats, _ = _build_od_cores(
        df,
        ZONES,
        bcfg,
        primary=PRIMARY,
        stripped={},
        groups=HOME,
        gateways={},
        external_lookup={},
        groups_work=groups_work,
    )
    return mats["wd_daily"]


# ---------------------------------------------------------------------------
# Persons to vehicle trips
# ---------------------------------------------------------------------------
class TestLegsOfTheDay:
    def test_worker_makes_two_vehicle_legs_not_four(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        daily = _daily(_rows(("1_meziobecni", "Kuřim", "Sokolnice", 100, 0)), bcfg)
        # 100 workers x 2 trips x car share 0.48 / occupancy 1.2
        assert daily.sum() == pytest.approx(80.0)
        assert daily[KURIM, SOKOLNICE] == pytest.approx(40.0)
        assert daily[SOKOLNICE, KURIM] == pytest.approx(40.0)

    def test_pupil_makes_two_vehicle_legs(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        daily = _daily(_rows(("1_meziobecni", "Kuřim", "Sokolnice", 0, 130)), bcfg)
        assert daily.sum() == pytest.approx(130 * 2 * 0.25 / 1.3)


# ---------------------------------------------------------------------------
# Census workplace categories
# ---------------------------------------------------------------------------
class TestCensusCategories:
    def test_default_counts_commuters_not_home_workers(self):
        cats = SIM_DEFAULTS["demand"]["sldb"]["include_lokalizace"]
        assert set(cats) == {"1_meziobecni", "3_v_ramci_obce"}

    def test_fallback_without_config_matches_default(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        assert set(bcfg.include_lokalizace) == {"1_meziobecni", "3_v_ramci_obce"}

    def test_filter_keeps_intra_municipal_and_drops_home_workers(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        df = _rows(
            ("0_na_adrese_OP", "Brno", "Brno", 50, 0),
            ("3_v_ramci_obce", "Brno", "Brno", 200, 0),
            ("1_meziobecni", "Kuřim", "Brno", 30, 0),
            ("4_bez_staleho_mista", "Brno", "Brno", 10, 0),
        )
        kept = _filter_commuting(df, bcfg)
        assert sorted(kept["lokalizace"]) == ["1_meziobecni", "3_v_ramci_obce"]


# ---------------------------------------------------------------------------
# Trip ends of the core city
# ---------------------------------------------------------------------------
class TestCoreCityTripEnds:
    def test_work_end_follows_employment(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        daily = _daily(_rows(("1_meziobecni", "Kuřim", "Brno", 100, 0)), bcfg, groups_work=WORK)
        assert daily[KURIM, DISTRICT_A] == pytest.approx(36.0)
        assert daily[KURIM, DISTRICT_B] == pytest.approx(4.0)
        # the return leg leaves from the same workplaces
        assert daily[DISTRICT_A, KURIM] == pytest.approx(36.0)
        assert daily[DISTRICT_B, KURIM] == pytest.approx(4.0)

    def test_home_end_follows_population(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        daily = _daily(_rows(("1_meziobecni", "Brno", "Kuřim", 100, 0)), bcfg, groups_work=WORK)
        assert daily[DISTRICT_A, KURIM] == pytest.approx(20.0)
        assert daily[DISTRICT_B, KURIM] == pytest.approx(20.0)

    def test_without_work_groups_both_ends_follow_population(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        daily = _daily(_rows(("1_meziobecni", "Kuřim", "Brno", 100, 0)), bcfg)
        assert daily[KURIM, DISTRICT_A] == pytest.approx(20.0)
        assert daily[KURIM, DISTRICT_B] == pytest.approx(20.0)

    def test_intra_city_commuting_crosses_home_and_work_weights(self, tmp_path):
        bcfg = _build_cfg(_cfg(tmp_path))
        daily = _daily(_rows(("3_v_ramci_obce", "Brno", "Brno", 100, 0)), bcfg, groups_work=WORK)
        # outbound 40 legs: home 50/50 x work 90/10; return the other way
        assert daily[DISTRICT_B, DISTRICT_A] == pytest.approx(40 * 0.5 * 0.9 + 40 * 0.1 * 0.5)
        assert daily.sum() == pytest.approx(80.0)
