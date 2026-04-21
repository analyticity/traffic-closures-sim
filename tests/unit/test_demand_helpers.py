"""Unit tests for pure helpers in sim.demand."""
from __future__ import annotations

from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from sim.demand import (
    _build_zone_name_index,
    _match_zone_id,
    _normalize_named_weights,
    _normalize_period_shares,
    _parse_gateway_pair_weights,
    _strip_geo_suffix,
    _validate_shares,
)


# ---------------------------------------------------------------------------
# _strip_geo_suffix
# ---------------------------------------------------------------------------
class TestStripGeoSuffix:
    def test_no_suffix(self):
        assert _strip_geo_suffix("brno") == "brno"

    def test_with_u(self):
        assert _strip_geo_suffix("kurim u brna") == "kurim"

    def test_with_nad(self):
        assert _strip_geo_suffix("blansko nad svitavou") == "blansko"

    def test_empty(self):
        assert _strip_geo_suffix("") == ""


# ---------------------------------------------------------------------------
# _normalize_period_shares
# ---------------------------------------------------------------------------
class TestNormalizePeriodShares:
    def test_empty_gives_uniform(self):
        result = _normalize_period_shares(None, ["am", "pm"])
        assert result == pytest.approx({"am": 0.5, "pm": 0.5})

    def test_zeros_give_uniform(self):
        result = _normalize_period_shares({"am": 0, "pm": 0}, ["am", "pm"])
        assert result == pytest.approx({"am": 0.5, "pm": 0.5})

    def test_normal_values(self):
        result = _normalize_period_shares({"am": 3, "pm": 1}, ["am", "pm"])
        assert result == pytest.approx({"am": 0.75, "pm": 0.25})

    def test_sums_to_one(self):
        result = _normalize_period_shares({"a": 10, "b": 20, "c": 30}, ["a", "b", "c"])
        assert sum(result.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _validate_shares (uses PeriodShares dataclass)
# ---------------------------------------------------------------------------
class TestValidateShares:
    @staticmethod
    def _make_shares(outbound: dict, return_: dict):
        from types import SimpleNamespace
        return SimpleNamespace(outbound=outbound, return_=return_)

    def test_valid(self):
        shares = self._make_shares({"am": 0.6, "pm": 0.4}, {"am": 0.5, "pm": 0.5})
        _validate_shares(["am", "pm"], shares, "test")

    def test_unknown_period_raises(self):
        shares = self._make_shares({"am": 0.5, "night": 0.5}, {"am": 1.0})
        with pytest.raises(ValueError, match="unknown period"):
            _validate_shares(["am", "pm"], shares, "test")

    def test_wrong_sum_raises(self):
        shares = self._make_shares({"am": 0.6, "pm": 0.6}, {"am": 0.5, "pm": 0.5})
        with pytest.raises(ValueError, match="sum to 1.0"):
            _validate_shares(["am", "pm"], shares, "test")


# ---------------------------------------------------------------------------
# _normalize_named_weights
# ---------------------------------------------------------------------------
class TestNormalizeNamedWeights:
    def test_empty_names(self):
        assert _normalize_named_weights([], None) == {}

    def test_equal_split_no_raw(self):
        result = _normalize_named_weights(["D1", "D2"], None)
        assert result == pytest.approx({"D1": 0.5, "D2": 0.5})

    def test_link_type_fallback(self):
        result = _normalize_named_weights(
            ["D1", "I43"],
            None,
            gateway_link_types={"D1": "motorway", "I43": "primary"},
        )
        assert result["D1"] > result["I43"]
        assert sum(result.values()) == pytest.approx(1.0)

    def test_explicit_weights(self):
        result = _normalize_named_weights(["A", "B"], {"A": 3.0, "B": 1.0})
        assert result == pytest.approx({"A": 0.75, "B": 0.25})


# ---------------------------------------------------------------------------
# _parse_gateway_pair_weights
# ---------------------------------------------------------------------------
class TestParseGatewayPairWeights:
    def test_no_pairs_uniform(self):
        result = _parse_gateway_pair_weights(None, ["D1", "D2"])
        assert ("D1", "D2") in result
        assert ("D2", "D1") in result
        assert ("D1", "D1") not in result

    def test_explicit_pair(self):
        pairs = [{"from": "D1", "to": "D2", "weight": 5.0}]
        result = _parse_gateway_pair_weights(pairs, ["D1", "D2"])
        assert result[("D1", "D2")] == 5.0
        assert ("D2", "D1") not in result

    def test_bidirectional(self):
        pairs = [{"from": "D1", "to": "D2", "weight": 3.0, "bidirectional": True}]
        result = _parse_gateway_pair_weights(pairs, ["D1", "D2"])
        assert result[("D1", "D2")] == 3.0
        assert result[("D2", "D1")] == 3.0

    def test_base_plus_overwrite(self):
        pairs = [{"from": "D1", "to": "D2", "weight": 10.0}]
        result = _parse_gateway_pair_weights(pairs, ["D1", "D2", "I43"], default_pair_weight=1.0)
        assert result[("D1", "D2")] == 10.0
        assert result[("D1", "I43")] == 1.0


# ---------------------------------------------------------------------------
# _build_zone_name_index + _match_zone_id
# ---------------------------------------------------------------------------
class TestZoneNameMatching:
    @staticmethod
    def _zones_gdf() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({
            "zone_id": [100, 200, 300],
            "name": ["Brno", "Kuřim u Brna", "Blansko"],
            "geometry": [Point(16.6, 49.2), Point(16.5, 49.3), Point(16.7, 49.4)],
        })

    def test_exact_match(self):
        gdf = self._zones_gdf()
        primary, stripped = _build_zone_name_index(gdf)
        assert _match_zone_id("Brno", primary, stripped) == 100

    def test_stripped_match(self):
        gdf = self._zones_gdf()
        primary, stripped = _build_zone_name_index(gdf)
        result = _match_zone_id("Kuřim", primary, stripped)
        assert result == 200

    def test_no_match(self):
        gdf = self._zones_gdf()
        primary, stripped = _build_zone_name_index(gdf)
        assert _match_zone_id("Olomouc", primary, stripped) is None

    def test_empty_name(self):
        gdf = self._zones_gdf()
        primary, stripped = _build_zone_name_index(gdf)
        assert _match_zone_id("", primary, stripped) is None
