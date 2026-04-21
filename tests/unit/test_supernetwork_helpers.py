"""Unit tests for pure helpers in sim.supernetwork."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString

from sim.supernetwork import (
    _edge_speed_kmh,
    _highway_default_speed_kmh,
    _iter_lines,
    _meters_to_seconds,
    _norm_obec_code,
    _parse_numeric,
    add_or_relax_edge,
    contract_graph,
    is_contractible,
)


# ---------------------------------------------------------------------------
# _norm_obec_code
# ---------------------------------------------------------------------------
class TestNormObecCode:
    def test_none(self):
        assert _norm_obec_code(None) == ""

    def test_nan(self):
        assert _norm_obec_code(float("nan")) == ""

    def test_numpy_nan(self):
        assert _norm_obec_code(np.nan) == ""

    def test_zero(self):
        assert _norm_obec_code(0) == "0"

    def test_numpy_zero(self):
        assert _norm_obec_code(np.int64(0)) == ""

    def test_empty_string(self):
        assert _norm_obec_code("") == ""

    def test_none_string(self):
        assert _norm_obec_code("none") == ""

    def test_float_with_decimal(self):
        assert _norm_obec_code("123.0") == "123"

    def test_normal_int(self):
        assert _norm_obec_code(582786) == "582786"

    def test_numpy_int(self):
        assert _norm_obec_code(np.int64(582786)) == "582786"

    def test_numpy_float(self):
        assert _norm_obec_code(np.float64(582786.0)) == "582786"


# ---------------------------------------------------------------------------
# _parse_numeric
# ---------------------------------------------------------------------------
class TestParseNumeric:
    def test_none(self):
        assert _parse_numeric(None) is None

    def test_int(self):
        assert _parse_numeric(50) == 50.0

    def test_float(self):
        assert _parse_numeric(80.5) == 80.5

    def test_string_number(self):
        assert _parse_numeric("90") == 90.0

    def test_string_with_unit(self):
        result = _parse_numeric("50 mph")
        assert result is not None
        assert result == pytest.approx(50 * 1.60934, rel=1e-3)

    def test_no_number(self):
        assert _parse_numeric("no speed") is None

    def test_empty(self):
        assert _parse_numeric("") is None

    def test_comma_decimal(self):
        assert _parse_numeric("3,14") == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# _highway_default_speed_kmh
# ---------------------------------------------------------------------------
class TestHighwayDefaultSpeed:
    def test_motorway(self):
        assert _highway_default_speed_kmh("motorway") == 130.0

    def test_secondary(self):
        assert _highway_default_speed_kmh("secondary") == 55.0

    def test_unknown(self):
        assert _highway_default_speed_kmh("service") == 60.0

    def test_none(self):
        assert _highway_default_speed_kmh(None) == 60.0


# ---------------------------------------------------------------------------
# _edge_speed_kmh
# ---------------------------------------------------------------------------
class TestEdgeSpeedKmh:
    def test_valid_maxspeed(self):
        assert _edge_speed_kmh("80", "primary") == 80.0

    def test_none_maxspeed_uses_highway_default(self):
        assert _edge_speed_kmh(None, "motorway") == 130.0

    def test_mph_conversion(self):
        result = _edge_speed_kmh("50 mph", "primary")
        assert result == pytest.approx(50 * 1.60934, rel=1e-3)


# ---------------------------------------------------------------------------
# _meters_to_seconds
# ---------------------------------------------------------------------------
class TestMetersToSeconds:
    def test_basic(self):
        assert _meters_to_seconds(1000, 36) == pytest.approx(100.0)

    def test_near_zero_speed(self):
        result = _meters_to_seconds(100, 0)
        assert result > 0 and math.isfinite(result)

    def test_fast(self):
        assert _meters_to_seconds(1000, 130) == pytest.approx(1000 / (130 / 3.6), rel=1e-6)


# ---------------------------------------------------------------------------
# _iter_lines
# ---------------------------------------------------------------------------
class TestIterLines:
    def test_none(self):
        assert list(_iter_lines(None)) == []

    def test_linestring(self):
        ls = LineString([(0, 0), (1, 1)])
        result = list(_iter_lines(ls))
        assert len(result) == 1
        assert result[0].equals(ls)

    def test_multilinestring(self):
        ml = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        result = list(_iter_lines(ml))
        assert len(result) == 2

    def test_empty_linestring(self):
        ls = LineString()
        assert list(_iter_lines(ls)) == []


# ---------------------------------------------------------------------------
# add_or_relax_edge
# ---------------------------------------------------------------------------
class TestAddOrRelaxEdge:
    def test_add_new(self):
        G = nx.DiGraph()
        add_or_relax_edge(G, 1, 2, 100.0, 10.0, "primary", "Main", "I/43")
        assert G.has_edge(1, 2)
        assert G[1][2]["travel_time_s"] == 10.0

    def test_relax_shorter(self):
        G = nx.DiGraph()
        add_or_relax_edge(G, 1, 2, 100.0, 10.0, "primary", "Main", "I/43")
        add_or_relax_edge(G, 1, 2, 50.0, 5.0, "primary", "Main", "I/43")
        assert G[1][2]["travel_time_s"] == 5.0
        assert G[1][2]["length_m"] == 50.0

    def test_ignore_longer(self):
        G = nx.DiGraph()
        add_or_relax_edge(G, 1, 2, 100.0, 10.0, "primary", "Main", "I/43")
        add_or_relax_edge(G, 1, 2, 200.0, 20.0, "primary", "Main", "I/43")
        assert G[1][2]["travel_time_s"] == 10.0

    def test_self_loop_ignored(self):
        G = nx.DiGraph()
        add_or_relax_edge(G, 1, 1, 100.0, 10.0, "primary", "Main", "I/43")
        assert not G.has_edge(1, 1)


# ---------------------------------------------------------------------------
# is_contractible + contract_graph
# ---------------------------------------------------------------------------
class TestGraphContraction:
    @staticmethod
    def _line_graph() -> nx.DiGraph:
        """1 -> 2 -> 3 -> 4 (chain)."""
        G = nx.DiGraph()
        for u, v in [(1, 2), (2, 3), (3, 4)]:
            G.add_edge(u, v, travel_time_s=10.0, length_m=100.0,
                       highway="primary", name="", ref="")
            G.add_edge(v, u, travel_time_s=10.0, length_m=100.0,
                       highway="primary", name="", ref="")
        return G

    def test_contractible_middle_nodes(self):
        G = self._line_graph()
        protected = {1, 4}
        assert is_contractible(G, 2, protected, degree_threshold=2)
        assert is_contractible(G, 3, protected, degree_threshold=2)

    def test_protected_not_contractible(self):
        G = self._line_graph()
        assert not is_contractible(G, 1, {1, 4}, degree_threshold=2)

    def test_threshold_too_low(self):
        G = self._line_graph()
        assert not is_contractible(G, 2, {1, 4}, degree_threshold=1)

    def test_contract_removes_middle(self):
        G = self._line_graph()
        contracted = contract_graph(G, protected={1, 4}, degree_threshold=2)
        assert 1 in contracted
        assert 4 in contracted
        assert 2 not in contracted
        assert 3 not in contracted
        assert contracted.has_edge(1, 4) or contracted.has_edge(4, 1)

    def test_contract_preserves_protected(self):
        G = self._line_graph()
        contracted = contract_graph(G, protected={1, 2, 3, 4}, degree_threshold=2)
        assert set(contracted.nodes()) == {1, 2, 3, 4}
