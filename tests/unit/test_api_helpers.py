"""Unit tests for pure helpers in sim.api."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError
from shapely.geometry import LineString

from sim.api import (
    ScenarioLinkInput,
    ScenarioRunRequest,
    _edge_coords,
    _estimate_intersection_delays,
    _sanitize_nan,
)


# ---------------------------------------------------------------------------
# _sanitize_nan
# ---------------------------------------------------------------------------
class TestSanitizeNan:
    def test_nan_becomes_none(self):
        assert _sanitize_nan(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _sanitize_nan(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _sanitize_nan(float("-inf")) is None

    def test_normal_float_unchanged(self):
        assert _sanitize_nan(3.14) == 3.14

    def test_nested_dict(self):
        result = _sanitize_nan({"a": float("nan"), "b": 1.0})
        assert result == {"a": None, "b": 1.0}

    def test_nested_list(self):
        result = _sanitize_nan([float("inf"), 2.0, "text"])
        assert result == [None, 2.0, "text"]

    def test_string_unchanged(self):
        assert _sanitize_nan("hello") == "hello"

    def test_int_unchanged(self):
        assert _sanitize_nan(42) == 42

    def test_deeply_nested(self):
        data = {"outer": [{"inner": float("nan")}]}
        result = _sanitize_nan(data)
        assert result == {"outer": [{"inner": None}]}


# ---------------------------------------------------------------------------
# _estimate_intersection_delays
# ---------------------------------------------------------------------------
class TestEstimateIntersectionDelays:
    def test_short_path_no_delay(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, link_type="primary")
        assert _estimate_intersection_delays(G, [1, 2]) == 0.0

    def test_degree_2_no_delay(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, link_type="primary")
        G.add_edge(2, 3, link_type="primary")
        assert _estimate_intersection_delays(G, [1, 2, 3]) == 0.0

    def test_signalized_intersection(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, link_type="primary")
        G.add_edge(2, 3, link_type="primary")
        G.add_edge(4, 2, link_type="secondary")
        assert _estimate_intersection_delays(G, [1, 2, 3]) == 20.0

    def test_minor_intersection(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, link_type="residential")
        G.add_edge(2, 3, link_type="residential")
        G.add_edge(4, 2, link_type="residential")
        assert _estimate_intersection_delays(G, [1, 2, 3]) == 6.0

    def test_motorway_through_skipped(self):
        G = nx.DiGraph()
        G.add_edge(1, 2, link_type="motorway")
        G.add_edge(2, 3, link_type="motorway")
        G.add_edge(4, 2, link_type="motorway")
        assert _estimate_intersection_delays(G, [1, 2, 3]) == 0.0

    def test_fallback_delay(self):
        """Non-signalized, non-minor types get the fallback 2s delay."""
        G = nx.DiGraph()
        G.add_edge(1, 2, link_type="cycleway")
        G.add_edge(2, 3, link_type="cycleway")
        G.add_edge(4, 2, link_type="cycleway")
        assert _estimate_intersection_delays(G, [1, 2, 3]) == 2.0


# ---------------------------------------------------------------------------
# _edge_coords
# ---------------------------------------------------------------------------
class TestEdgeCoords:
    def test_none_geometry(self):
        assert _edge_coords({}) == []

    def test_empty_geometry(self):
        assert _edge_coords({"geom": LineString()}) == []

    def test_forward(self):
        geom = LineString([(0, 0), (1, 1), (2, 2)])
        coords = _edge_coords({"geom": geom})
        assert coords[0] == (0, 0)
        assert coords[-1] == (2, 2)

    def test_reverse_geom(self):
        geom = LineString([(0, 0), (1, 1), (2, 2)])
        coords = _edge_coords({"geom": geom, "reverse_geom": True})
        assert coords[0] == (2, 2)
        assert coords[-1] == (0, 0)

    def test_prev_end_flips(self):
        geom = LineString([(0, 0), (10, 10)])
        coords = _edge_coords({"geom": geom}, prev_end=(10, 10))
        assert coords[0] == (10, 10)
        assert coords[-1] == (0, 0)

    def test_prev_end_keeps(self):
        geom = LineString([(0, 0), (10, 10)])
        coords = _edge_coords({"geom": geom}, prev_end=(0, 0))
        assert coords[0] == (0, 0)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TestScenarioLinkInput:
    def test_valid(self):
        link = ScenarioLinkInput(link_id=1, lanes=3, lanes_remaining=1)
        assert link.lanes_remaining == 1

    def test_lanes_remaining_exceeds_lanes(self):
        with pytest.raises(ValidationError, match="lanes_remaining"):
            ScenarioLinkInput(link_id=1, lanes=2, lanes_remaining=3)


class TestScenarioRunRequest:
    def test_deduplication(self):
        links = [
            ScenarioLinkInput(link_id=1, direction="ab"),
            ScenarioLinkInput(link_id=1, direction="ab"),
            ScenarioLinkInput(link_id=2, direction="both"),
        ]
        req = ScenarioRunRequest(links=links)
        assert len(req.links) == 2

    def test_different_directions_kept(self):
        links = [
            ScenarioLinkInput(link_id=1, direction="ab"),
            ScenarioLinkInput(link_id=1, direction="ba"),
        ]
        req = ScenarioRunRequest(links=links)
        assert len(req.links) == 2
