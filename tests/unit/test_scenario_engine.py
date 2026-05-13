"""Unit tests for scenario graph capacity edits."""
from __future__ import annotations

import pandas as pd

from sim.scenarios.engine import CLOSURE_CAPACITY, apply_scenario_to_graph


class _FakeGraph:
    def __init__(self, gdf: pd.DataFrame):
        self.graph = gdf


def test_lane_reduction_zero_remaining_triggers_full_closure_capacity():
    """lanes_remaining=0 must not clamp to 1 — otherwise partial severities collapse."""
    gdf = pd.DataFrame(
        {
            "link_id": [1, 1],
            "direction": [1, -1],
            "capacity": [2000.0, 2000.0],
            "free_flow_time": [60.0, 60.0],
        }
    )
    graph = _FakeGraph(gdf)
    scenario = [
        {
            "link_id": 1,
            "direction": "both",
            "closure_type": "lanes",
            "lanes": 2,
            "lanes_remaining": 0,
        }
    ]
    apply_scenario_to_graph(graph, scenario)
    assert float(gdf.loc[0, "capacity"]) == CLOSURE_CAPACITY
    assert float(gdf.loc[1, "capacity"]) == CLOSURE_CAPACITY


def test_lane_reduction_partial_ratio():
    gdf = pd.DataFrame(
        {
            "link_id": [7],
            "direction": [1],
            "capacity": [1000.0],
            "free_flow_time": [40.0],
        }
    )
    graph = _FakeGraph(gdf)
    scenario = [
        {
            "link_id": 7,
            "direction": "both",
            "closure_type": "lanes",
            "lanes": 4,
            "lanes_remaining": 2,
        }
    ]
    apply_scenario_to_graph(graph, scenario)
    assert abs(float(gdf.loc[0, "capacity"]) - 500.0) < 1e-6
