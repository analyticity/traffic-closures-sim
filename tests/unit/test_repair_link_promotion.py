"""Unit tests for promoting structurally-required divided-highway repair links.

The repair connectors are deliberately given 20 km/h and 200 veh/h so the
assignment avoids them.  When one is the *only* directed path to a group of
nodes the assignment must use it anyway, and BPR then turns the resulting V/C
into an absurd congested time -- on Brno one such link reached V/C 14.6 and
124 h, producing 96 % of the whole network's VHT.

``_promote_structural_repair_links`` separates the two cases by a structural
test (does the largest SCC lose nodes without this link?), not by a volume
threshold.  These tests pin that behaviour down without needing AequilibraE.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd
import pytest

from sim.network.connectivity import _build_digraph, _promote_structural_repair_links


class _FakeLinks:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df


class _FakeNetwork:
    def __init__(self, df: pd.DataFrame) -> None:
        self.links = _FakeLinks(df)


class _FakeConn:
    """Records the UPDATE statements the promotion would issue."""

    def __init__(self, sink: List[tuple]) -> None:
        self._sink = sink

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._sink.append((sql, params))


class _FakeProject:
    def __init__(self, df: pd.DataFrame) -> None:
        self.network = _FakeNetwork(df)


def _links(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _run(df: pd.DataFrame, new_ids, dead_cap, dead_lanes):
    """Call the promotion with the DB layer stubbed out."""
    executed: List[tuple] = []

    class _Ctx:
        def __enter__(self):
            return _FakeConn(executed)

        def __exit__(self, *exc):
            return False

    with patch("sim.network.connectivity.project_db", lambda _p: _Ctx()), \
         patch("sim.network.connectivity.refresh_network", lambda _p: None):
        promoted = _promote_structural_repair_links(
            _FakeProject(df), new_ids, dead_cap, dead_lanes,
        )
    return promoted, executed


# --- structural test itself -------------------------------------------------

def test_promotes_link_that_strands_a_zone():
    """A zone's connector sits behind link 99, so its demand must cross it."""
    df = _links([
        # a ring large enough that it is unambiguously the main component
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 0, "link_type": "motorway"},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 0, "link_type": "motorway"},
        {"link_id": 3, "a_node": 3, "b_node": 4, "direction": 0, "link_type": "motorway"},
        {"link_id": 6, "a_node": 4, "b_node": 5, "direction": 0, "link_type": "motorway"},
        {"link_id": 7, "a_node": 5, "b_node": 1, "direction": 0, "link_type": "motorway"},
        # a stub hanging off node 3, reachable only via link 99
        {"link_id": 99, "a_node": 3, "b_node": 10, "direction": 0, "link_type": "motorway"},
        {"link_id": 4, "a_node": 10, "b_node": 11, "direction": 0, "link_type": "motorway"},
        # a zone loads its demand onto node 11
        {"link_id": 5, "a_node": 900, "b_node": 11, "direction": 0,
         "link_type": "centroid_connector"},
    ])
    promoted, executed = _run(df, [99], {3: 4400.0, 10: 6600.0}, {3: 2, 10: 3})

    assert [p["link_id"] for p in promoted] == [99]
    assert promoted[0]["nodes_depending"] == 3      # nodes 10, 11 and the centroid
    # both endpoints of the connector (the centroid and the network node it
    # attaches to) fall behind the link, so the count is 2
    assert promoted[0]["stranded_connector_nodes"] == 2
    assert promoted[0]["capacity"] == 4400.0        # weakest adjacent carriageway
    assert promoted[0]["lanes"] == 2
    assert len(executed) == 1


def test_leaves_a_bypassable_crossover_alone():
    """A crossover with an alternative route keeps its penalty."""
    df = _links([
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 0, "link_type": "motorway"},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 0, "link_type": "motorway"},
        {"link_id": 3, "a_node": 3, "b_node": 1, "direction": 0, "link_type": "motorway"},
        # parallel to link 2 — removing it splits nothing
        {"link_id": 99, "a_node": 2, "b_node": 3, "direction": 0, "link_type": "motorway"},
    ])
    promoted, executed = _run(df, [99], {2: 4400.0, 3: 4400.0}, {2: 2, 3: 2})

    assert promoted == []
    assert executed == []


def test_one_way_trap_counts_as_structural():
    """An undirected alternative that is one-way the wrong way is no alternative."""
    df = _links([
        # a ring that stays connected on its own
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 0, "link_type": "trunk"},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 0, "link_type": "trunk"},
        {"link_id": 3, "a_node": 3, "b_node": 1, "direction": 0, "link_type": "trunk"},
        # 3 -> 7 only: without a return leg node 7 falls out of the SCC
        {"link_id": 6, "a_node": 3, "b_node": 7, "direction": 1, "link_type": "trunk"},
        # the repair link provides that return leg
        {"link_id": 99, "a_node": 7, "b_node": 1, "direction": 0, "link_type": "trunk"},
        {"link_id": 5, "a_node": 900, "b_node": 7, "direction": 0,
         "link_type": "centroid_connector"},
    ])
    promoted, _ = _run(df, [99], {7: 3600.0, 1: 3600.0}, {7: 2, 1: 2})

    assert [p["link_id"] for p in promoted] == [99]


def test_stub_without_a_zone_keeps_its_penalty():
    """The regression this criterion exists for.

    Link 99 is structurally required in the SCC sense, but nothing behind it
    generates or attracts trips.  Promoting it does not unblock any demand --
    it only opens a shortcut.  Measured on the Brno build: promoting such a
    link (117203) sent one screenline to 314 % and dropped holdout R^2 from
    0.725 to 0.625.
    """
    df = _links([
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 0, "link_type": "motorway"},
        {"link_id": 2, "a_node": 2, "b_node": 3, "direction": 0, "link_type": "motorway"},
        {"link_id": 3, "a_node": 3, "b_node": 1, "direction": 0, "link_type": "motorway"},
        {"link_id": 99, "a_node": 3, "b_node": 10, "direction": 0, "link_type": "motorway"},
        {"link_id": 4, "a_node": 10, "b_node": 11, "direction": 0, "link_type": "motorway"},
        # the only connector is on the main ring, not behind link 99
        {"link_id": 5, "a_node": 900, "b_node": 1, "direction": 0,
         "link_type": "centroid_connector"},
    ])
    promoted, executed = _run(df, [99], {3: 4400.0, 10: 4400.0}, {3: 2, 10: 2})

    assert promoted == []
    assert executed == []


# --- guard rails ------------------------------------------------------------

def test_no_new_links_is_a_noop():
    df = _links([{"link_id": 1, "a_node": 1, "b_node": 2, "direction": 0,
                  "link_type": "motorway"}])
    promoted, executed = _run(df, [], {}, {})
    assert promoted == []
    assert executed == []


def test_structural_link_without_known_capacity_is_reported_not_guessed():
    """No adjacent mainline capacity -> leave it penalised and warn, never invent."""
    df = _links([
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 0, "link_type": "motorway"},
        {"link_id": 99, "a_node": 2, "b_node": 10, "direction": 0, "link_type": "motorway"},
        {"link_id": 5, "a_node": 900, "b_node": 10, "direction": 0,
         "link_type": "centroid_connector"},
    ])
    promoted, executed = _run(df, [99], {}, {})
    assert promoted == []
    assert executed == []


def test_empty_network_is_a_noop():
    promoted, executed = _run(pd.DataFrame(), [99], {}, {})
    assert promoted == []
    assert executed == []


# --- the helper the test relies on -----------------------------------------

def test_build_digraph_respects_direction():
    df = _links([
        {"link_id": 1, "a_node": 1, "b_node": 2, "direction": 1, "link_type": "motorway"},
        {"link_id": 2, "a_node": 3, "b_node": 4, "direction": -1, "link_type": "motorway"},
        {"link_id": 3, "a_node": 5, "b_node": 6, "direction": 0, "link_type": "motorway"},
    ])
    g = _build_digraph(df)
    assert g.has_edge(1, 2) and not g.has_edge(2, 1)
    assert g.has_edge(4, 3) and not g.has_edge(3, 4)
    assert g.has_edge(5, 6) and g.has_edge(6, 5)
