"""Network connectivity analysis and repair.

Provides strongly-connected-component boundary repair, divided-highway
dead-end connection, and a general connectivity check.
"""
from __future__ import annotations

import logging
import struct
from collections import defaultdict
from typing import Any, Dict

import networkx as nx
import pandas as pd
from aequilibrae import Project

from sim.network.db import project_db, refresh_network
from sim.network.normalization import ensure_link_types_registered

logger = logging.getLogger(__name__)

_MAJOR_ROAD_TYPES = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})
# Only ramp/link types are eligible for SCC bidirectionalization — mainline
# carriageways (motorway, trunk) must never be reversed as that creates
# physically impossible routes.
_SCC_REPAIR_ELIGIBLE = frozenset({"motorway_link", "trunk_link"})


# ---------------------------------------------------------------------------
# SCC boundary repair
# ---------------------------------------------------------------------------

def repair_boundary_scc(
    project: Project,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Make one-way motorway/trunk links with nodes outside the largest directed
    SCC bidirectional, so that gateway nodes can participate in directed routing.

    When *dry_run* is ``True``, compute which links would be repaired and
    return the report **without** modifying the database.
    """
    links = project.network.links.data.copy()
    _empty = {"repaired": 0, "scc_before": 0, "scc_after": 0,
              "major_outside_before": 0, "major_outside_after": 0,
              "repaired_ids": [], "dry_run": dry_run}
    if links.empty:
        return _empty

    G = _build_digraph(links)
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return _empty

    largest_scc = max(sccs, key=len)
    scc_before = len(largest_scc)

    major_outside_before = _count_major_links_outside_scc(links, largest_scc)

    repair_ids = []
    for _, lk in links.iterrows():
        lt = str(lk.get("link_type", ""))
        if lt not in _SCC_REPAIR_ELIGIBLE:
            continue
        d = int(lk.get("direction", 0) or 0)
        if d == 0:
            continue
        a, b = int(lk["a_node"]), int(lk["b_node"])
        if a in largest_scc and b in largest_scc:
            continue
        repair_ids.append(int(lk["link_id"]))

    if not repair_ids:
        logger.info("Boundary SCC repair: nothing to fix (SCC=%d)", scc_before)
        return {**_empty, "scc_before": scc_before, "scc_after": scc_before,
                "major_outside_before": major_outside_before,
                "major_outside_after": major_outside_before}

    if dry_run:
        logger.info(
            "Boundary SCC repair (DRY RUN): would make %d one-way major links "
            "bidirectional (SCC=%d, major outside=%d)",
            len(repair_ids), scc_before, major_outside_before,
        )
        return {
            "repaired": len(repair_ids),
            "scc_before": scc_before, "scc_after": scc_before,
            "major_outside_before": major_outside_before,
            "major_outside_after": major_outside_before,
            "repaired_ids": repair_ids,
            "dry_run": True,
        }

    with project_db(project) as conn:
        for lid in repair_ids:
            conn.execute(
                """UPDATE links
                   SET direction = 0,
                       speed_ba     = speed_ab,
                       capacity_ba  = capacity_ab,
                       lanes_ba     = lanes_ab,
                       travel_time_ba = travel_time_ab
                 WHERE link_id = ? AND direction != 0""",
                (lid,),
            )

    refresh_network(project)

    links2 = project.network.links.data
    G2 = _build_digraph(links2)
    sccs2 = list(nx.strongly_connected_components(G2))
    largest_scc2 = max(sccs2, key=len) if sccs2 else set()
    scc_after = len(largest_scc2)
    major_outside_after = _count_major_links_outside_scc(links2, largest_scc2)

    logger.info(
        "Boundary SCC repair: made %d one-way major links bidirectional; "
        "SCC %d -> %d nodes (+%d); major links outside SCC %d -> %d",
        len(repair_ids), scc_before, scc_after, scc_after - scc_before,
        major_outside_before, major_outside_after,
    )
    return {
        "repaired": len(repair_ids),
        "scc_before": scc_before, "scc_after": scc_after,
        "major_outside_before": major_outside_before,
        "major_outside_after": major_outside_after,
        "repaired_ids": repair_ids,
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# Divided highway dead-end repair
# ---------------------------------------------------------------------------

def repair_divided_highway_dead_ends(
    project: Project,
    max_snap_distance_m: float = 600.0,
) -> Dict[str, Any]:
    """Connect dead-end carriageways of divided highways.

    Divided motorways/trunks imported from OSM often have two separate
    carriageways for each direction.  At the model boundary the two ways
    may terminate at different OSM nodes a few metres apart.  This
    function detects such dead-ends and adds a short bidirectional link.
    """
    from pyproj import Geod
    from shapely.geometry import LineString

    _IGNORE = _MAJOR_ROAD_TYPES | {"centroid_connector"}
    # Minor link types that don't disqualify a node as a dead end --
    # service roads and driveways often connect at highway endpoints
    # without providing real cross-traffic connectivity.
    _MINOR_ALLOWABLE = frozenset({
        "service", "living_street", "residential", "unclassified",
        "tertiary_link", "secondary_link", "primary_link",
    })

    with project_db(project) as conn:
        rows = conn.execute(
            "SELECT link_id, a_node, b_node, link_type, osm_ref_norm FROM links"
        ).fetchall()

        node_major: Dict[int, list] = defaultdict(list)
        node_nonmajor: Dict[int, int] = defaultdict(int)
        node_nonmajor_significant: Dict[int, int] = defaultdict(int)
        link_ref: Dict[int, str] = {}
        link_lt: Dict[int, str] = {}

        for lid, a, b, lt, ref in rows:
            lt_str = str(lt or "")
            if lt_str in _MAJOR_ROAD_TYPES:
                node_major[a].append(lid)
                node_major[b].append(lid)
                link_lt[int(lid)] = lt_str
                if ref:
                    link_ref[lid] = str(ref)
            elif lt_str not in _IGNORE:
                node_nonmajor[a] += 1
                node_nonmajor[b] += 1
                if lt_str not in _MINOR_ALLOWABLE:
                    node_nonmajor_significant[a] += 1
                    node_nonmajor_significant[b] += 1

        dead_ends: Dict[int, int] = {}
        for nid, major_lids in node_major.items():
            if len(major_lids) != 1:
                continue
            # Strict: no non-major links at all
            if node_nonmajor[nid] == 0:
                dead_ends[nid] = major_lids[0]
            # Relaxed: only minor/service links (no significant cross-traffic)
            elif node_nonmajor_significant[nid] == 0:
                dead_ends[nid] = major_lids[0]

        if not dead_ends:
            logger.info("Divided highway repair: no dead-end carriageway nodes found")
            return {"connected_pairs": 0, "new_link_ids": []}

        node_coords = _read_node_coords(conn, list(dead_ends.keys()))

    if not node_coords:
        logger.warning("Divided highway repair: could not read node coordinates")
        return {"connected_pairs": 0, "new_link_ids": []}

    geod = Geod(ellps="WGS84")
    dead_ref = {nid: link_ref.get(lid, "") for nid, lid in dead_ends.items()}
    dead_lt = {nid: link_lt.get(int(lid), "motorway") for nid, lid in dead_ends.items()}

    remaining = set(dead_ends.keys()) & set(node_coords.keys())
    pairs_to_create: list = []

    # Pass 1: pair dead ends that share the same osm_ref_norm
    while remaining:
        best_pair = None
        best_dist = max_snap_distance_m + 1
        remaining_list = sorted(remaining)
        for i, n1 in enumerate(remaining_list):
            ref1 = dead_ref.get(n1, "")
            if not ref1:
                continue
            c1 = node_coords[n1]
            for n2 in remaining_list[i + 1:]:
                if dead_ref.get(n2, "") != ref1:
                    continue
                c2 = node_coords[n2]
                _, _, dist = geod.inv(c1[0], c1[1], c2[0], c2[1])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (n1, n2, dist, ref1)
        if best_pair is None or best_dist > max_snap_distance_m:
            break
        n1, n2, dist, ref = best_pair
        remaining.discard(n1)
        remaining.discard(n2)
        pairs_to_create.append((n1, n2, dist, ref))

    # Pass 2: pair remaining dead ends by proximity + matching link_type,
    # even without osm_ref_norm (handles missing/inconsistent refs)
    if remaining:
        remaining_list = sorted(remaining)
        for i, n1 in enumerate(remaining_list):
            if n1 not in remaining:
                continue
            lt1 = dead_lt.get(n1, "")
            c1 = node_coords.get(n1)
            if not c1:
                continue
            best_n2 = None
            best_dist = max_snap_distance_m + 1
            best_ref = ""
            for n2 in remaining_list[i + 1:]:
                if n2 not in remaining or n2 == n1:
                    continue
                lt2 = dead_lt.get(n2, "")
                if lt1 != lt2:
                    continue
                c2 = node_coords.get(n2)
                if not c2:
                    continue
                _, _, dist = geod.inv(c1[0], c1[1], c2[0], c2[1])
                if dist < best_dist:
                    best_dist = dist
                    best_n2 = n2
                    best_ref = dead_ref.get(n1, "") or dead_ref.get(n2, "")
            if best_n2 is not None and best_dist <= max_snap_distance_m:
                remaining.discard(n1)
                remaining.discard(best_n2)
                pairs_to_create.append((n1, best_n2, best_dist, best_ref))

    if remaining:
        for nid in sorted(remaining):
            ref = dead_ref.get(nid, "<no ref>")
            lt = dead_lt.get(nid, "?")
            logger.warning(
                "Divided highway repair: unpaired dead-end node %d "
                "(ref=%s, type=%s) — may cause one-way disconnection",
                nid, ref, lt,
            )

    if not pairs_to_create:
        logger.info("Divided highway repair: no pairs within snap distance")
        return {"connected_pairs": 0, "new_link_ids": []}

    lt_needed = set()
    for n1, n2, _, _ in pairs_to_create:
        lt_needed.add(dead_lt.get(n1, "motorway"))
        lt_needed.add(dead_lt.get(n2, "motorway"))
    ensure_link_types_registered(project, lt_needed)

    new_link_ids: list = []
    connected_pairs: list = []

    for n1, n2, dist, ref in pairs_to_create:
        c1, c2 = node_coords[n1], node_coords[n2]
        lt_use = dead_lt.get(n1, dead_lt.get(n2, "motorway"))
        if dead_lt.get(n1) and dead_lt.get(n2) and dead_lt[n1] != dead_lt[n2]:
            lt_use = dead_lt[n1]

        # Penalty values: crossover links represent U-turns or service roads,
        # not mainline carriageways. Low speed + capacity discourages assignment
        # from routing through them unless no alternative exists.
        spd = 20.0
        cap_lane = 200.0
        lanes = 1
        tt = (dist / 1000.0) / max(spd, 1.0) * 3600.0 if dist > 0 else 0.01

        links_api = project.network.links
        new_link = links_api.new()
        new_link.geometry = LineString([c1, c2])
        new_link.direction = 0
        new_link.distance = dist
        new_link.modes = "tc"
        new_link.link_type = lt_use
        new_link.speed_ab = spd
        new_link.speed_ba = spd
        new_link.capacity_ab = cap_lane * lanes
        new_link.capacity_ba = cap_lane * lanes
        new_link.lanes_ab = lanes
        new_link.lanes_ba = lanes
        new_link.travel_time_ab = tt
        new_link.travel_time_ba = tt
        new_link.save()

        new_link_ids.append(new_link.link_id)
        connected_pairs.append((n1, n2, round(dist, 1), ref))
        logger.debug(
            "Connected dead-end pair %s: %d and %d (%.0fm) -> link %s (%s)",
            ref, n1, n2, dist, new_link.link_id, lt_use,
        )

    with project_db(project) as conn2:
        for lid, (n1, n2, _, ref) in zip(new_link_ids, pairs_to_create):
            lt_use = dead_lt.get(n1, dead_lt.get(n2, "motorway"))
            if dead_lt.get(n1) and dead_lt.get(n2) and dead_lt[n1] != dead_lt[n2]:
                lt_use = dead_lt[n1]
            conn2.execute(
                "UPDATE links SET a_node=?, b_node=?, osm_ref=?, osm_ref_norm=?, "
                "osm_highway=? WHERE link_id=?",
                (n1, n2, ref, ref, lt_use, lid),
            )

    refresh_network(project)

    logger.info("Divided highway repair: connected %d carriageway pair(s)", len(connected_pairs))
    return {"connected_pairs": len(connected_pairs), "new_link_ids": new_link_ids}


# ---------------------------------------------------------------------------
# General connectivity check
# ---------------------------------------------------------------------------

def check_connectivity(project: Project) -> Dict[str, Any]:
    """Check network connectivity and identify isolated components.

    Returns both undirected component analysis and directed (SCC) analysis
    with warnings for major road links outside the largest SCC.
    """
    links = project.network.links.data
    nodes = project.network.nodes.data

    G = nx.DiGraph()
    for _, node in nodes.iterrows():
        G.add_node(node["node_id"])

    for _, link in links.iterrows():
        a_node = link["a_node"]
        b_node = link["b_node"]
        direction = link.get("direction", 1)
        lt = str(link.get("link_type", ""))
        osm_hw = str(link.get("osm_highway", "")) if "osm_highway" in link.index else lt
        attrs = {"link_id": link["link_id"], "link_type": lt, "osm_highway": osm_hw}
        if direction == 1:
            G.add_edge(a_node, b_node, **attrs)
        elif direction == -1:
            G.add_edge(b_node, a_node, **attrs)
        elif direction == 0:
            G.add_edge(a_node, b_node, **attrs)
            G.add_edge(b_node, a_node, **attrs)

    G_undirected = G.to_undirected()
    components = sorted(nx.connected_components(G_undirected), key=len, reverse=True)
    largest_component = components[0] if components else set()
    isolated_nodes = [n for comp in components[1:] for n in comp if len(comp) == 1]
    isolated_components = [comp for comp in components[1:] if len(comp) > 1]

    sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    largest_scc = sccs[0] if sccs else set()

    major_outside_scc = _count_major_links_outside_scc(links, largest_scc)
    if major_outside_scc > 0:
        logger.warning(
            "%d motorway/trunk links are outside the largest strongly-connected component (%d nodes)",
            major_outside_scc, len(largest_scc),
        )

    return {
        "total_components": len(components),
        "largest_component_size": len(largest_component),
        "isolated_nodes_count": len(isolated_nodes),
        "isolated_components_count": len(isolated_components),
        "directed_scc_count": len(sccs),
        "largest_scc_size": len(largest_scc),
        "major_links_outside_scc": major_outside_scc,
        "components": [
            {"component_id": i, "size": len(comp), "nodes": list(comp)}
            for i, comp in enumerate(components)
        ],
    }


# ---------------------------------------------------------------------------
# Shared helpers (internal)
# ---------------------------------------------------------------------------

def _build_digraph(links: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, lk in links.iterrows():
        a, b = int(lk["a_node"]), int(lk["b_node"])
        d = int(lk.get("direction", 0) or 0)
        if d >= 0:
            G.add_edge(a, b)
        if d <= 0:
            G.add_edge(b, a)
    return G


def _count_major_links_outside_scc(links: pd.DataFrame, scc: set) -> int:
    count = 0
    for _, lk in links.iterrows():
        if str(lk.get("link_type", "")) not in _MAJOR_ROAD_TYPES:
            continue
        if int(lk["a_node"]) not in scc or int(lk["b_node"]) not in scc:
            count += 1
    return count


def _parse_spatialite_point(blob: bytes) -> tuple[float, float]:
    endian = '<' if blob[1] == 1 else '>'
    x = struct.unpack(endian + 'd', blob[43:51])[0]
    y = struct.unpack(endian + 'd', blob[51:59])[0]
    return x, y


def _read_node_coords(conn, node_ids: list) -> Dict[int, tuple[float, float]]:
    """Read node coordinates, falling back to raw blob parsing when
    SpatiaLite functions are unavailable."""
    placeholders = ",".join(str(n) for n in node_ids)
    try:
        rows = conn.execute(
            f"SELECT node_id, ST_X(geometry), ST_Y(geometry) FROM nodes "
            f"WHERE node_id IN ({placeholders})"
        ).fetchall()
    except Exception:
        rows = []

    if rows:
        return {int(r[0]): (float(r[1]), float(r[2])) for r in rows}

    result: Dict[int, tuple[float, float]] = {}
    for nid in node_ids:
        blob = conn.execute("SELECT geometry FROM nodes WHERE node_id = ?", (nid,)).fetchone()
        if blob and blob[0]:
            x, y = _parse_spatialite_point(blob[0])
            result[int(nid)] = (x, y)
    return result
