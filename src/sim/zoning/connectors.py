"""Centroid-connector creation and zone centroid computation."""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae import Project
from shapely.affinity import translate
from shapely.geometry import LineString, Point

from sim.network.db import project_db
from sim.zoning.geo import (
    force_to_target_crs,
    normalize_vector,
    point_is_too_close,
    to_wgs84_point,
)
from sim.zoning.network import (
    MAJOR_ROAD_WEIGHT_THRESHOLD,
    ROAD_CLASS_WEIGHT,
    eligible_road_nodes,
    network_ref,
)

logger = logging.getLogger(__name__)

_CANDIDATE_EXTERNAL_STEPS = [0.0, 2.0, 5.0, 10.0, 15.0, 25.0]


def _row_is_external(row: pd.Series) -> bool:
    val = row.get("is_external", 0)
    if pd.isna(val):
        return False
    return int(val) == 1


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def calculate_centroids(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute centroid points for each zone (representative point for internal,
    stored centroid_x/y for external)."""
    keep_cols = [
        c
        for c in [
            "zone_id",
            "name",
            "source_rank",
            "source_place",
            "is_external",
            "gateway_name",
            "anchor_node_id",
            "anchor_x",
            "anchor_y",
            "boundary_x",
            "boundary_y",
            "boundary_pos",
            "centroid_x",
            "centroid_y",
            "outward_dx",
            "outward_dy",
            "matched_ref",
            "matched_name",
            "link_type",
            "whitelist_token",
            "merged_from",
        ]
        if c in zones.columns
    ]

    pts = zones[keep_cols].copy()
    rep_points = zones.geometry.representative_point()

    geoms = []
    for idx, row in pts.iterrows():
        is_external_val = row.get("is_external", 0)
        is_external = int(0 if pd.isna(is_external_val) else is_external_val) == 1
        if is_external and "centroid_x" in pts.columns and "centroid_y" in pts.columns:
            cx = row.get("centroid_x")
            cy = row.get("centroid_y")
            if pd.notna(cx) and pd.notna(cy):
                geoms.append(Point(float(cx), float(cy)))
                continue
        geoms.append(rep_points.loc[idx])

    pts["geometry"] = geoms
    return gpd.GeoDataFrame(pts, crs=zones.crs)


# ---------------------------------------------------------------------------
# Connector cleanup
# ---------------------------------------------------------------------------

def delete_all_connectors_and_reset_centroids(project: Project) -> int:
    """Remove all centroid connectors and clear centroid flags, returning the old count."""
    with project_db(project) as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='centroid_connector'"
        ).fetchone()[0]
        conn.execute("DELETE FROM links WHERE link_type='centroid_connector'")
        conn.execute("UPDATE nodes SET is_centroid=0 WHERE is_centroid=1")

        orphans = conn.execute("""
            SELECT n.node_id FROM nodes n
            WHERE NOT EXISTS (SELECT 1 FROM links l WHERE l.a_node=n.node_id OR l.b_node=n.node_id)
        """).fetchall()
        if orphans:
            ids = [r[0] for r in orphans]
            conn.executemany("DELETE FROM nodes WHERE node_id=?", [(i,) for i in ids])
            logger.debug("Removed %d orphan nodes", len(ids))

    logger.info(
        "Cleaned: %d old connectors deleted, is_centroid flags cleared",
        before,
    )
    return before


# ---------------------------------------------------------------------------
# Diverse connector selection
# ---------------------------------------------------------------------------

def _select_diverse_connectors(
    eligible: gpd.GeoDataFrame,
    centroid_pt: Point,
    max_connectors: int,
    max_distance_m: float,
    pool_size: int = 40,
    min_major_connectors: int = 1,
) -> gpd.GeoDataFrame:
    """Select spatially-diverse connector target nodes around *centroid_pt*."""
    if eligible.empty or max_connectors <= 0:
        return eligible.head(0).copy()

    cand = eligible.copy()
    cand["_dist"] = cand.geometry.distance(centroid_pt)

    nearby = cand[cand["_dist"] <= float(max_distance_m)].copy()

    if nearby.empty:
        nearby = cand.nsmallest(min(pool_size, len(cand)), "_dist").copy()
    else:
        nearby = nearby.nsmallest(min(pool_size, len(nearby)), "_dist").copy()

    if nearby.empty:
        return nearby

    nearby["_dist"] = nearby["_dist"].clip(lower=1.0)

    cx, cy = centroid_pt.x, centroid_pt.y
    nearby["_angle"] = np.degrees(
        np.arctan2(nearby.geometry.y - cy, nearby.geometry.x - cx)
    ) % 360.0

    nearby["_score"] = nearby["road_weight"] / np.power(nearby["_dist"], 0.75)

    n_sectors = min(max_connectors, 8)
    sector_size = 360.0 / n_sectors
    nearby["_sector"] = (nearby["_angle"] // sector_size).astype(int)

    chosen_idx: List[int] = []

    for sec in range(n_sectors):
        sec_df = nearby[nearby["_sector"] == sec]
        if sec_df.empty:
            continue
        best_idx = sec_df["_score"].idxmax()
        chosen_idx.append(best_idx)
        if len(chosen_idx) >= max_connectors:
            break

    if len(chosen_idx) < max_connectors:
        remaining = nearby.drop(index=chosen_idx, errors="ignore").nlargest(
            max_connectors - len(chosen_idx), "_score"
        )
        chosen_idx.extend(remaining.index.tolist())

    chosen_idx = chosen_idx[:max_connectors]

    if min_major_connectors > 0 and chosen_idx:
        n_major = sum(
            1 for i in chosen_idx
            if nearby.loc[i, "road_weight"] >= MAJOR_ROAD_WEIGHT_THRESHOLD
        )
        if n_major < min_major_connectors:
            needed = min_major_connectors - n_major
            extended_radius = max_distance_m * 2.5
            wider = cand[cand["_dist"] <= extended_radius].copy()
            if wider.empty:
                wider = cand.nsmallest(min(pool_size * 2, len(cand)), "_dist").copy()
            if not wider.empty:
                wider["_dist"] = wider["_dist"].clip(lower=1.0)
                major_pool = wider[
                    (wider["road_weight"] >= MAJOR_ROAD_WEIGHT_THRESHOLD)
                    & (~wider.index.isin(chosen_idx))
                ].copy()
                if not major_pool.empty:
                    major_pool["_score_ext"] = (
                        major_pool["road_weight"]
                        / np.power(major_pool["_dist"], 0.75)
                    )
                    best_majors = major_pool.nlargest(needed, "_score_ext")
                    minor_sorted = sorted(
                        chosen_idx,
                        key=lambda i: nearby.loc[i, "road_weight"],
                    )
                    for j, maj_idx in enumerate(best_majors.index):
                        if j < len(minor_sorted):
                            chosen_idx[chosen_idx.index(minor_sorted[j])] = maj_idx

    all_pool = pd.concat(
        [nearby, cand[cand.index.isin(chosen_idx) & ~cand.index.isin(nearby.index)]],
        ignore_index=False,
    ).drop_duplicates(subset=["node_id"])
    valid_idx = [i for i in chosen_idx if i in all_pool.index]

    selected = all_pool.loc[valid_idx].copy()
    if "_score" not in selected.columns:
        selected["_score"] = (
            selected["road_weight"] / np.power(selected["_dist"].clip(lower=1.0), 0.75)
        )
    selected = selected.sort_values(
        ["_score", "road_weight", "_dist"],
        ascending=[False, False, True],
    ).copy()
    selected["dist"] = selected["_dist"]

    return selected


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def create_centroid_connectors(
    project: Project,
    centroids: gpd.GeoDataFrame,
    max_connectors: int,
    max_distance_m: float,
    speed_kmh: float = 30.0,
    capacity_vph: float = 5000.0,
    lanes: int = 4,
    access_penalty_s: float = 120.0,
    gateway_targets: Optional[Dict[str, List[int]]] = None,
    external_speed_kmh: Optional[float] = None,
    external_access_penalty_s: Optional[float] = None,
    internal_exclude_road_types: Optional[List[str]] = None,
) -> Dict[int, int]:
    """Create centroid connector links for all zones.

    ``internal_exclude_road_types``: road types that should NOT be connector targets
    for internal zones (e.g. ``["motorway", "motorway_link"]``).  A node is excluded
    only when ALL its incident non-connector links are of an excluded type.
    """
    logger.info("Create centroid connectors")

    gateway_targets = gateway_targets or {}
    nodes = project.network.nodes.data
    if "geometry" not in nodes.columns or len(nodes) == 0:
        raise RuntimeError("Network nodes missing geometry/empty")

    road_nids, node_weight = eligible_road_nodes(project, use_directed_scc=True)
    existing_nids = set(int(x) for x in nodes["node_id"].values)

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))
    nodes_gdf = force_to_target_crs(
        nodes_gdf,
        int(centroids.crs.to_epsg()),
        name="network.nodes(for connector candidates)",
    )

    eligible = nodes_gdf[nodes_gdf["node_id"].isin(road_nids)].copy()
    eligible["road_weight"] = eligible["node_id"].map(node_weight).fillna(0.5)
    logger.info(
        "Eligible road-network nodes: %d (from %d total, excluded %d non-car types)",
        len(eligible),
        len(nodes_gdf),
        len(nodes_gdf) - len(eligible),
    )

    gw_road_nids_scc, gw_node_weight_scc = eligible_road_nodes(project, use_directed_scc=True)
    gw_road_nids_all, gw_node_weight_all = eligible_road_nodes(project, use_directed_scc=False)
    gateway_eligible_scc = nodes_gdf[nodes_gdf["node_id"].isin(gw_road_nids_scc)].copy()
    gateway_eligible_scc["road_weight"] = gateway_eligible_scc["node_id"].map(gw_node_weight_scc).fillna(0.5)
    gateway_eligible_fallback = nodes_gdf[nodes_gdf["node_id"].isin(gw_road_nids_all)].copy()
    gateway_eligible_fallback["road_weight"] = gateway_eligible_fallback["node_id"].map(gw_node_weight_all).fillna(0.5)
    gateway_eligible = gateway_eligible_scc

    internal_eligible = eligible
    excl_types = set(internal_exclude_road_types or [])
    if excl_types:
        node_road_types: Dict[int, set] = {}
        links_data = project.network.links.data
        for _, r in links_data.iterrows():
            lt = str(r.get("link_type", "")).strip()
            if lt == "centroid_connector" or not lt:
                continue
            for nid in (int(r["a_node"]), int(r["b_node"])):
                node_road_types.setdefault(nid, set()).add(lt)

        motorway_only_nodes = {
            nid for nid, types in node_road_types.items()
            if types and types <= excl_types
        }
        internal_eligible = eligible[~eligible["node_id"].astype(int).isin(motorway_only_nodes)].copy()
        n_excluded = len(eligible) - len(internal_eligible)
        logger.debug(
            "Internal connector pool: %d nodes (excluded %d motorway-only nodes for internal zones)",
            len(internal_eligible),
            n_excluded,
        )

    occupied_points = list(nodes_gdf.geometry.dropna())

    if eligible.empty:
        logger.warning("no eligible road-network nodes found")
        return {}

    next_id = 1
    zone_to_centroid: Dict[int, int] = {}
    for zid in sorted(centroids["zone_id"].astype(int)):
        while next_id in existing_nids:
            next_id += 1
        zone_to_centroid[zid] = next_id
        existing_nids.add(next_id)
        next_id += 1

    logger.info(
        "Centroid IDs: %d-%d for %d zones",
        min(zone_to_centroid.values()),
        max(zone_to_centroid.values()),
        len(zone_to_centroid),
    )

    nodes_wgs84 = project.network.nodes.data[["node_id", "geometry"]].copy()
    wgs84_lookup = {int(r["node_id"]): r["geometry"] for _, r in nodes_wgs84.iterrows()}

    created = 0
    for _, c in centroids.iterrows():
        zone_id = int(c["zone_id"])
        centroid_node_id = zone_to_centroid[zone_id]

        is_external = bool(int(c.get("is_external", 0))) if pd.notna(c.get("is_external", 0)) else False
        gateway_name = str(c.get("gateway_name", "") or "").strip()

        if is_external and pd.notna(c.get("centroid_x")) and pd.notna(c.get("centroid_y")):
            base_pt = Point(float(c["centroid_x"]), float(c["centroid_y"]))
            ux, uy = normalize_vector(float(c.get("outward_dx", 1.0)), float(c.get("outward_dy", 0.0)))
        else:
            base_pt = c.geometry
            ux, uy = 1.0, 0.0

        saved_centroid_pt: Optional[Point] = None
        saved_centroid_wgs84 = None

        candidate_steps = _CANDIDATE_EXTERNAL_STEPS if is_external else [0.0]

        for step in candidate_steps:
            candidate_pt = translate(base_pt, xoff=ux * step, yoff=uy * step)

            if point_is_too_close(candidate_pt, occupied_points, min_sep_m=1.0):
                continue

            candidate_wgs84 = to_wgs84_point(candidate_pt, centroids.crs)

            try:
                new_node = (
                    project.network.nodes.new_centroid(centroid_node_id)
                    if hasattr(project.network.nodes, "new_centroid")
                    else project.network.nodes.new()
                )
                if not hasattr(project.network.nodes, "new_centroid"):
                    new_node.__dict__["node_id"] = centroid_node_id

                new_node.is_centroid = 1
                new_node.geometry = candidate_wgs84
                new_node.save()

                saved_centroid_pt = candidate_pt
                saved_centroid_wgs84 = candidate_wgs84
                occupied_points.append(candidate_pt)
                break

            except sqlite3.IntegrityError:
                continue

        if saved_centroid_pt is None or saved_centroid_wgs84 is None:
            raise RuntimeError(
                f"Could not place centroid node for zone_id={zone_id} "
                f"without overlapping an existing node"
            )

        centroid_pt = saved_centroid_pt
        centroid_wgs84 = saved_centroid_wgs84

        if is_external and gateway_name in gateway_targets:
            target_ids = [int(x) for x in gateway_targets[gateway_name]]
            cand = gateway_eligible[gateway_eligible["node_id"].astype(int).isin(target_ids)].copy()
            if cand.empty:
                cand = gateway_eligible_fallback[
                    gateway_eligible_fallback["node_id"].astype(int).isin(target_ids)
                ].copy()
                if not cand.empty:
                    logger.warning(
                        "gateway %s: no SCC targets, falling back to non-SCC pool (%d nodes)",
                        gateway_name,
                        len(cand),
                    )
            cand["dist"] = cand.geometry.distance(centroid_pt)
            cand = cand.sort_values(["dist", "road_weight"], ascending=[True, False]).head(len(target_ids))
        else:
            pool = internal_eligible if not is_external else eligible
            cand = _select_diverse_connectors(
                pool,
                centroid_pt,
                max_connectors,
                max_distance_m,
            )

        for _, n in cand.iterrows():
            target_node_id = int(n["node_id"])
            dist_m = float(n["dist"])

            target_geom = wgs84_lookup.get(target_node_id)
            if target_geom is None:
                continue

            link = project.network.links.new()
            link.__dict__["a_node"] = centroid_node_id
            link.__dict__["b_node"] = target_node_id
            link.direction = 0
            link.modes = "c"
            link.link_type = "centroid_connector"

            link_speed = (
                float(external_speed_kmh)
                if (is_external and external_speed_kmh is not None)
                else float(speed_kmh)
            )
            link_penalty = (
                float(external_access_penalty_s)
                if (is_external and external_access_penalty_s is not None)
                else float(access_penalty_s)
            )

            link.distance = dist_m
            link.speed_ab = link_speed
            link.speed_ba = link_speed
            link.capacity_ab = capacity_vph
            link.capacity_ba = capacity_vph
            link.lanes_ab = lanes
            link.lanes_ba = lanes

            time_s = (dist_m / 1000.0) / max(link_speed, 1e-6) * 3600.0 + link_penalty
            if hasattr(link, "travel_time_ab"):
                link.travel_time_ab = time_s
                link.travel_time_ba = time_s

            link.geometry = LineString([
                (centroid_wgs84.x, centroid_wgs84.y),
                (target_geom.x, target_geom.y),
            ])

            link.save()
            created += 1

    logger.info(
        "connectors created: %d (%d zones, %d per zone)",
        created,
        len(zone_to_centroid),
        max_connectors,
    )
    return zone_to_centroid
