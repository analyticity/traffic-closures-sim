"""Connector and gateway diagnostics / validation exports."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import networkx as nx
import pandas as pd
from aequilibrae import Project
from shapely.geometry import Point

from sim.network.db import project_db
from sim.zoning.network import pick_best_road_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connector diagnostics
# ---------------------------------------------------------------------------

def export_connector_diagnostics(
    project: Project,
    zone_to_centroid: Dict[int, int],
    output_dir: Path,
) -> None:
    """Write ``connector_diagnostics.csv`` with per-connector details."""
    with project_db(project) as conn:
        rows = conn.execute(
            "SELECT link_id, a_node, b_node, distance, speed_ab, travel_time_ab "
            "FROM links WHERE link_type='centroid_connector'"
        ).fetchall()

        road_link_types: Dict[int, str] = {}
        for nid, lt in conn.execute(
            "SELECT a_node, link_type FROM links WHERE link_type != 'centroid_connector' "
            "UNION ALL "
            "SELECT b_node, link_type FROM links WHERE link_type != 'centroid_connector'"
        ).fetchall():
            prev = road_link_types.get(int(nid), "")
            road_link_types[int(nid)] = pick_best_road_type(prev, str(lt or ""))

        dir_edges = conn.execute(
            "SELECT a_node, b_node, direction FROM links WHERE link_type != 'centroid_connector'"
        ).fetchall()

    G_dir = nx.DiGraph()
    for a, b, direction in dir_edges:
        d = int(direction or 0)
        if d >= 0:
            G_dir.add_edge(int(a), int(b))
        if d <= 0:
            G_dir.add_edge(int(b), int(a))
    largest_scc: set = set()
    if G_dir.number_of_nodes() > 0:
        largest_scc = max(nx.strongly_connected_components(G_dir), key=len)

    centroid_to_zone = {v: k for k, v in zone_to_centroid.items()}
    records = []
    for lid, a, b, dist, speed, tt in rows:
        zid = centroid_to_zone.get(a, centroid_to_zone.get(b))
        road_node = b if a in centroid_to_zone else a
        travel_km = (dist or 0) / 1000.0
        penalty_s = (tt or 0) - (travel_km / max(speed or 30, 1) * 3600) if tt and speed else 0
        records.append({
            "zone_id": zid,
            "centroid_node": a if a in centroid_to_zone else b,
            "road_node": road_node,
            "link_id": lid,
            "distance_m": round(dist or 0, 1),
            "speed_kmh": speed,
            "travel_time_s": round(tt or 0, 1),
            "access_penalty_s": round(max(penalty_s, 0), 1),
            "road_link_type": road_link_types.get(int(road_node), ""),
            "in_directed_scc": int(road_node) in largest_scc,
        })

    if records:
        df = pd.DataFrame(records)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "connector_diagnostics.csv"
        df.to_csv(out, index=False)

        gw_mask = df["zone_id"].astype(int) >= 8_000_000_000
        if gw_mask.any():
            gw = df[gw_mask]
            n_scc = int(gw["in_directed_scc"].sum())
            n_total = len(gw)
            logger.info("Connector diagnostics: %s (%d connectors)", out, len(df))
            logger.info("Gateway SCC status: %d/%d connectors target SCC nodes", n_scc, n_total)
        else:
            logger.info("Connector diagnostics: %s (%d connectors)", out, len(df))


# ---------------------------------------------------------------------------
# Connector validation
# ---------------------------------------------------------------------------

def validate_connectors(
    project: Project,
    zone_to_centroid: Dict[int, int],
    output_dir: Path,
    *,
    min_connectors: int = 2,
    warn_min_distance_m: float = 1500.0,
) -> Dict[str, Any]:
    """Post-connector validation: check counts, distances, road types, SCC membership."""
    with project_db(project) as conn:
        connector_rows = conn.execute(
            "SELECT link_id, a_node, b_node, distance FROM links WHERE link_type='centroid_connector'"
        ).fetchall()

        centroid_to_zone = {v: k for k, v in zone_to_centroid.items()}

        zone_connectors: Dict[int, List[Dict[str, Any]]] = {}
        for lid, a, b, dist in connector_rows:
            zid = centroid_to_zone.get(a, centroid_to_zone.get(b))
            if zid is None:
                continue
            road_node = b if a in centroid_to_zone else a
            zone_connectors.setdefault(zid, []).append({
                "link_id": lid, "road_node": road_node, "distance_m": dist or 0,
            })

        dir_edges = conn.execute(
            "SELECT a_node, b_node, direction FROM links WHERE link_type != 'centroid_connector'"
        ).fetchall()

    G = nx.DiGraph()
    for a, b, direction in dir_edges:
        d = int(direction or 0)
        if d >= 0:
            G.add_edge(int(a), int(b))
        if d <= 0:
            G.add_edge(int(b), int(a))

    largest_scc = set()
    if G.number_of_nodes() > 0:
        largest_scc = max(nx.strongly_connected_components(G), key=len)

    warnings: List[Dict[str, Any]] = []
    zone_reports: List[Dict[str, Any]] = []

    for zid in sorted(zone_to_centroid.keys()):
        conns = zone_connectors.get(zid, [])
        n_conn = len(conns)
        dists = [c["distance_m"] for c in conns]
        min_dist = min(dists) if dists else None
        max_dist = max(dists) if dists else None
        road_nodes_outside_scc = [
            c["road_node"] for c in conns if int(c["road_node"]) not in largest_scc
        ]

        report: Dict[str, Any] = {
            "zone_id": zid,
            "num_connectors": n_conn,
            "min_distance_m": round(min_dist, 1) if min_dist is not None else None,
            "max_distance_m": round(max_dist, 1) if max_dist is not None else None,
            "road_nodes_outside_scc": road_nodes_outside_scc,
        }

        if n_conn < min_connectors:
            w = f"zone {zid}: only {n_conn} connector(s) (minimum {min_connectors})"
            warnings.append({"zone_id": zid, "type": "few_connectors", "detail": w})
        if min_dist is not None and min_dist > warn_min_distance_m:
            w = f"zone {zid}: nearest connector is {min_dist:.0f} m away (threshold {warn_min_distance_m:.0f} m)"
            warnings.append({"zone_id": zid, "type": "far_connectors", "detail": w})
        if road_nodes_outside_scc:
            w = f"zone {zid}: {len(road_nodes_outside_scc)} road node(s) outside largest directed SCC"
            warnings.append({"zone_id": zid, "type": "outside_scc", "detail": w})

        zone_reports.append(report)

    result = {
        "total_zones": len(zone_to_centroid),
        "total_connectors": len(connector_rows),
        "largest_directed_scc_size": len(largest_scc),
        "warnings_count": len(warnings),
        "warnings": warnings,
        "zones": zone_reports,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "connector_validation.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if warnings:
        logger.warning("Connector validation: %d warning(s)", len(warnings))
        for w in warnings:
            logger.warning("%s", w["detail"])
    else:
        logger.info("Connector validation: all %d zones OK", len(zone_to_centroid))
    logger.info("Validation report: %s", out)

    return result


# ---------------------------------------------------------------------------
# Gateway diagnostics
# ---------------------------------------------------------------------------

def export_gateway_diagnostics(
    gateway_meta: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write ``gateway_diagnostics.csv`` with one row per gateway."""
    if not gateway_meta:
        return

    rows = []
    for gw_name, meta in gateway_meta.items():
        rows.append({
            "gateway_name": gw_name,
            "whitelist_token": meta.get("whitelist_token"),
            "whitelist_priority": meta.get("whitelist_priority"),
            "cluster_index": meta.get("cluster_index"),
            "anchor_node_id": meta.get("anchor_node_id"),
            "boundary_x": meta.get("boundary_x"),
            "boundary_y": meta.get("boundary_y"),
            "boundary_pos": meta.get("boundary_pos"),
            "anchor_x": meta.get("anchor_x"),
            "anchor_y": meta.get("anchor_y"),
            "outward_dx": meta.get("outward_dx"),
            "outward_dy": meta.get("outward_dy"),
            "matched_ref": meta.get("matched_ref"),
            "matched_name": meta.get("matched_name"),
            "link_type": meta.get("link_type"),
            "predominant_link_type": meta.get("predominant_link_type", meta.get("link_type")),
            "auto_discovered": meta.get("auto_discovered", False),
            "boundary_angle": meta.get("boundary_angle"),
            "dist_boundary_m": meta.get("dist_boundary_m"),
            "target_node_ids": ",".join(str(x) for x in meta.get("target_node_ids", [])),
            "merged_from": meta.get("merged_from", ""),
        })

    if rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "gateway_diagnostics.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Gateway diagnostics: %s (%d gateways)", out, len(rows))


# ---------------------------------------------------------------------------
# Gateway seed lookup
# ---------------------------------------------------------------------------

def export_gateway_seed_lookup(
    gateway_meta: Dict[str, Dict[str, Any]],
    output_path: Path,
    *,
    crs_epsg: int,
) -> None:
    """Export stable gateway seed lookup for ``build-supernetwork``."""
    if not gateway_meta:
        return

    rows = []
    for gw_name, meta in gateway_meta.items():
        rows.append({
            "gateway_name": gw_name,
            "whitelist_token": str(meta.get("whitelist_token", "")),
            "boundary_x": float(meta.get("boundary_x", 0.0)),
            "boundary_y": float(meta.get("boundary_y", 0.0)),
            "boundary_pos": float(meta.get("boundary_pos", 0.0)),
            "anchor_node_id": int(meta.get("anchor_node_id", 0)),
            "anchor_x": float(meta.get("anchor_x", 0.0)),
            "anchor_y": float(meta.get("anchor_y", 0.0)),
            "outward_dx": float(meta.get("outward_dx", 0.0)),
            "outward_dy": float(meta.get("outward_dy", 0.0)),
            "matched_ref": str(meta.get("matched_ref", "")),
            "matched_name": str(meta.get("matched_name", "")),
            "link_type": str(meta.get("link_type", "")),
            "predominant_link_type": str(meta.get("predominant_link_type", meta.get("link_type", ""))),
            "auto_discovered": bool(meta.get("auto_discovered", False)),
            "geometry": Point(float(meta.get("boundary_x", 0.0)), float(meta.get("boundary_y", 0.0))),
        })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{crs_epsg}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(output_path, index=False)
    elif suffix == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    elif suffix == ".csv":
        pd.DataFrame(gdf.drop(columns="geometry")).to_csv(output_path, index=False)
    else:
        out = output_path.with_suffix(".parquet")
        gdf.to_parquet(out, index=False)
        output_path = out

    logger.info("Gateway seed lookup: %s", output_path)
