"""Normalization pipeline orchestrator.

Runs attribute normalization, connectivity repairs, optional baseline
closures, and final network export in a single call.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd
from aequilibrae import Project

from sim.io_project import get_metric_epsg, load_config
from sim.network.closures import (
    apply_baseline_closures,
    load_closures,
    _write_closure_state_to_db,
)
from sim.network.connectivity import (
    check_connectivity,
    repair_boundary_scc,
    repair_divided_highway_dead_ends,
)
from sim.network.db import project_db
from sim.network.export import export_stable_network
from sim.network.normalization import normalize_network_attributes

logger = logging.getLogger(__name__)


def normalize_and_export_network(
    config_path: Union[str, Path] = "config/brno/sim.yaml",
    outputs_dir: Union[str, Path, None] = None,
) -> None:
    """Normalize link attributes, repair connectivity, apply closures, and export."""
    cfg = load_config(config_path)
    if outputs_dir is None:
        outputs_dir = cfg.get("network", {}).get("output_dir", "outputs/baseline/network")

    network_cfg = cfg.get("network") or {}
    experiment_profile = network_cfg.get("experiment_profile", "baseline")
    project_dir = Path(cfg["project_path"])

    project = Project()
    project.open(str(project_dir))

    try:
        links = _normalize(project, network_cfg, experiment_profile)
        links = _repair_connectivity(project, links, network_cfg)
        connectivity_info = check_connectivity(project)

        _log_connectivity(connectivity_info)

        links = _apply_closures_if_enabled(project, links, cfg)

        _export(project, links, connectivity_info, cfg, outputs_dir)

        logger.info("Normalization complete")
    finally:
        project.close()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _normalize(
    project: Project,
    network_cfg: dict,
    experiment_profile: str,
) -> pd.DataFrame:
    logger.info("Normalize attributes")
    links = normalize_network_attributes(project, network_cfg, experiment_profile=experiment_profile)
    logger.info("Normalized %d links", len(links))
    return links


def _repair_connectivity(
    project: Project,
    links: pd.DataFrame,
    network_cfg: dict,
) -> pd.DataFrame:
    logger.info("Repair boundary SCC")
    repair_info = repair_boundary_scc(project)
    if repair_info.get("repaired", 0) > 0:
        repaired_set = set(repair_info["repaired_ids"])
        mask = links["link_id"].astype(int).isin(repaired_set)
        links.loc[mask, "direction"] = 0
        links.loc[mask, "speed_ba"] = links.loc[mask, "speed_ab"]
        links.loc[mask, "capacity_ba"] = links.loc[mask, "capacity_ab"]
        links.loc[mask, "lanes_ba"] = links.loc[mask, "lanes_ab"]
        links.loc[mask, "travel_time_ba"] = links.loc[mask, "travel_time_ab"]

    logger.info("Repair divided highways")
    divided_info = repair_divided_highway_dead_ends(
        project,
        max_snap_distance_m=float(network_cfg.get("divided_highway_snap_m", 600)),
    )
    if divided_info["new_link_ids"]:
        with project_db(project) as conn_tmp:
            for new_lid in divided_info["new_link_ids"]:
                row = conn_tmp.execute(
                    "SELECT link_id, speed_ab, capacity_ab, lanes_ab, distance, link_type "
                    "FROM links WHERE link_id=?",
                    (new_lid,),
                ).fetchone()
                if row:
                    spd = float(row[1]) if row[1] else 0.0
                    dist_m = float(row[4])
                    tt = (dist_m / 1000.0 / spd * 3600.0) if spd else 0.0
                    new_row = pd.DataFrame([{
                        "link_id": row[0], "link_type": str(row[5] or "motorway"),
                        "direction": 0,
                        "speed_ab": row[1], "speed_ba": row[1],
                        "capacity_ab": row[2], "capacity_ba": row[2],
                        "lanes_ab": row[3], "lanes_ba": row[3],
                        "travel_time_ab": tt, "travel_time_ba": tt,
                        "distance": row[4],
                    }])
                    links = pd.concat([links, new_row], ignore_index=True)
    return links


def _log_connectivity(connectivity_info: dict) -> None:
    logger.info(
        "Connectivity: %d components, largest %d nodes, %d isolated nodes, %d isolated multi-node components",
        connectivity_info["total_components"],
        connectivity_info["largest_component_size"],
        connectivity_info["isolated_nodes_count"],
        connectivity_info["isolated_components_count"],
    )


def _apply_closures_if_enabled(
    project: Project,
    links: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    bc_cfg = cfg.get("baseline_closures") or {}
    if not bc_cfg.get("enabled", False):
        return links

    logger.info("Baseline closures")
    _cache = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))
    source_path = Path(bc_cfg.get("source_path", f"{_cache}/closures.parquet"))
    closures = load_closures(
        source_path,
        measurement_period=bc_cfg.get("measurement_period"),
        status_whitelist=bc_cfg.get("status_whitelist"),
    )
    if not closures:
        logger.info("No closures to apply (empty or file missing)")
        return links

    links = apply_baseline_closures(links, closures, cfg)

    with project_db(project) as conn:
        _write_closure_state_to_db(conn, links)

    return links


def _export(
    project: Project,
    links: pd.DataFrame,
    connectivity_info: dict,
    cfg: dict,
    outputs_dir: Union[str, Path],
) -> None:
    logger.info("Export network")
    crs_epsg = get_metric_epsg(cfg)
    export_stable_network(
        project, Path(outputs_dir), connectivity_info,
        normalized_links=links, output_crs_epsg=crs_epsg, cfg=cfg,
    )
