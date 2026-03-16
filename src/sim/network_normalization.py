from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from aequilibrae import Project

from sim.io_project import load_config


EXPERIMENT_PROFILES = {
    "baseline": {
        "speed_caps": {},
        "capacity_factors": {},
        "time_penalties": {},
    },
    "motorway_push": {
        "speed_caps": {
            "motorway": 130.0,
            "motorway_link": 110.0,
            "trunk": 100.0,
            "trunk_link": 85.0,
            "primary": 70.0,
            "primary_link": 55.0,
            "secondary": 48.0,
            "secondary_link": 40.0,
            "tertiary": 38.0,
            "tertiary_link": 32.0,
            "unclassified": 35.0,
            "road": 35.0,
            "residential": 30.0,
            "service": 20.0,
            "living_street": 20.0,
        },
        "speed_floors": {
            "motorway": 120.0,
            "motorway_link": 85.0,
            "trunk": 85.0,
            "trunk_link": 70.0,
            "primary": 60.0,
        },
        "capacity_factors": {
            "motorway": 1.35,
            "motorway_link": 1.20,
            "trunk": 1.25,
            "trunk_link": 1.10,
            "primary": 0.95,
            "primary_link": 0.95,
            "secondary": 0.90,
            "secondary_link": 0.90,
            "tertiary": 0.85,
            "tertiary_link": 0.85,
            "unclassified": 0.80,
            "road": 0.80,
            "residential": 0.75,
            "service": 0.60,
            "living_street": 0.50,
        },
        "time_penalties": {
            "primary": 4.0,
            "primary_link": 2.0,
            "secondary": 8.0,
            "secondary_link": 5.0,
            "tertiary": 8.0,
            "unclassified": 10.0,
            "road": 10.0,
            "residential": 15.0,
            "service": 25.0,
            "living_street": 35.0,
        },
    },
    "hierarchy_strong": {
        "speed_caps": {
            "motorway": 130.0,
            "motorway_link": 90.0,
            "trunk": 90.0,
            "trunk_link": 70.0,
            "primary": 70.0,
            "primary_link": 55.0,
            "secondary": 45.0,
            "secondary_link": 40.0,
            "tertiary": 35.0,
            "tertiary_link": 30.0,
            "unclassified": 30.0,
            "road": 30.0,
            "residential": 25.0,
            "service": 15.0,
            "living_street": 15.0,
        },
        "capacity_factors": {
            "motorway": 1.15,
            "motorway_link": 1.10,
            "trunk": 1.10,
            "trunk_link": 1.05,
            "primary": 1.00,
            "primary_link": 1.00,
            "secondary": 0.85,
            "secondary_link": 0.85,
            "tertiary": 0.75,
            "tertiary_link": 0.75,
            "unclassified": 0.65,
            "road": 0.65,
            "residential": 0.50,
            "service": 0.35,
            "living_street": 0.25,
        },
        "time_penalties": {},
    },
    "local_penalty": {
        "speed_caps": {},
        "capacity_factors": {},
        "time_penalties": {
            "tertiary": 5.0,
            "unclassified": 8.0,
            "road": 8.0,
            "residential": 15.0,
            "service": 30.0,
            "living_street": 45.0,
        },
    },
    "combined": {
        "speed_caps": {
            "motorway": 130.0,
            "motorway_link": 90.0,
            "trunk": 90.0,
            "trunk_link": 70.0,
            "primary": 70.0,
            "primary_link": 55.0,
            "secondary": 45.0,
            "secondary_link": 40.0,
            "tertiary": 35.0,
            "tertiary_link": 30.0,
            "unclassified": 30.0,
            "road": 30.0,
            "residential": 25.0,
            "service": 15.0,
            "living_street": 15.0,
        },
        "capacity_factors": {
            "motorway": 1.15,
            "motorway_link": 1.10,
            "trunk": 1.10,
            "trunk_link": 1.05,
            "primary": 1.00,
            "primary_link": 1.00,
            "secondary": 0.85,
            "secondary_link": 0.85,
            "tertiary": 0.75,
            "tertiary_link": 0.75,
            "unclassified": 0.65,
            "road": 0.65,
            "residential": 0.50,
            "service": 0.35,
            "living_street": 0.25,
        },
        "time_penalties": {
            "tertiary": 5.0,
            "unclassified": 8.0,
            "road": 8.0,
            "residential": 15.0,
            "service": 30.0,
            "living_street": 45.0,
        },
    },
}


def _lt_mask(links: pd.DataFrame, link_type: str) -> pd.Series:
    return links["link_type"].astype(str).str.fullmatch(link_type, case=False, na=False)

def _apply_speed_floors(links: pd.DataFrame, speed_floors: dict) -> None:
    for lt, floor_val in speed_floors.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "speed_ab"] = np.maximum(links.loc[m, "speed_ab"], floor_val)
        links.loc[m, "speed_ba"] = np.maximum(links.loc[m, "speed_ba"], floor_val)

def _apply_speed_caps(links: pd.DataFrame, speed_caps: dict) -> None:
    for lt, cap in speed_caps.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "speed_ab"] = np.minimum(links.loc[m, "speed_ab"], cap)
        links.loc[m, "speed_ba"] = np.minimum(links.loc[m, "speed_ba"], cap)


def _apply_capacity_factors(links: pd.DataFrame, capacity_factors: dict) -> None:
    for lt, factor in capacity_factors.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "capacity_ab"] = links.loc[m, "capacity_ab"] * factor
        links.loc[m, "capacity_ba"] = links.loc[m, "capacity_ba"] * factor


def _apply_time_penalties(links: pd.DataFrame, time_penalties: dict) -> None:
    for lt, penalty_s in time_penalties.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "travel_time_ab"] = links.loc[m, "travel_time_ab"] + penalty_s
        links.loc[m, "travel_time_ba"] = links.loc[m, "travel_time_ba"] + penalty_s


def _apply_experiment_profile(links: pd.DataFrame, experiment_profile: str) -> pd.DataFrame:
    profile = EXPERIMENT_PROFILES.get(experiment_profile, EXPERIMENT_PROFILES["baseline"])
    print(f"Applying network experiment profile: {experiment_profile}")

    _apply_speed_floors(links, profile.get("speed_floors", {}))
    _apply_speed_caps(links, profile.get("speed_caps", {}))
    _apply_capacity_factors(links, profile.get("capacity_factors", {}))

    links["speed_ab"] = links["speed_ab"].clip(lower=5.0)
    links["speed_ba"] = links["speed_ba"].clip(lower=5.0)
    links["capacity_ab"] = links["capacity_ab"].clip(lower=50.0)
    links["capacity_ba"] = links["capacity_ba"].clip(lower=50.0)

    return links

def normalize_network_attributes(
    project: Project,
    project_dir: Path | None = None,
    experiment_profile: str = "baseline",
) -> pd.DataFrame:
    """Normalize/compute link attributes, preserving AB/BA directional asymmetry.

    For two-way links (direction=0), when both AB and BA values exist from OSM,
    they are kept separately -- never averaged. Only when a value is missing
    is a default or the opposite-direction value used as fallback.

    Writes back per-direction values (speed_ab, speed_ba, ...) to the SQLite DB.
    """
    links = project.network.links.data.copy()

    links["estimated_distance"] = 0
    links["estimated_speed_ab"] = 0
    links["estimated_speed_ba"] = 0
    links["estimated_capacity_ab"] = 0
    links["estimated_capacity_ba"] = 0
    links["estimated_lanes_ab"] = 0
    links["estimated_lanes_ba"] = 0
    links["estimated_free_flow_time_ab"] = 0
    links["estimated_free_flow_time_ba"] = 0

    mask_twoway = links["direction"] == 0

    # ------------------------------------------------------------------
    # 1. Distance (from geometry if missing)
    # ------------------------------------------------------------------
    if "distance" not in links.columns:
        links["distance"] = None
    missing_dist = links["distance"].isna() | (links["distance"] == 0)
    if missing_dist.any():
        links.loc[missing_dist, "distance"] = links.loc[missing_dist, "geometry"].length
        links.loc[missing_dist, "estimated_distance"] = 1

    # ------------------------------------------------------------------
    # Helper: resolve per-direction attribute from OSM columns
    # ------------------------------------------------------------------
    default_speeds = {
        "motorway": 130.0,
        "motorway_link": 80.0,
        "trunk": 90.0,
        "trunk_link": 70.0,
        "primary": 70.0,
        "primary_link": 50.0,
        "secondary": 50.0,
        "secondary_link": 40.0,
        "tertiary": 40.0,
        "tertiary_link": 35.0,
        "unclassified": 35.0,
        "road": 35.0,
        "residential": 30.0,
        "service": 20.0,
        "living_street": 20.0,
    }

    default_lanes = {
        "motorway": 3,
        "motorway_link": 2,
        "trunk": 2,
        "trunk_link": 2,
        "primary": 2,
        "primary_link": 1,
        "secondary": 1,
        "secondary_link": 1,
        "tertiary": 1,
        "tertiary_link": 1,
        "unclassified": 1,
        "road": 1,
        "residential": 1,
        "service": 1,
        "living_street": 1,
    }

    lane_capacity = [
        ("motorway_link", 1800), ("motorway", 2200),
        ("trunk_link", 1500), ("trunk", 1800),
        ("primary_link", 1100), ("primary", 1400),
        ("secondary_link", 850), ("secondary", 1000),
        ("tertiary_link", 650), ("tertiary", 800),
        ("unclassified", 600),
        ("road", 600),
        ("residential", 500),
        ("service", 300),
        ("living_street", 150),
    ]

    def _resolve_directional(
        col_ab: str,
        col_ba: str,
        defaults: dict,
        fallback_val: float,
        est_ab: str,
        est_ba: str,
    ) -> None:
        """Populate col_ab / col_ba keeping asymmetry, filling missing with defaults."""
        if col_ab in links.columns:
            links[col_ab] = pd.to_numeric(links[col_ab], errors="coerce")
        else:
            links[col_ab] = None

        if col_ba in links.columns:
            links[col_ba] = pd.to_numeric(links[col_ba], errors="coerce")
        else:
            links[col_ba] = None

        ab_missing = links[col_ab].isna()
        ba_missing = links[col_ba].isna()

        # Two-way: if one direction present but other missing, copy
        tw_ab_only = mask_twoway & (~ab_missing) & ba_missing
        tw_ba_only = mask_twoway & ab_missing & (~ba_missing)
        links.loc[tw_ab_only, col_ba] = links.loc[tw_ab_only, col_ab]
        links.loc[tw_ab_only, est_ba] = 1
        links.loc[tw_ba_only, col_ab] = links.loc[tw_ba_only, col_ba]
        links.loc[tw_ba_only, est_ab] = 1

        # Fill remaining missing from defaults by link_type
        still_ab_missing = links[col_ab].isna()
        still_ba_missing = links[col_ba].isna()
        if "link_type" in links.columns:
            for lt, dv in defaults.items():
                m_ab = still_ab_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                m_ba = still_ba_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                links.loc[m_ab, col_ab] = dv
                links.loc[m_ab, est_ab] = 1
                links.loc[m_ba, col_ba] = dv
                links.loc[m_ba, est_ba] = 1

        # Final fallback
        links.loc[links[col_ab].isna(), est_ab] = 1
        links.loc[links[col_ab].isna(), col_ab] = fallback_val
        links.loc[links[col_ba].isna(), est_ba] = 1
        links.loc[links[col_ba].isna(), col_ba] = fallback_val

    # ------------------------------------------------------------------
    # 2. Speed (preserve AB/BA)
    # ------------------------------------------------------------------
    _resolve_directional(
        "speed_ab",
        "speed_ba",
        default_speeds,
        35.0,
        "estimated_speed_ab",
        "estimated_speed_ba",
    )

    # ------------------------------------------------------------------
    # 3. Lanes (preserve AB/BA)
    # ------------------------------------------------------------------
    _resolve_directional(
        "lanes_ab",
        "lanes_ba",
        default_lanes,
        1,
        "estimated_lanes_ab",
        "estimated_lanes_ba",
    )
    links["lanes_ab"] = links["lanes_ab"].clip(lower=1)
    links["lanes_ba"] = links["lanes_ba"].clip(lower=1)

    # ------------------------------------------------------------------
    # 4. Capacity (preserve AB/BA)
    # ------------------------------------------------------------------
    if "capacity_ab" in links.columns:
        links["capacity_ab"] = pd.to_numeric(links["capacity_ab"], errors="coerce")
    else:
        links["capacity_ab"] = None

    if "capacity_ba" in links.columns:
        links["capacity_ba"] = pd.to_numeric(links["capacity_ba"], errors="coerce")
    else:
        links["capacity_ba"] = None

    # Two-way: copy present direction to missing one
    tw_cab = mask_twoway & links["capacity_ab"].notna() & links["capacity_ba"].isna()
    tw_cba = mask_twoway & links["capacity_ba"].notna() & links["capacity_ab"].isna()
    links.loc[tw_cab, "capacity_ba"] = links.loc[tw_cab, "capacity_ab"]
    links.loc[tw_cab, "estimated_capacity_ba"] = 1
    links.loc[tw_cba, "capacity_ab"] = links.loc[tw_cba, "capacity_ba"]
    links.loc[tw_cba, "estimated_capacity_ab"] = 1

    # Estimate missing from lane_capacity * lanes
    for suffix, est_col, lanes_col in [
        ("ab", "estimated_capacity_ab", "lanes_ab"),
        ("ba", "estimated_capacity_ba", "lanes_ba"),
    ]:
        cap_col = f"capacity_{suffix}"
        cap_missing = links[cap_col].isna() | (links[cap_col] == 0)
        if cap_missing.any() and "link_type" in links.columns:
            for lt, cpl in lane_capacity:
                m = cap_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                links.loc[m, cap_col] = cpl * links.loc[m, lanes_col]
                links.loc[m, est_col] = 1

            remaining = links[cap_col].isna() | (links[cap_col] == 0)
            if remaining.any():
                links.loc[remaining, cap_col] = 900 * links.loc[remaining, lanes_col]
                links.loc[remaining, est_col] = 1

    # ------------------------------------------------------------------
    # 4.5 Experiment profile adjustments (speed/capacity hierarchy)
    # ------------------------------------------------------------------
    links = _apply_experiment_profile(links, experiment_profile)

    # ------------------------------------------------------------------
    # 5. Free-flow travel time (preserve AB/BA)
    # ------------------------------------------------------------------
    if "travel_time_ab" in links.columns:
        links["travel_time_ab"] = pd.to_numeric(links["travel_time_ab"], errors="coerce")
    else:
        links["travel_time_ab"] = None

    if "travel_time_ba" in links.columns:
        links["travel_time_ba"] = pd.to_numeric(links["travel_time_ba"], errors="coerce")
    else:
        links["travel_time_ba"] = None

    # Two-way: copy present direction to missing
    tw_tab = mask_twoway & links["travel_time_ab"].notna() & links["travel_time_ba"].isna()
    tw_tba = mask_twoway & links["travel_time_ba"].notna() & links["travel_time_ab"].isna()
    links.loc[tw_tab, "travel_time_ba"] = links.loc[tw_tab, "travel_time_ab"]
    links.loc[tw_tab, "estimated_free_flow_time_ba"] = 1
    links.loc[tw_tba, "travel_time_ab"] = links.loc[tw_tba, "travel_time_ba"]
    links.loc[tw_tba, "estimated_free_flow_time_ab"] = 1

    # ALWAYS recompute travel time from distance/speed to ensure consistency.
    for tt_col, spd_col in [("travel_time_ab", "speed_ab"), ("travel_time_ba", "speed_ba")]:
        valid_spd = (
            links[spd_col].notna()
            & (links[spd_col] > 0)
            & links["distance"].notna()
            & (links["distance"] > 0)
        )
        links.loc[valid_spd, tt_col] = (
            links.loc[valid_spd, "distance"] * 3.6 / links.loc[valid_spd, spd_col]
        )

    # ------------------------------------------------------------------
    # 5.5 Experiment profile adjustments (time penalties)
    # ------------------------------------------------------------------
    profile = EXPERIMENT_PROFILES.get(experiment_profile, EXPERIMENT_PROFILES["baseline"])
    _apply_time_penalties(links, profile.get("time_penalties", {}))

    # Floor travel time at 0.01s to prevent zero-cost links in BPR
    for tt_col in ("travel_time_ab", "travel_time_ba"):
        links[tt_col] = links[tt_col].clip(lower=0.01)

    # ------------------------------------------------------------------
    # DB write-back (per-direction, preserving asymmetry)
    # ------------------------------------------------------------------
    import sqlite3

    if project_dir is None:
        project_dir = Path(".")

    project_path = Path(project_dir)
    db_files = (
        list(project_path.glob("*.sqlite"))
        + list(project_path.glob("*.db"))
        + list(project_path.glob("*.sqlite3"))
    )

    if not db_files:
        print("Warning: project database not found, DB update skipped")
        return links

    db_path = db_files[0]
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    updates = []
    for _, row in links.iterrows():
        lid = int(row["link_id"])
        updates.append(
            (
                float(row["speed_ab"]),
                float(row["speed_ba"]),
                int(row["lanes_ab"]),
                int(row["lanes_ba"]),
                float(row["capacity_ab"]),
                float(row["capacity_ba"]),
                float(row["travel_time_ab"]),
                float(row["travel_time_ba"]),
                lid,
            )
        )

    cursor.executemany(
        "UPDATE links SET speed_ab=?, speed_ba=?, lanes_ab=?, lanes_ba=?, "
        "capacity_ab=?, capacity_ba=?, travel_time_ab=?, travel_time_ba=? "
        "WHERE link_id=?",
        updates,
    )
    conn.commit()
    conn.close()

    # Diagnostics
    n_asym_speed = int(((links["speed_ab"] - links["speed_ba"]).abs() > 0.1).sum())
    n_asym_lanes = int((links["lanes_ab"] != links["lanes_ba"]).sum())
    n_asym_cap = int(((links["capacity_ab"] - links["capacity_ba"]).abs() > 0.1).sum())
    n_est_speed = int((links["estimated_speed_ab"] | links["estimated_speed_ba"]).sum())
    n_est_cap = int((links["estimated_capacity_ab"] | links["estimated_capacity_ba"]).sum())

    print(f"Updated {len(updates)} links in DB (per-direction)")
    print(f"  Asymmetric: speed={n_asym_speed}, lanes={n_asym_lanes}, capacity={n_asym_cap}")
    print(f"  Estimated:  speed={n_est_speed}, capacity={n_est_cap}")

    return links


def check_connectivity(project: Project) -> Dict[str, Any]:
    """
    Kontrola konektivity sítě a identifikace izolovaných komponent.
    Vrací informace o komponentách a jejich velikostech.
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

        if direction == 1:
            G.add_edge(a_node, b_node, link_id=link["link_id"])
        elif direction == -1:
            G.add_edge(b_node, a_node, link_id=link["link_id"])
        elif direction == 0:
            G.add_edge(a_node, b_node, link_id=link["link_id"])
            G.add_edge(b_node, a_node, link_id=link["link_id"])

    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    components_sorted = sorted(components, key=len, reverse=True)

    largest_component = components_sorted[0] if components_sorted else set()
    isolated_nodes = [node for comp in components_sorted[1:] for node in comp if len(comp) == 1]
    isolated_components = [comp for comp in components_sorted[1:] if len(comp) > 1]

    result = {
        "total_components": len(components_sorted),
        "largest_component_size": len(largest_component),
        "isolated_nodes_count": len(isolated_nodes),
        "isolated_components_count": len(isolated_components),
        "components": [
            {
                "component_id": i,
                "size": len(comp),
                "nodes": list(comp),
            }
            for i, comp in enumerate(components_sorted)
        ],
    }

    return result


def export_stable_network(
    project: Project,
    output_path: Path,
    connectivity_info: Dict[str, Any] | None = None,
    normalized_links: pd.DataFrame | None = None,
) -> None:
    """
    Uloží síť se stabilními ID uzlů/hran do GeoJSON a Parquet.
    Stabilní ID jsou klíčem pro scénáře i delta analýzy.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if normalized_links is not None:
        links = normalized_links.copy()
    else:
        links = project.network.links.data.copy()

    link_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    optional_cols = [
        "link_type", "name", "osm_id",
        "speed_ab", "speed_ba", "lanes_ab", "lanes_ba",
        "capacity_ab", "capacity_ba", "travel_time_ab", "travel_time_ba",
        "speed", "lanes", "capacity", "free_flow_time",
    ]
    available_cols = [col for col in link_cols + optional_cols if col in links.columns]

    estimate_cols = [col for col in links.columns if col.startswith("estimated_")]
    available_cols.extend(estimate_cols)

    links_export = links[available_cols].copy()

    geojson_path = output_path / "network_links.geojson"
    links_export.to_file(geojson_path, driver="GeoJSON")

    links_parquet = links_export.drop(columns=["geometry"]) if "geometry" in links_export.columns else links_export
    parquet_path = output_path / "network_links.parquet"
    links_parquet.to_parquet(parquet_path, index=False)

    nodes = project.network.nodes.data.copy()
    node_cols = ["node_id", "is_centroid"]
    if "geometry" in nodes.columns:
        node_cols.append("geometry")
    available_node_cols = [col for col in node_cols if col in nodes.columns]

    nodes_export = nodes[available_node_cols].copy()

    nodes_geojson_path = output_path / "network_nodes.geojson"
    if "geometry" in nodes_export.columns:
        nodes_export.to_file(nodes_geojson_path, driver="GeoJSON")

    nodes_parquet = nodes_export.drop(columns=["geometry"]) if "geometry" in nodes_export.columns else nodes_export
    nodes_parquet_path = output_path / "network_nodes.parquet"
    nodes_parquet.to_parquet(nodes_parquet_path, index=False)

    if connectivity_info:
        connectivity_path = output_path / "connectivity_report.json"
        connectivity_path.write_text(json.dumps(connectivity_info, indent=2), encoding="utf-8")

    summary = {
        "links_count": len(links),
        "nodes_count": len(nodes),
        "connectivity": connectivity_info,
    }
    summary_path = output_path / "network_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported network to: {output_path}")
    print(f"  - Links: {len(links)}")
    print(f"  - Nodes: {len(nodes)}")
    if connectivity_info:
        print(f"  - Components: {connectivity_info['total_components']}")
        print(f"  - Largest component: {connectivity_info['largest_component_size']} nodes")


def normalize_and_export_network(
    config_path: str | Path = "config/sim.yaml",
    outputs_dir: str | Path | None = None,
) -> None:
    """
    Hlavní funkce: normalizuje atributy, kontroluje konektivitu a exportuje síť.
    """
    cfg = load_config(config_path)
    if outputs_dir is None:
        outputs_dir = cfg.get("network", {}).get("output_dir", "outputs/baseline/network")

    experiment_profile = cfg.get("network", {}).get("experiment_profile", "baseline")
    project_dir = Path(cfg["project_path"])

    project = Project()
    project.open(str(project_dir))

    try:
        print("=== NORMALIZACE ATRIBUTŮ ===")
        links = normalize_network_attributes(
            project,
            project_dir,
            experiment_profile=experiment_profile,
        )
        print(f"Normalizováno {len(links)} hran")

        print("\n=== KONTROLA KONEKTIVITY ===")
        connectivity_info = check_connectivity(project)
        print(f"Počet komponent: {connectivity_info['total_components']}")
        print(f"Největší komponenta: {connectivity_info['largest_component_size']} uzlů")
        print(f"Izolované uzly: {connectivity_info['isolated_nodes_count']}")
        print(f"Izolované komponenty: {connectivity_info['isolated_components_count']}")

        print("\n=== EXPORT SÍTĚ ===")
        out_dir = Path(outputs_dir)
        export_stable_network(project, out_dir, connectivity_info, normalized_links=links)

        print("\n=== NORMALIZACE DOKONČENA ===")

    finally:
        project.close()
