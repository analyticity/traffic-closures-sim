from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from aequilibrae import Project

from sim.aequilibrae_paths import resolve_project_database_path
from sim.io_project import load_config


def _resolved_experiment_profile(network_cfg: dict, experiment_profile: str) -> dict:
    profiles = network_cfg.get("experiment_profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    prof = profiles.get(experiment_profile)
    if prof is None:
        prof = profiles.get("baseline")
    if prof is None:
        prof = {
            "speed_caps": {},
            "speed_floors": {},
            "capacity_factors": {},
            "time_penalties": {},
        }
    return prof


def _normalization_defaults(network_cfg: dict) -> tuple[dict, dict]:
    norm = network_cfg.get("normalization")
    if not isinstance(norm, dict) or not isinstance(norm.get("defaults"), dict):
        raise ValueError(
            "network.normalization.defaults is missing. Set network.normalization_config in "
            "sim.yaml (see config/network_normalization.yaml) or define network.normalization "
            "inline."
        )
    defaults = norm["defaults"]
    for key in ("speed_by_link_type", "lanes_by_link_type", "capacity_per_lane_by_link_type"):
        if key not in defaults or not isinstance(defaults[key], dict):
            raise ValueError(f"network.normalization.defaults.{key} must be a mapping")
    thresholds = norm.get("thresholds") if isinstance(norm.get("thresholds"), dict) else {}
    return defaults, thresholds


def _threshold(thresholds: dict, key: str, default: float) -> float:
    if key not in thresholds:
        return float(default)
    return float(thresholds[key])


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


def _apply_experiment_profile(
    links: pd.DataFrame,
    profile: dict,
    thresholds: dict,
) -> pd.DataFrame:
    _apply_speed_floors(links, profile.get("speed_floors") or {})
    _apply_speed_caps(links, profile.get("speed_caps") or {})
    _apply_capacity_factors(links, profile.get("capacity_factors") or {})

    min_spd = _threshold(thresholds, "min_speed_kmh", 5.0)
    min_cap = _threshold(thresholds, "min_capacity_vph", 50.0)
    links["speed_ab"] = links["speed_ab"].clip(lower=min_spd)
    links["speed_ba"] = links["speed_ba"].clip(lower=min_spd)
    links["capacity_ab"] = links["capacity_ab"].clip(lower=min_cap)
    links["capacity_ba"] = links["capacity_ba"].clip(lower=min_cap)

    return links


def normalize_network_attributes(
    project: Project,
    network_cfg: dict,
    experiment_profile: str = "baseline",
) -> pd.DataFrame:
    """Normalize/compute link attributes, preserving AB/BA directional asymmetry.

    For two-way links (direction=0), when both AB and BA values exist from OSM,
    they are kept separately -- never averaged. Only when a value is missing
    is a default or the opposite-direction value used as fallback.

    Writes back per-direction values (speed_ab, speed_ba, ...) to the SQLite DB.
    """
    defaults, thresholds = _normalization_defaults(network_cfg)
    default_speeds = defaults["speed_by_link_type"]
    default_lanes = defaults["lanes_by_link_type"]
    cap_per_lane = defaults["capacity_per_lane_by_link_type"]
    fallback_speed = _threshold(thresholds, "fallback_speed_kmh", 50.0)
    generic_cpl = _threshold(thresholds, "generic_capacity_per_lane", 900.0)
    min_tt = _threshold(thresholds, "min_travel_time_s", 0.01)

    profile = _resolved_experiment_profile(network_cfg, experiment_profile)
    print(f"Applying network experiment profile: {experiment_profile}")

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
    def _resolve_directional(
        col_ab: str,
        col_ba: str,
        default_map: dict,
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
            for lt, dv in default_map.items():
                m_ab = still_ab_missing & links["link_type"].astype(str).str.fullmatch(
                    lt, case=False
                )
                m_ba = still_ba_missing & links["link_type"].astype(str).str.fullmatch(
                    lt, case=False
                )
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
        fallback_speed,
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
    # Total veh/h per direction: existing OSM/Aeq values kept; gaps filled as
    # capacity_* = capacity_per_lane[link_type] * lanes_* (see network_normalization.yaml).
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

    # Estimate missing from capacity_per_lane * lanes
    for suffix, est_col, lanes_col in [
        ("ab", "estimated_capacity_ab", "lanes_ab"),
        ("ba", "estimated_capacity_ba", "lanes_ba"),
    ]:
        cap_col = f"capacity_{suffix}"
        cap_missing = links[cap_col].isna() | (links[cap_col] == 0)
        if cap_missing.any() and "link_type" in links.columns:
            for lt, cpl in cap_per_lane.items():
                m = cap_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                links.loc[m, cap_col] = float(cpl) * links.loc[m, lanes_col]
                links.loc[m, est_col] = 1

            remaining = links[cap_col].isna() | (links[cap_col] == 0)
            if remaining.any():
                links.loc[remaining, cap_col] = generic_cpl * links.loc[remaining, lanes_col]
                links.loc[remaining, est_col] = 1

    # ------------------------------------------------------------------
    # 4.5 Experiment profile adjustments (speed/capacity hierarchy)
    # ------------------------------------------------------------------
    links = _apply_experiment_profile(links, profile, thresholds)

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
    _apply_time_penalties(links, profile.get("time_penalties") or {})

    # Floor travel time to prevent zero-cost links in BPR
    for tt_col in ("travel_time_ab", "travel_time_ba"):
        links[tt_col] = links[tt_col].clip(lower=min_tt)

    # ------------------------------------------------------------------
    # DB write-back (per-direction, preserving asymmetry)
    # ------------------------------------------------------------------
    db_path = resolve_project_database_path(project)
    if not db_path.is_file():
        print(f"Warning: project database not found at {db_path}, DB update skipped")
        return links

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
    output_crs_epsg: Optional[int] = None,
) -> None:
    """
    Uloží síť se stabilními ID uzlů/hran do GeoPackage (projekční CRS = sloupec distance v m),
    GeoJSON (WGS84 pro mapy), Parquet (bez geometrie).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_crs_epsg is None:
        output_crs_epsg = 5514

    if normalized_links is not None:
        links = normalized_links.copy()
    else:
        links = project.network.links.data.copy()

    link_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    optional_cols = [
        "link_type",
        "name",
        "osm_id",
        "osm_ref",
        "osm_ref_norm",
        "osm_name_raw",
        "osm_highway",
        "speed_ab",
        "speed_ba",
        "lanes_ab",
        "lanes_ba",
        "capacity_ab",
        "capacity_ba",
        "travel_time_ab",
        "travel_time_ba",
        "speed",
        "lanes",
        "capacity",
        "free_flow_time",
    ]
    available_cols = [col for col in link_cols + optional_cols if col in links.columns]

    estimate_cols = [col for col in links.columns if col.startswith("estimated_")]
    available_cols.extend(estimate_cols)

    links_export = links[available_cols].copy()

    export_meta: Dict[str, Any] = {
        "distance_column_units": "meters",
        "projected_crs_epsg": int(output_crs_epsg),
    }

    if "geometry" in links_export.columns:
        links_gdf = gpd.GeoDataFrame(
            links_export,
            geometry="geometry",
            crs=getattr(links_export, "crs", None),
        )
        if links_gdf.crs is None:
            links_gdf = links_gdf.set_crs(epsg=int(output_crs_epsg), allow_override=True)

        gpkg_links = output_path / "network_links.gpkg"
        links_gdf.to_file(gpkg_links, driver="GPKG", layer="links")

        links_wgs84 = links_gdf.to_crs(epsg=4326)
        links_wgs84.to_file(output_path / "network_links.geojson", driver="GeoJSON")

        links_parquet = links_gdf.drop(columns=["geometry"])
        export_meta["network_links.gpkg"] = {
            "crs_epsg": int(output_crs_epsg),
            "aligned_with_distance": True,
        }
        export_meta["network_links.geojson"] = {
            "crs_epsg": 4326,
            "note": "WGS84 for web maps; lengths in degrees — use `distance` (m) or GPKG for metrics.",
        }
    else:
        links_parquet = links_export
        export_meta["network_links.geojson"] = {"skipped": True, "reason": "no geometry column"}

    parquet_path = output_path / "network_links.parquet"
    links_parquet.to_parquet(parquet_path, index=False)

    nodes = project.network.nodes.data.copy()
    node_cols = ["node_id", "osm_id", "is_centroid"]
    if "geometry" in nodes.columns:
        node_cols.append("geometry")
    available_node_cols = [col for col in node_cols if col in nodes.columns]

    nodes_export = nodes[available_node_cols].copy()

    if "geometry" in nodes_export.columns:
        nodes_gdf = gpd.GeoDataFrame(
            nodes_export,
            geometry="geometry",
            crs=getattr(nodes_export, "crs", None),
        )
        if nodes_gdf.crs is None:
            nodes_gdf = nodes_gdf.set_crs(epsg=int(output_crs_epsg), allow_override=True)
        nodes_gdf.to_file(output_path / "network_nodes.gpkg", driver="GPKG", layer="nodes")
        nodes_gdf.to_crs(epsg=4326).to_file(output_path / "network_nodes.geojson", driver="GeoJSON")
        nodes_parquet = nodes_gdf.drop(columns=["geometry"])
        export_meta["network_nodes.gpkg"] = {"crs_epsg": int(output_crs_epsg)}
        export_meta["network_nodes.geojson"] = {"crs_epsg": 4326}
    else:
        nodes_parquet = nodes_export

    nodes_parquet_path = output_path / "network_nodes.parquet"
    nodes_parquet.to_parquet(nodes_parquet_path, index=False)

    if connectivity_info:
        connectivity_path = output_path / "connectivity_report.json"
        connectivity_path.write_text(json.dumps(connectivity_info, indent=2), encoding="utf-8")

    summary = {
        "links_count": len(links),
        "nodes_count": len(nodes),
        "connectivity": connectivity_info,
        "exports": export_meta,
    }
    summary_path = output_path / "network_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported network to: {output_path}")
    print(f"  - Links: {len(links)}")
    print(f"  - Nodes: {len(nodes)}")
    if "geometry" in links_export.columns:
        print(f"  - network_links.gpkg (EPSG:{output_crs_epsg}, metres, aligns with distance)")
        print("  - network_links.geojson (EPSG:4326, for maps only)")
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

    network_cfg = cfg.get("network") or {}
    experiment_profile = network_cfg.get("experiment_profile", "baseline")
    project_dir = Path(cfg["project_path"])

    project = Project()
    project.open(str(project_dir))

    try:
        print("=== NORMALIZACE ATRIBUTŮ ===")
        links = normalize_network_attributes(
            project,
            network_cfg,
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
        crs_epsg = int(cfg.get("crs_epsg", 5514))
        export_stable_network(
            project,
            out_dir,
            connectivity_info,
            normalized_links=links,
            output_crs_epsg=crs_epsg,
        )

        print("\n=== NORMALIZACE DOKONČENA ===")

    finally:
        project.close()
