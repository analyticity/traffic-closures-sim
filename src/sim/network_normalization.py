"""Network normalization: attribute computation, connectivity check, stable ID export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import networkx as nx
import pandas as pd
from aequilibrae import Project

from sim.io_project import load_config


def normalize_network_attributes(project: Project, project_dir: Path | None = None) -> pd.DataFrame:
    """
    Normalizuje/dopočítá atributy sítě s prioritou hodnot z OSM:
    - délka (distance) - z geometrie pokud chybí
    - rychlost (speed) - použije speed_ab/speed_ba z OSM, odhad pouze pokud chybí
    - pruhy (lanes) - použije lanes_ab/lanes_ba z OSM, odhad pouze pokud chybí
    - kapacita (capacity) - odhad podle typu linku a pruhů (OSM obvykle nemá)
    - volný čas (free_flow_time) - použije travel_time_ab/travel_time_ba z OSM, dopočítá pokud chybí
    - příznaky odhadu (estimated_*)
    """
    links = project.network.links.data.copy()
    
    # Přidání příznaků odhadu
    links["estimated_distance"] = 0
    links["estimated_speed"] = 0
    links["estimated_capacity"] = 0
    links["estimated_lanes"] = 0
    links["estimated_free_flow_time"] = 0
    
    # 1. Délka (distance) - z geometrie pokud chybí nebo je 0
    if "distance" not in links.columns or links["distance"].isna().any() or (links["distance"] == 0).any():
        missing_dist = links["distance"].isna() | (links["distance"] == 0)
        if missing_dist.any():
            links.loc[missing_dist, "distance"] = links.loc[missing_dist, "geometry"].length
            links.loc[missing_dist, "estimated_distance"] = 1
    
    # 2. Rychlost (speed) - použij hodnoty z OSM (speed_ab/speed_ba), odhad pouze pokud chybí
    # Vytvoření unified speed sloupce z speed_ab/speed_ba podle direction
    if "speed_ab" in links.columns or "speed_ba" in links.columns:
        # Převod na numerické hodnoty (může obsahovat stringy)
        if "speed_ab" in links.columns:
            links["speed_ab"] = pd.to_numeric(links["speed_ab"], errors="coerce")
        if "speed_ba" in links.columns:
            links["speed_ba"] = pd.to_numeric(links["speed_ba"], errors="coerce")
        
        # Pro one-way links použij příslušný směr
        # Pro two-way použij průměr nebo speed_ab
        links["speed"] = None
        
        # One-way A->B (direction == 1)
        mask_ab = links["direction"] == 1
        if "speed_ab" in links.columns:
            links.loc[mask_ab & links["speed_ab"].notna(), "speed"] = links.loc[mask_ab & links["speed_ab"].notna(), "speed_ab"]
        
        # One-way B->A (direction == -1)
        mask_ba = links["direction"] == -1
        if "speed_ba" in links.columns:
            links.loc[mask_ba & links["speed_ba"].notna(), "speed"] = links.loc[mask_ba & links["speed_ba"].notna(), "speed_ba"]
        
        # Two-way (direction == 0) - použij speed_ab nebo průměr
        mask_twoway = links["direction"] == 0
        if "speed_ab" in links.columns and "speed_ba" in links.columns:
            # Pokud máme oba, použij průměr
            both_available = mask_twoway & links["speed_ab"].notna() & links["speed_ba"].notna()
            links.loc[both_available, "speed"] = (links.loc[both_available, "speed_ab"] + links.loc[both_available, "speed_ba"]) / 2
            # Jinak použij co je k dispozici
            only_ab = mask_twoway & links["speed_ab"].notna() & links["speed_ba"].isna()
            links.loc[only_ab, "speed"] = links.loc[only_ab, "speed_ab"]
            only_ba = mask_twoway & links["speed_ba"].notna() & links["speed_ab"].isna()
            links.loc[only_ba, "speed"] = links.loc[only_ba, "speed_ba"]
        elif "speed_ab" in links.columns:
            links.loc[mask_twoway & links["speed_ab"].notna(), "speed"] = links.loc[mask_twoway & links["speed_ab"].notna(), "speed_ab"]
        elif "speed_ba" in links.columns:
            links.loc[mask_twoway & links["speed_ba"].notna(), "speed"] = links.loc[mask_twoway & links["speed_ba"].notna(), "speed_ba"]
    
    # Odhad rychlosti pouze pro chybějící hodnoty
    if "speed" not in links.columns or links["speed"].isna().any():
        if "speed" not in links.columns:
            links["speed"] = None
        
        default_speeds = {
            "motorway": 120.0,
            "motorway_link": 100.0,
            "trunk": 100.0,
            "trunk_link": 80.0,
            "primary": 80.0,
            "primary_link": 60.0,
            "secondary": 60.0,
            "secondary_link": 50.0,
            "tertiary": 50.0,
            "tertiary_link": 40.0,
            "residential": 30.0,
            "unclassified": 50.0,
            "service": 30.0,
        }
        
        missing_speed = links["speed"].isna()
        if missing_speed.any() and "link_type" in links.columns:
            for link_type, default_speed in default_speeds.items():
                mask = missing_speed & (links["link_type"].str.contains(link_type, case=False, na=False))
                links.loc[mask, "speed"] = default_speed
                links.loc[mask, "estimated_speed"] = 1
            # Fallback pro zbývající
            remaining = links["speed"].isna()
            if remaining.any():
                links.loc[remaining, "speed"] = 50.0
                links.loc[remaining, "estimated_speed"] = 1
    
    # 3. Pruhy (lanes) - použij hodnoty z OSM (lanes_ab/lanes_ba), odhad pouze pokud chybí
    if "lanes_ab" in links.columns or "lanes_ba" in links.columns:
        # Převod na numerické hodnoty
        if "lanes_ab" in links.columns:
            links["lanes_ab"] = pd.to_numeric(links["lanes_ab"], errors="coerce")
        if "lanes_ba" in links.columns:
            links["lanes_ba"] = pd.to_numeric(links["lanes_ba"], errors="coerce")
        
        links["lanes"] = None
        
        # One-way A->B
        mask_ab = links["direction"] == 1
        if "lanes_ab" in links.columns:
            links.loc[mask_ab & links["lanes_ab"].notna(), "lanes"] = links.loc[mask_ab & links["lanes_ab"].notna(), "lanes_ab"]
        
        # One-way B->A
        mask_ba = links["direction"] == -1
        if "lanes_ba" in links.columns:
            links.loc[mask_ba & links["lanes_ba"].notna(), "lanes"] = links.loc[mask_ba & links["lanes_ba"].notna(), "lanes_ba"]
        
        # Two-way - použij průměr nebo lanes_ab
        mask_twoway = links["direction"] == 0
        if "lanes_ab" in links.columns and "lanes_ba" in links.columns:
            both_available = mask_twoway & links["lanes_ab"].notna() & links["lanes_ba"].notna()
            links.loc[both_available, "lanes"] = (links.loc[both_available, "lanes_ab"] + links.loc[both_available, "lanes_ba"]) / 2
            only_ab = mask_twoway & links["lanes_ab"].notna() & links["lanes_ba"].isna()
            links.loc[only_ab, "lanes"] = links.loc[only_ab, "lanes_ab"]
            only_ba = mask_twoway & links["lanes_ba"].notna() & links["lanes_ab"].isna()
            links.loc[only_ba, "lanes"] = links.loc[only_ba, "lanes_ba"]
        elif "lanes_ab" in links.columns:
            links.loc[mask_twoway & links["lanes_ab"].notna(), "lanes"] = links.loc[mask_twoway & links["lanes_ab"].notna(), "lanes_ab"]
        elif "lanes_ba" in links.columns:
            links.loc[mask_twoway & links["lanes_ba"].notna(), "lanes"] = links.loc[mask_twoway & links["lanes_ba"].notna(), "lanes_ba"]
    
    # Odhad pruhů pouze pro chybějící hodnoty
    if "lanes" not in links.columns or links["lanes"].isna().any() or (links["lanes"] == 0).any():
        if "lanes" not in links.columns:
            links["lanes"] = None
        
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
            "residential": 1,
            "unclassified": 1,
            "service": 1,
        }
        
        missing_lanes = links["lanes"].isna() | (links["lanes"] == 0)
        if missing_lanes.any() and "link_type" in links.columns:
            for link_type, default_lane_count in default_lanes.items():
                mask = missing_lanes & (links["link_type"].str.contains(link_type, case=False, na=False))
                links.loc[mask, "lanes"] = default_lane_count
                links.loc[mask, "estimated_lanes"] = 1
            # Fallback
            remaining = links["lanes"].isna() | (links["lanes"] == 0)
            if remaining.any():
                links.loc[remaining, "lanes"] = 1
                links.loc[remaining, "estimated_lanes"] = 1
    
    # 4. Kapacita (capacity) - OSM obvykle nemá, takže odhad podle typu a pruhů
    # Kapacita na pruh podle typu
    lane_capacity = {
        "motorway": 1800,
        "motorway_link": 1500,
        "trunk": 1500,
        "trunk_link": 1200,
        "primary": 1000,
        "primary_link": 900,
        "secondary": 900,
        "secondary_link": 800,
        "tertiary": 800,
        "tertiary_link": 700,
        "residential": 600,
        "unclassified": 700,
        "service": 600,
    }
    
    # Zkus použít capacity_ab/capacity_ba pokud existují
    if "capacity_ab" in links.columns or "capacity_ba" in links.columns:
        # Převod na numerické hodnoty
        if "capacity_ab" in links.columns:
            links["capacity_ab"] = pd.to_numeric(links["capacity_ab"], errors="coerce")
        if "capacity_ba" in links.columns:
            links["capacity_ba"] = pd.to_numeric(links["capacity_ba"], errors="coerce")
        
        links["capacity"] = None
        mask_ab = links["direction"] == 1
        if "capacity_ab" in links.columns:
            links.loc[mask_ab & links["capacity_ab"].notna(), "capacity"] = links.loc[mask_ab & links["capacity_ab"].notna(), "capacity_ab"]
        mask_ba = links["direction"] == -1
        if "capacity_ba" in links.columns:
            links.loc[mask_ba & links["capacity_ba"].notna(), "capacity"] = links.loc[mask_ba & links["capacity_ba"].notna(), "capacity_ba"]
        mask_twoway = links["direction"] == 0
        if "capacity_ab" in links.columns and "capacity_ba" in links.columns:
            both_available = mask_twoway & links["capacity_ab"].notna() & links["capacity_ba"].notna()
            links.loc[both_available, "capacity"] = (links.loc[both_available, "capacity_ab"] + links.loc[both_available, "capacity_ba"]) / 2
            only_ab = mask_twoway & links["capacity_ab"].notna() & links["capacity_ba"].isna()
            links.loc[only_ab, "capacity"] = links.loc[only_ab, "capacity_ab"]
            only_ba = mask_twoway & links["capacity_ba"].notna() & links["capacity_ab"].isna()
            links.loc[only_ba, "capacity"] = links.loc[only_ba, "capacity_ba"]
        elif "capacity_ab" in links.columns:
            links.loc[mask_twoway & links["capacity_ab"].notna(), "capacity"] = links.loc[mask_twoway & links["capacity_ab"].notna(), "capacity_ab"]
        elif "capacity_ba" in links.columns:
            links.loc[mask_twoway & links["capacity_ba"].notna(), "capacity"] = links.loc[mask_twoway & links["capacity_ba"].notna(), "capacity_ba"]
    
    # Odhad kapacity pro chybějící hodnoty
    if "capacity" not in links.columns or links["capacity"].isna().any() or (links["capacity"] == 0).any():
        if "capacity" not in links.columns:
            links["capacity"] = None
        
        missing_capacity = links["capacity"].isna() | (links["capacity"] == 0)
        if missing_capacity.any() and "link_type" in links.columns:
            for link_type, cap_per_lane in lane_capacity.items():
                mask = missing_capacity & (links["link_type"].str.contains(link_type, case=False, na=False))
                # Kapacita = kapacita na pruh * počet pruhů
                links.loc[mask, "capacity"] = cap_per_lane * links.loc[mask, "lanes"]
                links.loc[mask, "estimated_capacity"] = 1
            # Fallback
            remaining = links["capacity"].isna() | (links["capacity"] == 0)
            if remaining.any():
                links.loc[remaining, "capacity"] = 900 * links.loc[remaining, "lanes"]
                links.loc[remaining, "estimated_capacity"] = 1
    
    # 5. Volný čas (free_flow_time) - použij travel_time z OSM, dopočítá pokud chybí
    if "travel_time_ab" in links.columns or "travel_time_ba" in links.columns:
        # Převod na numerické hodnoty
        if "travel_time_ab" in links.columns:
            links["travel_time_ab"] = pd.to_numeric(links["travel_time_ab"], errors="coerce")
        if "travel_time_ba" in links.columns:
            links["travel_time_ba"] = pd.to_numeric(links["travel_time_ba"], errors="coerce")
        
        links["free_flow_time"] = None
        
        # One-way A->B
        mask_ab = links["direction"] == 1
        if "travel_time_ab" in links.columns:
            links.loc[mask_ab & links["travel_time_ab"].notna(), "free_flow_time"] = links.loc[mask_ab & links["travel_time_ab"].notna(), "travel_time_ab"]
        
        # One-way B->A
        mask_ba = links["direction"] == -1
        if "travel_time_ba" in links.columns:
            links.loc[mask_ba & links["travel_time_ba"].notna(), "free_flow_time"] = links.loc[mask_ba & links["travel_time_ba"].notna(), "travel_time_ba"]
        
        # Two-way
        mask_twoway = links["direction"] == 0
        if "travel_time_ab" in links.columns and "travel_time_ba" in links.columns:
            both_available = mask_twoway & links["travel_time_ab"].notna() & links["travel_time_ba"].notna()
            links.loc[both_available, "free_flow_time"] = (links.loc[both_available, "travel_time_ab"] + links.loc[both_available, "travel_time_ba"]) / 2
            only_ab = mask_twoway & links["travel_time_ab"].notna() & links["travel_time_ba"].isna()
            links.loc[only_ab, "free_flow_time"] = links.loc[only_ab, "travel_time_ab"]
            only_ba = mask_twoway & links["travel_time_ba"].notna() & links["travel_time_ab"].isna()
            links.loc[only_ba, "free_flow_time"] = links.loc[only_ba, "travel_time_ba"]
        elif "travel_time_ab" in links.columns:
            links.loc[mask_twoway & links["travel_time_ab"].notna(), "free_flow_time"] = links.loc[mask_twoway & links["travel_time_ab"].notna(), "travel_time_ab"]
        elif "travel_time_ba" in links.columns:
            links.loc[mask_twoway & links["travel_time_ba"].notna(), "free_flow_time"] = links.loc[mask_twoway & links["travel_time_ba"].notna(), "travel_time_ba"]
    
    # Dopočítání free_flow_time z distance a speed pro chybějící hodnoty
    if "free_flow_time" not in links.columns or links["free_flow_time"].isna().any() or (links["free_flow_time"] == 0).any():
        if "free_flow_time" not in links.columns:
            links["free_flow_time"] = None
        
        missing_time = links["free_flow_time"].isna() | (links["free_flow_time"] == 0)
        if missing_time.any():
            # time (s) = distance (m) * 3.6 / speed (km/h)
            links.loc[missing_time, "free_flow_time"] = (
                links.loc[missing_time, "distance"] * 3.6 / links.loc[missing_time, "speed"]
            )
            links.loc[missing_time, "estimated_free_flow_time"] = 1
    
    # Aktualizace v projektu pomocí SQL
    # Najdeme SQLite databázi projektu
    import sqlite3
    
    if project_dir is None:
        # Zkus najít databázi v aktuálním adresáři nebo použít config
        project_dir = Path(".")
    
    project_path = Path(project_dir)
    db_files = list(project_path.glob("*.sqlite")) + list(project_path.glob("*.db")) + list(project_path.glob("*.sqlite3"))
    
    if not db_files:
        print("Varování: Nenalezena databáze projektu, aktualizace přeskočena")
        return links
    
    db_path = db_files[0]
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    updated_count = 0
    
    # Aktualizace speed_ab/speed_ba podle direction
    if "speed" in links.columns:
        for _, row in links.iterrows():
            if pd.notna(row.get("speed")):
                link_id = int(row["link_id"])
                direction = row.get("direction", 1)
                speed_val = float(row["speed"])
                
                if direction == 1:  # A->B
                    cursor.execute("UPDATE links SET speed_ab = ? WHERE link_id = ?", (speed_val, link_id))
                elif direction == -1:  # B->A
                    cursor.execute("UPDATE links SET speed_ba = ? WHERE link_id = ?", (speed_val, link_id))
                else:  # two-way
                    cursor.execute("UPDATE links SET speed_ab = ?, speed_ba = ? WHERE link_id = ?", (speed_val, speed_val, link_id))
                updated_count += 1
    
    # Aktualizace lanes_ab/lanes_ba
    if "lanes" in links.columns:
        for _, row in links.iterrows():
            if pd.notna(row.get("lanes")):
                link_id = int(row["link_id"])
                direction = row.get("direction", 1)
                lanes_val = int(row["lanes"])
                
                if direction == 1:  # A->B
                    cursor.execute("UPDATE links SET lanes_ab = ? WHERE link_id = ?", (lanes_val, link_id))
                elif direction == -1:  # B->A
                    cursor.execute("UPDATE links SET lanes_ba = ? WHERE link_id = ?", (lanes_val, link_id))
                else:  # two-way
                    cursor.execute("UPDATE links SET lanes_ab = ?, lanes_ba = ? WHERE link_id = ?", (lanes_val, lanes_val, link_id))
    
    # Aktualizace capacity_ab/capacity_ba
    if "capacity" in links.columns:
        for _, row in links.iterrows():
            if pd.notna(row.get("capacity")):
                link_id = int(row["link_id"])
                direction = row.get("direction", 1)
                capacity_val = float(row["capacity"])
                
                if direction == 1:  # A->B
                    cursor.execute("UPDATE links SET capacity_ab = ? WHERE link_id = ?", (capacity_val, link_id))
                elif direction == -1:  # B->A
                    cursor.execute("UPDATE links SET capacity_ba = ? WHERE link_id = ?", (capacity_val, link_id))
                else:  # two-way
                    cursor.execute("UPDATE links SET capacity_ab = ?, capacity_ba = ? WHERE link_id = ?", (capacity_val, capacity_val, link_id))
    
    # Aktualizace travel_time_ab/travel_time_ba
    if "free_flow_time" in links.columns:
        for _, row in links.iterrows():
            if pd.notna(row.get("free_flow_time")):
                link_id = int(row["link_id"])
                direction = row.get("direction", 1)
                time_val = float(row["free_flow_time"])
                
                if direction == 1:  # A->B
                    cursor.execute("UPDATE links SET travel_time_ab = ? WHERE link_id = ?", (time_val, link_id))
                elif direction == -1:  # B->A
                    cursor.execute("UPDATE links SET travel_time_ba = ? WHERE link_id = ?", (time_val, link_id))
                else:  # two-way
                    cursor.execute("UPDATE links SET travel_time_ab = ?, travel_time_ba = ? WHERE link_id = ?", (time_val, time_val, link_id))
    
    conn.commit()
    conn.close()
    
    print(f"Aktualizováno {updated_count} hran v databázi")
    
    return links


def check_connectivity(project: Project) -> Dict[str, Any]:
    """
    Kontrola konektivity sítě a identifikace izolovaných komponent.
    Vrací informace o komponentách a jejich velikostech.
    """
    links = project.network.links.data
    nodes = project.network.nodes.data
    
    # Vytvoření grafu
    G = nx.DiGraph()
    
    # Přidání uzlů
    for _, node in nodes.iterrows():
        G.add_node(node["node_id"])
    
    # Přidání hran podle směru
    for _, link in links.iterrows():
        a_node = link["a_node"]
        b_node = link["b_node"]
        direction = link.get("direction", 1)
        
        if direction == 1:  # one-way: a -> b
            G.add_edge(a_node, b_node, link_id=link["link_id"])
        elif direction == -1:  # one-way: b -> a
            G.add_edge(b_node, a_node, link_id=link["link_id"])
        elif direction == 0:  # two-way: oba směry
            G.add_edge(a_node, b_node, link_id=link["link_id"])
            G.add_edge(b_node, a_node, link_id=link["link_id"])
    
    # Nalezení komponent (pro neorientovaný graf)
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    
    # Seřazení podle velikosti (největší první)
    components_sorted = sorted(components, key=len, reverse=True)
    
    # Statistiky
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
    
    # Export links - použij normalized links pokud jsou k dispozici, jinak načti z projektu
    if normalized_links is not None:
        links = normalized_links.copy()
    else:
        links = project.network.links.data.copy()
    
    # Zajištění stabilních sloupců
    link_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    optional_cols = ["link_type", "name", "speed", "lanes", "capacity", "free_flow_time", "osm_id"]
    available_cols = [col for col in link_cols + optional_cols if col in links.columns]
    
    # Přidání příznaků odhadu pokud existují
    estimate_cols = [col for col in links.columns if col.startswith("estimated_")]
    available_cols.extend(estimate_cols)
    
    links_export = links[available_cols].copy()
    
    # GeoJSON export
    geojson_path = output_path / "network_links.geojson"
    links_export.to_file(geojson_path, driver="GeoJSON")
    
    # Parquet export (bez geometrie pro lepší kompatibilitu)
    links_parquet = links_export.drop(columns=["geometry"]) if "geometry" in links_export.columns else links_export
    parquet_path = output_path / "network_links.parquet"
    links_parquet.to_parquet(parquet_path, index=False)
    
    # Export nodes
    nodes = project.network.nodes.data.copy()
    node_cols = ["node_id", "is_centroid"]
    if "geometry" in nodes.columns:
        node_cols.append("geometry")
    available_node_cols = [col for col in node_cols if col in nodes.columns]
    
    nodes_export = nodes[available_node_cols].copy()
    
    # GeoJSON export nodes
    nodes_geojson_path = output_path / "network_nodes.geojson"
    if "geometry" in nodes_export.columns:
        nodes_export.to_file(nodes_geojson_path, driver="GeoJSON")
    
    # Parquet export nodes
    nodes_parquet = nodes_export.drop(columns=["geometry"]) if "geometry" in nodes_export.columns else nodes_export
    nodes_parquet_path = output_path / "network_nodes.parquet"
    nodes_parquet.to_parquet(nodes_parquet_path, index=False)
    
    # Export connectivity info
    if connectivity_info:
        connectivity_path = output_path / "connectivity_report.json"
        connectivity_path.write_text(json.dumps(connectivity_info, indent=2), encoding="utf-8")
    
    # Export summary
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

    project_dir = Path(cfg["project_path"])

    project = Project()
    project.open(str(project_dir))
    
    try:
        print("=== NORMALIZACE ATRIBUTŮ ===")
        links = normalize_network_attributes(project, project_dir)
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

