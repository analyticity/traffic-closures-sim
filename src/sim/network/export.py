"""Network export to GeoPackage, GeoJSON, and Parquet.

Pure data-export module -- no direct database mutations.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
from aequilibrae import Project

from sim.io_project import get_metric_epsg

logger = logging.getLogger(__name__)


def export_stable_network(
    project: Project,
    output_path: Path,
    connectivity_info: Dict[str, Any] | None = None,
    normalized_links: pd.DataFrame | None = None,
    output_crs_epsg: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Export network with stable node/link IDs to GeoPackage (projected CRS),
    GeoJSON (WGS84 for maps), and Parquet (no geometry)."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_crs_epsg is None:
        output_crs_epsg = get_metric_epsg(cfg or {})

    links = normalized_links.copy() if normalized_links is not None else project.network.links.data.copy()

    link_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    optional_cols = [
        "link_type", "name", "osm_id", "osm_ref", "osm_ref_norm",
        "osm_name_raw", "osm_highway",
        "speed_ab", "speed_ba", "lanes_ab", "lanes_ba",
        "capacity_ab", "capacity_ba", "travel_time_ab", "travel_time_ba",
        "speed", "lanes", "capacity", "free_flow_time",
    ]
    available_cols = [c for c in link_cols + optional_cols if c in links.columns]
    available_cols.extend(c for c in links.columns if c.startswith("estimated_"))
    links_export = links[available_cols].copy()

    export_meta: Dict[str, Any] = {
        "distance_column_units": "meters",
        "projected_crs_epsg": int(output_crs_epsg),
    }

    if "geometry" in links_export.columns:
        links_gdf = gpd.GeoDataFrame(
            links_export, geometry="geometry",
            crs=getattr(links_export, "crs", None),
        )
        if links_gdf.crs is None:
            links_gdf = links_gdf.set_crs(epsg=int(output_crs_epsg), allow_override=True)

        links_gdf.to_file(output_path / "network_links.gpkg", driver="GPKG", layer="links")
        links_gdf.to_crs(epsg=4326).to_file(output_path / "network_links.geojson", driver="GeoJSON")

        links_parquet = links_gdf.drop(columns=["geometry"])
        export_meta["network_links.gpkg"] = {"crs_epsg": int(output_crs_epsg), "aligned_with_distance": True}
        export_meta["network_links.geojson"] = {
            "crs_epsg": 4326,
            "note": "WGS84 for web maps; lengths in degrees — use `distance` (m) or GPKG for metrics.",
        }
    else:
        links_parquet = links_export
        export_meta["network_links.geojson"] = {"skipped": True, "reason": "no geometry column"}

    links_parquet.to_parquet(output_path / "network_links.parquet", index=False)

    # --- Nodes ---
    nodes = project.network.nodes.data.copy()
    node_cols = ["node_id", "osm_id", "is_centroid"]
    if "geometry" in nodes.columns:
        node_cols.append("geometry")
    available_node_cols = [c for c in node_cols if c in nodes.columns]
    nodes_export = nodes[available_node_cols].copy()

    if "geometry" in nodes_export.columns:
        nodes_gdf = gpd.GeoDataFrame(
            nodes_export, geometry="geometry",
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

    nodes_parquet.to_parquet(output_path / "network_nodes.parquet", index=False)

    # --- Metadata files ---
    if connectivity_info:
        (output_path / "connectivity_report.json").write_text(
            json.dumps(connectivity_info, indent=2), encoding="utf-8",
        )

    summary = {
        "links_count": len(links),
        "nodes_count": len(nodes),
        "connectivity": connectivity_info,
        "exports": export_meta,
    }
    (output_path / "network_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Exported network to %s (links: %d, nodes: %d)", output_path, len(links), len(nodes))
    if "geometry" in links_export.columns:
        logger.info(
            "Wrote network_links.gpkg (EPSG:%d) and network_links.geojson (EPSG:4326, for maps only)",
            output_crs_epsg,
        )
    if connectivity_info:
        logger.info(
            "Components: %d, largest: %d nodes",
            connectivity_info["total_components"], connectivity_info["largest_component_size"],
        )
        if "directed_scc_count" in connectivity_info:
            logger.info(
                "Directed SCCs: %d, largest SCC: %d nodes",
                connectivity_info["directed_scc_count"], connectivity_info["largest_scc_size"],
            )
            if connectivity_info.get("major_links_outside_scc", 0) > 0:
                logger.warning("%d major links outside largest SCC", connectivity_info["major_links_outside_scc"])
