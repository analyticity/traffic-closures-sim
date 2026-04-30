"""``sim.network`` -- modular network-build and normalization pipeline.

Public API re-exported here for convenience.  Downstream code can import
from ``sim.network`` directly or from the individual sub-modules.
"""
from sim.network.closures import (
    apply_baseline_closures,
    load_closures,
    strip_closures,
    swap_db_closures,
)
from sim.network.connectivity import (
    check_connectivity,
    repair_boundary_scc,
    repair_divided_highway_dead_ends,
)
from sim.network.crs import (
    as_gdf,
    compute_bbox_from_links_raw,
    compute_urban_trim_bbox,
    guess_crs_from_coords,
    native_bbox_from_wgs84_bbox,
    network_bbox_wgs84_from_project,
    network_links_gdf_with_crs,
)
from sim.network.db import project_db, refresh_network
from sim.network.export import export_stable_network
from sim.network.filtering import (
    DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES,
    prune_orphan_nodes,
    remove_disconnected_components_keep_largest,
    remove_non_drivable_links,
    trim_network_to_bbox_raw,
)
from sim.network.map_export import (
    NETWORK_MAP_EXPORT_DPI,
    NETWORK_MAP_EXPORT_FIGSIZE,
    NETWORK_MAP_PALETTE,
    save_network_links_map_png,
)
from sim.network.normalization import normalize_network_attributes
from sim.network.normalization_pipeline import normalize_and_export_network
from sim.network.osm_enrichment import (
    aggregate_osm_edge_attributes,
    buffer_polygon_km,
    choose_best,
    download_osm_drive_edges,
    enrich_links_from_osm,
    extract_osm_ids,
    fill_missing_refs_from_named_corridors,
    geocode_place,
    listify,
    normalize_ref,
)
from sim.network.pipeline import build_network_from_osm, create_or_open_project

__all__ = [
    # pipeline (build)
    "build_network_from_osm",
    "create_or_open_project",
    # pipeline (normalization)
    "normalize_and_export_network",
    # normalization
    "normalize_network_attributes",
    # connectivity
    "check_connectivity",
    "repair_boundary_scc",
    "repair_divided_highway_dead_ends",
    # export
    "export_stable_network",
    # closures
    "apply_baseline_closures",
    "load_closures",
    "strip_closures",
    "swap_db_closures",
    # map export constants
    "NETWORK_MAP_PALETTE",
    "NETWORK_MAP_EXPORT_DPI",
    "NETWORK_MAP_EXPORT_FIGSIZE",
    "save_network_links_map_png",
    # filtering
    "DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES",
    "prune_orphan_nodes",
    "remove_non_drivable_links",
    "remove_disconnected_components_keep_largest",
    "trim_network_to_bbox_raw",
    # crs
    "as_gdf",
    "compute_bbox_from_links_raw",
    "compute_urban_trim_bbox",
    "guess_crs_from_coords",
    "native_bbox_from_wgs84_bbox",
    "network_bbox_wgs84_from_project",
    "network_links_gdf_with_crs",
    # osm
    "aggregate_osm_edge_attributes",
    "buffer_polygon_km",
    "choose_best",
    "download_osm_drive_edges",
    "enrich_links_from_osm",
    "extract_osm_ids",
    "fill_missing_refs_from_named_corridors",
    "geocode_place",
    "listify",
    "normalize_ref",
    # db
    "project_db",
    "refresh_network",
]
