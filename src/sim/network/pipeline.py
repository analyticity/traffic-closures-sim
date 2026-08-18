"""High-level network-build pipeline: OSM download -> filter -> trim -> enrich -> export.

This module is the single entry point called by ``run.py``.  Each pipeline
stage is a dedicated function that delegates to the specialised sub-modules
(``crs``, ``filtering``, ``osm_enrichment``, ``map_export``).
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
from aequilibrae import Project
from shapely.geometry import box

from sim.io_project import get_metric_epsg, load_config
from sim.network.crs import (
    as_gdf,
    compute_bbox_from_links_raw,
    compute_urban_trim_bbox,
    native_bbox_from_wgs84_bbox,
    network_bbox_wgs84_from_project,
)
from sim.network.filtering import (
    DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES,
    remove_disconnected_components_keep_largest,
    remove_non_drivable_links,
    trim_network_to_bbox_raw,
)
from sim.network.map_export import (
    NETWORK_MAP_PALETTE,
    save_network_links_map_png,
)
from sim.network.osm_enrichment import (
    buffer_polygon_km,
    enrich_links_from_osm,
    geocode_place,
)

logger = logging.getLogger(__name__)

_CONNECTIVITY_MARGIN_KM = 5.0

_GEOJSON_BASE_COLS = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
_GEOJSON_SPEED_COLS = ["speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "travel_time_ab", "travel_time_ba"]
_GEOJSON_META_COLS = ["link_type", "name", "osm_id", "osm_ref", "osm_ref_norm", "osm_name_raw", "osm_highway"]
_GEOJSON_ACTIVE_COLS = ["cycleway", "cycleway_left", "cycleway_right", "busway", "busway_left", "busway_right"]
_GEOJSON_CAP_COLS = ["capacity_ab", "capacity_ba"]
_GEOJSON_DESIRED_COLS = _GEOJSON_BASE_COLS + _GEOJSON_SPEED_COLS + _GEOJSON_META_COLS + _GEOJSON_ACTIVE_COLS + _GEOJSON_CAP_COLS


# ---------------------------------------------------------------------------
# Data class for resolved download geometry
# ---------------------------------------------------------------------------

@dataclass
class _DownloadArea:
    """Resolved geometry for the OSM download and the model trim."""
    bbox_cfg: Optional[Any] = None
    place_name: Optional[str] = None
    buffer_km: float = 0.0
    buffered_polygon: Optional[Any] = None
    model_bbox_polygon: Optional[Any] = None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _resolve_download_area(cfg: Dict[str, Any], crs_epsg_hint: int) -> _DownloadArea:
    """Parse config to determine download polygon / bbox and model trim polygon."""
    osm_cfg = cfg.get("osm", {}) or {}
    place_name: Optional[str] = osm_cfg.get("place_name")
    bbox_cfg = cfg.get("model_bbox") or osm_cfg.get("bbox")
    buffer_km = float(osm_cfg.get("buffer_km", 0))

    if not bbox_cfg and not place_name:
        raise ValueError("Missing config model_bbox or osm.place_name in sim.yaml")

    area = _DownloadArea(bbox_cfg=bbox_cfg, place_name=place_name, buffer_km=buffer_km)

    if place_name and buffer_km > 0 and not bbox_cfg:
        download_km = buffer_km + _CONNECTIVITY_MARGIN_KM
        logger.info(
            "Geocoding %r, buffer %s km (+ %s km connectivity margin -> %s km download)",
            place_name, buffer_km, _CONNECTIVITY_MARGIN_KM, download_km,
        )
        place_poly = geocode_place(place_name)
        area.model_bbox_polygon = box(*buffer_polygon_km(place_poly, buffer_km, crs_epsg_hint).bounds)
        area.buffered_polygon = box(*buffer_polygon_km(place_poly, download_km, crs_epsg_hint).bounds)
        logger.info("model bbox (WGS84): %s", area.model_bbox_polygon.bounds)
        logger.info("download bbox (WGS84): %s", area.buffered_polygon.bounds)

    return area


def create_or_open_project(project_dir: Path) -> Project:
    """Open an existing AequilibraE project or create a new one."""
    project = Project()
    has_db = (
        any(project_dir.glob("*.sqlite"))
        or any(project_dir.glob("*.db"))
        or any(project_dir.glob("*.sqlite3"))
    )
    if has_db:
        project.open(str(project_dir))
    else:
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project.new(str(project_dir))
    return project


def _ensure_fresh_project(
    project: Project,
    project_dir: Path,
    area: _DownloadArea,
    crs_epsg_hint: int,
) -> Tuple[Project, int, int]:
    """Rebuild the project from scratch if the existing network is too small for the buffer."""
    links_before = project.network.count_links()
    nodes_before = project.network.count_nodes()

    if links_before > 0 and area.buffered_polygon is not None:
        try:
            net_bbox = network_bbox_wgs84_from_project(project, crs_epsg_hint, pad_ratio=0, min_pad_deg=0)
            net_box = box(*net_bbox)
            buf_box = box(*area.buffered_polygon.bounds)
            coverage = net_box.area / buf_box.area if buf_box.area > 0 else 1.0
            if coverage < 0.95:
                logger.info(
                    "Existing network covers only %s of buffered area; rebuilding from scratch",
                    format(coverage, ".0%"),
                )
                project.close()
                shutil.rmtree(project_dir)
                project = create_or_open_project(project_dir)
                return project, 0, 0
            logger.info("Existing network already covers the buffered area; skipping rebuild")
        except Exception as e:
            logger.warning("could not check network extent vs buffer (%s), keeping existing network", e)

    return project, links_before, nodes_before


def _import_osm_network(
    project: Project,
    area: _DownloadArea,
    links_before: int,
    nodes_before: int,
) -> bool:
    """Import OSM into the project if it is empty.  Returns True when import ran."""
    if links_before > 0 and nodes_before > 0:
        return False

    if area.bbox_cfg:
        west, south, east, north = [float(x) for x in area.bbox_cfg]
        project.network.create_from_osm(model_area=box(west, south, east, north))
        logger.info("Network created from model_bbox (WGS84): %s", area.bbox_cfg)
    elif area.buffered_polygon is not None:
        project.network.create_from_osm(model_area=area.buffered_polygon)
        logger.info("Network from place_name %r + %s km buffer", area.place_name, area.buffer_km)
    else:
        project.network.create_from_osm(place_name=area.place_name)
        logger.info("Network created from place_name: %s", area.place_name)
    return True


def _apply_network_filters(
    project: Project,
    project_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply drivable-network and isolated-component filters.  Returns their stats."""
    dn = (cfg.get("network") or {}).get("drivable_network") or {}
    drivable_stats: Dict[str, Any] = {}
    if dn.get("enabled", True):
        raw_excl = dn.get("excluded_link_types")
        excluded = (
            {str(x).strip().lower() for x in raw_excl if str(x).strip()}
            if raw_excl
            else set(DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES)
        )
        require_car = bool(dn.get("require_mode_car", True))
        logger.info(
            "Drivable-network filter: %s excluded link_type values, require_mode_car=%s",
            len(excluded), require_car,
        )
        drivable_stats = remove_non_drivable_links(
            project, project_dir, excluded_link_types=excluded, require_mode_car=require_car,
        )
        logger.info("  removed links: %s", drivable_stats.get("links_removed", 0))
        logger.info("  pruned orphan nodes: %s", drivable_stats.get("nodes_pruned", 0))

    iso_cfg = (cfg.get("network") or {}).get("isolated_components") or {}
    isolated_stats: Dict[str, Any] = {}
    if iso_cfg.get("enabled", True):
        logger.info("Largest-component filter: removing disconnected subgraphs")
        isolated_stats = remove_disconnected_components_keep_largest(project, project_dir)
        logger.info(
            "  components (before): %s | removed links: %s | pruned orphan nodes: %s",
            isolated_stats.get("components", 0),
            isolated_stats.get("links_removed", 0),
            isolated_stats.get("nodes_pruned", 0),
        )

    return drivable_stats, isolated_stats


def _trim_to_urban_core(
    project: Project,
    project_dir: Path,
    area: _DownloadArea,
    crs_epsg_hint: int,
    iso_enabled: bool,
) -> Dict[str, Any]:
    """Compute urban-core bbox, trim non-corridor links, re-run isolation filter."""
    trim_bbox = compute_urban_trim_bbox(project, pad_ratio=0.02)
    if area.model_bbox_polygon is not None:
        model_trim = native_bbox_from_wgs84_bbox(
            project, list(area.model_bbox_polygon.bounds), crs_epsg_hint,
        )
        trim_bbox = (
            min(trim_bbox[0], model_trim[0]),
            min(trim_bbox[1], model_trim[1]),
            max(trim_bbox[2], model_trim[2]),
            max(trim_bbox[3], model_trim[3]),
        )

    logger.info("Trimming network to urban-core bbox (native): %s", trim_bbox)
    trim_stats = trim_network_to_bbox_raw(project, trim_bbox, project_dir)
    logger.info("  trimmed: %s links, %s nodes", trim_stats["links_deleted"], trim_stats["nodes_deleted"])

    if trim_stats["links_deleted"] > 0 and iso_enabled:
        logger.info("  re-running isolated-component filter after trim")
        iso2 = remove_disconnected_components_keep_largest(project, project_dir)
        trim_stats["post_trim_iso_links_removed"] = iso2.get("links_removed", 0)
        trim_stats["post_trim_iso_nodes_pruned"] = iso2.get("nodes_pruned", 0)
        logger.info(
            "  post-trim components: %s | removed links: %s",
            iso2.get("components", 0), iso2.get("links_removed", 0),
        )

    bbox_native = compute_bbox_from_links_raw(project)
    logger.info("  final network extent (native): %s", bbox_native)
    return trim_stats


def _export_artifacts(
    project: Project,
    out_dir: Path,
    crs_epsg_hint: int,
    area: _DownloadArea,
    *,
    bbox_native: Tuple[float, float, float, float],
    bbox_wgs84_used: Optional[Tuple[float, ...]],
    osm_imported: bool,
    trim_stats: Dict[str, Any],
    drivable_stats: Dict[str, Any],
    isolated_stats: Dict[str, Any],
    enrich_stats: Dict[str, Any],
) -> None:
    """Write PNG maps, GeoJSON layers, and ``network_counts.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-processing snapshot (only on fresh import) ---
    if osm_imported:
        try:
            links_df_pre = project.network.links.data
            if len(links_df_pre) > 0 and "geometry" in links_df_pre.columns:
                links_gdf_pre = gpd.GeoDataFrame(
                    links_df_pre, geometry="geometry", crs=getattr(links_df_pre, "crs", None),
                )
                bbox_pre = compute_bbox_from_links_raw(project)
                bbox_gdf_pre = gpd.GeoDataFrame({"geometry": [box(*bbox_pre)]}, crs=links_gdf_pre.crs)
                save_network_links_map_png(
                    links_gdf_pre, bbox_gdf_pre,
                    out_dir / "links_native_before_processing.png",
                    title="Network after OSM import (before filters & trim)",
                    links_color=NETWORK_MAP_PALETTE["links_before"],
                )
                logger.info("Wrote (pre-processing map): %s", out_dir / "links_native_before_processing.png")
        except Exception as e:
            logger.warning("could not write pre-processing map (%s)", e)
    else:
        logger.info("Skipping links_native_before_processing.png (loaded existing project)")

    # --- JSON counts ---
    links_n = project.network.count_links()
    nodes_n = project.network.count_nodes()

    counts_path = out_dir / "network_counts.json"
    counts_path.write_text(
        json.dumps(
            {
                "links": int(links_n),
                "nodes": int(nodes_n),
                "place_name": area.place_name or "",
                "buffer_km": area.buffer_km,
                "osm_bbox_used_for_import_wgs84": area.bbox_cfg or None,
                "trim_bbox_wgs84_requested": list(bbox_wgs84_used) if bbox_wgs84_used else None,
                "trim_bbox_native_used": {
                    "minx": bbox_native[0], "miny": bbox_native[1],
                    "maxx": bbox_native[2], "maxy": bbox_native[3],
                },
                "trim_stats": trim_stats,
                "drivable_network_stats": drivable_stats,
                "isolated_components_stats": isolated_stats,
                "enrich_stats": enrich_stats,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # --- Native-CRS map + GeoJSON ---
    links_df = project.network.links.data
    links_gdf_native = gpd.GeoDataFrame(links_df, geometry="geometry", crs=getattr(links_df, "crs", None))
    bbox_gdf_native = gpd.GeoDataFrame({"geometry": [box(*bbox_native)]}, crs=links_gdf_native.crs)

    save_network_links_map_png(
        links_gdf_native, bbox_gdf_native,
        out_dir / "links_native.png",
        title="AequilibraE links + reference bbox (native)",
        links_color=NETWORK_MAP_PALETTE["links_after"],
    )

    available_cols = [c for c in _GEOJSON_DESIRED_COLS if c in links_gdf_native.columns]
    links_gdf_native[available_cols].to_file(out_dir / "links_native.geojson", driver="GeoJSON")
    bbox_gdf_native.to_file(out_dir / "model_bbox_native.geojson", driver="GeoJSON")

    # --- WGS-84 map + GeoJSON (best-effort) ---
    links_gdf_plot = links_gdf_native.copy()
    if links_gdf_plot.crs is None:
        links_gdf_plot = as_gdf(links_gdf_plot, crs_epsg_hint)

    wgs_files: list[Path] = []
    try:
        links_wgs84 = links_gdf_plot.to_crs(epsg=4326)
        bbox_wgs84_gdf = bbox_gdf_native.copy()
        if bbox_wgs84_gdf.crs is None:
            bbox_wgs84_gdf = bbox_wgs84_gdf.set_crs(links_gdf_plot.crs, allow_override=True)
        bbox_wgs84_gdf = bbox_wgs84_gdf.to_crs(epsg=4326)

        save_network_links_map_png(
            links_wgs84, bbox_wgs84_gdf,
            out_dir / "links_wgs84.png",
            title="AequilibraE links + bbox (WGS84, best-effort)",
            links_color=NETWORK_MAP_PALETTE["links_after"],
        )
        links_wgs84[available_cols].to_file(out_dir / "links_wgs84.geojson", driver="GeoJSON")
        bbox_wgs84_gdf.to_file(out_dir / "model_bbox_wgs84.geojson", driver="GeoJSON")
        wgs_files = [
            out_dir / "links_wgs84.png",
            out_dir / "links_wgs84.geojson",
            out_dir / "model_bbox_wgs84.geojson",
        ]
    except Exception:
        pass

    logger.info(
        "Network build done: project=%s nodes=%s links=%s bbox_native=%s",
        out_dir.resolve(), nodes_n, links_n, bbox_native,
    )
    if bbox_wgs84_used is not None:
        logger.info("Config reference bbox (WGS84): %s", bbox_wgs84_used)
    logger.info("Trim stats: %s", trim_stats)
    logger.info("Enrich stats: %s", enrich_stats)
    logger.info(
        "Wrote: %s, %s, %s, %s%s",
        counts_path,
        out_dir / "links_native.png",
        out_dir / "links_native.geojson",
        out_dir / "model_bbox_native.geojson",
        (", " + ", ".join(str(p) for p in wgs_files)) if wgs_files else "",
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_network_from_osm(
    config_path: str | Path = "config/brno/sim.yaml",
    outputs_dir: str | Path | None = None,
) -> None:
    """End-to-end network build: download -> clean -> enrich -> export."""
    cfg = load_config(config_path)
    if outputs_dir is None:
        outputs_dir = cfg.get("network", {}).get("maps_dir", "outputs/baseline/maps")

    project_dir = Path(cfg["project_path"])
    crs_epsg_hint = get_metric_epsg(cfg)

    # 1. Resolve download geometry from config
    area = _resolve_download_area(cfg, crs_epsg_hint)

    # 2. Open (or create) AequilibraE project; rebuild if stale
    project = create_or_open_project(project_dir)
    project, links_before, nodes_before = _ensure_fresh_project(
        project, project_dir, area, crs_epsg_hint,
    )

    # 3. Import OSM network (only when the project is empty)
    osm_imported = _import_osm_network(project, area, links_before, nodes_before)

    # 4. Compute reference bbox for maps / metadata
    if area.bbox_cfg:
        bbox_native = native_bbox_from_wgs84_bbox(project, area.bbox_cfg, crs_epsg_hint)
        bbox_wgs84_used: Optional[Tuple[float, ...]] = tuple(float(x) for x in area.bbox_cfg)
        logger.info("Reference bbox from config (WGS84): %s", bbox_wgs84_used)
        logger.info("Reference bbox (native): %s", bbox_native)
    else:
        bbox_native = compute_bbox_from_links_raw(project)
        bbox_wgs84_used = None
        logger.info("Network extent bbox (native): %s", bbox_native)

    out_dir = Path(outputs_dir)

    # 5. Apply network filters (drivable + isolated-component)
    drivable_stats, isolated_stats = _apply_network_filters(project, project_dir, cfg)

    # 6. Trim to urban core
    iso_cfg = (cfg.get("network") or {}).get("isolated_components") or {}
    trim_stats = _trim_to_urban_core(
        project, project_dir, area, crs_epsg_hint, iso_cfg.get("enabled", True),
    )
    bbox_native = compute_bbox_from_links_raw(project)

    # 7. OSM enrichment (prefers the local PBF over a second Overpass query)
    from sim.io_project import as_path, get_nested

    pbf_path = as_path(
        get_nested(cfg, ["supernetwork", "pbf_path"],
                   "data/sources/osm/czech-republic-latest.osm.pbf")
    )
    enrich_stats = enrich_links_from_osm(
        project, project_dir,
        crs_epsg_hint=crs_epsg_hint,
        place_name=area.place_name,
        bbox_cfg=area.bbox_cfg,
        buffered_polygon=area.buffered_polygon,
        pbf_path=pbf_path,
    )
    logger.info("OSM enrichment: %s", enrich_stats)

    # 8. Export visual / data artifacts
    _export_artifacts(
        project, out_dir, crs_epsg_hint, area,
        bbox_native=bbox_native,
        bbox_wgs84_used=bbox_wgs84_used,
        osm_imported=osm_imported,
        trim_stats=trim_stats,
        drivable_stats=drivable_stats,
        isolated_stats=isolated_stats,
        enrich_stats=enrich_stats,
    )

    project.close()
