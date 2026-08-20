"""Zoning pipeline orchestrator -- the single public entry point."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
from aequilibrae import Project

from sim.io_project import get_metric_epsg, load_config
from sim.zoning.sources import load_zones_from_sources
from sim.zoning.sources.shared import remove_overlaps_by_priority

from sim.zoning.aoi import build_model_area, filter_zones_centroid_in_bbox
from sim.zoning.connectors import (
    calculate_centroids,
    create_centroid_connectors,
    delete_all_connectors_and_reset_centroids,
    _row_is_external,
)
from sim.zoning.diagnostics import (
    export_connector_diagnostics,
    export_gateway_diagnostics,
    export_gateway_seed_lookup,
    validate_connectors,
)
from sim.zoning.gateways import (
    auto_discover_boundary_roads,
    build_external_gateway_zones,
    resolve_blacklist,
    resolve_whitelist,
    road_blacklist_tokens,
    spec_matches_road_blacklist,
    select_gateway_target_nodes,
)
from sim.zoning.map_export import export_map_png

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def _employment_needs_remap(emp_path: Path, zones_geojson: Optional[Path] = None) -> bool:
    """Return True when zone_employment.parquet is missing, stale, or out of sync with zones."""
    try:
        df = pd.read_parquet(emp_path)
        if df.empty or "zone_id" not in df.columns:
            return True
        if bool((df["zone_id"] == 0).all()):
            return True
        if "match_engine_version" not in df.columns or int(df["match_engine_version"].min()) < 3:
            return True
        if zones_geojson is not None and zones_geojson.exists():
            z = gpd.read_file(zones_geojson)
            zids = {int(x) for x in z["zone_id"].astype(int).tolist()}
            eids = {int(x) for x in df["zone_id"].astype(int).tolist()}
            if zids != eids:
                return True
    except Exception:
        return True
    return False


def _population_needs_remap(pop_path: Path) -> bool:
    """Return True when zone_population.parquet exists but has only the no_zones fallback."""
    try:
        df = pd.read_parquet(pop_path)
        if df.empty or "zone_id" not in df.columns:
            return True
        return bool((df["zone_id"] == 0).all())
    except Exception:
        return True


def _ensure_zone_population(
    cfg: dict, pop_path: Path, zones_geojson: Path,
) -> Path:
    """Regenerate zone_population.parquet if the raw CSV is available and
    the current parquet is missing or contains the no_zones fallback."""
    if pop_path.exists() and not _population_needs_remap(pop_path):
        return pop_path

    pop_source = (
        cfg.get("datasets", {}).get("sources", {})
        .get("population_sldb2021", {}).get("out_path", "")
    )
    pop_csv = Path(pop_source) if pop_source else Path()
    if not pop_csv.exists():
        return pop_path

    from sim.datasets.population import preprocess_population_sldb2021

    logger.info("Regenerating zone_population.parquet with real zone IDs")
    preprocess_population_sldb2021(
        pop_csv,
        pop_path,
        zones_geojson=zones_geojson if zones_geojson.exists() else None,
        cfg=cfg,
    )
    return pop_path


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_zones_and_connectors(
    config_path: str | Path | Dict[str, Any] = "config/brno/sim.yaml",
) -> None:
    """End-to-end zoning pipeline: AOI, zone loading, gateways, connectors, exports."""
    logger.info("Zoning: loading config")
    cfg = load_config(config_path) if not isinstance(config_path, dict) else config_path

    project_dir = Path(cfg["project_path"])
    crs_epsg = get_metric_epsg(cfg)
    zoning_cfg = cfg.get("zoning", {}) or {}

    sources = zoning_cfg.get("sources")
    if not sources:
        raise ValueError("config.zoning.sources must be set")

    area_filter_cfg = zoning_cfg.get("zone_area_filter", {}) or {}
    cache_file = zoning_cfg.get("cache_file")
    output_dir = Path(zoning_cfg.get("output_dir", "outputs/baseline/zones"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_connectors = int(zoning_cfg.get("max_connectors", 6))
    max_distance_m = float(zoning_cfg.get("max_distance_m", 3000.0))
    connector_speed = float(zoning_cfg.get("connector_speed_kmh", 20.0))
    connector_capacity = float(zoning_cfg.get("connector_capacity_vph", 2000.0))
    connector_lanes = int(zoning_cfg.get("connector_lanes", 1))
    connector_penalty = float(zoning_cfg.get("connector_access_penalty_s", 300.0))
    internal_exclude_road_types = list(
        zoning_cfg.get("internal_exclude_road_types", ["motorway", "motorway_link"])
    )

    ext_cfg = zoning_cfg.get("external_gateways", {}) or {}
    export_lookup = bool(ext_cfg.get("export_lookup", True))
    _cache = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))
    export_lookup_path = Path(ext_cfg.get("export_lookup_path", f"{_cache}/gateway_lookup_seed.parquet"))

    logger.info("Zoning: opening project")
    project = Project()
    project.open(str(project_dir))

    try:
        # --- AOI ---
        aoi_cfg = zoning_cfg.get("aoi", {}) or {}
        model_area = build_model_area(
            project,
            crs_epsg,
            quantile=float(aoi_cfg.get("quantile", 0.01)),
            pad_ratio=float(aoi_cfg.get("pad_ratio", 0.008)),
            extra_margin_m=float(aoi_cfg.get("extra_margin_m", 200.0)),
            shift_x_m=float(aoi_cfg.get("shift_x_m", 0.0)),
            shift_y_m=float(aoi_cfg.get("shift_y_m", 0.0)),
        )

        # --- Zone loading & filtering ---
        logger.info("Zoning: loading zones (cache or sources)")
        zones = load_zones_from_sources(
            sources=sources,
            crs_epsg=crs_epsg,
            cache_file=Path(cache_file) if cache_file else None,
            area_filter_cfg=area_filter_cfg,
        )

        zones = filter_zones_centroid_in_bbox(zones, model_area, crs_epsg)
        logger.info("zones with representative_point in AOI: %d", len(zones))

        zones = remove_overlaps_by_priority(zones, rank_col="source_rank", min_area_m2=25.0)
        logger.info("zones after priority de-overlap: %d", len(zones))

        # --- External gateways ---
        gateway_targets: Dict[str, List[int]] = {}
        gateway_meta: Dict[str, Dict[str, Any]] = {}
        debug_corridors = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")
        debug_points = gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{crs_epsg}")

        if bool(ext_cfg.get("enabled", False)):
            whitelist_specs = resolve_whitelist(ext_cfg)
            blacklist_specs = resolve_blacklist(ext_cfg)
            blacklist_tokens = road_blacklist_tokens(blacklist_specs)

            auto_cfg = ext_cfg.get("auto_discover") or {}
            whitelist_empty = len(whitelist_specs) == 0
            auto_enabled = bool(auto_cfg.get("enabled", False)) or whitelist_empty
            all_auto_types: Optional[list] = None
            if auto_enabled:
                existing_refs = {s["raw"] for s in whitelist_specs}
                if whitelist_empty:
                    all_auto_types = [
                        "motorway", "motorway_link", "trunk", "trunk_link",
                        "primary", "primary_link", "secondary", "secondary_link",
                    ]
                    default_max = 50
                else:
                    auto_types = auto_cfg.get("link_types", [
                        "secondary", "secondary_link",
                    ])
                    all_auto_types = list(ext_cfg.get(
                        "allowed_link_types",
                        ["motorway", "motorway_link", "trunk", "trunk_link",
                         "primary", "primary_link"],
                    )) + list(auto_types)
                    default_max = 10
                discovered = auto_discover_boundary_roads(
                    project=project,
                    target_epsg=crs_epsg,
                    model_area=model_area,
                    existing_refs=existing_refs,
                    boundary_buffer_m=float(ext_cfg.get("boundary_buffer_m", 1000.0)) * 1.5,
                    min_link_types=all_auto_types,
                    min_lanes=int(auto_cfg.get("min_lanes", 1)),
                    blacklist_tokens=blacklist_tokens or None,
                )
                max_auto = int(auto_cfg.get("max_gateways", default_max))
                discovered = discovered[:max_auto]
                if discovered:
                    for d in discovered:
                        d["auto_discovered"] = True
                    refs_str = ", ".join(d["raw"] for d in discovered)
                    logger.info(
                        "Auto-discovered %d boundary roads: %s",
                        len(discovered),
                        refs_str,
                    )
                    whitelist_specs.extend(discovered)

            if blacklist_tokens:
                before = len(whitelist_specs)
                whitelist_specs = [
                    s for s in whitelist_specs
                    if not spec_matches_road_blacklist(s, blacklist_tokens)
                ]
                n_skipped = before - len(whitelist_specs)
                if n_skipped:
                    logger.info(
                        "External gateways blacklist skipped %d road spec(s) (%d remaining)",
                        n_skipped,
                        len(whitelist_specs),
                    )

            _default_allowed = (
                all_auto_types
                if all_auto_types is not None
                else ["motorway", "motorway_link", "trunk", "trunk_link",
                      "primary", "primary_link"]
            )
            gateway_targets, gateway_meta, debug_corridors, debug_points = select_gateway_target_nodes(
                project=project,
                target_epsg=crs_epsg,
                model_area=model_area,
                whitelist_specs=whitelist_specs,
                nodes_per_gateway=int(ext_cfg.get("connectors_per_gateway", 2)),
                boundary_buffer_m=float(ext_cfg.get("boundary_buffer_m", 1000.0)),
                min_gateway_separation_m=float(ext_cfg.get("min_gateway_separation_m", 800.0)),
                allowed_link_types=ext_cfg.get(
                    "allowed_link_types",
                    _default_allowed,
                ),
                max_anchor_distance_m=float(ext_cfg.get("max_anchor_distance_m", 2000.0)),
            )

            if debug_corridors is not None and not debug_corridors.empty:
                dbg = debug_corridors.copy()

                extra_geom_cols = [
                    c for c in dbg.columns if c != dbg.geometry.name and dbg[c].dtype == "geometry"
                ]
                if extra_geom_cols:
                    dbg = dbg.drop(columns=extra_geom_cols)

                dbg.to_file(output_dir / "gateway_corridors_debug.geojson", driver="GeoJSON")
                logger.debug("gateway corridor debug: %s", output_dir / "gateway_corridors_debug.geojson")

            if debug_points is not None and not debug_points.empty:
                debug_points.to_file(output_dir / "gateway_points_debug.geojson", driver="GeoJSON")
                logger.debug("gateway points debug: %s", output_dir / "gateway_points_debug.geojson")

            ext_zones = build_external_gateway_zones(
                gateway_meta=gateway_meta,
                target_epsg=crs_epsg,
                zone_offset_m=float(ext_cfg.get("zone_offset_m", 10.0)),
                zone_size_m=float(ext_cfg.get("zone_size_m", 40.0)),
                start_id=int(ext_cfg.get("synthetic_zone_id_start", 8_000_000_000)),
            )

            if not ext_zones.empty:
                zones = gpd.GeoDataFrame(
                    pd.concat([zones, ext_zones], ignore_index=True),
                    crs=zones.crs,
                )
                logger.info("synthetic external gateway zones added: %d", len(ext_zones))

            export_gateway_diagnostics(gateway_meta, output_dir)

            if export_lookup:
                export_gateway_seed_lookup(
                    gateway_meta,
                    export_lookup_path,
                    crs_epsg=crs_epsg,
                )

        # --- CRS enforcement ---
        if zones.crs is None:
            zones = zones.set_crs(epsg=crs_epsg, allow_override=True)
        if zones.crs.to_epsg() != crs_epsg:
            zones = zones.to_crs(epsg=crs_epsg)

        centroids = calculate_centroids(zones)

        # --- Pre-connector map ---
        map_dpi = int(zoning_cfg["map_dpi"]) if zoning_cfg.get("map_dpi") is not None else None
        export_map_png(
            project=project,
            zones=zones,
            centroids=centroids,
            model_area_bbox=model_area,
            output_dir=output_dir,
            filename="zones_map_pre_connectors.png",
            title_suffix=" (pre-connectors)",
            dpi=map_dpi,
            debug_corridors=debug_corridors,
            debug_points=debug_points,
            crs_epsg=crs_epsg,
        )

        # --- Connectors ---
        delete_all_connectors_and_reset_centroids(project)
        project.close()

        project = Project()
        project.open(str(project_dir))

        zone_to_centroid = create_centroid_connectors(
            project=project,
            centroids=centroids,
            max_connectors=max_connectors,
            max_distance_m=max_distance_m,
            speed_kmh=connector_speed,
            capacity_vph=connector_capacity,
            lanes=connector_lanes,
            access_penalty_s=connector_penalty,
            gateway_targets=gateway_targets,
            external_speed_kmh=float(ext_cfg.get("external_connector_speed_kmh", 90.0)),
            external_access_penalty_s=float(ext_cfg.get("external_connector_access_penalty_s", 5.0)),
            internal_exclude_road_types=internal_exclude_road_types,
        )

        export_connector_diagnostics(project, zone_to_centroid, output_dir)

        connector_warn_dist = float(zoning_cfg.get("connector_warn_distance_m", 1500.0))
        connector_strict = bool(zoning_cfg.get("connector_strict", False))
        validation_result = validate_connectors(
            project, zone_to_centroid, output_dir,
            warn_min_distance_m=connector_warn_dist,
        )

        warn_list = validation_result.get("warnings", [])
        n_far = sum(1 for w in warn_list if w.get("type") == "far_connectors")
        n_scc = sum(1 for w in warn_list if w.get("type") == "outside_scc")
        if warn_list:
            logger.info(
                "Connector validation: %d zone(s) with warnings "
                "(%d far, %d outside SCC)",
                len({w["zone_id"] for w in warn_list}), n_far, n_scc,
            )
        if connector_strict and (n_far > 0 or n_scc > 0):
            raise RuntimeError(
                f"Connector validation failed in strict mode: "
                f"{n_far} zone(s) with far connectors, "
                f"{n_scc} zone(s) outside SCC. "
                f"Fix connector geometry or set zoning.connector_strict=false."
            )

        # --- Final exports ---
        centroids["centroid_node_id"] = centroids["zone_id"].map(
            lambda z: zone_to_centroid.get(int(z), int(z))
        ).astype(int)

        output_dir.mkdir(parents=True, exist_ok=True)
        zones.to_file(output_dir / "zones.geojson", driver="GeoJSON")
        centroids.to_file(output_dir / "centroids.geojson", driver="GeoJSON")

        # --- Population ---
        pop_path = Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")) / "zone_population.parquet"
        pop_path = _ensure_zone_population(cfg, pop_path, output_dir / "zones.geojson")

        if pop_path.exists():
            pop_df = pd.read_parquet(pop_path)
            pop_map = dict(zip(pop_df["zone_id"], pop_df["population"]))

            centroids["population"] = centroids.apply(
                lambda r: 0 if _row_is_external(r) else int(pop_map.get(int(r["zone_id"]), 0)),
                axis=1,
            ).astype(int)

            zones["population"] = zones.apply(
                lambda r: 0 if _row_is_external(r) else int(pop_map.get(int(r["zone_id"]), 0)),
                axis=1,
            ).astype(int)

            logger.info(
                "population data loaded: %d zones, total=%s",
                len(pop_map),
                format(sum(pop_map.values()), ","),
            )
            logger.info("external zones exported with population=0")

            zones.to_file(output_dir / "zones.geojson", driver="GeoJSON")
            centroids.to_file(output_dir / "centroids.geojson", driver="GeoJSON")
        else:
            logger.warning(
                "%s not found; run fetch-data to download population CSV",
                pop_path,
            )

        # --- Employment (derived from commuting destinations) ---
        emp_path = Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")) / "zone_employment.parquet"
        _emp_needs_regen = (
            not emp_path.exists()
            or _employment_needs_remap(emp_path, output_dir / "zones.geojson")
        )
        if _emp_needs_regen:
            reason = "missing" if not emp_path.exists() else "stale zone_id=0"
            logger.info("zone_employment.parquet %s; regenerating ...", reason)
            try:
                from sim.datasets.employment import derive_zone_employment
                from sim.datasets.paths import resolved_commuting_full_cr_parquet_path
                comm_path = resolved_commuting_full_cr_parquet_path(cfg)
                if comm_path.exists():
                    emp_info = derive_zone_employment(
                        comm_path, emp_path,
                        zones_geojson=output_dir / "zones.geojson",
                        cfg=cfg,
                    )
                    logger.info(
                        "zone_employment.parquet regenerated: %d zones, %d matched, total=%s",
                        emp_info.get("zones_total", 0),
                        emp_info.get("matched", 0),
                        format(emp_info.get("total_employment", 0), ","),
                    )
                else:
                    logger.warning(
                        "Cannot regenerate employment: commuting source %s not found. "
                        "Run fetch-data first.",
                        comm_path,
                    )
            except Exception:
                logger.warning("Employment regeneration failed", exc_info=True)

        mapping_path = output_dir / "zone_centroid_mapping.json"
        mapping_path.write_text(
            json.dumps({str(k): v for k, v in zone_to_centroid.items()}, indent=2),
            encoding="utf-8",
        )
        logger.info("zone->centroid mapping: %s", mapping_path)

        gpd.GeoDataFrame({"geometry": [model_area]}, crs=f"EPSG:{crs_epsg}").to_file(
            output_dir / "model_area.geojson", driver="GeoJSON"
        )

        logger.info("exported: %s (zones=%d)", output_dir, len(zones))

        # --- Final map ---
        export_map_png(
            project=project,
            zones=zones,
            centroids=centroids,
            model_area_bbox=model_area,
            output_dir=output_dir,
            filename="zones_map.png",
            title_suffix=" (final)",
            dpi=map_dpi,
            crs_epsg=crs_epsg,
        )

    finally:
        project.close()
