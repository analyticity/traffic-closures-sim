"""Demand build pipeline: orchestrator and preflight check."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sim.io_project import get_metric_epsg, get_nested, load_config, load_zone_population, load_zones
from sim.demand.config import _build_cfg, _normalize_period_shares, _parse_gateway_pair_weights
from sim.demand.naming import _build_zone_name_index
from sim.demand.gateways import (
    _load_external_gateway_lookup_directional,
    _load_gateways,
    _load_through_gateway_pairs,
    _preflight_external_inputs,
)
from sim.demand.hub import _build_hub_group
from sim.demand.commuting_io import _filter_commuting, _read_commuting
from sim.demand.seeds import (
    _build_external_local_seed,
    _build_external_through_from_pairs,
    _build_external_through_seed,
    _build_gravity_seed,
    _estimate_total_daily_trips_from_csd,
    _zero_matrix,
)
from sim.demand.od_builder import _build_od_cores
from sim.demand.aem_io import _ensure_dir, _register_in_project, _write_aem
from sim.distribution.impedance import _load_zone_employment

logger = logging.getLogger(__name__)


def load_or_build_od_matrix(config_path: str | Path = "config/brno/sim.yaml", cfg: dict | None = None) -> None:
    t0 = time.perf_counter()
    marks: Dict[str, float] = {}
    t_prev = t0

    def _mark(name: str) -> None:
        nonlocal t_prev
        now = time.perf_counter()
        marks[name] = round(now - t_prev, 3)
        t_prev = now

    if cfg is None:
        cfg = load_config(config_path)
    bcfg = _build_cfg(cfg)

    logger.info("Loading zones")
    zones_gdf = load_zones(cfg)
    _mark("load_zones")

    external_zone_ids: set[int] = set(
        zones_gdf.loc[zones_gdf["is_external"].fillna(0).astype(int) == 1, "zone_id"].astype(int)
    )
    logger.info("External zones: %d", len(external_zone_ids))

    zone_ids = np.array(sorted(zones_gdf["zone_id"].astype(int).unique()), dtype=np.int64)

    zoning_dir = get_nested(cfg, ["zoning", "output_dir"], "outputs/baseline/zones")
    mapping_path = Path(zoning_dir) / "zone_centroid_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"zone_centroid_mapping.json not found at {mapping_path}. Run build-zones first."
        )

    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    zone_to_centroid: Dict[int, int] = {int(k): int(v) for k, v in raw_mapping.items()}
    logger.info("Loaded zone->centroid mapping (%d entries)", len(zone_to_centroid))

    centroid_ids = np.array([zone_to_centroid.get(int(z), int(z)) for z in zone_ids], dtype=np.int64)
    logger.info(
        "%d zones loaded (centroid IDs: %d-%d)",
        len(zone_ids),
        int(centroid_ids.min()),
        int(centroid_ids.max()),
    )

    primary, stripped = _build_zone_name_index(zones_gdf)
    logger.info("Name index: %d primary, %d stripped entries", len(primary), len(stripped))

    logger.info("Reading commuting data")
    df_raw = _read_commuting(bcfg)
    _mark("read_commuting")

    zone_population = load_zone_population(
        Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))
    )
    _mark("load_zone_population")

    origin_place_col = "op_obec" if "op_obec" in df_raw.columns else None
    dest_place_col = "doj_obec" if "doj_obec" in df_raw.columns else None
    csv_places: set[str] = set()
    if origin_place_col:
        csv_places |= set(df_raw[origin_place_col].dropna().astype(str).unique())
    if dest_place_col:
        csv_places |= set(df_raw[dest_place_col].dropna().astype(str).unique())

    hub_cfg = get_nested(cfg, ["demand", "hub_group"], {}) or {}
    hub_name = str(hub_cfg.get("name", "")).strip()
    if not hub_name:
        place = str(get_nested(cfg, ["osm", "place_name"], "") or "")
        hub_name = place.split(",")[0].strip() if place else ""

    hub_source_rank = hub_cfg.get("source_rank")
    if hub_source_rank is not None:
        hub_source_rank = int(hub_source_rank)

    groups = _build_hub_group(
        zones_gdf,
        primary,
        stripped,
        csv_places,
        hub_name=hub_name,
        hub_source_rank=hub_source_rank,
        zone_population=zone_population,
    )
    if groups:
        for group_name, members in groups.items():
            logger.debug("Group %r: %d zones", group_name, len(members))

    # The core city is one place in the census, so both ends of its trips have
    # to be split among its zones.  Home ends follow population; work and
    # school ends follow employment, which otherwise lands where people sleep.
    groups_work = groups
    hub_dest_weights = str(hub_cfg.get("destination_weights", "employment")).strip().lower()
    if not groups:
        hub_dest_weights = "none"
    elif hub_dest_weights == "employment":
        zone_employment = _load_zone_employment(
            Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))
        )
        if zone_employment:
            groups_work = _build_hub_group(
                zones_gdf,
                primary,
                stripped,
                csv_places,
                hub_name=hub_name,
                hub_source_rank=hub_source_rank,
                zone_population=zone_employment,
                weight_label="employment",
            ) or groups
        else:
            hub_dest_weights = "population"
            logger.warning(
                "demand.hub_group.destination_weights=employment but "
                "zone_employment.parquet is missing -- work ends of the core "
                "city follow population. Run fetch-data after build-zones."
            )
    else:
        hub_dest_weights = "population"
    _mark("build_hub_groups")

    gateways = _load_gateways(zones_gdf)
    if gateways:
        logger.info("Gateways: %d corridors", len(gateways))
        for gateway_name, gateway_zones in gateways.items():
            logger.debug("%s: %d zones", gateway_name, len(gateway_zones))
    else:
        logger.info("No external gateway zones found")

    _preflight_external_inputs(bcfg, gateways)
    _mark("preflight_external_inputs")

    external_lookup: Dict[str, List[Tuple[int, float]]] = {}
    external_lookup_out: Dict[str, List[Tuple[int, float]]] = {}
    if bcfg.external.enabled:
        (
            external_lookup,
            external_lookup_out,
            _gw_in_names,
            _gw_out_names,
        ) = _load_external_gateway_lookup_directional(
            bcfg.external.gateway_lookup_path,
            gateways,
        )

    df = _filter_commuting(df_raw, bcfg)
    logger.info("%d rows after filtering", len(df))
    _mark("filter_commuting")

    logger.info("Building commuting OD cores")
    cores, summary = _build_od_cores(
        df,
        zone_ids,
        bcfg,
        primary=primary,
        stripped=stripped,
        groups=groups,
        gateways=gateways,
        external_lookup=external_lookup,
        external_lookup_out=external_lookup_out,
        groups_work=groups_work,
    )
    _mark("build_commuting_cores")

    segments_cfg = get_nested(cfg, ["demand", "segments"], {}) or {}
    other_cfg = segments_cfg.get("other", {}) or {}
    external_local_cfg = segments_cfg.get("external_local", {}) or {}
    external_through_cfg = segments_cfg.get("external_through", {}) or {}

    n_zones = len(zone_ids)

    # --- Internal "other" trips ---
    if other_cfg and other_cfg.get("source", "gravity") == "gravity":
        logger.info("Building 'other' trips (gravity seed)")
        other_defaults = (other_cfg.get("defaults") or {})
        other_daily = _build_gravity_seed(
            zone_ids,
            zone_population,
            trip_rate=float(other_cfg.get("trip_rate", other_defaults.get("trip_rate", 1.0))),
            car_share=float(other_cfg.get("car_share", other_defaults.get("car_share", 0.38))),
            occupancy=float(other_cfg.get("occupancy", other_defaults.get("occupancy", 1.45))),
            beta=float(other_cfg.get("beta", other_defaults.get("beta", 0.00030))),
            centroids_gdf=zones_gdf,
            excluded_zone_ids=external_zone_ids,
            metric_epsg=get_metric_epsg(cfg),
        )
        logger.info("'other' daily total: %.0f", float(other_daily.sum()))
    else:
        other_daily = _zero_matrix(n_zones)
    _mark("build_other_seed")

    # --- Residual synthetic external_local (optional) ---
    _el_source = external_local_cfg.get("source", "gateway_local") if external_local_cfg else None
    zoning_out = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))
    gw_diag_path = zoning_out / "gateway_diagnostics.csv"

    _el_trips_raw = external_local_cfg.get("total_daily_trips", "auto") if external_local_cfg else 0
    _el_trips_mode = str(_el_trips_raw).strip().lower()

    if _el_trips_mode == "auto" or _el_trips_raw is None:
        _el_trips_val = _estimate_total_daily_trips_from_csd(cfg, gw_diag_path)
        if _el_trips_val is None:
            logger.warning(
                "total_daily_trips='auto' but CSD estimation failed; "
                "falling back to 0 (no external-local seed)"
            )
            _el_trips_val = 0.0
    else:
        _el_trips_val = float(_el_trips_raw)

    if (
        external_local_cfg
        and bool(external_local_cfg.get("enabled", True))
        and _el_source == "gateway_local"
        and gateways
        and _el_trips_val > 0
    ):
        logger.info("Building residual synthetic external-local trips (gateway <-> internal)")
        gw_link_types: Optional[Dict[str, str]] = None
        if gw_diag_path.exists():
            _gd = pd.read_csv(gw_diag_path)
            if "gateway_name" in _gd.columns:
                lt_col = "predominant_link_type" if "predominant_link_type" in _gd.columns else "link_type"
                if lt_col in _gd.columns:
                    gw_link_types = dict(zip(_gd["gateway_name"].astype(str), _gd[lt_col].astype(str)))
        external_local_daily = _build_external_local_seed(
            zone_ids,
            gateways,
            zone_population,
            total_daily_trips=_el_trips_val,
            corridor_weights=external_local_cfg.get("corridor_weights", {}) or {},
            gateway_link_types=gw_link_types,
        )
        logger.info("'external_local' daily total: %.0f", float(external_local_daily.sum()))
    else:
        external_local_daily = _zero_matrix(n_zones)
    _mark("build_external_local")

    # --- Data-driven external_through from coarse supernetwork ---
    data_driven_through_daily = _zero_matrix(n_zones)
    if bcfg.external.enabled and bcfg.external.use_through_traffic:
        through_pairs_df = _load_through_gateway_pairs(bcfg.external.through_pairs_path)
        if not through_pairs_df.empty:
            logger.info("Building data-driven external-through trips from supernetwork gateway pairs")
            data_driven_through_daily = _build_external_through_from_pairs(
                zone_ids,
                gateways,
                through_pairs_df,
            )
            through_scale = float(
                get_nested(cfg, ["demand", "sldb", "external_processing", "through_traffic_scale"], 0.65) or 0.65
            )
            if through_scale != 1.0:
                data_driven_through_daily *= through_scale
                logger.info("Applied through_traffic_scale=%.2f", through_scale)
            logger.info("'external_through_data' daily total: %.0f", float(data_driven_through_daily.sum()))
    _mark("build_external_through_data")

    # --- Optional residual synthetic external_through ---
    if (
        external_through_cfg
        and bool(external_through_cfg.get("enabled", True))
        and external_through_cfg.get("source") == "gateway_pairs"
        and gateways
        and float(external_through_cfg.get("total_daily_trips", 0.0)) > 0
    ):
        logger.info("Building residual synthetic external-through trips (gateway <-> gateway)")
        pair_weights = _parse_gateway_pair_weights(
            external_through_cfg.get("pairs", []),
            sorted(gateways.keys()),
            default_pair_weight=float(external_through_cfg.get("default_pair_weight", 0.0)),
        )
        residual_external_through_daily = _build_external_through_seed(
            zone_ids,
            gateways,
            total_daily_trips=float(external_through_cfg.get("total_daily_trips", 0.0)),
            pair_weights=pair_weights,
        )
        logger.info("'external_through_residual' daily total: %.0f", float(residual_external_through_daily.sum()))
    else:
        residual_external_through_daily = _zero_matrix(n_zones)
    _mark("build_external_through_residual")

    external_through_daily = data_driven_through_daily + residual_external_through_daily

    other_period_shares = _normalize_period_shares(
        get_nested(cfg, ["demand", "time_slices", "segments", "other"], None),
        bcfg.periods,
    )
    external_local_period_shares = _normalize_period_shares(
        get_nested(cfg, ["demand", "time_slices", "segments", "external_local"], None),
        bcfg.periods,
    )
    external_through_period_shares = _normalize_period_shares(
        get_nested(cfg, ["demand", "time_slices", "segments", "external_through"], None),
        bcfg.periods,
    )

    all_cores: Dict[str, np.ndarray] = {}

    for period in bcfg.periods:
        all_cores[f"wd_{period}_commuting"] = cores.get(f"wd_{period}", _zero_matrix(n_zones)).copy()
    all_cores["wd_daily_commuting"] = cores.get("wd_daily", _zero_matrix(n_zones)).copy()

    for period in bcfg.periods:
        all_cores[f"wd_{period}_other"] = other_daily * other_period_shares[period]
        all_cores[f"wd_{period}_external_local"] = external_local_daily * external_local_period_shares[period]
        all_cores[f"wd_{period}_external_through"] = external_through_daily * external_through_period_shares[period]
        all_cores[f"wd_{period}_external"] = (
            all_cores[f"wd_{period}_external_local"]
            + all_cores[f"wd_{period}_external_through"]
        )

    all_cores["wd_daily_other"] = other_daily.copy()
    all_cores["wd_daily_external_local"] = external_local_daily.copy()
    all_cores["wd_daily_external_through_data"] = data_driven_through_daily.copy()
    all_cores["wd_daily_external_through_residual"] = residual_external_through_daily.copy()
    all_cores["wd_daily_external_through"] = external_through_daily.copy()
    all_cores["wd_daily_external"] = external_local_daily + external_through_daily
    all_cores["wd_daily_local"] = (
        all_cores["wd_daily_commuting"] + other_daily + external_local_daily
    )

    for period in bcfg.periods:
        all_cores[f"wd_{period}"] = (
            all_cores[f"wd_{period}_commuting"]
            + all_cores[f"wd_{period}_other"]
            + all_cores[f"wd_{period}_external_local"]
            + all_cores[f"wd_{period}_external_through"]
        )

    all_cores["wd_daily"] = sum(all_cores[f"wd_{period}"] for period in bcfg.periods)
    _mark("assemble_cores")

    segment_totals = {
        "commuting": round(float(all_cores["wd_daily_commuting"].sum()), 0),
        "other": round(float(other_daily.sum()), 0),
        "external_local": round(float(external_local_daily.sum()), 0),
        "external_through_data": round(float(data_driven_through_daily.sum()), 0),
        "external_through_residual": round(float(residual_external_through_daily.sum()), 0),
        "external_through_total": round(float(external_through_daily.sum()), 0),
        "external_total": round(float(all_cores["wd_daily_external"].sum()), 0),
        "combined_daily": round(float(all_cores["wd_daily"].sum()), 0),
    }
    logger.info("Segment totals: %s", segment_totals)

    logger.info("Writing AEM matrix: %s", bcfg.matrix_path)
    _write_aem(bcfg.matrix_path, centroid_ids, all_cores, matrix_name=bcfg.matrix_name)

    # Seed reference for ODME's drift report.  Anchored here because this is a
    # moment the matrix is provably uncalibrated; when distribution runs it
    # moves the reference forward to its own output, which is the real
    # uncalibrated demand.  Calibration used to refresh .aem.orig whenever the
    # working matrix differed from it — which is exactly what its own previous run
    # caused — so the "seed" silently became the last calibrated matrix and
    # prior_drift_pct measured zero by construction.
    import shutil as _shutil

    _seed_backup = Path(bcfg.matrix_path).with_suffix(".aem.orig")
    _shutil.copy2(bcfg.matrix_path, _seed_backup)
    logger.info("Seed matrix reference: %s", _seed_backup)
    _mark("write_aem_matrix")

    _ensure_dir(bcfg.output_dir)

    summary_data = {
        "commuting_source": str(
            bcfg.commuting_full_cr_parquet
            if (bcfg.external.enabled and bcfg.external.use_full_cr_dataset and bcfg.commuting_full_cr_parquet.exists())
            else (bcfg.commuting_filtered_parquet if bcfg.commuting_filtered_parquet.exists() else bcfg.commuting_csv)
        ),
        "external_gateway_lookup": str(bcfg.external.gateway_lookup_path),
        "through_gateway_pairs": str(bcfg.external.through_pairs_path),
        "matrix_path": str(bcfg.matrix_path),
        "conversion": {
            "work": {
                "car_share": bcfg.conv_work.car_share,
                "occupancy": bcfg.conv_work.occupancy,
                "trips_per_person": bcfg.conv_work.trips_per_person,
            },
            "school": {
                "car_share": bcfg.conv_school.car_share,
                "occupancy": bcfg.conv_school.occupancy,
                "trips_per_person": bcfg.conv_school.trips_per_person,
            },
        },
        "include_lokalizace": bcfg.include_lokalizace,
        "hub_destination_weights": hub_dest_weights,
        "periods": bcfg.periods,
        "segments": segment_totals,
        "groups": {k: {"zones": len(v)} for k, v in groups.items()},
        "gateways": {k: len(v) for k, v in gateways.items()},
        "external_lookup_places_inbound": len(external_lookup),
        "external_lookup_places_outbound": len(external_lookup_out),
        **summary,
        "final_cores_sum": {name: round(float(all_cores[name].sum()), 1) for name in sorted(all_cores.keys())},
        "final_nonzero_cells": {name: int(np.count_nonzero(all_cores[name])) for name in sorted(all_cores.keys())},
    }

    summary_path = bcfg.output_dir / "od_summary.json"
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    project_dir = cfg.get("project_path")
    if project_dir and Path(project_dir).exists():
        logger.info("Registering matrix in AequilibraE project: %s", project_dir)
        _register_in_project(Path(project_dir), bcfg.matrix_path)
    _mark("register_matrix")

    summary_data["timing_breakdown_s"] = marks
    summary_data["elapsed_total_s"] = round(time.perf_counter() - t0, 3)
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "OD build: zones=%s rows_in=%s pairs_used=%s direct=%s group=%s ext_lookup=%s "
        "legacy_fallback=%s skipped_ext_ext=%s missing_o=%s missing_d=%s",
        summary["zones"],
        summary["rows_in"],
        summary["pairs_used"],
        summary["mapped_direct"],
        summary["mapped_group"],
        summary["mapped_external_lookup"],
        summary["mapped_external_legacy_fallback"],
        summary["skipped_external_external_rows"],
        summary["missing_origin"],
        summary["missing_destination"],
    )
    for core_name in sorted(all_cores.keys()):
        total = round(float(all_cores[core_name].sum()), 1)
        nonzero = int(np.count_nonzero(all_cores[core_name]))
        logger.debug("%s total=%.1f nonzero=%d", core_name, total, nonzero)

    logger.info("Matrix: %s", bcfg.matrix_path)
    logger.info("Summary: %s", summary_path)
    logger.info("Timing: %s", marks)


def assert_build_demand_prerequisites(cfg: dict) -> None:
    """Fail fast before ``load_or_build_od_matrix`` (used by ``run.py`` preflight)."""
    from sim.io_project import as_path

    root = Path(cfg["_meta"]["project_root"])
    zoning = cfg.get("zoning") or {}
    zout = Path(zoning.get("output_dir", "outputs/baseline/zones"))
    zones_path = zout / "zones.geojson"
    mapping_path = zout / "zone_centroid_mapping.json"
    if not zones_path.exists():
        raise FileNotFoundError(
            f"build-demand: missing zones export {zones_path}. Run build-zones first."
        )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"build-demand: missing centroid mapping {mapping_path}. Run build-zones first."
        )

    ext = ((cfg.get("demand") or {}).get("sldb") or {}).get("external_processing") or {}
    if not ext.get("enabled", True):
        return

    demand = cfg.get("demand") or {}
    seg_ext_through = (demand.get("segments") or {}).get("external_through") or {}
    _cache_dir = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))

    def _resolve(p: Any) -> Path:
        path = Path(p) if not isinstance(p, Path) else p
        return path if path.is_absolute() else (root / path).resolve()

    gw = _resolve(
        ext.get("external_gateway_lookup_path", f"{_cache_dir}/external_gateway_lookup.parquet")
    )
    tp_default = ext.get("through_gateway_pairs_path")
    if not tp_default:
        tp_default = seg_ext_through.get(
            "data_driven_pairs_path", f"{_cache_dir}/through_gateway_pairs.parquet"
        )
    tp = _resolve(tp_default)

    if not gw.exists():
        raise FileNotFoundError(
            f"build-demand: missing external gateway lookup {gw}. Run build-supernetwork first."
        )
    if ext.get("use_through_traffic", True) and not tp.exists():
        raise FileNotFoundError(
            f"build-demand: missing through gateway pairs {tp}. Run build-supernetwork first."
        )

    sn = cfg.get("supernetwork") or {}
    sn_out = _resolve(sn.get("output_dir", "outputs/baseline/supernetwork"))
    summary = sn_out / "supernetwork_summary.json"
    if not summary.exists():
        raise FileNotFoundError(
            f"build-demand: missing supernetwork summary {summary}. Run build-supernetwork first."
        )
