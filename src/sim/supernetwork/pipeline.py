"""Supernetwork pipeline orchestrator -- end-to-end entry point."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd

from sim.datasets.utils import ensure_dir
from sim.io_project import load_config
from sim.supernetwork.classification import classify_relations
from sim.supernetwork.config import SuperCfg, build_cfg
from sim.supernetwork.graph import (
    build_graph_from_roads,
    contract_graph,
    graph_from_parquets,
    graph_to_gdfs,
    load_or_extract_major_roads,
    snap_points_to_graph,
)
from sim.supernetwork.inputs import (
    aggregate_commuting_pairs,
    load_gateways,
    load_internal_zone_names,
    load_model_area,
    load_place_centroids,
    read_commuting,
    resolve_external_units,
)
from sim.supernetwork.plot import _external_units_outside_model_area, plot_overview
from sim.supernetwork.routing import build_gateway_costs, build_gateway_lookup

logger = logging.getLogger(__name__)


def _gateway_boundary_angles(gateways: gpd.GeoDataFrame, cfg: SuperCfg) -> Dict[str, float]:
    """Map gateway_name → boundary_angle (degrees on AOI), from frame or diagnostics CSV."""
    gdf = gateways
    if "boundary_angle" not in gdf.columns and cfg.gateway_diagnostics_path.exists():
        diag = pd.read_csv(cfg.gateway_diagnostics_path, usecols=["gateway_name", "boundary_angle"])
        gdf = gdf.merge(diag, on="gateway_name", how="left")
    if "boundary_angle" not in gdf.columns:
        return {}
    out: Dict[str, float] = {}
    for _, row in gdf.iterrows():
        ang = row.get("boundary_angle")
        if pd.notna(ang):
            out[str(row["gateway_name"])] = float(ang)
    return out


def run(config_path: str = "config/brno/sim.yaml") -> Dict[str, Any]:
    t_run = time.perf_counter()
    phase_t = time.perf_counter()
    profile: Dict[str, Any] = {"phases_s": {}}
    cfg_root = load_config(config_path)
    cfg = build_cfg(cfg_root)
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.cache_dir)

    model_area = load_model_area(cfg.model_area_path, cfg.metric_epsg)
    gateways_all = load_gateways(cfg)

    if cfg.eligible_gateway_types and "link_type" in gateways_all.columns:
        eligible_set = {t.strip() for t in cfg.eligible_gateway_types}
        type_col = "predominant_link_type" if "predominant_link_type" in gateways_all.columns else "link_type"
        type_ok = gateways_all[type_col].astype(str).isin(eligible_set)
        if type_col != "link_type":
            type_ok = type_ok | gateways_all["link_type"].astype(str).isin(eligible_set)
        is_whitelist = ~gateways_all["auto_discovered"].astype(bool) if "auto_discovered" in gateways_all.columns else pd.Series(True, index=gateways_all.index)
        mask = type_ok | is_whitelist
        dropped = gateways_all[~mask]["gateway_name"].tolist()
        gateways = gateways_all[mask].copy()
        if dropped:
            logger.info(
                "Supernetwork: filtered auto-discovered gateways by eligible types %s",
                sorted(eligible_set),
            )
            logger.info("Kept %d: %s", len(gateways), sorted(gateways["gateway_name"].tolist()))
            logger.info("Dropped %d auto-discovered: %s", len(dropped), dropped)
    else:
        gateways = gateways_all

    internal_zone_names = load_internal_zone_names(cfg.zones_path)
    profile["phases_s"]["load_core_inputs"] = round(time.perf_counter() - phase_t, 3)

    phase_t = time.perf_counter()
    commuting_raw = read_commuting(cfg)
    commuting_pairs = aggregate_commuting_pairs(commuting_raw)
    place_centroids = load_place_centroids(cfg)
    external_units = resolve_external_units(place_centroids, commuting_pairs, internal_zone_names, cfg.unresolved_places_path)
    profile["phases_s"]["load_and_prepare_datasets"] = round(time.perf_counter() - phase_t, 3)
    profile["relation_counts"] = {
        "commuting_pairs": int(len(commuting_pairs)),
        "external_units": int(len(external_units)),
    }

    gw_hash = hashlib.sha1(
        ",".join(sorted(gateways["gateway_name"].astype(str))).encode()
    ).hexdigest()[:12]
    gw_hash_path = cfg.national_nodes_path.with_suffix(".gw_hash")
    _cache_valid = (
        cfg.national_nodes_path.exists()
        and cfg.national_edges_path.exists()
        and gw_hash_path.exists()
        and gw_hash_path.read_text().strip() == gw_hash
    )
    if _cache_valid:
        phase_t = time.perf_counter()
        G = graph_from_parquets(cfg.national_nodes_path, cfg.national_edges_path)
        nodes_metric = gpd.read_parquet(cfg.national_nodes_path)
        edges_metric = gpd.read_parquet(cfg.national_edges_path)
        profile["phases_s"]["load_cached_graph"] = round(time.perf_counter() - phase_t, 3)
        profile["graph_source"] = "cache"
    else:
        phase_t = time.perf_counter()
        roads = load_or_extract_major_roads(cfg)
        profile["phases_s"]["load_or_extract_roads"] = round(time.perf_counter() - phase_t, 3)

        phase_t = time.perf_counter()
        G_raw, graph_build_stats = build_graph_from_roads(roads)
        profile["graph_build"] = graph_build_stats
        profile["phases_s"]["build_graph_from_roads"] = round(time.perf_counter() - phase_t, 3)

        gw_metric = gateways.to_crs(epsg=cfg.metric_epsg) if gateways.crs and gateways.crs.to_epsg() != cfg.metric_epsg else gateways.copy()
        units_metric = external_units.to_crs(epsg=cfg.metric_epsg) if external_units.crs.to_epsg() != cfg.metric_epsg else external_units.copy()

        gw_raw = gw_metric.merge(snap_points_to_graph(G_raw, gw_metric[["gateway_name", "geometry"]], "gateway_name"), on="gateway_name", how="left")
        unit_raw = units_metric.merge(
            snap_points_to_graph(G_raw, units_metric[["unit_id", "geometry"]], "unit_id"),
            on="unit_id",
            how="left",
        )
        protected = set(gw_raw["graph_node"].dropna().astype(int)) | set(unit_raw["graph_node"].dropna().astype(int))
        phase_t = time.perf_counter()
        G = contract_graph(G_raw, protected, cfg.contract_degree) if cfg.contract_graph else G_raw
        profile["phases_s"]["contract_graph"] = round(time.perf_counter() - phase_t, 3)
        profile["contract_graph"] = {
            "enabled": bool(cfg.contract_graph),
            "protected_nodes": int(len(protected)),
            "nodes_before": int(G_raw.number_of_nodes()),
            "edges_before": int(G_raw.number_of_edges()),
            "nodes_after": int(G.number_of_nodes()),
            "edges_after": int(G.number_of_edges()),
        }
        phase_t = time.perf_counter()
        nodes_metric, edges_metric = graph_to_gdfs(G, cfg.metric_epsg)
        nodes_metric.to_parquet(cfg.national_nodes_path, index=False)
        edges_metric.to_parquet(cfg.national_edges_path, index=False)
        gw_hash_path.write_text(gw_hash)
        profile["phases_s"]["persist_graph_cache"] = round(time.perf_counter() - phase_t, 3)
        profile["graph_source"] = "rebuilt"

    phase_t = time.perf_counter()
    gateways_join = gateways.merge(snap_points_to_graph(G, gateways[["gateway_name", "geometry"]].to_crs(epsg=cfg.metric_epsg), "gateway_name"), on="gateway_name", how="left")
    units_join = external_units.merge(
        snap_points_to_graph(
            G,
            external_units[["unit_id", "geometry"]].to_crs(epsg=cfg.metric_epsg),
            "unit_id",
        ),
        on="unit_id",
        how="left",
    )
    units_join = units_join.dropna(subset=["graph_node"]).copy()
    profile["phases_s"]["snap_gateways_and_units"] = round(time.perf_counter() - phase_t, 3)

    unique_unit_nodes = int(units_join["graph_node"].nunique()) if not units_join.empty else 0
    if len(units_join) > 100 and unique_unit_nodes <= 1:
        raise RuntimeError(
            "Invalid supernetwork cache: all external units snapped to a single graph node. "
            f"Remove stale cache files in {cfg.cache_dir} and rerun build-supernetwork."
        )

    unit_lookup = units_join[
        ["unit_id", "place_code", "place_name", "district_name", "admin_level", "place_key", "graph_node", "graph_x", "graph_y", "geometry"]
    ].rename(columns={"graph_node": "unit_graph_node"})
    phase_t = time.perf_counter()
    unit_lookup.to_parquet(cfg.external_unit_lookup_path, index=False)
    profile["phases_s"]["write_unit_lookup"] = round(time.perf_counter() - phase_t, 3)

    costs_from_gateway, costs_to_gateway, gateway_pair_costs = build_gateway_costs(G, gateways_join, profile=profile)
    gw_snap_dists = {
        str(r["gateway_name"]): float(r["snap_distance_m"])
        for _, r in gateways_join.iterrows()
        if pd.notna(r.get("snap_distance_m"))
    }
    gateway_lookup = build_gateway_lookup(
        units_join, costs_from_gateway, costs_to_gateway, cfg,
        gateway_snap_distances=gw_snap_dists,
    )
    profile["gateway_lookup"] = gateway_lookup.attrs.get("profile", {})
    phase_t = time.perf_counter()
    gateway_lookup.to_parquet(cfg.external_gateway_lookup_path, index=False)
    profile["phases_s"]["write_gateway_lookup"] = round(time.perf_counter() - phase_t, 3)

    phase_t = time.perf_counter()
    gateway_boundary_angles = _gateway_boundary_angles(gateways, cfg)
    classified, through_pairs = classify_relations(
        G,
        commuting_pairs,
        gateway_lookup,
        gateway_pair_costs,
        internal_zone_names,
        cfg_root,
        cfg,
        gateway_boundary_angles=gateway_boundary_angles,
    )
    profile["classify_relations"] = classified.attrs.get("profile", {})
    profile["phases_s"]["classify_relations"] = round(time.perf_counter() - phase_t, 3)
    phase_t = time.perf_counter()
    classified.to_parquet(cfg.classified_relations_path, index=False)
    through_pairs.to_parquet(cfg.through_gateway_pairs_path, index=False)
    profile["phases_s"]["write_relation_outputs"] = round(time.perf_counter() - phase_t, 3)

    rejected_ext_ext = classified[(classified["classification"] == "external_external") & (classified["accepted"] == False)]  # noqa: E712
    reason_counts = (
        rejected_ext_ext["rejection_reason"].fillna("unspecified").value_counts().to_dict()
        if not rejected_ext_ext.empty else {}
    )
    ext_ext_total = int((classified["classification"] == "external_external").sum())
    ext_ext_accepted = int(((classified["classification"] == "external_external") & (classified["accepted"] == True)).sum())  # noqa: E712
    resolved_coverage = (len(units_join) / max(len(external_units), 1)) * 100.0
    logger.info(
        "External unit coverage: %d/%d (%.1f%%)",
        len(units_join),
        len(external_units),
        resolved_coverage,
    )
    logger.info("External-external accepted: %d/%d", ext_ext_accepted, ext_ext_total)
    if reason_counts:
        logger.info("External-external rejection reasons: %s", reason_counts)

    ei_ok = classified[(classified["classification"] == "external_internal") & (classified["accepted"] == True)]  # noqa: E712
    inbound_by_gw: Dict[str, float] = {}
    if not ei_ok.empty and "gateway_in" in ei_ok.columns:
        inbound_by_gw = ei_ok.groupby("gateway_in", dropna=False)["vehicles_daily"].sum().sort_values(ascending=False).to_dict()
        inbound_by_gw = {str(k): float(v) for k, v in inbound_by_gw.items() if str(k).strip()}

    ie_ok = classified[(classified["classification"] == "internal_external") & (classified["accepted"] == True)]  # noqa: E712
    outbound_by_gw: Dict[str, float] = {}
    if not ie_ok.empty and "gateway_out" in ie_ok.columns:
        outbound_by_gw = ie_ok.groupby("gateway_out", dropna=False)["vehicles_daily"].sum().sort_values(ascending=False).to_dict()
        outbound_by_gw = {str(k): float(v) for k, v in outbound_by_gw.items() if str(k).strip()}

    through_by_pair: List[Dict[str, Any]] = []
    if not through_pairs.empty:
        for _, tr in through_pairs.iterrows():
            through_by_pair.append({
                "gateway_in": str(tr["gateway_in"]),
                "gateway_out": str(tr["gateway_out"]),
                "vehicles_daily": float(tr["vehicles_daily"]),
            })

    if inbound_by_gw:
        logger.info("Gateway inbound (external -> model), vehicles/day:")
        for k, v in sorted(inbound_by_gw.items(), key=lambda x: -x[1]):
            logger.info("  %s: %s", k, f"{v:,.0f}")
    if outbound_by_gw:
        logger.info("Gateway outbound (model -> external), vehicles/day:")
        for k, v in sorted(outbound_by_gw.items(), key=lambda x: -x[1]):
            logger.info("  %s: %s", k, f"{v:,.0f}")
    if through_by_pair:
        logger.info("Through traffic (gateway -> gateway), vehicles/day:")
        for item in sorted(through_by_pair, key=lambda x: -x["vehicles_daily"])[:20]:
            logger.info(
                "  %s -> %s: %s",
                item["gateway_in"],
                item["gateway_out"],
                f"{item['vehicles_daily']:,.0f}",
            )
        if len(through_by_pair) > 20:
            logger.info("  ... +%d more pairs", len(through_by_pair) - 20)

    gateways_geojson = cfg.output_dir / "gateway_points.geojson"
    units_geojson = cfg.output_dir / "used_external_units.geojson"
    plot_png = cfg.output_dir / "supernetwork_overview.png"

    phase_t = time.perf_counter()
    gateways_join.to_crs(epsg=4326).to_file(gateways_geojson, driver="GeoJSON")
    units_map = _external_units_outside_model_area(units_join.to_crs(epsg=cfg.metric_epsg), model_area)
    units_map.to_crs(epsg=4326).to_file(units_geojson, driver="GeoJSON")
    plot_overview(edges_metric, model_area, gateways_join.to_crs(epsg=cfg.metric_epsg), units_join.to_crs(epsg=cfg.metric_epsg), plot_png)
    profile["phases_s"]["write_geo_outputs_and_plot"] = round(time.perf_counter() - phase_t, 3)

    # --- Gateway health diagnostics ---
    through_in_by_gw: Dict[str, float] = {}
    through_out_by_gw: Dict[str, float] = {}
    if not through_pairs.empty:
        for _, tr in through_pairs.iterrows():
            gw_in = str(tr["gateway_in"])
            gw_out = str(tr["gateway_out"])
            vd = float(tr["vehicles_daily"])
            through_in_by_gw[gw_in] = through_in_by_gw.get(gw_in, 0.0) + vd
            through_out_by_gw[gw_out] = through_out_by_gw.get(gw_out, 0.0) + vd

    gw_health: List[Dict[str, Any]] = []
    gw_health_warnings: List[str] = []
    gw_graph_nodes: Dict[str, int] = {}
    for _, gw in gateways_join.iterrows():
        name = str(gw["gateway_name"])
        graph_node = int(gw["graph_node"]) if pd.notna(gw.get("graph_node")) else None
        snap_dist = float(gw["snap_distance_m"]) if pd.notna(gw.get("snap_distance_m")) else None
        inb = inbound_by_gw.get(name, 0.0)
        outb = outbound_by_gw.get(name, 0.0)
        thr_in = through_in_by_gw.get(name, 0.0)
        thr_out = through_out_by_gw.get(name, 0.0)
        total_traffic = inb + outb + thr_in + thr_out

        issues: List[str] = []
        if total_traffic == 0:
            issues.append("zero_traffic")
        if snap_dist is not None and snap_dist > 500.0:
            issues.append(f"large_snap_distance_{snap_dist:.0f}m")
        if graph_node is not None:
            if graph_node in gw_graph_nodes.values():
                dup_name = [k for k, v in gw_graph_nodes.items() if v == graph_node][0]
                issues.append(f"shares_graph_node_with_{dup_name}")
            gw_graph_nodes[name] = graph_node

        entry = {
            "gateway_name": name,
            "graph_node": graph_node,
            "snap_distance_m": round(snap_dist, 1) if snap_dist is not None else None,
            "inbound_vehicles_daily": round(inb, 1),
            "outbound_vehicles_daily": round(outb, 1),
            "through_in_vehicles_daily": round(thr_in, 1),
            "through_out_vehicles_daily": round(thr_out, 1),
            "total_vehicles_daily": round(total_traffic, 1),
            "issues": issues,
        }
        gw_health.append(entry)
        if issues:
            gw_health_warnings.append(f"{name}: {', '.join(issues)}")

    if gw_health_warnings:
        for w in gw_health_warnings:
            logger.warning("%s", w)

    matrix_csv_path = cfg.output_dir / "gateway_through_matrix.csv"
    matrix_txt_path = cfg.output_dir / "gateway_through_matrix.txt"
    gw_order = sorted({str(x) for x in gateways_join["gateway_name"].tolist()})
    if not through_pairs.empty:
        tp = through_pairs.copy()
        tp["gateway_in"] = tp["gateway_in"].astype(str)
        tp["gateway_out"] = tp["gateway_out"].astype(str)
        pivot = tp.pivot_table(
            index="gateway_in",
            columns="gateway_out",
            values="vehicles_daily",
            aggfunc="sum",
            fill_value=0.0,
        )
        gw_order = sorted(set(gw_order) | set(pivot.index) | set(pivot.columns))
        through_matrix = pivot.reindex(index=gw_order, columns=gw_order, fill_value=0.0)
    else:
        through_matrix = pd.DataFrame(0.0, index=gw_order, columns=gw_order) if gw_order else pd.DataFrame()
    through_matrix.to_csv(matrix_csv_path, encoding="utf-8")
    matrix_txt_path.write_text(through_matrix.to_string() if not through_matrix.empty else "", encoding="utf-8")

    summary = {
        "gateway_count": int(len(gateways_join)),
        "resolved_external_units": int(len(units_join)),
        "unresolved_external_places_path": str(cfg.unresolved_places_path),
        "classified_relations": int(len(classified)),
        "accepted_external_external": ext_ext_accepted,
        "external_external_total": ext_ext_total,
        "external_external_rejections": reason_counts,
        "external_unit_coverage_pct": round(resolved_coverage, 2),
        "through_pairs_count": int(len(through_pairs)),
        "through_pairs_total_vehicles_daily": float(through_pairs["vehicles_daily"].sum()) if not through_pairs.empty else 0.0,
        "gateway_inbound_vehicles_daily": inbound_by_gw,
        "gateway_outbound_vehicles_daily": outbound_by_gw,
        "through_gateway_pairs_detail": through_by_pair,
        "gateway_health": gw_health,
        "gateway_health_warnings": gw_health_warnings,
        "graph_nodes": int(len(nodes_metric)),
        "graph_edges": int(len(edges_metric)),
        "profiling": profile,
        "elapsed_total_s": round(time.perf_counter() - t_run, 3),
        "outputs": {
            "national_nodes": str(cfg.national_nodes_path),
            "national_edges": str(cfg.national_edges_path),
            "external_unit_lookup": str(cfg.external_unit_lookup_path),
            "external_gateway_lookup": str(cfg.external_gateway_lookup_path),
            "through_gateway_pairs": str(cfg.through_gateway_pairs_path),
            "classified_relations": str(cfg.classified_relations_path),
            "gateways_geojson": str(gateways_geojson),
            "used_units_geojson": str(units_geojson),
            "plot": str(plot_png),
            "gateway_through_matrix_csv": str(matrix_csv_path),
            "gateway_through_matrix_txt": str(matrix_txt_path),
        },
    }
    summary_path = cfg.output_dir / "supernetwork_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_build_supernetwork(config_path: str | Path = "config/brno/sim.yaml") -> dict[str, Any]:
    return run(config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified supernetwork builder")
    parser.add_argument("--config", default="config/brno/sim.yaml")
    args = parser.parse_args()

    summary = run(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
