"""SuperCfg dataclass and builder -- all supernetwork configuration in one place."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim.datasets.paths import (
    resolved_commuting_full_cr_parquet_path,
    resolved_cz_place_centroids_parquet_path,
)
from sim.io_project import as_path, get_metric_epsg, get_nested


@dataclass(frozen=True)
class SuperCfg:
    metric_epsg: int
    output_dir: Path
    cache_dir: Path
    pbf_path: Path
    place_centroids_path: Path
    place_centroids_crs_epsg: int
    highway_types: List[str]
    contract_graph: bool
    contract_degree: int
    max_candidate_gateways: int
    detour_ratio_max: float
    max_extra_minutes: float
    allow_same_gateway_pair: bool
    reject_same_boundary_sector: bool
    model_area_path: Path
    zones_path: Path
    gateway_seed_lookup_path: Path
    gateway_diagnostics_path: Path
    full_cr_commuting_parquet: Path
    full_cr_commuting_csv: Path
    national_nodes_path: Path
    national_edges_path: Path
    external_unit_lookup_path: Path
    external_gateway_lookup_path: Path
    through_gateway_pairs_path: Path
    unresolved_places_path: Path
    classified_relations_path: Path
    raw_major_roads_cache: Path
    eligible_gateway_types: Optional[List[str]]


def build_cfg(cfg_root: Dict[str, Any]) -> SuperCfg:
    sn = cfg_root.get("supernetwork") or {}
    outputs = sn.get("outputs") or {}
    cache_dir = as_path(sn.get("cache_dir", "data/cache/supernetwork"))
    output_dir = as_path(sn.get("output_dir", "outputs/baseline/supernetwork"))
    zoning_output_dir = as_path(get_nested(cfg_root, ["zoning", "output_dir"], "outputs/baseline/zones"))
    commute_src = get_nested(cfg_root, ["datasets", "sources", "commuting_sldb2021"], {}) or {}
    general_cache = as_path(get_nested(cfg_root, ["datasets", "cache_dir"], "data/cache"))
    highway_types = list(
        get_nested(sn, ["national_network", "highway_types"], ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"])
    )
    hw_sig = hashlib.sha1(
        "|".join(sorted({str(x).strip() for x in highway_types if str(x).strip()})).encode("utf-8")
    ).hexdigest()[:12]

    return SuperCfg(
        metric_epsg=get_metric_epsg(cfg_root),
        output_dir=output_dir,
        cache_dir=cache_dir,
        pbf_path=as_path(sn.get("pbf_path", "data/sources/osm/czech-republic-latest.osm.pbf")),
        place_centroids_path=(
            as_path(sn["place_centroids_path"])
            if sn.get("place_centroids_path")
            else resolved_cz_place_centroids_parquet_path(cfg_root)
        ),
        place_centroids_crs_epsg=int(sn.get("place_centroids_crs_epsg", 4326)),
        highway_types=highway_types,
        contract_graph=bool(sn.get("contract_graph", True)),
        contract_degree=int(sn.get("contract_exclude_degree_leq", 2)),
        max_candidate_gateways=int(get_nested(sn, ["gateway_mapping", "max_candidate_gateways"], 3)),
        detour_ratio_max=float(get_nested(sn, ["relation_filter", "detour_ratio_max"], 1.40)),
        max_extra_minutes=float(get_nested(sn, ["relation_filter", "max_extra_minutes"], 25.0)),
        allow_same_gateway_pair=bool(get_nested(sn, ["relation_filter", "allow_same_gateway_pair"], False)),
        reject_same_boundary_sector=bool(
            get_nested(sn, ["relation_filter", "reject_same_boundary_sector"], False)
        ),
        model_area_path=zoning_output_dir / "model_area.geojson",
        zones_path=zoning_output_dir / "zones.geojson",
        gateway_seed_lookup_path=as_path(get_nested(cfg_root, ["zoning", "external_gateways", "export_lookup_path"], str(general_cache / "gateway_lookup_seed.parquet"))),
        gateway_diagnostics_path=zoning_output_dir / "gateway_diagnostics.csv",
        full_cr_commuting_parquet=(
            as_path(commute_src["full_cr_out_parquet"])
            if commute_src.get("full_cr_out_parquet")
            else resolved_commuting_full_cr_parquet_path(cfg_root)
        ),
        full_cr_commuting_csv=as_path(commute_src.get("out_path", "data/sources/csu/sldb2021/dojizdka_obce.csv")),
        national_nodes_path=as_path(
            outputs.get("national_nodes", cache_dir / f"national_nodes_{hw_sig}.parquet")
        ),
        national_edges_path=as_path(
            outputs.get("national_edges", cache_dir / f"national_edges_{hw_sig}.parquet")
        ),
        external_unit_lookup_path=as_path(outputs.get("external_unit_lookup", str(general_cache / "external_unit_lookup.parquet"))),
        external_gateway_lookup_path=as_path(outputs.get("external_gateway_lookup", str(general_cache / "external_gateway_lookup.parquet"))),
        through_gateway_pairs_path=as_path(outputs.get("through_gateway_pairs", str(general_cache / "through_gateway_pairs.parquet"))),
        unresolved_places_path=cache_dir / "unresolved_external_places.parquet",
        classified_relations_path=cache_dir / "classified_external_relations.parquet",
        raw_major_roads_cache=cache_dir / f"major_roads_raw_{hw_sig}.parquet",
        eligible_gateway_types=sn.get("eligible_gateway_types", None),
    )
