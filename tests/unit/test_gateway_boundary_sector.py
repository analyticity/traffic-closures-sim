"""Through-traffic filter: reject external-external pairs on the same AOI boundary sector."""
from __future__ import annotations

import pandas as pd

from sim.supernetwork.classification import _boundary_sector, classify_relations
from sim.supernetwork.config import SuperCfg


def _minimal_super_cfg(**overrides) -> SuperCfg:
    base = dict(
        metric_epsg=5514,
        output_dir=overrides.get("output_dir"),
        cache_dir=overrides.get("cache_dir"),
        pbf_path=overrides.get("pbf_path"),
        place_centroids_path=overrides.get("place_centroids_path"),
        place_centroids_crs_epsg=4326,
        highway_types=["motorway", "trunk"],
        contract_graph=False,
        contract_degree=2,
        max_candidate_gateways=3,
        detour_ratio_max=1.25,
        max_extra_minutes=18.0,
        allow_same_gateway_pair=False,
        reject_same_boundary_sector=overrides.get("reject_same_boundary_sector", True),
        model_area_path=overrides.get("model_area_path"),
        zones_path=overrides.get("zones_path"),
        gateway_seed_lookup_path=overrides.get("gateway_seed_lookup_path"),
        gateway_diagnostics_path=overrides.get("gateway_diagnostics_path"),
        full_cr_commuting_parquet=overrides.get("full_cr_commuting_parquet"),
        full_cr_commuting_csv=overrides.get("full_cr_commuting_csv"),
        national_nodes_path=overrides.get("national_nodes_path"),
        national_edges_path=overrides.get("national_edges_path"),
        external_unit_lookup_path=overrides.get("external_unit_lookup_path"),
        external_gateway_lookup_path=overrides.get("external_gateway_lookup_path"),
        through_gateway_pairs_path=overrides.get("through_gateway_pairs_path"),
        unresolved_places_path=overrides.get("unresolved_places_path"),
        classified_relations_path=overrides.get("classified_relations_path"),
        raw_major_roads_cache=overrides.get("raw_major_roads_cache"),
        eligible_gateway_types=None,
    )
    return SuperCfg(**base)


def test_boundary_sector_groups_south_gateways() -> None:
    # Brno diagnostics: 52_S ≈ 179°, D2_S ≈ 162° → same 45° sector
    assert _boundary_sector(179.28) == _boundary_sector(161.77)


def test_boundary_sector_west_vs_east() -> None:
    assert _boundary_sector(268.31) != _boundary_sector(106.57)


def test_classify_rejects_same_sector_pair(tmp_path) -> None:
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge(1, 2, weight=100.0)

    commuting = pd.DataFrame([{
        "origin_place": "A",
        "dest_place": "B",
        "origin_place_norm": "a",
        "dest_place_norm": "b",
        "origin_place_key": "a|",
        "dest_place_key": "b|",
        "origin_unit_id": "u1",
        "dest_unit_id": "u2",
        "persons_work": 100,
        "persons_school": 0,
    }])

    gateway_lookup = pd.DataFrame([
        {
            "unit_id": "u1",
            "place_key": "a|",
            "gateway_name": "52_S",
            "rank_in": 1,
            "rank_out": 1,
            "route_cost_to_gateway_s": 10.0,
            "route_cost_from_gateway_s": 10.0,
            "unit_graph_node": 1,
        },
        {
            "unit_id": "u2",
            "place_key": "b|",
            "gateway_name": "D2_S",
            "rank_in": 1,
            "rank_out": 1,
            "route_cost_to_gateway_s": 10.0,
            "route_cost_from_gateway_s": 10.0,
            "unit_graph_node": 2,
        },
    ])

    pair_costs = pd.DataFrame([
        {"gateway_in": "52_S", "gateway_out": "D2_S", "internal_cost_s": 50.0},
    ])

    p = tmp_path
    cfg = _minimal_super_cfg(
        output_dir=p / "out",
        cache_dir=p / "cache",
        pbf_path=p / "x.pbf",
        place_centroids_path=p / "pc.parquet",
        model_area_path=p / "ma.geojson",
        zones_path=p / "z.geojson",
        gateway_seed_lookup_path=p / "gw.parquet",
        gateway_diagnostics_path=p / "gw.csv",
        full_cr_commuting_parquet=p / "c.parquet",
        full_cr_commuting_csv=p / "c.csv",
        national_nodes_path=p / "n.parquet",
        national_edges_path=p / "e.parquet",
        external_unit_lookup_path=p / "u.parquet",
        external_gateway_lookup_path=p / "gl.parquet",
        through_gateway_pairs_path=p / "tp.parquet",
        unresolved_places_path=p / "ur.parquet",
        classified_relations_path=p / "cr.parquet",
        raw_major_roads_cache=p / "r.parquet",
        reject_same_boundary_sector=True,
    )

    angles = {"52_S": 179.28, "D2_S": 161.77, "D1_W": 268.31, "D1_E": 106.57}
    classified, through = classify_relations(
        G,
        commuting,
        gateway_lookup,
        pair_costs,
        internal_zone_names=set(),
        cfg_root={"demand": {"vehicle_occupancy": {"work": 1.0, "school": 1.0}}},
        cfg=cfg,
        gateway_boundary_angles=angles,
    )

    row = classified.iloc[0]
    assert row["rejection_reason"] == "same_boundary_sector"
    assert through.empty
