"""Default Czech dataset source registry and YAML merge logic."""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

_CZ_DEFAULT_SOURCES: Dict[str, Dict[str, Any]] = {
    "commuting_sldb2021": {
        "enabled": True,
        "provider": "csu_open_data_csv",
        "url": "https://csu.gov.cz/docs/107508/4dbdab3b-905c-deff-4cfa-e4828a6fa2de/dojizdka_obce.csv?version=1.0",
        "out_path": "data/sources/csu/sldb2021/dojizdka_obce.csv",
        "purposes": ["work", "school"],
        "preprocess": {"write_filtered": True, "write_full_cr": True},
    },
    "validation_csd2025_v2": {
        "enabled": True,
        "year": 2025,
        "provider": "http_file",
        "url": "https://www.rsd.cz/documents/38144/3734982/V2_CSD_2025.xlsx/50663492-395b-0fd4-f365-d18440997ca5?t=1773922634346",
        "out_path": "data/sources/rsd/csd2025/V2_CSD2025.xlsx",
        "format": {"type": "xlsx"},
        "usage": {"validation_target": "aadt_screenlines"},
    },
    "population_sldb2021": {
        "enabled": True,
        "provider": "http_file",
        "url": "https://csu.gov.cz/docs/107508/79c509a0-261c-b4dd-d58d-c05955a24a2c/sldb2021_pohlavi.csv",
        "out_path": "data/sources/csu/sldb2021/populace_pohlavi.csv",
        "format": {"type": "csv"},
        "usage": {"socioeconomic": "population_per_zone"},
    },
    "cz_place_centroids": {
        "enabled": True,
        "provider": "atom_file",
        "url": "https://atom.cuzk.gov.cz/get.ashx?theme=RUIAN-CSV-ADR-ST",
        "feed_cache_path": "data/sources/cz/places/ruian_csv_adr_st.atom.xml",
        "out_path": "data/sources/cz/places/ruian_csv_adr_st.zip",
        "format": {
            "type": "zip_csv",
            "asset_pattern": r"(?i)(^|/)[0-9]{8}_OB_ADR_csv\.zip$",
            "member_pattern": r"(?i)\.csv$",
            "delimiter": ";",
            "encoding": "cp1250",
            "source_crs_epsg": 2065,
            "output_crs_epsg": 4326,
        },
        "columns": {
            "place_code": "Kód obce",
            "place_name": "Název obce",
            "x": "Souřadnice X",
            "y": "Souřadnice Y",
        },
        "usage": {"supernetwork_places": "grouped_point_centroids"},
    },
    "closures_pg": {
        "enabled": True,
        "provider": "postgres_closures",
        "table": "restrictions",
        "status_whitelist": [],
        "min_observed_days": 2,
        "usage": {"calibration_target": "baseline_closures"},
    },
    "cz_roads_major_pbf": {
        "enabled": True,
        "provider": "http_file",
        "url": "https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf",
        "out_path": "data/sources/osm/czech-republic-latest.osm.pbf",
    },
}


def merge_dataset_sources(yaml_sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge YAML dataset source overrides on top of built-in CZ defaults."""
    merged = copy.deepcopy(_CZ_DEFAULT_SOURCES)
    if not yaml_sources:
        return merged
    for key, overrides in yaml_sources.items():
        if key in merged:
            if overrides is None:
                continue
            merged[key].update(overrides)
        else:
            merged[key] = overrides or {}
    return merged
