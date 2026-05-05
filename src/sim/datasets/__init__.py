"""Public API for the ``sim.datasets`` package.

Re-exports the most commonly imported names so that existing
``from sim.datasets import X`` statements keep working.
"""
from sim.datasets.pipeline import run_fetch_datasets
from sim.datasets.paths import (
    resolved_commuting_full_cr_parquet_path,
    resolved_csd2025_validation_parquet_path,
    resolved_cz_place_centroids_parquet_path,
)
from sim.datasets.csd import (
    ensure_csd2025_validation_parquet,
    normalize_csd_count_columns,
)
from sim.datasets.population import preprocess_population_sldb2021
from sim.datasets.employment import derive_zone_employment
from sim.datasets.registry import merge_dataset_sources

__all__ = [
    "run_fetch_datasets",
    "resolved_commuting_full_cr_parquet_path",
    "resolved_csd2025_validation_parquet_path",
    "resolved_cz_place_centroids_parquet_path",
    "ensure_csd2025_validation_parquet",
    "normalize_csd_count_columns",
    "preprocess_population_sldb2021",
    "derive_zone_employment",
    "merge_dataset_sources",
]
