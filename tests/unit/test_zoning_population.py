"""Unit tests for population auto-remap logic in sim.zoning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sim.zoning import _population_needs_remap


class TestPopulationNeedsRemap:
    def test_missing_file(self, tmp_path: Path):
        assert _population_needs_remap(tmp_path / "nonexistent.parquet") is True

    def test_all_zone_id_zero(self, tmp_path: Path):
        p = tmp_path / "zone_population.parquet"
        df = pd.DataFrame({
            "zone_id": [0, 0, 0],
            "zone_name": ["A", "B", "C"],
            "population": [100, 200, 300],
            "match": ["no_zones", "no_zones", "no_zones"],
        })
        df.to_parquet(p, index=False)
        assert _population_needs_remap(p) is True

    def test_real_zone_ids(self, tmp_path: Path):
        p = tmp_path / "zone_population.parquet"
        df = pd.DataFrame({
            "zone_id": [423541, 423542, 423543],
            "zone_name": ["X", "Y", "Z"],
            "population": [1000, 2000, 3000],
            "match": ["obec_exact", "obec_exact", "mc_area:Brno-střed"],
        })
        df.to_parquet(p, index=False)
        assert _population_needs_remap(p) is False

    def test_empty_parquet(self, tmp_path: Path):
        p = tmp_path / "zone_population.parquet"
        df = pd.DataFrame({"zone_id": pd.Series([], dtype=int), "population": pd.Series([], dtype=int)})
        df.to_parquet(p, index=False)
        assert _population_needs_remap(p) is True

    def test_missing_zone_id_column(self, tmp_path: Path):
        p = tmp_path / "zone_population.parquet"
        df = pd.DataFrame({"population": [100, 200]})
        df.to_parquet(p, index=False)
        assert _population_needs_remap(p) is True

    def test_mixed_ids_with_one_zero(self, tmp_path: Path):
        """If at least one zone_id is non-zero, the file is considered valid."""
        p = tmp_path / "zone_population.parquet"
        df = pd.DataFrame({
            "zone_id": [0, 423541, 423542],
            "zone_name": ["fallback", "A", "B"],
            "population": [100, 1000, 2000],
            "match": ["no_zones", "obec_exact", "obec_exact"],
        })
        df.to_parquet(p, index=False)
        assert _population_needs_remap(p) is False
