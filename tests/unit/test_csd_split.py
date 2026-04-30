"""Unit tests for CSD calibration/validation split and link-count conversion.

Tests cover ``split_csd_for_calibration`` (all three strategies),
``load_csd_as_link_counts``, ``match_csd_to_links``, and edge cases
without requiring AequilibraE or network access.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from sim.calibration import (
    _CSD_COMPATIBLE_LINK_TYPES,
    _classify_csd_road,
    load_csd_as_link_counts,
    match_csd_to_links,
    split_csd_for_calibration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_csd(roads: list[tuple[str, float, float, float, str]] | None = None) -> pd.DataFrame:
    """Build a minimal CSD-like DataFrame.

    Each tuple: (sil, sv, o, tv, nazev_mesta).
    """
    if roads is None:
        roads = [
            ("D1",  50000, 40000, 10000, "Brno"),
            ("D1",  48000, 38000, 10000, "Brno"),
            ("D1",  46000, 36000, 10000, "Modřice"),
            ("D2",  30000, 25000, 5000,  "Brno"),
            ("D2",  28000, 23000, 5000,  "Rajhrad"),
            ("43",  20000, 18000, 2000,  "Kuřim"),
            ("43",  19000, 17000, 2000,  "Lipůvka"),
            ("150", 8000,  7000,  1000,  ""),
            ("150", 7500,  6500,  1000,  ""),
            ("373", 5000,  4500,  500,   ""),
            ("373", 4800,  4300,  500,   ""),
            ("3792", 2000, 1800,  200,   ""),
            ("3792", 1800, 1600,  200,   ""),
        ]
    return pd.DataFrame(roads, columns=["sil", "sv", "o", "tv", "nazev_mesta"])


def _make_links() -> gpd.GeoDataFrame:
    """Network links with osm_ref values matching the test CSD roads."""
    data = [
        {"link_id": 1, "osm_ref": "D1",  "link_type": "motorway",  "geometry": LineString([(0, 0), (1, 0)])},
        {"link_id": 2, "osm_ref": "D1",  "link_type": "motorway",  "geometry": LineString([(1, 0), (2, 0)])},
        {"link_id": 3, "osm_ref": "D2",  "link_type": "motorway",  "geometry": LineString([(0, 1), (1, 1)])},
        {"link_id": 4, "osm_ref": "43",  "link_type": "trunk",     "geometry": LineString([(0, 2), (1, 2)])},
        {"link_id": 5, "osm_ref": "150", "link_type": "secondary", "geometry": LineString([(0, 3), (1, 3)])},
        {"link_id": 6, "osm_ref": "373", "link_type": "secondary", "geometry": LineString([(0, 4), (1, 4)])},
        {"link_id": 7, "osm_ref": "3792","link_type": "tertiary",  "geometry": LineString([(0, 5), (1, 5)])},
        {"link_id": 8, "osm_ref": "999", "link_type": "residential","geometry": LineString([(0, 6), (1, 6)])},
    ]
    return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:5514")


def _make_links_with_volumes() -> gpd.GeoDataFrame:
    """Network links that also carry model volumes (for match_csd_to_links)."""
    data = [
        {"link_id": 1, "osm_ref": "D1",  "link_type": "motorway",  "distance": 5000,
         "total_vehicles_tot": 45000, "geometry": LineString([(0, 0), (1, 0)])},
        {"link_id": 2, "osm_ref": "D1",  "link_type": "motorway",  "distance": 3000,
         "total_vehicles_tot": 50000, "geometry": LineString([(1, 0), (2, 0)])},
        {"link_id": 3, "osm_ref": "D2",  "link_type": "motorway",  "distance": 4000,
         "total_vehicles_tot": 28000, "geometry": LineString([(0, 1), (1, 1)])},
        {"link_id": 4, "osm_ref": "43",  "link_type": "trunk",     "distance": 6000,
         "total_vehicles_tot": 18000, "geometry": LineString([(0, 2), (1, 2)])},
        {"link_id": 5, "osm_ref": "150", "link_type": "secondary", "distance": 7000,
         "total_vehicles_tot": 7000,  "geometry": LineString([(0, 3), (1, 3)])},
        {"link_id": 6, "osm_ref": "373", "link_type": "secondary", "distance": 3000,
         "total_vehicles_tot": 4500,  "geometry": LineString([(0, 4), (1, 4)])},
        {"link_id": 7, "osm_ref": "3792","link_type": "tertiary",  "distance": 2000,
         "total_vehicles_tot": 1500,  "geometry": LineString([(0, 5), (1, 5)])},
        {"link_id": 8, "osm_ref": "999", "link_type": "residential","distance": 1000,
         "total_vehicles_tot": 500,   "geometry": LineString([(0, 6), (1, 6)])},
    ]
    return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:5514")


# ---------------------------------------------------------------------------
# split_csd_for_calibration — alternating (default)
# ---------------------------------------------------------------------------

class TestSplitAlternating:
    def test_default_split(self):
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="alternating")
        assert len(calib) > 0
        assert len(valid) > 0
        assert len(calib) + len(valid) == len(csd)

    def test_all_road_classes_in_both(self):
        """Both subsets must contain all road classes."""
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="alternating")
        all_classes = set(csd["sil"].apply(_classify_csd_road).unique())
        calib_classes = set(calib["road_class"].unique())
        valid_classes = set(valid["road_class"].unique())
        assert calib_classes == all_classes
        assert valid_classes == all_classes

    def test_no_overlap(self):
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="alternating")
        shared = set(calib.index) & set(valid.index)
        assert shared == set()

    def test_each_road_split(self):
        """Roads with >=2 sections must appear in both subsets."""
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="alternating")
        for road in csd["sil"].unique():
            n_total = (csd["sil"] == road).sum()
            if n_total >= 2:
                assert (calib["sil"] == road).any(), f"{road} missing from calib"
                assert (valid["sil"] == road).any(), f"{road} missing from valid"

    def test_is_default_strategy(self):
        """Calling without strategy should use alternating."""
        csd = _make_csd()
        c1, v1 = split_csd_for_calibration(csd)
        c2, v2 = split_csd_for_calibration(csd, strategy="alternating")
        pd.testing.assert_frame_equal(c1.reset_index(drop=True), c2.reset_index(drop=True))

    def test_single_section_roads_go_to_calibration(self):
        """Roads with exactly one section should land in calibration."""
        csd = _make_csd([
            ("D1", 50000, 40000, 10000, "Brno"),
            ("43", 20000, 18000, 2000, "Kuřim"),
        ])
        calib, valid = split_csd_for_calibration(csd, strategy="alternating")
        assert len(calib) == 2
        assert len(valid) == 0


# ---------------------------------------------------------------------------
# split_csd_for_calibration — stratified
# ---------------------------------------------------------------------------

class TestSplitStratified:
    def test_deterministic(self):
        csd = _make_csd()
        c1, v1 = split_csd_for_calibration(csd, strategy="stratified", random_seed=42)
        c2, v2 = split_csd_for_calibration(csd, strategy="stratified", random_seed=42)
        pd.testing.assert_frame_equal(c1.reset_index(drop=True), c2.reset_index(drop=True))
        pd.testing.assert_frame_equal(v1.reset_index(drop=True), v2.reset_index(drop=True))

    def test_covers_all(self):
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="stratified", random_seed=0)
        assert len(calib) + len(valid) == len(csd)

    def test_all_road_classes_preserved(self):
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="stratified")
        all_classes = set(csd["sil"].apply(_classify_csd_road).unique())
        assert set(calib["road_class"].unique()) == all_classes

    def test_validation_has_road_classes(self):
        """Validation subset should also have multiple road classes."""
        csd = _make_csd()
        _, valid = split_csd_for_calibration(csd, strategy="stratified")
        assert len(valid["road_class"].unique()) >= 2

    def test_custom_share(self):
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(
            csd, strategy="stratified", calib_share=0.80,
        )
        assert len(calib) > len(valid)


# ---------------------------------------------------------------------------
# split_csd_for_calibration — spatial
# ---------------------------------------------------------------------------

class TestSplitSpatial:
    def test_urban_rural(self):
        csd = _make_csd()
        calib, valid = split_csd_for_calibration(csd, strategy="spatial")
        assert len(calib) > 0
        assert len(valid) > 0
        assert all(calib["nazev_mesta"].str.strip() != "")
        assert all(valid["nazev_mesta"].fillna("").str.strip() == "")

    def test_all_urban_falls_back(self):
        """When all sections are urban, spatial should fall back to alternating."""
        csd = _make_csd([
            ("D1", 50000, 40000, 10000, "Brno"),
            ("D1", 48000, 38000, 10000, "Brno"),
            ("43", 20000, 18000, 2000, "Kuřim"),
            ("43", 19000, 17000, 2000, "Lipůvka"),
        ])
        calib, valid = split_csd_for_calibration(csd, strategy="spatial")
        assert len(calib) > 0
        assert len(valid) > 0

    def test_all_rural_falls_back(self):
        """When all sections are rural, spatial should fall back to alternating."""
        csd = _make_csd([
            ("150", 8000, 7000, 1000, ""),
            ("150", 7500, 6500, 1000, ""),
            ("373", 5000, 4500, 500, ""),
            ("373", 4800, 4300, 500, ""),
        ])
        calib, valid = split_csd_for_calibration(csd, strategy="spatial")
        assert len(calib) > 0
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# split_csd_for_calibration — error handling / edge cases
# ---------------------------------------------------------------------------

class TestSplitErrors:
    def test_unknown_strategy(self):
        csd = _make_csd()
        with pytest.raises(ValueError, match="Unknown CSD split strategy"):
            split_csd_for_calibration(csd, strategy="nonexistent")

    def test_calib_share_zero_raises(self):
        csd = _make_csd()
        with pytest.raises(ValueError, match="calib_share must be in"):
            split_csd_for_calibration(csd, calib_share=0.0)

    def test_calib_share_one_raises(self):
        csd = _make_csd()
        with pytest.raises(ValueError, match="calib_share must be in"):
            split_csd_for_calibration(csd, calib_share=1.0)

    def test_calib_share_negative_raises(self):
        csd = _make_csd()
        with pytest.raises(ValueError, match="calib_share must be in"):
            split_csd_for_calibration(csd, calib_share=-0.1)

    def test_numeric_sil_handled(self):
        """Numeric sil column should not crash the split."""
        csd = pd.DataFrame({
            "sil": [1, 1, 43, 43, 150, 150],
            "sv": [50000, 48000, 20000, 19000, 8000, 7500],
            "o": [40000, 38000, 18000, 17000, 7000, 6500],
            "tv": [10000, 10000, 2000, 2000, 1000, 1000],
            "nazev_mesta": ["Brno", "Brno", "Kuřim", "", "", ""],
        })
        calib, valid = split_csd_for_calibration(csd, strategy="alternating")
        assert len(calib) + len(valid) == len(csd)


# ---------------------------------------------------------------------------
# load_csd_as_link_counts
# ---------------------------------------------------------------------------

class TestLoadCsdAsLinkCounts:
    def test_basic_conversion(self):
        csd = _make_csd()
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        for col in ("observed_car", "observed_total", "observed_motor_total", "observed_truck"):
            assert col in result.columns
        assert "geometry" in result.columns
        assert (result["observed_total"] > 0).all()

    def test_matched_roads(self):
        csd = _make_csd()
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        matched_roads = set(result["csd_road"].unique())
        assert "D1" in matched_roads
        assert "D2" in matched_roads

    def test_one_row_per_road(self):
        """Each CSD road should produce exactly one row (no per-link duplication)."""
        csd = _make_csd()
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        assert result["csd_road"].is_unique

    def test_no_match_returns_empty(self):
        csd = pd.DataFrame({
            "sil": ["ZZZZ"],
            "sv": [1000],
            "o": [900],
            "tv": [100],
            "nazev_mesta": [""],
        })
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        assert result.empty

    def test_missing_columns(self):
        csd = pd.DataFrame({"foo": [1]})
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        assert result.empty

    def test_crs_preserved(self):
        csd = _make_csd()
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        assert result.crs == links.crs

    def test_observed_car_equals_o_mean(self):
        """observed_car should equal the CSD 'o' column mean for the road."""
        csd = _make_csd([
            ("D1", 50000, 40000, 10000, "Brno"),
            ("D1", 50000, 42000, 8000, "Brno"),
        ])
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        d1_rows = result[result["csd_road"] == "D1"]
        assert len(d1_rows) > 0
        assert d1_rows["observed_car"].iloc[0] == pytest.approx(41000.0)

    def test_numeric_sil_handled(self):
        """Integer sil values should not crash the function."""
        csd = pd.DataFrame({
            "sil": [43, 43],
            "sv": [20000, 19000],
            "o": [18000, 17000],
            "tv": [2000, 2000],
            "nazev_mesta": ["", ""],
        })
        links = _make_links()
        result = load_csd_as_link_counts(csd, links)
        assert len(result) > 0
        assert result["csd_road"].iloc[0] == "43"

    def test_composite_osm_ref(self):
        """Links with semicolon-separated osm_ref should match CSD roads."""
        csd = pd.DataFrame({
            "sil": ["D1", "43"],
            "sv": [50000, 20000],
            "o": [40000, 18000],
            "tv": [10000, 2000],
            "nazev_mesta": ["Brno", "Kuřim"],
        })
        links = gpd.GeoDataFrame([
            {"link_id": 1, "osm_ref": "D1;43", "link_type": "motorway",
             "geometry": LineString([(0, 0), (1, 0)])},
        ], geometry="geometry", crs="EPSG:5514")
        result = load_csd_as_link_counts(csd, links)
        assert "D1" in result["csd_road"].values

    def test_non_car_links_excluded(self):
        """Footway/cycleway links should not be matched."""
        csd = pd.DataFrame({
            "sil": ["43"],
            "sv": [20000],
            "o": [18000],
            "tv": [2000],
            "nazev_mesta": ["Kuřim"],
        })
        links = gpd.GeoDataFrame([
            {"link_id": 1, "osm_ref": "43", "link_type": "cycleway",
             "geometry": LineString([(0, 0), (1, 0)])},
        ], geometry="geometry", crs="EPSG:5514")
        result = load_csd_as_link_counts(csd, links)
        assert result.empty


# ---------------------------------------------------------------------------
# match_csd_to_links
# ---------------------------------------------------------------------------

class TestMatchCsdToLinks:
    def test_basic_matching(self):
        csd = _make_csd()
        links = _make_links_with_volumes()
        result = match_csd_to_links(csd, links)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "road" in result.columns
        assert "csd_mean_sv" in result.columns
        assert "model_lw_mean" in result.columns

    def test_matched_roads_present(self):
        csd = _make_csd()
        links = _make_links_with_volumes()
        result = match_csd_to_links(csd, links)
        matched = set(result["road"].values)
        assert "D1" in matched
        assert "43" in matched

    def test_no_match_returns_empty(self):
        csd = pd.DataFrame({
            "sil": ["ZZZZ"],
            "sv": [1000],
            "o": [900],
            "tv": [100],
        })
        links = _make_links_with_volumes()
        result = match_csd_to_links(csd, links)
        assert result.empty

    def test_missing_sil_returns_empty(self):
        csd = pd.DataFrame({"foo": [1]})
        links = _make_links_with_volumes()
        result = match_csd_to_links(csd, links)
        assert result.empty

    def test_missing_osm_ref_returns_empty(self):
        csd = _make_csd()
        links = gpd.GeoDataFrame([
            {"link_id": 1, "link_type": "motorway", "distance": 5000,
             "total_vehicles_tot": 45000, "geometry": LineString([(0, 0), (1, 0)])},
        ], geometry="geometry", crs="EPSG:5514")
        result = match_csd_to_links(csd, links)
        assert result.empty

    def test_numeric_sil_handled(self):
        """Integer sil values should not crash."""
        csd = pd.DataFrame({
            "sil": [43, 43],
            "sv": [20000, 19000],
            "o": [18000, 17000],
            "tv": [2000, 2000],
        })
        links = _make_links_with_volumes()
        result = match_csd_to_links(csd, links)
        assert len(result) > 0

    def test_geh_computed(self):
        """GEH column should be present and non-negative."""
        csd = _make_csd()
        links = _make_links_with_volumes()
        result = match_csd_to_links(csd, links)
        if not result.empty:
            assert "geh" in result.columns
            assert (result["geh"] >= 0).all()

    def test_no_volume_columns_returns_empty(self):
        """Links without any *_tot columns should return empty."""
        csd = _make_csd()
        links = gpd.GeoDataFrame([
            {"link_id": 1, "osm_ref": "D1", "link_type": "motorway",
             "geometry": LineString([(0, 0), (1, 0)])},
        ], geometry="geometry", crs="EPSG:5514")
        result = match_csd_to_links(csd, links)
        assert result.empty


# ---------------------------------------------------------------------------
# _classify_csd_road (used internally by the split)
# ---------------------------------------------------------------------------

class TestClassifyCsdRoad:
    @pytest.mark.parametrize("sil,expected", [
        ("D1", "motorway"),
        ("D2", "motorway"),
        ("D52", "motorway"),
        ("43", "trunk"),
        ("52", "trunk"),
        ("150", "secondary"),
        ("602", "secondary"),
        ("3792", "tertiary"),
        ("abc", "other"),
    ])
    def test_classification(self, sil, expected):
        assert _classify_csd_road(sil) == expected

    def test_numeric_input(self):
        """Function should handle integer input gracefully."""
        assert _classify_csd_road(43) == "trunk"
        assert _classify_csd_road(150) == "secondary"
        assert _classify_csd_road(3792) == "tertiary"


# ---------------------------------------------------------------------------
# _CSD_COMPATIBLE_LINK_TYPES consistency
# ---------------------------------------------------------------------------

class TestCsdCompatibleLinkTypes:
    def test_secondary_does_not_include_trunk(self):
        """Secondary road class should not match trunk links."""
        assert "trunk" not in _CSD_COMPATIBLE_LINK_TYPES["secondary"]
        assert "trunk_link" not in _CSD_COMPATIBLE_LINK_TYPES["secondary"]

    def test_each_class_includes_own_type(self):
        for cls in ("motorway", "trunk", "secondary", "tertiary"):
            assert cls in _CSD_COMPATIBLE_LINK_TYPES[cls]


# ---------------------------------------------------------------------------
# normalize_csd_count_columns (class-breakdown XLSX → sv / o / tv)
# ---------------------------------------------------------------------------

class TestNormalizeCsdCountColumns:
    def test_maps_total_s_and_light_l(self):
        from sim.datasets.csd import normalize_csd_count_columns

        df = pd.DataFrame({
            "sil": ["4-0131", "4-0132"],
            "s": [10000, 20000],
            "l": [8000, 15000],
            "t": [1500, 3500],
            "m": [500, 1500],
        })
        out = normalize_csd_count_columns(df)
        assert out["sv"].tolist() == [10000, 20000]
        assert out["o"].tolist() == [8000, 15000]
        # Heavy column sums T (+ PN/TN/A/AL); motorcycles stay in ``sv`` only.
        assert out["tv"].tolist()[0] == pytest.approx(1500.0)
        assert out["tv"].tolist()[1] == pytest.approx(3500.0)

    def test_preserves_official_sv_o_tv(self):
        from sim.datasets.csd import normalize_csd_count_columns

        df = pd.DataFrame({"sil": ["13"], "sv": [12000.0], "o": [9000.0], "tv": [3000.0]})
        out = normalize_csd_count_columns(df)
        assert float(out["sv"].iloc[0]) == 12000.0
        assert float(out["o"].iloc[0]) == 9000.0
        assert float(out["tv"].iloc[0]) == 3000.0
