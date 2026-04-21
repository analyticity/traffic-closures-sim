"""Unit tests for deterministic helpers in sim.network_pipeline.

These tests cover pure-function logic that does not require AequilibraE,
OSM downloads, or any network access.
"""
from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from sim.network_pipeline import (
    _aggregate_osm_edge_attributes,
    _buffer_polygon_km,
    _choose_best,
    _clean_text,
    _compute_urban_trim_bbox,
    _extract_osm_ids,
    _guess_crs_from_coords,
    _listify,
    _normalize_name,
    _normalize_ref,
)


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------
class TestCleanText:
    def test_none(self):
        assert _clean_text(None) is None

    def test_nan(self):
        assert _clean_text(float("nan")) is None

    def test_empty_string(self):
        assert _clean_text("") is None
        assert _clean_text("   ") is None

    def test_normal(self):
        assert _clean_text("  hello ") == "hello"

    def test_number(self):
        assert _clean_text(42) == "42"


# ---------------------------------------------------------------------------
# _normalize_name
# ---------------------------------------------------------------------------
class TestNormalizeName:
    def test_empty_inputs(self):
        assert _normalize_name(None) == ""
        assert _normalize_name("") == ""
        assert _normalize_name(float("nan")) == ""

    def test_strips_diacritics_and_uppercases(self):
        assert _normalize_name("Brněnská") == "BRNENSKA"

    def test_removes_spaces_hyphens_slashes(self):
        assert _normalize_name("Praha - Brno / CZ") == "PRAHABRNOCZ"

    def test_number_input(self):
        assert _normalize_name(123) == "123"


# ---------------------------------------------------------------------------
# _normalize_ref
# ---------------------------------------------------------------------------
class TestNormalizeRef:
    def test_basic(self):
        assert _normalize_ref("I/43") == "I43"

    def test_spaces_and_dashes(self):
        assert _normalize_ref(" D - 1 ") == "D1"

    def test_backslash_normalised(self):
        assert _normalize_ref("I\\43") == "I43"

    def test_none(self):
        assert _normalize_ref(None) == ""


# ---------------------------------------------------------------------------
# _listify
# ---------------------------------------------------------------------------
class TestListify:
    def test_none(self):
        assert _listify(None) == []

    def test_nan(self):
        assert _listify(float("nan")) == []

    def test_scalar(self):
        assert _listify(42) == [42]
        assert _listify("abc") == ["abc"]

    def test_list(self):
        assert _listify([1, 2, 3]) == [1, 2, 3]

    def test_nested(self):
        assert _listify([[1, 2], 3]) == [1, 2, 3]

    def test_set(self):
        result = _listify({10})
        assert result == [10]


# ---------------------------------------------------------------------------
# _extract_osm_ids
# ---------------------------------------------------------------------------
class TestExtractOsmIds:
    def test_none(self):
        assert _extract_osm_ids(None) == []

    def test_nan(self):
        assert _extract_osm_ids(float("nan")) == []

    def test_int(self):
        assert _extract_osm_ids(12345) == [12345]

    def test_float(self):
        assert _extract_osm_ids(12345.0) == [12345]

    def test_string_single(self):
        assert _extract_osm_ids("98765") == [98765]

    def test_string_list_format(self):
        result = _extract_osm_ids("[100, 200, 300]")
        assert result == [100, 200, 300]

    def test_tuple(self):
        assert _extract_osm_ids((5, 6)) == [5, 6]

    def test_deduplication(self):
        result = _extract_osm_ids([10, 10, 20])
        assert result == [10, 20]


# ---------------------------------------------------------------------------
# _choose_best
# ---------------------------------------------------------------------------
class TestChooseBest:
    def test_empty(self):
        assert _choose_best(Counter()) is None

    def test_single(self):
        assert _choose_best(Counter({"D1": 3})) == "D1"

    def test_picks_most_common(self):
        c = Counter({"D1": 5, "D2": 2, "I/43": 1})
        assert _choose_best(c) == "D1"


# ---------------------------------------------------------------------------
# _aggregate_osm_edge_attributes
# ---------------------------------------------------------------------------
class TestAggregateOsmEdgeAttributes:
    @staticmethod
    def _make_edges(rows: list[dict]) -> gpd.GeoDataFrame:
        df = pd.DataFrame(rows)
        if "geometry" not in df.columns:
            df["geometry"] = [
                LineString([(16.6, 49.2), (16.61, 49.21)]) for _ in range(len(df))
            ]
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    def test_simple_aggregation(self):
        edges = self._make_edges([
            {"osmid": 100, "ref": "D1", "name": "Dálnice D1", "highway": "motorway"},
            {"osmid": 100, "ref": "D1", "name": "Dálnice D1", "highway": "motorway"},
            {"osmid": 200, "ref": "I/43", "name": "Svitavská", "highway": "primary"},
        ])
        result = _aggregate_osm_edge_attributes(edges)

        assert 100 in result
        assert result[100]["osm_ref"] == "D1"
        assert result[100]["osm_name_raw"] == "Dálnice D1"
        assert result[100]["osm_highway"] == "motorway"

        assert 200 in result
        assert result[200]["osm_ref"] == "I/43"
        assert result[200]["osm_ref_norm"] == "I43"

    def test_missing_ref(self):
        edges = self._make_edges([
            {"osmid": 300, "ref": None, "name": "Ulice", "highway": "residential"},
        ])
        result = _aggregate_osm_edge_attributes(edges)
        assert 300 in result
        assert result[300]["osm_ref"] is None
        assert result[300]["osm_ref_norm"] is None

    def test_list_osmid(self):
        edges = self._make_edges([
            {"osmid": [400, 401], "ref": "D2", "name": "D2", "highway": "motorway"},
        ])
        result = _aggregate_osm_edge_attributes(edges)
        assert 400 in result
        assert 401 in result
        assert result[400]["osm_ref"] == "D2"


# ---------------------------------------------------------------------------
# _guess_crs_from_coords
# ---------------------------------------------------------------------------
class TestGuessCrsFromCoords:
    def test_wgs84(self):
        pts = gpd.GeoSeries([Point(16.6, 49.2), Point(16.7, 49.3)])
        assert _guess_crs_from_coords(pts) == "EPSG:4326"

    def test_metric_fallback(self):
        pts = gpd.GeoSeries([Point(-598000, -1160000), Point(-597000, -1159000)])
        assert _guess_crs_from_coords(pts) == "EPSG:5514"

    def test_custom_fallback(self):
        pts = gpd.GeoSeries([Point(-598000, -1160000)])
        assert _guess_crs_from_coords(pts, fallback_epsg=32633) == "EPSG:32633"

    def test_empty(self):
        pts = gpd.GeoSeries([], dtype="geometry")
        assert _guess_crs_from_coords(pts) == "EPSG:5514"


# ---------------------------------------------------------------------------
# _buffer_polygon_km
# ---------------------------------------------------------------------------
class TestBufferPolygonKm:
    def test_buffer_enlarges(self):
        poly = box(16.5, 49.1, 16.7, 49.3)
        buffered = _buffer_polygon_km(poly, buffer_km=5, crs_epsg=5514)
        orig_bounds = poly.bounds
        buf_bounds = buffered.bounds
        assert buf_bounds[0] < orig_bounds[0]
        assert buf_bounds[1] < orig_bounds[1]
        assert buf_bounds[2] > orig_bounds[2]
        assert buf_bounds[3] > orig_bounds[3]

    def test_zero_buffer_roughly_same(self):
        poly = box(16.5, 49.1, 16.7, 49.3)
        buffered = _buffer_polygon_km(poly, buffer_km=0, crs_epsg=5514)
        for orig, buf in zip(poly.bounds, buffered.bounds):
            assert abs(orig - buf) < 0.01


# ---------------------------------------------------------------------------
# _compute_urban_trim_bbox
# ---------------------------------------------------------------------------
class TestComputeUrbanTrimBbox:
    @staticmethod
    def _fake_project(link_rows: list[dict], node_rows: list[dict]):
        """Build a mock object that quacks like aequilibrae.Project for bbox calc."""
        links_df = pd.DataFrame(link_rows)
        if "geometry" not in links_df.columns and link_rows:
            links_df["geometry"] = [
                LineString([(r["ax"], r["ay"]), (r["bx"], r["by"])])
                for r in link_rows
            ]
        nodes_df = pd.DataFrame(node_rows)
        if "geometry" not in nodes_df.columns and node_rows:
            nodes_df["geometry"] = [Point(r["x"], r["y"]) for r in node_rows]

        links_gdf = gpd.GeoDataFrame(links_df, geometry="geometry") if not links_df.empty else links_df
        nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry="geometry") if not nodes_df.empty else nodes_df

        return SimpleNamespace(
            network=SimpleNamespace(
                links=SimpleNamespace(data=links_gdf),
                nodes=SimpleNamespace(data=nodes_gdf),
            )
        )

    def test_residential_nodes_form_bbox(self):
        nodes = [
            {"node_id": i, "x": 10.0 + i * 0.1, "y": 50.0 + i * 0.1}
            for i in range(100)
        ]
        links = [
            {
                "link_id": i,
                "a_node": i,
                "b_node": i + 1,
                "link_type": "residential",
                "ax": 10.0 + i * 0.1,
                "ay": 50.0 + i * 0.1,
                "bx": 10.0 + (i + 1) * 0.1,
                "by": 50.0 + (i + 1) * 0.1,
            }
            for i in range(99)
        ]
        proj = self._fake_project(links, nodes)
        result = _compute_urban_trim_bbox(proj, pad_ratio=0.0, quantile=0.0)

        assert len(result) == 4
        minx, miny, maxx, maxy = result
        assert minx < maxx
        assert miny < maxy
        assert minx == pytest.approx(10.0, abs=0.5)

    def test_corridor_links_excluded(self):
        """Motorway-only nodes should be excluded from the urban core calc
        when enough non-corridor nodes exist."""
        nodes = []
        links = []
        for i in range(80):
            nodes.append({"node_id": i, "x": 10.0 + i * 0.01, "y": 50.0 + i * 0.01})
            if i > 0:
                links.append({
                    "link_id": i,
                    "a_node": i - 1,
                    "b_node": i,
                    "link_type": "residential",
                    "ax": 10.0 + (i - 1) * 0.01,
                    "ay": 50.0 + (i - 1) * 0.01,
                    "bx": 10.0 + i * 0.01,
                    "by": 50.0 + i * 0.01,
                })
        outlier_id = 200
        nodes.append({"node_id": outlier_id, "x": 20.0, "y": 60.0})
        nodes.append({"node_id": outlier_id + 1, "x": 20.1, "y": 60.1})
        links.append({
            "link_id": 999,
            "a_node": outlier_id,
            "b_node": outlier_id + 1,
            "link_type": "motorway",
            "ax": 20.0, "ay": 60.0,
            "bx": 20.1, "by": 60.1,
        })

        proj = self._fake_project(links, nodes)
        minx, miny, maxx, maxy = _compute_urban_trim_bbox(proj, pad_ratio=0.0, quantile=0.0)
        assert maxx < 15.0, "Motorway outlier should be excluded from urban trim bbox"
