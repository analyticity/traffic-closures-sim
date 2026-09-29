"""Unit tests for reading the PostgreSQL-backed sources from CSV dumps.

These cover the contract the fetchers rely on: a raw table dump must come out
of :mod:`sim.datasets.csv_source` with the same column names and derived
columns the SQL query would have produced.
"""
from __future__ import annotations

import pandas as pd
import pytest

shapely = pytest.importorskip("shapely")

from sim.datasets.csv_source import (  # noqa: E402
    add_lonlat_from_geog,
    add_wkt_from_geog,
    load_closures_csv,
    load_event_links_csv,
    load_jams_csv,
    load_segments_csv,
    resolve_csv_path,
)

# POINT(16.6068 49.1951) and LINESTRING(16.60 49.19, 16.62 49.21), SRID 4326,
# as PostGIS writes them into a CSV dump (hex EWKB).
_POINT_EWKB = shapely.wkb.dumps(
    shapely.geometry.Point(16.6068, 49.1951), hex=True, srid=4326
)
_LINE_EWKB = shapely.wkb.dumps(
    shapely.geometry.LineString([(16.60, 49.19), (16.62, 49.21)]), hex=True, srid=4326
)


def _write(tmp_path, name: str, rows: list[dict]) -> str:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# --- geometry decoding ---------------------------------------------------

def test_lonlat_from_point_geog():
    df = pd.DataFrame({"g": [_POINT_EWKB]})
    out = add_lonlat_from_geog(df, "g")
    assert out["lon"].iloc[0] == pytest.approx(16.6068, abs=1e-6)
    assert out["lat"].iloc[0] == pytest.approx(49.1951, abs=1e-6)
    assert "g" not in out.columns


def test_lonlat_from_line_geog_uses_centroid():
    """Mirrors ST_Centroid in the SQL: a line collapses to its midpoint."""
    df = pd.DataFrame({"g": [_LINE_EWKB]})
    out = add_lonlat_from_geog(df, "g")
    assert out["lon"].iloc[0] == pytest.approx(16.61, abs=1e-6)
    assert out["lat"].iloc[0] == pytest.approx(49.20, abs=1e-6)


def test_missing_geometry_does_not_raise():
    """A malformed or NULL row yields NaN rather than killing the import."""
    df = pd.DataFrame({"g": [_POINT_EWKB, None, "not-hex"]})
    out = add_lonlat_from_geog(df, "g")
    assert out["lon"].notna().sum() == 1
    assert out["lon"].isna().sum() == 2


def test_wkt_from_geog():
    df = pd.DataFrame({"g": [_LINE_EWKB, None]})
    out = add_wkt_from_geog(df, "g", wkt_col="line_wkt")
    assert out["line_wkt"].iloc[0].startswith("LINESTRING")
    # Rows without a line stay missing; closures_fetch gates on pd.notna, so
    # None and NaN are equivalent here.
    assert pd.isna(out["line_wkt"].iloc[1])


# --- per-source loaders ---------------------------------------------------

def test_load_segments_csv(tmp_path):
    path = _write(tmp_path, "road_segments.csv", [
        {"id": 1, "osm_id": 111, "geog": _LINE_EWKB, "name": "Jihlavská",
         "road_ref": "602", "road_class": "secondary", "city": "Brno", "max_speed": 50},
    ])
    df = load_segments_csv(path)
    assert set(df.columns) >= {"id", "osm_id", "name", "road_ref", "road_class",
                               "city", "max_speed", "lat", "lon"}
    assert df["osm_id"].iloc[0] == 111


def test_load_jams_csv_joins_segments(tmp_path):
    jams = _write(tmp_path, "jams.csv", [
        {"id": 10, "first_seen": "2026-06-01 07:00:00+00",
         "last_seen": "2026-06-01 07:30:00+00", "city": "Brno",
         "street_name": "Jihlavská", "road_number": "602", "road_type_code": "secondary",
         "delay_seconds": 120, "length_m": 400, "speed_kmh": 8,
         "speed_normal_kmh": 50, "severity": "heavy", "quality_score": 80,
         "segment_id": 1, "jam_line_geog": _LINE_EWKB, "raw": "{}"},
    ])
    segs = _write(tmp_path, "road_segments.csv", [
        {"id": 1, "osm_id": 111, "geog": _LINE_EWKB, "name": "Jihlavská",
         "road_ref": "602", "road_class": "secondary", "city": "Brno", "max_speed": 50},
    ])

    df = load_jams_csv(jams, segments_csv_path=segs)
    assert df["osm_id"].iloc[0] == 111
    assert df["segment_road_class"].iloc[0] == "secondary"
    assert df["segment_max_speed"].iloc[0] == 50
    # `raw` is the bulk of the real dump and nothing consumes it.
    assert "raw" not in df.columns


def test_load_jams_csv_drops_rows_without_geometry(tmp_path):
    """The SQL has WHERE jam_line_geog IS NOT NULL; the CSV path must match."""
    path = _write(tmp_path, "jams.csv", [
        {"id": 1, "segment_id": 1, "jam_line_geog": _LINE_EWKB},
        {"id": 2, "segment_id": 2, "jam_line_geog": None},
    ])
    df = load_jams_csv(path)
    assert list(df["id"]) == [1]


def test_load_jams_csv_without_segments_still_works(tmp_path):
    path = _write(tmp_path, "jams.csv", [
        {"id": 1, "segment_id": 5, "jam_line_geog": _POINT_EWKB},
    ])
    df = load_jams_csv(path)
    assert "osm_id" in df.columns
    assert df["osm_id"].isna().all()


def test_load_closures_csv_applies_sql_aliases(tmp_path):
    path = _write(tmp_path, "restrictions.csv", [
        {"id": 94462, "external_ids": '{"ndic": "x"}', "restriction_type": "roadworks",
         "restriction_subtype": None, "severity": "serious", "status": "expired",
         "road_number": "42", "street_name": "VMO Dobrovského", "city": "Brno",
         "description_cs": "uzavírka", "max_speed_kmh": None,
         "valid_from": "2026-06-30 22:00:00+00", "valid_to": "2026-07-13 21:59:00+00",
         "first_seen": "2026-06-30 22:00:00+00", "last_seen": "2026-07-13 21:59:00+00",
         "direction": "both", "km_from": None, "km_to": None,
         "road_type_code": "trunk", "quality_score": 100, "urgency": None,
         "probability": None, "segment_id": 20112,
         "location_point_geog": _POINT_EWKB, "location_line_geog": _LINE_EWKB},
    ])
    df = load_closures_csv(path)
    # SQL aliases the fetcher's post-processing depends on
    assert "pg_severity" in df.columns and "severity" not in df.columns
    assert "closure_direction" in df.columns and "direction" not in df.columns
    assert df["line_wkt"].iloc[0].startswith("LINESTRING")
    # kept on purpose: the only way to tell NDIC roadworks from Waze alerts
    assert "ndic" in df["external_ids"].iloc[0]


def test_load_closures_csv_requires_point_geometry(tmp_path):
    path = _write(tmp_path, "restrictions.csv", [
        {"id": 1, "location_point_geog": _POINT_EWKB, "location_line_geog": None},
        {"id": 2, "location_point_geog": None, "location_line_geog": None},
    ])
    df = load_closures_csv(path)
    assert list(df["id"]) == [1]


def test_load_event_links_csv_filters_and_joins(tmp_path):
    links = _write(tmp_path, "event_links.csv", [
        {"id": 1, "source_type": "restriction", "source_id": 94462,
         "target_type": "traffic_jam", "target_id": 10, "link_type": "caused_by",
         "confidence": 70, "description": ""},
        # dropped: not a causal link
        {"id": 2, "source_type": "alert", "source_id": 5, "target_type": "alert",
         "target_id": 6, "link_type": "related_to", "confidence": 40, "description": ""},
    ])
    jams = _write(tmp_path, "jams.csv", [
        {"id": 10, "first_seen": "2026-07-01 07:00:00+00",
         "last_seen": "2026-07-01 08:00:00+00", "delay_seconds": 300,
         "length_m": 800, "speed_kmh": 6, "speed_normal_kmh": 50,
         "severity": "heavy", "segment_id": 20112, "city": "Brno",
         "road_number": "42"},
    ])

    df = load_event_links_csv(links, jams_csv_path=jams)
    assert len(df) == 1
    assert "link_id" in df.columns  # SQL aliases el.id -> link_id
    assert df["jam_delay_s"].iloc[0] == 300
    assert df["jam_segment_id"].iloc[0] == 20112


# --- path resolution ------------------------------------------------------

def test_resolve_csv_path_prefers_explicit(tmp_path):
    cfg = {"datasets": {"csv_dir": str(tmp_path)}}
    assert resolve_csv_path(cfg, {"csv_path": "/x/y.csv"}, "jams.csv") == "/x/y.csv"


def test_resolve_csv_path_falls_back_to_dir(tmp_path):
    (tmp_path / "jams.csv").write_text("id\n1\n")
    cfg = {"datasets": {"csv_dir": str(tmp_path)}}
    assert resolve_csv_path(cfg, {}, "jams.csv") == str(tmp_path / "jams.csv")


def test_resolve_csv_path_none_keeps_postgres_branch(tmp_path):
    """No csv_dir and no missing file => None, so the DB branch stays in charge."""
    assert resolve_csv_path({"datasets": {}}, {}, "jams.csv") is None
    cfg = {"datasets": {"csv_dir": str(tmp_path)}}
    assert resolve_csv_path(cfg, {}, "not_there.csv") is None
