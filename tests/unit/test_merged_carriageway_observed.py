"""Tests for divided-highway screenline merge and observed aggregation."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from sim.calibration.validation import (
    _observed_total_for_merged_carriageway,
    merge_divided_highway_screenline_per_link,
)


def test_observed_split_halves_with_concentrated_modeled() -> None:
    """Most I/13 style: each link ~half AADT; assignment stacks on one arc."""
    modeled = [12950.0, 0.0]
    observed = [6486.0, 6486.0]
    assert _observed_total_for_merged_carriageway(modeled, observed) == 12972.0


def test_observed_duplicate_full_on_each_link() -> None:
    """Matcher duplicated full corridor AADT onto each parallel link."""
    modeled = [12950.0, 0.0]
    observed = [12971.0, 12971.0]
    assert _observed_total_for_merged_carriageway(modeled, observed) == 12971.0


def test_observed_balanced_modeled_balanced_obs() -> None:
    modeled = [5500.0, 5500.0]
    observed = [5509.0, 5509.0]
    assert _observed_total_for_merged_carriageway(modeled, observed) == 11018.0


def test_merge_bidirectional_trunk_same_osm_id() -> None:
    """Twin carriageways digitized as two bidirectional links (direction=0)."""
    per_link = [
        {"link_id": 2768, "direction": 0, "modeled": 6000.0, "observed": 6486.0},
        {"link_id": 2769, "direction": 0, "modeled": 6500.0, "observed": 6485.0},
    ]
    links = pd.DataFrame(
        {
            "link_id": [2768, 2769],
            "osm_id": [139510607, 139510607],
            "link_type": ["trunk", "trunk"],
            "direction": [0, 0],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 1), (1, 1)]),
            ],
        }
    )
    gdf = gpd.GeoDataFrame(links, geometry="geometry", crs="EPSG:4326")
    out = merge_divided_highway_screenline_per_link(per_link, gdf)
    assert len(out) == 1
    assert out[0]["modeled"] == 12500.0
    assert out[0]["observed"] == 12971.0
    assert out[0].get("divided_highway_merged") is True


def test_merge_one_way_trunk_same_osm_id() -> None:
    """Two one-way carriageways sharing an OSM way id (split graph links)."""
    per_link = [
        {"link_id": 101, "direction": 1, "modeled": 3400.0, "observed": 5509.0},
        {"link_id": 102, "direction": 1, "modeled": 8200.0, "observed": 5509.0},
    ]
    links = pd.DataFrame(
        {
            "link_id": [101, 102],
            "osm_id": [9001, 9001],
            "link_type": ["trunk", "trunk"],
            "direction": [1, 1],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 1), (1, 1)]),
            ],
        }
    )
    gdf = gpd.GeoDataFrame(links, geometry="geometry", crs="EPSG:4326")
    out = merge_divided_highway_screenline_per_link(per_link, gdf)
    assert len(out) == 1
    assert out[0]["modeled"] == 11600.0
    assert out[0]["observed"] == 11018.0


def test_merge_skips_different_osm_ids() -> None:
    per_link = [
        {"link_id": 1, "direction": 1, "modeled": 100.0, "observed": 50.0},
        {"link_id": 2, "direction": 1, "modeled": 200.0, "observed": 60.0},
    ]
    links = pd.DataFrame(
        {
            "link_id": [1, 2],
            "osm_id": [111, 222],
            "link_type": ["trunk", "trunk"],
            "direction": [1, 1],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(2, 0), (3, 0)]),
            ],
        }
    )
    gdf = gpd.GeoDataFrame(links, geometry="geometry", crs="EPSG:4326")
    out = merge_divided_highway_screenline_per_link(per_link, gdf)
    assert len(out) == 2
