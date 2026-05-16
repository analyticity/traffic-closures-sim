"""CSD anchor must match link osm_ref when require_csd_road_ref_match is enabled."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from sim.calibration.matching import match_counts_to_links


def test_csd_road_ref_blocks_cross_ref_match():
    counts = gpd.GeoDataFrame([
        {
            "objectid": 9001,
            "csd_road": "430",
            "observed_car": 8619.0,
            "observed_motor_total": 10110.0,
            "observed_total": 10110.0,
            "geometry": Point(0.05, 0),
        },
    ], geometry="geometry", crs="EPSG:5514")
    links = gpd.GeoDataFrame([
        {
            "link_id": 79141,
            "osm_ref": "50",
            "link_type": "trunk",
            "PCE_tot": 8350.0,
            "geometry": LineString([(0, 0), (0.1, 0)]),
        },
        {
            "link_id": 18045,
            "osm_ref": "430",
            "link_type": "primary",
            "PCE_tot": 5000.0,
            "geometry": LineString([(0.04, 0), (0.06, 0)]),
        },
    ], geometry="geometry", crs="EPSG:5514")

    loose = match_counts_to_links(
        counts, links, buffer_m=50, vol_col="PCE_tot", aggregate_corridor=False,
        require_csd_road_ref_match=False,
    )
    assert int(loose.iloc[0]["link_id"]) == 79141

    strict = match_counts_to_links(
        counts, links, buffer_m=50, vol_col="PCE_tot", aggregate_corridor=False,
        require_csd_road_ref_match=True,
    )
    assert int(strict.iloc[0]["link_id"]) == 18045
