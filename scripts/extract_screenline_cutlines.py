"""One-time helper: extract WKT cut-line geometries for existing screenlines.

Reads the current screenlines.yaml (with hardcoded link_ids), loads the
network links, and for each screenline generates a short LINESTRING that
crosses all the referenced links perpendicularly.  Outputs new YAML to stdout.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import transform as shapely_transform
import pyproj

METRIC_EPSG = 5514
WGS84 = 4326
CUT_LENGTH_M = 150  # half-length of cut-line on each side of centroid


def _link_bearing(geom: LineString) -> float:
    """Compute average bearing (degrees) of a linestring."""
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return math.degrees(math.atan2(dx, dy)) % 360


def _perpendicular_cut(centroid: Point, bearing_deg: float, half_len: float) -> LineString:
    """Build a line perpendicular to *bearing_deg*, centered on *centroid*."""
    perp = math.radians(bearing_deg + 90)
    dx = half_len * math.sin(perp)
    dy = half_len * math.cos(perp)
    return LineString([
        (centroid.x - dx, centroid.y - dy),
        (centroid.x + dx, centroid.y + dy),
    ])


def main() -> None:
    links_path = Path("outputs/baseline/network/network_links.geojson")
    sl_path = Path("config/screenlines.yaml")

    if not links_path.exists():
        sys.exit(f"Network links not found: {links_path}")
    if not sl_path.exists():
        sys.exit(f"Screenlines YAML not found: {sl_path}")

    links_gdf = gpd.read_file(links_path)
    if links_gdf.crs is None:
        links_gdf = links_gdf.set_crs(epsg=WGS84)

    links_metric = links_gdf.to_crs(epsg=METRIC_EPSG) if links_gdf.crs.to_epsg() != METRIC_EPSG else links_gdf

    to_wgs = pyproj.Transformer.from_crs(f"EPSG:{METRIC_EPSG}", f"EPSG:{WGS84}", always_xy=True)

    with open(sl_path) as f:
        raw = yaml.safe_load(f)

    new_screenlines = []
    for sl in raw.get("screenlines", []):
        name = sl["name"]
        link_ids = [int(lk["link_id"]) for lk in sl.get("links", []) if int(lk.get("link_id", 0)) > 0]
        if not link_ids:
            print(f"# SKIP {name}: no link_ids", file=sys.stderr)
            continue

        matched = links_metric[links_metric["link_id"].isin(link_ids)]
        if matched.empty:
            print(f"# SKIP {name}: link_ids {link_ids} not in network", file=sys.stderr)
            continue

        geoms = [g for g in matched.geometry if g is not None]
        combined = MultiLineString(geoms) if len(geoms) > 1 else geoms[0]
        centroid = combined.centroid

        bearings = [_link_bearing(g) for g in geoms]
        avg_bearing = np.mean(bearings)

        cut_metric = _perpendicular_cut(centroid, avg_bearing, CUT_LENGTH_M)
        cut_wgs = shapely_transform(to_wgs.transform, cut_metric)

        coords_str = ", ".join(
            f"{c[0]:.6f} {c[1]:.6f}" for c in cut_wgs.coords
        )
        wkt = f"LINESTRING({coords_str})"

        highway_vals = set()
        name_vals = set()
        ref_vals = set()
        for _, row in matched.iterrows():
            if "osm_highway" in row and row["osm_highway"] and str(row["osm_highway"]) != "nan":
                highway_vals.add(str(row["osm_highway"]).strip())
            if "name" in row and row["name"] and str(row["name"]) != "nan":
                name_vals.add(str(row["name"]).strip())
            if "osm_ref_norm" in row and row["osm_ref_norm"] and str(row["osm_ref_norm"]) != "nan":
                ref_vals.add(str(row["osm_ref_norm"]).strip())

        attr_filter = {}
        if highway_vals:
            attr_filter["osm_highway"] = ",".join(sorted(highway_vals))
        if ref_vals:
            attr_filter["osm_ref_norm"] = ",".join(sorted(ref_vals))
        elif name_vals:
            attr_filter["name"] = ",".join(sorted(name_vals))

        entry: dict = {
            "name": name,
            "description": sl.get("description", ""),
            "type": sl.get("type", "radial"),
        }
        if sl.get("observed_aadt_cars"):
            entry["observed_aadt_cars"] = sl["observed_aadt_cars"]
        if sl.get("observed_aadt_all"):
            entry["observed_aadt_all"] = sl["observed_aadt_all"]
        entry["geometry_wkt"] = wkt
        if attr_filter:
            entry["filter"] = attr_filter

        entry["_old_link_ids"] = link_ids

        new_screenlines.append(entry)
        print(
            f"# {name}: {len(geoms)} link(s), bearing={avg_bearing:.0f}°, "
            f"filter={attr_filter}, link_ids={link_ids}",
            file=sys.stderr,
        )

    output = {"screenlines": new_screenlines}
    print(yaml.dump(output, default_flow_style=False, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
