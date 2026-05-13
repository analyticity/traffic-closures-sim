#!/usr/bin/env python3
"""Export count points and matched link geometries for worst-GEH Pentlogram rows.

Reads ``matching_diagnostics.csv`` and the AequilibraE ``links`` layer so you
can open the GeoJSON pair in QGIS and visually check ref, bearing, and
side-of-road vs. the count location (plan: Olomouc GEH diagnostics).

Example::

    python scripts/export_olomouc_matching_geojson.py \\
        --csv outputs/olomouc/baseline/demand/matching_diagnostics.csv \\
        --db project/olomouc_aeq/project_database.sqlite \\
        --out-dir outputs/olomouc/baseline/demand/map_export \\
        --top 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["GEH"] = pd.to_numeric(df["GEH"], errors="coerce")
    top = df.nlargest(args.top, "GEH").copy()

    lng = pd.to_numeric(top.get("_count_lng"), errors="coerce")
    lat = pd.to_numeric(top.get("_count_lat"), errors="coerce")
    top["geometry"] = [Point(xy) for xy in zip(lng, lat)]
    counts = gpd.GeoDataFrame(
        top[
            [
                c
                for c in (
                    "objectid",
                    "link_id",
                    "osm_ref",
                    "name",
                    "link_type",
                    "GEH",
                    "_dist",
                    "_bearing_diff",
                    "_corridor_volume",
                    "_corridor_n_links",
                    "_excluded",
                    "_match_quality",
                    "observed_car",
                )
                if c in top.columns
            ]
        ],
        geometry=top["geometry"],
        crs="EPSG:4326",
    )

    lids = [int(x) for x in top["link_id"].dropna().unique()]
    if not lids:
        raise SystemExit("No link_id values in top rows.")
    in_list = ",".join(str(i) for i in lids)
    links = gpd.read_file(
        args.db,
        sql=(
            "SELECT link_id, osm_ref, name, link_type, direction, "
            "capacity_ab, capacity_ba, geometry FROM links "
            f"WHERE link_id IN ({in_list})"
        ),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = args.out_dir / "matching_counts_top_geh.geojson"
    links_path = args.out_dir / "matching_links_top_geh.geojson"
    counts.to_file(counts_path, driver="GeoJSON")
    links.to_file(links_path, driver="GeoJSON")
    print(f"Wrote {counts_path}")
    print(f"Wrote {links_path}")


if __name__ == "__main__":
    main()
