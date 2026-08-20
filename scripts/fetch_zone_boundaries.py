#!/usr/bin/env python3
"""Fetch administrative boundaries from Overpass into a GeoJSON zone source.

Why this exists: ``osmnx.features.features_from_place`` silently drops some
boundary relations.  For Brno it returns 26 of the 29 city districts — missing
Brno-střed, Brno-Komín and Brno-Chrlice — with no warning and no difference in
their OSM tagging (all are ``type=boundary``, ``place=borough``,
``admin_level=9``).  Verified on osmnx 2.1.1 against three query variants
(okres place, city place, 20x ``max_query_area_size``): all three return the
same 26.  Brno-střed is the largest trip attractor in the model, so this is not
a cosmetic loss — it leaves a hole in the middle of the zone system.

Overpass itself returns all 29, so this script asks it directly and assembles
the relation members into polygons instead of relying on osmnx.

The result is a plain GeoJSON that ``zoning.sources`` can consume as
``type: file``, which also takes Overpass off the critical path of
``build-zones``.

    python scripts/fetch_zone_boundaries.py \\
        --bbox 49.10 16.42 49.31 16.75 \\
        --admin-level 9 --name-prefix "Brno-" \\
        --out config/brno/zones_admin9.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Rovnaký prepínač ako zvyšok pipeline (sim/overpass.py): OVERPASS_ENDPOINT.
ENDPOINT = os.environ.get("OVERPASS_ENDPOINT", "https://overpass-api.de/api").rstrip("/") \
    + "/interpreter"

Point = Tuple[float, float]


def overpass(query: str, retries: int = 3, timeout: int = 180) -> Dict[str, Any]:
    """POST a query, retrying — overpass-api.de round-robins onto a dead host.

    Uses ``requests`` (bundles its own CA store) rather than urllib, which fails
    certificate verification on a stock python.org install.
    """
    import requests

    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            # Bez vlastného User-Agent vracia overpass-api.de 406.
            r = requests.post(ENDPOINT, data={"data": query}, timeout=timeout,
                              headers={"User-Agent": "traffic-sim-backend/1.0 (zone boundaries)"})
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — retry on anything transient
            last = e
            print(f"  pokus {attempt}/{retries} zlyhal: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise SystemExit(f"Overpass nedostupný po {retries} pokusoch: {last}")


def assemble_rings(members: List[Dict[str, Any]]) -> List[List[Point]]:
    """Stitch relation member ways into closed rings by matching endpoints."""
    segs: List[List[Point]] = [
        [(p["lon"], p["lat"]) for p in m["geometry"]]
        for m in members
        if m.get("type") == "way" and m.get("geometry") and m.get("role") in ("outer", "", None)
    ]
    rings: List[List[Point]] = []
    while segs:
        cur = segs.pop(0)
        moved = True
        while moved and cur[0] != cur[-1]:
            moved = False
            for i, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur = cur + s[1:]
                elif s[-1] == cur[-1]:
                    cur = cur + s[::-1][1:]
                elif s[-1] == cur[0]:
                    cur = s[:-1] + cur
                elif s[0] == cur[0]:
                    cur = s[::-1][:-1] + cur
                else:
                    continue
                segs.pop(i)
                moved = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
    return rings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("S", "W", "N", "E"), help="south west north east (WGS84)")
    ap.add_argument("--admin-level", default="9")
    ap.add_argument("--name-prefix", default="", help="keep only names starting with this")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--precision", type=int, default=6, help="coordinate decimals")
    args = ap.parse_args()

    s, w, n, e = args.bbox
    q = (f'[out:json][timeout:180];'
         f'rel["boundary"="administrative"]["admin_level"="{args.admin_level}"]'
         f'({s},{w},{n},{e});out geom;')
    print(f"Overpass: admin_level={args.admin_level} v bbox {s},{w},{n},{e}")
    data = overpass(q)

    features = []
    preskocene = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        meno = tags.get("name", "")
        if args.name_prefix and not meno.startswith(args.name_prefix):
            continue
        rings = assemble_rings(el.get("members", []))
        if not rings:
            preskocene.append(meno or el.get("id"))
            continue
        rings.sort(key=len, reverse=True)
        r = args.precision
        coords = [[[round(x, r), round(y, r)] for x, y in ring] for ring in rings]
        geom = ({"type": "Polygon", "coordinates": [coords[0]]} if len(coords) == 1
                else {"type": "MultiPolygon", "coordinates": [[c] for c in coords]})
        features.append({
            "type": "Feature",
            "properties": {"zone_id": int(el["id"]), "name": meno,
                           "admin_level": tags.get("admin_level"),
                           "source": f"OSM relation {el['id']}"},
            "geometry": geom,
        })

    if not features:
        raise SystemExit("Overpass nevrátil žiadnu hranicu — skontroluj bbox a admin_level.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8")

    print(f"  zapísaných {len(features)} hraníc → {args.out} "
          f"({args.out.stat().st_size // 1024} kB)")
    if preskocene:
        print(f"  ! bez použiteľnej geometrie: {preskocene}")
    for f in sorted(features, key=lambda f: f["properties"]["name"])[:40]:
        print(f"     {f['properties']['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
