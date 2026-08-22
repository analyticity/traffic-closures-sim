#!/usr/bin/env python3
"""Count workplaces per zone from OSM, for splitting a city's employment.

``datasets/employment.py`` derives employment from SLDB commuting destinations,
which are municipalities, so a city that is subdivided into districts has all of
its jobs filed under one name.  Spreading them over the districts needs a key,
and the fallback key — resident population — puts jobs where people sleep: Brno
centre ends up with 17.8 % of the city's jobs, well under its real share.

This script builds a better key by counting workplace features per zone:
``office=*``, ``shop=*`` and the amenities that employ people.  The output plugs
into ``datasets.employment.city_split_weights``.

    python scripts/zone_workplace_weights.py \\
        --zones config/brno/zones_admin9.geojson \\
        --out config/brno/zone_employment_weights.json

Known bias, worth stating wherever the numbers are used: a POI count treats one
factory the same as one hairdresser, so industrial districts come out low and
retail-dense ones high.  The script therefore also reports industrial land area
per zone, as a sanity check against the weights it produces — if the two
disagree wildly for a zone, do not trust the weight there.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ENDPOINT = os.environ.get("OVERPASS_ENDPOINT", "https://overpass-api.de/api").rstrip("/") \
    + "/interpreter"

# One query per group — a single combined query kept timing out on the server.
GROUPS: Dict[str, str] = {
    "office": 'nwr["office"]',
    "shop": 'nwr["shop"]',
    "amenity": 'nwr["amenity"~"^(university|college|school|kindergarten|hospital|clinic|'
               'doctors|townhall|courthouse|police|fire_station|library|theatre|research_institute|'
               'bank|post_office|restaurant|cafe|bar|fast_food|pharmacy)$"]',
}
LANDUSE_QUERY = 'way["landuse"~"^(industrial|commercial|retail)$"]'


def overpass(query: str, retries: int = 3, timeout: int = 180) -> Dict[str, Any]:
    import requests

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(ENDPOINT, data={"data": query}, timeout=timeout,
                              headers={"User-Agent": "traffic-sim-backend/1.0 (zone weights)"})
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  pokus {attempt}/{retries} zlyhal: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise SystemExit(f"Overpass nedostupný: {last}")


def point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def load_zones(path: Path) -> List[Tuple[int, str, List[List[List[float]]], Tuple[float, ...]]]:
    gj = json.loads(path.read_text(encoding="utf-8"))
    zones = []
    for f in gj["features"]:
        g = f.get("geometry") or {}
        rings: List[List[List[float]]] = []
        if g.get("type") == "Polygon":
            rings = [g["coordinates"][0]]
        elif g.get("type") == "MultiPolygon":
            rings = [part[0] for part in g["coordinates"]]
        if not rings:
            continue
        xs = [p[0] for r in rings for p in r]
        ys = [p[1] for r in rings for p in r]
        props = f.get("properties") or {}
        zones.append((int(props["zone_id"]), str(props.get("name", "")), rings,
                      (min(xs), min(ys), max(xs), max(ys))))
    return zones


def assign(lon: float, lat: float, zones) -> int | None:
    for zid, _name, rings, (x0, y0, x1, y1) in zones:
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue  # bbox reject first — this runs per POI
        if any(point_in_ring(lon, lat, r) for r in rings):
            return zid
    return None


def ring_area_m2(ring: List[List[float]]) -> float:
    """Shoelace on a local equirectangular projection — good enough for a ratio."""
    if len(ring) < 3:
        return 0.0
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx = 111320.0 * math.cos(math.radians(lat0))
    pts = [(p[0] * kx, p[1] * 110540.0) for p in ring]
    s = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts)))
    return abs(s) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", required=True, type=Path)
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("S", "W", "N", "E"), help="default: bbox of the zone file")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    zones = load_zones(args.zones)
    if not zones:
        raise SystemExit(f"Žiadne zóny v {args.zones}")
    if args.bbox:
        s, w, n, e = args.bbox
    else:
        w = min(z[3][0] for z in zones); s = min(z[3][1] for z in zones)
        e = max(z[3][2] for z in zones); n = max(z[3][3] for z in zones)
    print(f"zón: {len(zones)} | bbox {s:.4f},{w:.4f},{n:.4f},{e:.4f}")

    counts: Dict[int, int] = {z[0]: 0 for z in zones}
    per_group: Dict[str, int] = {}
    for name, selector in GROUPS.items():
        q = f'[out:json][timeout:150];{selector}({s},{w},{n},{e});out center;'
        data = overpass(q)
        hit = 0
        for el in data.get("elements", []):
            c = el.get("center") or el
            lon, lat = c.get("lon"), c.get("lat")
            if lon is None or lat is None:
                continue
            zid = assign(lon, lat, zones)
            if zid is not None:
                counts[zid] += 1
                hit += 1
        per_group[name] = hit
        print(f"  {name:<8} {len(data.get('elements', [])):>6} nájdených, {hit:>6} v zónach")

    # Sanity check only — not part of the weights.
    industrial: Dict[int, float] = {z[0]: 0.0 for z in zones}
    data = overpass(f'[out:json][timeout:150];{LANDUSE_QUERY}({s},{w},{n},{e});out geom;')
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 3:
            continue
        ring = [[p["lon"], p["lat"]] for p in geom]
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        zid = assign(cx, cy, zones)
        if zid is not None:
            industrial[zid] += ring_area_m2(ring) / 10_000.0  # ha

    total = sum(counts.values())
    if total <= 0:
        raise SystemExit("Nenašlo sa ani jedno pracovisko — skontroluj bbox a endpoint.")

    weights = {str(zid): round(c / total, 5) for zid, c in counts.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\npracovísk spolu: {total:,}".replace(",", " "))
    print(f"{'zóna':<28} {'pracovísk':>10} {'podiel':>8} {'priemysel ha':>13}")
    for zid, name, _r, _b in sorted(zones, key=lambda z: -counts[z[0]])[:12]:
        print(f"{name:<28} {counts[zid]:>10,} {counts[zid] / total:>7.1%} "
              f"{industrial[zid]:>13,.0f}".replace(",", " "))
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
