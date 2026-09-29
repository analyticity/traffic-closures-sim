#!/usr/bin/env python3
"""Closures reported by Waze, as extra cases for ``waze_closure_check.py``.

The rules below were fixed before the scenarios were run (25 Sep 2026).

* Waze ``road_closed`` records of one street are one closure when their
  periods overlap or are at most 2 days apart and their points lie within
  1 km of each other.  The closure lasts from the first to the last report.
* A closure must last at least 7 days.
* The closed part: links of the same street (normalised name) within 100 m of
  any report point of the closure, closed fully in both directions -- Waze
  does not say more.  Streets named only by a road number are skipped.
* The model must carry at least 1 000 veh/day on the closed part in the null run.
* A closure that is one of the eight NDIC closures of the article (report
  within 500 m of its closed links, periods overlapping) is left out; the NDIC
  record is used instead.
* The closure needs a comparison period inside the Waze data (the windows of
  ``waze_closure_check.py``).

Each kept closure is solved through the running API.  The per-link results go
to ``<variant>/vyhodnotenie_v2/waze_closures/links_<key>.csv`` and the list of
closures to ``closures.json`` in the same folder.

Usage::

    python scripts/waze_closures_build.py [--variant build_2026-09-23] [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import wkb
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detour_distance_baseline import (  # noqa: E402
    KEYS, M_PER_DEG_LAT, M_PER_DEG_LON, job_for, load_geometry)
from evaluate_detours import (  # noqa: E402
    CSV_DIR, NULLS, ROOT, Api, load_links, norm, run_scenario)
from waze_closure_check import MIN_WEEKDAYS, local_day, windows  # noqa: E402

MERGE_GAP_DAYS, MERGE_DIST_M = 2, 1000.0
MIN_DAYS, NEAR_M, MIN_VOLUME, NDIC_DIST_M = 7, 100.0, 1000.0, 500.0


def to_m(lon, lat):
    return Point(lon * M_PER_DEG_LON, lat * M_PER_DEG_LAT)


def waze_closures():
    """Waze road_closed records merged into closures; also the table indexed by id."""
    r = pd.read_csv(CSV_DIR / "restrictions.csv", na_values=NULLS, low_memory=False)
    by_id = r.set_index("id")
    ids = r["external_ids"].fillna("")
    r = r[ids.str.contains('"waze"') & ~ids.str.contains('"ndic"')
          & (r["restriction_type"] == "road_closed") & r["street_name"].notna()].copy()
    r["start"], r["end"] = local_day(r["first_seen"]), local_day(r["last_seen"])
    r["t0"] = pd.to_datetime(r["first_seen"], utc=True)
    r["t1"] = pd.to_datetime(r["last_seen"], utc=True)
    r["key"] = r["street_name"].map(norm)
    r["pt"] = r["location_point_geog"].map(
        lambda h: (lambda p: to_m(p.x, p.y))(wkb.loads(h, hex=True)) if isinstance(h, str) else None)
    r = r[r["pt"].notna()].sort_values(["key", "t0"])

    closures = []
    for key, grp in r.groupby("key"):
        cur = None
        for _, x in grp.iterrows():
            if (cur is not None and x.t0 <= cur["t1"] + pd.Timedelta(days=MERGE_GAP_DAYS)
                    and min(x.pt.distance(p) for p in cur["pts"]) <= MERGE_DIST_M):
                cur["t1"] = max(cur["t1"], x.t1)
                cur["end"] = max(cur["end"], x.end)
                cur["pts"].append(x.pt)
                cur["ids"].append(int(x.id))
            else:
                if cur is not None:
                    closures.append(cur)
                cur = {"key": key, "street": x.street_name, "t0": x.t0, "t1": x.t1,
                       "start": x.start, "end": x.end, "pts": [x.pt], "ids": [int(x.id)]}
        if cur is not None:
            closures.append(cur)
    return closures, by_id


def closed_part(c, by_name, geoms):
    """Links of the closure's street within NEAR_M of any of its report points."""
    return [lid for lid in by_name.get(c["key"], [])
            if min(geoms[lid].distance(p) for p in c["pts"]) <= NEAR_M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-23")
    args = ap.parse_args()

    ev = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    out = ev / "waze_closures"
    out.mkdir(exist_ok=True)
    api = Api(args.api)

    closures, ndic_ids = waze_closures()
    print(f"Waze closures {len(closures):,}", flush=True)

    # --- model network
    links = {l["link_id"]: l for l in load_links(api)}
    geoms = load_geometry(api)
    by_name = {}
    for lid, l in links.items():
        if l["name"] and lid in geoms:
            by_name.setdefault(norm(l["name"]), []).append(lid)
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))
    ndic = []
    for key in KEYS:
        job = job_for(key)
        recs = ndic_ids.loc[job["records"]]
        geom = shapely.union_all([geoms[i] for i in meta["runs"][key]["links"] if i in geoms])
        ndic.append((key, local_day(recs["valid_from"]).min(), local_day(recs["valid_to"]).max(), geom))

    kept, report = [], []
    for c in closures:
        row = {"street": c["street"], "start": f"{c['start']:%Y-%m-%d}", "end": f"{c['end']:%Y-%m-%d}",
               "records": len(c["ids"])}
        days = (c["t1"] - c["t0"]).total_seconds() / 86400
        if days < MIN_DAYS:
            continue                                   # short closures are not listed
        row["days"] = round(days, 1)
        if c["key"].replace(" ", "").isdigit():
            report.append(dict(row, result="road number only"))
            continue
        sel = closed_part(c, by_name, geoms)
        if not sel:
            report.append(dict(row, result="not in the network"))
            continue
        vol = max(links[lid]["vol"] for lid in sel)
        row.update({"links": len(sel), "max_volume": round(vol)})
        if vol < MIN_VOLUME:
            report.append(dict(row, result="below 1 000 veh/day"))
            continue
        dup = [k for k, s, e, g in ndic
               if s <= c["end"] and c["start"] <= e and min(g.distance(p) for p in c["pts"]) <= NDIC_DIST_M]
        if dup:
            report.append(dict(row, result=f"same as NDIC {dup[0]}"))
            continue
        before, during, after = windows(c["start"], c["end"])
        if len(during) < MIN_WEEKDAYS or len(before) + len(after) < MIN_WEEKDAYS:
            report.append(dict(row, result="no comparison period"))
            continue
        key = f"waze_{c['key'].replace(' ', '_')}_{c['start']:%m%d}"
        t0 = time.time()
        df, _ = run_scenario(api, [links[lid] for lid in sel], "full")
        df.to_csv(out / f"links_{key}.csv", index=False)
        kept.append({"key": key, "label": f"{c['street']} (Waze)", "street": c["street"],
                     "start": f"{c['start']:%Y-%m-%d}", "end": f"{c['end']:%Y-%m-%d}",
                     "links": sel, "closed_names": [c["street"]], "records": c["ids"],
                     "max_volume": round(vol)})
        report.append(dict(row, result=f"kept as {key} ({time.time() - t0:.0f} s)"))
        print(f"✓ {key}: {len(sel)} links, {vol:,.0f} veh/day", flush=True)

    (out / "closures.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
    R = pd.DataFrame(report)
    R.to_csv(out / "selection.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    print(f"\nkept {len(kept)} closures; saved to {out}")


if __name__ == "__main__":
    sys.exit(main())
