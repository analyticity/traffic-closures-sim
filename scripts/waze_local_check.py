#!/usr/bin/env python3
"""Waze check v5: streets around each closure, with the concurrent closures.

Two faults of v3 and v4, both raised by the author on 25 Sep 2026, are fixed:

1. Concurrent closures.  The model predicted each closure alone, although other
   roads were closed at the same time, and streets that were closed themselves
   were measured.  Here the prediction for a closure is the change between the
   network with its concurrent closures and the same network with this closure
   added, and streets closed by another closure are not measured.
2. The effect is local.  Only streets around the closure are compared.

The design was fixed before the first run, and every result is reported.

* Closures studied: the NDIC and Waze closures of ``waze_closure_check.py
  --extra``, with the same windows.
* Other closures: the eight NDIC closures, and every Waze road_closed closure
  of at least one day, merged and located as in ``waze_closures_build.py``.
  For a studied closure c, S = the other closures that are active on at least
  half of the weekdays of c's during window and close a link with modelled
  traffic.  A closure whose closed links overlap c's by at least half (of the
  smaller set) is the same closure and is not in S.
* Prediction: V(S + c) - V(S), both solved through the API with the same code
  (c alone against the null run when S is empty).
* Streets: named streets with modelled traffic whose nearest link lies within
  2 km of c's closed links.  The closed street and every street with a link
  closed by another closure active on any day of c's windows are left out.
  Jam reports (quality >= 50, all hours) are counted on the street's links
  within 2 km (link midpoint within 15 m of the jam line).  A street needs at
  least 5 reports in the control window, and a closure at least 10 streets.
* Per street: predicted change = sum of the change / sum of the null volume
  over those links; observed change y = ln((r_during + 0.5) / (r_control +
  0.5)), r = reports per weekday.
* Per closure: Spearman correlation between predicted and observed change.
  T = mean over closures; one-sided p from 20 000 permutations of y within
  closure and road class group.
* Secondary: rush hours (7-9, 15-18); streets the model gives at least 10 %
  more traffic against the other streets of the area (mean y difference within
  road class groups, same permutation).
* ``--ignore-concurrent`` repeats the test with each closure alone and without
  leaving out the streets closed by other closures, to show what the
  concurrent closures change.  It changes two things at once, so ``--alone``
  and ``--keep-disturbed`` switch them separately (added 28 Sep 2026).
* Added after the first run (25 Sep 2026), as secondary outcomes reported
  whatever they show: Waze hazard alerts and police accidents on the same
  streets and days, each report counted on its nearest link within 20 m.

Usage::

    python scripts/waze_local_check.py [--variant build_2026-09-23] [--api http://localhost:8000]
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
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detour_distance_baseline import (  # noqa: E402
    KEYS, M_PER_DEG_LAT, M_PER_DEG_LON, job_for, load_geometry)
from evaluate_detours import CSV_DIR, NULLS, ROOT, Api, load_links, norm  # noqa: E402
from waze_closure_check import (  # noqa: E402
    DATA_END, DATA_START, HOLIDAYS, MAIN, MIN_WEEKDAYS, local_day, windows)
from waze_closures_build import closed_part, waze_closures  # noqa: E402
from waze_link_check import jam_matrix  # noqa: E402

AREA_M, MIN_REPORTS, MIN_STREETS, GAIN = 2000.0, 5, 10, 0.10
ACTIVE_SHARE, SAME_SHARE, EPS = 0.5, 0.5, 0.5
N_PERM, SEED = 20_000, 20260927
POINT_M = 20.0


def point_matrix(csv_name, time_col, link_ids, geoms, days):
    """Reports per (link, day) for point data; each report counts on its nearest link within POINT_M."""
    d = pd.read_csv(CSV_DIR / csv_name, na_values=NULLS, low_memory=False,
                    usecols=[time_col, "location_geog"])
    d = d[d["location_geog"].notna()].copy()
    d["day"] = local_day(d[time_col])
    d = d[d["day"].isin(days)].copy()
    d["di"] = d["day"].map({x: k for k, x in enumerate(days)})
    pts = shapely.from_wkb(d["location_geog"].values)
    pts = shapely.transform(pts, lambda c: c * np.array([M_PER_DEG_LON, M_PER_DEG_LAT]))
    tree = shapely.STRtree(np.array([geoms[i] for i in link_ids], dtype=object))
    pi, li = tree.query_nearest(pts, max_distance=POINT_M)
    first = np.unique(pi, return_index=True)[1]          # one link per report
    pi, li = pi[first], li[first]
    M = np.zeros((len(link_ids), len(days)), dtype=np.int32)
    np.add.at(M, (li, d["di"].values[pi].astype(int)), 1)
    print(f"{csv_name}: {len(d):,} reports on weekdays, {len(pi):,} on a link", flush=True)
    return M


def run_links(api, link_modes, links):
    """Solve the network with *link_modes* closed; volume per link."""
    payload = []
    for lid, mode in link_modes.items():
        n = max(int(float(links[lid]["lanes"] or 2)), 1)
        payload.append({"link_id": int(lid), "direction": "both", "closure_type": mode, "lanes": n,
                        "lanes_remaining": (n - 1 if mode == "lanes" and n > 1 else 0)})
    jid = api.post("/api/scenarios/run", {"links": payload})["id"]
    while True:
        st = api.get(f"/api/scenarios/{jid}/status")
        if st["status"] == "done":
            break
        if st["status"] == "error":
            raise RuntimeError(st.get("error"))
        time.sleep(2)
    res = api.get(f"/api/scenarios/{jid}/results")
    return pd.Series({int(f["properties"]["link_id"]): float(f["properties"].get("wd_daily_tot") or 0)
                      for f in res["features"]})


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-23")
    ap.add_argument("--ignore-concurrent", action="store_true",
                    help="sensitivity run: each closure alone, closed streets of other closures kept "
                         "(= --alone --keep-disturbed)")
    ap.add_argument("--alone", action="store_true",
                    help="predict each closure alone, without the concurrent closures")
    ap.add_argument("--keep-disturbed", action="store_true",
                    help="keep the streets closed by other closures")
    args = ap.parse_args()
    alone = args.alone or args.ignore_concurrent
    keep_disturbed = args.keep_disturbed or args.ignore_concurrent

    ev = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    if alone and keep_disturbed:
        out = ev / "waze_v5_no_concurrent"
    elif alone:
        out = ev / "waze_v5_alone_excluded"
    elif keep_disturbed:
        out = ev / "waze_v5_concurrent_kept"
    else:
        out = ev / "waze_v5"
    out.mkdir(exist_ok=True)
    cache = ev / "waze_v5"          # scenario runs with the concurrent closures
    cache.mkdir(exist_ok=True)
    api = Api(args.api)
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))

    wz, by_id = waze_closures()
    links = {l["link_id"]: l for l in load_links(api)}
    geoms = load_geometry(api)
    by_name = {}
    for lid, l in links.items():
        if l["name"] and lid in geoms:
            by_name.setdefault(norm(l["name"]), []).append(lid)
    base = pd.read_csv(ev / "links_dobrovskeho.csv").set_index("link_id")
    vol = base["baseline"]

    # --- every closure known in the period
    allc = []
    for key in KEYS:
        job = job_for(key)
        recs = by_id.loc[job["records"]]
        names = {norm(n) for n in base.loc[base.index.intersection(meta["runs"][key]["links"]), "name"].dropna()}
        if isinstance(by_id.loc[job["anchor"], "street_name"], str):
            names.add(norm(by_id.loc[job["anchor"], "street_name"]))
        names |= {norm(x) for x in job.get("exclude", [])}
        allc.append({"id": key, "source": "ndic", "label": job["label"], "mode": job["mode"],
                     "links": set(meta["runs"][key]["links"]), "names": names,
                     "start": local_day(recs["valid_from"]).min(), "end": local_day(recs["valid_to"]).max()})
    for c in wz:
        if (c["t1"] - c["t0"]).total_seconds() < 86400:
            continue
        sel = closed_part(c, by_name, geoms)
        if sel:
            allc.append({"id": f"w{c['ids'][0]}", "source": "waze", "label": f"{c['street']} (Waze)",
                         "mode": "full", "links": set(sel), "names": {c["key"]}, "ids": set(c["ids"]),
                         "start": c["start"], "end": c["end"]})

    studied = [c for c in allc if c["source"] == "ndic"]
    for s in json.loads((ev / "waze_closures" / "closures.json").read_text(encoding="utf-8")):
        own = next(c for c in allc if c["source"] == "waze" and set(s["records"]) <= c.get("ids", set()))
        studied.append(dict(own, id=s["key"], links=set(s["links"])))

    def same(a, b):
        if a is b or (a.get("ids") and b.get("ids") and a["ids"] & b["ids"]):
            return True
        inter = len(a["links"] & b["links"])
        return inter >= SAME_SHARE * min(len(a["links"]), len(b["links"]))

    loaded = {i for i in base.index if vol.get(i, 0) > 0}
    link_ids = [i for i in base.index if i in geoms and base.loc[i, "link_type"] != "centroid_connector"]
    pos = {i: k for k, i in enumerate(link_ids)}
    days = [d for d in pd.date_range(DATA_START, DATA_END, freq="D") if d.weekday() < 5 and d not in HOLIDAYS]
    dpos = {d: k for k, d in enumerate(days)}
    A, P = jam_matrix(link_ids, geoms, days)
    H = point_matrix("alerts.csv", "first_seen", link_ids, geoms, days)
    X = point_matrix("accidents.csv", "event_time", link_ids, geoms, days)
    geo_arr = np.array([geoms[i] for i in link_ids], dtype=object)
    names_arr = np.array([norm(base.loc[i, "name"]) if isinstance(base.loc[i, "name"], str) else "" for i in link_ids])
    raw_names = np.array([base.loc[i, "name"] if isinstance(base.loc[i, "name"], str) else "" for i in link_ids])
    vol_arr = vol.reindex(link_ids).fillna(0).values
    len_arr = base["distance"].reindex(link_ids).fillna(0).values
    main_arr = base["link_type"].reindex(link_ids).isin(MAIN).values

    units, rows = [], []
    for c in studied:
        before, during, after = windows(c["start"], c["end"])
        control = before + after
        row = {"key": c["id"], "label": c["label"], "source": c["source"]}
        if len(during) < MIN_WEEKDAYS or len(control) < MIN_WEEKDAYS:
            rows.append(dict(row, result="no comparison period"))
            continue
        others = [o for o in allc if not same(o, c)]
        S = [o for o in others
             if sum(o["start"] <= d <= o["end"] for d in during) >= ACTIVE_SHARE * len(during)
             and o["links"] & loaded]
        if alone:
            S = []
        win = before + during + after
        disturbed = set()
        for o in others:
            if any(o["start"] <= d <= o["end"] for d in win) and not keep_disturbed:
                disturbed |= o["names"]

        # prediction with the concurrent closures
        f_s, f_sc = cache / f"scen_{c['id']}_S.csv", cache / f"scen_{c['id']}_Sc.csv"
        modes = {}
        for o in S:
            for lid in o["links"]:
                modes[lid] = "full" if modes.get(lid) == "full" or o["mode"] == "full" else o["mode"]
        if not S:
            src = ev / "waze_closures" / f"links_{c['id']}.csv" if c["source"] == "waze" else ev / f"links_{c['id']}.csv"
            delta = pd.read_csv(src).set_index("link_id")["delta"]
        else:
            if f_s.exists() and f_sc.exists():
                v_s = pd.read_csv(f_s, index_col=0).iloc[:, 0]
                v_sc = pd.read_csv(f_sc, index_col=0).iloc[:, 0]
            else:
                t0 = time.time()
                v_s = run_links(api, modes, links)
                both = dict(modes)
                for lid in c["links"]:
                    both[lid] = "full" if both.get(lid) == "full" or c["mode"] == "full" else c["mode"]
                v_sc = run_links(api, both, links)
                v_s.to_csv(f_s)
                v_sc.to_csv(f_sc)
                print(f"  {c['id']}: {len(S)} concurrent closures, {time.time() - t0:.0f} s", flush=True)
            delta = v_sc - v_s
        d_arr = delta.reindex(link_ids).fillna(0).values

        # streets around the closure
        closed_geom = shapely.union_all([geoms[i] for i in c["links"] if i in geoms])
        dist = shapely.distance(geo_arr, closed_geom)
        closed_mask = np.array([i in c["links"] for i in link_ids])
        local = (dist <= AREA_M) & ~closed_mask & (names_arr != "")
        local &= ~np.isin(names_arr, list(c["names"] | disturbed))
        streets = []
        for nm in np.unique(names_arr[local]):
            m = local & (names_arr == nm)
            b = vol_arr[m].sum()
            if b <= 0:
                continue
            jc = A[m][:, [dpos[d] for d in control]].sum()
            if jc < MIN_REPORTS:
                continue
            jd = A[m][:, [dpos[d] for d in during]].sum()
            pc = P[m][:, [dpos[d] for d in control]].sum()
            pd_ = P[m][:, [dpos[d] for d in during]].sum()
            hc = H[m][:, [dpos[d] for d in control]].sum()
            hd = H[m][:, [dpos[d] for d in during]].sum()
            xc = X[m][:, [dpos[d] for d in control]].sum()
            xd = X[m][:, [dpos[d] for d in during]].sum()
            streets.append({"street": raw_names[m][0], "pred": d_arr[m].sum() / b,
                            "main": (len_arr[m] * main_arr[m]).sum() >= 0.5 * len_arr[m].sum(),
                            "y": np.log((jd / len(during) + EPS) / (jc / len(control) + EPS)),
                            "y_rush": np.log((pd_ / len(during) + EPS) / (pc / len(control) + EPS)),
                            "y_hazards": np.log((hd / len(during) + EPS) / (hc / len(control) + EPS)),
                            "y_accidents": np.log((xd / len(during) + EPS) / (xc / len(control) + EPS)),
                            "hazards": int(hc + hd), "accidents": int(xc + xd)})
        St = pd.DataFrame(streets)
        row.update({"concurrent": len(S), "streets": len(St)})
        if len(St) < MIN_STREETS:
            rows.append(dict(row, result="too few streets"))
            continue
        row.update({"rho": spearman(St["pred"], St["y"]), "rho_rush": spearman(St["pred"], St["y_rush"]),
                    "gainers": int((St["pred"] >= GAIN).sum()), "result": "used"})
        rows.append(row)
        St["closure"] = c["id"]
        units.append(St)

    rng = np.random.default_rng(SEED)

    def perm_test(col, kind):
        obs, null = [], np.zeros(N_PERM)
        for St in units:
            x, y, g = St["pred"].values, St[col].values, St["main"].values
            if kind == "rho":
                o = spearman(x, y)
            else:
                gain = x >= GAIN
                if gain.sum() == 0 or (~gain).sum() == 0:
                    continue
                o = gain_diff(gain, y, g)
            if np.isnan(o):
                continue
            obs.append(o)
            perm = np.empty(N_PERM)
            yp = np.tile(y, (N_PERM, 1))
            for grp in (True, False):
                idx = np.where(g == grp)[0]
                if len(idx) > 1:
                    order = np.argsort(rng.random((N_PERM, len(idx))), axis=1)
                    yp[:, idx] = y[idx][order]
            if kind == "rho":
                rx = rankdata(x) - (len(x) + 1) / 2
                ry = np.apply_along_axis(rankdata, 1, yp) - (len(x) + 1) / 2
                perm = (ry @ rx) / np.sqrt((rx ** 2).sum() * (ry ** 2).sum(axis=1))
            else:
                perm = np.array([gain_diff(gain, yp[k], g) for k in range(N_PERM)])
            null += perm
        T = float(np.mean(obs))
        null /= len(obs)
        return {"T": T, "p": float((null >= T).mean()), "closures": len(obs),
                "positive": int(sum(o > 0 for o in obs))}

    def gain_diff(gain, y, g):
        num, w = 0.0, 0
        for grp in (True, False):
            t, k = gain & (g == grp), ~gain & (g == grp)
            if t.sum() and k.sum():
                num += t.sum() * (y[t].mean() - y[k].mean())
                w += t.sum()
        return num / w if w else np.nan

    results = {"spearman_all_hours": perm_test("y", "rho"),
               "spearman_rush_hours": perm_test("y_rush", "rho"),
               "gainers_vs_others": perm_test("y", "gain"),
               "spearman_hazards": perm_test("y_hazards", "rho"),
               "spearman_accidents": perm_test("y_accidents", "rho")}
    all_st = pd.concat(units)
    results["counts_on_studied_streets"] = {"hazards": int(all_st["hazards"].sum()),
                                            "accidents": int(all_st["accidents"].sum())}

    R = pd.DataFrame(rows)
    R.to_csv(out / "closures.csv", index=False)
    pd.concat(units).to_csv(out / "streets.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    print()
    for k, r in results.items():
        if "T" not in r:
            print(f"{k:22s} {r}")
            continue
        print(f"{k:22s} T = {r['T']:+.3f}  p = {r['p']:.4f}  positive {r['positive']} of {r['closures']}")
    ex = next((u for u in units if u["closure"].iloc[0] == "jihlavska"), None)
    if ex is not None:
        print("\nJihlavská, streets around the closure (predicted vs observed change):")
        print(ex.sort_values("pred", ascending=False).head(12).round(3).to_string(index=False))
    print("\nsaved to", out)


if __name__ == "__main__":
    sys.exit(main())
