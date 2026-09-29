#!/usr/bin/env python3
"""Waze cross-check of the closure predictions, v3.

Question: do the streets that the model predicts to gain the most traffic
during a closure show a larger change in Waze congestion reports than other
streets of the same road class on the same days?

The design below was fixed before the first run on the data of 25 Sep 2026,
and every result is reported, whatever it shows.

* Closures: the eight of the article (link-selection rule of Section V-A;
  scenario results saved by ``evaluate_detours.py``).  The closure period is
  the union of the validity of its records.  Weekdays only, Czech public
  holidays excluded.
* Windows.  Waze data start on 14 Apr 2026, so most closures have no period
  before them, but all except one ended while the data were collected.
  - control: the 14 days before the start and/or the 14 days from the third
    day after the end, whichever lies inside the data (both if both do);
  - during: 28 days of the closure next to the control window (from the third
    day of the closure if a before window exists, otherwise the last 28 days
    before the end), inside the data;
  - a closure needs at least 5 weekdays in each window.
* Streets: named streets that carry modelled traffic in the null run, matched
  to Waze by normalised name, with at least 5 jam reports (quality >= 50) in
  the control window.  Road class group: main if motorway, trunk, primary and
  secondary links (with their ramps) make up at least half of its length,
  otherwise minor.
* Treated: the model's 20 streets with the largest predicted increase (the
  ranking of Section V-B), if eligible.  Controls: the other eligible streets,
  without the model's top 50 for this closure and without the top 50 of any
  other of the eight closures that is active in a window of this one.
* Outcome: y = ln((r_during + 0.1) / (r_control + 0.1)), r = reports per
  weekday.
* Statistic: per closure, the mean y of the treated streets minus the mean y
  of the controls, within road class groups, weighted by the treated streets
  in each group; T = mean over closures.
* Inference: 20 000 permutations of the treated label within closure and road
  class group; one-sided p = share of permutations with T_perm >= T.
* Secondary outcomes, same design and streets: jam delay (seconds per
  weekday), Waze hazard alerts, police accidents.  Robustness: top 10 and top
  50 as treated; streets whose nearest link lies within 300 m of the closed
  section left out.
* With ``--extra``, the closures reported by Waze (``waze_closures_build.py``)
  are added with the same design, and the NDIC and Waze closures are also
  tested separately.

Usage::

    python scripts/waze_closure_check.py [--variant build_2026-09-23] [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detour_distance_baseline import KEYS, job_for, load_geometry  # noqa: E402
from evaluate_detours import CSV_DIR, NULLS, ROOT, Api, norm, rank_streets  # noqa: E402

DATA_START, DATA_END = pd.Timestamp("2026-04-15"), pd.Timestamp("2026-09-24")  # full days
HOLIDAYS = {pd.Timestamp(d) for d in ("2026-05-01", "2026-05-08", "2026-07-06", "2026-09-28")}
RAMP, CONTROL_DAYS, DURING_DAYS, MIN_WEEKDAYS = 2, 14, 28, 5
QUALITY_MIN, MIN_REPORTS, TOP_N, EXCL_N, NEAR_M = 50, 5, 20, 50, 300.0
N_PERM, SEED = 20_000, 20260925
MAIN = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link"}
OUTCOMES = ["jams", "delay", "hazards", "accidents"]


def street_key(name):
    k = norm(name)
    for t in ("trida ", " trida"):
        k = k.replace(t, " ")
    return " ".join(k.split())


def weekdays(first, last):
    days = pd.date_range(first, last, freq="D")
    return [d for d in days if d.weekday() < 5 and d not in HOLIDAYS
            and DATA_START <= d <= DATA_END]


def local_day(series):
    ts = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Europe/Prague")
    return ts.dt.tz_localize(None).dt.normalize()


def daily(df, day_col, value=None):
    """(street_key, day) -> count or sum of *value*."""
    df = df[df["street_name"].notna()].copy()
    df["key"] = df["street_name"].map(street_key)
    g = df.groupby(["key", day_col])
    return (g.size() if value is None else g[value].sum()).rename("v")


def load_waze():
    jams = pd.read_csv(CSV_DIR / "jams.csv", na_values=NULLS, low_memory=False,
                       usecols=["first_seen", "street_name", "delay_seconds", "quality_score"])
    jams = jams[jams["quality_score"].fillna(0) >= QUALITY_MIN].copy()
    # Waze writes -1 for a blocked road; that is not a delay in seconds.
    jams["delay_seconds"] = jams["delay_seconds"].where(jams["delay_seconds"] >= 0)
    jams["day"] = local_day(jams["first_seen"])
    alerts = pd.read_csv(CSV_DIR / "alerts.csv", na_values=NULLS, low_memory=False,
                         usecols=["first_seen", "street_name"])
    alerts["day"] = local_day(alerts["first_seen"])
    acc = pd.read_csv(CSV_DIR / "accidents.csv", na_values=NULLS, low_memory=False,
                      usecols=["event_time", "street_name"])
    acc["day"] = local_day(acc["event_time"])
    return {
        "jams": daily(jams, "day"),
        "delay": daily(jams, "day", "delay_seconds"),
        "hazards": daily(alerts, "day"),
        "accidents": daily(acc, "day"),
    }


def rate(series, keys, days):
    """Mean per weekday over *days* for each street key."""
    if not days:
        return pd.Series(np.nan, index=keys)
    sub = series[series.index.get_level_values(1).isin(days)]
    tot = sub.groupby(level=0).sum().reindex(keys).fillna(0.0)
    return tot / len(days)


def windows(start, end):
    before = weekdays(start - pd.Timedelta(days=CONTROL_DAYS), start - pd.Timedelta(days=1)) \
        if start - pd.Timedelta(days=CONTROL_DAYS) >= DATA_START else []
    a0 = end + pd.Timedelta(days=RAMP + 1)
    after = weekdays(a0, a0 + pd.Timedelta(days=CONTROL_DAYS - 1)) \
        if a0 + pd.Timedelta(days=CONTROL_DAYS - 1) <= DATA_END else []
    d0 = start + pd.Timedelta(days=RAMP)
    if before:
        during = weekdays(d0, min(end, d0 + pd.Timedelta(days=DURING_DAYS - 1)))
    else:
        during = weekdays(max(d0, end - pd.Timedelta(days=DURING_DAYS - 1)), end)
    return before, during, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-23")
    ap.add_argument("--extra", action="store_true",
                    help="add the Waze-reported closures of waze_closures_build.py")
    args = ap.parse_args()

    ev = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    out = ev / ("waze_v3_extended" if args.extra else "waze_v3")
    out.mkdir(exist_ok=True)
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))
    restr = pd.read_csv(CSV_DIR / "restrictions.csv", na_values=NULLS,
                        low_memory=False).set_index("id")
    waze = load_waze()
    geoms = load_geometry(Api(args.api))
    print({k: int(v.sum()) for k, v in waze.items()}, flush=True)

    # --- closures: the eight NDIC ones and, with --extra, those reported by Waze
    specs = []
    for key in KEYS:
        job = job_for(key)
        recs = restr.loc[job["records"]]
        df = pd.read_csv(ev / f"links_{key}.csv")
        closed_ids = meta["runs"][key]["links"]
        closed_names = {norm(n) for n in df.loc[df["link_id"].isin(closed_ids), "name"].dropna()}
        if isinstance(restr.loc[job["anchor"], "street_name"], str):
            closed_names.add(norm(restr.loc[job["anchor"], "street_name"]))
        for x in job.get("exclude", []):
            closed_names.add(norm(x))
        specs.append((key, job["label"], "ndic", local_day(recs["valid_from"]).min(),
                      local_day(recs["valid_to"]).max(), df, closed_ids, closed_names))
    if args.extra:
        wc = ev / "waze_closures"
        for s in json.loads((wc / "closures.json").read_text(encoding="utf-8")):
            df = pd.read_csv(wc / f"links_{s['key']}.csv")
            names = {norm(n) for n in df.loc[df["link_id"].isin(s["links"]), "name"].dropna()}
            specs.append((s["key"], s["label"], "waze", pd.Timestamp(s["start"]), pd.Timestamp(s["end"]),
                          df, s["links"], names | {norm(n) for n in s["closed_names"]}))

    # --- model side: street table, ranking, class group, distance per closure
    C = {}
    for key, label, source, start, end, df, closed_ids, closed_names in specs:
        g = rank_streets(df, closed_names)
        d = df[df["name"].notna() & (df["link_type"] != "centroid_connector")]
        d = d[~d["name"].map(lambda n: norm(n) in closed_names)].copy()
        main_len = d["distance"].where(d["link_type"].isin(MAIN), 0.0)
        share = main_len.groupby(d["name"]).sum() / d.groupby("name")["distance"].sum().clip(lower=1e-9)
        g["group"] = np.where(share.reindex(g.index).fillna(0) >= 0.5, "main", "minor")
        closed = shapely.union_all([geoms[i] for i in closed_ids if i in geoms])
        d["dist_m"] = shapely.distance(np.array([geoms.get(i) for i in d["link_id"]], dtype=object), closed)
        g["dist_m"] = d.groupby("name")["dist_m"].min().reindex(g.index)
        g["key"] = [street_key(n) for n in g.index]
        ranked = g[g["modelled"]].sort_values("delta", ascending=False)
        C[key] = {"label": label, "source": source, "start": start, "end": end, "g": g,
                  "top": {n: list(ranked.index[:n]) for n in (10, 20, 50)}}
        C[key]["before"], C[key]["during"], C[key]["after"] = windows(start, end)

    def active_in(key, days):
        s, e = C[key]["start"], C[key]["end"]
        return any(s <= d <= e for d in days)

    # --- per closure: eligible streets and outcomes
    rows, detail, units = [], [], {}
    for key in C:
        c = C[key]
        control = c["before"] + c["after"]
        testable = len(c["during"]) >= MIN_WEEKDAYS and len(control) >= MIN_WEEKDAYS
        row = {"key": key, "label": c["label"], "source": c["source"],
               "closure": f"{c['start']:%Y-%m-%d}..{c['end']:%Y-%m-%d}",
               "before_days": len(c["before"]), "during_days": len(c["during"]),
               "after_days": len(c["after"]),
               "during": f"{c['during'][0]:%m-%d}..{c['during'][-1]:%m-%d}" if c["during"] else "",
               "testable": testable}
        if not testable:
            rows.append(row)
            continue
        g = c["g"]
        base = g[g["baseline"] > 0].copy()
        base = base[base["key"].isin(set(waze["jams"].index.get_level_values(0)))]
        keys = list(base["key"])
        rc = rate(waze["jams"], keys, control)
        base = base[(rc * len(control)).values >= MIN_REPORTS]
        excl = set(c["top"][EXCL_N])
        for other in C:
            if other != key and active_in(other, c["during"] + control):
                excl |= set(C[other]["top"][EXCL_N])
        base["y"] = np.nan
        for oc in OUTCOMES:
            r_c = rate(waze[oc], list(base["key"]), control).values
            r_d = rate(waze[oc], list(base["key"]), c["during"]).values
            base[f"rc_{oc}"], base[f"rd_{oc}"] = r_c, r_d
            base[f"y_{oc}"] = np.log((r_d + 0.1) / (r_c + 0.1))
        units[key] = {"base": base, "excl": excl}
        row.update({"eligible": len(base)})
        rows.append(row)

    rng = np.random.default_rng(SEED)

    def test(outcome, top_n, drop_near=False, source=None):
        """Observed T, permutation p and per-closure differences."""
        strata, per = [], {}
        for key, u in units.items():
            if source and C[key]["source"] != source:
                continue
            b = u["base"]
            if drop_near:
                b = b[~(b["dist_m"] < NEAR_M)]
            treated = set(C[key]["top"][top_n])
            is_t = b.index.isin(list(treated))
            is_c = ~is_t & ~b.index.isin(list(u["excl"] - treated))
            n_t = int(is_t.sum())
            if n_t == 0 or is_c.sum() == 0:
                continue
            diff = 0.0
            for grp in ("main", "minor"):
                t = b.loc[is_t & (b["group"] == grp).values, f"y_{outcome}"].values
                k = b.loc[is_c & (b["group"] == grp).values, f"y_{outcome}"].values
                if len(t) == 0:
                    continue
                if len(k) == 0:          # no control in this group: drop the treated there
                    n_t -= len(t)
                    continue
                strata.append((key, np.concatenate([t, k]), len(t)))
                diff += len(t) * (t.mean() - k.mean())
            if n_t > 0:
                per[key] = {"n_treated": n_t, "n_control": int(is_c.sum()), "diff": diff / n_t}
        if not per:
            return None
        T = float(np.mean([v["diff"] for v in per.values()]))
        perm = np.zeros(N_PERM)
        wsum = {key: sum(nt for kk, _, nt in strata if kk == key) for key in per}
        for key in per:
            acc = np.zeros(N_PERM)
            for kk, pool, nt in strata:
                if kk != key:
                    continue
                idx = np.argsort(rng.random((N_PERM, len(pool))), axis=1)
                sel = pool[idx]
                acc += nt * (sel[:, :nt].mean(axis=1) - sel[:, nt:].mean(axis=1))
            perm += acc / wsum[key]
        perm /= len(per)
        p = float((perm >= T).mean())
        return {"T": T, "p": p, "closures": len(per), "per_closure": per}

    results = {}
    for oc in OUTCOMES:
        results[f"{oc}_top20"] = test(oc, 20)
    results["jams_top10"] = test("jams", 10)
    results["jams_top50"] = test("jams", 50)
    results["jams_top20_without_near"] = test("jams", 20, drop_near=True)
    if args.extra:
        results["jams_top20_ndic_only"] = test("jams", 20, source="ndic")
        results["jams_top20_waze_only"] = test("jams", 20, source="waze")
        results["delay_top20_waze_only"] = test("delay", 20, source="waze")

    R = pd.DataFrame(rows)
    prim = results["jams_top20"]["per_closure"] if results["jams_top20"] else {}
    R["n_treated"] = R["key"].map(lambda k: prim.get(k, {}).get("n_treated"))
    R["n_control"] = R["key"].map(lambda k: prim.get(k, {}).get("n_control"))
    R["diff_jams"] = R["key"].map(lambda k: prim.get(k, {}).get("diff"))
    R.to_csv(out / "closures.csv", index=False)
    for key, u in units.items():
        b = u["base"].copy()
        b["treated"] = b.index.isin(C[key]["top"][20])
        b["closure"] = key
        detail.append(b.reset_index())
    pd.concat(detail).to_csv(out / "streets.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(R.to_string(index=False))
    print()
    for name, r in results.items():
        if r is None:
            print(f"{name:28s} n/a")
            continue
        print(f"{name:28s} T = {r['T']:+.3f}  (x{np.exp(r['T']):.2f})  p = {r['p']:.4f}  closures = {r['closures']}")
    print("\nsaved to", out)


if __name__ == "__main__":
    sys.exit(main())
