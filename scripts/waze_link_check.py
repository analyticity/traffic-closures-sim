#!/usr/bin/env python3
"""Waze check v4: jams on the road sections where the model moves traffic, rush hours.

``waze_closure_check.py`` counts jam reports on whole streets over the whole
day.  This version counts them only on the links where the model adds traffic,
and only in the rush hours.  The design was fixed before the first run
(25 Sep 2026), and every result is reported.

* Closures: the NDIC closures of the article and the Waze-reported closures of
  ``waze_closures_build.py``, with the windows of ``waze_closure_check.py``
  (weekdays, holidays excluded; control = 14 days before and/or after, during =
  up to 28 days of the closure).
* Jams: Waze jam reports with quality >= 50 that start on a weekday between
  7:00 and 8:59 or between 15:00 and 17:59.  A report counts for every model
  link whose midpoint lies within 15 m of the jam line.
* Links of a closure (closed links left out):
  - affected: the model adds at least 10 % and at least 500 veh/day to the
    link's volume in the null run;
  - control: at least 500 veh/day in the null run, the model changes the volume
    by at most 1 % and at most 100 veh/day, and no other closure active in a
    window of this one changes it by 5 % or more.
  Both sets are split into main links (motorway, trunk, primary, secondary and
  their ramps) and minor links.  A group is used when both sets have at least 5
  reports in the control window.
* Per closure: D = ln(change of the rate on the affected links) - ln(change of
  the rate on the control links), rate = reports per weekday + 0.5, averaged
  over the groups weighted by the number of affected links.
* Test: mean D over closures; one-sided p from 200 000 random sign flips of
  the closure values.  Robustness: all hours; a 20 % threshold for affected
  links.
* Event study: the same difference week by week, aligned on the start (weeks
  -2..3) and on the end (weeks -3..2) of the closures, relative to the weeks
  outside the closure, averaged over closures.

Usage::

    python scripts/waze_link_check.py [--variant build_2026-09-23] [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detour_distance_baseline import (  # noqa: E402
    KEYS, M_PER_DEG_LAT, M_PER_DEG_LON, job_for, load_geometry)
from evaluate_detours import CSV_DIR, NULLS, ROOT, Api  # noqa: E402
from waze_closure_check import (  # noqa: E402
    DATA_END, DATA_START, HOLIDAYS, MAIN, MIN_WEEKDAYS, QUALITY_MIN, local_day, windows)

PEAK_HOURS = {7, 8, 15, 16, 17}
MATCH_M = 15.0
AFFECTED_SHARE, AFFECTED_MIN = 0.10, 500.0
CONTROL_MIN_VOL, CONTROL_SHARE, CONTROL_ABS, OTHER_SHARE = 500.0, 0.01, 100.0, 0.05
MIN_REPORTS, EPS, N_FLIP, SEED = 5, 0.5, 200_000, 20260926
START_WEEKS, END_WEEKS = [-2, -1, 0, 1, 2, 3], [-3, -2, -1, 1, 2]


def jam_matrix(link_ids, geoms, days):
    """Reports per (link, day): all hours and rush hours."""
    j = pd.read_csv(CSV_DIR / "jams.csv", na_values=NULLS, low_memory=False,
                    usecols=["first_seen", "quality_score", "jam_line_geog"])
    j = j[(j["quality_score"].fillna(0) >= QUALITY_MIN) & j["jam_line_geog"].notna()].copy()
    ts = pd.to_datetime(j["first_seen"], errors="coerce", utc=True).dt.tz_convert("Europe/Prague")
    ts = ts.dt.tz_localize(None)
    j["day"], j["hour"] = ts.dt.normalize(), ts.dt.hour
    j = j[j["day"].isin(days)].copy()
    j["di"] = j["day"].map({d: k for k, d in enumerate(days)})
    lines = shapely.from_wkb(j["jam_line_geog"].values)
    lines = shapely.transform(lines, lambda c: c * np.array([M_PER_DEG_LON, M_PER_DEG_LAT]))
    mids = shapely.point_on_surface(np.array([geoms[i] for i in link_ids], dtype=object))
    jam_i, link_i = shapely.STRtree(mids).query(lines, predicate="dwithin", distance=MATCH_M)
    di = j["di"].values[jam_i].astype(int)
    peak = j["hour"].isin(PEAK_HOURS).values[jam_i]
    A = np.zeros((len(link_ids), len(days)), dtype=np.int32)
    P = np.zeros_like(A)
    np.add.at(A, (link_i, di), 1)
    np.add.at(P, (link_i[peak], di[peak]), 1)
    print(f"jam reports {len(j):,} -> {len(jam_i):,} link matches", flush=True)
    return A, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-23")
    args = ap.parse_args()

    ev = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    out = ev / "waze_v4"
    out.mkdir(exist_ok=True)
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))
    restr = pd.read_csv(CSV_DIR / "restrictions.csv", na_values=NULLS, low_memory=False).set_index("id")

    # --- closures
    specs = []
    for key in KEYS:
        job = job_for(key)
        recs = restr.loc[job["records"]]
        specs.append({"key": key, "label": job["label"], "source": "ndic",
                      "start": local_day(recs["valid_from"]).min(), "end": local_day(recs["valid_to"]).max(),
                      "df": pd.read_csv(ev / f"links_{key}.csv"), "closed": set(meta["runs"][key]["links"])})
    wc = ev / "waze_closures"
    for s in json.loads((wc / "closures.json").read_text(encoding="utf-8")):
        specs.append({"key": s["key"], "label": s["label"], "source": "waze",
                      "start": pd.Timestamp(s["start"]), "end": pd.Timestamp(s["end"]),
                      "df": pd.read_csv(wc / f"links_{s['key']}.csv"), "closed": set(s["links"])})

    geoms = load_geometry(Api(args.api))
    base = specs[0]["df"].set_index("link_id")
    link_ids = [i for i in base.index if i in geoms and base.loc[i, "link_type"] != "centroid_connector"]
    pos = {i: k for k, i in enumerate(link_ids)}
    days = [d for d in pd.date_range(DATA_START, DATA_END, freq="D") if d.weekday() < 5 and d not in HOLIDAYS]
    dpos = {d: k for k, d in enumerate(days)}
    A, P = jam_matrix(link_ids, geoms, days)

    vol = base.loc[link_ids, "baseline"].values
    main_link = base.loc[link_ids, "link_type"].isin(MAIN).values
    for s in specs:
        d = s["df"].set_index("link_id")["delta"].reindex(link_ids).fillna(0.0).values
        s["delta"] = d
        s["share"] = np.where(vol > 0, np.abs(d) / np.maximum(vol, 1e-9), 0.0)
        s["closed_mask"] = np.array([i in s["closed"] for i in link_ids])
        s["before"], s["during"], s["after"] = windows(s["start"], s["end"])

    def active(s, days_):
        return any(s["start"] <= x <= s["end"] for x in days_)

    def sets(s, share_thr):
        others = [o for o in specs if o is not s and active(o, s["during"] + s["before"] + s["after"])]
        disturbed = np.zeros(len(link_ids), bool)
        for o in others:
            disturbed |= o["share"] >= OTHER_SHARE
        ok = ~s["closed_mask"] & (vol > 0)
        aff = ok & (s["delta"] >= share_thr * vol) & (s["delta"] >= AFFECTED_MIN)
        ctl = (ok & (vol >= CONTROL_MIN_VOL) & (s["share"] <= CONTROL_SHARE)
               & (np.abs(s["delta"]) <= CONTROL_ABS) & ~disturbed)
        return aff, ctl

    def rate(M, mask, days_):
        idx = [dpos[x] for x in days_]
        return M[mask][:, idx].sum() / len(idx)

    def closure_d(s, M, share_thr):
        control = s["before"] + s["after"]
        if len(s["during"]) < MIN_WEEKDAYS or len(control) < MIN_WEEKDAYS:
            return None
        aff, ctl = sets(s, share_thr)
        parts = []
        for grp in (True, False):
            a, c = aff & (main_link == grp), ctl & (main_link == grp)
            if a.sum() == 0 or c.sum() == 0:
                continue
            if rate(M, a, control) * len(control) < MIN_REPORTS or rate(M, c, control) * len(control) < MIN_REPORTS:
                continue
            ra = np.log((rate(M, a, s["during"]) + EPS) / (rate(M, a, control) + EPS))
            rc = np.log((rate(M, c, s["during"]) + EPS) / (rate(M, c, control) + EPS))
            parts.append((int(a.sum()), ra - rc, grp, a, c))
        if not parts:
            return None
        w = np.array([p[0] for p in parts], float)
        return {"D": float(np.dot(w, [p[1] for p in parts]) / w.sum()), "parts": parts,
                "affected": int(sum(p[3].sum() for p in parts)), "control": int(sum(p[4].sum() for p in parts))}

    rng = np.random.default_rng(SEED)

    def test(M, share_thr):
        per = {s["key"]: closure_d(s, M, share_thr) for s in specs}
        per = {k: v for k, v in per.items() if v is not None}
        D = np.array([v["D"] for v in per.values()])
        T = float(D.mean())
        flips = rng.choice([-1.0, 1.0], size=(N_FLIP, len(D)))
        p = float(((flips * np.abs(D)).mean(axis=1) >= T).mean())
        return {"T": T, "p": p, "closures": len(D), "positive": int((D > 0).sum()),
                "per_closure": {k: v["D"] for k, v in per.items()}}, per

    results = {}
    results["rush_hours_10pct"], per_main = test(P, AFFECTED_SHARE)
    results["all_hours_10pct"], _ = test(A, AFFECTED_SHARE)
    results["rush_hours_20pct"], _ = test(P, 0.20)

    # --- event study (rush hours, 10 %), weekly, relative to the weeks outside the closure
    def week_value(s, parts, first, last):
        days_ = [x for x in pd.date_range(first, last, freq="D") if x in dpos]
        if len(days_) < 3:
            return np.nan
        w = np.array([p[0] for p in parts], float)
        vals = [np.log(rate(P, a, days_) + EPS) - np.log(rate(P, c, days_) + EPS) for _, _, _, a, c in parts]
        return float(np.dot(w, vals) / w.sum())

    ev_rows = []
    for s in specs:
        v = per_main.get(s["key"])
        if v is None:
            continue
        st, en = s["start"], s["end"]
        start_w = {}
        for k in START_WEEKS:
            f = st + pd.Timedelta(days=7 * k)
            l = f + pd.Timedelta(days=6)
            if k >= 0 and l > en:
                continue
            start_w[k] = week_value(s, v["parts"], f, l)
        end_w = {}
        for k in END_WEEKS:
            if k < 0:
                l = en + pd.Timedelta(days=7 * (k + 1))
                f = l - pd.Timedelta(days=6)
                if f < st:
                    continue
            else:
                f = en + pd.Timedelta(days=1 + 7 * (k - 1))
                l = f + pd.Timedelta(days=6)
            end_w[k] = week_value(s, v["parts"], f, l)
        ref_s = np.nanmean([start_w.get(-2, np.nan), start_w.get(-1, np.nan)])
        ref_e = np.nanmean([end_w.get(1, np.nan), end_w.get(2, np.nan)])
        for k, x in start_w.items():
            ev_rows.append({"key": s["key"], "align": "start", "week": k, "value": x - ref_s})
        for k, x in end_w.items():
            ev_rows.append({"key": s["key"], "align": "end", "week": k, "value": x - ref_e})
    E = pd.DataFrame(ev_rows).dropna()
    E.to_csv(out / "event_study.csv", index=False)
    summ = E.groupby(["align", "week"])["value"].agg(["mean", "std", "count"]).reset_index()
    summ["se"] = summ["std"] / np.sqrt(summ["count"])
    results["event_study"] = summ.to_dict(orient="records")

    ink, muted, blue, grid = "#0b0b0b", "#52514e", "#2a78d6", "#d9d8d4"
    fig, axes = plt.subplots(1, 2, figsize=(7.13, 2.6), sharey=True)
    for ax, align, title, line_at in ((axes[0], "start", "aligned on the start", -0.5),
                                      (axes[1], "end", "aligned on the end", 0.0)):
        sm = summ[summ["align"] == align].sort_values("week")
        x = sm["week"].values.astype(float)
        m = 100 * (np.exp(sm["mean"].values) - 1)
        lo = 100 * (np.exp(sm["mean"].values - 1.96 * sm["se"].values) - 1)
        hi = 100 * (np.exp(sm["mean"].values + 1.96 * sm["se"].values) - 1)
        ax.axhline(0, color=grid, lw=1)
        ax.axvline(line_at, color=muted, lw=1, ls="--")
        ax.fill_between(x, lo, hi, color=blue, alpha=.15, lw=0)
        ax.plot(x, m, color=blue, lw=2, marker="o", ms=5)
        for xi, n in zip(x, sm["count"].values):
            ax.annotate(f"n={n}", (xi, lo.min() if len(lo) else 0), textcoords="offset points",
                        xytext=(0, -2), ha="center", va="top", fontsize=6.5, color=muted)
        ax.set_title(title, fontsize=8, color=ink)
        ax.set_xlabel("week relative to the closure", fontsize=7.5, color=ink)
        ax.tick_params(labelsize=7, colors=ink)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("rush-hour jams on affected links\nvs control links (%)", fontsize=7.5, color=ink)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"event_study.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    C = pd.DataFrame([{"key": s["key"], "source": s["source"],
                       "D": per_main[s["key"]]["D"] if s["key"] in per_main else np.nan,
                       "affected_links": per_main[s["key"]]["affected"] if s["key"] in per_main else np.nan,
                       "control_links": per_main[s["key"]]["control"] if s["key"] in per_main else np.nan}
                      for s in specs])
    C.to_csv(out / "closures.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(C.round(3).to_string(index=False))
    print()
    for name in ("rush_hours_10pct", "all_hours_10pct", "rush_hours_20pct"):
        r = results[name]
        print(f"{name:18s} {100 * (np.exp(r['T']) - 1):+.1f} %  p = {r['p']:.4f}  "
              f"positive {r['positive']} of {r['closures']}")
    print()
    print(summ.round(3).to_string(index=False))
    print("\nsaved to", out)


if __name__ == "__main__":
    sys.exit(main())
