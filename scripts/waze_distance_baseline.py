#!/usr/bin/env python3
"""Distance baseline for the Waze check (Section V-C of the article).

Would the streets around a closure, ordered only by their distance from the
closed links, predict the change in jam reports as well as the model does?

The script reuses the per-street table of ``waze_local_check.py`` (v5, with the
concurrent closures): the same closures, streets, observed changes y and the
same permutation test (y permuted within closure and road class group).  Only
the predictor changes:

* model     -- predicted relative change in volume (as in v5, for reference)
* distance  -- minus the distance of the street's nearest link from the closed
               links (closer = larger expected increase)
* model | distance -- partial Spearman correlation of the model with y after
               removing the rank-linear effect of distance from both

The street's distance is taken over the same links as in v5: links with its
name, not closed, within 2 km of the closed links.  Link geometry comes from
the running API; no scenario is run.

Usage::

    python scripts/waze_distance_baseline.py [--variant build_2026-09-23] [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detour_distance_baseline import load_geometry  # noqa: E402
from evaluate_detours import ROOT, Api, norm  # noqa: E402
from waze_local_check import AREA_M, N_PERM, SEED  # noqa: E402


def closed_links(ev, meta):
    out = {k: set(v["links"]) for k, v in meta["runs"].items()}
    for s in json.loads((ev / "waze_closures" / "closures.json").read_text(encoding="utf-8")):
        out[s["key"]] = set(s["links"])
    return out


def partial_rank(x, y, z):
    """Spearman correlation of x and y after removing the linear effect of rank(z)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))

    def resid(r):
        rz_c = rz - rz.mean()
        den = (rz_c ** 2).sum()
        b = (rz_c * (r - r.mean())).sum() / den if den > 0 else 0.0
        return r - r.mean() - b * rz_c

    ex, ey = resid(rx), resid(ry)
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else np.nan


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def perm_test(units, stat, col="y"):
    """Mean of *stat* over closures; one-sided p, y permuted within closure and road class group."""
    rng = np.random.default_rng(SEED)
    obs, null = [], np.zeros(N_PERM)
    for St in units:
        y, g = St[col].values, St["main"].values
        o = stat(St, y)
        if np.isnan(o):
            continue
        obs.append(o)
        order_all = np.tile(np.arange(len(y)), (N_PERM, 1))
        for grp in (True, False):
            idx = np.where(g == grp)[0]
            if len(idx) > 1:
                order = np.argsort(rng.random((N_PERM, len(idx))), axis=1)
                order_all[:, idx] = idx[order]
        null += np.array([stat(St, y[order_all[k]]) for k in range(N_PERM)])
    T = float(np.mean(obs))
    null /= len(obs)
    return {"T": T, "p": float((null >= T).mean()), "closures": len(obs),
            "positive": int(sum(o > 0 for o in obs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-23")
    ap.add_argument("--source", default="waze_v5", help="per-street table to reuse")
    args = ap.parse_args()

    ev = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    out = ev / f"{args.source}_distance"
    out.mkdir(exist_ok=True)
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))
    closed = closed_links(ev, meta)
    streets = pd.read_csv(ev / args.source / "streets.csv")

    base = pd.read_csv(ev / "links_dobrovskeho.csv").set_index("link_id")
    geoms = load_geometry(Api(args.api))
    missing = [i for i in base.index if i not in geoms and base.loc[i, "link_type"] != "centroid_connector"]
    print(f"geometry for {len(geoms):,} links; {len(missing)} links of the build without geometry", flush=True)

    link_ids = [i for i in base.index if i in geoms and base.loc[i, "link_type"] != "centroid_connector"]
    geo_arr = np.array([geoms[i] for i in link_ids], dtype=object)
    names_arr = np.array([norm(n) for n in base["name"].reindex(link_ids)])

    parts = []
    for key, St in streets.groupby("closure", sort=False):
        cl = closed[key]
        cg = shapely.union_all([geoms[i] for i in cl if i in geoms])
        dist = shapely.distance(geo_arr, cg)
        ok = (dist <= AREA_M) & ~np.isin(link_ids, list(cl))
        d = pd.Series(dist[ok]).groupby(names_arr[ok]).min()
        St = St.copy()
        St["dist_m"] = St["street"].map(lambda s: d.get(norm(s), np.nan))
        parts.append(St)
    S = pd.concat(parts)
    if S["dist_m"].isna().any():
        print("streets without distance:", S.loc[S["dist_m"].isna(), ["closure", "street"]].to_string())
    S["neg_dist"] = -S["dist_m"]
    units = [g for _, g in S.groupby("closure", sort=False)]

    rows = []
    for St in units:
        rows.append({"closure": St["closure"].iloc[0], "streets": len(St),
                     "rho_model": spearman(St["pred"], St["y"]),
                     "rho_distance": spearman(St["neg_dist"], St["y"]),
                     "rho_model_vs_distance": spearman(St["pred"], St["neg_dist"]),
                     "partial_model_given_distance": partial_rank(St["pred"], St["y"], St["neg_dist"]),
                     "median_dist_m": St["dist_m"].median()})
    R = pd.DataFrame(rows)

    results = {}
    for col in ("y", "y_rush"):
        results[col] = {
            "model": perm_test(units, lambda St, y: spearman(St["pred"].values, y), col),
            "distance": perm_test(units, lambda St, y: spearman(St["neg_dist"].values, y), col),
            "model_given_distance": perm_test(
                units, lambda St, y: partial_rank(St["pred"].values, y, St["neg_dist"].values), col),
        }
    results["mean_rho_model_vs_distance"] = float(R["rho_model_vs_distance"].mean())
    results["closures_model_beats_distance"] = int((R["rho_model"] > R["rho_distance"]).sum())

    S.to_csv(out / "streets.csv", index=False)
    R.to_csv(out / "closures.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    print()
    print(json.dumps(results, indent=1, default=float))
    print("\nsaved to", out)


if __name__ == "__main__":
    sys.exit(main())
