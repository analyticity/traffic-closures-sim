#!/usr/bin/env python3
"""Distance baseline for the detour evaluation of the article.

Would a ranking without any traffic model -- the loaded streets ordered by
their distance from the closed section -- place the prescribed detour streets
as high as the model does?  The script reads the per-link scenario results
that ``evaluate_detours.py`` saved (``links_<key>.csv``, ``meta.json``) and
takes only the link geometry from the running API, so no scenario is re-run.

It also reports how many streets and links without modelled traffic in the
null run gain some in the scenario (Section IV-A of the article), and the
probability of the observed number of closures with a prescribed street in the
top tenth if the streets were ranked at random.

Usage::

    python scripts/detour_distance_baseline.py [--variant build_2026-09-23]
        [--api http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.stats import rankdata
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_detours import (  # noqa: E402
    CSV_DIR, INTERVENTIONS, NULLS, ROOT, VARIANTS, Api, norm, rank_streets, resolve_name,
)

# The eight closures of the article, with the link-selection rule of Section V-A.
KEYS = ["dobrovskeho", "kralovopolsky", "olomoucka", "moravske", "jihlavska",
        "bauerova_ramps", "sladkova_i42", "hlavacova_frycajova"]
# Local projection, the same as evaluate_detours.dist_m.
M_PER_DEG_LON, M_PER_DEG_LAT = 73_000.0, 111_000.0


def job_for(key):
    for it in INTERVENTIONS:
        if it["key"] == key:
            return it
    v = VARIANTS[key]
    base = next(i for i in INTERVENTIONS if i["key"] == v["base"])
    return dict(base, key=key, label=v["label"])


def load_geometry(api):
    """link_id -> (Multi)LineString in metres."""
    geoms = {}
    for f in api.get("/api/links", min_volume=0)["features"]:
        g = f.get("geometry") or {}
        parts = g.get("coordinates") or []
        if g.get("type") == "LineString":
            parts = [parts]
        lines = [LineString([(x * M_PER_DEG_LON, y * M_PER_DEG_LAT) for x, y in p])
                 for p in parts if len(p) >= 2]
        if lines:
            geoms[int(f["properties"]["link_id"])] = (
                lines[0] if len(lines) == 1 else shapely.union_all(lines))
    return geoms


def p_best_in_top(n, k, share=0.10):
    """P(best of k randomly placed streets ranks within the top *share* of n)."""
    m = math.floor(share * n)
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def p_at_least(probs, hits):
    """Poisson-binomial tail P(X >= hits)."""
    total = 0.0
    for combo in product([0, 1], repeat=len(probs)):
        if sum(combo) >= hits:
            pr = 1.0
            for c, q in zip(combo, probs):
                pr *= q if c else (1.0 - q)
            total += pr
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-23")
    args = ap.parse_args()

    out = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    restr = pd.read_csv(CSV_DIR / "restrictions.csv", na_values=NULLS,
                        low_memory=False).set_index("id")
    geoms = load_geometry(Api(args.api))
    print(f"geometry for {len(geoms):,} links", flush=True)

    summary, detail, unloaded = [], [], []
    for key in KEYS:
        job = job_for(key)
        df = pd.read_csv(out / f"links_{key}.csv")
        closed_ids = meta["runs"][key]["links"]

        # Same street universe and model ranks as evaluate_detours.py.
        closed_names = {norm(n) for n in df.loc[df["link_id"].isin(closed_ids), "name"].dropna()}
        row = restr.loc[job["anchor"]]
        if isinstance(row.street_name, str):
            closed_names.add(norm(row.street_name))
        for x in job.get("exclude", []):
            closed_names.add(norm(x))
        g = rank_streets(df, closed_names)

        # Distance of each street from the closed section: its nearest link.
        closed = shapely.union_all([geoms[i] for i in closed_ids if i in geoms])
        d = df[df["name"].notna() & (df["link_type"] != "centroid_connector")]
        d = d[~d["name"].map(lambda n: norm(n) in closed_names)].copy()
        d["dist_m"] = shapely.distance(
            np.array([geoms.get(i) for i in d["link_id"]], dtype=object), closed)
        g["dist_m"] = d.groupby("name")["dist_m"].min().reindex(g.index).fillna(np.inf)
        loaded = g["modelled"]
        n_loaded = int(loaded.sum())
        g["rank_dist_loaded"] = np.nan
        g.loc[loaded, "rank_dist_loaded"] = rankdata(g.loc[loaded, "dist_m"].values,
                                                     method="average")

        model_names = set(df["name"].dropna())
        rows = []
        for name in job["prescribed"]:
            resolved = resolve_name(name, model_names)
            if resolved is None or resolved not in g.index or not g.loc[resolved, "modelled"]:
                continue
            r = g.loc[resolved]
            rows.append({"key": key, "street": name, "delta": round(r["delta"]),
                         "dist_m": round(r["dist_m"]),
                         "rank_model": r["rank_loaded"],
                         "pct_model": 100 * r["rank_loaded"] / n_loaded,
                         "rank_dist": r["rank_dist_loaded"],
                         "pct_dist": 100 * r["rank_dist_loaded"] / n_loaded})
        detail.extend(rows)
        best_m = min(rows, key=lambda r: r["rank_model"])
        best_d = min(rows, key=lambda r: r["rank_dist"])
        summary.append({
            "key": key, "label": job["label"], "n_loaded": n_loaded, "k": len(rows),
            "best_rank_model": best_m["rank_model"], "best_pct_model": best_m["pct_model"],
            "best_street_model": best_m["street"],
            "best_rank_dist": best_d["rank_dist"], "best_pct_dist": best_d["pct_dist"],
            "best_street_dist": best_d["street"],
            "streets_at_distance_0": int((g.loc[loaded, "dist_m"] == 0).sum()),
            "p_random_top10": p_best_in_top(n_loaded, len(rows)),
        })

        # Section IV-A: unloaded in the null run, some traffic in the scenario.
        lk = df[df["link_type"] != "centroid_connector"]
        ul = lk[lk["baseline"] <= 0]
        us = g[g["baseline"] <= 0]
        unloaded.append({
            "key": key,
            "links_unloaded": len(ul), "links_gain_gt0": int((ul["scenario"] > 0).sum()),
            "links_gain_gt1": int((ul["scenario"] > 1).sum()),
            "streets_unloaded": len(us), "streets_gain_gt0": int((us["scenario"] > 0).sum()),
            "streets_gain_gt1": int((us["scenario"] > 1).sum()),
        })

    S, D, U = pd.DataFrame(summary), pd.DataFrame(detail), pd.DataFrame(unloaded)
    S.to_csv(out / "distance_baseline.csv", index=False)
    D.to_csv(out / "distance_detail.csv", index=False)
    U.to_csv(out / "unloaded_gain.csv", index=False)

    hits_model = int((S["best_pct_model"] <= 10).sum())
    hits_dist = int((S["best_pct_dist"] <= 10).sum())
    probs = S["p_random_top10"].tolist()
    stats = {
        "closures": len(S),
        "model_hits_top10": hits_model,
        "distance_hits_top10": hits_dist,
        "random_expected_hits": sum(probs),
        "random_p_at_least_model_hits": p_at_least(probs, hits_model),
        "median_best_pct_model": float(S["best_pct_model"].median()),
        "median_best_pct_dist": float(S["best_pct_dist"].median()),
    }
    (out / "distance_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(S.drop(columns=["label"]).round(2).to_string(index=False))
    print()
    print(D.round(1).to_string(index=False))
    print()
    print(U.to_string(index=False))
    print()
    print(json.dumps(stats, indent=1))
    print("\nsaved to", out)


if __name__ == "__main__":
    sys.exit(main())
