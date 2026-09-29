#!/usr/bin/env python3
"""Closure scenarios against prescribed detours -- article evaluation, v2.

Re-implements sections 4-7 of ``notebooks/vyhodnotenie_uzavierok.ipynb`` as a
script against the running model API, with these differences:

* NDIC records that describe one intervention (identical description, staged
  works) are merged into ONE unit of analysis -- the notebook counted
  94509/94507/94511 (Bauerova ramps) and 94488/94482 (Moravské náměstí) twice.
* The prescribed street list is a manually verified table (``INTERVENTIONS``).
  The regex extractor of the notebook also picked up junction references
  ("od křižovatky ul. Pisárecká x ul. Žabovřeská ...") and direction
  references ("pro směr do ul. Lidická"), which are not detour streets.
* Ties get an average rank (streets with no modelled change used to be ranked
  by the arbitrary order of ``sort_values``) and percentiles are reported for
  two universes: all named streets, and streets that carry modelled traffic
  in the null run or in the scenario.
* A length-weighted ranking (delta vehicle-km) is computed next to the sum
  over links, as a robustness check of the aggregation.
* Two variants are run in addition to the eight interventions: Hlaváčova
  re-anchored on II/383 Fryčajova (the road the record actually closes), and
  the Bauerova ramps closed as ramps (``*_link`` links) instead of the tunnel
  section that the automatic rule selected.

Usage::

    python scripts/evaluate_detours.py [--only key,key] [--variant NAME]
        [--api http://localhost:8000]

Outputs go to ``simulation_for_article/<VARIANT>/vyhodnotenie_v2/`` (default variant
``build_2026-09-04`` = the docker image ``simulation-brno`` built on 4 Sep 2026, which is
the model the article's closure results come from -- NOT the frozen ``updated_version_12``).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data/data_vyhodnotenie_brno"
RADIUS_M = 400.0
NULLS = ["NULL", "null", "\\N"]

# ---------------------------------------------------------------------------
# Manually verified interventions.  ``records`` are all NDIC ids that describe
# the intervention; ``anchor`` is the record whose segment anchors the link
# selection (as in the notebook); ``prescribed`` are the streets the detour
# follows according to the description, without junction / direction
# references and without the closed street.
# ---------------------------------------------------------------------------
INTERVENTIONS = [
    {
        "key": "dobrovskeho",
        "label": "I/42 Dobrovského",
        "records": [94462],
        "anchor": 94462,
        "mode": "full",
        "prescribed": ["Provazníkova", "třída Generála Píky"],
        "note": "Objížďka: silnice I/42, Brno-Královo Pole - ulice Provazníkova, přes: třída Generála Píky",
    },
    {
        "key": "kralovopolsky",
        "label": "Královopolský tunnel",
        "records": [94502],
        "anchor": 94502,
        "mode": "full",
        # Žabovřeská is the closed road itself ("silnice I/42 (ulice Žabovřeská) ... tunel uzavřen")
        "prescribed": ["Hradecká", "Palackého třída", "Sportovní"],
        "exclude": ["Žabovřeská"],
        "note": "Objížďka: I/42 Žabovřeská, II/640 Hradecká, Palackého třída, I/43 Sportovní",
    },
    {
        "key": "olomoucka",
        "label": "Olomoucká",
        "records": [94415, 94417],
        "anchor": 94415,
        "mode": "full",
        "prescribed": ["Těžební", "Průmyslová", "Ostravská"],
        "note": "Objízdná trasa je vedena po ul. Těžební, Průmyslová, Ostravská",
    },
    {
        "key": "moravske",
        "label": "Moravské náměstí",
        "records": [94488, 94482],
        "anchor": 94488,
        "mode": "lanes",
        # Kounicova / Žerotínovo náměstí are junction references, Lidická a
        # direction reference ("pro směr do ul. Lidická"); Moravské náměstí is
        # the closed street.
        "prescribed": ["Koliště", "Rooseveltova", "Dvořákova", "Za divadlem", "Jezuitská"],
        "note": "Etapa I: ... dále ul. Moravské náměstí a ul. Koliště; pro směr na Koliště: Rooseveltova, Dvořákova, Za divadlem, Jezuitská",
    },
    {
        "key": "jihlavska",
        "label": "Jihlavská",
        "records": [94493],
        "anchor": 94493,
        "mode": "lanes",
        "prescribed": ["Kamenice", "Akademická", "Osová", "Pod Nemocnicí"],
        "note": "pro ul. Na Pískové cestě: Kamenice - Akademická - Jihlavská - Osová - Pod Nemocnicí; pro ul. Kamenice: Akademická",
    },
    {
        "key": "bauerova",
        "label": "I/42 Bauerova ramps",
        "records": [94509, 94507, 94511],
        "anchor": 94507,
        "mode": "full",
        # Pisárecká / Žabovřeská / Bauerova name the junction where the detour
        # starts, not streets it follows.
        "prescribed": ["Hlinky", "Veletržní", "Křížová", "Poříčí", "Rybnická", "Bítešská"],
        "note": "I: Hlinky, Veletržní, Křížová, Poříčí; II: Hlinky, Rybnická, Bítešská; III: text truncated",
    },
    {
        "key": "sladkova",
        "label": "Sládkova (I/42)",
        "records": [94460],
        "anchor": 94460,
        "mode": "full",
        "prescribed": ["Křižíkova", "Merhautova"],
        "note": "Objížďka: silnice I/42 (ulice Křižíkova) - silnice III/37915 (ulice Merhautova); record does not describe the closed section",
    },
    {
        "key": "hlavacova",
        "label": "Hlaváčova (II/383 Fryčajova)",
        "records": [94342],
        "anchor": 94342,
        "mode": "full",
        "prescribed": ["Rokytova", "Žarošická", "Jedovnická"],
        "note": "in-network part of the detour; the rest runs via Ochoz, Kanice, Řícmanice, Bílovice outside the network",
    },
]

VARIANTS = {
    # The record closes II/383 Fryčajova; the automatic rule anchored on the
    # segment named Hlaváčova.  Re-anchor on links named Fryčajova / ref 383
    # within 600 m of the segment.
    "hlavacova_frycajova": {
        "base": "hlavacova",
        "label": "Hlaváčova re-anchored on II/383 Fryčajova",
        "select": {"names": ["Fryčajova"], "refs": ["383"], "radius": 600.0},
    },
    # Ramps only: *_link links with ref 42 around both anchors of the staged
    # works (segments 15711 and 15718).
    "bauerova_ramps": {
        "base": "bauerova",
        "label": "I/42 Bauerova ramps closed as ramps",
        "select": {"refs": ["42"], "types": ["trunk_link", "motorway_link", "primary_link"],
                   "segments": [15711, 15718], "radius": 400.0},
    },
    # Mainline representation of the same works (what the notebook did for
    # records 94509/94511: every link with ref 42 within 400 m).
    "bauerova_mainline": {
        "base": "bauerova",
        "label": "I/42 Bauerova ramps closed as the I/42 carriageway",
        "select": {"refs": ["42"], "segments": [15718], "radius": 400.0},
    },
    # The record's header says "silnice I/42"; the record field and the
    # segment carry no road number, so the notebook fell back to the six
    # nearest links (service roads).  Anchor on I/42 instead.
    "sladkova_i42": {
        "base": "sladkova",
        "label": "Sládkova anchored on I/42 from the description header",
        "select": {"refs": ["42"], "radius": 400.0},
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def norm(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c)).strip()


def dist_m(lat1, lon1, lat2, lon2):
    return math.hypot((lon2 - lon1) * 73_000, (lat2 - lat1) * 111_000)


def wkb_centroid(hexstr):
    """Centroid of a WKB (Multi)LineString/Point with SRID header, no shapely."""
    if not isinstance(hexstr, str) or not hexstr:
        return None
    b = bytes.fromhex(hexstr)
    pos = 0

    def read_geom(pos):
        order = b[pos]
        pos += 1
        fmt = "<" if order == 1 else ">"
        gtype = struct.unpack(fmt + "I", b[pos:pos + 4])[0]
        pos += 4
        if gtype & 0x20000000:
            pos += 4  # SRID
        gtype &= 0xFF
        pts = []
        if gtype == 1:
            x, y = struct.unpack(fmt + "dd", b[pos:pos + 16])
            pos += 16
            pts.append((x, y))
        elif gtype == 2:
            n = struct.unpack(fmt + "I", b[pos:pos + 4])[0]
            pos += 4
            for _ in range(n):
                x, y = struct.unpack(fmt + "dd", b[pos:pos + 16])
                pos += 16
                pts.append((x, y))
        elif gtype in (4, 5, 7):
            n = struct.unpack(fmt + "I", b[pos:pos + 4])[0]
            pos += 4
            for _ in range(n):
                sub, pos = read_geom(pos)
                pts.extend(sub)
        else:
            raise ValueError(f"unsupported wkb type {gtype}")
        return pts, pos

    pts, _ = read_geom(pos)
    if not pts:
        return None
    return (sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts))  # (lat, lon)


class Api:
    def __init__(self, base):
        self.base = base

    def get(self, path, **params):
        url = f"{self.base}{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        with urllib.request.urlopen(url, timeout=900) as r:
            return json.load(r)

    def post(self, path, payload):
        req = urllib.request.Request(f"{self.base}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)


def load_links(api):
    gj = api.get("/api/links", min_volume=0)
    links = []
    for f in gj["features"]:
        p, g = f["properties"], (f.get("geometry") or {})
        cs = g.get("coordinates") or []
        if g.get("type") == "MultiLineString":
            cs = [c for part in cs for c in part]
        if not cs:
            continue
        lats = [c[1] for c in cs]
        lons = [c[0] for c in cs]
        links.append({
            "link_id": int(p["link_id"]), "name": p.get("name"),
            "ref": str(p.get("osm_ref") or ""), "type": p.get("link_type"),
            "lanes": p.get("lanes"), "vol": float(p.get("wd_daily_tot") or 0),
            "distance": float(p.get("distance") or 0),
            "centre": (sum(lats) / len(lats), sum(lons) / len(lons)),
        })
    return links


def select_links(links, seg_centre, refs=None, names=None, types=None, radius=RADIUS_M):
    """Notebook rule: links within *radius* of the anchor with the same road
    ref (or, without a ref, the same name); fallback = 6 nearest links."""
    near = [(dist_m(l["centre"][0], l["centre"][1], seg_centre[0], seg_centre[1]), l) for l in links]
    near = [(d, l) for d, l in near if d <= radius]
    refs = {r for r in (refs or []) if r}
    names = {norm(n) for n in (names or []) if n}
    if refs:
        sel = [l for d, l in near if l["ref"] in refs]
        how = f"road ref {'/'.join(sorted(refs))}"
    elif names:
        sel = [l for d, l in near if norm(l["name"]) in names]
        how = f"segment name {'/'.join(sorted(names))}"
    else:
        sel = []
        how = ""
    if types:
        sel = [l for l in sel if l["type"] in types]
        how += f" restricted to {','.join(types)}"
    if not sel and not types:
        sel = [l for d, l in sorted(near, key=lambda t: t[0])[:6]]
        how = "nearest links (no ref/name match)"
    return sel, how


def run_scenario(api, links, mode):
    payload = []
    for l in links:
        n = max(int(float(l["lanes"] or 2)), 1)
        payload.append({"link_id": l["link_id"], "direction": "both",
                        "closure_type": mode, "lanes": n,
                        "lanes_remaining": (n - 1 if mode == "lanes" and n > 1 else 0)})
    jid = api.post("/api/scenarios/run", {"links": payload})["id"]
    while True:
        st = api.get(f"/api/scenarios/{jid}/status")
        if st["status"] == "done":
            break
        if st["status"] == "error":
            raise RuntimeError(st.get("error"))
        time.sleep(3)
    res = api.get(f"/api/scenarios/{jid}/results")
    rows = []
    for f in res["features"]:
        p = f["properties"]
        rows.append({"link_id": int(p["link_id"]), "name": p.get("name"),
                     "link_type": p.get("link_type"),
                     "delta": float(p.get("delta_vol") or 0),
                     "baseline": float(p.get("baseline_vol") or 0),
                     "scenario": float(p.get("wd_daily_tot") or 0),
                     "distance": float(p.get("distance") or 0)})
    return pd.DataFrame(rows), st.get("delta_summary", {})


def resolve_name(name, model_names):
    """Exact name, else the 'třída' variants, else None."""
    if name in model_names:
        return name
    for cand in (f"{name} třída", f"třída {name}", name.replace(" třída", ""), name.replace("třída ", "")):
        if cand in model_names:
            return cand
    return None


def rank_streets(df, closed_names):
    """Street-level aggregation and tie-aware ranks in two universes."""
    from scipy.stats import rankdata

    d = df[df["name"].notna() & (df["link_type"] != "centroid_connector")].copy()
    d = d[~d["name"].map(lambda n: norm(n) in closed_names)]
    d["vkm"] = d["delta"] * d["distance"] / 1000.0
    g = d.groupby("name").agg(delta=("delta", "sum"), vkm=("vkm", "sum"),
                              baseline=("baseline", "sum"), scenario=("scenario", "sum"),
                              links=("delta", "size"))
    g["rank_all"] = rankdata(-g["delta"].values, method="average")
    g["rank_vkm_all"] = rankdata(-g["vkm"].values, method="average")
    loaded = (g["baseline"] > 0) | (g["scenario"] > 0)
    g["modelled"] = loaded
    g["rank_loaded"] = np.nan
    g.loc[loaded, "rank_loaded"] = rankdata(-g.loc[loaded, "delta"].values, method="average")
    g["rank_vkm_loaded"] = np.nan
    g.loc[loaded, "rank_vkm_loaded"] = rankdata(-g.loc[loaded, "vkm"].values, method="average")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--variant", default="build_2026-09-04")
    ap.add_argument("--only", default="", help="comma-separated keys (interventions or variants)")
    ap.add_argument("--skip-variants", action="store_true")
    args = ap.parse_args()

    api = Api(args.api)
    out = ROOT / "simulation_for_article" / args.variant / "vyhodnotenie_v2"
    out.mkdir(parents=True, exist_ok=True)

    meta = api.get("/api/meta")
    val = api.get("/api/reports/validation")
    hv = (val.get("benchmarks") or {}).get("holdout_validation") or {}
    print(f"API {args.api}: {meta.get('place_name')}; holdout r2={hv.get('r2')} slope={hv.get('slope')} "
          f"pct_rmse={hv.get('pct_rmse')}", flush=True)

    links = load_links(api)
    model_names = {l["name"] for l in links if l["name"]}
    print(f"links {len(links):,} | named streets {len(model_names):,}", flush=True)

    restr = pd.read_csv(CSV_DIR / "restrictions.csv", na_values=NULLS, low_memory=False).set_index("id")
    segs = pd.read_csv(CSV_DIR / "road_segments.csv", na_values=NULLS, low_memory=False).set_index("id")

    def anchor_of(rec_id):
        row = restr.loc[rec_id]
        seg = segs.loc[int(row.segment_id)]
        centre = wkb_centroid(seg.geog)
        refs = [str(x).strip() for x in (seg.road_ref, row.road_number)
                if isinstance(x, str) and str(x).strip()]
        # road_number may be numeric in the CSV
        for x in (seg.road_ref, row.road_number):
            if isinstance(x, (int, float)) and not pd.isna(x):
                refs.append(str(int(x)))
        names = [seg["name"]] if isinstance(seg["name"], str) else []
        return centre, refs, names, row

    jobs = []
    for it in INTERVENTIONS:
        jobs.append(dict(it, variant=None))
    if not args.skip_variants:
        for vkey, v in VARIANTS.items():
            base = next(i for i in INTERVENTIONS if i["key"] == v["base"])
            jobs.append(dict(base, key=vkey, label=v["label"], variant=v))
    if args.only:
        keep = {k.strip() for k in args.only.split(",")}
        jobs = [j for j in jobs if j["key"] in keep]

    summary, detail, meta_out = [], [], {"api": args.api, "holdout": hv, "links": len(links),
                                         "named_streets": len(model_names), "runs": {}}
    for job in jobs:
        key = job["key"]
        centre, refs, names, row = anchor_of(job["anchor"])
        v = job.get("variant")
        if v and v["select"].get("segments"):
            centres = [wkb_centroid(segs.loc[s].geog) for s in v["select"]["segments"]]
        else:
            centres = [centre]
        sel, how = [], ""
        for c in centres:
            if v:
                s, how = select_links(links, c, refs=v["select"].get("refs"),
                                      names=v["select"].get("names"),
                                      types=v["select"].get("types"),
                                      radius=v["select"].get("radius", RADIUS_M))
            else:
                s, how = select_links(links, c, refs=refs, names=names)
            sel.extend(x for x in s if x["link_id"] not in {y["link_id"] for y in sel})
        closed_vol = sum(l["vol"] for l in sel)
        if not sel:
            print(f"!! {key}: no links selected", flush=True)
            continue
        t0 = time.time()
        df, ds = run_scenario(api, sel, job["mode"])
        el = time.time() - t0
        df.to_csv(out / f"links_{key}.csv", index=False)

        closed_names = {norm(l["name"]) for l in sel if l["name"]}
        if isinstance(row.street_name, str):
            closed_names.add(norm(row.street_name))
        for x in job.get("exclude", []):
            closed_names.add(norm(x))
        g = rank_streets(df, closed_names)
        n_all, n_loaded = len(g), int(g["modelled"].sum())
        g.sort_values("delta", ascending=False).head(60).to_csv(out / f"streets_{key}.csv")

        rows = []
        for name in job["prescribed"]:
            resolved = resolve_name(name, model_names)
            if resolved is None or resolved not in g.index:
                rows.append({"key": key, "street": name, "in_network": False})
                continue
            r = g.loc[resolved]
            rows.append({"key": key, "street": name, "resolved": resolved, "in_network": True,
                         "modelled": bool(r["modelled"]), "delta": round(r["delta"]),
                         "delta_vkm": round(r["vkm"]), "baseline": round(r["baseline"]),
                         "rank_all": r["rank_all"], "pct_all": 100 * r["rank_all"] / n_all,
                         "rank_loaded": r["rank_loaded"],
                         "pct_loaded": (100 * r["rank_loaded"] / n_loaded) if r["modelled"] else np.nan,
                         "rank_vkm_all": r["rank_vkm_all"], "rank_vkm_loaded": r["rank_vkm_loaded"]})
        detail.extend(rows)
        inn = [r for r in rows if r["in_network"]]
        mod = [r for r in inn if r["modelled"]]
        s = {
            "key": key, "label": job["label"], "records": "+".join(map(str, job["records"])),
            "mode": job["mode"], "links_closed": len(sel), "closed_volume": round(closed_vol),
            "selection": how, "sum_abs_delta": round(ds.get("sum_abs_delta_vol", 0)),
            "n_streets_all": n_all, "n_streets_loaded": n_loaded,
            "prescribed": len(job["prescribed"]), "in_network": len(inn), "modelled": len(mod),
            "no_change": sum(1 for r in inn if r["delta"] == 0),
            "losing": sum(1 for r in inn if r["delta"] < 0),
            "best_rank_all": min(r["rank_all"] for r in inn) if inn else np.nan,
            "median_rank_all": float(np.median([r["rank_all"] for r in inn])) if inn else np.nan,
            "best_pct_all": min(r["pct_all"] for r in inn) if inn else np.nan,
            "median_pct_all": float(np.median([r["pct_all"] for r in inn])) if inn else np.nan,
            "best_rank_loaded": min(r["rank_loaded"] for r in mod) if mod else np.nan,
            "best_pct_loaded": min(r["pct_loaded"] for r in mod) if mod else np.nan,
            "median_pct_loaded": float(np.median([r["pct_loaded"] for r in mod])) if mod else np.nan,
            "top100_all": sum(1 for r in inn if r["rank_all"] <= 100),
            "top10pct_loaded": sum(1 for r in mod if r["pct_loaded"] <= 10),
            "best_rank_vkm_all": min(r["rank_vkm_all"] for r in inn) if inn else np.nan,
            "top100_vkm_all": sum(1 for r in inn if r["rank_vkm_all"] <= 100),
            "seconds": round(el),
        }
        summary.append(s)
        meta_out["runs"][key] = {"links": [l["link_id"] for l in sel], "selection": how,
                                 "closed_volume": round(closed_vol), "delta_summary": ds}
        print(f"✓ {key}: {len(sel)} links ({how}), {closed_volume_fmt(closed_vol)} veh/day closed, "
              f"mode {job['mode']}, {el:.0f}s | streets {n_all}/{n_loaded} loaded | "
              f"best {s['best_rank_all']} | top100 {s['top100_all']}/{len(inn)} | "
              + ", ".join(f"{r['street']}={r['rank_all'] if r['in_network'] else 'n/a'}" for r in rows),
              flush=True)

    S = pd.DataFrame(summary)
    D = pd.DataFrame(detail)
    S.to_csv(out / "summary.csv", index=False)
    D.to_csv(out / "detail.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta_out, indent=1, ensure_ascii=False), encoding="utf-8")

    # --- statistics over the eight interventions (variants excluded) ---
    main_keys = [i["key"] for i in INTERVENTIONS]
    M = S[S["key"].isin(main_keys)]
    if len(M):
        from scipy.stats import wilcoxon
        stats = {}
        for col in ("best_pct_all", "median_pct_all", "best_pct_loaded", "median_pct_loaded"):
            v = M[col].dropna().values
            if len(v) >= 5:
                p = wilcoxon(v - 50, alternative="less").pvalue
                stats[col] = {"n": int(len(v)), "median": float(np.median(v)), "p_one_sided": float(p)}
        n_in = int(M["in_network"].sum())
        n_mod = int(M["modelled"].sum())
        stats["top100_all"] = {"hits": int(M["top100_all"].sum()), "streets": n_in,
                               "expected_random": float((M["in_network"] * 100 / M["n_streets_all"]).sum())}
        stats["top10pct_loaded"] = {"hits": int(M["top10pct_loaded"].sum()), "streets": n_mod,
                                    "expected_random": float(0.10 * n_mod)}
        stats["closures_with_hit_top100"] = int((M["best_rank_all"] <= 100).sum())
        stats["closures"] = int(len(M))
        (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print("\n== statistics (8 interventions) ==")
        print(json.dumps(stats, indent=1))
    print("\nsaved to", out)


def closed_volume_fmt(v):
    return f"{v:,.0f}"


if __name__ == "__main__":
    sys.exit(main())
