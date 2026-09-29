#!/usr/bin/env python3
"""Render the closure-impact figure from a frozen result set.

Why this exists as a script and not a notebook cell: the figure for the paper
has to be regenerable from the archived artefacts alone, without a running API
or container.  It reads ``simulation_for_article/<version>/`` and writes a
self-contained HTML next to it.

**What is different from the notebook map.**  The notebook draws a pure
difference map: width = |delta|, colour = sign, links below the threshold are
not drawn at all.  That hides the thing a reader needs most -- how big the road
was in the first place.  A +150 veh/day change on a service street and on a
motorway come out identical, and every unchanged link disappears, so there is
no way to tell whether traffic moved onto a corridor or into a side street.

Here the two variables are separated, which is the usual convention for flow
maps in transport planning:

    width  = baseline volume   (how important is this road)
    colour = what happened to it

Unchanged links stay on the map in grey, so the network context survives.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("make_article_maps")

ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = ROOT / "simulation_for_article"

#: Links below this baseline volume are left off the map entirely -- at city
#: scale they are visual noise, not context.
DEFAULT_MIN_VOLUME = 750.0
#: Below this the change is treated as "no meaningful change" and drawn grey.
DEFAULT_MIN_DELTA = 100.0
#: A relative change is only meaningful once the link carries something.
REL_VOLUME_FLOOR = 500.0

#: CARTO started requiring an API key and now stamps "API KEY REQUIRED" across
#: every tile, which lands in the figure.  Plain OSM needs no key; a greyscale
#: filter brings it close to the light basemap the figure was designed around.
BASEMAPS = {
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "© OpenStreetMap",
        "grey": True,
    },
    "carto": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": "© OpenStreetMap, © CARTO",
        "grey": False,
    },
    "none": {"url": None, "attr": "", "grey": False},
}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def geometry_from_maps(version_dir: Path) -> Dict[int, List[List[float]]]:
    """Recover ``link_id -> [[lon, lat], ...]`` from previously rendered maps.

    The archived parquets carry no geometry (it came from the API at render
    time), and the on-disk network export belongs to a different build whose
    link ids were renumbered.  The maps written alongside the parquets are
    therefore the only geometry that is guaranteed to match this result set.
    """
    geom: Dict[int, List[List[float]]] = {}
    for html in sorted(version_dir.glob("mapa*.html")):
        text = html.read_text(encoding="utf-8")
        match = re.search(r"const D\s*=\s*(\{.*?\});\s*\n", text, re.S)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("%s: embedded data is not valid JSON, skipped", html.name)
            continue
        for link in payload.get("hrany", []):
            geom.setdefault(int(link["id"]), link["c"])
    return geom


def gateways_from_maps(version_dir: Path) -> List[Dict[str, Any]]:
    """Recover the gateway markers (entry points of external demand)."""
    for html in sorted(version_dir.glob("mapa*.html")):
        match = re.search(r"const D\s*=\s*(\{.*?\});\s*\n", html.read_text(encoding="utf-8"), re.S)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if payload.get("brany"):
            return payload["brany"]
    return []


def geometry_from_geojson(path: Path, wanted: set) -> Dict[int, List[List[float]]]:
    """Read geometry from a network export, if its link ids match this run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    geom: Dict[int, List[List[float]]] = {}
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        lid = props.get("link_id")
        g = feat.get("geometry") or {}
        if lid is None or not g.get("coordinates"):
            continue
        coords = g["coordinates"]
        if g.get("type") == "MultiLineString":
            coords = [c for part in coords for c in part]
        geom[int(lid)] = [[round(x, 5), round(y, 5)] for x, y in coords]

    overlap = len(set(geom) & wanted) / max(len(wanted), 1)
    if overlap < 0.5:
        raise SystemExit(
            f"{path.name} matches only {overlap:.0%} of this run's link ids — "
            "it belongs to a different network build. Omit --geometry to read "
            "the geometry out of the archived maps instead."
        )
    return geom


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)


def build_payload(
    version_dir: Path,
    scenario: str,
    geom: Dict[int, List[List[float]]],
    min_volume: float,
    min_delta: float,
) -> Dict[str, Any]:
    scen_path = version_dir / f"scenario_{scenario}.parquet"
    if not scen_path.exists():
        available = sorted(p.stem.replace("scenario_", "") for p in version_dir.glob("scenario_*.parquet"))
        raise SystemExit(f"{scen_path.name} not found. Available: {', '.join(available) or 'none'}")

    df = pd.read_parquet(scen_path)
    base = _num(df, "baseline_vol")
    scen = _num(df, "wd_daily_tot")
    delta = scen - base

    meta = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    missing = 0
    for i, lid in enumerate(df["link_id"].astype(int)):
        b, s, d = float(base.iat[i]), float(scen.iat[i]), float(delta.iat[i])
        # Keep a link if it matters either as context or as an effect.
        if b < min_volume and abs(d) < min_delta:
            continue
        coords = geom.get(lid)
        if coords is None:
            missing += 1
            continue
        name = df["name"].iat[i] if "name" in df.columns else None
        rows.append({
            "id": lid,
            "n": (str(name) if isinstance(name, str) and name.strip() else "(bez názvu)"),
            "t": str(df["link_type"].iat[i]) if "link_type" in df.columns else "",
            "b": round(b, 1),
            "s": round(s, 1),
            "los": str(df["LOS_max"].iat[i]) if "LOS_max" in df.columns else "",
            "c": coords,
        })

    if missing:
        logger.warning("%d link(s) had no geometry and were dropped", missing)
    if not rows:
        raise SystemExit("nothing to draw — check --min-volume / --min-delta")

    lats = [p[1] for r in rows for p in r["c"]]
    lons = [p[0] for r in rows for p in r["c"]]

    closed = [r["id"] for r in rows if r["b"] >= min_delta and r["s"] < 1]
    n_changed = sum(1 for r in rows if abs(r["s"] - r["b"]) >= min_delta)

    return {
        "hrany": rows,
        "stred": [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2],
        "zavrete_ids": closed,
        "min_delta": min_delta,
        "scenar": scenario,
        "mesto": (meta.get("_meta") or {}).get("mesto", ""),
        "variant": (meta.get("_meta") or {}).get("variant", version_dir.name),
        "n_zmenenych": n_changed,
        "sigma_delta": round(sum(abs(r["s"] - r["b"]) for r in rows)),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

_TEMPLATE = """<!doctype html><html lang="sk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 :root{--bg:#fff;--fg:#16181d;--mut:#6b7280;--line:#e3e6ea;--panel:rgba(255,255,255,.97)}
 *{box-sizing:border-box} html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
 #map{position:absolute;inset:0}
 .box{position:absolute;z-index:1000;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;box-shadow:0 6px 24px rgba(0,0,0,.16);backdrop-filter:blur(8px)}
 #ovl{top:14px;left:14px;padding:14px 16px;max-width:330px}
 #leg{bottom:14px;left:14px;padding:12px 14px;min-width:250px}
 h1{margin:0 0 2px;font-size:15px;font-weight:650;letter-spacing:-.01em}
 .sub{color:var(--mut);font-size:12px;margin-bottom:10px}
 label.f{display:block;font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:var(--mut);margin:10px 0 5px}
 select{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:#fff;font:inherit}
 .k{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px}
 .sw{width:26px;height:4px;border-radius:2px;flex:none}
 .hint{font-size:11px;color:var(--mut);margin-top:8px;line-height:1.35}
 .stat{font-size:12px;margin-top:8px;padding-top:8px;border-top:1px solid var(--line)}
 .stat b{font-variant-numeric:tabular-nums}
 #none{display:none;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:1200;
  padding:18px 22px;text-align:center;max-width:340px}
 #none b{display:block;font-size:15px;margin-bottom:4px}
 __GREY__
</style></head><body>
<div id="map"></div>
<div id="ovl" class="box">
 <h1>__H1__</h1>
 <div class="sub">__SUB__</div>
 <label class="f" for="m">Čo zobraziť farbou</label>
 <select id="m">
  <option value="dopad">Dopad uzávierky (absolútne)</option>
  <option value="rel">Dopad uzávierky (relatívne)</option>
  <option value="stav">Základný stav — úroveň kvality (LOS)</option>
 </select>
 <div class="stat" id="st"></div>
 <div class="hint">Hrúbka čiary vždy zodpovedá <b>objemu v základnom stave</b>,
  takže hlavné ťahy zostávajú rozoznateľné aj tam, kde sa nič nezmenilo.</div>
</div>
<div id="leg" class="box"></div>
<div id="none" class="box"><b>Žiadna zmena</b>
 Ani jedna hrana sa nezmenila o viac než __MINDELTA__ voz/deň. Pri null scenári
 je to očakávaný výsledok — je to zmerané šumové dno priradenia.</div>
<script>
const D = __DATA__;
const map = L.map('map',{preferCanvas:true,zoomControl:true}).setView(D.stred, 12);
__BASEMAP_JS__
const g = L.layerGroup().addTo(map);
const gGw = L.layerGroup().addTo(map);

(D.brany||[]).forEach(b => {
  L.circleMarker([b.lat, b.lon], {radius:7, color:'#7a5b00', weight:1.5,
      fillColor:'#fbbf24', fillOpacity:.95})
   .bindTooltip(`<b>${b.brana}</b><br>${b.ref||''}<br>${nf(b.objem)} voz/deň`, {sticky:true})
   .addTo(gGw);
});

const nf = n => Math.round(n).toLocaleString('sk-SK').replace(/ /g,' ');
const kvantil = (a,q) => { const b=a.filter(Number.isFinite).sort((x,y)=>x-y);
  return b.length ? b[Math.min(b.length-1, Math.floor(q*b.length))] : 0; };

const NEUTRAL = '#b9c0c9';                    // bez podstatnej zmeny
const UP   = ['#f2b8b5','#e57373','#d32f2f','#8e1b16'];   // pribudlo
const DOWN = ['#b3d4e8','#6aaed6','#2b7bba','#12496e'];   // ubudlo
const CLOSED = '#7b3294';
const LOS = {A:'#2c7fb8',B:'#41b6c4',C:'#a1dab4',D:'#fecc5c',E:'#fd8d3c',F:'#e31a1c'};

// Sirka nesie objem v zakladnom stave. Odmocnina, aby diaľnica neprehlusila
// zvysok siete tak, ako by to spravil linearny prevod.
let WMAX = 1;
const sirka = b => 0.8 + 5.6 * Math.sqrt(Math.min(1, b / WMAX));

function odtien(ramp, t){ return ramp[Math.min(ramp.length-1, Math.floor(t*ramp.length))]; }

function farba(mode, l, dmax, rmax){
 const d = l.s - l.b;
 if (mode === 'stav') return LOS[l.los] || NEUTRAL;
 if (D.zavrete_ids.includes(l.id)) return CLOSED;
 if (Math.abs(d) < D.min_delta) return NEUTRAL;
 if (mode === 'rel'){
   if (l.b < __RELFLOOR__) return NEUTRAL;
   const t = Math.min(1, Math.abs(d / l.b) / (rmax || 1));
   return odtien(d > 0 ? UP : DOWN, t);
 }
 const t = Math.min(1, Math.abs(d) / (dmax || 1));
 return odtien(d > 0 ? UP : DOWN, t);
}

function kresli(){
 const mode = document.getElementById('m').value;
 g.clearLayers();
 WMAX = kvantil(D.hrany.map(l => l.b), 0.98) || 1;
 const dmax = kvantil(D.hrany.map(l => Math.abs(l.s - l.b)).filter(v => v >= D.min_delta), 0.90) || 1;
 const rmax = kvantil(D.hrany.filter(l => l.b >= __RELFLOOR__)
                             .map(l => Math.abs((l.s - l.b) / l.b)), 0.90) || 1;

 // Nezmenene hrany kreslime prve, aby zmena zostala navrchu a bola citatelna.
 const poradie = [...D.hrany].sort((a,b) =>
   Math.abs(a.s - a.b) - Math.abs(b.s - b.b));

 poradie.forEach(l => {
  const d = l.s - l.b, c = farba(mode, l, dmax, rmax);
  const tlmene = (c === NEUTRAL);
  L.polyline(l.c.map(p => [p[1], p[0]]),
    {color:c, weight:sirka(l.b), opacity: tlmene ? .38 : .9, lineCap:'round'})
   .bindTooltip(`<b>${l.n}</b><span style="opacity:.6"> · ${l.t}</span>`
     + `<br>${nf(l.b)} → ${nf(l.s)} voz/deň`
     + (Math.abs(d) >= 1 ? `<br><b>${d > 0 ? '+' : ''}${nf(d)}</b>`
          + (l.b >= 1 ? ` (${d > 0 ? '+' : ''}${(100*d/l.b).toFixed(0)} %)` : '') : '<br>bez zmeny')
     , {sticky:true})
   .addTo(g);
 });
 legenda(mode, dmax, rmax);
 document.getElementById('none').style.display =
   (mode !== 'stav' && D.n_zmenenych === 0) ? 'block' : 'none';
}

function legenda(mode, dmax, rmax){
 const el = document.getElementById('leg');
 const riadok = (c,t) => `<div class="k"><span class="sw" style="background:${c}"></span>${t}</div>`;
 let h = '';
 if (mode === 'stav'){
   h += '<div class="sub" style="margin:0 0 6px">Farba — úroveň kvality dopravy</div>';
   ['A','B','C','D','E','F'].forEach(k => h += riadok(LOS[k], k));
 } else {
   const jed = mode === 'rel' ? '%' : ' voz/deň';
   const m = mode === 'rel' ? (100*rmax).toFixed(0) : nf(dmax);
   h += `<div class="sub" style="margin:0 0 6px">Farba — zmena oproti základnému stavu</div>`;
   h += riadok(UP[2],   `nárast (do ${m}${jed})`);
   h += riadok(DOWN[2], `pokles (do ${m}${jed})`);
   h += riadok(NEUTRAL, `bez podstatnej zmeny (&lt; ${nf(D.min_delta)} voz/deň)`);
   if (D.zavrete_ids.length) h += riadok(CLOSED, 'vyprázdnená / zavretá');
 }
 h += '<div class="k" style="margin-top:9px">'
    + '<span class="sw" style="background:#111;height:1.2px"></span>'
    + '<span class="sw" style="background:#111;height:6px"></span>'
    + '<span style="font-size:11px;color:var(--mut)">hrúbka = objem v základnom stave</span></div>';
 el.innerHTML = h;
}

document.getElementById('st').innerHTML =
  `Hrán na mape: <b>${nf(D.hrany.length)}</b><br>`
  + `so zmenou ≥ ${nf(D.min_delta)} voz/deň: <b>${nf(D.n_zmenenych)}</b><br>`
  + `celková redistribúcia: <b>${nf(D.sigma_delta)}</b> voz/deň`;
document.getElementById('m').addEventListener('change', kresli);
document.getElementById('gw').addEventListener('change', e =>
  e.target.checked ? gGw.addTo(map) : map.removeLayer(gGw));
kresli();
</script></body></html>
"""


def render(payload: Dict[str, Any], out_path: Path, rel_floor: float,
           basemap: str = "osm") -> None:
    spec = BASEMAPS[basemap]
    if spec["url"]:
        tile_js = (f"L.tileLayer('{spec['url']}',\n"
                   f" {{maxZoom:19, attribution:'{spec['attr']}'}}).addTo(map);")
    else:
        tile_js = "// bez podkladovej mapy"
    grey_css = (".leaflet-tile-pane{filter:grayscale(1) contrast(.92) brightness(1.06)}"
                if spec["grey"] else "")

    title = f"{payload['mesto']} — {payload['scenar']}"
    html = (_TEMPLATE
            .replace("__BASEMAP_JS__", tile_js)
            .replace("__GREY__", grey_css)
            .replace("__TITLE__", title)
            .replace("__H1__", f"Dopad uzávierky — {payload['scenar']}")
            .replace("__SUB__", f"{payload['mesto']} · {payload['variant']}")
            .replace("__MINDELTA__", f"{payload['min_delta']:.0f}")
            .replace("__RELFLOOR__", str(rel_floor))
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="updated_version_12",
                    help="folder under simulation_for_article/")
    ap.add_argument("--scenario", default="Jihlavská",
                    help="scenario name, matching scenario_<name>.parquet")
    ap.add_argument("--geometry", type=Path, default=None,
                    help="optional network_links.geojson; by default the geometry "
                         "is recovered from the archived maps of the same run")
    ap.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME)
    ap.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    ap.add_argument("--basemap", choices=sorted(BASEMAPS), default="osm",
                    help="tile provider; 'carto' now watermarks every tile with "
                         "'API KEY REQUIRED', 'none' gives a plain white background")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    version_dir = ARTICLE_DIR / args.version
    if not version_dir.is_dir():
        raise SystemExit(f"not found: {version_dir}")

    scen_ids = set(
        pd.read_parquet(version_dir / f"scenario_{args.scenario}.parquet")["link_id"].astype(int)
    )
    if args.geometry:
        geom = geometry_from_geojson(args.geometry, scen_ids)
        logger.info("geometry: %s (%d links)", args.geometry.name, len(geom))
    else:
        geom = geometry_from_maps(version_dir)
        logger.info("geometry: recovered from archived maps (%d links)", len(geom))
        if not geom:
            raise SystemExit(
                "no geometry found in the archived maps — pass --geometry with a "
                "network_links.geojson from the same build"
            )

    payload = build_payload(version_dir, args.scenario, geom,
                            args.min_volume, args.min_delta)
    payload["brany"] = gateways_from_maps(version_dir)
    out = args.out or version_dir / f"fig_{args.scenario}.html"
    render(payload, out, REL_VOLUME_FLOOR, basemap=args.basemap)
    logger.info(
        "wrote %s — %d links, %d changed, sum|delta| = %s veh/day",
        out, len(payload["hrany"]), payload["n_zmenenych"], f"{payload['sigma_delta']:,}",
    )


if __name__ == "__main__":
    main()
