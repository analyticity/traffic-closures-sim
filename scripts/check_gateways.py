#!/usr/bin/env python3
"""Verify that every whitelisted gateway road actually produced a gateway.

Run right after ``build-zones``.  A whitelist entry whose road ref does not
match anything in the network fails silently in the pipeline (it only logs
``token ...: no ref match in whole network``), so the model quietly keeps the
old set of entry points.  This script turns that into a pass/fail.

Usage:
    python scripts/check_gateways.py --config config/brno/sim.yaml

Needs only pyyaml + pandas — no geo stack, so it also runs outside the
pipeline container.

Exit code 0 = every whitelisted road has at least one gateway.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

# CSD 2025 AADT on the Brno-mesto / Brno-venkov district boundary, for context
# in the report. Roads not listed here simply print "-".
_CSD_BOUNDARY_AADT = {
    "52": 50911, "43": 40582, "50": 24271, "602": 18523, "380": 12348,
    "6401": 10998, "430": 10115, "15286": 9977, "3844": 9615, "3846": 7430,
    "383": 7035, "373": 6425, "15275": 5723, "15276": 4854, "37915": 3800,
    "41614": 3357, "417": 2622,
}


def norm_text(value: object) -> str:
    """Same normalisation the gateway matcher uses."""
    text = str(value or "").strip().upper()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for ch in (" ", "/", "\\", "-", "."):
        text = text.replace(ch, "")
    return text


def numeric_part(token: str) -> str:
    """Strip a road-class prefix: II602 -> 602, I43 -> 43, D1 -> 1."""
    t = norm_text(token)
    for prefix in ("III", "II", "I", "D", "R"):
        if t.startswith(prefix) and t[len(prefix):].isdigit():
            return t[len(prefix):]
    return t


def whitelist_tokens(cfg: dict) -> list[str]:
    raw = ((cfg.get("zoning") or {}).get("external_gateways") or {}).get("whitelist") or []
    out = []
    for item in raw:
        ref = item.get("ref") if isinstance(item, dict) else item
        if ref:
            out.append(str(ref))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/brno/sim.yaml")
    ap.add_argument("--zones-dir", default=None,
                    help="override zoning.output_dir (default: read from config)")
    args = ap.parse_args()

    # Prefer the merged config (zoning.output_dir is filled in by load_config,
    # not present in sim.yaml), fall back to raw YAML outside the pipeline env.
    try:
        from sim.io_project import load_config

        cfg = load_config(args.config)
    except Exception:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    city = Path(args.config).parent.name
    zones_dir = Path(
        args.zones_dir
        or (cfg.get("zoning") or {}).get("output_dir")
        or f"outputs/{city}/baseline/zones"
    )
    diag_path = zones_dir / "gateway_diagnostics.csv"

    if not diag_path.exists():
        zones_geojson = zones_dir / "zones.geojson"
        print(f"CHYBA: {diag_path} neexistuje.\n")
        if zones_dir.exists():
            found = sorted(p.name for p in zones_dir.iterdir())
            print(f"Adresár {zones_dir} existuje a obsahuje {len(found)} položiek:")
            for name in found[:25]:
                print(f"    {name}")
            if len(found) > 25:
                print(f"    ... a ďalších {len(found) - 25}")
        else:
            print(f"Adresár {zones_dir} vôbec neexistuje — build-zones nedobehol.")
            return 2

        print()
        if zones_geojson.exists():
            print("zones.geojson JE na mieste, takže build-zones prebehol.")
            print("gateway_diagnostics.csv sa nezapisuje, keď sa nenájde ANI JEDNA brána")
            print("(export_gateway_diagnostics() končí hneď pri prázdnom gateway_meta).")
            print()
            print("V logu build-zones hľadaj:")
            print("    'Discover whitelist token ...'      ktoré refy sa vôbec skúšali")
            print("    'no ref match in whole network'     ref sa nenašiel v sieti")
            print("    'no boundary-near matched links'    ref sa našiel, ale nie pri hranici AOI")
            print("    'synthetic external gateway zones added: N'")
            print()
            print("Najčastejšia príčina: zoning.external_gateways.allowed_link_types")
            print("nezahŕňa typ cesty, na ktorej ten ref v OSM je.")
        else:
            print("Chýba aj zones.geojson — build-zones spadol skôr, než čokoľvek zapísal.")
        return 2

    gw = pd.read_csv(diag_path)
    tokens = whitelist_tokens(cfg)

    print(f"config      : {args.config}")
    print(f"diagnostika : {diag_path}")
    print(f"brán spolu  : {len(gw)}   (whitelist má {len(tokens)} ciest)\n")

    # --- one row per whitelisted road -------------------------------------
    gw["_tok"] = gw["whitelist_token"].map(numeric_part)
    gw["_ref"] = gw.get("matched_ref", pd.Series(dtype=str)).map(numeric_part)

    print(f"{'cesta':<10} {'brán':>4}  {'CSD hranica':>11}  {'typ':<16} názvy brán")
    print("-" * 92)

    missing: list[str] = []
    for tok in tokens:
        num = numeric_part(tok)
        rows = gw[(gw["_tok"] == num) | (gw["_ref"] == num)]
        aadt = _CSD_BOUNDARY_AADT.get(num)
        aadt_s = f"{aadt:,}".replace(",", " ") if aadt else "-"
        if rows.empty:
            missing.append(tok)
            print(f"{tok:<10} {0:>4}  {aadt_s:>11}  {'—':<16} *** NENAŠLA SA ***")
            continue
        typ = str(rows["predominant_link_type"].mode().iat[0]) if "predominant_link_type" in rows else "?"
        names = ", ".join(sorted(rows["gateway_name"].astype(str)))
        print(f"{tok:<10} {len(rows):>4}  {aadt_s:>11}  {typ:<16} {names}")

    # --- gateways not traceable to any whitelist entry --------------------
    known = {numeric_part(t) for t in tokens}
    orphans = gw[~gw["_tok"].isin(known) & ~gw["_ref"].isin(known)]
    if not orphans.empty:
        print(f"\nBrány mimo whitelistu ({len(orphans)}) — auto-objavené alebo zlúčené:")
        for _, r in orphans.iterrows():
            print(f"  {r['gateway_name']}  token={r.get('whitelist_token')} ref={r.get('matched_ref')}")

    # --- quality warnings --------------------------------------------------
    warn: list[str] = []
    if "dist_boundary_m" in gw.columns:
        far = gw[pd.to_numeric(gw["dist_boundary_m"], errors="coerce") > 1000]
        for _, r in far.iterrows():
            warn.append(f"{r['gateway_name']}: kotva {float(r['dist_boundary_m']):.0f} m od hranice AOI")
    if "target_node_ids" in gw.columns:
        empty = gw[gw["target_node_ids"].astype(str).str.strip().isin(["", "nan"])]
        for _, r in empty.iterrows():
            warn.append(f"{r['gateway_name']}: žiadne cieľové uzly — brána nebude pripojená")
    dup = gw["gateway_name"].value_counts()
    for name, n in dup[dup > 1].items():
        warn.append(f"{name}: duplicitný názov brány ({n}x)")

    if warn:
        print("\nUpozornenia:")
        for w in warn:
            print(f"  ! {w}")

    print()
    if missing:
        print(f"ZLYHALO: {len(missing)} ciest z whitelistu nemá bránu: {', '.join(missing)}")
        print("Pravdepodobná príčina: OSM tá cesta má iný `ref`, alebo jej `link_type`")
        print("nie je v zoning.external_gateways.allowed_link_types.")
        return 1

    print(f"OK: všetkých {len(tokens)} ciest z whitelistu má bránu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
