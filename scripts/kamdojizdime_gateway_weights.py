#!/usr/bin/env python3
"""Derive ``external_local.corridor_weights`` from kamdojizdime mobile data.

Without ``corridor_weights`` the external_local segment is split across gateways
by road class alone (``_LINK_TYPE_WEIGHT_SCALE`` in ``sim/demand/config.py``:
motorway 1.0, trunk 0.9, primary 0.7, secondary 0.4).  That is the only thing the
model otherwise knows about the size of a corridor, and it is coarse enough to
show up in validation — I43_N was drawing 12.1 % of the segment against 8.2 % of
observed cordon traffic (modelled/observed 1.38) while D1_E and I52_S drew less
than their share (0.69 and 0.65).

Kamdojizdime is a better source for this particular weight than the CSD counts:

* It measures the right thing.  ``external_local`` is the non-commuting external
  segment, so this script sums T2 + T3 + N (``typ_cesty`` 2, 3, 6) — services and
  visitors.  A CSD cordon count is all vehicles, so its motorway share is inflated
  by through traffic that does not belong to this segment at all.  The two sources
  agree within 1-3 pp on the motorway gateways and diverge by a factor of two to
  three on the class II roads, which is exactly where that difference should land.
* It keeps CSD independent.  Weighting the seed with CSD would tune the model on
  the same counts the gateway screenlines validate against.

Caveats worth repeating in the thesis: ~63 % of relations are below the
publication threshold of 7 persons (empty cell, not zero), and roughly 13 % of
the volume belongs to municipalities missing from the gateway lookup.

    python scripts/kamdojizdime_gateway_weights.py \\
        --data-dir data --lookup data/brno/cache/external_gateway_lookup.parquet

``external_gateway_lookup.parquet`` is written by ``build-demand``; pull it out of
a built image with ``docker cp`` if you do not have a local run.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# typ_cesty per the INTENS methodology, table 2 column E:
#   1 T1 commuting | 2 T2 intensive services | 3 T3 occasional services
#   4 D2 second home (already counted in T2/T3) | 5 PN overnight | 6 N visitor
EXTERNAL_LOCAL_TYPES = {"2", "3", "6"}


def read_side(pattern: str, code_column: str, value_column: str) -> Dict[str, Dict[str, float]]:
    """Sum persons per municipality code, per season."""
    per_season: Dict[str, Dict[str, float]] = {}
    for path in sorted(glob.glob(pattern)):
        season = Path(path).parent.name.replace("kam_dojizdime_", "")
        acc: Dict[str, float] = collections.Counter()
        with open(path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter=";"):
                if row.get("typ_cesty") not in EXTERNAL_LOCAL_TYPES:
                    continue
                raw = (row.get(value_column) or "").strip().replace(",", ".")
                if not raw:
                    continue  # suppressed cell — empty is not zero
                try:
                    acc[str(row[code_column]).strip()] += float(raw)
                except (ValueError, KeyError):
                    continue
        per_season[season] = acc
    return per_season


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--city-code", default="582786", help="obec code of the modelled city")
    ap.add_argument("--lookup", required=True, type=Path,
                    help="external_gateway_lookup.parquet from build-demand")
    ap.add_argument("--value-column", default="pocet_osob_pd",
                    choices=["pocet_osob_pd", "pocet_osob"])
    args = ap.parse_args()

    inbound = read_side(str(args.data_dir / "kam_dojizdime_*" / f"{args.city_code}_dojizdka.csv"),
                        "kod_zdroj", args.value_column)
    outbound = read_side(str(args.data_dir / "kam_dojizdime_*" / f"{args.city_code}_vyjizdka.csv"),
                         "kod_cil", args.value_column)
    seasons = sorted(set(inbound) | set(outbound))
    if not seasons:
        print(f"Nenašiel som žiadne kamdojizdime CSV pod {args.data_dir}/kam_dojizdime_*",
              file=sys.stderr)
        return 2

    per_place: Dict[str, float] = collections.Counter()
    for season in seasons:
        for code, value in inbound.get(season, {}).items():
            per_place[code] += value / len(seasons)
        for code, value in outbound.get(season, {}).items():
            per_place[code] += value / len(seasons)
    per_place.pop(args.city_code, None)

    print(f"sezóny: {', '.join(seasons)}")
    print(f"obcí s nenulovým T2+T3+N: {len(per_place):,}".replace(",", " "))

    lookup = pd.read_parquet(args.lookup)
    lookup["unit_id"] = lookup["unit_id"].astype(str).str.strip()
    lookup = lookup[lookup["gateway_name"].notna()].copy()
    # Same weighting the demand builder uses: inverse route cost across candidates.
    lookup["inv"] = 1.0 / lookup["route_cost_s"].clip(lower=1.0)
    lookup["w"] = lookup["inv"] / lookup.groupby("unit_id")["inv"].transform("sum")

    places = pd.DataFrame({"unit_id": list(per_place), "osoby": list(per_place.values())})
    merged = places.merge(lookup[["unit_id", "gateway_name", "w"]], on="unit_id", how="inner")
    if merged.empty:
        print("Žiadna obec sa nenapárovala na bránu — sedia kódy obcí s unit_id?",
              file=sys.stderr)
        return 1

    covered = merged.groupby("unit_id")["osoby"].first().sum()
    total = sum(per_place.values())
    print(f"napárovaných na bránu: {merged['unit_id'].nunique():,} obcí "
          f"({covered / total:.1%} objemu)".replace(",", " "))

    weights = merged.assign(v=merged["osoby"] * merged["w"]).groupby("gateway_name")["v"].sum()
    weights = (weights / weights.sum()).sort_values(ascending=False)

    print("\n      corridor_weights:")
    for name, value in weights.items():
        print(f"        {name}: {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
