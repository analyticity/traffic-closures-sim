#!/usr/bin/env python3
"""Propose the next iteration of ``external_local.corridor_weights``.

Once the demand total is right, what is left on the cordon is the split between
gateways.  This reads the gateway screenlines from a validation report, compares
modelled against observed, and scales each corridor weight toward closing the
gap.

Damping matters, so the correction is not applied at full strength.  A gateway
screenline counts *all* traffic crossing it — commuting, through, and
external_local — while the weight only steers the last of those.  Multiplying a
weight by observed/modelled therefore asks one segment to absorb an error that
belongs to three, and overshoots.  ``--damping`` (default 0.5, i.e. the square
root of the ratio) keeps each step conservative; run it two or three times
rather than once at full strength.

    python scripts/tune_corridor_weights.py \\
        --report outputs/brno/baseline/demand/validation_report.json \\
        --config config/brno/sim.yaml

Prints YAML ready to paste. After applying it, re-run build-demand → validate
and check two things: the screenline error went down, and ``seed_deviation.
prior_drift_pct`` did not go up. The second is the guard — if drift grows, the
new weights are further from the truth and ODME is compensating harder, no
matter what the screenlines say.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, type=Path,
                    help="validation_report.json from the run to correct")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--damping", type=float, default=0.5,
                    help="exponent on observed/modelled; 1.0 = full correction (default 0.5)")
    ap.add_argument("--max-step", type=float, default=2.0,
                    help="cap on how much a single weight may change per iteration")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    screenlines = report.get("screenlines") or {}
    ratios: Dict[str, float] = {}
    for name, row in screenlines.items():
        if not name.startswith("auto_gw_"):
            continue
        ratio = row.get("ratio")
        if ratio:
            ratios[name[len("auto_gw_"):]] = float(ratio)
    if not ratios:
        print("V reporte nie sú žiadne gateway screenliny (auto_gw_*).", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    el = (((cfg.get("demand") or {}).get("segments") or {}).get("external_local") or {})
    current = {k: float(v) for k, v in (el.get("corridor_weights") or {}).items()}
    if not current:
        print("V configu nie sú corridor_weights — nastav ich najprv "
              "(scripts/kamdojizdime_gateway_weights.py).", file=sys.stderr)
        return 1

    missing = set(current) - set(ratios)
    if missing:
        print(f"Bez screenline, váha ostáva nezmenená: {sorted(missing)}")

    proposed: Dict[str, float] = {}
    for gw, w in current.items():
        ratio = ratios.get(gw)
        if not ratio:
            proposed[gw] = w
            continue
        factor = (1.0 / ratio) ** args.damping
        factor = max(1.0 / args.max_step, min(args.max_step, factor))
        proposed[gw] = w * factor

    total = sum(proposed.values())
    proposed = {k: v / total for k, v in proposed.items()}

    print(f"\n{'brána':<9} {'pomer':>6} {'váha teraz':>11} {'navrhovaná':>11} {'zmena':>8}")
    for gw in sorted(proposed, key=lambda g: -proposed[g]):
        r = ratios.get(gw)
        change = (proposed[gw] / current[gw] - 1.0) * 100 if current[gw] else 0.0
        print(f"{gw:<9} {r if r else float('nan'):>6.2f} {current[gw]:>11.4f} "
              f"{proposed[gw]:>11.4f} {change:>+7.1f}%")

    worst = max(ratios.items(), key=lambda kv: abs(kv[1] - 1.0))
    print(f"\nnajhoršia brána teraz: {worst[0]} (pomer {worst[1]:.2f}), "
          f"tlmenie {args.damping}")
    print("\n      corridor_weights:")
    for gw in sorted(proposed, key=lambda g: -proposed[g]):
        print(f"        {gw}: {proposed[gw]:.4f}")
    print("\nPo aplikovaní: build-demand -> validate, potom skontroluj, že klesla "
          "screenline odchýlka A NEZVÝŠIL sa prior_drift_pct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
