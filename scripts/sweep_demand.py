#!/usr/bin/env python3
"""Run demand-parameter variants and collect their validation metrics.

Only the demand half of the pipeline is re-run — network, zones and the
supernetwork are untouched — so a variant costs minutes, not a rebuild.

Two modes:

``sweep``
    Re-runs the chain once per value of one parameter and prints a comparison
    table.  Nothing else in the config is touched, so the runs differ by
    exactly one number (which is the only way the comparison means anything).

``commuting-scale``
    ``external_commuting_scale`` is a ratio applied to the SLDB-derived external
    commuting volume, not an absolute number, so a target from an outside source
    cannot be written into the config directly.  This mode measures the
    denominator with two cheap ``build-demand`` runs and prints the scale that
    hits ``--target`` (99043 veh-trips/day for T1 in kamdojizdime).

Usage (inside a container that already has the built network):

    docker run -d --name sweep simulation-brno sleep infinity
    docker cp config/brno/sim.yaml sweep:/app/config/brno/sim.yaml
    docker cp scripts/sweep_demand.py sweep:/app/scripts/sweep_demand.py
    docker exec sweep python scripts/sweep_demand.py sweep \\
        --param demand.segments.external_local.total_daily_trips \\
        --values 45000 90000 130000 auto
    docker cp sweep:/app/outputs/brno/sweeps ./outputs/brno/sweeps

Every run's ``validation_report.json`` and ``calibration_report.json`` are kept
under ``outputs/<city>/sweeps/<variant>/`` next to the resulting table.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# The demand half of the pipeline.  `build-network`, `fetch-data`,
# `normalize-network`, `build-zones` and `build-supernetwork` are deliberately
# absent — no demand parameter can change their output.
STEPS = [
    "build-demand",
    "assign-warm-skims",
    "distribute",
    "assign",
    "audit-supply",
    "calibrate",
    "validate",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> Any:
    """Set ``a.b.c`` in a nested dict, creating levels.  Returns the old value."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise SystemExit(f"'{dotted}': '{k}' is not a mapping")
    old = node.get(keys[-1])
    node[keys[-1]] = value
    return old


def coerce(raw: str) -> Any:
    if raw.lower() in ("auto", "none", "null"):
        return "auto" if raw.lower() == "auto" else None
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def write_variant(config: Path, param: str, value: Any) -> Path:
    """Variant configu vedľa originálu.

    Pôvodný sim.yaml sa nikdy neprepisuje — yaml.safe_dump by z neho zmazal
    všetky komentáre a tie nesú odôvodnenie hodnôt.  Mesto sa odvádza z názvu
    priečinka, takže výstupy idú do rovnakého outputs/<city>/.
    """
    cfg = load_yaml(config)
    set_dotted(cfg, param, value)
    tmp = config.parent / "_sweep_tmp.yaml"
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return tmp


def run_steps(config: Path, steps: List[str], log: Path) -> bool:
    """Run pipeline steps, streaming into *log*.  False on the first failure."""
    with log.open("w", encoding="utf-8") as fh:
        for step in steps:
            print(f"    {step} ... ", end="", flush=True)
            t0 = time.time()
            fh.write(f"\n{'=' * 70}\n=== {step}\n{'=' * 70}\n")
            fh.flush()
            proc = subprocess.run(
                [sys.executable, "run.py", "--config", str(config), step],
                stdout=fh, stderr=subprocess.STDOUT,
            )
            if proc.returncode != 0:
                print(f"ZLYHALO (pozri {log})")
                return False
            print(f"{time.time() - t0:.0f} s")
    return True


def read_metrics(out_dir: Path) -> Dict[str, Any]:
    """Pull the numbers worth comparing out of the two reports."""
    val = _read(out_dir / "validation_report.json")
    cal = _read(out_dir / "calibration_report.json")
    od = _read(out_dir / "od_summary.json")

    bench = (val.get("benchmarks") or {})
    cf = bench.get("calibration_fit") or {}
    hv = bench.get("holdout_validation") or {}
    sd = cal.get("seed_deviation") or {}
    hist = cal.get("history") or []

    return {
        "od_spolu": (od.get("segments") or {}).get("combined_daily"),
        "kal_n": (val.get("calibration_reference") or {}).get("n"),
        "kal_r2": cf.get("r2"),
        "hold_r2": hv.get("r2"),
        "hold_slope": hv.get("slope"),
        "hold_rmse": hv.get("pct_rmse"),
        "hold_bias": hv.get("bias_abs_pct"),
        "screenline": bench.get("screenline_max_error_pct"),
        "adekvat": bench.get("holdout_adequacy"),
        "pass": bench.get("overall_pass"),
        # Úroveň 3 z KAMDOJIZDIME_PLAN.md: lepší vstup => ODME pokriví maticu menej.
        "drift_pct": sd.get("prior_drift_pct"),
        "buniek_nad": sd.get("n_cells_exceeding_threshold"),
        "Z_koniec": (hist[-1] or {}).get("Z_objective") if hist else None,
    }


def _read(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def demand_dir(config: Path) -> Path:
    city = config.parent.name
    return Path("outputs") / city / "baseline" / "demand"


def tabulka(rows: List[Dict[str, Any]]) -> str:
    stlpce = [("variant", 22), ("od_spolu", 10), ("kal_n", 6), ("kal_r2", 7),
              ("hold_r2", 8), ("hold_slope", 6), ("hold_rmse", 6), ("hold_bias", 6),
              ("screenline", 11), ("drift_pct", 10), ("Z_koniec", 11), ("pass", 6)]
    out = ["  ".join(n.ljust(w) for n, w in stlpce), "-" * (sum(w + 2 for _, w in stlpce))]
    for r in rows:
        bunky = []
        for n, w in stlpce:
            v = r.get(n)
            if isinstance(v, float):
                s = (f"{v:,.0f}".replace(",", " ") if abs(v) >= 1000 else f"{v:.4g}")
            else:
                s = "—" if v is None else str(v)
            bunky.append(s.ljust(w))
        out.append("  ".join(bunky))
    return "\n".join(out)


def cmd_sweep(args: argparse.Namespace) -> int:
    config = Path(args.config)
    city = config.parent.name
    sweeps = Path("outputs") / city / "sweeps"
    sweeps.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    try:
        for raw in args.values:
            hodnota = coerce(raw)
            nazov = f"{args.param.rsplit('.', 1)[-1]}_{raw}"
            cieL = sweeps / nazov
            cieL.mkdir(parents=True, exist_ok=True)
            print(f"\n=== {nazov}  ({args.param} = {hodnota!r})")

            variant = write_variant(config, args.param, hodnota)
            shutil.copy2(variant, cieL / "sim.yaml.variant")

            if not run_steps(variant, STEPS, cieL / "run.log"):
                rows.append({"variant": nazov, "pass": "CHYBA"})
                continue

            src = demand_dir(config)
            for meno in ("validation_report.json", "calibration_report.json",
                         "od_summary.json", "pre_odme_convergence.json"):
                if (src / meno).exists():
                    shutil.copy2(src / meno, cieL / meno)
            rows.append({"variant": nazov, **read_metrics(cieL)})
    finally:
        (config.parent / "_sweep_tmp.yaml").unlink(missing_ok=True)

    tab = tabulka(rows)
    (sweeps / "porovnanie.txt").write_text(tab + "\n", encoding="utf-8")
    (sweeps / "porovnanie.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print("\n" + tab)
    print(f"\n→ {sweeps}/porovnanie.txt")
    print("Verdikt sa berie z holdoutu; pri 'thin' holdoute (14 bodov) uvádzaj aj kal_n.")
    return 0


def cmd_commuting_scale(args: argparse.Namespace) -> int:
    config = Path(args.config)
    kluc = "demand.sldb.external_processing.external_commuting_scale"
    merania: Dict[float, Optional[float]] = {}

    try:
        for scale in (1.0, args.reference):
            print(f"\n=== build-demand s {kluc} = {scale}")
            variant = write_variant(config, kluc, scale)
            log = Path("outputs") / config.parent.name / "sweeps"
            log.mkdir(parents=True, exist_ok=True)
            if not run_steps(variant, ["build-demand"], log / f"commuting_{scale}.log"):
                return 1
            od = _read(demand_dir(config) / "od_summary.json")
            merania[scale] = (od.get("segments") or {}).get("commuting")
            print(f"    commuting = {merania[scale]:,.0f}".replace(",", " "))
    finally:
        (config.parent / "_sweep_tmp.yaml").unlink(missing_ok=True)

    c1, c2 = merania.get(1.0), merania.get(args.reference)
    if not c1 or not c2:
        print("Nepodarilo sa odčítať segments.commuting z oboch behov.")
        return 1

    externa = (c1 - c2) / (1.0 - args.reference)
    if externa <= 0:
        print(f"Externá zložka vyšla {externa:,.0f} — scale zjavne nie je lineárny "
              f"alebo sa medzi behmi zmenilo niečo iné.".replace(",", " "))
        return 1

    scale = args.target / externa
    print(f"\n  commuting pri scale 1.00 : {c1:>12,.0f}".replace(",", " "))
    print(f"  commuting pri scale {args.reference:.2f} : {c2:>12,.0f}".replace(",", " "))
    print(f"  => externá dojížďka SLDB : {externa:>12,.0f} voz-ciest/deň".replace(",", " "))
    print(f"  => cieľ (kamdojizdime T1): {args.target:>12,.0f}".replace(",", " "))
    print(f"\n  {kluc}: {scale:.4f}")
    print("\nZapíš to do sim.yaml a prežeň cez `sweep` proti dnešnej hodnote, "
          "nech je vidieť, či to holdoutu pomohlo.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/brno/sim.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run one parameter over several values")
    s.add_argument("--param", required=True, help="dotted key in sim.yaml")
    s.add_argument("--values", nargs="+", required=True)
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("commuting-scale", help="derive external_commuting_scale from a target")
    c.add_argument("--target", type=float, default=99043.0,
                   help="veh-trips/day the scale should produce (default: kamdojizdime T1)")
    c.add_argument("--reference", type=float, default=0.65,
                   help="second measurement point (default: today's effective value)")
    c.set_defaults(func=cmd_commuting_scale)

    args = ap.parse_args()
    if not Path(args.config).exists():
        print(f"Config {args.config} neexistuje.")
        return 2
    if not Path("run.py").exists():
        print("Spúšťaj z koreňa repa (chýba run.py).")
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
