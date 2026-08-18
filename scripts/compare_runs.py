#!/usr/bin/env python3
"""Compare two or more pipeline runs side by side.

Reads ``metrics.json`` (and ``calibration_report.json`` when present) from each
run directory and prints a comparison table. The first directory is the
reference; every later column also shows the delta against it.

Usage:
    python scripts/compare_runs.py simulation_for_article/base_model \
                                   simulation_for_article/updated_version_1 \
                                   simulation_for_article/updated_version_2

    python scripts/compare_runs.py --markdown run_a run_b > porovnanie.md

Only the standard library is needed.

Metric key names drifted between runs (``od_spolu`` vs ``combined_daily``,
``bias_pct`` vs ``bias_abs_pct``, ...), so every field is looked up through a
list of aliases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# (label, [section.key aliases], kind)
#   kind: "int" | "float1" | "float3" | "pct" | "str"
#   Sign convention: "up" = higher is better, "down" = lower is better,
#   "-" = neither (context only).
_ROWS: list[tuple[str, str, list[str], str, str]] = [
    ("SIEŤ",       "hrán",                    ["siet.hrany"],                              "int",    "-"),
    ("SIEŤ",       "zón TAZ",                 ["siet.zony_taz"],                           "int",    "-"),
    ("SIEŤ",       "  z toho externých",      ["siet.zony_externe"],                       "int",    "-"),
    ("SIEŤ",       "  z toho interných",      ["siet.zony_interne"],                       "int",    "-"),

    ("DOPYT",      "OD spolu (voz/deň)",      ["dopyt.combined_daily", "dopyt.od_spolu"],  "int",    "-"),
    ("DOPYT",      "  commuting",             ["dopyt.commuting"],                         "int",    "-"),
    ("DOPYT",      "  other",                 ["dopyt.other"],                             "int",    "-"),
    ("DOPYT",      "  external_local",        ["dopyt.external_local"],                    "int",    "-"),
    ("DOPYT",      "  external_through",      ["dopyt.external_through_total",
                                               "dopyt.external_through"],                  "int",    "-"),

    ("KALIBRÁCIA", "n úsekov",                ["kalibracia.n"],                            "int",    "up"),
    ("KALIBRÁCIA", "R²",                      ["kalibracia.r2"],                           "float3", "up"),
    ("KALIBRÁCIA", "slope",                   ["kalibracia.slope"],                        "float3", "up"),
    ("KALIBRÁCIA", "%RMSE",                   ["kalibracia.pct_rmse"],                     "float1", "down"),
    ("KALIBRÁCIA", "|bias| %",                ["kalibracia.bias_abs_pct",
                                               "kalibracia.bias_pct"],                     "float1", "down"),

    ("HOLDOUT",    "R²",                      ["holdout.r2"],                              "float3", "up"),
    ("HOLDOUT",    "slope",                   ["holdout.slope"],                           "float3", "up"),
    ("HOLDOUT",    "%RMSE",                   ["holdout.pct_rmse"],                        "float1", "down"),
    ("HOLDOUT",    "|bias| %",                ["holdout.bias_abs_pct",
                                               "holdout.bias_pct"],                        "float1", "down"),
    ("HOLDOUT",    "adekvátnosť",             ["holdout.adekvatnost"],                     "str",    "-"),
    ("HOLDOUT",    "prijaté",                 ["holdout.prijate"],                         "str",    "-"),

    ("SCREENLINE", "max odchýlka %",          ["screenlines.max_odchylka_pct"],            "float1", "down"),
    ("SCREENLINE", "porovnaných",             ["screenlines.porovnanych"],                 "int",    "-"),

    ("ODME",       "iterácií",                ["odme.iteracii"],                           "int",    "-"),
    ("ODME",       "konvergovalo",            ["odme.konvergovalo"],                       "str",    "-"),
    ("ODME",       "Z start",                 ["odme.Z_start"],                            "int",    "-"),
    ("ODME",       "Z koniec",                ["odme.Z_koniec"],                           "int",    "down"),

    ("KPI",        "VKT (voz-km)",            ["baseline_kpi.vkt_voz_km"],                 "int",    "-"),
    ("KPI",        "VHT (voz-h)",             ["baseline_kpi.vht_voz_h"],                  "int",    "-"),
    ("KPI",        "preťažených hrán",        ["baseline_kpi.pretazenych_hran"],           "int",    "down"),
]


def dig(data: dict, dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def lookup(data: dict, aliases: Sequence[str]) -> Any:
    for a in aliases:
        v = dig(data, a)
        if v is not None:
            return v
    return None


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "str":
        return {True: "áno", False: "nie"}.get(value, str(value))
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "int":
        return f"{v:,.0f}".replace(",", " ")
    if kind == "float1":
        return f"{v:.1f}"
    if kind == "float3":
        return f"{v:.3f}"
    return str(value)


def delta_str(new: Any, ref: Any, kind: str, better: str) -> str:
    if new is None or ref is None or kind == "str":
        return ""
    try:
        a, b = float(new), float(ref)
    except (TypeError, ValueError):
        return ""
    d = a - b
    if abs(d) < 1e-9:
        return "="
    if kind == "int":
        core = f"{d:+,.0f}".replace(",", " ")
        if abs(b) > 1e-9:
            core += f" ({d / b * 100:+.1f} %)"
    elif kind == "float3":
        core = f"{d:+.3f}"
    else:
        core = f"{d:+.1f}"
    if better == "up":
        mark = "✓" if d > 0 else "✗"
    elif better == "down":
        mark = "✓" if d < 0 else "✗"
    else:
        mark = " "
    return f"{core} {mark}"


def gateway_table(runs: list[tuple[str, Path]]) -> list[tuple[str, list[str]]]:
    """modeled/observed per auto_gw_* screenline, from calibration_report.json."""
    per_run: list[dict[str, tuple[float, float]]] = []
    for _, path in runs:
        rep = path / "calibration_report.json"
        gws: dict[str, tuple[float, float]] = {}
        if rep.exists():
            try:
                sl = json.loads(rep.read_text(encoding="utf-8")).get("screenlines") or {}
                for name, row in sl.items():
                    if not str(name).startswith("auto_gw_") or not isinstance(row, dict):
                        continue
                    gws[name] = (
                        float(row.get("modeled_total") or 0.0),
                        float(row.get("observed_total") or 0.0),
                    )
            except Exception:
                pass
        per_run.append(gws)

    names = sorted({n for g in per_run for n in g})
    out = []
    for n in names:
        cells = []
        for g in per_run:
            if n not in g:
                cells.append("—")
                continue
            mod, obs = g[n]
            ratio = mod / obs if obs else float("nan")
            cells.append(f"{mod:,.0f} / {obs:,.0f}".replace(",", " ") + f"  ({ratio:.2f})")
        out.append((n.replace("auto_gw_", ""), cells))
    return out


def render(runs: list[tuple[str, Path]], data: list[dict], markdown: bool) -> str:
    names = [n for n, _ in runs]
    lines: list[str] = []
    w0 = max([len(lbl) for _, lbl, *_ in _ROWS] + [20])
    wc = max([len(n) for n in names] + [16])

    def row(cells: list[str], label: str) -> str:
        if markdown:
            return "| " + " | ".join([label] + cells) + " |"
        return f"{label:<{w0}}" + "".join(f"  {c:>{wc}}" for c in cells)

    if markdown:
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
    else:
        lines.append(row(names, ""))
        lines.append("─" * (w0 + (wc + 2) * len(names)))

    section = None
    for sec, label, aliases, kind, better in _ROWS:
        if sec != section:
            section = sec
            lines.append(row([""] * len(names), f"**{sec}**" if markdown else f"── {sec} ──"))
        vals = [lookup(d, aliases) for d in data]
        cells = [fmt(vals[0], kind)]
        for v in vals[1:]:
            d = delta_str(v, vals[0], kind, better)
            cells.append(fmt(v, kind) + (f"   {d}" if d and not markdown else (f" ({d})" if d else "")))
        lines.append(row(cells, label))

    gw = gateway_table(runs)
    if gw:
        lines.append("")
        lines.append(row([""] * len(names), "**BRÁNY** model / observed (pomer)" if markdown
                         else "── BRÁNY: model / observed (pomer) ──"))
        for label, cells in gw:
            lines.append(row(cells, label))

    metas = [(d.get("_meta") or {}) for d in data]
    lines.append("")
    for (n, _), m in zip(runs, metas):
        note = str(m.get("popis", "")).strip()
        stamp = str(m.get("vytvorene", ""))[:16]
        commit = str(m.get("git_commit", ""))
        bits = " · ".join(x for x in (stamp, commit, note) if x)
        lines.append(f"{'- ' if markdown else '  '}{n}: {bits}" if bits else f"  {n}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="adresáre s metrics.json (prvý je referenčný)")
    ap.add_argument("--markdown", action="store_true", help="výstup ako markdown tabuľka")
    args = ap.parse_args()

    runs: list[tuple[str, Path]] = []
    data: list[dict] = []
    for r in args.runs:
        p = Path(r)
        mp = p / "metrics.json"
        if not mp.exists():
            print(f"CHYBA: {mp} neexistuje", file=sys.stderr)
            return 2
        runs.append((p.name, p))
        data.append(json.loads(mp.read_text(encoding="utf-8")))

    print(render(runs, data, args.markdown))
    print("\n✓ = zlepšenie oproti prvému stĺpcu, ✗ = zhoršenie."
          "\nPozor: pri rôznom n nie sú metriky priamo porovnateľné.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
