#!/usr/bin/env python3
"""Experiment 01: Baseline plausibility (daily model, sparse CSD).

No GEH pass-rate gates — instead:
- Scatter observed vs modeled + basic stats
- Factor-of-2 band (% stations within 0.5×–2× observed)
- Mean volumes by OSM ``link_type`` (sanity ordering)
- Optional sums by Czech road-number class (D / I / II / III) from ``osm_ref``
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    compute_factor_of_2_pct,
    init_experiment,
    load_assignment_results,
    load_network_links,
    save_figure,
    save_json,
)

NAME = "exp01_baseline_plausibility"

# Expected high-to-low capacity hierarchy (index used only for rank correlation).
_LINK_TYPE_VOLUME_ORDER = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
]


def _resolve_obs_mod_columns(diag: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    obs_candidates = ["observed", "observed_total", "observed_motor_total"]
    mod_candidates = ["modeled", "mod", "modeled_total", "lw_mean", "model_lw_mean"]
    obs_col = next((c for c in obs_candidates if c in diag.columns), None)
    mod_col = next((c for c in mod_candidates if c in diag.columns), None)
    return obs_col, mod_col


def _czech_road_bucket(osm_ref: Any) -> str:
    s = str(osm_ref or "").strip().upper()
    if re.match(r"^D\d", s) or s.startswith("D/"):
        return "dalnice"
    if s.startswith("I/") or re.match(r"^I\d", s):
        return "I_trida"
    if s.startswith("II/") or re.match(r"^II\d", s):
        return "II_trida"
    if s.startswith("III/") or re.match(r"^III\d", s):
        return "III_trida"
    return "other"


def _rank_coherence(mean_vol_by_type: pd.Series) -> Dict[str, Any]:
    """Spearman between empirical mean-volume rank and *a priori* link-type order."""
    scores: List[float] = []
    ranks: List[int] = []
    for lt, mean_v in mean_vol_by_type.items():
        lt_s = str(lt)
        if lt_s not in _LINK_TYPE_VOLUME_ORDER:
            continue
        scores.append(float(mean_v))
        ranks.append(_LINK_TYPE_VOLUME_ORDER.index(lt_s))
    if len(scores) < 3:
        return {"n_types": len(scores), "spearman_rho": None, "p_value": None}
    rho, p = sp_stats.spearmanr(ranks, scores)
    return {
        "n_types": len(scores),
        "spearman_rho": round(float(rho), 4) if np.isfinite(rho) else None,
        "p_value": float(p) if np.isfinite(p) else None,
    }


def main() -> None:
    cfg, out_dir = init_experiment(NAME)
    demand_dir = Path(cfg["demand"]["output_dir"])

    vr_path = demand_dir / "validation_report.json"
    vr = json.loads(vr_path.read_text(encoding="utf-8")) if vr_path.exists() else {}

    assign_df = load_assignment_results(cfg)
    links = load_network_links(cfg)
    base_cols = ["link_id", "link_type"]
    for c in ("name", "osm_ref"):
        if c in links.columns:
            base_cols.append(c)
    merged = links[base_cols].merge(assign_df, on="link_id", how="inner")
    vol_col = "wd_daily_tot" if "wd_daily_tot" in merged.columns else None

    summary: Dict[str, Any] = {
        "experiment": NAME,
        "note": "Daily model — plausibility metrics, not industrial GEH thresholds.",
        "validation_report_keys": list(vr.keys())[:20] if isinstance(vr, dict) else [],
    }

    # --- Mean modeled volume by link_type (network sanity) ---
    if vol_col and "link_type" in merged.columns:
        by_type = (
            merged.groupby("link_type", dropna=False)[vol_col]
            .mean()
            .sort_values(ascending=False)
        )
        summary["model_mean_volume_by_link_type"] = {
            str(k): round(float(v), 1) for k, v in by_type.items()
        }
        summary["volume_rank_vs_osm_hierarchy"] = _rank_coherence(by_type)

        fig, ax = plt.subplots(figsize=(10, 5))
        by_type.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_ylabel(f"Mean {vol_col} (veh/day per link)")
        ax.set_xlabel("link_type (OSM)")
        ax.set_title("Model — mean daily volume by road class")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        save_figure(fig, out_dir / "volume_by_class.png")
        plt.close(fig)

    # --- CSD matched diagnostics: scatter, factor-of-2, per-type bias, road buckets ---
    diag_path = demand_dir / "matching_diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        obs_col, mod_col = _resolve_obs_mod_columns(diag)
        if obs_col and mod_col:
            obs = pd.to_numeric(diag[obs_col], errors="coerce").values
            mod = pd.to_numeric(diag[mod_col], errors="coerce").values
            valid = np.isfinite(obs) & np.isfinite(mod) & (obs > 0)
            obs_v, mod_v = obs[valid], mod[valid]

            from sim.calibration.metrics import compute_stats

            stats = compute_stats(mod_v, obs_v)
            summary["holdout_stats"] = stats
            summary["factor_of_2"] = compute_factor_of_2_pct(mod_v, obs_v)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(obs_v, mod_v, alpha=0.55, s=28, edgecolors="none")
            lim = max(float(obs_v.max()), float(mod_v.max())) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
            ax.plot([0, lim], [0, 0.5 * lim], "g:", lw=0.9, alpha=0.75, label="0.5× / 2× band")
            ax.plot([0, lim], [0, 2.0 * lim], "g:", lw=0.9, alpha=0.75)
            if stats.get("slope") and np.isfinite(stats["slope"]):
                xs = np.array([0.0, lim])
                ax.plot(
                    xs,
                    stats["slope"] * xs + float(stats.get("intercept") or 0),
                    "r-",
                    lw=0.9,
                    label=f"regression: y = {stats['slope']:.2f}x + {stats.get('intercept', 0):.0f}",
                )
            ax.set_xlabel(f"Observed ({obs_col})")
            ax.set_ylabel(f"Modeled ({mod_col})")
            ax.set_title(
                f"Baseline plausibility  (n={stats.get('n', len(obs_v))},  "
                f"factor-of-2: {summary['factor_of_2'].get('pct_in_factor_of_2')}%)"
            )
            ax.legend(loc="upper left", fontsize=8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            save_figure(fig, out_dir / "scatter.png")
            plt.close(fig)

            # Factor-of-2 visual: ratios
            ratio = mod_v / obs_v
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(ratio, bins=40, edgecolor="white", alpha=0.85, color="teal")
            ax.axvline(0.5, color="orange", ls="--", lw=1)
            ax.axvline(2.0, color="orange", ls="--", lw=1)
            ax.axvline(1.0, color="black", ls="-", lw=0.8)
            ax.set_xlabel("model / observed")
            ax.set_ylabel("Count stations")
            ax.set_title(
                f"Model/observed ratio  (median={summary['factor_of_2'].get('median_ratio')})"
            )
            save_figure(fig, out_dir / "factor_of_2.png")
            plt.close(fig)

            # Per link_type in diagnostics
            if "link_type" in diag.columns:
                per_lt = []
                for lt, g in diag.groupby("link_type"):
                    o = pd.to_numeric(g[obs_col], errors="coerce")
                    m = pd.to_numeric(g[mod_col], errors="coerce")
                    ok = o > 0
                    if ok.sum() < 1:
                        continue
                    bias_pct = float((m[ok] - o[ok]).mean() / o[ok].mean() * 100)
                    per_lt.append({
                        "link_type": str(lt),
                        "n": int(ok.sum()),
                        "mean_observed": round(float(o[ok].mean()), 1),
                        "mean_modeled": round(float(m[ok].mean()), 1),
                        "bias_pct": round(bias_pct, 2),
                    })
                summary["per_link_type_matched"] = sorted(per_lt, key=lambda x: -x["n"])

            # Czech road-number buckets (osm_ref on matched rows)
            ref_col = "osm_ref" if "osm_ref" in diag.columns else None
            if ref_col:
                d2 = diag.assign(_bucket=diag[ref_col].map(_czech_road_bucket))
                rows = []
                for bkt, g in d2.groupby("_bucket"):
                    o = pd.to_numeric(g[obs_col], errors="coerce")
                    m = pd.to_numeric(g[mod_col], errors="coerce")
                    ok = o > 0
                    if ok.sum() < 1:
                        continue
                    rows.append({
                        "bucket": str(bkt),
                        "n_stations": int(ok.sum()),
                        "sum_observed": round(float(o[ok].sum()), 1),
                        "sum_modeled": round(float(m[ok].sum()), 1),
                        "bias_pct": round(
                            float((m[ok].sum() - o[ok].sum()) / max(o[ok].sum(), 1) * 100),
                            2,
                        ),
                    })
                summary["road_number_class_totals"] = sorted(
                    rows, key=lambda r: r["n_stations"], reverse=True,
                )
        else:
            summary["diagnostics_warning"] = (
                f"Could not resolve obs/mod columns. Columns: {list(diag.columns)}"
            )
    else:
        summary["diagnostics_warning"] = f"Missing {diag_path}"

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
