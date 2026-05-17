#!/usr/bin/env python3
"""Experiment 01: Comprehensive baseline model validation.

Validates the reference (no-closure) model across multiple dimensions,
following FHWA TMIP / WebTAG-style validation tiers:

1. **Structural checks**: Volume hierarchy by OSM link type, Spearman ρ
2. **Count validation**: CSD scatter, factor-of-2 band, R², regression stats
3. **Screenline validation**: Aggregated model vs observed per screenline, GEH
4. **System-level aggregates**: Total VMT, VHT, mean V/C, overloaded links
5. **Trip length distribution**: Histogram of OD trip distances from skims
6. **Per-class bias**: Czech road-number buckets (D / I / II / III)
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
    compute_vht,
    compute_vkt,
    init_experiment,
    load_assignment_results,
    load_network_links,
    save_csv,
    save_figure,
    save_json,
)
from sim._metrics import aggregate_daily_volumes

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
    mod_candidates = ["modeled", "mod", "modeled_total", "_corridor_volume", "lw_mean", "model_lw_mean"]
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
        "note": "Denní model — metriky plausibility, ne průmyslové GEH prahy.",
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
        ax.set_ylabel(f"Průměrný {vol_col} (voz/den na hraně)")
        ax.set_xlabel("link_type (OSM)")
        ax.set_title("Model — průměrný denní objem podle typu komunikace")
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
            ax.plot([0, lim], [0, 0.5 * lim], "g:", lw=0.9, alpha=0.75, label="0.5× / 2× pásmo")
            ax.plot([0, lim], [0, 2.0 * lim], "g:", lw=0.9, alpha=0.75)
            if stats.get("slope") and np.isfinite(stats["slope"]):
                xs = np.array([0.0, lim])
                ax.plot(
                    xs,
                    stats["slope"] * xs + float(stats.get("intercept") or 0),
                    "r-",
                    lw=0.9,
                    label=f"regrese: y = {stats['slope']:.2f}x + {stats.get('intercept', 0):.0f}",
                )
            ax.set_xlabel(f"Pozorované ({obs_col})")
            ax.set_ylabel(f"Modelované ({mod_col})")
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
            ax.set_xlabel("model / pozorované")
            ax.set_ylabel("Počet stanic")
            ax.set_title(
                f"Poměr model/pozorované  (median={summary['factor_of_2'].get('median_ratio')})"
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

    # --- System-level aggregates (VMT, VHT, overloaded links) ---
    aggregate_daily_volumes(merged)
    vht = compute_vht(merged)
    vkt = compute_vkt(merged)
    n_overloaded = int((merged.get("VOC_max", pd.Series(dtype=float)).fillna(0) > 1.0).sum())
    mean_vc = float(merged["VOC_max"].mean()) if "VOC_max" in merged.columns else None
    summary["system_aggregates"] = {
        "total_vht_hours": round(vht, 1),
        "total_vkt_km": round(vkt, 1),
        "n_links": len(merged),
        "n_overloaded_links": n_overloaded,
        "pct_overloaded": round(n_overloaded / max(len(merged), 1) * 100, 2),
        "mean_voc": round(mean_vc, 4) if mean_vc is not None else None,
    }

    # --- Screenline validation (load from calibration/validation reports) ---
    for report_name in ("validation_report.json", "calibration_report.json"):
        rp = demand_dir / report_name
        if not rp.exists():
            continue
        try:
            report = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sl_data = report.get("screenlines") or report.get("screenline_results")
        if not sl_data or not isinstance(sl_data, dict):
            continue

        sl_rows = []
        for sl_name, sl_info in sl_data.items():
            if not isinstance(sl_info, dict):
                continue
            obs = sl_info.get("observed_total", 0)
            mod = sl_info.get("modeled_total", 0)
            ratio = mod / obs if obs > 0 else None
            geh = sl_info.get("geh")
            if geh is None and obs > 0:
                geh = float(np.sqrt(2.0 * (mod - obs) ** 2 / (mod + obs))) if (mod + obs) > 0 else None
            sl_rows.append({
                "screenline": sl_name,
                "observed": round(float(obs), 0),
                "modeled": round(float(mod), 0),
                "ratio": round(float(ratio), 3) if ratio is not None else None,
                "geh": round(float(geh), 2) if geh is not None else None,
                "excluded": sl_info.get("excluded_from_benchmark", False),
            })

        if sl_rows:
            sl_df = pd.DataFrame(sl_rows)
            save_csv(sl_df, out_dir / "screenline_validation.csv")

            eligible = [r for r in sl_rows if not r.get("excluded") and r["ratio"] is not None]
            if eligible:
                ratios = [r["ratio"] for r in eligible]
                gehs = [r["geh"] for r in eligible if r["geh"] is not None]
                summary["screenline_validation"] = {
                    "source": report_name,
                    "n_total": len(sl_rows),
                    "n_eligible": len(eligible),
                    "mean_ratio": round(float(np.mean(ratios)), 3),
                    "median_ratio": round(float(np.median(ratios)), 3),
                    "max_abs_deviation_pct": round(max(abs(r - 1.0) for r in ratios) * 100, 1),
                    "pct_within_20pct": round(
                        sum(1 for r in ratios if 0.8 <= r <= 1.2) / len(ratios) * 100, 1
                    ),
                    "mean_geh": round(float(np.mean(gehs)), 2) if gehs else None,
                }

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                ax = axes[0]
                names = [r["screenline"][:25] for r in eligible]
                obs_vals = [r["observed"] for r in eligible]
                mod_vals = [r["modeled"] for r in eligible]
                x = np.arange(len(names))
                w = 0.35
                ax.barh(x - w / 2, obs_vals, w, label="Pozorované", color="steelblue")
                ax.barh(x + w / 2, mod_vals, w, label="Modelované", color="coral")
                ax.set_yticks(x)
                ax.set_yticklabels(names, fontsize=7)
                ax.set_xlabel("Denní objem (voz/den)")
                ax.set_title("Screenline: pozorované vs modelované")
                ax.legend(fontsize=8)
                ax.invert_yaxis()

                ax = axes[1]
                r_vals = [r["ratio"] for r in eligible]
                colors = ["green" if 0.8 <= r <= 1.2 else "orange" if 0.5 <= r <= 2.0 else "red" for r in r_vals]
                ax.barh(range(len(eligible)), r_vals, color=colors, edgecolor="white")
                ax.axvline(1.0, color="black", ls="-", lw=0.8)
                ax.axvline(0.8, color="green", ls="--", lw=0.7, alpha=0.6)
                ax.axvline(1.2, color="green", ls="--", lw=0.7, alpha=0.6)
                ax.set_yticks(range(len(eligible)))
                ax.set_yticklabels(names, fontsize=7)
                ax.set_xlabel("Poměr model/pozorované")
                ax.set_title("Screenline: poměr mod/obs")
                ax.invert_yaxis()

                fig.tight_layout()
                save_figure(fig, out_dir / "screenline_validation.png")
                plt.close(fig)
            break

    # --- Trip length distribution from skims ---
    try:
        from aequilibrae.matrix import AequilibraeMatrix

        demand_cfg = cfg.get("demand", {})
        skim_path = Path(demand_cfg.get("output_dir", "")) / "skims.aem"
        matrix_path = Path(demand_cfg.get("matrix_path", ""))

        if skim_path.exists() and matrix_path.exists():
            skims = AequilibraeMatrix()
            skims.load(str(skim_path))

            mat = AequilibraeMatrix()
            mat.load(str(matrix_path))

            skim_core = None
            for cn in ("distance_blended", "distance_min", "free_flow_time_min"):
                if cn in skims.names:
                    skim_core = cn
                    break
            if skim_core is None and skims.names:
                skim_core = skims.names[0]

            calib_cfg = cfg.get("calibration", {})
            core_name = str(calib_cfg.get("core_name", "wd_daily"))
            od_core = core_name if core_name in mat.names else (mat.names[0] if mat.names else None)

            if skim_core and od_core:
                skims.computational_view([skim_core])
                mat.computational_view([od_core])

                dist_vals = skims.matrix_view.flatten()
                trip_vals = mat.matrix_view.flatten()

                mask = (trip_vals > 0) & np.isfinite(dist_vals) & (dist_vals > 0)
                distances = dist_vals[mask]
                trips = trip_vals[mask]

                if skim_core.startswith("free_flow_time"):
                    dist_label = "Cestovní čas (min)"
                    unit = "min"
                else:
                    distances = distances / 1000.0
                    dist_label = "Vzdálenost (km)"
                    unit = "km"

                if len(distances) > 10:
                    weighted_mean = float(np.average(distances, weights=trips))
                    weighted_median_idx = np.searchsorted(
                        np.cumsum(trips[np.argsort(distances)]) / trips.sum(), 0.5,
                    )
                    sorted_d = np.sort(distances)
                    weighted_median = float(sorted_d[min(weighted_median_idx, len(sorted_d) - 1)])

                    summary["trip_length_distribution"] = {
                        "skim_core": skim_core,
                        "unit": unit,
                        "n_od_pairs": int(mask.sum()),
                        "total_trips": round(float(trips.sum()), 0),
                        "weighted_mean": round(weighted_mean, 2),
                        "weighted_median": round(weighted_median, 2),
                        "p10": round(float(np.percentile(np.repeat(distances, trips.astype(int).clip(1, 100)), 10)), 2),
                        "p90": round(float(np.percentile(np.repeat(distances, trips.astype(int).clip(1, 100)), 90)), 2),
                    }

                    fig, ax = plt.subplots(figsize=(10, 5))
                    max_d = min(float(np.percentile(distances, 98)), weighted_mean * 4)
                    bins = np.linspace(0, max_d, 50)
                    ax.hist(distances, bins=bins, weights=trips, color="steelblue",
                            edgecolor="white", alpha=0.85, density=True)
                    ax.axvline(weighted_mean, color="red", ls="--", lw=1.2,
                               label=f"Vážený průměr: {weighted_mean:.1f} {unit}")
                    ax.axvline(weighted_median, color="orange", ls="--", lw=1.2,
                               label=f"Vážený medián: {weighted_median:.1f} {unit}")
                    ax.set_xlabel(dist_label)
                    ax.set_ylabel("Hustota (normalizovaná)")
                    ax.set_title("Distribuce délky cest (váženo počtem cest)")
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    save_figure(fig, out_dir / "trip_length_distribution.png")
                    plt.close(fig)

            skims.close()
            mat.close()
    except Exception as e:
        summary["trip_length_warning"] = f"Could not compute trip length distribution: {e}"

    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
