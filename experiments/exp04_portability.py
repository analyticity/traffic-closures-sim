#!/usr/bin/env python3
"""Experiment 04: Cross-city portability comparison.

Evaluates whether the same pipeline (same code, same defaults) produces
credible baseline models for multiple Czech cities.  Compares key validation
metrics side-by-side for every city whose pipeline outputs exist.

Metrics compared:
- R², slope, %RMSE, bias from CSD matched diagnostics
- Factor-of-2 percentage
- Screenline mean ratio & deviation
- System aggregates: VMT, VHT, mean V/C, overloaded link share
- Network size and complexity indicators

The experiment reads outputs from already-run pipelines — it does NOT rerun
assignments.  Cities with missing outputs are gracefully skipped.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    compute_factor_of_2_pct,
    compute_vht,
    compute_vkt,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp04_portability"

CITY_CONFIGS = [
    "config/brno/sim.yaml",
    "config/olomouc/sim.yaml",
    "config/most/sim.yaml",
]


def _load_city_data(config_path: str) -> Optional[Dict[str, Any]]:
    """Load pipeline outputs for one city. Returns None if critical files missing."""
    try:
        import sys
        _REPO = Path(__file__).resolve().parent.parent
        if str(_REPO / "src") not in sys.path:
            sys.path.insert(0, str(_REPO / "src"))
        from sim.io_project import load_config
        from sim._metrics import aggregate_daily_volumes
    except ImportError:
        return None

    cfg_path = _REPO / config_path
    if not cfg_path.exists():
        return None

    try:
        cfg = load_config(str(cfg_path))
    except Exception as e:
        logger.warning("Cannot load config %s: %s", config_path, e)
        return None

    city_slug = Path(config_path).parts[-2]
    demand_dir = Path(cfg.get("demand", {}).get("output_dir", ""))
    network_dir = Path(cfg.get("network", {}).get("output_dir", ""))

    assign_path = demand_dir / "assignment_results.parquet"
    if not assign_path.exists():
        logger.info("Skipping %s — no assignment results", city_slug)
        return None

    result: Dict[str, Any] = {"city": city_slug, "config": config_path}

    # Assignment results → system aggregates
    try:
        import geopandas as gpd
        assign = pd.read_parquet(assign_path)
        aggregate_daily_volumes(assign)

        for ext in ("gpkg", "parquet", "geojson"):
            lp = network_dir / f"network_links.{ext}"
            if lp.exists():
                links = gpd.read_file(lp) if ext != "parquet" else gpd.GeoDataFrame(pd.read_parquet(lp))
                break
        else:
            links = None

        if links is not None:
            vol_cols = [c for c in assign.columns if c != "link_id" and c not in links.columns]
            merged = links.merge(assign[["link_id"] + vol_cols], on="link_id", how="left")
            aggregate_daily_volumes(merged)
        else:
            merged = assign

        vht = compute_vht(merged)
        vkt = compute_vkt(merged)
        n_overloaded = int((merged.get("VOC_max", pd.Series(dtype=float)).fillna(0) > 1.0).sum())
        result["system"] = {
            "n_links": len(merged),
            "total_vht": round(vht, 1),
            "total_vkt": round(vkt, 1),
            "n_overloaded": n_overloaded,
            "pct_overloaded": round(n_overloaded / max(len(merged), 1) * 100, 2),
            "mean_voc": round(float(merged["VOC_max"].mean()), 4) if "VOC_max" in merged.columns else None,
        }
    except Exception as e:
        logger.warning("%s: system aggregates failed: %s", city_slug, e)
        result["system"] = None

    # CSD matched diagnostics
    diag_path = demand_dir / "matching_diagnostics.csv"
    if diag_path.exists():
        try:
            diag = pd.read_csv(diag_path)
            obs_candidates = ["observed", "observed_total", "observed_motor_total"]
            mod_candidates = ["modeled", "mod", "modeled_total", "_corridor_volume", "lw_mean", "model_lw_mean"]
            obs_col = next((c for c in obs_candidates if c in diag.columns), None)
            mod_col = next((c for c in mod_candidates if c in diag.columns), None)

            if obs_col and mod_col:
                obs = pd.to_numeric(diag[obs_col], errors="coerce").values
                mod = pd.to_numeric(diag[mod_col], errors="coerce").values
                valid = np.isfinite(obs) & np.isfinite(mod) & (obs > 0)
                obs_v, mod_v = obs[valid], mod[valid]

                from sim.calibration.metrics import compute_stats
                stats = compute_stats(mod_v, obs_v)
                f2 = compute_factor_of_2_pct(mod_v, obs_v)
                result["count_validation"] = {**stats, **f2}
        except Exception as e:
            logger.warning("%s: CSD diagnostics failed: %s", city_slug, e)

    # Validation / calibration report → screenlines
    for rn in ("validation_report.json", "calibration_report.json"):
        rp = demand_dir / rn
        if not rp.exists():
            continue
        try:
            report = json.loads(rp.read_text(encoding="utf-8"))
            sl_data = report.get("screenlines") or report.get("screenline_results")
            if sl_data and isinstance(sl_data, dict):
                ratios = []
                for sl_info in sl_data.values():
                    if not isinstance(sl_info, dict):
                        continue
                    obs = sl_info.get("observed_total", 0)
                    mod = sl_info.get("modeled_total", 0)
                    if obs > 0:
                        ratios.append(mod / obs)
                if ratios:
                    result["screenlines"] = {
                        "source": rn,
                        "n_screenlines": len(ratios),
                        "mean_ratio": round(float(np.mean(ratios)), 3),
                        "median_ratio": round(float(np.median(ratios)), 3),
                        "max_deviation_pct": round(max(abs(r - 1.0) for r in ratios) * 100, 1),
                        "pct_within_20pct": round(
                            sum(1 for r in ratios if 0.8 <= r <= 1.2) / len(ratios) * 100, 1
                        ),
                    }
                break
        except (json.JSONDecodeError, OSError):
            continue

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=f"Experiment: {NAME}")
    parser.add_argument("--config", default="config/brno/sim.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    from _common import EXPERIMENTS_OUTPUT, city_from_config
    city = city_from_config(args.config)
    out_dir = EXPERIMENTS_OUTPUT / city / NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    city_results: List[Dict[str, Any]] = []
    for cfg_path in CITY_CONFIGS:
        data = _load_city_data(cfg_path)
        if data is not None:
            city_results.append(data)
            logger.info("Loaded %s", data["city"])

    if len(city_results) < 2:
        save_json(
            {"status": "skipped", "reason": f"Need ≥2 cities, found {len(city_results)}", "experiment": NAME},
            out_dir / "summary.json",
        )
        print(f"[{NAME}] Skipped — fewer than 2 cities available")
        return

    # Build comparison table
    rows = []
    for cr in city_results:
        row: Dict[str, Any] = {"city": cr["city"]}
        cv = cr.get("count_validation", {})
        row["R²"] = cv.get("r2")
        row["slope"] = cv.get("slope")
        row["pct_rmse"] = cv.get("pct_rmse")
        row["bias_pct"] = cv.get("bias_pct")
        row["factor_of_2_pct"] = cv.get("pct_in_factor_of_2")
        row["n_counts"] = cv.get("n")

        sl = cr.get("screenlines", {})
        row["sl_mean_ratio"] = sl.get("mean_ratio")
        row["sl_pct_within_20"] = sl.get("pct_within_20pct")

        sys = cr.get("system", {}) or {}
        row["n_links"] = sys.get("n_links")
        row["vht"] = sys.get("total_vht")
        row["vkt"] = sys.get("total_vkt")
        row["pct_overloaded"] = sys.get("pct_overloaded")
        row["mean_voc"] = sys.get("mean_voc")
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    save_csv(comp_df, out_dir / "city_comparison.csv")

    # --- Visualization: radar/comparison charts ---
    cities = comp_df["city"].tolist()
    n_cities = len(cities)

    # 1) Bar chart: key metrics side-by-side
    metrics_to_plot = [
        ("R²", "R²", None),
        ("factor_of_2_pct", "Factor-of-2 (%)", None),
        ("pct_overloaded", "Přetíž. hrany (%)", None),
    ]
    available_metrics = [(col, label, _) for col, label, _ in metrics_to_plot if col in comp_df.columns and comp_df[col].notna().any()]

    if available_metrics:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]

        city_colors = plt.cm.Set2(np.linspace(0, 1, n_cities))
        for ax, (col, label, _) in zip(axes, available_metrics):
            vals = comp_df[col].fillna(0).values
            ax.bar(cities, vals, color=city_colors[:n_cities], edgecolor="white")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Porovnání validačních metrik napříč městy", fontsize=14)
        fig.tight_layout()
        save_figure(fig, out_dir / "city_comparison_bars.png")
        plt.close(fig)

    # 2) Scatter: R² vs network size
    if "R²" in comp_df.columns and "n_links" in comp_df.columns:
        valid = comp_df.dropna(subset=["R²", "n_links"])
        if len(valid) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(valid["n_links"], valid["R²"], s=120, zorder=5, color="steelblue")
            for _, row in valid.iterrows():
                ax.annotate(row["city"], (row["n_links"], row["R²"]),
                            textcoords="offset points", xytext=(8, 8), fontsize=11)
            ax.set_xlabel("Počet hran v síti")
            ax.set_ylabel("R² (CSD validace)")
            ax.set_title("Přenositelnost: R² vs. velikost sítě")
            ax.grid(True, alpha=0.3)
            save_figure(fig, out_dir / "r2_vs_network_size.png")
            plt.close(fig)

    # 3) Comprehensive comparison table as figure
    fig, ax = plt.subplots(figsize=(12, max(3, 1 + n_cities * 0.6)))
    ax.axis("off")
    display_cols = [c for c in ["city", "n_links", "R²", "slope", "bias_pct",
                                 "factor_of_2_pct", "sl_mean_ratio", "pct_overloaded"]
                    if c in comp_df.columns]
    cell_text = []
    for _, row in comp_df[display_cols].iterrows():
        cell_text.append([
            str(v) if pd.notna(v) else "—"
            for v in row
        ])

    col_labels = {
        "city": "Město", "n_links": "Hrany", "R²": "R²", "slope": "Sklon",
        "bias_pct": "Bias (%)", "factor_of_2_pct": "F2 (%)",
        "sl_mean_ratio": "SL ratio", "pct_overloaded": "Přetíž. (%)",
    }
    headers = [col_labels.get(c, c) for c in display_cols]

    table = ax.table(cellText=cell_text, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#e8e8e8"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Srovnání validačních metrik: přenositelnost pipeline", fontsize=13, pad=20)
    save_figure(fig, out_dir / "comparison_table.png")
    plt.close(fig)

    summary = {
        "experiment": NAME,
        "cities_compared": cities,
        "n_cities": n_cities,
        "per_city": city_results,
    }
    save_json(summary, out_dir / "summary.json")
    print(f"[{NAME}] Done — {n_cities} cities compared → {out_dir}")


if __name__ == "__main__":
    main()
