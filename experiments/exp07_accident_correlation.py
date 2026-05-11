#!/usr/bin/env python3
"""Experiment 07: Traffic accidents vs model congestion.

Correlates baseline V/C ratios with accident blackspot frequency.
Also fits a simple logistic regression P(accident) ~ V/C + road_type.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    _cache_dir,
    init_experiment,
    load_assignment_results,
    load_network_links,
    load_segment_map,
    save_csv,
    save_figure,
    save_json,
)

logger = logging.getLogger(__name__)
NAME = "exp07_accident_correlation"


def _load_accidents(cfg: dict) -> pd.DataFrame:
    """Load accidents from cache or fetch from DB."""
    cache = _cache_dir(cfg)
    path = cache / "accidents.parquet"
    if path.exists():
        return pd.read_parquet(path)

    # Fallback: try to fetch directly
    db_cfg = cfg.get("datasets", {}).get("postgres", {})
    if not db_cfg:
        raise FileNotFoundError(
            f"accidents.parquet not found at {path} and no postgres config available. "
            "Run fetch-data first."
        )

    import psycopg2
    conn = psycopg2.connect(**db_cfg)
    try:
        df = pd.read_sql("""
            SELECT id, segment_id, severity, lat, lon,
                   created_utc, road_type
            FROM accidents
        """, conn)
    finally:
        conn.close()

    df.to_parquet(path, index=False)
    return df


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    links = load_network_links(cfg)
    assign = load_assignment_results(cfg)
    seg_map = load_segment_map(cfg, links)

    from sim._metrics import aggregate_daily_volumes
    aggregate_daily_volumes(assign)

    try:
        accidents = _load_accidents(cfg)
    except FileNotFoundError as e:
        print(f"[{NAME}] {e}")
        return

    # Map accidents to link_ids
    if "segment_id" in accidents.columns:
        accidents["link_id"] = accidents["segment_id"].map(seg_map)
    else:
        logger.warning("No segment_id in accidents — attempting spatial join")
        import geopandas as _gpd
        from shapely.geometry import Point

        if "lat" in accidents.columns and "lon" in accidents.columns:
            valid = accidents.dropna(subset=["lat", "lon"])
            geom = [Point(r["lon"], r["lat"]) for _, r in valid.iterrows()]
            acc_gdf = _gpd.GeoDataFrame(valid, geometry=geom, crs="EPSG:4326")
            if links.crs and links.crs.to_epsg() != 4326:
                acc_gdf = acc_gdf.to_crs(links.crs)
            joined = _gpd.sjoin_nearest(acc_gdf, links[["link_id", "geometry"]], how="left", max_distance=150)
            accidents = pd.DataFrame(joined)
        else:
            print(f"[{NAME}] Cannot map accidents to links.")
            return

    acc_per_link = (
        accidents.dropna(subset=["link_id"])
        .groupby("link_id")
        .size()
        .reset_index(name="accident_count")
    )
    acc_per_link["link_id"] = acc_per_link["link_id"].astype(int)

    # Build analysis dataframe
    net = links[["link_id", "link_type"]].merge(assign, on="link_id", how="left")
    aggregate_daily_volumes(net)

    if "VOC_max" in net.columns:
        net["vc"] = net["VOC_max"].fillna(0)
    elif "capacity_ab" in links.columns:
        cap = links.set_index("link_id")[["capacity_ab", "capacity_ba"]].max(axis=1)
        vol = net.set_index("link_id").get("wd_daily_tot", pd.Series(dtype=float)).fillna(0)
        net["vc"] = (vol / cap.clip(lower=1)).values
    else:
        net["vc"] = 0

    df = net.merge(acc_per_link, on="link_id", how="left")
    df["accident_count"] = df["accident_count"].fillna(0).astype(int)

    # Only keep links with vc > 0
    df = df[df["vc"] > 0].copy()

    if df.empty:
        print(f"[{NAME}] No data after merge.")
        return

    # ------------------------------------------------------------------
    # Spearman correlation
    # ------------------------------------------------------------------
    has_acc = df[df["accident_count"] > 0]
    if len(has_acc) >= 5:
        rho, p_val = sp_stats.spearmanr(has_acc["vc"], has_acc["accident_count"])
    else:
        rho, p_val = float("nan"), float("nan")

    summary = {
        "n_links_total": len(df),
        "n_links_with_accidents": len(has_acc),
        "total_accidents": int(df["accident_count"].sum()),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_val),
    }

    # ------------------------------------------------------------------
    # Box plot: accidents by V/C quartile
    # ------------------------------------------------------------------
    df["vc_quartile"] = pd.qcut(df["vc"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                                 duplicates="drop")

    if df["vc_quartile"].nunique() >= 2:
        quartile_stats = df.groupby("vc_quartile", observed=True).agg(
            links=("link_id", "count"),
            links_with_accident=("accident_count", lambda x: (x > 0).sum()),
            total_accidents=("accident_count", "sum"),
            mean_accidents=("accident_count", "mean"),
        ).reset_index()
        save_csv(quartile_stats, out_dir / "quartile_stats.csv")

        fig, ax = plt.subplots(figsize=(7, 5))
        df.boxplot(column="accident_count", by="vc_quartile", ax=ax, showfliers=False)
        ax.set_xlabel("V/C kvartil")
        ax.set_ylabel("Počet nehod")
        ax.set_title("Distribuce nehod podle V/C kvartilů")
        fig.suptitle("")
        save_figure(fig, out_dir / "boxplot.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Scatter (aggregated)
    # ------------------------------------------------------------------
    if len(has_acc) >= 5:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(has_acc["vc"], has_acc["accident_count"], alpha=0.3, s=10, edgecolors="none")
        ax.set_xlabel("V/C poměr")
        ax.set_ylabel("Počet nehod")
        ax.set_title(f"V/C vs nehody  (ρ = {rho:.3f},  n = {len(has_acc)})")
        save_figure(fig, out_dir / "correlation.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Logistic regression: P(any accident) ~ V/C + road_type
    # ------------------------------------------------------------------
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        logreg_df = df[["vc", "link_type", "accident_count"]].dropna().copy()
        logreg_df["has_accident"] = (logreg_df["accident_count"] > 0).astype(int)

        le = LabelEncoder()
        logreg_df["road_type_enc"] = le.fit_transform(logreg_df["link_type"].astype(str))

        X = logreg_df[["vc", "road_type_enc"]].values
        y = logreg_df["has_accident"].values

        if y.sum() >= 10 and (len(y) - y.sum()) >= 10:
            model = LogisticRegression(max_iter=500)
            model.fit(X, y)
            summary["logreg_accuracy"] = round(float(model.score(X, y)), 3)
            summary["logreg_vc_coef"] = round(float(model.coef_[0][0]), 4)
            summary["logreg_road_coef"] = round(float(model.coef_[0][1]), 4)
    except ImportError:
        logger.info("sklearn not available, skipping logistic regression")

    save_json(summary, out_dir / "correlation.json")
    print(f"[{NAME}] Done → {out_dir}")


if __name__ == "__main__":
    main()
