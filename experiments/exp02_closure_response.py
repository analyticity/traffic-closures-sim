#!/usr/bin/env python3
"""Experiment 02: Closure impact analysis — rerouting, congestion & system KPIs.

For a small set of real closures (full closure on matched links):
- System-level KPIs: ΔVHT, ΔVKT, change in overloaded links
- Volume drop on closed links (sanity check: volume → 0)
- Top links by absolute Δvolume — where does traffic reroute?
- Δvolume distribution histogram (how many links affected, by how much)
- GeoJSON delta map for interactive exploration
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

from _common import (
    city_display_name,
    compute_scenario_kpis,
    ensure_metric_links,
    init_experiment,
    load_baseline_links,
    load_closures,
    match_links_near_point,
    null_baseline_assignment,
    run_scenario_assignment,
    save_csv,
    save_figure,
    save_geojson,
    save_json,
    top_n_links_by_delta,
)

logger = logging.getLogger(__name__)
NAME = "exp02_closure_response"

N_CLOSURES = 3
MAP_BUFFER_M = 1500.0
VOL_COL = "wd_daily_tot"
DELTA_THRESHOLD = 500


def _select_closures(closures: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """Pick *N_CLOSURES* closures on higher functional classes when possible."""
    c = closures.copy()
    sev_col = "pg_severity" if "pg_severity" in c.columns else "severity"
    if "quality_score" in c.columns:
        c["_qs"] = pd.to_numeric(c["quality_score"], errors="coerce").fillna(0)
        c = c.sort_values("_qs", ascending=False)
    major = {"motorway", "trunk", "primary", "secondary"}
    rtc = c.get("road_type_code", pd.Series([""] * len(c)))
    is_major = rtc.isin(major)
    in_city = (
        c["city"].str.contains(city_name, case=False, na=False)
        if "city" in c.columns and city_name
        else pd.Series(True, index=c.index)
    )
    lat_ok = c["lat"].notna() & c["lon"].notna()

    pool = c[lat_ok & (is_major | in_city)]
    if len(pool) < N_CLOSURES:
        pool = c[lat_ok]

    chosen: List[pd.Series] = []
    seen_ids: Set[int] = set()
    for _, row in pool.iterrows():
        rid = int(row.get("id", -1))
        if rid in seen_ids:
            continue
        chosen.append(row)
        seen_ids.add(rid)
        if len(chosen) >= N_CLOSURES:
            break
    return pd.DataFrame(chosen) if chosen else pd.DataFrame()


def _closure_to_scenario_links_full(
    closure: pd.Series,
    links: gpd.GeoDataFrame,
    *,
    max_dist_m: float = 200.0,
    max_links: int = 50,
) -> List[Dict[str, Any]]:
    """Map closure point to nearby links; apply **full** closure (both directions)."""
    lat, lon = closure.get("lat"), closure.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        return []

    nearby = match_links_near_point(
        float(lat), float(lon), links, max_dist_m=max_dist_m, max_links=max_links,
    )
    direction = "both"
    raw_dir = str(closure.get("closure_direction", "")).strip().lower()
    if raw_dir == "aligned":
        direction = "ab"
    elif raw_dir == "opposite":
        direction = "ba"

    result: List[Dict[str, Any]] = []
    for _, link in nearby.iterrows():
        lanes = max(int(link.get("lanes", 2) or 2), 1)
        result.append({
            "link_id": int(link["link_id"]),
            "direction": direction,
            "closure_type": "full",
            "lanes": lanes,
            "lanes_remaining": 0,
        })
    return result


def _delta_frame(
    baseline_merged: pd.DataFrame,
    scenario_df: pd.DataFrame,
    vol_col: str = VOL_COL,
) -> pd.DataFrame:
    b = baseline_merged[["link_id", vol_col]].rename(columns={vol_col: "vol_base"})
    s = scenario_df[["link_id", vol_col]].rename(columns={vol_col: "vol_scen"})
    out = b.merge(s, on="link_id", how="outer")
    out["vol_base"] = out["vol_base"].fillna(0)
    out["vol_scen"] = out["vol_scen"].fillna(0)
    out["delta_vol"] = out["vol_scen"] - out["vol_base"]
    out["abs_delta_vol"] = out["delta_vol"].abs()
    return out


def _buffer_delta_geojson(
    links: gpd.GeoDataFrame,
    delta_df: pd.DataFrame,
    lat: float,
    lon: float,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    mlinks = ensure_metric_links(links)
    pt = gpd.GeoDataFrame(geometry=[Point(float(lon), float(lat))], crs="EPSG:4326").to_crs(
        mlinks.crs,
    ).geometry.iloc[0]
    buf = pt.buffer(buffer_m)
    sel = mlinks[mlinks.geometry.intersects(buf)].copy()
    if sel.empty:
        sel = mlinks.loc[[mlinks.geometry.distance(pt).idxmin()]].copy()

    sel = sel.merge(delta_df, on="link_id", how="left")
    sel["delta_vol"] = sel["delta_vol"].fillna(0)
    sel["abs_delta_vol"] = sel["abs_delta_vol"].fillna(0)
    return sel


def main() -> None:
    cfg, out_dir = init_experiment(NAME)

    closures = load_closures(cfg)
    links = load_baseline_links(cfg)

    # Deltas are taken against a null run — the untouched network solved by the
    # same code path as the scenarios — not against assignment_results.parquet,
    # which comes from a different run and carries a large noise floor.
    baseline_assign = null_baseline_assignment(cfg)

    from sim._metrics import aggregate_daily_volumes

    aggregate_daily_volumes(baseline_assign)

    baseline_merged = links.copy()
    for c in baseline_assign.columns:
        if c != "link_id" and c not in baseline_merged.columns:
            baseline_merged = baseline_merged.merge(
                baseline_assign[["link_id", c]], on="link_id", how="left",
            )
    aggregate_daily_volumes(baseline_merged)

    selected = _select_closures(closures, city_display_name(cfg))
    if selected.empty:
        print(f"[{NAME}] No closures with coordinates.")
        return

    all_results: List[Dict[str, Any]] = []
    all_deltas: List[pd.DataFrame] = []

    for idx, (_, closure) in enumerate(selected.iterrows(), start=1):
        label = f"closure_{idx}"
        desc = str(closure.get("description_cs", closure.get("road_number", f"id_{closure['id']}")))[:80]
        scenario_links = _closure_to_scenario_links_full(closure, links)
        if not scenario_links:
            logger.warning("Skipping %s — no matched links", label)
            continue

        closed_ids = {int(sl["link_id"]) for sl in scenario_links}

        # Skip closures whose links already carry no traffic in baseline.
        base_vol_on_closed = baseline_merged.loc[
            baseline_merged["link_id"].isin(closed_ids), VOL_COL
        ].sum() if VOL_COL in baseline_merged.columns else 1.0
        if base_vol_on_closed <= 0:
            logger.info("Skipping %s — links already have 0 baseline volume (closure already active)", label)
            continue

        try:
            scenario_df = run_scenario_assignment(cfg, scenario_links)
        except Exception as e:
            logger.error("%s assignment failed: %s", label, e)
            continue

        aggregate_daily_volumes(scenario_df)
        delta_df = _delta_frame(baseline_merged, scenario_df, VOL_COL)

        closed_rows = delta_df[delta_df["link_id"].isin(closed_ids)]
        vol_drop = {
            "n_closed_links": len(closed_ids),
            "mean_vol_base_on_closed": round(float(closed_rows["vol_base"].mean()), 1)
            if len(closed_rows)
            else None,
            "mean_vol_scen_on_closed": round(float(closed_rows["vol_scen"].mean()), 1)
            if len(closed_rows)
            else None,
            "sum_vol_base_on_closed": round(float(closed_rows["vol_base"].sum()), 1)
            if len(closed_rows)
            else None,
            "sum_vol_scen_on_closed": round(float(closed_rows["vol_scen"].sum()), 1)
            if len(closed_rows)
            else None,
        }

        top = top_n_links_by_delta(
            baseline_merged,
            scenario_df,
            vol_col=VOL_COL,
            n=15,
            exclude_link_ids=closed_ids,
        )
        # Enrich top links with attributes
        attr_cols = [c for c in ("name", "osm_ref", "link_type") if c in baseline_merged.columns]
        if not top.empty and attr_cols:
            top = top.merge(
                baseline_merged[["link_id"] + attr_cols].drop_duplicates("link_id"),
                on="link_id",
                how="left",
            )
        save_csv(top, out_dir / f"{label}_top_links.csv")

        # System-level KPIs
        kpis = compute_scenario_kpis(baseline_merged, scenario_df)

        n_high = int((delta_df["abs_delta_vol"] > DELTA_THRESHOLD).sum())
        total_abs = float(delta_df["abs_delta_vol"].sum())

        lat, lon = float(closure["lat"]), float(closure["lon"])
        gj = _buffer_delta_geojson(links, delta_df, lat, lon, MAP_BUFFER_M)
        keep_cols = ["link_id", "delta_vol", "abs_delta_vol", "vol_base", "vol_scen", "geometry"]
        keep_cols = [c for c in keep_cols if c in gj.columns]
        save_geojson(gj[keep_cols], out_dir / f"{label}_delta_map.geojson")

        rec: Dict[str, Any] = {
            "label": label,
            "closure_id": int(closure["id"]),
            "description": desc,
            "n_scenario_links": len(scenario_links),
            "volume_on_closed_links": vol_drop,
            "system_kpis": kpis,
            "n_links_abs_delta_gt_threshold": n_high,
            "threshold_vol": DELTA_THRESHOLD,
            "sum_abs_delta_vol_network": round(total_abs, 1),
        }
        all_results.append(rec)
        all_deltas.append(delta_df.assign(_closure=label))
        save_json(rec, out_dir / f"{label}_summary.json")

    if not all_results:
        save_json({"closures": [], "status": "no_valid_closures"}, out_dir / "summary.json")
        print(f"[{NAME}] No valid closures processed.")
        return

    # --- Visualization 1: Closed-link volume sanity ---
    labels = [r["label"] for r in all_results]
    base_means = [
        (r["volume_on_closed_links"].get("mean_vol_base_on_closed") or 0) for r in all_results
    ]
    scen_means = [
        (r["volume_on_closed_links"].get("mean_vol_scen_on_closed") or 0) for r in all_results
    ]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, base_means, w, label="Baseline prům. objem (uzavřené)")
    ax.bar(x + w / 2, scen_means, w, label="Scénář prům. objem (uzavřené)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Průměr {VOL_COL}")
    ax.set_title("Sanity: objem na uzavřených hranách klesá?")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir / "rerouting_summary.png")
    plt.close(fig)

    # --- Visualization 2: System KPI comparison across closures ---
    kpi_df = pd.DataFrame([
        {
            "closure": r["label"],
            "description": r["description"][:40],
            "ΔVHT (voz·h)": r["system_kpis"]["delta_vht"],
            "ΔVHT (%)": r["system_kpis"]["delta_vht_pct"],
            "ΔVKT (voz·km)": r["system_kpis"]["delta_vkt"],
            "Δ přetížených": r["system_kpis"]["delta_overloaded"],
        }
        for r in all_results
    ])
    save_csv(kpi_df, out_dir / "closure_kpis.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.barh(kpi_df["closure"], kpi_df["ΔVHT (voz·h)"], color="steelblue")
    ax.set_xlabel("ΔVHT (voz·h)")
    ax.set_title("Dopad na celkový čas v síti")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(kpi_df["closure"], kpi_df["ΔVHT (%)"], color="coral")
    ax.set_xlabel("ΔVHT (%)")
    ax.set_title("Relativní dopad na VHT")
    ax.invert_yaxis()

    ax = axes[2]
    ax.barh(kpi_df["closure"], kpi_df["Δ přetížených"], color="darkred")
    ax.set_xlabel("Δ počet přetížených hran")
    ax.set_title("Změna přetížených hran")
    ax.invert_yaxis()

    fig.suptitle("Systémový dopad uzavírek", fontsize=14)
    fig.tight_layout()
    save_figure(fig, out_dir / "system_kpis.png")
    plt.close(fig)

    # --- Visualization 3: Delta volume distribution ---
    if all_deltas:
        combined_delta = pd.concat(all_deltas, ignore_index=True)
        for cl_label in combined_delta["_closure"].unique():
            cl_data = combined_delta[combined_delta["_closure"] == cl_label]
            nonzero = cl_data[cl_data["abs_delta_vol"] > 10]["delta_vol"]
            if len(nonzero) < 5:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            clip_val = float(np.percentile(nonzero.abs(), 95))
            bins = np.linspace(-clip_val, clip_val, 60)
            ax.hist(nonzero.clip(-clip_val, clip_val), bins=bins,
                    color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Δ objem (voz/den)")
            ax.set_ylabel("Počet hran")
            ax.set_title(f"{cl_label}: distribuce změn objemu (hrany s |Δ|>10)")
            ax.grid(True, alpha=0.3)
            save_figure(fig, out_dir / f"{cl_label}_delta_distribution.png")
            plt.close(fig)

    save_json({"closures": all_results}, out_dir / "summary.json")
    print(f"[{NAME}] Done — {len(all_results)} closures → {out_dir}")


if __name__ == "__main__":
    main()
