"""Iterative FSM calibration and independent validation.

Calibration loop
----------------
1. Run traffic assignment with current OD matrix
2. Match assigned link volumes to observed counts (pentlogram)
3. Compute GEH / RMSE metrics
4. If converged → stop
5. Scale OD matrix using observed/modeled ratios (global or sector-based)
6. Go to 1

Validation
----------
After calibration converges, ``validate`` compares the *final* assignment
to an independent dataset (CSD2020) that was **not** used during calibration.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import load_config
from sim.assignment import execute_assignment, fix_node_ids, _detect_volume_col

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get(cfg: Any, path: List[str], default: Any = None) -> Any:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# Observed data loaders
# ---------------------------------------------------------------------------

def load_pentlogram(cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    geojson_path = Path(_get(
        cfg,
        ["datasets", "sources", "calibration_brno_pentlogram_2024", "out_path"],
        "data/sources/brno/intensity/intenzita_dopravy_pentlogram_2024.geojson",
    ))
    if not geojson_path.exists():
        raise FileNotFoundError(f"Pentlogram not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    out_epsg = int(_get(
        cfg,
        ["datasets", "sources", "calibration_brno_pentlogram_2024", "output", "out_epsg"],
        5514,
    ))
    gdf = gdf.set_crs(epsg=out_epsg, allow_override=True)

    for col in ("car_24", "truc_24"):
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    # Pentlogram data is in hundreds of vehicles/24h (Czech standard for flow diagrams)
    gdf["observed_total"] = (gdf.get("car_24", 0) + gdf.get("truc_24", 0)) * 100
    return gdf[gdf["observed_total"] > 0].copy()


def load_csd2020(cfg: Dict[str, Any], region_code: str = "CZ064") -> pd.DataFrame:
    cache_dir = Path(_get(cfg, ["datasets", "cache_dir"], "data/cache"))
    parquet_path = cache_dir / "v2_csd2020.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"CSD2020 parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "kk" in df.columns:
        df = df[df["kk"].astype(str).str.contains(region_code.replace("CZ", ""), na=False)].copy()
    for col in ("sv", "o", "tv"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------

def _load_network_links(project_dir: Path) -> gpd.GeoDataFrame:
    project = Project()
    project.open(str(project_dir))
    try:
        links_df = project.network.links.data.copy()
        crs = getattr(project.network, "crs", "EPSG:4326")
    finally:
        project.close()
    return gpd.GeoDataFrame(links_df, geometry="geometry", crs=crs)


def match_counts_to_links(
    counts: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    *,
    buffer_m: float = 50.0,
    metric_epsg: int = 5514,
    id_col: str = "objectid",
) -> gpd.GeoDataFrame:
    """Spatial-join observed count points/lines to nearest network links (metric CRS)."""
    counts = counts.to_crs(epsg=metric_epsg)
    links = links.to_crs(epsg=metric_epsg)

    pts = counts.copy()
    pts["geometry"] = pts.geometry.centroid

    keep = ["link_id", "link_type", "name", "geometry"] + [
        c for c in links.columns
        if c not in ("link_id", "link_type", "name", "geometry", "ogc_fid")
        and links[c].dtype in ("float64", "float32", "int64")
    ]
    keep = [c for c in keep if c in links.columns]

    joined = gpd.sjoin_nearest(pts, links[keep], how="left",
                               max_distance=buffer_m, distance_col="_dist")
    if id_col in joined.columns:
        joined = joined.drop_duplicates(subset=[id_col], keep="first")
    return joined


# ---------------------------------------------------------------------------
# Statistics: GEH, R², RMSE
# ---------------------------------------------------------------------------

def compute_geh(modeled: np.ndarray, observed: np.ndarray) -> np.ndarray:
    m, c = np.asarray(modeled, dtype=float), np.asarray(observed, dtype=float)
    denom = m + c
    mask = denom > 0
    geh = np.full_like(m, np.nan)
    geh[mask] = np.sqrt(2.0 * (m[mask] - c[mask]) ** 2 / denom[mask])
    return geh


def compute_stats(modeled: np.ndarray, observed: np.ndarray) -> Dict[str, Any]:
    m = np.asarray(modeled, dtype=float)
    c = np.asarray(observed, dtype=float)
    valid = np.isfinite(m) & np.isfinite(c) & (c > 0)
    m, c = m[valid], c[valid]
    n = len(m)
    if n == 0:
        return {"n": 0}

    geh = compute_geh(m, c)
    rmse = float(np.sqrt(np.mean((m - c) ** 2)))
    mean_obs = float(np.mean(c))
    pct_rmse = rmse / mean_obs * 100 if mean_obs > 0 else float("nan")

    ss_res = float(np.sum((m - c) ** 2))
    ss_tot = float(np.sum((c - np.mean(c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n": int(n),
        "r2": round(r2, 4) if np.isfinite(r2) else None,
        "rmse": round(rmse, 1),
        "pct_rmse": round(pct_rmse, 1) if np.isfinite(pct_rmse) else None,
        "geh_mean": round(float(np.nanmean(geh)), 2),
        "geh_median": round(float(np.nanmedian(geh)), 2),
        "geh_lt5_pct": round(float(np.nanmean(geh < 5) * 100), 1),
        "geh_lt10_pct": round(float(np.nanmean(geh < 10) * 100), 1),
        "sum_modeled": round(float(m.sum()), 0),
        "sum_observed": round(float(c.sum()), 0),
    }


# ---------------------------------------------------------------------------
# OD matrix scaling (FSM calibration step)
# ---------------------------------------------------------------------------

def _compute_global_factor(
    matched: pd.DataFrame,
    vol_col: str,
    *,
    damping: float,
    min_factor: float,
    max_factor: float,
) -> float:
    """Single factor = sum(observed) / sum(modeled), with damping."""
    m = matched[vol_col].values.astype(float)
    c = matched["observed_total"].values.astype(float)
    mask = (m > 0) & (c > 0)
    if mask.sum() == 0:
        return 1.0
    raw = float(c[mask].sum() / m[mask].sum())
    return float(np.clip(1.0 + damping * (raw - 1.0), min_factor, max_factor))


def _compute_sector_factors(
    matched: pd.DataFrame,
    vol_col: str,
    *,
    damping: float,
    min_factor: float,
    max_factor: float,
) -> Dict[str, float]:
    """Per-road-class factor = weighted-mean(observed / modeled) within class."""
    factors: Dict[str, float] = {}
    for lt in matched["link_type"].dropna().unique():
        sub = matched[matched["link_type"] == lt]
        m = sub[vol_col].values.astype(float)
        c = sub["observed_total"].values.astype(float)
        mask = (m > 0) & (c > 0)
        if mask.sum() < 3:
            continue
        raw = float(np.average(c[mask] / m[mask], weights=c[mask]))
        factors[str(lt)] = float(np.clip(1.0 + damping * (raw - 1.0), min_factor, max_factor))
    return factors


def scale_matrix(
    mat: AequilibraeMatrix,
    core_name: str,
    factor: float | Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> None:
    """Scale the matrix core in-place.

    *factor* is either a single float (global) or a dict of per-sector
    floats.  For the dict case, a weighted average is computed using
    *weights* (typically total observed volume per sector).
    """
    if isinstance(factor, dict):
        if not factor:
            return
        if weights:
            total_w = sum(weights.get(k, 1.0) for k in factor)
            avg = sum(f * weights.get(k, 1.0) for k, f in factor.items()) / max(total_w, 1e-9)
        else:
            avg = float(np.mean(list(factor.values())))
        mat.matrix[core_name][:, :] *= avg
    else:
        mat.matrix[core_name][:, :] *= factor


# ---------------------------------------------------------------------------
# Aggregate CSD2020 helpers (for validation)
# ---------------------------------------------------------------------------

def _classify_csd_road(sil: str) -> str:
    s = str(sil).strip().upper()
    if s.startswith("D"):
        return "motorway"
    try:
        num = int(s.replace("M", ""))
        if num < 100:
            return "primary"
        if num < 400:
            return "secondary"
        return "tertiary"
    except ValueError:
        return "primary" if "M" in s else "other"


def aggregate_csd_by_class(csd: pd.DataFrame) -> pd.DataFrame:
    csd = csd.copy()
    csd["road_class"] = csd["sil"].apply(_classify_csd_road)
    return csd.groupby("road_class").agg(
        sections=("sv", "count"),
        mean_sv=("sv", "mean"),
        mean_o=("o", "mean"),
    ).reset_index()


def aggregate_model_by_class(links: gpd.GeoDataFrame, vol_col: str) -> pd.DataFrame:
    if vol_col not in links.columns:
        return pd.DataFrame()
    return links.groupby("link_type").agg(
        links=("link_id", "count"),
        mean_vol=(vol_col, "mean"),
        total_vol=(vol_col, "sum"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Iterative calibration
# ---------------------------------------------------------------------------

def run_calibration(config_path: str | Path = "config/sim.yaml") -> None:
    """FSM iterative calibration: assign → compare → scale → repeat."""
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)

    # Assignment params
    algorithm = str(calib_cfg.get("algorithm", "bfw"))
    max_iter_assign = int(calib_cfg.get("max_iter", 100))
    rgap = float(calib_cfg.get("rgap_target", 0.001))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))
    buffer_m = float(calib_cfg.get("match_buffer_m", 50.0))

    # Iteration params
    max_iterations = int(calib_cfg.get("max_iterations", 10))
    conv_cfg = calib_cfg.get("convergence") or {}
    geh_target = float(conv_cfg.get("geh_lt5_target_pct", 85.0))
    min_improvement = float(conv_cfg.get("min_improvement_pct", 1.0))
    scale_cfg = calib_cfg.get("scaling") or {}
    scale_method = str(scale_cfg.get("method", "sector"))
    damping = float(scale_cfg.get("damping", 0.5))
    min_factor = float(scale_cfg.get("min_factor", 0.5))
    max_factor = float(scale_cfg.get("max_factor", 2.0))

    print("=== FSM ITERATIVE CALIBRATION ===")
    print(f"  max_iterations={max_iterations}, target GEH<5 >= {geh_target}%")
    print(f"  scaling: {scale_method}, damping={damping}")

    # Pre-flight
    fix_node_ids(project_dir)

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}")

    # Keep a backup of the original matrix
    backup = matrix_path.with_suffix(".aem.orig")
    if not backup.exists():
        shutil.copy2(matrix_path, backup)

    # Load calibration counts once
    pent = load_pentlogram(cfg)
    links_gdf = _load_network_links(project_dir)
    print(f"  Pentlogram: {len(pent)} observed segments")

    # Load matrix (stays open across iterations)
    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])

    project = Project()
    project.open(str(project_dir))

    history: List[Dict[str, Any]] = []
    prev_geh5 = 0.0

    try:
        for it in range(1, max_iterations + 1):
            print(f"\n── Iteration {it}/{max_iterations} ──")

            # 1) Assignment
            total_demand = float(mat.matrix_view.sum())
            print(f"  Demand total: {total_demand:,.0f}")
            vol_df = execute_assignment(
                project, mat,
                algorithm=algorithm,
                max_iter=max_iter_assign,
                rgap_target=rgap,
            )

            vol_col = _detect_volume_col(vol_df)
            total_vol = float(vol_df[vol_col].sum()) if vol_col else 0.0
            print(f"  Assigned volume: {total_vol:,.0f}  (col={vol_col})")

            # 2) Match to pentlogram
            links_with_vol = links_gdf.copy()
            if vol_col and "link_id" in vol_df.columns:
                links_with_vol = links_with_vol.merge(
                    vol_df[["link_id", vol_col]], on="link_id", how="left",
                )

            matched = match_counts_to_links(pent, links_with_vol, buffer_m=buffer_m)
            vc = vol_col if vol_col and vol_col in matched.columns else next(
                (c for c in matched.columns if vol_col and vol_col in c), None
            )

            # 3) Compute stats
            if vc and vc in matched.columns:
                valid = matched.dropna(subset=[vc, "observed_total"])
                valid = valid[valid["observed_total"] > 0].copy()
                stats = compute_stats(valid[vc].values, valid["observed_total"].values)
            else:
                valid = pd.DataFrame()
                stats = {"n": 0}

            geh5 = float(stats.get("geh_lt5_pct", 0))
            geh10 = float(stats.get("geh_lt10_pct", 0))
            r2 = stats.get("r2")
            print(f"  GEH<5: {geh5:.1f}%  GEH<10: {geh10:.1f}%  R²: {r2}")

            iter_record = {
                "iteration": it,
                "demand_total": round(total_demand, 0),
                "assigned_total": round(total_vol, 0),
                **stats,
            }
            history.append(iter_record)

            # 4) Convergence check
            if geh5 >= geh_target:
                print(f"  CONVERGED: GEH<5 = {geh5:.1f}% >= target {geh_target}%")
                break

            improvement = geh5 - prev_geh5
            if it > 1 and improvement < min_improvement:
                print(f"  STALLED: improvement {improvement:.2f}% < {min_improvement}%")
                break

            prev_geh5 = geh5

            # 5) Scale OD matrix
            if not valid.empty and vc and total_vol > 0:
                if scale_method == "global":
                    factor = _compute_global_factor(
                        valid, vc, damping=damping,
                        min_factor=min_factor, max_factor=max_factor,
                    )
                    print(f"  Global scaling factor: {factor:.3f}")
                    scale_matrix(mat, core_name, factor)
                else:
                    factors = _compute_sector_factors(
                        valid, vc, damping=damping,
                        min_factor=min_factor, max_factor=max_factor,
                    )
                    if factors:
                        # Weights = total observed per sector for weighted average
                        obs_weights = {}
                        for lt in factors:
                            sub = valid[valid["link_type"] == lt]
                            obs_weights[lt] = float(sub["observed_total"].sum())
                        total_w = sum(obs_weights.values())
                        wavg = sum(f * obs_weights.get(k, 0) for k, f in factors.items()) / max(total_w, 1e-9)
                        print(f"  Sector factors ({len(factors)}): weighted_avg={wavg:.3f}  {factors}")
                        scale_matrix(mat, core_name, factors, weights=obs_weights)
                    else:
                        print("  No sector factors computed — skipping scaling")
            else:
                print("  Cannot scale — no valid matched volumes")

            # Persist scaled matrix for next iteration
            mat.save()

    finally:
        mat.close()
        project.close()

    # Save calibration report
    report = {
        "iterations": len(history),
        "converged": history[-1].get("geh_lt5_pct", 0) >= geh_target if history else False,
        "history": history,
        "final": history[-1] if history else {},
        "config": {
            "max_iterations": max_iterations,
            "geh_target": geh_target,
            "scale_method": scale_method,
            "damping": damping,
        },
    }
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCalibration report: {report_path}")

    # Save final assignment
    if history:
        out_path = output_dir / "assignment_results.parquet"
        vol_df.to_parquet(str(out_path), index=False)
        print(f"Final assignment: {out_path}")


# ---------------------------------------------------------------------------
# Independent validation (CSD2020)
# ---------------------------------------------------------------------------

def run_validation_only(config_path: str | Path = "config/sim.yaml") -> None:
    """Compare the calibrated assignment to independent CSD2020 data."""
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)
    buffer_m = float(_get(cfg, ["calibration", "match_buffer_m"], 50.0))

    print("=== INDEPENDENT VALIDATION (CSD2020) ===")

    # Load last assignment results
    results_path = output_dir / "assignment_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No assignment results at {results_path}. Run 'calibrate' or 'assign' first."
        )
    vol_df = pd.read_parquet(str(results_path))
    vol_col = _detect_volume_col(vol_df)
    print(f"  Loaded assignment: {len(vol_df)} links, vol_col={vol_col}")

    links_gdf = _load_network_links(project_dir)
    if vol_col and "link_id" in vol_df.columns:
        links_gdf = links_gdf.merge(vol_df[["link_id", vol_col]], on="link_id", how="left")

    report: Dict[str, Any] = {}

    # Pentlogram (same data as calibration — just for reference)
    print("\n1) Pentlogram comparison (reference) ...")
    try:
        pent = load_pentlogram(cfg)
        matched = match_counts_to_links(pent, links_gdf, buffer_m=buffer_m)
        vc = vol_col if vol_col and vol_col in matched.columns else None
        if vc:
            valid = matched.dropna(subset=[vc, "observed_total"])
            valid = valid[valid["observed_total"] > 0]
            stats = compute_stats(valid[vc].values, valid["observed_total"].values)
            report["pentlogram"] = {"matched": int(len(valid)), **stats}
            print(f"  Matched: {len(valid)}  GEH<5: {stats.get('geh_lt5_pct')}%  "
                  f"R²: {stats.get('r2')}")
        else:
            print("  No volume column on links")
    except Exception as e:
        print(f"  SKIP: {e}")

    # CSD2020 — independent validation
    print("\n2) CSD2020 aggregate validation (independent) ...")
    try:
        csd = load_csd2020(cfg)
        csd_agg = aggregate_csd_by_class(csd)
        report["csd2020_observed"] = csd_agg.to_dict(orient="records")

        if vol_col and vol_col in links_gdf.columns:
            model_agg = aggregate_model_by_class(links_gdf, vol_col)
            report["csd2020_modeled"] = model_agg.to_dict(orient="records")

        print(f"  CSD2020 JMK: {len(csd)} sections")
        for _, r in csd_agg.iterrows():
            print(f"    {r['road_class']:12s}  sections={int(r['sections']):4d}  "
                  f"mean_AADT={r['mean_sv']:>8.0f}  mean_cars={r['mean_o']:>8.0f}")
    except Exception as e:
        print(f"  SKIP: {e}")

    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nValidation report: {report_path}")
