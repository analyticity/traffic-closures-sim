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
import math
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    # Robust CRS handling: detect GeoJSON files with metric coordinates
    # mislabeled as EPSG:4326 (common when ArcGIS outSR != 4326 is used
    # with f=geojson format, violating RFC 7946).
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=out_epsg)
    else:
        sample = gdf.geometry.dropna()
        if not sample.empty:
            try:
                bounds = sample.total_bounds
                max_abs = max(abs(bounds[0]), abs(bounds[1]), abs(bounds[2]), abs(bounds[3]))
            except Exception:
                max_abs = 0

            declared_epsg = gdf.crs.to_epsg()
            if declared_epsg == 4326 and max_abs > 1000:
                print(f"  WARNING: GeoJSON declared as EPSG:4326 but coordinates "
                      f"are metric (max={max_abs:.0f}). Overriding to EPSG:{out_epsg}.")
                gdf = gdf.set_crs(epsg=out_epsg, allow_override=True)
            elif declared_epsg != out_epsg:
                gdf = gdf.to_crs(epsg=out_epsg)

    # Fail-fast geometry validation
    valid_geom = gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~valid_geom).sum())
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid}/{len(gdf)} pentlogram features have invalid geometry")
        gdf = gdf[valid_geom].copy()

    for col in ("car_24", "truc_24"):
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    # Pentlogram data from data.Brno: values are "v tisicich" (in thousands)
    # per 24h. Multiplier is configurable in case a different layer uses
    # different units (hundreds, absolute, etc.).
    units_cfg = _get(cfg, [
        "datasets", "sources", "calibration_brno_pentlogram_2024", "units",
    ], {}) or {}
    car_mult = float(units_cfg.get("car_24_multiplier", 1000))
    truc_mult = float(units_cfg.get("truc_24_multiplier", 1000))

    gdf["observed_car"] = gdf.get("car_24", 0) * car_mult
    gdf["observed_truck"] = gdf.get("truc_24", 0) * truc_mult
    gdf["observed_total"] = gdf["observed_car"] + gdf["observed_truck"]

    # Sanity check: warn if observed values are outside plausible range
    nz = gdf[gdf["observed_car"] > 0]["observed_car"]
    if not nz.empty:
        p50 = float(nz.median())
        p_max = float(nz.max())
        if p_max < 500:
            print(f"  WARNING: observed_car max={p_max:.0f} seems very low. "
                  f"Check car_24_multiplier (currently {car_mult}).")
        elif p50 > 100000:
            print(f"  WARNING: observed_car median={p50:.0f} seems very high. "
                  f"Check car_24_multiplier (currently {car_mult}).")
        else:
            print(f"  Pentlogram observed_car: median={p50:,.0f} max={p_max:,.0f} "
                  f"(multiplier={car_mult})")

    return gdf[gdf["observed_total"] > 0].copy()


def validate_geometries_or_fail(
    gdf: gpd.GeoDataFrame,
    *,
    name: str = "dataset",
    expected_epsg: Optional[int] = None,
    min_valid_pct: float = 0.99,
) -> None:
    """Fail-fast validation of geometry integrity after loading + CRS ops.

    Raises ValueError with a remediation message if data is corrupt.
    """
    if gdf is None or len(gdf) == 0:
        raise ValueError(f"{name}: empty dataset (0 features).")

    valid_basic = gdf.geometry.notna() & ~gdf.geometry.is_empty
    pct_valid = float(valid_basic.mean())
    if pct_valid < min_valid_pct:
        n_bad = int((~valid_basic).sum())
        raise ValueError(
            f"{name}: {n_bad}/{len(gdf)} features have null/empty geometry "
            f"({pct_valid:.1%} valid, need {min_valid_pct:.0%}). "
            "Remediation: delete cached GeoJSON and re-run fetch-data."
        )

    centroids = gdf[valid_basic].geometry.centroid
    cx = centroids.x.values.astype(float)
    cy = centroids.y.values.astype(float)
    finite_mask = np.isfinite(cx) & np.isfinite(cy)
    pct_finite = float(finite_mask.mean()) if len(finite_mask) > 0 else 0.0
    if pct_finite < min_valid_pct:
        n_bad = int((~finite_mask).sum())
        raise ValueError(
            f"{name}: {n_bad} centroid(s) have NaN/inf coordinates. "
            "This typically means CRS mismatch: GeoJSON with metric coordinates "
            "interpreted as WGS84 degrees. "
            "Remediation: delete the GeoJSON file and re-fetch with outSR=4326."
        )

    bounds = gdf[valid_basic].total_bounds
    max_abs = float(np.nanmax(np.abs(bounds)))
    epsg = gdf.crs.to_epsg() if gdf.crs else None

    if epsg == 4326 and max_abs > 180:
        raise ValueError(
            f"{name}: CRS=EPSG:4326 but bounds {tuple(round(float(x), 2) for x in bounds)} "
            f"exceed degree range. Coordinates are likely metric, not WGS84. "
            "Remediation: re-fetch GeoJSON with outSR=4326 or fix CRS override."
        )
    if expected_epsg == 5514 and epsg == 5514 and max_abs < 1000:
        raise ValueError(
            f"{name}: CRS=EPSG:5514 but bounds look like degrees (max={max_abs:.1f}). "
            "Likely the file was not reprojected. "
            "Remediation: check that to_crs(5514) was applied after loading."
        )


def load_csd2020(cfg: Dict[str, Any], region_code: str = "CZ064") -> pd.DataFrame:
    cache_dir = Path(_get(cfg, ["datasets", "cache_dir"], "data/cache"))
    parquet_path = cache_dir / "v2_csd2025.parquet"
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


def _bearing_from_geom(geom) -> Optional[float]:
    """Bearing (0-360) from first to last vertex of a LineString."""
    try:
        coords = list(geom.coords)
        if len(coords) < 2:
            return None
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        return math.degrees(math.atan2(dx, dy)) % 360
    except Exception:
        return None


def _bearing_diff(a: float, b: float) -> float:
    """Absolute angular difference (0-180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _aggregate_corridor_volumes(
    joined: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    buffer_m: float,
    bearing_tol: float = 30.0,
) -> gpd.GeoDataFrame:
    """Sum volumes from parallel corridor links for bidirectional count stations.

    Restrictions to prevent accidental aggregation of ramps/frontage roads:
    - Only links with the same ``link_type`` as the matched link are included
    - Spatial index is used for efficient candidate search
    - Bearing tolerance filters out perpendicular links
    """
    vol_cols = [c for c in joined.columns if c.endswith("_tot")]
    if not vol_cols or "link_id" not in joined.columns:
        return joined

    vc = vol_cols[0]
    joined["_corridor_volume"] = joined[vc].copy()
    joined["_corridor_n_links"] = 1

    link_lid = links["link_id"].values if "link_id" in links.columns else np.array([])
    link_lt = links["link_type"].astype(str).values if "link_type" in links.columns else np.array([])
    link_vol = links[vc].values if vc in links.columns else np.zeros(len(links))
    link_bearing = np.array([_bearing_from_geom(g) for g in links.geometry], dtype=object)

    has_sindex = hasattr(links, "sindex")

    for idx, row in joined.iterrows():
        if pd.isna(row.get("link_id")):
            continue
        matched_lid = int(row["link_id"])
        count_pt = row.geometry
        matched_lt = str(row.get("link_type", ""))

        matched_idx_in_links = np.where(link_lid == matched_lid)[0]
        if len(matched_idx_in_links) == 0:
            continue
        mb = link_bearing[matched_idx_in_links[0]]
        if mb is None:
            continue

        if has_sindex:
            buf_geom = count_pt.buffer(buffer_m)
            cand_idxs = list(links.sindex.query(buf_geom, predicate="intersects"))
        else:
            cand_idxs = range(len(links))

        corridor_vol = float(row.get(vc, 0) or 0)
        n_links = 1

        for ci in cand_idxs:
            lid = int(link_lid[ci])
            if lid == matched_lid:
                continue
            lt = str(link_lt[ci])
            if lt != matched_lt:
                continue
            lb = link_bearing[ci]
            if lb is None:
                continue
            if _bearing_diff(mb, lb) > bearing_tol:
                continue
            if not has_sindex:
                dist = count_pt.distance(links.geometry.iloc[ci])
                if dist > buffer_m:
                    continue

            corridor_vol += float(link_vol[ci])
            n_links += 1

        joined.at[idx, "_corridor_volume"] = corridor_vol
        joined.at[idx, "_corridor_n_links"] = n_links

    return joined


_NON_CAR_LINK_TYPES = frozenset({
    "footway", "path", "track", "steps", "cycleway", "pedestrian",
    "corridor", "bridleway", "proposed", "construction", "elevator",
    "service", "rest_area", "services", "traffic_mirror", "virtual",
    "crossing", "busway",
})


def match_counts_to_links(
    counts: gpd.GeoDataFrame,
    links: gpd.GeoDataFrame,
    *,
    buffer_m: float = 50.0,
    metric_epsg: int = 5514,
    id_col: str = "objectid",
    direction_aware: bool = True,
    conflict_resolution: str = "nearest",
    aggregate_corridor: bool = True,
) -> gpd.GeoDataFrame:
    """Spatial-join observed count points/lines to nearest network links.

    Features:
    - Filters to car-driveable links only (excludes footway, track, etc.)
    - Direction-aware scoring (bearing similarity between count and link)
    - Conflict detection (multiple counts on one link)
    - Corridor aggregation: sums volumes from parallel links (divided highways)
    - Match quality columns: _dist, _bearing_diff, _match_quality
    """
    counts = counts.to_crs(epsg=metric_epsg)
    links = links.to_crs(epsg=metric_epsg)

    n_before = len(links)
    if "link_type" in links.columns:
        links = links[~links["link_type"].astype(str).isin(_NON_CAR_LINK_TYPES)].copy()
    if "modes" in links.columns:
        links = links[links["modes"].astype(str).str.contains("c", na=False)].copy()
    n_after = len(links)
    if n_before != n_after:
        print(f"  Matching: filtered {n_before} -> {n_after} car-driveable links")

    pts = counts.copy()
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty].copy()
    count_geom_backup = pts.geometry.copy()
    pts["_count_bearing"] = count_geom_backup.apply(_bearing_from_geom)
    centroids = pts.geometry.centroid
    valid_cent = np.isfinite(centroids.x) & np.isfinite(centroids.y)
    n_dropped = int((~valid_cent).sum())
    if n_dropped > 0:
        print(f"  WARNING: dropped {n_dropped} counts with invalid centroid geometry")
    pts = pts[valid_cent].copy()
    pts["geometry"] = centroids[valid_cent]

    keep = ["link_id", "link_type", "name", "geometry"] + [
        c for c in links.columns
        if c not in ("link_id", "link_type", "name", "geometry", "ogc_fid")
        and links[c].dtype in ("float64", "float32", "int64")
    ]
    keep = [c for c in keep if c in links.columns]

    links_sel = links[keep].copy()
    links_sel["_link_bearing"] = links_sel.geometry.apply(_bearing_from_geom)

    joined = gpd.sjoin_nearest(
        pts, links_sel, how="left", max_distance=buffer_m, distance_col="_dist",
    )

    if direction_aware:
        has_both = joined["_count_bearing"].notna() & joined["_link_bearing"].notna()
        joined["_bearing_diff"] = np.nan
        if has_both.any():
            joined.loc[has_both, "_bearing_diff"] = joined.loc[has_both].apply(
                lambda r: _bearing_diff(r["_count_bearing"], r["_link_bearing"]), axis=1,
            )
        bearing_penalty = joined["_bearing_diff"].fillna(0) / 180.0
        dist_norm = joined["_dist"].fillna(buffer_m) / max(buffer_m, 1.0)
        joined["_match_quality"] = 1.0 - 0.6 * dist_norm - 0.4 * bearing_penalty
    else:
        joined["_bearing_diff"] = np.nan
        joined["_match_quality"] = 1.0 - joined["_dist"].fillna(buffer_m) / max(buffer_m, 1.0)

    # De-duplicate counts: keep best match per count station
    if id_col in joined.columns:
        joined = joined.sort_values("_match_quality", ascending=False)
        joined = joined.drop_duplicates(subset=[id_col], keep="first")

    # Conflict detection: multiple counts matched to the same link
    if "link_id" in joined.columns:
        link_counts = joined.groupby("link_id").size()
        conflicts = link_counts[link_counts > 1]
        joined["_link_conflict"] = joined["link_id"].isin(conflicts.index)
        n_conflicts = int(conflicts.sum() - len(conflicts))

        if n_conflicts > 0 and conflict_resolution == "nearest":
            joined = joined.sort_values("_dist")
            joined = joined.drop_duplicates(subset=["link_id"], keep="first")
    else:
        joined["_link_conflict"] = False

    joined["_matched"] = joined["link_id"].notna() if "link_id" in joined.columns else False

    # Corridor aggregation: sum volumes from parallel links (divided highways)
    if aggregate_corridor:
        joined = _aggregate_corridor_volumes(joined, links_sel, buffer_m)

    for col in ("_count_bearing", "_link_bearing"):
        if col in joined.columns:
            joined = joined.drop(columns=[col])

    return joined


def match_quality_report(matched: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Summary statistics about count-to-link matching quality."""
    m = matched[matched["_matched"]] if "_matched" in matched.columns else matched
    n_total = len(matched)
    n_matched = int(m["_matched"].sum()) if "_matched" in m.columns else len(m)
    n_unmatched = n_total - n_matched
    n_conflicts = int(m["_link_conflict"].sum()) if "_link_conflict" in m.columns else 0
    dists = m["_dist"].dropna()
    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "n_link_conflicts": n_conflicts,
        "mean_match_distance_m": round(float(dists.mean()), 1) if len(dists) else None,
        "max_match_distance_m": round(float(dists.max()), 1) if len(dists) else None,
        "median_match_distance_m": round(float(dists.median()), 1) if len(dists) else None,
        "mean_bearing_diff_deg": round(float(m["_bearing_diff"].dropna().mean()), 1)
            if "_bearing_diff" in m.columns and m["_bearing_diff"].notna().any() else None,
    }


def _export_matching_diagnostics(
    matched: gpd.GeoDataFrame,
    model_col: Optional[str],
    obs_col: str,
    output_dir: Path,
) -> None:
    """Export per-count matching diagnostics to CSV for external audit."""
    diag_cols = ["objectid", "link_id", "link_type", "name"]
    for c in [obs_col, "observed_car", "observed_truck", "observed_total",
              "_dist", "_bearing_diff", "_match_quality",
              "_corridor_volume", "_corridor_n_links", "_matched"]:
        if c not in diag_cols:
            diag_cols.append(c)
    if model_col and model_col not in diag_cols:
        diag_cols.append(model_col)

    available = [c for c in diag_cols if c in matched.columns]
    diag = matched[available].copy()

    m_arr = diag[model_col].values.astype(float) if model_col and model_col in diag.columns else np.zeros(len(diag))
    o_arr = diag[obs_col].values.astype(float) if obs_col in diag.columns else np.zeros(len(diag))
    denom = m_arr + o_arr
    mask = denom > 0
    geh = np.full(len(diag), np.nan)
    geh[mask] = np.sqrt(2.0 * (m_arr[mask] - o_arr[mask]) ** 2 / denom[mask])
    diag["GEH"] = geh

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "matching_diagnostics.csv"
    diag.to_csv(out_path, index=False)
    print(f"  Matching diagnostics: {out_path} ({len(diag)} rows)")


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
    obs_col: str = "observed_total",
) -> float:
    """Single factor = sum(observed) / sum(modeled), with damping."""
    m = matched[vol_col].values.astype(float)
    c = matched[obs_col].values.astype(float)
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
    obs_col: str = "observed_total",
) -> Dict[str, float]:
    """Per-road-class factor = weighted-mean(observed / modeled) within class."""
    factors: Dict[str, float] = {}
    for lt in matched["link_type"].dropna().unique():
        sub = matched[matched["link_type"] == lt]
        m = sub[vol_col].values.astype(float)
        c = sub[obs_col].values.astype(float)
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
# Sector-based OD scaling (geographic zones → sector pairs)
# ---------------------------------------------------------------------------

def _assign_zone_sectors(
    project_dir: Path,
    n_sectors: int = 5,
    sector_labels: Optional[List[str]] = None,
) -> Dict[int, str]:
    """Assign each zone to a geographic sector based on azimuth from network centroid.

    Sectors are angular slices centred on the network's geographic centre.
    A special 'C' sector captures zones within the inner 20% radius.
    Returns ``{zone_id: sector_label}``.
    """
    zones_path = Path("outputs/baseline/zones/centroids.geojson")
    if not zones_path.exists():
        zones_path = Path("outputs/baseline/zones/zones.geojson")
    if not zones_path.exists():
        return {}

    gdf = gpd.read_file(zones_path)
    if gdf.empty or "zone_id" not in gdf.columns:
        return {}

    if gdf.crs is not None and gdf.crs.to_epsg() != 5514:
        gdf = gdf.to_crs(epsg=5514)

    pts = gdf.geometry.representative_point() if gdf.geom_type.iloc[0] in ("Polygon", "MultiPolygon") else gdf.geometry
    cx = float(pts.x.mean())
    cy = float(pts.y.mean())

    if sector_labels is None:
        if n_sectors == 4:
            sector_labels = ["N", "E", "S", "W"]
        elif n_sectors == 5:
            sector_labels = ["N", "E", "S", "W", "C"]
        else:
            sector_labels = [f"S{i}" for i in range(n_sectors)]

    has_center = "C" in sector_labels
    n_angular = n_sectors - 1 if has_center else n_sectors
    sector_size = 360.0 / n_angular if n_angular > 0 else 360.0

    dists = np.sqrt((pts.x - cx) ** 2 + (pts.y - cy) ** 2)
    r_threshold = float(np.percentile(dists, 20)) if has_center else 0.0

    angular_labels = [s for s in sector_labels if s != "C"]
    result: Dict[int, str] = {}
    for i, (_, row) in enumerate(gdf.iterrows()):
        zid = int(row["zone_id"])
        pt = pts.iloc[i]
        d = float(dists.iloc[i])
        if has_center and d <= r_threshold:
            result[zid] = "C"
        else:
            angle = math.degrees(math.atan2(pt.x - cx, pt.y - cy)) % 360
            sec_idx = int(angle // sector_size) % n_angular
            result[zid] = angular_labels[sec_idx]
    return result


def _compute_sector_pair_factors(
    matched: pd.DataFrame,
    vol_col: str,
    links_gdf: gpd.GeoDataFrame,
    zone_sector_map: Dict[int, str],
    *,
    damping: float,
    min_factor: float,
    max_factor: float,
    obs_col: str = "observed_total",
) -> Dict[Tuple[str, str], float]:
    """Compute correction factors per geographic sector pair.

    Each matched link is assigned to a sector based on its midpoint location.
    Links in the same sector contribute to that sector's factor.
    The factor is then applied to OD pairs whose O and D fall in sectors
    estimated from the link's geographic position.
    """
    valid = matched.dropna(subset=[vol_col, obs_col])
    valid = valid[(valid[vol_col] > 0) & (valid[obs_col] > 0)].copy()
    if valid.empty:
        return {}

    all_sectors = sorted(set(zone_sector_map.values()))
    if not all_sectors:
        return {}

    if links_gdf.crs is not None and links_gdf.crs.to_epsg() != 5514:
        links_metric = links_gdf.to_crs(epsg=5514)
    else:
        links_metric = links_gdf
    if valid.crs is not None and valid.crs.to_epsg() != 5514:
        valid = valid.to_crs(epsg=5514)

    # Compute sector for each link midpoint
    inv_map = {}
    for zid, sec in zone_sector_map.items():
        inv_map.setdefault(sec, []).append(zid)

    cx = float(links_metric.geometry.centroid.x.mean())
    cy = float(links_metric.geometry.centroid.y.mean())

    n_angular = len([s for s in all_sectors if s != "C"])
    has_center = "C" in all_sectors
    angular_labels = [s for s in all_sectors if s != "C"]
    sector_size = 360.0 / n_angular if n_angular > 0 else 360.0

    dists_all = np.sqrt((links_metric.geometry.centroid.x - cx) ** 2
                        + (links_metric.geometry.centroid.y - cy) ** 2)
    r_threshold = float(np.percentile(dists_all, 20)) if has_center else 0.0

    def _link_sector(geom) -> str:
        mp = geom.centroid
        d = math.sqrt((mp.x - cx) ** 2 + (mp.y - cy) ** 2)
        if has_center and d <= r_threshold:
            return "C"
        angle = math.degrees(math.atan2(mp.x - cx, mp.y - cy)) % 360
        idx = int(angle // sector_size) % n_angular
        return angular_labels[idx]

    valid["_link_sector"] = valid.geometry.apply(_link_sector)

    # Per-sector factor
    sector_factor: Dict[str, float] = {}
    for sec in all_sectors:
        sub = valid[valid["_link_sector"] == sec]
        if len(sub) < 2:
            continue
        m = sub[vol_col].values.astype(float)
        c = sub[obs_col].values.astype(float)
        raw = float(c.sum() / max(m.sum(), 1e-9))
        sector_factor[sec] = float(np.clip(1.0 + damping * (raw - 1.0), min_factor, max_factor))

    # Build sector-pair factors: for (O_sector, D_sector) use geometric mean
    # of the two sectors' factors (heuristic until path-based proportions)
    result: Dict[Tuple[str, str], float] = {}
    for os in all_sectors:
        fo = sector_factor.get(os, 1.0)
        for ds in all_sectors:
            fd = sector_factor.get(ds, 1.0)
            result[(os, ds)] = float(np.sqrt(fo * fd))
    return result


def scale_matrix_by_sectors(
    mat: AequilibraeMatrix,
    core_name: str,
    zone_sector_map: Dict[int, str],
    sector_factors: Dict[Tuple[str, str], float],
) -> None:
    """Apply differentiated scaling by origin-destination sector pairs."""
    idx = list(mat.index)
    data = mat.matrix[core_name]
    for i, oz in enumerate(idx):
        os = zone_sector_map.get(int(oz), "C")
        for ds_label in set(zone_sector_map.values()):
            f = sector_factors.get((os, ds_label), 1.0)
            if abs(f - 1.0) < 1e-6:
                continue
            for j, dz in enumerate(idx):
                if zone_sector_map.get(int(dz), "C") == ds_label:
                    data[i, j] *= f


def _do_sector_od_scaling(
    mat: AequilibraeMatrix,
    core_name: str,
    valid: pd.DataFrame,
    vol_col: str,
    *,
    links_gdf: gpd.GeoDataFrame,
    project_dir: Path,
    damping: float,
    min_factor: float,
    max_factor: float,
    obs_col: str,
    sector_cfg: Optional[Dict] = None,
) -> None:
    """Orchestrate sector-based OD matrix scaling."""
    n_sectors = 5
    sector_labels = None
    if sector_cfg:
        n_sectors = int(sector_cfg.get("n_sectors", 5))
        sector_labels = sector_cfg.get("labels")

    zone_sector_map = _assign_zone_sectors(project_dir, n_sectors=n_sectors,
                                            sector_labels=sector_labels)
    if not zone_sector_map:
        print("  WARNING: could not assign zone sectors — falling back to global")
        factor = _compute_global_factor(valid, vol_col, damping=damping,
                                         min_factor=min_factor, max_factor=max_factor,
                                         obs_col=obs_col)
        scale_matrix(mat, core_name, factor)
        return

    sector_factors = _compute_sector_pair_factors(
        valid, vol_col, links_gdf, zone_sector_map,
        damping=damping, min_factor=min_factor, max_factor=max_factor,
        obs_col=obs_col,
    )

    if not sector_factors:
        print("  WARNING: no sector factors computed — falling back to global")
        factor = _compute_global_factor(valid, vol_col, damping=damping,
                                         min_factor=min_factor, max_factor=max_factor,
                                         obs_col=obs_col)
        scale_matrix(mat, core_name, factor)
        return

    unique_factors = set(round(f, 4) for f in sector_factors.values())
    print(f"  Sector-OD factors: {len(sector_factors)} pairs, "
          f"range [{min(sector_factors.values()):.3f}, {max(sector_factors.values()):.3f}], "
          f"{len(unique_factors)} distinct values")

    scale_matrix_by_sectors(mat, core_name, zone_sector_map, sector_factors)


# ---------------------------------------------------------------------------
# Select-link based OD correction (Spiess-like proportional adjustment)
# ---------------------------------------------------------------------------

def _apply_select_link_od_correction(
    mat: AequilibraeMatrix,
    core_name: str,
    sl_matrices: Dict[str, np.ndarray],
    sl_results: Dict[str, Any],
    damping: float = 0.3,
    min_factor: float = 0.5,
    max_factor: float = 2.0,
) -> Dict[str, float]:
    """Adjust OD cells proportionally based on select-link screenline errors.

    For each screenline with significant error, scales OD cells that
    contribute to that screenline, weighted by their select-link proportion.
    """
    if not sl_matrices or not sl_results:
        return {}

    data = mat.matrix[core_name]
    total_od = data[:, :].copy().astype(np.float64)
    corrections: Dict[str, float] = {}

    for sl_name, sl_od in sl_matrices.items():
        sr = sl_results.get(sl_name, {})
        obs = sr.get("observed_total", 0)
        mod = sr.get("modeled_total", 0)
        if obs <= 0 or mod <= 0:
            continue

        geh = sr.get("geh", 0)
        if geh is not None and geh < 3.0:
            continue

        raw_factor = obs / mod
        f = 1.0 + damping * (raw_factor - 1.0)
        f = float(np.clip(f, min_factor, max_factor))
        corrections[sl_name] = round(f, 4)

        proportion = np.zeros_like(total_od)
        mask = total_od > 0
        proportion[mask] = sl_od[mask] / total_od[mask]
        proportion = np.clip(proportion, 0, 1)

        adjustment = 1.0 + (f - 1.0) * proportion
        adjustment = np.clip(adjustment, min_factor, max_factor)
        data[:, :] *= adjustment

    return corrections


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
# Journey time validation
# ---------------------------------------------------------------------------

_CAR_LINK_TYPES_FOR_SPEED = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "unclassified",
    "living_street",
})

_CZECH_REFERENCE_SPEEDS = {
    "motorway": 130.0,
    "motorway_link": 80.0,
    "trunk": 90.0,
    "trunk_link": 60.0,
    "primary": 70.0,
    "primary_link": 50.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "unclassified": 40.0,
    "living_street": 20.0,
}


def compute_class_speed_comparison(
    links_gdf: gpd.GeoDataFrame,
    csd_data: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compare modeled average speeds per car-driveable road class against
    Czech reference design speeds.
    """
    result: Dict[str, Dict[str, float]] = {}
    for lt in sorted(_CAR_LINK_TYPES_FOR_SPEED):
        sub = links_gdf[links_gdf["link_type"] == lt] if "link_type" in links_gdf.columns else pd.DataFrame()
        if sub.empty:
            continue

        for spd_col in ("speed_ab", "speed"):
            if spd_col in sub.columns:
                speeds = pd.to_numeric(sub[spd_col], errors="coerce").dropna()
                if not speeds.empty:
                    model_speed = float(speeds.mean())
                    ref_speed = _CZECH_REFERENCE_SPEEDS.get(lt, 50.0)
                    pct_diff = (model_speed - ref_speed) / max(ref_speed, 1) * 100
                    result[lt] = {
                        "modeled_kmh": round(model_speed, 1),
                        "reference_kmh": round(ref_speed, 1),
                        "pct_diff": round(pct_diff, 1),
                    }
                    break

    return result


def validate_journey_times(
    skim_matrix: np.ndarray,
    zone_ids: np.ndarray,
    routes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate journey times from skim matrix against reference routes.

    Each route dict has: name, from_zone, to_zone, reference_time_min, tolerance_pct.
    """
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    results: List[Dict[str, Any]] = []

    for route in routes:
        name = route.get("name", "unnamed")
        fz = int(route.get("from_zone", 0))
        tz = int(route.get("to_zone", 0))
        ref_min = float(route.get("reference_time_min", 0))
        tol_pct = float(route.get("tolerance_pct", 15))

        if fz not in z2i or tz not in z2i or ref_min <= 0:
            results.append({"name": name, "status": "skipped", "reason": "invalid zone or time"})
            continue

        fi, ti = z2i[fz], z2i[tz]
        model_sec = float(skim_matrix[fi, ti])
        model_min = model_sec / 60.0 if model_sec > 0 else float("nan")

        if not np.isfinite(model_min) or model_min <= 0:
            results.append({"name": name, "status": "no_path", "model_min": None, "ref_min": ref_min})
            continue

        diff_pct = (model_min - ref_min) / ref_min * 100
        diff_min = model_min - ref_min
        within_pct = abs(diff_pct) <= tol_pct
        within_1min = abs(diff_min) <= 1.0
        passed = within_pct or within_1min

        results.append({
            "name": name,
            "from_zone": fz,
            "to_zone": tz,
            "model_min": round(model_min, 1),
            "ref_min": ref_min,
            "diff_pct": round(diff_pct, 1),
            "diff_min": round(diff_min, 1),
            "pass": passed,
        })

    return results


# ---------------------------------------------------------------------------
# Supply parameter tuning (outer loop)
# ---------------------------------------------------------------------------

@dataclass
class SupplyParams:
    speed_factors: Dict[str, float] = field(default_factory=lambda: {})
    capacity_factors: Dict[str, float] = field(default_factory=lambda: {})
    connector_penalty_s: float = 120.0


def _save_base_values(project_dir: Path) -> None:
    """Copy current speed/capacity to _base columns in SQLite (runs once)."""
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(links)").fetchall()]
        if "_base_speed_ab" in cols:
            conn.close()
            return

        for base, src in [
            ("_base_speed_ab", "speed_ab"), ("_base_speed_ba", "speed_ba"),
            ("_base_capacity_ab", "capacity_ab"), ("_base_capacity_ba", "capacity_ba"),
        ]:
            conn.execute(f"ALTER TABLE links ADD COLUMN {base} REAL")
            conn.execute(f"UPDATE links SET {base} = {src}")
        conn.commit()
    finally:
        conn.close()


def apply_supply_params(project_dir: Path, params: SupplyParams) -> int:
    """Apply class-specific speed/capacity factors to the network DB.

    Reads from _base_* columns (preserved originals) and writes
    factored values to speed_ab/speed_ba/capacity_ab/capacity_ba.
    """
    _save_base_values(project_dir)
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    updated = 0
    try:
        links = conn.execute(
            "SELECT link_id, link_type, _base_speed_ab, _base_speed_ba, "
            "_base_capacity_ab, _base_capacity_ba FROM links"
        ).fetchall()

        for lid, lt, bsab, bsba, bcab, bcba in links:
            lt_str = str(lt or "").lower()
            sf = params.speed_factors.get(lt_str, 1.0)
            cf = params.capacity_factors.get(lt_str, 1.0)

            for road_class, factor_s, factor_c in [
                ("motorway", sf, cf), ("trunk", sf, cf), ("primary", sf, cf),
                ("secondary", sf, cf), ("tertiary", sf, cf), ("residential", sf, cf),
            ]:
                if road_class in lt_str:
                    sf = params.speed_factors.get(road_class, sf)
                    cf = params.capacity_factors.get(road_class, cf)
                    break

            new_sab = (bsab or 50) * sf
            new_sba = (bsba or 50) * sf
            new_cab = (bcab or 900) * cf
            new_cba = (bcba or 900) * cf

            tt_ab = 0
            tt_ba = 0
            dist = conn.execute(
                "SELECT distance FROM links WHERE link_id=?", (lid,)
            ).fetchone()
            if dist and dist[0] and new_sab > 0:
                tt_ab = dist[0] * 3.6 / new_sab
            if dist and dist[0] and new_sba > 0:
                tt_ba = dist[0] * 3.6 / new_sba

            conn.execute(
                "UPDATE links SET speed_ab=?, speed_ba=?, capacity_ab=?, capacity_ba=?, "
                "travel_time_ab=?, travel_time_ba=? WHERE link_id=?",
                (new_sab, new_sba, new_cab, new_cba, tt_ab, tt_ba, lid),
            )
            updated += 1

        conn.commit()
    finally:
        conn.close()
    return updated


def compute_objective(
    count_stats: Dict[str, Any],
    screenline_results: Dict[str, Any],
    jt_results: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> float:
    """Weighted composite objective for supply calibration (lower is better)."""
    w = weights
    geh_term = (100.0 - float(count_stats.get("geh_lt5_pct", 0))) * w.get("geh", 1.0)

    sl_term = 0.0
    for sr in screenline_results.values():
        ratio = sr.get("ratio", 1.0)
        if ratio is not None:
            sl_term += abs(ratio - 1.0)
    sl_term *= w.get("screenline", 2.0)

    jt_fail = sum(1 for r in jt_results if not r.get("pass", True))
    jt_term = jt_fail / max(len(jt_results), 1) * 100 * w.get("jt", 1.0) if jt_results else 0.0

    return geh_term + sl_term + jt_term


def run_supply_tuning(config_path: str | Path = "config/sim.yaml") -> None:
    """Outer-loop supply parameter tuning via coordinate descent."""
    cfg = load_config(config_path)
    calib_cfg = cfg.get("calibration") or {}
    tuning_cfg = calib_cfg.get("supply_tuning") or {}

    if not tuning_cfg.get("enabled", False):
        print("Supply tuning disabled in config.")
        return

    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)

    road_classes = tuning_cfg.get("road_classes", ["motorway", "trunk", "primary", "secondary", "tertiary"])
    speed_range = tuning_cfg.get("speed_factor_range", [0.8, 0.9, 1.0, 1.1, 1.2])
    cap_range = tuning_cfg.get("capacity_factor_range", [0.8, 0.9, 1.0, 1.1, 1.2])
    inner_max_iter = int(tuning_cfg.get("inner_max_iterations", 3))
    obj_weights = tuning_cfg.get("objective_weights", {"geh": 1.0, "screenline": 2.0, "jt": 1.0})

    print("=== SUPPLY PARAMETER TUNING ===")
    print(f"  Road classes: {road_classes}")
    print(f"  Speed range: {speed_range}")
    print(f"  Capacity range: {cap_range}")

    _save_base_values(project_dir)

    best_params = SupplyParams(
        speed_factors={rc: 1.0 for rc in road_classes},
        capacity_factors={rc: 1.0 for rc in road_classes},
    )
    best_obj = float("inf")

    orig_max_iter = calib_cfg.get("max_iterations", 10)

    def _evaluate(params: SupplyParams) -> float:
        """Apply params, run short calibration, return objective."""
        apply_supply_params(project_dir, params)
        calib_cfg["max_iterations"] = inner_max_iter

        try:
            run_calibration(config_path)
        except Exception as e:
            print(f"    Calibration failed: {e}")
            return float("inf")

        report_path = output_dir / "calibration_report.json"
        if not report_path.exists():
            return float("inf")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        count_stats = report.get("final", {})
        sl_results = report.get("screenlines", {})

        jt_cfg = calib_cfg.get("journey_time_validation", {})
        jt_results = []
        if jt_cfg.get("reference_routes"):
            skim_path = output_dir / "skims.aem"
            if skim_path.exists():
                try:
                    mat = AequilibraeMatrix()
                    mat.load(str(skim_path))
                    names = list(mat.names)
                    if names:
                        zone_ids = mat.index[:].copy()
                        skim_data = mat.matrix[names[0]][:, :].copy()
                        mat.close()
                        jt_results = validate_journey_times(
                            skim_data, zone_ids, jt_cfg["reference_routes"],
                        )
                except Exception:
                    pass

        return compute_objective(count_stats, sl_results, jt_results, obj_weights)

    best_obj = _evaluate(best_params)
    print(f"\n  Baseline objective: {best_obj:.2f}")

    for rc in road_classes:
        print(f"\n  --- Tuning {rc} ---")

        # Speed factor
        best_sf = best_params.speed_factors.get(rc, 1.0)
        for sf in speed_range:
            if sf == best_sf:
                continue
            trial = SupplyParams(
                speed_factors={**best_params.speed_factors, rc: sf},
                capacity_factors=dict(best_params.capacity_factors),
            )
            print(f"    speed_factor[{rc}]={sf} ... ", end="", flush=True)
            obj = _evaluate(trial)
            print(f"obj={obj:.2f}")
            if obj < best_obj:
                best_obj = obj
                best_params = trial
                best_sf = sf
        best_params.speed_factors[rc] = best_sf

        # Capacity factor
        best_cf = best_params.capacity_factors.get(rc, 1.0)
        for cf in cap_range:
            if cf == best_cf:
                continue
            trial = SupplyParams(
                speed_factors=dict(best_params.speed_factors),
                capacity_factors={**best_params.capacity_factors, rc: cf},
            )
            print(f"    capacity_factor[{rc}]={cf} ... ", end="", flush=True)
            obj = _evaluate(trial)
            print(f"obj={obj:.2f}")
            if obj < best_obj:
                best_obj = obj
                best_params = trial
                best_cf = cf
        best_params.capacity_factors[rc] = best_cf

    # Apply best params and run final full calibration
    calib_cfg["max_iterations"] = orig_max_iter
    print(f"\n  Best params: speed={best_params.speed_factors}, "
          f"capacity={best_params.capacity_factors}, obj={best_obj:.2f}")
    apply_supply_params(project_dir, best_params)

    tuning_report = {
        "best_objective": round(best_obj, 3),
        "speed_factors": best_params.speed_factors,
        "capacity_factors": best_params.capacity_factors,
        "config": {
            "road_classes": road_classes,
            "speed_range": speed_range,
            "capacity_range": cap_range,
            "inner_max_iterations": inner_max_iter,
            "objective_weights": obj_weights,
        },
    }
    report_path = output_dir / "supply_tuning_report.json"
    report_path.write_text(json.dumps(tuning_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Supply tuning report: {report_path}")
    print("  Running final calibration with best parameters ...")
    run_calibration(config_path)


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
    direction_aware = bool(calib_cfg.get("match_direction_aware", True))
    conflict_res = str(calib_cfg.get("match_conflict_resolution", "nearest"))
    agg_corridor = bool(calib_cfg.get("aggregate_corridor", True))

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

    count_target = str(calib_cfg.get("count_target", "total"))
    obs_col = "observed_car" if count_target == "car_only" else "observed_total"

    print("=== FSM ITERATIVE CALIBRATION ===")
    print(f"  max_iterations={max_iterations}, target GEH<5 >= {geh_target}%")
    print(f"  scaling: {scale_method}, damping={damping}")
    print(f"  count_target: {count_target} (comparing against '{obs_col}')")

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
    validate_geometries_or_fail(pent, name="pentlogram", expected_epsg=5514)
    links_gdf = _load_network_links(project_dir)
    print(f"  Pentlogram: {len(pent)} observed segments")

    # Load screenlines (for select-link OD correction and evaluation)
    from sim.screenlines import load_screenlines, resolve_screenline_links, evaluate_all_screenlines
    sl_path = str(calib_cfg.get("screenlines_path", "config/screenlines.yaml"))
    screenlines = load_screenlines(sl_path)
    sl_query: Optional[Dict[str, list]] = None
    if screenlines:
        sl_query = {}
        for sl in screenlines:
            resolved = sl.links or resolve_screenline_links(sl, links_gdf)
            if resolved:
                sl.links = resolved
                sl_query[sl.name] = [(lid, d) for lid, d in resolved]
        print(f"  Screenlines: {len(screenlines)} defined, {len(sl_query)} with links")

    # Load matrix (stays open across iterations)
    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])

    project = Project()
    project.open(str(project_dir))

    history: List[Dict[str, Any]] = []
    prev_geh5 = 0.0
    mq: Dict[str, Any] = {}

    try:
        for it in range(1, max_iterations + 1):
            print(f"\n── Iteration {it}/{max_iterations} ──")

            # 1) Assignment
            total_demand = float(mat.matrix_view.sum())
            print(f"  Demand total: {total_demand:,.0f}")
            save_skims_now = bool(calib_cfg.get("save_skims", False)) and it == 1
            use_select_link = scale_method == "select_link" and sl_query
            vol_df, skims, sl_matrices = execute_assignment(
                project, mat,
                algorithm=algorithm,
                max_iter=max_iter_assign,
                rgap_target=rgap,
                save_skims=save_skims_now,
                select_links=sl_query if use_select_link else None,
            )
            if skims is not None:
                skim_path = output_dir / "skims.aem"
                try:
                    skims.export(str(skim_path))
                    print(f"  Skims saved: {skim_path}")
                except Exception:
                    pass

            vol_col = _detect_volume_col(vol_df)
            total_vol = float(vol_df[vol_col].sum()) if vol_col else 0.0
            print(f"  Assigned volume: {total_vol:,.0f}  (col={vol_col})")

            # 2) Match to pentlogram
            links_with_vol = links_gdf.copy()
            if vol_col and "link_id" in vol_df.columns:
                links_with_vol = links_with_vol.merge(
                    vol_df[["link_id", vol_col]], on="link_id", how="left",
                )

            matched = match_counts_to_links(
                pent, links_with_vol, buffer_m=buffer_m,
                direction_aware=direction_aware,
                conflict_resolution=conflict_res,
                aggregate_corridor=agg_corridor,
            )
            vc = vol_col if vol_col and vol_col in matched.columns else next(
                (c for c in matched.columns if vol_col and vol_col in c), None
            )

            # 3) Compute stats (use corridor volume when available)
            compare_col = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
            if vc and compare_col and compare_col in matched.columns:
                valid = matched.dropna(subset=[compare_col, obs_col])
                valid = valid[valid[obs_col] > 0].copy()
                stats = compute_stats(valid[compare_col].values, valid[obs_col].values)
            else:
                valid = pd.DataFrame()
                stats = {"n": 0}

            geh5 = float(stats.get("geh_lt5_pct", 0))
            geh10 = float(stats.get("geh_lt10_pct", 0))
            r2 = stats.get("r2")
            print(f"  GEH<5: {geh5:.1f}%  GEH<10: {geh10:.1f}%  R²: {r2}")

            mq = match_quality_report(matched)
            if it == 1:
                print(f"  Match quality: {mq['n_matched']}/{mq['n_total']} matched, "
                      f"{mq['n_link_conflicts']} conflicts, "
                      f"mean_dist={mq['mean_match_distance_m']}m")

                # Export matching diagnostics CSV
                _export_matching_diagnostics(matched, compare_col, obs_col, output_dir)

            # Evaluate screenlines
            sl_results: Dict[str, Any] = {}
            if screenlines and vol_col:
                sl_res = evaluate_all_screenlines(
                    screenlines, vol_df, matched, vol_col, obs_col, links_gdf,
                )
                for sn, sr in sl_res.items():
                    sl_results[sn] = sr.to_dict()
                    if it == 1 or it == max_iterations:
                        print(f"  Screenline '{sn}': mod={sr.modeled_total:,.0f} "
                              f"obs={sr.observed_total:,.0f} ratio={sr.ratio:.2f} GEH={sr.geh:.1f}")

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

            # 5) Scale OD matrix (use compare_col -- corridor volume when available)
            if not valid.empty and compare_col and total_vol > 0:
                if scale_method == "global":
                    factor = _compute_global_factor(
                        valid, compare_col, damping=damping,
                        min_factor=min_factor, max_factor=max_factor,
                        obs_col=obs_col,
                    )
                    print(f"  Global scaling factor: {factor:.3f}")
                    scale_matrix(mat, core_name, factor)
                elif scale_method == "select_link" and sl_matrices:
                    corrections = _apply_select_link_od_correction(
                        mat, core_name, sl_matrices, sl_results,
                        damping=damping, min_factor=min_factor, max_factor=max_factor,
                    )
                    if corrections:
                        print(f"  Select-link corrections: {corrections}")
                    else:
                        print("  Select-link: no corrections applied, falling back to global")
                        factor = _compute_global_factor(
                            valid, compare_col, damping=damping,
                            min_factor=min_factor, max_factor=max_factor,
                            obs_col=obs_col,
                        )
                        scale_matrix(mat, core_name, factor)
                elif scale_method == "sector_od":
                    _do_sector_od_scaling(
                        mat, core_name, valid, compare_col,
                        links_gdf=links_gdf,
                        project_dir=project_dir,
                        damping=damping, min_factor=min_factor,
                        max_factor=max_factor, obs_col=obs_col,
                        sector_cfg=scale_cfg.get("sectors"),
                    )
                else:
                    factors = _compute_sector_factors(
                        valid, compare_col, damping=damping,
                        min_factor=min_factor, max_factor=max_factor,
                        obs_col=obs_col,
                    )
                    if factors:
                        obs_weights = {}
                        for lt in factors:
                            sub = valid[valid["link_type"] == lt]
                            obs_weights[lt] = float(sub[obs_col].sum())
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

    # Per-period validation (uses daily observed × period shares)
    period_stats: Dict[str, Any] = {}
    period_cfg = calib_cfg.get("period_calibration") or {}
    if period_cfg.get("enabled", False) and history:
        try:
            from sim.temporal import load_profile, get_demand_period_shares
            profile = load_profile(cfg)
            period_shares = get_demand_period_shares(profile)
            cal_periods = period_cfg.get("periods", ["am", "pm", "daily"])
            dir_ratios = period_cfg.get("directional_ratios", {})

            print("\n=== PER-PERIOD VALIDATION ===")

            mat_p = AequilibraeMatrix()
            mat_p.load(str(matrix_path))
            project_p = Project()
            project_p.open(str(project_dir))

            try:
                for period in cal_periods:
                    p_core = f"wd_{period}" if period != "daily" else "wd_daily"
                    if p_core not in mat_p.names:
                        print(f"  Skipping {period}: core '{p_core}' not in matrix")
                        continue

                    p_share = period_shares.get(period, 1.0)
                    print(f"\n  Period: {period} (share={p_share:.3f}, core={p_core})")

                    mat_p.computational_view([p_core])
                    p_demand = float(mat_p.matrix_view.sum())
                    print(f"    Demand: {p_demand:,.0f}")

                    vol_df_p, _, _sl_p = execute_assignment(
                        project_p, mat_p,
                        algorithm=algorithm,
                        max_iter=max_iter_assign,
                        rgap_target=rgap,
                    )

                    vc_p = _detect_volume_col(vol_df_p)
                    if not vc_p:
                        continue

                    lv = links_gdf.copy()
                    lv = lv.merge(vol_df_p[["link_id", vc_p]], on="link_id", how="left")
                    m_p = match_counts_to_links(pent, lv, buffer_m=buffer_m)

                    obs_period_col = f"_obs_{period}"
                    if period == "daily":
                        m_p[obs_period_col] = m_p[obs_col]
                    else:
                        m_p[obs_period_col] = m_p[obs_col] * p_share

                    vp = m_p.dropna(subset=[vc_p, obs_period_col])
                    vp = vp[vp[obs_period_col] > 0].copy()
                    if len(vp) > 0:
                        ps = compute_stats(vp[vc_p].values, vp[obs_period_col].values)
                        period_stats[period] = ps
                        print(f"    GEH<5: {ps.get('geh_lt5_pct', 0):.1f}%  "
                              f"R²: {ps.get('r2')}  n={ps.get('n')}")
                    else:
                        print(f"    No valid matched volumes for {period}")
            finally:
                mat_p.close()
                project_p.close()
        except Exception as e:
            print(f"  Period validation error: {e}")

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
            "count_target": count_target,
            "obs_col": obs_col,
        },
        "observed_summary": {
            "total_car": round(float(pent["observed_car"].sum()), 0),
            "total_truck": round(float(pent["observed_truck"].sum()), 0),
            "total_all": round(float(pent["observed_total"].sum()), 0),
            "n_count_stations": len(pent),
        },
        "match_quality": mq,
        "period_stats": period_stats,
        "screenlines": sl_results,
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

def match_csd_to_links(
    csd: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
    metric_epsg: int = 5514,
) -> pd.DataFrame:
    """Match CSD2020 sections to model links by road class and spatial proximity."""
    csd = csd.copy()
    if "sil" in csd.columns:
        csd["road_class"] = csd["sil"].apply(_classify_csd_road)
    else:
        csd["road_class"] = "other"

    model_class_map = {}
    if "link_type" in links_gdf.columns:
        for lt in links_gdf["link_type"].dropna().unique():
            lt_str = str(lt).lower()
            for rc in ("motorway", "trunk", "primary", "secondary", "tertiary"):
                if rc in lt_str:
                    model_class_map[str(lt)] = rc
                    break

    matched_rows = []
    for rc in csd["road_class"].unique():
        csd_sub = csd[csd["road_class"] == rc]
        model_types = [k for k, v in model_class_map.items() if v == rc]
        if not model_types:
            continue
        links_sub = links_gdf[links_gdf["link_type"].isin(model_types)]
        if links_sub.empty:
            continue

        csd_mean_aadt = float(csd_sub["o"].mean()) if "o" in csd_sub.columns else 0
        csd_total_aadt = float(csd_sub["o"].sum()) if "o" in csd_sub.columns else 0
        model_mean_vol = 0.0
        model_total_vol = 0.0
        vol_cols = [c for c in links_sub.columns if c.endswith("_tot") and links_sub[c].sum() > 0]
        if vol_cols:
            vc = vol_cols[0]
            model_mean_vol = float(links_sub[vc].mean())
            model_total_vol = float(links_sub[vc].sum())

        matched_rows.append({
            "road_class": rc,
            "csd_sections": len(csd_sub),
            "model_links": len(links_sub),
            "csd_mean_cars": round(csd_mean_aadt, 0),
            "csd_total_cars": round(csd_total_aadt, 0),
            "model_mean_vol": round(model_mean_vol, 0),
            "model_total_vol": round(model_total_vol, 0),
        })

    return pd.DataFrame(matched_rows)


def compute_validation_benchmarks(
    count_stats: Dict[str, Any],
    screenline_results: Dict[str, Any],
    jt_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check FHWA/Scottish validation benchmarks."""
    geh5 = float(count_stats.get("geh_lt5_pct", 0))
    geh_pass = geh5 >= 85.0

    jt_pass_count = sum(1 for r in jt_results if r.get("pass", False))
    jt_total = len(jt_results) if jt_results else 0
    jt_pct = jt_pass_count / max(jt_total, 1) * 100
    jt_pass = jt_pct >= 85.0 if jt_total > 0 else None

    sl_max_error = 0.0
    for sr in screenline_results.values():
        ratio = sr.get("ratio")
        if ratio is not None:
            sl_max_error = max(sl_max_error, abs(ratio - 1.0) * 100)

    return {
        "geh_lt5_pct": round(geh5, 1),
        "geh_benchmark_pass": geh_pass,
        "jt_within_tolerance_pct": round(jt_pct, 1) if jt_total > 0 else None,
        "jt_benchmark_pass": jt_pass,
        "jt_routes_checked": jt_total,
        "screenline_max_error_pct": round(sl_max_error, 1),
        "overall_pass": geh_pass and (jt_pass is None or jt_pass),
    }


def run_validation_only(config_path: str | Path = "config/sim.yaml") -> None:
    """Comprehensive independent validation against CSD2020 + screenlines + journey times."""
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)
    buffer_m = float(_get(cfg, ["calibration", "match_buffer_m"], 50.0))

    print("=== COMPREHENSIVE VALIDATION ===")

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

    count_target = str(calib_cfg.get("count_target", "total"))
    obs_col = "observed_car" if count_target == "car_only" else "observed_total"

    report: Dict[str, Any] = {}

    # 1) Pentlogram (reference -- same data as calibration)
    print("\n1) Pentlogram comparison (reference) ...")
    pent_stats: Dict[str, Any] = {"n": 0}
    try:
        pent = load_pentlogram(cfg)
        validate_geometries_or_fail(pent, name="pentlogram", expected_epsg=5514)
        agg_corr = bool(calib_cfg.get("aggregate_corridor", True))
        matched = match_counts_to_links(pent, links_gdf, buffer_m=buffer_m,
                                         aggregate_corridor=agg_corr)
        vc = vol_col if vol_col and vol_col in matched.columns else None
        compare_vc = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
        if compare_vc and compare_vc in matched.columns:
            valid = matched.dropna(subset=[compare_vc, obs_col])
            valid = valid[valid[obs_col] > 0]
            pent_stats = compute_stats(valid[compare_vc].values, valid[obs_col].values)
            report["pentlogram"] = {"matched": int(len(valid)), **pent_stats}
            print(f"  Matched: {len(valid)}  GEH<5: {pent_stats.get('geh_lt5_pct')}%  "
                  f"R²: {pent_stats.get('r2')}")
        else:
            print("  No volume column on links")
    except Exception as e:
        print(f"  SKIP: {e}")

    # 2) CSD2020 -- independent link-level validation
    print("\n2) CSD2020 independent validation ...")
    csd_match_df = pd.DataFrame()
    try:
        csd = load_csd2020(cfg)
        csd_agg = aggregate_csd_by_class(csd)
        report["csd2020_observed"] = csd_agg.to_dict(orient="records")

        if vol_col and vol_col in links_gdf.columns:
            model_agg = aggregate_model_by_class(links_gdf, vol_col)
            report["csd2020_modeled"] = model_agg.to_dict(orient="records")

        csd_match_df = match_csd_to_links(csd, links_gdf)
        if not csd_match_df.empty:
            report["csd2020_link_matching"] = csd_match_df.to_dict(orient="records")

        print(f"  CSD2020 JMK: {len(csd)} sections")
        for _, r in csd_agg.iterrows():
            print(f"    {r['road_class']:12s}  sections={int(r['sections']):4d}  "
                  f"mean_AADT={r['mean_sv']:>8.0f}  mean_cars={r['mean_o']:>8.0f}")
        if not csd_match_df.empty:
            print(f"  Link-matched: {len(csd_match_df)} road classes")
            for _, r in csd_match_df.iterrows():
                print(f"    {r['road_class']:12s}  csd_mean={r['csd_mean_cars']:>8.0f}  "
                      f"model_mean={r['model_mean_vol']:>8.0f}")
    except Exception as e:
        print(f"  SKIP: {e}")

    # 3) Screenline validation
    print("\n3) Screenline validation ...")
    sl_results: Dict[str, Any] = {}
    try:
        from sim.screenlines import load_screenlines, evaluate_all_screenlines
        sl_path = str(calib_cfg.get("screenlines_path", "config/screenlines.yaml"))
        screenlines = load_screenlines(sl_path)
        if screenlines:
            sl_res = evaluate_all_screenlines(
                screenlines, vol_df,
                    matched if "matched" in locals() else gpd.GeoDataFrame(),
                vol_col or "", obs_col, links_gdf,
            )
            for sn, sr in sl_res.items():
                sl_results[sn] = sr.to_dict()
                print(f"  {sn}: mod={sr.modeled_total:,.0f} obs={sr.observed_total:,.0f} "
                      f"ratio={sr.ratio:.2f} GEH={sr.geh:.1f}")
            report["screenlines"] = sl_results
        else:
            print("  No screenlines defined")
    except Exception as e:
        print(f"  SKIP: {e}")

    # 4) Journey time validation
    print("\n4) Journey time validation ...")
    jt_results: List[Dict[str, Any]] = []
    speed_comparison: Dict[str, Any] = {}
    try:
        # Class-level speed comparison from CSD2020
        if "csd" in dir() and not csd.empty:
            speed_comparison = compute_class_speed_comparison(links_gdf, csd)
            report["class_speed_comparison"] = speed_comparison
            if speed_comparison:
                print("  Speed comparison (model vs CSD2020):")
                for lt, sc in speed_comparison.items():
                    print(f"    {lt:20s}  model={sc['modeled_kmh']:>5.1f}  "
                          f"ref={sc.get('reference_kmh', sc.get('csd_implied_kmh', 0)):>5.1f}  "
                          f"diff={sc['pct_diff']:>+5.1f}%")

        # Reference route checks from skims
        jt_cfg = calib_cfg.get("journey_time_validation", {})
        routes = jt_cfg.get("reference_routes", [])
        if routes and jt_cfg.get("enabled", True):
            skim_path = output_dir / "skims.aem"
            if skim_path.exists():
                mat = AequilibraeMatrix()
                mat.load(str(skim_path))
                names = list(mat.names)
                if names:
                    zone_ids = mat.index[:].copy()
                    skim_data = mat.matrix[names[0]][:, :].copy()
                    mat.close()
                    jt_results = validate_journey_times(skim_data, zone_ids, routes)
                    report["journey_time_routes"] = jt_results
                    for r in jt_results:
                        status = "PASS" if r.get("pass") else "FAIL"
                        print(f"  {r.get('name', '?'):20s}  model={r.get('model_min', '?')}min  "
                              f"ref={r.get('ref_min', '?')}min  {status}")
                else:
                    mat.close()
            else:
                print("  No skims available -- skip route checks")
        else:
            print("  No reference routes configured")
    except Exception as e:
        print(f"  SKIP: {e}")

    # 5) Benchmark summary
    print("\n5) FHWA / Scottish benchmarks ...")
    benchmarks = compute_validation_benchmarks(pent_stats, sl_results, jt_results)
    report["benchmarks"] = benchmarks
    overall = "PASS" if benchmarks["overall_pass"] else "FAIL"
    print(f"  GEH<5 >= 85%:  {benchmarks['geh_lt5_pct']:.1f}%  "
          f"{'PASS' if benchmarks['geh_benchmark_pass'] else 'FAIL'}")
    if benchmarks["jt_benchmark_pass"] is not None:
        print(f"  JT within tol:  {benchmarks['jt_within_tolerance_pct']:.1f}%  "
              f"{'PASS' if benchmarks['jt_benchmark_pass'] else 'FAIL'}")
    print(f"  Screenline max error: {benchmarks['screenline_max_error_pct']:.1f}%")
    print(f"  OVERALL: {overall}")

    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nValidation report: {report_path}")
