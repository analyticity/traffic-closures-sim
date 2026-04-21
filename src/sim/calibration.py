"""Traffic model calibration and independent validation.

ODME calibration (preferred — ``run_odme_calibration``)
-------------------------------------------------------
Spiess-style gradient-based OD Matrix Estimation.  Bi-level optimization:
  Lower level: AequilibraE equilibrium assignment
  Upper level: minimize weighted sum-of-squared residuals between modeled
               and observed link volumes via relative-gradient descent.
Select-link OD proportions from screenlines steer the gradient;
a damped global residual correction handles aggregate bias from all
matched count posts.  Elasticity bounds prevent overfitting.

Legacy calibration (``run_calibration``)
----------------------------------------
Iterative FSM loop: assign → compare → scale OD → repeat.
Preserved for backward compatibility.

Validation
----------
After calibration converges, ``validate`` compares the *final* assignment
to an independent dataset (CSD) that was **not** used during calibration.
"""
from __future__ import annotations

import json
import math
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import get_metric_epsg, load_config
from sim.assignment import (
    build_graph,
    execute_assignment,
    fix_node_ids,
    resolve_daily_cap_factor_default,
    _detect_volume_col,
)
import logging

logger = logging.getLogger(__name__)

from sim.calibration_state import CalibrationRun

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


_SCREENLINE_TO_GATEWAY = {
    "D1_west": "D1_NW",
    "D1_east": "D1_E",
    "D2_south": "D2_S",
    "I43_north": "I43_N",
    "I52_south": "I52_S",
}


# ---------------------------------------------------------------------------
# Gateway calibration helpers
# ---------------------------------------------------------------------------

def _load_gateway_zone_map(
    cfg: Dict[str, Any],
    mat_index: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build gateway_name -> array of matrix row/col indices for external zones.

    Reads ``zones.geojson`` from the zoning output directory, selects rows
    where ``is_external == 1``, and maps each ``gateway_name`` to the
    corresponding matrix indices (positions in ``mat_index``).

    The matrix uses centroid IDs (1..N) while zones.geojson uses zone_ids
    (which may be large synthetic values like 8000000020 for external zones).
    A zone_centroid_mapping.json bridges the two ID spaces.
    """
    zoning_dir = Path(_get(cfg, ["zoning", "output_dir"], "outputs/baseline/zones"))
    zones_path = zoning_dir / "zones.geojson"
    if not zones_path.exists():
        return {}

    zones_gdf = gpd.read_file(zones_path)
    if "is_external" not in zones_gdf.columns or "zone_id" not in zones_gdf.columns:
        return {}

    external = zones_gdf[zones_gdf["is_external"].fillna(0).astype(int) == 1].copy()
    if external.empty:
        return {}

    mapping_path = zoning_dir / "zone_centroid_mapping.json"
    zone_to_centroid: Dict[int, int] = {}
    if mapping_path.exists():
        import json
        with open(mapping_path) as f:
            raw = json.load(f)
        zone_to_centroid = {int(k): int(v) for k, v in raw.items()}

    idx_lookup = {int(v): i for i, v in enumerate(mat_index)}
    result: Dict[str, np.ndarray] = {}

    for _, row in external.iterrows():
        gw = str(row.get("gateway_name", "")).strip()
        if not gw:
            name = str(row.get("name", "")).strip()
            gw = name.replace("EXT_", "", 1) if name.startswith("EXT_") else name
        if not gw:
            continue
        zid = int(row["zone_id"])
        centroid_id = zone_to_centroid.get(zid, zid)
        mat_idx = idx_lookup.get(centroid_id)
        if mat_idx is not None:
            result.setdefault(gw, []).append(mat_idx)

    return {gw: np.array(indices, dtype=int) for gw, indices in result.items()}


def _load_gateway_observed(
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Load observed AADT per gateway for gateway calibration.

    Tries the parquet at ``calibration.gateway_calibration.observed_path``
    first (columns: ``gateway_name``, ``observed_aadt``).  If not found,
    falls back to CSD values from the screenlines YAML, using a hardcoded
    screenline-name -> gateway-name mapping.
    """
    gw_cfg = _get(cfg, ["calibration", "gateway_calibration"], {}) or {}
    obs_path = Path(gw_cfg.get("observed_path", "data/cache/gateway_counts_2025.parquet"))

    if obs_path.exists():
        try:
            df = pd.read_parquet(obs_path)
            if "gateway_name" in df.columns and "observed_aadt" in df.columns:
                return {
                    str(r["gateway_name"]).strip(): float(r["observed_aadt"])
                    for _, r in df.iterrows()
                    if float(r["observed_aadt"]) > 0
                }
        except Exception:
            logger.debug(
                "Failed to load gateway observed counts from parquet; falling back to YAML",
                exc_info=True,
            )

    from sim.screenlines import load_screenlines
    sl_path = str(_get(cfg, ["calibration", "screenlines_path"], "config/screenlines.yaml"))
    screenlines = load_screenlines(sl_path)

    result: Dict[str, float] = {}
    for sl in screenlines:
        gw = _SCREENLINE_TO_GATEWAY.get(sl.name)
        aadt = getattr(sl, "observed_aadt_cars", None)
        if gw and aadt and float(aadt) > 0:
            result[gw] = float(aadt)

    return result


def _apply_gateway_calibration(
    demand: np.ndarray,
    gateway_zone_map: Dict[str, np.ndarray],
    gateway_observed: Dict[str, float],
    gateway_modeled: Dict[str, float],
    *,
    damping: float = 0.08,
    min_factor: float = 0.90,
    max_factor: float = 1.10,
    seed_lower: Optional[np.ndarray] = None,
    seed_upper: Optional[np.ndarray] = None,
) -> List[str]:
    """Scale OD rows/columns for each gateway's external zones toward observed AADT.

    For each gateway with both observed and modeled totals, compute
    ``factor = 1 + damping * (obs/mod - 1)`` clipped to ``[min_factor, max_factor]``
    and multiply all OD cells where either origin or destination belongs
    to that gateway's external zones.

    Returns a list of log strings describing corrections applied.
    """
    corrections: List[str] = []
    n = demand.shape[0]

    for gw_name, ext_indices in gateway_zone_map.items():
        obs = gateway_observed.get(gw_name, 0.0)
        mod = gateway_modeled.get(gw_name, 0.0)
        if obs <= 0 or mod <= 0:
            continue

        ratio = obs / mod
        if ratio > 10.0 or ratio < 0.1:
            continue

        factor = 1.0 + damping * (ratio - 1.0)
        factor = float(np.clip(factor, min_factor, max_factor))

        if abs(factor - 1.0) < 0.001:
            continue

        mask = np.zeros(n, dtype=bool)
        mask[ext_indices] = True

        demand[mask, :] *= factor
        demand[:, mask] *= factor
        # Undo double-scaling of ext-ext cells within same gateway
        demand[np.ix_(mask, mask)] /= factor

        if seed_lower is not None and seed_upper is not None:
            rows_to_clip = np.where(mask)[0]
            for ri in rows_to_clip:
                np.clip(demand[ri, :], seed_lower[ri, :], seed_upper[ri, :], out=demand[ri, :])
                np.clip(demand[:, ri], seed_lower[:, ri], seed_upper[:, ri], out=demand[:, ri])

        corrections.append(f"{gw_name}: obs={obs:,.0f} mod={mod:,.0f} "
                           f"ratio={ratio:.2f} factor={factor:.4f}")

    return corrections


def _compute_gateway_modeled_volumes(
    vol_df: pd.DataFrame,
    screenlines: list,
    vol_col: str,
) -> Dict[str, float]:
    """Sum modeled volume on each screenline and map to gateway names."""
    from sim.screenlines import _get_link_volume

    result: Dict[str, float] = {}
    if not screenlines or not vol_col:
        return result

    for sl in screenlines:
        gw = _SCREENLINE_TO_GATEWAY.get(sl.name)
        if not gw or not sl.links:
            continue

        total = 0.0
        for link_id, direction in sl.links:
            total += _get_link_volume(vol_df, link_id, direction, vol_col)
        result[gw] = total

    return result


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
        get_metric_epsg(cfg),
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
                logger.debug(
                    "Failed to compute pentlogram sample bounds; assuming non-metric CRS",
                    exc_info=True,
                )
                max_abs = 0

            declared_epsg = gdf.crs.to_epsg()
            if declared_epsg == 4326 and max_abs > 1000:
                logger.warning(
                    f"  WARNING: GeoJSON declared as EPSG:4326 but coordinates "
                    f"are metric (max={max_abs:.0f}). Overriding to EPSG:{out_epsg}."
                )
                gdf = gdf.set_crs(epsg=out_epsg, allow_override=True)
            elif declared_epsg != out_epsg:
                gdf = gdf.to_crs(epsg=out_epsg)

    # Fail-fast geometry validation
    valid_geom = gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~valid_geom).sum())
    if n_invalid > 0:
        logger.warning(f"  WARNING: {n_invalid}/{len(gdf)} pentlogram features have invalid geometry")
        gdf = gdf[valid_geom].copy()

    for col in ("car_24", "truc_24"):
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    # Pentlogram data from data.Brno:
    #   car_24  = total motor vehicles in thousands per 24h (ALL vehicles, not cars-only)
    #   truc_24 = percentage of trucks/buses (0-100), NOT absolute count
    units_cfg = _get(cfg, [
        "datasets", "sources", "calibration_brno_pentlogram_2024", "units",
    ], {}) or {}
    car_mult = float(units_cfg.get("car_24_multiplier", 1000))

    total_vehicles = gdf.get("car_24", 0) * car_mult
    truck_pct = gdf.get("truc_24", 0).clip(0, 100)

    gdf["observed_motor_total"] = total_vehicles
    gdf["observed_truck"] = total_vehicles * truck_pct / 100.0
    gdf["observed_total"] = total_vehicles
    # Backward-compat alias (deprecated — use observed_motor_total)
    gdf["observed_car"] = gdf["observed_motor_total"]

    nz = gdf[gdf["observed_motor_total"] > 0]["observed_motor_total"]
    if not nz.empty:
        p50 = float(nz.median())
        p_max = float(nz.max())
        if p_max < 500:
            logger.warning(
                f"  WARNING: observed_motor_total max={p_max:.0f} seems very low. "
                f"Check car_24_multiplier (currently {car_mult})."
            )
        elif p50 > 100000:
            logger.warning(
                f"  WARNING: observed_motor_total median={p50:.0f} seems very high. "
                f"Check car_24_multiplier (currently {car_mult})."
            )
        else:
            logger.info(
                f"  Pentlogram observed_motor_total: median={p50:,.0f} max={p_max:,.0f} "
                f"(multiplier={car_mult})"
            )

    gdf = gdf[gdf["observed_total"] > 0].copy()

    # Spatial consistency check: flag segments whose value is drastically
    # lower than bearing-aligned neighbors (service roads / ramps running
    # parallel to a major road).  These cause mis-matches when a 9k ramp
    # segment gets assigned to a 26k trunk model link.
    gdf = _flag_neighbor_outliers(gdf, metric_epsg=out_epsg)

    return gdf


def _flag_neighbor_outliers(
    gdf: gpd.GeoDataFrame,
    *,
    metric_epsg: int = 5514,
    search_radius_m: float = 100.0,
    bearing_tol: float = 30.0,
    min_neighbors: int = 2,
    low_ratio: float = 0.30,
) -> gpd.GeoDataFrame:
    """Drop pentlogram segments inconsistent with bearing-aligned neighbors.

    A segment is dropped when its ``observed_car`` is below *low_ratio*
    of the median of nearby segments running in the same (or opposite)
    direction.  These are typically service roads / ramps measured
    separately but spatially overlapping with a major road.
    """
    work = gdf.to_crs(epsg=metric_epsg) if gdf.crs and gdf.crs.to_epsg() != metric_epsg else gdf

    bearings = np.array([_bearing_from_geom(g) for g in work.geometry], dtype=object)
    centroids = work.geometry.centroid
    obs = gdf["observed_car"].values.astype(float)

    drop_mask = np.zeros(len(gdf), dtype=bool)

    has_sindex = hasattr(work, "sindex")

    for i in range(len(work)):
        b = bearings[i]
        if b is None or obs[i] <= 0:
            continue
        pt = centroids.iloc[i]
        if pt is None or pt.is_empty:
            continue

        if has_sindex:
            buf = pt.buffer(search_radius_m)
            cand_idxs = list(work.sindex.query(buf, predicate="intersects"))
        else:
            continue

        neighbor_vals = []
        for ci in cand_idxs:
            if ci == i:
                continue
            nb = bearings[ci]
            if nb is None:
                continue
            bdiff = _bearing_diff(b, nb)
            if bdiff > bearing_tol and abs(bdiff - 180) > bearing_tol:
                continue
            if obs[ci] > 0:
                neighbor_vals.append(obs[ci])

        if len(neighbor_vals) < min_neighbors:
            continue

        med = float(np.median(neighbor_vals))
        if med > 0 and obs[i] / med < low_ratio:
            drop_mask[i] = True

    n_dropped = int(drop_mask.sum())
    if n_dropped > 0:
        logger.info(
            f"  Pentlogram validation: dropped {n_dropped} segments "
            f"inconsistent with neighbors (ratio < {low_ratio})"
        )
    return gdf[~drop_mask].copy()


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


def load_csd(cfg: Dict[str, Any], region_code: str | None = None) -> pd.DataFrame:
    """Load CSD parquet with optional region filter from locale config.

    When *region_code* is ``None`` (default), the region filter is read
    from ``config/locale.yaml`` (``csd_region_filter``).  Pass an explicit
    code like ``"CZ064"`` to override.
    """
    from sim.io_project import load_locale

    cache_dir = Path(_get(cfg, ["datasets", "cache_dir"], "data/cache"))
    parquet_path = cache_dir / "v2_csd2025.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"CSD parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if region_code is not None:
        if "kk" in df.columns:
            df = df[df["kk"].astype(str).str.contains(
                region_code.replace("CZ", ""), na=False,
            )].copy()
    else:
        locale = load_locale(cfg)
        csd_filter = locale.get("csd_region_filter") or {}
        filter_col = csd_filter.get("column", "kk")
        filter_val = str(csd_filter.get("contains", "064"))
        if filter_col in df.columns:
            df = df[df[filter_col].astype(str).str.contains(filter_val, na=False)].copy()

    if "sil" in df.columns:
        df["sil"] = df["sil"].astype(str)
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
    """Sum volumes from the single best opposing carriageway for one-way links.

    For one-way (``direction != 0``) links the pentlogram count is
    bidirectional, so we need the volume from exactly **one** opposing
    carriageway link to reconstruct the full cross-section.  Candidates
    are scored by: same road ``name`` (strong bonus), opposite bearing
    (within *bearing_tol* of 180°), and proximity.  Only the **single
    best** candidate is added (max 2 links total).

    Bidirectional links (``direction == 0``) already carry AB+BA and are
    left untouched.
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
    link_dir = links["direction"].values if "direction" in links.columns else np.zeros(len(links))
    link_name = (
        links["name"].fillna("").astype(str).values
        if "name" in links.columns
        else np.array([""] * len(links))
    )

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

        mi = matched_idx_in_links[0]
        matched_direction = int(link_dir[mi])

        if matched_direction == 0:
            continue

        mb = link_bearing[mi]
        if mb is None:
            continue
        matched_name = str(link_name[mi])

        if has_sindex:
            buf_geom = count_pt.buffer(buffer_m)
            cand_idxs = list(links.sindex.query(buf_geom, predicate="intersects"))
        else:
            cand_idxs = range(len(links))

        best_score = -1.0
        best_vol = 0.0

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

            bdiff = _bearing_diff(mb, lb)
            if abs(bdiff - 180) > bearing_tol:
                continue

            if not has_sindex:
                dist = count_pt.distance(links.geometry.iloc[ci])
                if dist > buffer_m:
                    continue
                dist_score = 1.0 - min(dist / buffer_m, 1.0)
            else:
                d = count_pt.distance(links.geometry.iloc[ci])
                dist_score = 1.0 - min(d / buffer_m, 1.0)

            bearing_score = 1.0 - abs(bdiff - 180) / bearing_tol

            name_score = 0.0
            cand_name = str(link_name[ci])
            if matched_name and cand_name and matched_name == cand_name:
                name_score = 1.0

            score = 0.30 * dist_score + 0.30 * bearing_score + 0.40 * name_score
            if score > best_score:
                best_score = score
                best_vol = float(link_vol[ci])

        if best_score >= 0:
            base_vol = float(row.get(vc, 0) or 0)
            joined.at[idx, "_corridor_volume"] = base_vol + best_vol
            joined.at[idx, "_corridor_n_links"] = 2

    return joined


_NON_CAR_LINK_TYPES = frozenset({
    "footway", "path", "track", "steps", "cycleway", "pedestrian",
    "corridor", "bridleway", "proposed", "construction", "elevator",
    "service", "rest_area", "services", "traffic_mirror", "virtual",
    "crossing", "busway", "centroid_connector",
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
    vol_col: Optional[str] = None,
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
        logger.info(f"  Matching: filtered {n_before} -> {n_after} car-driveable links")

    pts = counts.copy()
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty].copy()
    count_geom_backup = pts.geometry.copy()
    pts["_count_bearing"] = count_geom_backup.apply(_bearing_from_geom)
    centroids = pts.geometry.centroid
    valid_cent = np.isfinite(centroids.x) & np.isfinite(centroids.y)
    n_dropped = int((~valid_cent).sum())
    if n_dropped > 0:
        logger.warning(f"  WARNING: dropped {n_dropped} counts with invalid centroid geometry")
    pts = pts[valid_cent].copy()
    pts["geometry"] = centroids[valid_cent]

    keep = ["link_id", "link_type", "name", "osm_ref", "geometry"] + [
        c for c in links.columns
        if c not in ("link_id", "link_type", "name", "osm_ref", "geometry", "ogc_fid")
        and links[c].dtype in ("float64", "float32", "int64")
    ]
    keep = [c for c in keep if c in links.columns]

    links_sel = links[keep].copy()
    links_sel["_link_bearing"] = links_sel.geometry.apply(_bearing_from_geom)

    # --- Multi-candidate matching with composite scoring ---
    # For each count point, evaluate ALL candidate links within buffer_m and
    # pick the best by a weighted composite of distance, bearing, and road
    # class.  This replaces sjoin_nearest + bearing re-match, preventing
    # mis-matches to ramps when a mainline link is nearby.
    _ROAD_CLASS_W: Dict[str, float] = {
        "motorway": 1.0, "trunk": 0.875, "motorway_link": 0.875,
        "trunk_link": 0.75, "primary": 0.5, "primary_link": 0.44,
        "secondary": 0.25, "secondary_link": 0.225, "tertiary": 0.15,
        "tertiary_link": 0.125, "unclassified": 0.09, "road": 0.09,
        "residential": 0.05, "service": 0.025, "living_street": 0.006,
    }
    link_cols_to_copy = [c for c in links_sel.columns if c not in ("geometry", "_link_bearing")]

    records: list = []
    for _, pt_row in pts.iterrows():
        pt = pt_row.geometry
        base: dict = {c: pt_row[c] for c in pts.columns if c != "geometry"}
        base["geometry"] = pt

        if pt is None or pt.is_empty:
            base.update({"_dist": np.nan, "_bearing_diff": np.nan, "_match_quality": 0.0})
            records.append(base)
            continue

        cb = pt_row.get("_count_bearing")
        cand_idxs = list(links_sel.sindex.query(pt.buffer(buffer_m), predicate="intersects"))

        best_q = -1.0
        best_data: Optional[dict] = None
        for ci in cand_idxs:
            cand = links_sel.iloc[ci]
            d = float(pt.distance(cand.geometry))
            if d > buffer_m:
                continue

            bd = 0.0
            if direction_aware:
                lb = cand.get("_link_bearing")
                if cb is not None and lb is not None:
                    try:
                        if not np.isnan(float(cb)) and not np.isnan(float(lb)):
                            bd = _bearing_diff(float(cb), float(lb))
                    except (TypeError, ValueError):
                        pass

            lt = str(cand.get("link_type", ""))
            rw = _ROAD_CLASS_W.get(lt, 0.06)
            dn = d / max(buffer_m, 1.0)
            bp = bd / 180.0

            vb = 0.0
            if vol_col and vol_col in cand.index:
                try:
                    v = float(cand[vol_col])
                    if v > 0:
                        vb = 1.0
                        obs_v = float(pt_row.get("observed_car", 0))
                        if obs_v > 0:
                            ratio = min(obs_v, v) / max(obs_v, v)
                            if ratio < 0.15:
                                vb = 0.1
                            elif ratio < 0.30:
                                vb = 0.4
                except (TypeError, ValueError):
                    pass

            q = 0.30 * (1.0 - dn) + 0.30 * (1.0 - bp) + 0.20 * rw + 0.20 * vb

            if q > best_q:
                best_q = q
                best_data = {c: cand[c] for c in link_cols_to_copy if c in cand.index}
                best_data["_dist"] = d
                best_data["_bearing_diff"] = bd
                best_data["_match_quality"] = q

        if best_data is not None:
            base.update(best_data)
        else:
            base.update({"_dist": np.nan, "_bearing_diff": np.nan, "_match_quality": 0.0})
        records.append(base)

    joined = gpd.GeoDataFrame(records, crs=pts.crs)

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
            obs_col_for_sort = "observed_car" if "observed_car" in joined.columns else None
            _MAJOR_TYPES = {"trunk", "trunk_link", "motorway", "motorway_link", "primary", "primary_link"}
            is_major = (
                joined["link_type"].astype(str).isin(_MAJOR_TYPES)
                if "link_type" in joined.columns
                else pd.Series(False, index=joined.index)
            )

            if obs_col_for_sort and is_major.any():
                major = joined[is_major].sort_values(obs_col_for_sort, ascending=False)
                major = major.drop_duplicates(subset=["link_id"], keep="first")
                minor = joined[~is_major].sort_values("_dist")
                minor = minor.drop_duplicates(subset=["link_id"], keep="first")
                joined = pd.concat([major, minor], ignore_index=True)
                joined = joined.drop_duplicates(subset=["link_id"], keep="first")
            else:
                joined = joined.sort_values("_dist")
                joined = joined.drop_duplicates(subset=["link_id"], keep="first")
    else:
        joined["_link_conflict"] = False

    joined["_matched"] = joined["link_id"].notna() if "link_id" in joined.columns else False

    # Flag structurally unreliable matches: very low modeled volume relative
    # to a high observed count — almost always a wrong-link match (e.g. ramp
    # or parallel residential road instead of the actual measured carriageway).
    joined["_excluded"] = False
    if vol_col and vol_col in joined.columns:
        obs_cols = [c for c in joined.columns if c.startswith("observed")]
        if obs_cols:
            obs_max = joined[obs_cols].max(axis=1).fillna(0)
            vol_vals = pd.to_numeric(joined[vol_col], errors="coerce").fillna(0)
            bad_zero = (vol_vals <= 0) & (obs_max >= 5000)
            bad_ratio = (obs_max >= 2000) & (vol_vals < obs_max * 0.01)
            bad = bad_zero | bad_ratio
            n_excluded = int(bad.sum())
            if n_excluded > 0:
                joined.loc[bad, "_excluded"] = True
                logger.info(
                    f"  Matching: excluded {n_excluded} zero-volume links with high observed counts"
                )

        # Reverse mismatch: observed is far below modeled on major roads.
        # E.g. a 9k pentlogram segment from a road below a bridge matched
        # to a 62k trunk link above — clearly a spatial mis-match.
        if "link_type" in joined.columns:
            _MAJOR_EX = {"trunk", "trunk_link", "motorway", "motorway_link"}
            is_major = joined["link_type"].astype(str).isin(_MAJOR_EX)
            obs_car = pd.to_numeric(
                joined["observed_car"] if "observed_car" in joined.columns else 0,
                errors="coerce",
            ).fillna(0)
            extreme_low = (
                is_major
                & (vol_vals > 15000)
                & (obs_car > 0)
                & (obs_car / vol_vals.clip(lower=1) < 0.25)
            )
            n_extreme = int(extreme_low.sum())
            if n_extreme > 0:
                joined.loc[extreme_low, "_excluded"] = True
                logger.info(
                    f"  Matching: excluded {n_extreme} major-road matches "
                    f"with obs/model ratio < 0.25"
                )

    # Corridor aggregation: sum volumes from parallel links (divided highways)
    if aggregate_corridor:
        joined = _aggregate_corridor_volumes(joined, links_sel, buffer_m)

    # Post-aggregation exclusion: compare observed against the final
    # corridor volume (which sums both carriageways for one-way links).
    # This catches cases like a 9k service road segment matched to a
    # 62k trunk corridor — the pre-aggregation check only sees ~31k
    # per direction and misses the mismatch.
    corr_col = "_corridor_volume" if "_corridor_volume" in joined.columns else None
    if corr_col and "link_type" in joined.columns:
        _MAJOR_POST = {"trunk", "trunk_link", "motorway", "motorway_link"}
        is_major_p = joined["link_type"].astype(str).isin(_MAJOR_POST)
        obs_car_p = pd.to_numeric(
            joined["observed_car"] if "observed_car" in joined.columns else 0,
            errors="coerce",
        ).fillna(0)
        corr_vals = pd.to_numeric(joined[corr_col], errors="coerce").fillna(0)
        extreme_corr = (
            is_major_p
            & ~joined["_excluded"]
            & (corr_vals > 15000)
            & (obs_car_p > 0)
            & (obs_car_p / corr_vals.clip(lower=1) < 0.25)
        )
        n_ext = int(extreme_corr.sum())
        if n_ext > 0:
            joined.loc[extreme_corr, "_excluded"] = True
            logger.info(
                f"  Matching: excluded {n_ext} major-road matches "
                f"with obs/corridor ratio < 0.25"
            )

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
    diag_cols = ["objectid", "link_id", "link_type", "name", "osm_ref"]
    for c in [obs_col, "observed_car", "observed_truck", "observed_total",
              "_dist", "_bearing_diff", "_match_quality",
              "_corridor_volume", "_corridor_n_links", "_matched", "_excluded"]:
        if c not in diag_cols:
            diag_cols.append(c)
    if model_col and model_col not in diag_cols:
        diag_cols.append(model_col)

    available = [c for c in diag_cols if c in matched.columns]
    diag = matched[available].copy()

    m_arr = diag[model_col].values.astype(float) if model_col and model_col in diag.columns else np.zeros(len(diag))
    o_arr = diag[obs_col].values.astype(float) if obs_col in diag.columns else np.zeros(len(diag))
    diag["GEH"] = compute_geh(m_arr, o_arr)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "matching_diagnostics.csv"
    diag.to_csv(out_path, index=False)
    logger.info(f"  Matching diagnostics: {out_path} ({len(diag)} rows)")


# ---------------------------------------------------------------------------
# Statistics: GEH, R², RMSE
# ---------------------------------------------------------------------------

from sim._metrics import compute_geh  # noqa: E402 — re-exported for backward compat


def compute_stats(
    modeled: np.ndarray,
    observed: np.ndarray,
    *,
    daily_capacity_factor: float = 1.0,
) -> Dict[str, Any]:
    """Compute link-level fit statistics.

    When *daily_capacity_factor* > 1 the model operates on daily aggregates;
    an adjusted GEH threshold (``5 * sqrt(K)``) is used alongside the
    standard hourly GEH<5 so that daily-model convergence criteria are
    meaningful (per FHWA/DMRB, GEH<5 targets apply to hourly flows).
    """
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

    # OLS regression slope / intercept  (model = slope * observed + intercept)
    if n >= 2:
        coeffs = np.polyfit(c, m, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
    else:
        slope = float("nan")
        intercept = float("nan")

    # MAPE: mean(|M-O|/O) * 100 — only for positive observed
    mape = float(np.mean(np.abs(m - c) / c) * 100.0)

    # Overall volume bias
    sum_m, sum_c = float(m.sum()), float(c.sum())
    bias_pct = (sum_m - sum_c) / sum_c * 100.0 if sum_c > 0 else float("nan")

    # Daily-adjusted GEH threshold: GEH scales as ~sqrt(K) on daily
    # volumes relative to hourly, so the "GEH<5" criterion becomes
    # "GEH < 5*sqrt(K)" for an equivalent daily acceptance band.
    k = max(daily_capacity_factor, 1.0)
    daily_geh_thr = 5.0 * math.sqrt(k)
    daily_geh_lt_adj_pct = float(np.nanmean(geh < daily_geh_thr) * 100.0)

    return {
        "n": int(n),
        "r2": round(r2, 4) if np.isfinite(r2) else None,
        "slope": round(slope, 4) if np.isfinite(slope) else None,
        "intercept": round(intercept, 1) if np.isfinite(intercept) else None,
        "rmse": round(rmse, 1),
        "pct_rmse": round(pct_rmse, 1) if np.isfinite(pct_rmse) else None,
        "mape_pct": round(mape, 1) if np.isfinite(mape) else None,
        "bias_pct": round(bias_pct, 2) if np.isfinite(bias_pct) else None,
        "geh_mean": round(float(np.nanmean(geh)), 2),
        "geh_median": round(float(np.nanmedian(geh)), 2),
        "geh_lt5_pct": round(float(np.nanmean(geh < 5) * 100), 1),
        "geh_lt10_pct": round(float(np.nanmean(geh < 10) * 100), 1),
        "daily_geh_threshold": round(daily_geh_thr, 1),
        "daily_geh_lt_adj_pct": round(daily_geh_lt_adj_pct, 1),
        "sum_modeled": round(sum_m, 0),
        "sum_observed": round(sum_c, 0),
    }


def _coarse_road_class(link_type: object) -> str:
    """Map OSM link_type to a small set of buckets for bias / MAE reporting."""
    s = str(link_type).lower()
    for rc in ("motorway", "trunk", "primary", "secondary", "tertiary"):
        if rc in s:
            return rc
    return "other"


def compute_extended_link_metrics(
    matched: pd.DataFrame,
    model_col: str,
    obs_col: str,
) -> Dict[str, Any]:
    """Extra fit diagnostics for metric validity experiments (wMAPE, class MAE, Spearman).

    *matched* should be rows with positive observed counts and finite modeled volumes.
    """
    if model_col not in matched.columns or obs_col not in matched.columns:
        return {}

    sub = matched[[model_col, obs_col]].dropna()
    sub = sub[(sub[obs_col] > 0) & np.isfinite(sub[model_col])].copy()
    if sub.empty:
        return {}

    m = sub[model_col].astype(float).values
    o = sub[obs_col].astype(float).values

    wmape = float(np.sum(np.abs(m - o)) / max(np.sum(o), 1e-9) * 100.0)
    denom_smape = np.abs(m) + np.abs(o)
    mask = denom_smape > 0
    smape = float(
        np.mean(2.0 * np.abs(m[mask] - o[mask]) / denom_smape[mask]) * 100.0
    ) if mask.any() else float("nan")

    srs = pd.Series(m)
    srs_o = pd.Series(o)
    rho = srs.corr(srs_o, method="spearman")
    spearman_rho = round(float(rho), 4) if rho is not None and np.isfinite(rho) else None

    sum_m, sum_o = float(m.sum()), float(o.sum())
    sum_ratio = round(sum_m / max(sum_o, 1e-9), 4) if sum_o > 0 else None

    mae_by_class: Dict[str, float] = {}
    bias_pct_by_class: Dict[str, float] = {}
    n_by_class: Dict[str, int] = {}

    if "link_type" in matched.columns:
        lt_sub = matched.loc[sub.index]
        coarse = lt_sub["link_type"].map(_coarse_road_class)
        for rc in coarse.unique():
            idx = coarse == rc
            if not idx.any():
                continue
            take = idx.to_numpy()
            mc = m[take]
            oc = o[take]
            mae_by_class[str(rc)] = round(float(np.mean(np.abs(mc - oc))), 1)
            bias_pct_by_class[str(rc)] = round(
                float(np.mean((mc - oc) / np.maximum(oc, 1e-9)) * 100.0), 2
            )
            n_by_class[str(rc)] = int(idx.sum())

    biases = list(bias_pct_by_class.values()) if bias_pct_by_class else []
    class_bias_max_abs = round(float(max(abs(b) for b in biases)), 2) if biases else None

    return {
        "wmape_pct": round(wmape, 2),
        "smape_pct": round(smape, 2) if np.isfinite(smape) else None,
        "spearman_rho": spearman_rho,
        "sum_ratio": sum_ratio,
        "mae_by_class": mae_by_class,
        "bias_pct_by_class": bias_pct_by_class,
        "n_by_class": n_by_class,
        "class_bias_max_abs_pct": class_bias_max_abs,
    }


def compute_class_volume_breakdown(
    vol_df: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """Per-road-class volume breakdown split by traffic class (local vs through).

    Joins assignment results with network link_type and groups total assigned
    volume by coarse road class.  When multi-class columns are present
    (e.g. ``local_tot``, ``through_tot``), reports each class separately so
    callers can see how much local vs through traffic uses motorway vs trunk.
    """
    if vol_df.empty:
        return {}

    merged = vol_df.copy()
    if "link_type" not in merged.columns and "link_id" in merged.columns:
        lt_map = links_gdf.set_index("link_id")["link_type"] if "link_type" in links_gdf.columns else None
        if lt_map is not None:
            merged["link_type"] = merged["link_id"].map(lt_map)

    if "link_type" not in merged.columns:
        return {}

    merged["_road_class"] = merged["link_type"].map(_coarse_road_class)

    tot_cols = [c for c in merged.columns if c.endswith("_tot")]
    if not tot_cols:
        return {}

    breakdown: Dict[str, Any] = {}
    for col in tot_cols:
        grp = merged.groupby("_road_class")[col].agg(["sum", "count"])
        breakdown[col] = {
            rc: {"total_volume": round(float(row["sum"]), 0), "n_links": int(row["count"])}
            for rc, row in grp.iterrows()
        }

    total_col = tot_cols[0]
    if len(tot_cols) > 1:
        for rc in ("motorway", "trunk"):
            rc_mask = merged["_road_class"] == rc
            if not rc_mask.any():
                continue
            parts = {col: round(float(merged.loc[rc_mask, col].sum()), 0) for col in tot_cols}
            total = sum(parts.values())
            shares = {col: round(v / max(total, 1), 3) for col, v in parts.items()}
            breakdown.setdefault("_class_shares", {})[rc] = {
                "volumes": parts, "shares": shares,
            }

    return breakdown




# ---------------------------------------------------------------------------
# Aggregate CSD helpers (for validation)
# ---------------------------------------------------------------------------

def _classify_csd_road(sil: str) -> str:
    s = str(sil).strip().upper()
    if s.startswith("D"):
        return "motorway"
    try:
        num = int(s.replace("M", ""))
        if num < 100:
            return "trunk"
        if num < 1000:
            return "secondary"
        return "tertiary"
    except ValueError:
        return "trunk" if "M" in s else "other"


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
# CSD split for calibration / validation
# ---------------------------------------------------------------------------

def split_csd_for_calibration(
    csd: pd.DataFrame,
    strategy: str = "alternating",
    calib_share: float = 0.65,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split CSD sections into calibration and validation subsets.

    Both subsets always contain **all road classes** so that the
    calibration can correct inter-class volume imbalances.

    Returns ``(calib_df, valid_df)``.

    Strategies
    ----------
    alternating  *(default)*
        Within each road number (``sil``), sections are sorted by their
        original order and assigned to calibration / validation in a
        round-robin fashion.  Every road class appears in both subsets,
        and geographically adjacent sections end up in different sets.
    stratified
        Stratified random split preserving ``road_class`` proportions
        (``calib_share`` controls the calibration fraction, default 65 %).
    spatial
        Sections whose ``nazev_mesta`` is non-empty (urban) are used for
        calibration; sections without a city name (rural/inter-urban)
        become the validation set.
    """
    if not (0.0 < calib_share < 1.0):
        raise ValueError(
            f"calib_share must be in (0, 1), got {calib_share}"
        )

    csd = csd.copy()
    csd["sil"] = csd["sil"].astype(str)
    if "road_class" not in csd.columns:
        csd["road_class"] = csd["sil"].apply(_classify_csd_road)

    if strategy == "alternating":
        period = max(2, round(1 / (1 - calib_share)))
        is_calib = pd.Series(False, index=csd.index)
        for _, grp in csd.groupby("sil"):
            idxs = grp.index.tolist()
            if len(idxs) == 1:
                is_calib.at[idxs[0]] = True
                continue
            for i, idx in enumerate(idxs):
                is_calib.at[idx] = (i % period) != 0
            is_calib.at[idxs[0]] = True
            is_calib.at[idxs[-1]] = False
        calib_df = csd[is_calib].copy()
        valid_df = csd[~is_calib].copy()

    elif strategy == "stratified":
        from sklearn.model_selection import train_test_split

        test_size = max(0.05, min(0.95, 1.0 - calib_share))
        rc_counts = csd["road_class"].value_counts()
        n_classes = len(rc_counts)
        n_test = max(1, round(len(csd) * test_size))
        can_stratify = (
            (rc_counts >= 2).all()
            and len(csd) >= 4
            and n_test >= n_classes
        )
        calib_df, valid_df = train_test_split(
            csd,
            test_size=test_size,
            random_state=random_seed,
            stratify=csd["road_class"] if can_stratify else None,
        )
        calib_df = calib_df.copy()
        valid_df = valid_df.copy()

    elif strategy == "spatial":
        has_city = csd["nazev_mesta"].fillna("").str.strip().astype(bool)
        calib_df = csd[has_city].copy()
        valid_df = csd[~has_city].copy()
        if calib_df.empty or valid_df.empty:
            logger.warning(
                "Spatial CSD split produced an empty %s subset — "
                "all sections are %s. Falling back to alternating.",
                "calibration" if calib_df.empty else "validation",
                "rural" if calib_df.empty else "urban",
            )
            return split_csd_for_calibration(
                csd, strategy="alternating",
                calib_share=calib_share, random_seed=random_seed,
            )

    else:
        raise ValueError(f"Unknown CSD split strategy: {strategy!r}")

    logger.info(
        "CSD split (%s): calibration=%d sections (%d roads), "
        "validation=%d sections (%d roads)",
        strategy,
        len(calib_df), calib_df["sil"].nunique() if "sil" in calib_df.columns else 0,
        len(valid_df), valid_df["sil"].nunique() if "sil" in valid_df.columns else 0,
    )
    return calib_df, valid_df


def load_csd_as_link_counts(
    csd: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Convert CSD sections into a pentlogram-compatible GeoDataFrame.

    For each CSD road (``sil``), finds network links whose ``osm_ref``
    matches, then assigns the CSD AADT as observed counts on those links.
    The returned GeoDataFrame has the same columns the calibration loop
    expects from ``load_pentlogram``: ``observed_car``, ``observed_total``,
    ``observed_motor_total``, ``observed_truck``, and link ``geometry``.
    """
    if "sil" not in csd.columns or "osm_ref" not in links_gdf.columns:
        logger.warning("CSD→link_counts: missing 'sil' or 'osm_ref' column")
        return gpd.GeoDataFrame()

    csd = csd.copy()
    csd["sil"] = csd["sil"].astype(str)
    for col in ("o", "sv", "tv"):
        if col in csd.columns:
            csd[col] = pd.to_numeric(csd[col], errors="coerce").fillna(0)
    if "road_class" not in csd.columns:
        csd["road_class"] = csd["sil"].apply(_classify_csd_road)

    csd_sil = csd["sil"].str.strip()
    raw_refs = links_gdf["osm_ref"].fillna("").astype(str).str.strip()
    ref_components = raw_refs.str.split(";").explode().str.strip()
    ref_components = ref_components[ref_components != ""]

    rows: list[dict] = []
    for road in csd_sil.unique():
        csd_sub = csd[csd_sil == road]
        if csd_sub.empty:
            continue

        matching_indices = ref_components.index[ref_components == road]
        if len(matching_indices) == 0:
            continue

        csd_mean_o = float(csd_sub["o"].mean())
        csd_mean_sv = float(csd_sub["sv"].mean())
        csd_mean_tv = float(csd_sub["tv"].mean())
        road_class = csd_sub["road_class"].iloc[0]

        matched_links = links_gdf.loc[matching_indices.unique()]
        if "link_type" in matched_links.columns:
            lt = matched_links["link_type"].astype(str)
            compatible = _CSD_COMPATIBLE_LINK_TYPES.get(road_class)
            if compatible:
                matched_links = matched_links[lt.reindex(matched_links.index).isin(compatible)]
            matched_links = matched_links[
                ~lt.reindex(matched_links.index).isin(_NON_CAR_LINK_TYPES)
            ]
        if matched_links.empty:
            continue

        rows.append({
            "link_id": matched_links.iloc[0].get("link_id", matched_links.index[0]),
            "geometry": matched_links.iloc[0].geometry,
            "observed_car": csd_mean_o,
            "observed_motor_total": csd_mean_sv,
            "observed_total": csd_mean_sv,
            "observed_truck": max(0.0, csd_mean_sv - csd_mean_o),
            "csd_road": road,
            "csd_road_class": road_class,
            "_n_matched_links": len(matched_links),
        })

    if not rows:
        logger.warning("CSD→link_counts: no CSD roads matched any network links")
        return gpd.GeoDataFrame()

    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=links_gdf.crs)
    result = result[result["observed_total"] > 0].copy()
    logger.info(
        "CSD→link_counts: %d road-level observations from %d CSD roads",
        len(result), result["csd_road"].nunique(),
    )
    return result


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
) -> Dict[str, Dict[str, float]]:
    """Compare modeled average speeds per road class against reference speeds."""
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
    """Weighted composite objective for supply calibration (lower is better).

    Includes R², slope, and %RMSE terms so that daily-model supply tuning
    optimises for regression fit rather than hourly-specific GEH.
    """
    w = weights
    geh_term = (100.0 - float(count_stats.get("geh_lt5_pct", 0))) * w.get("geh", 1.0)

    r2 = float(count_stats.get("r2") or 0.0)
    r2_term = (1.0 - r2) * 100.0 * w.get("r2", 1.0)

    slope = float(count_stats.get("slope") or 1.0)
    slope_term = abs(slope - 1.0) * 100.0 * w.get("slope", 1.0)

    prmse = float(count_stats.get("pct_rmse") or 0.0)
    rmse_term = prmse * w.get("pct_rmse", 0.5)

    sl_term = 0.0
    for sr in screenline_results.values():
        ratio = sr.get("ratio", 1.0)
        if ratio is not None:
            sl_term += abs(ratio - 1.0)
    sl_term *= w.get("screenline", 2.0)

    jt_fail = sum(1 for r in jt_results if not r.get("pass", True))
    jt_term = jt_fail / max(len(jt_results), 1) * 100 * w.get("jt", 1.0) if jt_results else 0.0

    return geh_term + r2_term + slope_term + rmse_term + sl_term + jt_term


def run_supply_tuning(config_path: str | Path = "config/sim.yaml") -> None:
    """Outer-loop supply parameter tuning via coordinate descent."""
    cfg = load_config(config_path)
    calib_cfg = cfg.get("calibration") or {}
    tuning_cfg = calib_cfg.get("supply_tuning") or {}

    if not tuning_cfg.get("enabled", False):
        logger.info("Supply tuning disabled in config.")
        return

    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)

    road_classes = tuning_cfg.get("road_classes", ["motorway", "trunk", "primary", "secondary", "tertiary"])
    speed_range = tuning_cfg.get("speed_factor_range", [0.8, 0.9, 1.0, 1.1, 1.2])
    cap_range = tuning_cfg.get("capacity_factor_range", [0.8, 0.9, 1.0, 1.1, 1.2])
    inner_max_iter = int(tuning_cfg.get("inner_max_iterations", 3))
    obj_weights = {"geh": 1.0, "screenline": 2.0, "jt": 1.0}

    logger.info("=== SUPPLY PARAMETER TUNING ===")
    logger.info(f"  Road classes: {road_classes}")
    logger.info(f"  Speed range: {speed_range}")
    logger.info(f"  Capacity range: {cap_range}")

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
        except Exception:
            logger.exception("Calibration failed during supply tuning evaluation")
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
                    logger.debug(
                        "Journey time validation from skim matrix failed",
                        exc_info=True,
                    )

        return compute_objective(count_stats, sl_results, jt_results, obj_weights)

    best_obj = _evaluate(best_params)
    logger.info(f"\n  Baseline objective: {best_obj:.2f}")

    for rc in road_classes:
        logger.info(f"\n  --- Tuning {rc} ---")

        # Speed factor
        best_sf = best_params.speed_factors.get(rc, 1.0)
        for sf in speed_range:
            if sf == best_sf:
                continue
            trial = SupplyParams(
                speed_factors={**best_params.speed_factors, rc: sf},
                capacity_factors=dict(best_params.capacity_factors),
            )
            obj = _evaluate(trial)
            logger.info(f"    speed_factor[{rc}]={sf} ... obj={obj:.2f}")
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
            obj = _evaluate(trial)
            logger.info(f"    capacity_factor[{rc}]={cf} ... obj={obj:.2f}")
            if obj < best_obj:
                best_obj = obj
                best_params = trial
                best_cf = cf
        best_params.capacity_factors[rc] = best_cf

    # Apply best params and run final full calibration
    calib_cfg["max_iterations"] = orig_max_iter
    logger.info(
        f"\n  Best params: speed={best_params.speed_factors}, "
        f"capacity={best_params.capacity_factors}, obj={best_obj:.2f}"
    )
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
    logger.info(f"\n  Supply tuning report: {report_path}")
    logger.info("  Running final calibration with best parameters ...")
    run_calibration(config_path)


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def _check_final_convergence(
    history: List[Dict[str, Any]],
    model_time_period: str,
    geh_target: float,
    daily_conv: Dict[str, Any],
) -> bool:
    """Check whether the final iteration satisfies convergence criteria.

    For daily models the screenline criterion is included alongside
    R², slope, %RMSE, and bias — matching the iteration-loop guards.
    """
    if not history:
        return False
    final = history[-1]
    if model_time_period == "daily":
        r2 = float(final.get("r2") or 0.0)
        slp = float(final.get("slope") or 0.0)
        prmse = float(final.get("pct_rmse") or 999.0)
        bias = abs(float(final.get("bias_pct") or 999.0))
        sr = daily_conv.get("slope_range", [0.85, 1.15])
        sl_max_dev = float(final.get("max_screenline_pct_dev") or 999.0)
        sl_target = float(daily_conv.get("screenline_max_pct_deviation", 15.0))
        return (
            r2 >= float(daily_conv.get("r2_target", 0.80))
            and float(sr[0]) <= slp <= float(sr[1])
            and prmse <= float(daily_conv.get("pct_rmse_max", 35.0))
            and bias <= float(daily_conv.get("bias_abs_max_pct", 15.0))
            and sl_max_dev <= sl_target
        )
    return float(final.get("geh_lt5_pct", 0)) >= geh_target


# ---------------------------------------------------------------------------
# ODME – Spiess gradient OD matrix estimation
# ---------------------------------------------------------------------------

def _sr_val(sr: Any, key: str) -> float:
    """Extract a numeric value from a screenline result (dict or dataclass)."""
    if isinstance(sr, dict):
        return float(sr.get(key, 0) or 0)
    return float(getattr(sr, key, 0) or 0)


def _compute_count_weights(
    observed: np.ndarray,
    method: str = "inverse_sqrt",
) -> np.ndarray:
    """Weight vector for count-post observations.

    ``inverse_sqrt`` (default): w_a = 1 / sqrt(max(obs_a, 100)).
    Balances high- and low-volume count posts so the objective function
    is not dominated by motorway links (Aimsun OD-adj. TN, 2019).
    """
    obs = np.maximum(np.asarray(observed, dtype=np.float64), 100.0)
    if method == "inverse_sqrt":
        return 1.0 / np.sqrt(obs)
    if method == "inverse":
        return 1.0 / obs
    # uniform — plain least squares
    return np.ones_like(obs)


def _odme_objective(
    modeled: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted sum-of-squares objective  Z = sum_a w_a*(v_a - v_a^obs)^2."""
    residuals = np.asarray(modeled, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    return float(np.sum(weights * residuals ** 2))


class _CalibrationContext:
    """Shared setup / teardown for calibration loops.

    Encapsulates config loading, matrix backup/restore, observed data,
    screenline resolution, project/graph setup, and seed elasticity bounds
    so that ``run_calibration`` and ``run_odme_calibration`` can reuse
    the same initialisation logic.
    """

    def __init__(self, config_path: str | Path = "config/sim.yaml") -> None:
        from sim.screenlines import load_screenlines, resolve_screenline_links

        self.config_path = config_path
        cfg = load_config(config_path)
        self.cfg = cfg
        self.project_dir = Path(cfg["project_path"])

        demand_cfg = cfg.get("demand") or {}
        self.calib_cfg = calib_cfg = cfg.get("calibration") or {}
        self.matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
        self.output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
        _ensure_dir(self.output_dir)

        # Assignment parameters
        self.algorithm = str(calib_cfg.get("algorithm", "bfw"))
        self.max_iter_assign = int(calib_cfg.get("max_iter", 100))
        self.rgap = float(calib_cfg.get("rgap_target", 0.001))
        self.core_name = str(calib_cfg.get("core_name", "wd_daily"))

        assign_cfg = cfg.get("assignment") or {}
        gc_cfg = assign_cfg.get("generalized_cost") or {}
        gc_enabled = bool(gc_cfg.get("enabled", False))
        self.gc_field: Optional[str] = (
            str(gc_cfg["fixed_cost_field"])
            if gc_enabled and "fixed_cost_field" in gc_cfg else None
        )
        self.gc_mult: float = float(gc_cfg.get("fixed_cost_multiplier", 0.0)) if gc_enabled else 0.0
        self.gc_vot: float = float(gc_cfg.get("vot", 1.0))
        bpr_cfg = assign_cfg.get("bpr") or {}
        self.cfg_bpr: Optional[Dict[str, object]] = dict(bpr_cfg) if bpr_cfg else None
        mc_cfg = assign_cfg.get("multi_class") or {}
        self.cfg_multi: Optional[list] = (
            list(mc_cfg["classes"]) if mc_cfg.get("enabled") and "classes" in mc_cfg else None
        )
        self.daily_cap_factor = resolve_daily_cap_factor_default(bpr_cfg)
        self.cores = int(assign_cfg.get("cores", 0))

        # Matching parameters
        self.buffer_m = float(calib_cfg.get("match_buffer_m", 50.0))
        self.direction_aware = bool(calib_cfg.get("match_direction_aware", True))
        self.conflict_res = str(calib_cfg.get("match_conflict_resolution", "nearest"))
        self.agg_corridor = bool(calib_cfg.get("aggregate_corridor", True))
        self.count_target = str(calib_cfg.get("count_target", "total"))
        _COUNT_TARGET_COL = {
            "car_only": "observed_car",
            "motor_total": "observed_motor_total",
            "total": "observed_total",
        }
        self.obs_col = _COUNT_TARGET_COL.get(self.count_target, "observed_total")
        if self.count_target not in _COUNT_TARGET_COL:
            logger.warning(
                f"  WARNING: unknown count_target '{self.count_target}', "
                f"falling back to observed_total"
            )

        # Convergence
        conv_cfg = calib_cfg.get("convergence") or {}
        self.daily_conv = conv_cfg.get("daily") or {}

        # ODME / elasticity
        self.odme_cfg = calib_cfg.get("odme") or {}
        self.max_deviation = float(self.odme_cfg.get("max_deviation", 3.0))

        # Pre-flight
        fix_node_ids(self.project_dir)
        if not self.matrix_path.exists():
            raise FileNotFoundError(f"OD matrix not found: {self.matrix_path}")

        # Backup / restore seed matrix — versioned by content hash
        import hashlib

        def _file_hash(path: Path) -> str:
            h = hashlib.sha256()
            h.update(path.read_bytes())
            return h.hexdigest()[:16]

        backup = self.matrix_path.with_suffix(".aem.orig")
        current_hash = _file_hash(self.matrix_path)

        if backup.exists():
            backup_hash = _file_hash(backup)
            if backup_hash != current_hash:
                logger.warning(
                    "Seed matrix changed since last backup "
                    "(backup hash=%s, current hash=%s). "
                    "Updating .aem.orig to new seed.",
                    backup_hash, current_hash,
                )
                shutil.copy2(self.matrix_path, backup)
        else:
            shutil.copy2(self.matrix_path, backup)
            logger.info("Created seed matrix backup: %s", backup)

        reset_matrix = bool(calib_cfg.get("reset_matrix_before_run", True))
        if reset_matrix and backup.exists():
            shutil.copy2(backup, self.matrix_path)
            logger.info("  Restored OD matrix from .aem.orig")

        # Observed counts + network links
        self.links_gdf = _load_network_links(self.project_dir)
        self.count_source = str(calib_cfg.get("count_source", "pentlogram"))

        if self.count_source == "csd_split":
            split_cfg = calib_cfg.get("csd_split") or {}
            csd_full = load_csd(cfg)
            calib_csd, _valid_csd = split_csd_for_calibration(
                csd_full,
                strategy=str(split_cfg.get("strategy", "alternating")),
                calib_share=float(split_cfg.get("calib_share", 0.65)),
                random_seed=int(split_cfg.get("random_seed", 42)),
            )
            self.pent = load_csd_as_link_counts(calib_csd, self.links_gdf)
            if self.pent.empty:
                raise RuntimeError(
                    "CSD split produced no link-level observations for calibration. "
                    "Check that osm_ref values match CSD road numbers."
                )
            logger.info(f"  CSD-split calibration: {len(self.pent)} link observations")
        else:
            self.pent = load_pentlogram(cfg)
            validate_geometries_or_fail(
                self.pent, name="pentlogram", expected_epsg=get_metric_epsg(self.cfg),
            )
            logger.info(f"  Pentlogram: {len(self.pent)} observed segments")

        # Screenlines
        sl_path = str(calib_cfg.get("screenlines_path", "config/screenlines.yaml"))
        self.screenlines = load_screenlines(sl_path)
        self.sl_query: Optional[Dict[str, list]] = None
        if self.screenlines:
            self.sl_query = {}
            for sl in self.screenlines:
                resolved = resolve_screenline_links(sl, self.links_gdf)
                if resolved:
                    sl.links = resolved
                    self.sl_query[sl.name] = [(lid, d) for lid, d in resolved]
            logger.info(
                f"  Screenlines: {len(self.screenlines)} defined, "
                f"{len(self.sl_query)} with links"
            )

        # Open matrix (stays open across iterations)
        self.mat = AequilibraeMatrix()
        self.mat.load(str(self.matrix_path))
        self.mat.computational_view([self.core_name])

        # Seed matrix for elasticity bounds
        seed = self.mat.matrix[self.core_name][:, :].copy().astype(np.float64)
        self.seed_lower = seed / self.max_deviation
        self.seed_upper = seed * self.max_deviation
        self.seed_lower[seed <= 0] = 0.0
        self.seed_upper[seed <= 0] = 0.0

        # Swap closures for the calibration period before opening the project
        bc_cfg = cfg.get("baseline_closures") or {}
        calib_period = bc_cfg.get("calibration_period")
        if bc_cfg.get("enabled", False) and calib_period:
            from sim.network_normalization import swap_db_closures
            swap_db_closures(config_path, measurement_period=calib_period)
            logger.info("  Closures swapped to calibration period: %s", calib_period)

        # AequilibraE project + graph (reused across iterations)
        self.project = Project()
        self.project.open(str(self.project_dir))
        self.cached_graph = build_graph(self.project, self.mat, bpr_parameters=self.cfg_bpr)

        # Gateway calibration
        gw_cal_cfg = calib_cfg.get("gateway_calibration") or {}
        self.gw_cal_enabled = bool(gw_cal_cfg.get("enabled", False))
        self.gw_cal_damping = float(gw_cal_cfg.get("damping", 0.08))
        self.gw_cal_min_factor = float(gw_cal_cfg.get("min_factor", 0.90))
        self.gw_cal_max_factor = float(gw_cal_cfg.get("max_factor", 1.10))
        self.gw_zone_map: Dict[str, np.ndarray] = {}
        self.gw_observed: Dict[str, float] = {}

        if self.gw_cal_enabled:
            try:
                self.gw_zone_map = _load_gateway_zone_map(cfg, self.mat.index[:])
                self.gw_observed = _load_gateway_observed(cfg)
                if self.gw_zone_map and self.gw_observed:
                    active = set(self.gw_zone_map) & set(self.gw_observed)
                    logger.info(
                        f"  Gateway calibration: {len(active)} gateways with observed data "
                        f"({', '.join(sorted(active))})"
                    )
                else:
                    logger.info(
                        "  Gateway calibration: no matching zone/observed data — disabled"
                    )
                    self.gw_cal_enabled = False
            except Exception:
                logger.exception("Gateway calibration init failed")
                self.gw_cal_enabled = False

        # Tracking state (encapsulated in CalibrationRun for FSM audit)
        self.run = CalibrationRun()
        self.best_Z: float = float("inf")
        self.best_demand: Optional[np.ndarray] = None
        self.best_iteration: int = 0
        self.history: List[Dict[str, Any]] = []

    # -- Shared helpers for the iteration body --

    def run_assignment(self, *, save_skims: bool = False) -> tuple:
        """Execute one equilibrium assignment pass."""
        return execute_assignment(
            self.project, self.mat,
            algorithm=self.algorithm,
            max_iter=self.max_iter_assign,
            rgap_target=self.rgap,
            save_skims=save_skims,
            select_links=self.sl_query,
            bpr_parameters=self.cfg_bpr,
            fixed_cost_field=self.gc_field,
            fixed_cost_multiplier=self.gc_mult,
            vot=self.gc_vot,
            multi_class=self.cfg_multi,
            graph=self.cached_graph,
            cores=self.cores,
        )

    def restore_best_and_close(self) -> None:
        """Restore best-Z demand matrix, save, and close resources."""
        try:
            if self.best_demand is not None:
                self.mat.matrix[self.core_name][:, :] = self.best_demand
                self.mat.save()
                logger.info(
                    f"  Restored best matrix from iteration {self.best_iteration} "
                    f"(Z={self.best_Z:,.1f})"
                )
            self.mat.close()
            self.project.close()
        except Exception:
            logger.exception("restore_best_and_close failed")

    def finalize_best_state(self) -> Optional[pd.DataFrame]:
        """Run a final assignment on the restored best-demand matrix.

        Must be called *after* ``restore_best_and_close()`` so that the
        on-disk matrix already contains the best demand.  Returns the
        vol_df from this final assignment so the caller can persist
        artifacts that are consistent with the best OD matrix.
        """
        if self.best_demand is None:
            return None

        mat = AequilibraeMatrix()
        mat.load(str(self.matrix_path))
        mat.computational_view([self.core_name])

        project = Project()
        project.open(str(self.project_dir))
        try:
            graph = build_graph(project, mat, bpr_parameters=self.cfg_bpr)
            vol_df, _skims, _sl = execute_assignment(
                project, mat,
                algorithm=self.algorithm,
                max_iter=self.max_iter_assign,
                rgap_target=self.rgap,
                bpr_parameters=self.cfg_bpr,
                fixed_cost_field=self.gc_field,
                fixed_cost_multiplier=self.gc_mult,
                vot=self.gc_vot,
                multi_class=self.cfg_multi,
                graph=graph,
                cores=self.cores,
            )
            logger.info(
                f"  Final assignment on best-state demand completed "
                f"(iteration {self.best_iteration})"
            )
            return vol_df
        finally:
            mat.close()
            project.close()

    def save_results(self, vol_df: Optional[pd.DataFrame] = None) -> None:
        """Persist assignment results parquet and report path."""
        if self.history and vol_df is not None:
            results_path = self.output_dir / "assignment_results.parquet"
            vol_df.to_parquet(str(results_path), index=False)
            logger.info(f"  Saved: {results_path}")


def run_odme_calibration(config_path: str | Path = "config/sim.yaml") -> None:
    """Spiess-style gradient ODME calibration.

    Bi-level optimization:
      Lower level – AequilibraE equilibrium assignment (BFW)
      Upper level – minimize weighted squared residuals between modeled
                    and observed link volumes by adjusting the OD matrix
                    via relative-gradient descent with optimal step length.

    The gradient is computed from select-link OD proportions (screenlines).
    Remaining aggregate bias (from all matched count posts) is corrected
    with a damped global scalar after the gradient step.
    """
    ctx = _CalibrationContext(config_path)
    calib_cfg = ctx.calib_cfg
    odme_cfg = ctx.odme_cfg

    from sim.screenlines import evaluate_all_screenlines

    # Aliases for loop body backward compat
    mat = ctx.mat
    project = ctx.project
    cached_graph = ctx.cached_graph
    links_gdf = ctx.links_gdf
    pent = ctx.pent
    sl_query = ctx.sl_query
    screenlines = ctx.screenlines
    output_dir = ctx.output_dir
    core_name = ctx.core_name
    algorithm = ctx.algorithm
    max_iter_assign = ctx.max_iter_assign
    rgap = ctx.rgap
    gc_field = ctx.gc_field
    gc_mult = ctx.gc_mult
    gc_vot = ctx.gc_vot
    cfg_bpr = ctx.cfg_bpr
    cfg_multi = ctx.cfg_multi
    daily_cap_factor = ctx.daily_cap_factor
    cores = ctx.cores
    buffer_m = ctx.buffer_m
    direction_aware = ctx.direction_aware
    conflict_res = ctx.conflict_res
    agg_corridor = ctx.agg_corridor
    obs_col = ctx.obs_col
    count_target = ctx.count_target
    daily_conv = ctx.daily_conv
    max_deviation = ctx.max_deviation
    seed_lower = ctx.seed_lower
    seed_upper = ctx.seed_upper

    max_outer = int(odme_cfg.get("max_outer_iterations", 20))
    gd_inner = int(odme_cfg.get("gradient_descent_iterations", 3))
    weight_method = str(odme_cfg.get("weight_function", "inverse_sqrt"))
    conv_tol = float(odme_cfg.get("convergence_tol", 0.01))
    global_residual_damping = float(odme_cfg.get("global_residual_damping", 0.3))

    logger.info("=== ODME GRADIENT CALIBRATION (Spiess method) ===")
    logger.info(f"  max_outer={max_outer}, gd_inner={gd_inner}, max_deviation={ctx.max_deviation}")
    logger.info(f"  weight_function={weight_method}, convergence_tol={conv_tol}")
    logger.info(f"  global_residual_damping={global_residual_damping}")

    history = ctx.history
    prev_Z = float("inf")
    best_Z = ctx.best_Z
    best_demand = ctx.best_demand
    best_iteration = ctx.best_iteration
    effective_global_damping = global_residual_damping
    sl_results: Dict[str, Any] = {}

    try:
        for outer_it in range(1, max_outer + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"  ODME Outer Iteration {outer_it}/{max_outer}")
            logger.info(f"{'='*60}")

            total_demand = float(mat.matrix_view.sum())
            logger.info(f"  Demand total: {total_demand:,.0f}")

            # --- 1) Equilibrium assignment with select-link ---
            save_skims_now = bool(calib_cfg.get("save_skims", False)) and outer_it == 1
            vol_df, skims, sl_matrices = execute_assignment(
                project, mat,
                algorithm=algorithm,
                max_iter=max_iter_assign,
                rgap_target=rgap,
                save_skims=save_skims_now,
                select_links=sl_query,
                fixed_cost_field=gc_field,
                fixed_cost_multiplier=gc_mult,
                vot=gc_vot,
                bpr_parameters=cfg_bpr,
                multi_class=cfg_multi,
                graph=cached_graph,
                cores=cores,
            )
            if skims is not None and outer_it == 1:
                skim_path = output_dir / "skims.aem"
                try:
                    skims.export(str(skim_path))
                except Exception:
                    logger.debug("Skim export failed", exc_info=True)

            vol_col = _detect_volume_col(vol_df)
            class_tot_cols = [
                c for c in vol_df.columns
                if c.endswith("_tot") and c not in ("PCE_tot", "Preload_tot")
                and vol_df[c].sum() > 0
            ]
            if len(class_tot_cols) > 1:
                vol_df["total_vehicles_tot"] = vol_df[class_tot_cols].sum(axis=1)
                vol_col = "total_vehicles_tot"

            total_vol = float(vol_df[vol_col].sum()) if vol_col else 0.0
            logger.info(f"  Assigned volume: {total_vol:,.0f}  (col={vol_col})")

            # --- 2) Match ALL pentlogram counts to links ---
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
                vol_col=vol_col,
            )

            # Build persistent exclusion set on first iteration
            if outer_it == 1:
                if "_excluded" in matched.columns and "objectid" in matched.columns:
                    excl_ids = set(matched.loc[matched["_excluded"], "objectid"].dropna().astype(int))
                    if excl_ids:
                        n_before = len(pent)
                        pent = pent[~pent["objectid"].isin(excl_ids)].copy()
                        logger.info(
                            f"  Pre-filter: removed {n_before - len(pent)} excluded stations"
                        )
                _export_matching_diagnostics(
                    matched,
                    "_corridor_volume" if "_corridor_volume" in matched.columns else vol_col,
                    obs_col, output_dir,
                )

            compare_col = "_corridor_volume" if "_corridor_volume" in matched.columns else vol_col

            # --- 3) Extract valid count posts for objective function ---
            valid = matched.dropna(subset=[compare_col, obs_col])
            valid = valid[valid[obs_col] > 0].copy()
            if "_excluded" in valid.columns:
                valid = valid[~valid["_excluded"]].copy()

            if valid.empty:
                logger.warning("  WARNING: no valid matched counts — cannot compute gradient")
                continue

            mod_all = valid[compare_col].values.astype(np.float64)
            obs_all = valid[obs_col].values.astype(np.float64)
            w_all = _compute_count_weights(obs_all, method=weight_method)

            Z_current = _odme_objective(mod_all, obs_all, w_all)

            # Compute standard metrics for reporting
            stats = compute_stats(mod_all, obs_all, daily_capacity_factor=daily_cap_factor)
            geh5 = float(stats.get("geh_lt5_pct", 0))
            r2 = stats.get("r2")
            slope = stats.get("slope")
            pct_rmse = stats.get("pct_rmse")
            bias_pct = stats.get("bias_pct")

            # Track best state — restore at the end for guaranteed best output
            if Z_current < best_Z:
                best_Z = Z_current
                best_demand = mat.matrix[core_name][:, :].copy()
                best_iteration = outer_it

            # Adaptive damping: if Z increased, reduce global damping
            if outer_it > 1 and Z_current > prev_Z:
                effective_global_damping = max(effective_global_damping * 0.7, 0.05)
                logger.warning(
                    f"  WARNING: Z increased — reducing global_damping to {effective_global_damping:.3f}"
                )

            logger.info(f"  Z={Z_current:,.1f}  (prev={prev_Z:,.1f}  delta={Z_current - prev_Z:+,.1f})")
            logger.info(f"  R²={r2}  slope={slope}  %RMSE={pct_rmse}  bias={bias_pct}%")
            logger.info(f"  GEH<5: {geh5:.1f}%  n_counts={len(valid)}")

            # Evaluate screenlines for reporting
            sl_results: Dict[str, Any] = {}
            max_sl_pct_dev: float = 0.0
            if screenlines and vol_col:
                sl_res = evaluate_all_screenlines(
                    screenlines, vol_df, matched, vol_col, obs_col, links_gdf,
                )
                for sn, sr in sl_res.items():
                    sl_results[sn] = sr.to_dict()
                    if sr.observed_total > 0 and np.isfinite(sr.ratio):
                        dev = abs(sr.ratio - 1.0) * 100.0
                        max_sl_pct_dev = max(max_sl_pct_dev, dev)
                    logger.info(
                        f"  SL '{sn}': mod={sr.modeled_total:,.0f} "
                        f"obs={sr.observed_total:,.0f} ratio={sr.ratio:.2f} GEH={sr.geh:.1f}"
                    )

            # Record iteration history
            iter_record = {
                "iteration": outer_it,
                "demand_total": round(total_demand, 0),
                "assigned_total": round(total_vol, 0),
                "Z_objective": round(Z_current, 1),
                **stats,
                "max_screenline_pct_dev": round(max_sl_pct_dev, 1),
                "n_count_posts": len(valid),
            }
            history.append(iter_record)

            # --- 4) Convergence check ---
            if outer_it > 1:
                rel_change = abs(Z_current - prev_Z) / max(prev_Z, 1.0)
                if rel_change < conv_tol:
                    logger.info(f"  CONVERGED: |delta Z|/Z = {rel_change:.6f} < {conv_tol}")
                    break

            # Stall detection: if best Z hasn't improved in 4 outer iterations
            if outer_it - best_iteration >= 4:
                logger.info(f"  STALLED: no Z improvement since iteration {best_iteration}")
                break

            # Also check daily convergence criteria
            cur_r2 = float(stats.get("r2") or 0.0)
            cur_slope = float(stats.get("slope") or 0.0)
            cur_prmse = float(stats.get("pct_rmse") or 999.0)
            cur_bias = abs(float(stats.get("bias_pct") or 999.0))
            daily_r2_target = float(daily_conv.get("r2_target", 0.80))
            daily_slope_range = daily_conv.get("slope_range", [0.85, 1.15])
            daily_pct_rmse_max = float(daily_conv.get("pct_rmse_max", 35.0))
            daily_bias_max = float(daily_conv.get("bias_abs_max_pct", 15.0))
            daily_sl_max_dev = float(daily_conv.get("screenline_max_pct_deviation", 15.0))

            daily_ok = (
                cur_r2 >= daily_r2_target
                and float(daily_slope_range[0]) <= cur_slope <= float(daily_slope_range[1])
                and cur_prmse <= daily_pct_rmse_max
                and cur_bias <= daily_bias_max
                and (max_sl_pct_dev <= daily_sl_max_dev if sl_results else True)
            )
            if daily_ok:
                logger.info(f"  CONVERGED: all daily criteria met")
                break

            prev_Z = Z_current

            # --- 5) Spiess gradient step: per-screenline multiplicative update ---
            # The Spiess relative gradient update applied per screenline:
            #   g_i *= 1 + damping * (ratio_sl - 1) * p_i(sl)
            # where p_i(sl) = sl_od[i] / demand[i] is the fraction of OD pair
            # i that routes through screenline sl.  This multiplicative form
            # preserves zeros and moves OD cells proportionally to their
            # contribution to each screenline's flow.
            #
            # With N overlapping screenlines, damping = 1/sqrt(N) prevents
            # double-correction while remaining responsive.
            data = mat.matrix[core_name]
            demand = data[:, :].copy().astype(np.float64)
            n_sl_used = 0

            if sl_matrices:
                n_sl_active = sum(
                    1 for sn in sl_matrices
                    if _sr_val(sl_results.get(sn, {}), "observed_total") > 0
                    and _sr_val(sl_results.get(sn, {}), "modeled_total") > 0
                )
                sl_damping = 1.0 / max(np.sqrt(n_sl_active), 1.0)

                for gd_it in range(1, gd_inner + 1):
                    corrections_applied = []
                    for sl_name, sl_od_raw in sl_matrices.items():
                        obs_sl = _sr_val(sl_results.get(sl_name, {}), "observed_total")
                        mod_sl = _sr_val(sl_results.get(sl_name, {}), "modeled_total")
                        if obs_sl <= 0 or mod_sl <= 0:
                            continue
                        ratio = obs_sl / mod_sl
                        # Skip extreme ratios (connectivity issue, not demand)
                        if ratio > 5.0 or ratio < 0.2:
                            continue

                        sl_od = np.asarray(sl_od_raw, dtype=np.float64).reshape(demand.shape)
                        proportion = np.where(
                            demand > 0,
                            np.clip(sl_od / np.maximum(demand, 1e-9), 0.0, 1.0),
                            0.0,
                        )

                        # Multiplicative Spiess update
                        adjustment = 1.0 + sl_damping * (ratio - 1.0) * proportion
                        np.clip(adjustment, 0.5, 2.0, out=adjustment)
                        demand *= adjustment
                        corrections_applied.append((sl_name, round(ratio, 3)))

                    if gd_it == 1:
                        n_sl_used = len(corrections_applied)

                    # Elasticity: clip to [seed/max_dev, seed*max_dev]
                    np.clip(demand, seed_lower, seed_upper, out=demand)
                    np.maximum(demand, 0.0, out=demand)

                    logger.info(
                        f"    GD inner {gd_it}: {len(corrections_applied)} SLs applied, "
                        f"demand_total={demand.sum():,.0f}"
                    )

                if corrections_applied:
                    logger.info(
                        f"  Screenline ratios: "
                        f"{', '.join(f'{n}={r}' for n, r in corrections_applied)}"
                    )

            # --- 6) Global residual correction from ALL count posts ---
            # After the select-link gradient step, compute aggregate bias
            # from ALL 2000+ matched count posts and apply a damped
            # uniform correction.  Stratified (per-road-class) correction
            # was tested but is counterproductive: it cuts through-traffic
            # uniformly when some corridors need more and others less.
            # The screenline gradient handles the directional component;
            # the global scalar handles only the aggregate level.
            if len(valid) > 0 and total_vol > 0:
                sum_obs = float(obs_all.sum())
                sum_mod = float(mod_all.sum())
                if sum_mod > 0 and sum_obs > 0:
                    global_ratio = sum_obs / sum_mod
                    global_factor = 1.0 + effective_global_damping * (global_ratio - 1.0)
                    global_factor = float(np.clip(global_factor, 0.8, 1.25))

                    if abs(global_factor - 1.0) > 0.003:
                        demand *= global_factor
                        np.clip(demand, seed_lower, seed_upper, out=demand)
                        np.maximum(demand, 0.0, out=demand)
                        logger.info(
                            f"  Global residual: obs/mod={global_ratio:.3f} "
                            f"→ factor={global_factor:.4f}  "
                            f"demand={demand.sum():,.0f}"
                        )

            # --- 7) Gateway calibration — per-gateway OD scaling ---
            if ctx.gw_cal_enabled and vol_col:
                gw_modeled = _compute_gateway_modeled_volumes(
                    vol_df, screenlines, vol_col,
                )
                gw_corrections = _apply_gateway_calibration(
                    demand,
                    ctx.gw_zone_map,
                    ctx.gw_observed,
                    gw_modeled,
                    damping=ctx.gw_cal_damping,
                    min_factor=ctx.gw_cal_min_factor,
                    max_factor=ctx.gw_cal_max_factor,
                    seed_lower=seed_lower,
                    seed_upper=seed_upper,
                )
                if gw_corrections:
                    logger.info(f"  Gateway calibration ({len(gw_corrections)}):")
                    for gc_line in gw_corrections:
                        logger.info(f"    {gc_line}")

            # --- 8) Write updated demand back to matrix ---
            data[:, :] = demand
            mat.save()
            logger.info(f"  Matrix saved. Total demand: {demand.sum():,.0f}")

    finally:
        ctx.best_Z = best_Z
        ctx.best_demand = best_demand
        ctx.best_iteration = best_iteration
        ctx.restore_best_and_close()

    # Run final assignment on best-state demand so artifacts are consistent
    best_vol_df = ctx.finalize_best_state()
    if best_vol_df is not None:
        vol_df = best_vol_df

    # Use best iteration stats for the report when available
    model_time_period = str(calib_cfg.get("model_time_period", "daily"))
    best_final = history[best_iteration - 1] if history and 0 < best_iteration <= len(history) else (history[-1] if history else {})

    # --- Save calibration report ---
    report = {
        "method": "odme_spiess_gradient",
        "iterations": len(history),
        "converged": len(history) > 0 and (
            (len(history) >= 2 and abs(history[-1].get("Z_objective", 0) - history[-2].get("Z_objective", 1)) / max(abs(history[-2].get("Z_objective", 1)), 1) < conv_tol)
            or _check_final_convergence(history, model_time_period, 85.0, daily_conv)
        ),
        "history": history,
        "final": best_final,
        "config": {
            "max_outer_iterations": max_outer,
            "gradient_descent_iterations": gd_inner,
            "max_deviation": max_deviation,
            "weight_function": weight_method,
            "convergence_tol": conv_tol,
            "global_residual_damping": global_residual_damping,
            "algorithm": algorithm,
            "count_target": count_target,
            "obs_col": obs_col,
            "aggregate_corridor": agg_corridor,
        },
        "screenlines": sl_results,
    }
    report_path = output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nODME report: {report_path}")

    if history:
        z_start = history[0].get("Z_objective", 0)
        z_end = best_Z if best_Z < float("inf") else history[-1].get("Z_objective", 0)
        logger.info(
            f"  Z: {z_start:,.1f} → {z_end:,.1f}  "
            f"(reduction: {(1 - z_end / max(z_start, 1)) * 100:.1f}%)  "
            f"best at iteration {best_iteration}"
        )

    # Save final assignment results
    if history:
        out_path = output_dir / "assignment_results.parquet"
        vol_df.to_parquet(str(out_path), index=False)
        logger.info(f"  Final assignment: {out_path}")


# ---------------------------------------------------------------------------
# Legacy iterative calibration
# ---------------------------------------------------------------------------

def run_calibration(config_path: str | Path = "config/sim.yaml") -> None:
    """FSM iterative calibration: assign → compare → scale → repeat."""
    ctx = _CalibrationContext(config_path)
    cfg = ctx.cfg
    calib_cfg = ctx.calib_cfg

    # Aliases for loop body backward compat
    mat = ctx.mat
    project = ctx.project
    cached_graph = ctx.cached_graph
    links_gdf = ctx.links_gdf
    pent = ctx.pent
    sl_query = ctx.sl_query
    screenlines = ctx.screenlines
    output_dir = ctx.output_dir
    project_dir = ctx.project_dir
    matrix_path = ctx.matrix_path
    core_name = ctx.core_name
    algorithm = ctx.algorithm
    max_iter_assign = ctx.max_iter_assign
    rgap = ctx.rgap
    gc_field = ctx.gc_field
    gc_mult = ctx.gc_mult
    gc_vot = ctx.gc_vot
    cfg_bpr = ctx.cfg_bpr
    cfg_multi = ctx.cfg_multi
    daily_cap_factor = ctx.daily_cap_factor
    cores = ctx.cores
    buffer_m = ctx.buffer_m
    direction_aware = ctx.direction_aware
    conflict_res = ctx.conflict_res
    agg_corridor = ctx.agg_corridor
    obs_col = ctx.obs_col
    count_target = ctx.count_target
    daily_conv = ctx.daily_conv
    seed_lower = ctx.seed_lower
    seed_upper = ctx.seed_upper

    model_time_period = str(calib_cfg.get("model_time_period", "daily"))

    max_iterations = int(calib_cfg.get("max_iterations", 10))
    conv_cfg = calib_cfg.get("convergence") or {}
    geh_target = float(conv_cfg.get("geh_lt5_target_pct", 85.0))
    min_improvement = float(conv_cfg.get("min_improvement_pct", 1.0))

    daily_r2_target = float(daily_conv.get("r2_target", 0.80))
    daily_slope_range = daily_conv.get("slope_range", [0.85, 1.15])
    daily_slope_lo = float(daily_slope_range[0])
    daily_slope_hi = float(daily_slope_range[1])
    daily_pct_rmse_max = float(daily_conv.get("pct_rmse_max", 35.0))
    daily_sl_max_dev = float(daily_conv.get("screenline_max_pct_deviation", 15.0))
    daily_bias_max = float(daily_conv.get("bias_abs_max_pct", 15.0))

    scale_cfg = calib_cfg.get("scaling") or {}
    scale_method = str(scale_cfg.get("method", "sector"))
    scale_enabled = bool(scale_cfg.get("enabled", True))
    damping = float(scale_cfg.get("damping", 0.5))
    # NOTE: min_factor, max_factor, and adaptive_data_driven from scaling
    # config are loaded for reporting but NOT used in the actual update step.
    # Elasticity is controlled by seed_lower / seed_upper (from max_deviation).
    _unused_scale_params = {
        k: scale_cfg[k]
        for k in ("min_factor", "max_factor", "adaptive_data_driven")
        if k in scale_cfg
    }
    if _unused_scale_params:
        logger.debug(
            "Scaling config params present but unused (elasticity uses max_deviation): %s",
            _unused_scale_params,
        )
    quality_cfg = calib_cfg.get("quality_gates") or {}
    q_bias_hard = float(quality_cfg.get("hard_class_bias_max_abs_pct", 90.0))
    q_wmape_warn = float(quality_cfg.get("warn_wmape_pct", 47.0))
    q_geh_warn = float(quality_cfg.get("warn_geh_lt5_pct", 7.0))
    q_obj_patience = int(quality_cfg.get("objective_patience", 3))
    q_obj_weights = quality_cfg.get("objective_weights") or {}
    w_rho = float(q_obj_weights.get("spearman", 120.0))
    w_r2 = float(q_obj_weights.get("r2", 80.0))
    w_slope = float(q_obj_weights.get("slope_penalty", 50.0))
    w_geh = float(q_obj_weights.get("geh_lt5", 1.8))
    w_wmape = float(q_obj_weights.get("wmape_pct", 1.0))
    w_bias = float(q_obj_weights.get("class_bias_max_abs_pct", 0.35))
    w_rmse = float(q_obj_weights.get("pct_rmse", 0.35))

    logger.info("=== FSM ITERATIVE CALIBRATION ===")
    logger.info(f"  model_time_period={model_time_period}, max_iterations={max_iterations}")
    if model_time_period == "daily":
        logger.info(
            f"  Daily convergence: R²>={daily_r2_target}, slope∈[{daily_slope_lo},{daily_slope_hi}], "
            f"%RMSE<={daily_pct_rmse_max}, SL_dev<={daily_sl_max_dev}%, |bias|<={daily_bias_max}%"
        )
        logger.info(
            f"  (GEH<5 >= {geh_target}% kept as diagnostic; daily_capacity_factor={daily_cap_factor})"
        )
    else:
        logger.info(f"  Hourly convergence: target GEH<5 >= {geh_target}%")
    logger.info(f"  scaling: enabled={scale_enabled}, method={scale_method}, damping={damping}")
    logger.info(f"  count_target: {count_target} (comparing against '{obs_col}')")

    from sim.screenlines import evaluate_all_screenlines

    history = ctx.history
    prev_geh5 = 0.0
    mq: Dict[str, Any] = {}
    last_extended: Dict[str, Any] = {}
    obj_best = float("-inf")
    obj_non_improve = 0
    best_Z = float("inf")
    best_demand: Optional[np.ndarray] = None
    best_iteration = 0
    prev_Z = float("inf")

    try:
        for it in range(1, max_iterations + 1):
            logger.info(f"\n── Iteration {it}/{max_iterations} ──")

            # 1) Assignment
            total_demand = float(mat.matrix_view.sum())
            logger.info(f"  Demand total: {total_demand:,.0f}")
            save_skims_now = bool(calib_cfg.get("save_skims", False)) and it == 1
            vol_df, skims, sl_matrices = execute_assignment(
                project,
                mat,
                algorithm=algorithm,
                max_iter=max_iter_assign,
                rgap_target=rgap,
                save_skims=save_skims_now,
                select_links=sl_query,
                fixed_cost_field=gc_field,
                fixed_cost_multiplier=gc_mult,
                vot=gc_vot,
                bpr_parameters=cfg_bpr,
                multi_class=cfg_multi,
                graph=cached_graph,
                cores=cores,
            )
            if skims is not None:
                skim_path = output_dir / "skims.aem"
                try:
                    skims.export(str(skim_path))
                    logger.info(f"  Skims saved: {skim_path}")
                except Exception as exc:
                    import warnings as _w
                    _w.warn(f"Skim export failed ({skim_path}): {exc}", RuntimeWarning, stacklevel=1)

            vol_col = _detect_volume_col(vol_df)

            # Multi-class fix: sum per-class vehicle volumes into a single
            # column so calibration compares TOTAL vehicles against observed
            # counts, not just the first class (which misses through traffic).
            class_tot_cols = [
                c for c in vol_df.columns
                if c.endswith("_tot") and c not in ("PCE_tot", "Preload_tot")
                and vol_df[c].sum() > 0
            ]
            if len(class_tot_cols) > 1:
                vol_df["total_vehicles_tot"] = vol_df[class_tot_cols].sum(axis=1)
                vol_col = "total_vehicles_tot"

            total_vol = float(vol_df[vol_col].sum()) if vol_col else 0.0
            logger.info(f"  Assigned volume: {total_vol:,.0f}  (col={vol_col})")
            if total_vol > 0 and total_demand > 0:
                logger.info(f"  Route amplification: {total_vol / total_demand:.1f} links/trip")

            if it == 1:
                try:
                    bkdn = compute_class_volume_breakdown(vol_df, links_gdf)
                    if bkdn:
                        logger.info("  Volume breakdown by road class / traffic class:")
                        for col_name, by_rc in bkdn.items():
                            if col_name.startswith("_"):
                                continue
                            for rc in ("motorway", "trunk", "primary", "secondary", "tertiary", "other"):
                                info = by_rc.get(rc)
                                if info:
                                    logger.info(
                                        f"    {col_name:20s}  {rc:12s}  vol={info['total_volume']:>12,.0f}  links={info['n_links']}"
                                    )
                        shares = bkdn.get("_class_shares", {})
                        for rc, detail in shares.items():
                            logger.info(f"    {rc} class shares: {detail['shares']}")
                except Exception as ex:
                    logger.warning(f"  WARNING: volume breakdown failed: {ex}")

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
                vol_col=vol_col,
            )
            vc = vol_col if vol_col and vol_col in matched.columns else next(
                (c for c in matched.columns if vol_col and vol_col in c), None
            )

            # 3) Compute stats (use corridor volume when available)
            compare_col = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
            if vc and compare_col and compare_col in matched.columns:
                valid = matched.dropna(subset=[compare_col, obs_col])
                valid = valid[valid[obs_col] > 0].copy()
                if "_excluded" in valid.columns:
                    valid = valid[~valid["_excluded"]].copy()
                stats = compute_stats(
                    valid[compare_col].values, valid[obs_col].values,
                    daily_capacity_factor=daily_cap_factor,
                )
            else:
                valid = pd.DataFrame()
                stats = {"n": 0}

            geh5 = float(stats.get("geh_lt5_pct", 0))
            geh10 = float(stats.get("geh_lt10_pct", 0))
            r2 = stats.get("r2")
            slope = stats.get("slope")
            pct_rmse = stats.get("pct_rmse")
            bias_pct = stats.get("bias_pct")
            daily_geh_adj = stats.get("daily_geh_lt_adj_pct", 0)

            # Track weighted objective Z for best-state restoration
            Z_current = float("inf")
            if not valid.empty and compare_col in valid.columns:
                _mod = valid[compare_col].values.astype(np.float64)
                _obs = valid[obs_col].values.astype(np.float64)
                _w = _compute_count_weights(_obs)
                Z_current = _odme_objective(_mod, _obs, _w)
                if Z_current < best_Z:
                    best_Z = Z_current
                    best_demand = mat.matrix[core_name][:, :].copy()
                    best_iteration = it

            logger.info(f"  R²={r2}  slope={slope}  %RMSE={pct_rmse}  bias={bias_pct}%")
            logger.info(
                f"  GEH<5: {geh5:.1f}%  GEH<10: {geh10:.1f}%  "
                f"daily-adj GEH<{stats.get('daily_geh_threshold', 5):.0f}: {daily_geh_adj:.1f}%"
            )
            logger.info(f"  Z={Z_current:,.0f}  (best={best_Z:,.0f} at it={best_iteration})")

            if not valid.empty and "link_id" in valid.columns and "direction" in links_gdf.columns:
                _dir_lookup = links_gdf[["link_id", "direction"]].drop_duplicates("link_id")
                _dir_lookup = _dir_lookup.rename(columns={"direction": "_link_dir"})
                vdir = valid.merge(_dir_lookup, on="link_id", how="left")
                for dval, label in [(0, "bidir"), (1, "oneway")]:
                    mask = vdir["_link_dir"] == dval if dval == 0 else vdir["_link_dir"] != 0
                    sub = vdir.loc[mask]
                    if len(sub) >= 2:
                        s_stats = compute_stats(
                            sub[compare_col].values, sub[obs_col].values,
                            daily_capacity_factor=daily_cap_factor,
                        )
                        logger.info(
                            f"    {label} (n={len(sub)}): slope={s_stats.get('slope')}  "
                            f"bias={s_stats.get('bias_pct')}%  R²={s_stats.get('r2')}"
                        )

            if it == 1 and not valid.empty and "link_type" in valid.columns:
                _rc = valid["link_type"].map(_coarse_road_class)
                for rc in ("motorway", "trunk", "primary", "secondary", "tertiary", "other"):
                    rc_rows = valid[_rc == rc]
                    if len(rc_rows) < 2:
                        continue
                    _m = rc_rows[compare_col].values.astype(float)
                    _o = rc_rows[obs_col].values.astype(float)
                    _ratio = float(_m.sum() / max(_o.sum(), 1))
                    logger.info(
                        f"    {rc:12s} (n={len(rc_rows):4d}): "
                        f"sum_mod={_m.sum():>12,.0f}  sum_obs={_o.sum():>12,.0f}  "
                        f"ratio={_ratio:.3f}"
                    )

            last_extended = {}
            if not valid.empty and compare_col and compare_col in valid.columns:
                try:
                    last_extended = compute_extended_link_metrics(
                        valid, compare_col, obs_col,
                    )
                except Exception as ex:
                    logger.warning(f"  WARNING: extended link metrics failed: {ex}")
                    last_extended = {}

            mq = match_quality_report(matched)
            if it == 1:
                logger.info(
                    f"  Match quality: {mq['n_matched']}/{mq['n_total']} matched, "
                    f"{mq['n_link_conflicts']} conflicts, "
                    f"mean_dist={mq['mean_match_distance_m']}m"
                )

                # Build persistent exclusion set from iteration-1 matching
                # so that bad stations (e.g. 9k service road matched to 62k
                # trunk) never influence OD scaling in subsequent iterations.
                if "_excluded" in matched.columns and "objectid" in matched.columns:
                    excl_ids = set(matched.loc[matched["_excluded"], "objectid"].dropna().astype(int))
                    if excl_ids:
                        n_before = len(pent)
                        pent = pent[~pent["objectid"].isin(excl_ids)].copy()
                        logger.info(
                            f"  Pre-filter: removed {n_before - len(pent)} excluded stations "
                            f"from pentlogram ({len(pent)} remaining)"
                        )

                # Export matching diagnostics CSV
                _export_matching_diagnostics(matched, compare_col, obs_col, output_dir)

                # Top links by absolute bias (motorway + trunk only)
                if compare_col and compare_col in valid.columns and "link_type" in valid.columns:
                    try:
                        diag = valid[[compare_col, obs_col, "link_type"]].copy()
                        diag["_bias"] = diag[compare_col] - diag[obs_col]
                        diag["_road_class"] = diag["link_type"].map(_coarse_road_class)
                        for rc in ("motorway", "trunk"):
                            rc_rows = diag[diag["_road_class"] == rc].copy()
                            if rc_rows.empty:
                                continue
                            worst = rc_rows.reindex(rc_rows["_bias"].abs().nlargest(5).index)
                            logger.info(f"  Top-5 {rc} links by |bias|:")
                            for _, row in worst.iterrows():
                                logger.info(
                                    f"    mod={row[compare_col]:>8,.0f}  obs={row[obs_col]:>8,.0f}  "
                                    f"bias={row['_bias']:>+8,.0f}  type={row['link_type']}"
                                )
                    except Exception:
                        logger.debug(
                            "Top-5 bias by road class diagnostic failed",
                            exc_info=True,
                        )

            # Evaluate screenlines
            sl_results: Dict[str, Any] = {}
            max_sl_pct_dev: float = 0.0
            if screenlines and vol_col:
                sl_res = evaluate_all_screenlines(
                    screenlines, vol_df, matched, vol_col, obs_col, links_gdf,
                )
                for sn, sr in sl_res.items():
                    sl_results[sn] = sr.to_dict()
                    if sr.observed_total > 0 and np.isfinite(sr.ratio):
                        dev = abs(sr.ratio - 1.0) * 100.0
                        max_sl_pct_dev = max(max_sl_pct_dev, dev)
                    if it == 1 or it == max_iterations:
                        logger.info(
                            f"  Screenline '{sn}': mod={sr.modeled_total:,.0f} "
                            f"obs={sr.observed_total:,.0f} ratio={sr.ratio:.2f} GEH={sr.geh:.1f}"
                        )
                if sl_results:
                    logger.info(f"  Screenline max %deviation: {max_sl_pct_dev:.1f}%")

            iter_record = {
                "iteration": it,
                "demand_total": round(total_demand, 0),
                "assigned_total": round(total_vol, 0),
                **stats,
                "max_screenline_pct_dev": round(max_sl_pct_dev, 1),
            }
            if last_extended:
                iter_record["wmape_pct"] = last_extended.get("wmape_pct")
                iter_record["class_bias_max_abs_pct"] = last_extended.get("class_bias_max_abs_pct")
                iter_record["spearman_rho"] = last_extended.get("spearman_rho")

            # Quality objective (higher is better) — daily models weight R²,
            # slope, and %RMSE; hourly models lean on GEH.
            if last_extended:
                try:
                    rho = float(last_extended.get("spearman_rho") or 0.0)
                    wmape = float(last_extended.get("wmape_pct") or 0.0)
                    bias_abs = float(last_extended.get("class_bias_max_abs_pct") or 0.0)
                    cur_r2 = float(stats.get("r2") or 0.0)
                    cur_slope = float(stats.get("slope") or 1.0)
                    slope_dev = abs(cur_slope - 1.0)
                    cur_pct_rmse = float(stats.get("pct_rmse") or 0.0)

                    obj_q = (
                        w_rho * rho
                        + w_r2 * cur_r2
                        - w_slope * slope_dev
                        + w_geh * geh5
                        - w_wmape * wmape
                        - w_bias * bias_abs
                        - w_rmse * cur_pct_rmse
                    )
                    iter_record["quality_objective"] = round(obj_q, 4)
                    if obj_q > obj_best + 1e-9:
                        obj_best = obj_q
                        obj_non_improve = 0
                    else:
                        obj_non_improve += 1
                except Exception:
                    logger.debug(
                        "Quality objective computation failed",
                        exc_info=True,
                    )
            history.append(iter_record)

            # 4) Convergence check — daily vs hourly criteria
            converged = False
            if model_time_period == "daily":
                cur_r2 = float(stats.get("r2") or 0.0)
                cur_slope = float(stats.get("slope") or 0.0)
                cur_prmse = float(stats.get("pct_rmse") or 999.0)
                cur_bias = abs(float(stats.get("bias_pct") or 999.0))
                checks = {
                    f"R²>={daily_r2_target}": cur_r2 >= daily_r2_target,
                    f"slope∈[{daily_slope_lo},{daily_slope_hi}]": daily_slope_lo <= cur_slope <= daily_slope_hi,
                    f"%RMSE<={daily_pct_rmse_max}": cur_prmse <= daily_pct_rmse_max,
                    f"|bias|<={daily_bias_max}%": cur_bias <= daily_bias_max,
                }
                if sl_results:
                    checks[f"SL_dev<={daily_sl_max_dev}%"] = max_sl_pct_dev <= daily_sl_max_dev

                passed = [k for k, v in checks.items() if v]
                failed = [k for k, v in checks.items() if not v]
                logger.info(f"  Daily convergence: {len(passed)}/{len(checks)} criteria met")
                if failed:
                    logger.info(f"    PASS: {', '.join(passed) if passed else 'none'}")
                    logger.info(f"    FAIL: {', '.join(failed)}")

                if all(checks.values()):
                    logger.info(f"  CONVERGED (daily): all criteria met")
                    converged = True
            else:
                if geh5 >= geh_target:
                    logger.info(f"  CONVERGED: GEH<5 = {geh5:.1f}% >= target {geh_target}%")
                    converged = True

            if converged:
                break

            # Stall detection: Z-based (primary) + quality-objective (secondary)
            if it - best_iteration >= q_obj_patience + 1:
                logger.info(f"  Z-STALL: no Z improvement since iteration {best_iteration}")
                break
            if model_time_period == "daily":
                if q_obj_patience > 0 and obj_non_improve >= q_obj_patience:
                    logger.info(
                        f"  QUALITY STOP: objective non-improving for "
                        f"{obj_non_improve} iterations"
                    )
                    break
            else:
                improvement = geh5 - prev_geh5
                if it > 1 and improvement < min_improvement:
                    logger.info(
                        f"  STALLED: GEH improvement {improvement:.2f}% "
                        f"< {min_improvement}%"
                    )
                    break
                if q_obj_patience > 0 and obj_non_improve >= q_obj_patience:
                    logger.info(
                        f"  QUALITY STOP: objective non-improving for "
                        f"{obj_non_improve} iterations"
                    )
                    break

            prev_geh5 = geh5

            # 5) Spiess-style OD update: screenline corrections + global residual + elasticity clip
            if scale_enabled and not valid.empty and compare_col and total_vol > 0:
                data = mat.matrix[core_name]
                demand = data[:, :].copy().astype(np.float64)

                # (a) Screenline Spiess corrections
                if sl_matrices and sl_results:
                    n_sl_active = sum(
                        1 for sn in sl_matrices
                        if _sr_val(sl_results.get(sn, {}), "observed_total") > 0
                        and _sr_val(sl_results.get(sn, {}), "modeled_total") > 0
                    )
                    sl_damp = 1.0 / max(np.sqrt(n_sl_active), 1.0)
                    corrections_log = []

                    for sl_name, sl_od_raw in sl_matrices.items():
                        obs_sl = _sr_val(sl_results.get(sl_name, {}), "observed_total")
                        mod_sl = _sr_val(sl_results.get(sl_name, {}), "modeled_total")
                        if obs_sl <= 0 or mod_sl <= 0:
                            continue
                        ratio = obs_sl / mod_sl
                        if ratio > 5.0 or ratio < 0.2:
                            continue

                        sl_od = np.asarray(sl_od_raw, dtype=np.float64).reshape(demand.shape)
                        proportion = np.where(
                            demand > 0,
                            np.clip(sl_od / np.maximum(demand, 1e-9), 0.0, 1.0),
                            0.0,
                        )
                        adjustment = 1.0 + sl_damp * (ratio - 1.0) * proportion
                        np.clip(adjustment, 0.5, 2.0, out=adjustment)
                        demand *= adjustment
                        corrections_log.append(f"{sl_name}={ratio:.2f}")

                    if corrections_log:
                        logger.info(
                            f"  Spiess SL corrections ({len(corrections_log)}): "
                            f"{', '.join(corrections_log)}"
                        )

                # (b) Global residual correction from all count posts
                sum_obs = float(valid[obs_col].sum())
                sum_mod = float(valid[compare_col].sum())
                if sum_mod > 0 and sum_obs > 0:
                    global_ratio = sum_obs / sum_mod
                    global_factor = 1.0 + damping * (global_ratio - 1.0)
                    global_factor = float(np.clip(global_factor, 0.8, 1.25))
                    if abs(global_factor - 1.0) > 0.003:
                        demand *= global_factor
                        logger.info(
                            f"  Global residual: obs/mod={global_ratio:.3f} "
                            f"→ factor={global_factor:.4f}"
                        )

                # (c) Gateway calibration — per-gateway OD scaling
                if ctx.gw_cal_enabled and vol_col:
                    gw_modeled = _compute_gateway_modeled_volumes(
                        vol_df, screenlines, vol_col,
                    )
                    gw_corrections = _apply_gateway_calibration(
                        demand,
                        ctx.gw_zone_map,
                        ctx.gw_observed,
                        gw_modeled,
                        damping=ctx.gw_cal_damping,
                        min_factor=ctx.gw_cal_min_factor,
                        max_factor=ctx.gw_cal_max_factor,
                        seed_lower=seed_lower,
                        seed_upper=seed_upper,
                    )
                    if gw_corrections:
                        logger.info(f"  Gateway calibration ({len(gw_corrections)}):")
                        for gc_line in gw_corrections:
                            logger.info(f"    {gc_line}")

                # (d) Elasticity clip
                np.clip(demand, seed_lower, seed_upper, out=demand)
                np.maximum(demand, 0.0, out=demand)
                data[:, :] = demand
                logger.info(f"  Demand after update: {demand.sum():,.0f}")

            elif not scale_enabled:
                logger.info("  Scaling disabled (principle: frozen / diagnostic run)")
            else:
                logger.warning("  Cannot scale — no valid matched volumes")

            # Persist scaled matrix for next iteration
            mat.save()

    finally:
        ctx.best_Z = best_Z
        ctx.best_demand = best_demand
        ctx.best_iteration = best_iteration
        ctx.restore_best_and_close()

    # Run final assignment on best-state demand so artifacts are consistent
    best_vol_df = ctx.finalize_best_state()
    if best_vol_df is not None:
        vol_df = best_vol_df

    # Per-period validation (uses daily observed × period shares)
    period_stats: Dict[str, Any] = {}
    period_cfg = calib_cfg.get("period_calibration") or {}
    if period_cfg.get("enabled", False) and history:
        try:
            from sim.temporal import load_profile, get_demand_period_shares
            profile = load_profile(cfg)
            period_shares = get_demand_period_shares(profile)
            cal_periods = period_cfg.get("periods", ["am", "pm", "daily"])

            logger.info("\n=== PER-PERIOD VALIDATION ===")

            mat_p = AequilibraeMatrix()
            mat_p.load(str(matrix_path))
            project_p = Project()
            project_p.open(str(project_dir))

            try:
                for period in cal_periods:
                    p_core = f"wd_{period}" if period != "daily" else "wd_daily"
                    if p_core not in mat_p.names:
                        logger.warning(f"  Skipping {period}: core '{p_core}' not in matrix")
                        continue

                    p_share = period_shares.get(period, 1.0)
                    logger.info(f"\n  Period: {period} (share={p_share:.3f}, core={p_core})")

                    mat_p.computational_view([p_core])
                    p_demand = float(mat_p.matrix_view.sum())
                    logger.info(f"    Demand: {p_demand:,.0f}")

                    vol_df_p, _, _sl_p = execute_assignment(
                        project_p,
                        mat_p,
                        algorithm=algorithm,
                        max_iter=max_iter_assign,
                        rgap_target=rgap,
                        fixed_cost_field=gc_field,
                        fixed_cost_multiplier=gc_mult,
                        vot=gc_vot,
                        bpr_parameters=cfg_bpr,
                        multi_class=cfg_multi,
                        cores=cores,
                    )

                    vc_p = _detect_volume_col(vol_df_p)
                    if not vc_p:
                        continue

                    lv = links_gdf.copy()
                    lv = lv.merge(vol_df_p[["link_id", vc_p]], on="link_id", how="left")
                    m_p = match_counts_to_links(pent, lv, buffer_m=buffer_m, vol_col=vc_p)

                    obs_period_col = f"_obs_{period}"
                    if period == "daily":
                        m_p[obs_period_col] = m_p[obs_col]
                    else:
                        m_p[obs_period_col] = m_p[obs_col] * p_share

                    vp = m_p.dropna(subset=[vc_p, obs_period_col])
                    vp = vp[vp[obs_period_col] > 0].copy()
                    if "_excluded" in vp.columns:
                        vp = vp[~vp["_excluded"]].copy()
                    if len(vp) > 0:
                        p_dcf = daily_cap_factor if period == "daily" else 1.0
                        ps = compute_stats(
                            vp[vc_p].values, vp[obs_period_col].values,
                            daily_capacity_factor=p_dcf,
                        )
                        period_stats[period] = ps
                        logger.info(f"    R²={ps.get('r2')}  slope={ps.get('slope')}  "
                              f"%RMSE={ps.get('pct_rmse')}  n={ps.get('n')}")
                    else:
                        logger.warning(f"    No valid matched volumes for {period}")
            finally:
                mat_p.close()
                project_p.close()
        except Exception:
            logger.exception("Period calibration validation failed")

    # Use best iteration stats for the report when available
    best_final = history[best_iteration - 1] if history and 0 < best_iteration <= len(history) else (history[-1] if history else {})

    # Save calibration report
    report = {
        "iterations": len(history),
        "model_time_period": model_time_period,
        "converged": _check_final_convergence(history, model_time_period, geh_target, daily_conv),
        "history": history,
        "final": best_final,
        "config": {
            "max_iterations": max_iterations,
            "model_time_period": model_time_period,
            "daily_capacity_factor": daily_cap_factor,
            "geh_target": geh_target,
            "daily_convergence": daily_conv if model_time_period == "daily" else None,
            "scale_method": scale_method,
            "scale_enabled": scale_enabled,
            "damping": damping,
            "count_target": count_target,
            "obs_col": obs_col,
            "aggregate_corridor": agg_corridor,
            "matching": calib_cfg.get("matching") or {},
        },
        "extended_metrics": last_extended,
        "quality_gates": {
            "hard_class_bias_max_abs_pct": q_bias_hard,
            "warn_wmape_pct": q_wmape_warn,
            "warn_geh_lt5_pct": q_geh_warn,
            "objective_patience": q_obj_patience,
            "objective_weights": {
                "spearman": w_rho,
                "r2": w_r2,
                "slope_penalty": w_slope,
                "geh_lt5": w_geh,
                "wmape_pct": w_wmape,
                "class_bias_max_abs_pct": w_bias,
                "pct_rmse": w_rmse,
            },
            "hard_reject": bool((last_extended or {}).get("class_bias_max_abs_pct", 0.0) > q_bias_hard) if last_extended else False,
            "warnings": [
                w for w in [
                    (f"wmape_pct>{q_wmape_warn}" if last_extended and (last_extended.get("wmape_pct") or 0.0) > q_wmape_warn else None),
                    (f"geh_lt5_pct<{q_geh_warn}" if history and (history[-1].get("geh_lt5_pct", 0.0) < q_geh_warn) else None),
                ] if w is not None
            ],
        },
        "observed_summary": {
            "total_motor": round(float(pent["observed_motor_total"].sum()), 0),
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
    logger.info(f"\nCalibration report: {report_path}")

    # Save final assignment
    if history:
        out_path = output_dir / "assignment_results.parquet"
        vol_df.to_parquet(str(out_path), index=False)
        logger.info(f"Final assignment: {out_path}")


# ---------------------------------------------------------------------------
# Independent validation (CSD)
# ---------------------------------------------------------------------------

_CSD_COMPATIBLE_LINK_TYPES: dict = {
    "motorway": {"motorway", "motorway_link"},
    "trunk": {"trunk", "trunk_link", "primary", "primary_link"},
    "secondary": {"secondary", "secondary_link", "primary", "primary_link"},
    "tertiary": {"tertiary", "tertiary_link", "secondary", "secondary_link", "unclassified", "residential"},
}

_MIN_VOL_FOR_CSD_LW = 100  # boundary links with < 100 veh/day are artifacts


def match_csd_to_links(
    csd: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Per-road matching via ``osm_ref`` ↔ CSD ``sil``.

    For each CSD road number (e.g. D1, 52, 152), finds all model links
    whose ``osm_ref`` contains that road number (splitting composite
    refs like ``"D1;50"`` on ``";"``) and computes a length-weighted
    mean model volume, compared against the CSD section-averaged AADT.

    Links with near-zero volume (< 100 veh/day) are excluded from the
    length-weighted mean to avoid dilution by boundary artifacts.
    Link types are filtered to be compatible with the CSD road class.
    """
    csd = csd.copy()
    if "sil" not in csd.columns or "osm_ref" not in links_gdf.columns:
        return pd.DataFrame()

    csd["sil"] = csd["sil"].astype(str)
    for col in ("o", "sv", "tv"):
        if col in csd.columns:
            csd[col] = pd.to_numeric(csd[col], errors="coerce").fillna(0)

    csd["road_class"] = csd["sil"].apply(_classify_csd_road)
    csd_sil = csd["sil"].str.strip()

    vol_cols = sorted(
        c for c in links_gdf.columns
        if c.endswith("_tot") and c not in ("PCE_tot", "Preload_tot")
    )
    tot_col = None
    if "total_vehicles_tot" in vol_cols:
        tot_col = "total_vehicles_tot"
    else:
        for vc in vol_cols:
            if links_gdf[vc].sum() > 0:
                tot_col = vc
                break
    if tot_col is None:
        return pd.DataFrame()

    links_work = links_gdf.copy()
    links_work[tot_col] = pd.to_numeric(links_work[tot_col], errors="coerce").fillna(0)
    if "distance" in links_work.columns:
        links_work["distance"] = pd.to_numeric(links_work["distance"], errors="coerce").fillna(0)

    raw_refs = links_work["osm_ref"].fillna("").str.strip()

    # Build a reverse index: for each link, explode composite osm_ref
    # ("D1;50" → ["D1", "50"]) so CSD roads match any component.
    ref_components = raw_refs.str.split(";").explode().str.strip()
    ref_components = ref_components[ref_components != ""]

    matched_roads: list = []
    csd_roads = csd_sil.unique()

    for road in csd_roads:
        csd_sub = csd[csd_sil == road]
        if csd_sub.empty:
            continue

        matching_indices = ref_components.index[ref_components == road]
        if len(matching_indices) == 0:
            continue
        model_sub = links_work.loc[matching_indices.unique()]

        csd_mean_sv = float(csd_sub["sv"].mean())
        csd_mean_o = float(csd_sub["o"].mean())
        n_csd = len(csd_sub)
        road_class = csd_sub["road_class"].iloc[0]

        car_links = model_sub
        if "link_type" in car_links.columns:
            lt = car_links["link_type"].astype(str)
            car_links = car_links[~lt.isin(_NON_CAR_LINK_TYPES)]
            compatible = _CSD_COMPATIBLE_LINK_TYPES.get(road_class)
            if compatible:
                car_links = car_links[lt.reindex(car_links.index).isin(compatible)]

        n_model = len(car_links)
        if n_model == 0:
            continue

        active = car_links[car_links[tot_col] >= _MIN_VOL_FOR_CSD_LW]
        if active.empty:
            active = car_links

        vols = active[tot_col].values
        if "distance" in active.columns:
            dists = active["distance"].values
            dsum = float(dists.sum())
            if dsum > 0:
                model_lw_mean = float((vols * dists).sum() / dsum)
            else:
                model_lw_mean = float(vols.mean())
        else:
            model_lw_mean = float(vols.mean())

        geh = float(compute_geh(np.array([model_lw_mean]), np.array([csd_mean_sv]))[0])

        matched_roads.append({
            "road": road,
            "road_class": road_class,
            "csd_sections": n_csd,
            "model_links": n_model,
            "csd_mean_sv": round(csd_mean_sv, 0),
            "csd_mean_o": round(csd_mean_o, 0),
            "model_lw_mean": round(model_lw_mean, 0),
            "geh": round(geh, 1),
        })

    if not matched_roads:
        return pd.DataFrame()

    result = pd.DataFrame(matched_roads)

    # Summary statistics across matched roads
    obs = result["csd_mean_sv"].values.astype(float)
    mod = result["model_lw_mean"].values.astype(float)

    if len(obs) >= 2:
        mask = (obs > 0) & (mod > 0)
        if mask.sum() >= 2:
            o_valid = obs[mask]
            m_valid = mod[mask]
            ss_res = float(((m_valid - o_valid) ** 2).sum())
            ss_tot = float(((o_valid - o_valid.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            bias = float((m_valid.sum() - o_valid.sum()) / o_valid.sum() * 100)
            pct_rmse = float(np.sqrt(((m_valid - o_valid) ** 2).mean()) / o_valid.mean() * 100)
            result.attrs["summary"] = {
                "n_roads": int(mask.sum()),
                "r2": round(r2, 3),
                "bias_pct": round(bias, 1),
                "pct_rmse": round(pct_rmse, 1),
                "mean_geh": round(float(result.loc[mask, "geh"].mean()), 1),
            }

    return result



def compute_validation_benchmarks(
    count_stats: Dict[str, Any],
    screenline_results: Dict[str, Any],
    jt_results: List[Dict[str, Any]],
    *,
    model_time_period: str = "daily",
    daily_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check validation benchmarks for the model's time aggregation.

    For daily models: R², slope, %RMSE, bias, screenline deviations.
    GEH is diagnostic only (FHWA target is for hourly flows).
    Thresholds default to calibration.convergence.daily if not provided.
    """
    geh5 = float(count_stats.get("geh_lt5_pct", 0))
    geh_pass_hourly = geh5 >= 85.0

    jt_pass_count = sum(1 for r in jt_results if r.get("pass", False))
    jt_total = len(jt_results) if jt_results else 0
    jt_pct = jt_pass_count / max(jt_total, 1) * 100
    jt_pass = jt_pct >= 85.0 if jt_total > 0 else None

    sl_max_error = 0.0
    for sr in screenline_results.values():
        ratio = sr.get("ratio")
        if ratio is not None:
            sl_max_error = max(sl_max_error, abs(ratio - 1.0) * 100)

    result: Dict[str, Any] = {
        "model_time_period": model_time_period,
        "screenline_max_error_pct": round(sl_max_error, 1),
        "jt_within_tolerance_pct": round(jt_pct, 1) if jt_total > 0 else None,
        "jt_benchmark_pass": jt_pass,
        "jt_routes_checked": jt_total,
        # Hourly GEH — always reported; authoritative only for hourly models
        "geh_lt5_pct": round(geh5, 1),
        "geh_benchmark_pass_hourly": geh_pass_hourly,
    }

    if model_time_period == "daily":
        dt = daily_thresholds or {}
        r2 = float(count_stats.get("r2") or 0.0)
        slope = float(count_stats.get("slope") or 0.0)
        prmse = float(count_stats.get("pct_rmse") or 999.0)
        bias = abs(float(count_stats.get("bias_pct") or 999.0))
        daily_geh_adj = float(count_stats.get("daily_geh_lt_adj_pct") or 0.0)

        r2_target = float(dt.get("r2_target", 0.80))
        slope_range = dt.get("slope_range", [0.85, 1.15])
        prmse_max = float(dt.get("pct_rmse_max", 35.0))
        bias_max = float(dt.get("bias_abs_max_pct", 15.0))
        sl_max = float(dt.get("screenline_max_pct_deviation", 15.0))

        r2_pass = r2 >= r2_target
        slope_pass = slope_range[0] <= slope <= slope_range[1]
        prmse_pass = prmse <= prmse_max
        bias_pass = bias <= bias_max
        sl_pass = sl_max_error <= sl_max

        daily_pass = r2_pass and slope_pass and prmse_pass and bias_pass and sl_pass
        result.update({
            "daily_r2": round(r2, 4),
            "daily_r2_pass": r2_pass,
            "daily_slope": round(slope, 4),
            "daily_slope_pass": slope_pass,
            "daily_pct_rmse": round(prmse, 1),
            "daily_pct_rmse_pass": prmse_pass,
            "daily_bias_abs_pct": round(bias, 2),
            "daily_bias_pass": bias_pass,
            "daily_screenline_pass": sl_pass,
            "daily_geh_lt_adj_pct": round(daily_geh_adj, 1),
            "daily_overall_pass": daily_pass and (jt_pass is None or jt_pass),
            "overall_pass": daily_pass and (jt_pass is None or jt_pass),
            "note": "GEH<5 target applies to hourly flows; daily model uses R²/slope/%RMSE/bias",
        })
    else:
        result["overall_pass"] = geh_pass_hourly and (jt_pass is None or jt_pass)

    return result


def run_validation_only(config_path: str | Path = "config/sim.yaml") -> None:
    """Comprehensive independent validation against CSD + screenlines + journey times."""
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)
    buffer_m = float(_get(cfg, ["calibration", "match_buffer_m"], 50.0))

    # Swap closures to validation period (e.g. 2025)
    bc_cfg = cfg.get("baseline_closures") or {}
    valid_period = bc_cfg.get("validation_period")
    if bc_cfg.get("enabled", False) and valid_period:
        from sim.network_normalization import swap_db_closures
        swap_db_closures(config_path, measurement_period=valid_period)
        logger.info("  Closures swapped to validation period: %s", valid_period)

    logger.info("=== COMPREHENSIVE VALIDATION ===")

    results_path = output_dir / "assignment_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No assignment results at {results_path}. Run 'calibrate' or 'assign' first."
        )
    vol_df = pd.read_parquet(str(results_path))

    # Use total_vehicles_tot if the calibration loop already created it;
    # otherwise sum per-class columns (excluding any pre-existing total).
    if "total_vehicles_tot" in vol_df.columns and vol_df["total_vehicles_tot"].sum() > 0:
        vol_col = "total_vehicles_tot"
    else:
        vol_col = _detect_volume_col(vol_df)
        class_tot_cols = [
            c for c in vol_df.columns
            if c.endswith("_tot")
            and c not in ("PCE_tot", "Preload_tot", "total_vehicles_tot")
            and vol_df[c].sum() > 0
        ]
        if len(class_tot_cols) > 1:
            vol_df["total_vehicles_tot"] = vol_df[class_tot_cols].sum(axis=1)
            vol_col = "total_vehicles_tot"

    logger.info(f"  Loaded assignment: {len(vol_df)} links, vol_col={vol_col}")

    links_gdf = _load_network_links(project_dir)
    if vol_col and "link_id" in vol_df.columns:
        links_gdf = links_gdf.merge(vol_df[["link_id", vol_col]], on="link_id", how="left")

    count_target = str(calib_cfg.get("count_target", "total"))
    _ct_map = {"car_only": "observed_car", "motor_total": "observed_motor_total", "total": "observed_total"}
    obs_col = _ct_map.get(count_target, "observed_total")

    assign_cfg = cfg.get("assignment") or {}
    bpr_cfg = assign_cfg.get("bpr") or {}
    daily_cap_factor = resolve_daily_cap_factor_default(bpr_cfg)
    model_time_period = str(calib_cfg.get("model_time_period", "daily"))

    report: Dict[str, Any] = {"model_time_period": model_time_period}

    count_source = str(calib_cfg.get("count_source", "pentlogram"))
    report["count_source"] = count_source

    # 1) Reference comparison (calibration data check)
    pent_stats: Dict[str, Any] = {"n": 0}
    matched = gpd.GeoDataFrame()

    if count_source == "csd_split":
        logger.info("\n1) CSD calibration-subset reference comparison ...")
        try:
            split_cfg = calib_cfg.get("csd_split") or {}
            csd_full = load_csd(cfg)
            calib_csd, _ = split_csd_for_calibration(
                csd_full,
                strategy=str(split_cfg.get("strategy", "alternating")),
                calib_share=float(split_cfg.get("calib_share", 0.65)),
                random_seed=int(split_cfg.get("random_seed", 42)),
            )
            pent = load_csd_as_link_counts(calib_csd, links_gdf)
            if not pent.empty:
                agg_corr = bool(calib_cfg.get("aggregate_corridor", True))
                matched = match_counts_to_links(pent, links_gdf, buffer_m=buffer_m,
                                                 aggregate_corridor=agg_corr,
                                                 vol_col=vol_col)
                vc = vol_col if vol_col and vol_col in matched.columns else None
                compare_vc = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
                if compare_vc and compare_vc in matched.columns:
                    valid = matched.dropna(subset=[compare_vc, obs_col])
                    valid = valid[valid[obs_col] > 0]
                    if "_excluded" in valid.columns:
                        valid = valid[~valid["_excluded"]].copy()
                    pent_stats = compute_stats(
                        valid[compare_vc].values, valid[obs_col].values,
                        daily_capacity_factor=daily_cap_factor,
                    )
                    report["calibration_reference"] = {"matched": int(len(valid)), **pent_stats}
                    logger.info(f"  Matched: {len(valid)}  R²={pent_stats.get('r2')}  "
                          f"bias={pent_stats.get('bias_pct')}%")
        except Exception:
            logger.exception("CSD calibration-subset reference comparison skipped")
    else:
        logger.info(f"\n1) Pentlogram comparison (reference, time_period={model_time_period}) ...")
        try:
            pent = load_pentlogram(cfg)
            validate_geometries_or_fail(
                pent, name="pentlogram", expected_epsg=get_metric_epsg(cfg),
            )
            agg_corr = bool(calib_cfg.get("aggregate_corridor", True))
            matched = match_counts_to_links(pent, links_gdf, buffer_m=buffer_m,
                                             aggregate_corridor=agg_corr,
                                             vol_col=vol_col)
            vc = vol_col if vol_col and vol_col in matched.columns else None
            compare_vc = "_corridor_volume" if "_corridor_volume" in matched.columns else vc
            if compare_vc and compare_vc in matched.columns:
                valid = matched.dropna(subset=[compare_vc, obs_col])
                valid = valid[valid[obs_col] > 0]
                if "_excluded" in valid.columns:
                    valid = valid[~valid["_excluded"]].copy()
                pent_stats = compute_stats(
                    valid[compare_vc].values, valid[obs_col].values,
                    daily_capacity_factor=daily_cap_factor,
                )
                report["pentlogram"] = {"matched": int(len(valid)), **pent_stats}
                logger.info(f"  Matched: {len(valid)}  R²={pent_stats.get('r2')}  "
                      f"slope={pent_stats.get('slope')}  %RMSE={pent_stats.get('pct_rmse')}  "
                      f"bias={pent_stats.get('bias_pct')}%")
                logger.info(f"  GEH<5: {pent_stats.get('geh_lt5_pct')}%  "
                      f"daily-adj GEH<{pent_stats.get('daily_geh_threshold', 5):.0f}: "
                      f"{pent_stats.get('daily_geh_lt_adj_pct')}%")
                ext_metrics = compute_extended_link_metrics(valid, compare_vc, obs_col)
                if ext_metrics:
                    report["extended_metrics"] = ext_metrics
            else:
                logger.warning("  No volume column on links")
        except Exception:
            logger.exception("Pentlogram comparison step skipped")

    # 2) CSD -- independent per-road validation
    #    When count_source=="csd_split", only the validation subset is used.
    logger.info("\n2) CSD independent validation (per-road matching via osm_ref) ...")
    csd_match_df = pd.DataFrame()
    csd = None
    try:
        if count_source == "csd_split":
            split_cfg = calib_cfg.get("csd_split") or {}
            csd_full = load_csd(cfg)
            _, csd = split_csd_for_calibration(
                csd_full,
                strategy=str(split_cfg.get("strategy", "alternating")),
                calib_share=float(split_cfg.get("calib_share", 0.65)),
                random_seed=int(split_cfg.get("random_seed", 42)),
            )
            logger.info(f"  Using CSD validation subset ({len(csd)} sections)")
        else:
            csd = load_csd(cfg)

        csd_agg = aggregate_csd_by_class(csd)
        report["csd_observed"] = csd_agg.to_dict(orient="records")

        if vol_col and vol_col in links_gdf.columns:
            model_agg = aggregate_model_by_class(links_gdf, vol_col)
            report["csd_modeled"] = model_agg.to_dict(orient="records")

        csd_match_df = match_csd_to_links(csd, links_gdf)
        if not csd_match_df.empty:
            report["csd_link_matching"] = csd_match_df.to_dict(orient="records")

        logger.info(f"  CSD: {len(csd)} sections")
        for _, r in csd_agg.iterrows():
            logger.info(f"    {r['road_class']:12s}  sections={int(r['sections']):4d}  "
                  f"mean_AADT={r['mean_sv']:>8.0f}  mean_cars={r['mean_o']:>8.0f}")

        if not csd_match_df.empty:
            logger.info(f"  Per-road comparison: {len(csd_match_df)} roads matched")
            for _, r in csd_match_df.iterrows():
                logger.info(
                    f"    {r['road']:>8s} ({r['road_class']:>10s})  "
                    f"csd_sv={r['csd_mean_sv']:>8.0f}  model={r['model_lw_mean']:>8.0f}  "
                    f"GEH={r['geh']:>5.1f}  sections={r['csd_sections']}  links={r['model_links']}"
                )
            summary = getattr(csd_match_df, "attrs", {}).get("summary")
            if summary:
                logger.info(f"  Summary ({summary['n_roads']} roads): "
                      f"R²={summary['r2']:.3f}  bias={summary['bias_pct']:.1f}%  "
                      f"%RMSE={summary['pct_rmse']:.1f}  mean_GEH={summary['mean_geh']:.1f}")
                report["csd_summary"] = summary
    except Exception:
        logger.exception("CSD independent validation step skipped")

    # 3) Screenline validation
    logger.info("\n3) Screenline validation ...")
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
                logger.info(f"  {sn}: mod={sr.modeled_total:,.0f} obs={sr.observed_total:,.0f} "
                      f"ratio={sr.ratio:.2f} GEH={sr.geh:.1f}")
            report["screenlines"] = sl_results
        else:
            logger.warning("  No screenlines defined")
    except Exception:
        logger.exception("Screenline validation step skipped")

    # 4) Journey time validation
    logger.info("\n4) Journey time validation ...")
    jt_results: List[Dict[str, Any]] = []
    speed_comparison: Dict[str, Any] = {}
    try:
        # Class-level speed comparison
        if csd is not None and not csd.empty:
            speed_comparison = compute_class_speed_comparison(links_gdf)
            report["class_speed_comparison"] = speed_comparison
            if speed_comparison:
                logger.info("  Speed comparison (model vs CSD):")
                for lt, sc in speed_comparison.items():
                    logger.info(f"    {lt:20s}  model={sc['modeled_kmh']:>5.1f}  "
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
                        logger.info(f"  {r.get('name', '?'):20s}  model={r.get('model_min', '?')}min  "
                              f"ref={r.get('ref_min', '?')}min  {status}")
                else:
                    mat.close()
            else:
                logger.warning("  No skims available -- skip route checks")
        else:
            logger.info("  No reference routes configured")
    except Exception:
        logger.exception("Journey time validation step skipped")

    # 5) Benchmark summary
    logger.info(f"\n5) Validation benchmarks (model_time_period={model_time_period}) ...")
    daily_conv = _get(cfg, ["calibration", "convergence", "daily"], {})
    benchmarks = compute_validation_benchmarks(
        pent_stats, sl_results, jt_results,
        model_time_period=model_time_period,
        daily_thresholds=daily_conv,
    )
    report["benchmarks"] = benchmarks
    qcfg = (calib_cfg.get("quality_gates") or {})
    q_bias_hard = float(qcfg.get("hard_class_bias_max_abs_pct", 90.0))
    q_wmape_warn = float(qcfg.get("warn_wmape_pct", 47.0))
    q_geh_warn = float(qcfg.get("warn_geh_lt5_pct", 7.0))
    ext = report.get("extended_metrics") or {}
    bias = float(ext.get("class_bias_max_abs_pct", 0.0)) if ext else None
    wmape = float(ext.get("wmape_pct", 0.0)) if ext else None
    geh = float(benchmarks.get("geh_lt5_pct", 0.0))
    q_warns: List[str] = []
    if wmape is not None and wmape > q_wmape_warn:
        q_warns.append(f"wmape_pct>{q_wmape_warn:g}")
    if geh < q_geh_warn:
        q_warns.append(f"geh_lt5_pct<{q_geh_warn:g}")
    report["quality_gates"] = {
        "hard_class_bias_max_abs_pct": q_bias_hard,
        "warn_wmape_pct": q_wmape_warn,
        "warn_geh_lt5_pct": q_geh_warn,
        "hard_reject": bool((bias is not None) and (bias > q_bias_hard)),
        "warnings": q_warns,
    }
    overall = "PASS" if benchmarks["overall_pass"] else "FAIL"
    if model_time_period == "daily":
        dt = daily_conv or {}
        r2_tgt = float(dt.get("r2_target", 0.80))
        sl_range = dt.get("slope_range", [0.85, 1.15])
        prmse_max = float(dt.get("pct_rmse_max", 35.0))
        bias_max = float(dt.get("bias_abs_max_pct", 15.0))
        sl_max = float(dt.get("screenline_max_pct_deviation", 15.0))
        logger.info(f"  R² >= {r2_tgt}:     {benchmarks.get('daily_r2', 0):.4f}  "
              f"{'PASS' if benchmarks.get('daily_r2_pass') else 'FAIL'}")
        logger.info(f"  slope [{sl_range[0]}-{sl_range[1]}]: {benchmarks.get('daily_slope', 0):.4f}  "
              f"{'PASS' if benchmarks.get('daily_slope_pass') else 'FAIL'}")
        logger.info(f"  %RMSE <= {prmse_max:g}%:   {benchmarks.get('daily_pct_rmse', 0):.1f}%  "
              f"{'PASS' if benchmarks.get('daily_pct_rmse_pass') else 'FAIL'}")
        logger.info(f"  |bias| <= {bias_max:g}%:  {benchmarks.get('daily_bias_abs_pct', 0):.2f}%  "
              f"{'PASS' if benchmarks.get('daily_bias_pass') else 'FAIL'}")
        logger.info(f"  SL dev <= {sl_max:g}%:  {benchmarks.get('screenline_max_error_pct', 0):.1f}%  "
              f"{'PASS' if benchmarks.get('daily_screenline_pass') else 'FAIL'}")
        logger.info(f"  (GEH<5: {geh:.1f}% — diagnostic only for daily model)")
    else:
        logger.info(f"  GEH<5 >= 85%:  {geh:.1f}%  "
              f"{'PASS' if benchmarks.get('geh_benchmark_pass_hourly') else 'FAIL'}")
    if benchmarks["jt_benchmark_pass"] is not None:
        logger.info(f"  JT within tol:  {benchmarks['jt_within_tolerance_pct']:.1f}%  "
              f"{'PASS' if benchmarks['jt_benchmark_pass'] else 'FAIL'}")
    logger.info(f"  Screenline max error: {benchmarks['screenline_max_error_pct']:.1f}%")
    logger.info(f"  OVERALL: {overall}")

    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"\nValidation report: {report_path}")

    # Strip closures and run a clean final assignment
    if bc_cfg.get("enabled", False) and valid_period:
        from sim.network_normalization import swap_db_closures
        from sim.assignment import run_assignment

        logger.info("\n=== POST-VALIDATION: stripping closures, clean assignment ===")
        swap_db_closures(config_path, measurement_period=None)
        run_assignment(config_path)
        logger.info("  Clean (closure-free) assignment saved.")


# ---------------------------------------------------------------------------
# Lightweight diagnostics refresh (no assignment, no calibration)
# ---------------------------------------------------------------------------

def run_match_diagnostics(config_path: str | Path = "config/sim.yaml") -> None:
    """Regenerate matching_diagnostics.csv from existing assignment results.

    This is a fast (~seconds) step that re-matches the cleaned pentlogram
    data against the current assignment results and exports the diagnostics
    CSV used by the frontend's bias / corridor panels.  It does **not**
    re-run traffic assignment or OD scaling.
    """
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    _ensure_dir(output_dir)

    buffer_m = float(calib_cfg.get("match_buffer_m", 50.0))
    direction_aware = bool(calib_cfg.get("match_direction_aware", True))
    conflict_res = str(calib_cfg.get("match_conflict_resolution", "nearest"))
    agg_corridor = bool(calib_cfg.get("aggregate_corridor", True))

    count_target = str(calib_cfg.get("count_target", "total"))
    _ct_map = {"car_only": "observed_car", "motor_total": "observed_motor_total", "total": "observed_total"}
    obs_col = _ct_map.get(count_target, "observed_total")

    assign_cfg = cfg.get("assignment") or {}
    bpr_cfg = assign_cfg.get("bpr") or {}
    daily_cap_factor = resolve_daily_cap_factor_default(bpr_cfg)

    logger.info("=== MATCH DIAGNOSTICS (lightweight refresh) ===")

    results_path = output_dir / "assignment_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No assignment results at {results_path}. Run 'calibrate' or 'assign' first."
        )
    vol_df = pd.read_parquet(str(results_path))

    if "total_vehicles_tot" in vol_df.columns and vol_df["total_vehicles_tot"].sum() > 0:
        vol_col = "total_vehicles_tot"
    else:
        vol_col = _detect_volume_col(vol_df)
        class_tot_cols = [
            c for c in vol_df.columns
            if c.endswith("_tot")
            and c not in ("PCE_tot", "Preload_tot", "total_vehicles_tot")
            and vol_df[c].sum() > 0
        ]
        if len(class_tot_cols) > 1:
            vol_df["total_vehicles_tot"] = vol_df[class_tot_cols].sum(axis=1)
            vol_col = "total_vehicles_tot"

    logger.info(f"  Loaded assignment: {len(vol_df)} links, vol_col={vol_col}")

    pent = load_pentlogram(cfg)
    validate_geometries_or_fail(
        pent, name="pentlogram", expected_epsg=get_metric_epsg(cfg),
    )
    logger.info(f"  Pentlogram: {len(pent)} segments (after cleaning)")

    links_gdf = _load_network_links(project_dir)
    if vol_col and "link_id" in vol_df.columns:
        links_gdf = links_gdf.merge(vol_df[["link_id", vol_col]], on="link_id", how="left")

    matched = match_counts_to_links(
        pent, links_gdf,
        buffer_m=buffer_m,
        direction_aware=direction_aware,
        conflict_resolution=conflict_res,
        aggregate_corridor=agg_corridor,
        vol_col=vol_col,
    )

    compare_col = "_corridor_volume" if "_corridor_volume" in matched.columns else vol_col
    _export_matching_diagnostics(matched, compare_col, obs_col, output_dir)

    if compare_col and compare_col in matched.columns:
        valid = matched.dropna(subset=[compare_col, obs_col])
        valid = valid[valid[obs_col] > 0]
        if "_excluded" in valid.columns:
            valid = valid[~valid["_excluded"]].copy()
        stats = compute_stats(
            valid[compare_col].values, valid[obs_col].values,
            daily_capacity_factor=daily_cap_factor,
        )
        logger.info(f"  Matched: {len(valid)}  R²={stats.get('r2')}  slope={stats.get('slope')}  "
              f"%RMSE={stats.get('pct_rmse')}  bias={stats.get('bias_pct')}%")
        logger.info(f"  GEH<5: {stats.get('geh_lt5_pct')}%")

    mq = match_quality_report(matched)
    logger.info(f"  Match quality: {mq['n_matched']}/{mq['n_total']} matched, "
          f"{mq['n_link_conflicts']} conflicts, "
          f"mean_dist={mq['mean_match_distance_m']}m")
    logger.info(f"  Done. Diagnostics CSV: {output_dir / 'matching_diagnostics.csv'}")
