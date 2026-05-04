"""Baseline road closures: load, apply, strip, and swap.

Closures reduce capacity and speed of matched network links.
Pre-closure values are stored in ``_preclosure_*`` DB columns for
later restoration by :func:`strip_closures`.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
from aequilibrae import Project

from sim.io_project import get_metric_epsg, load_config
from sim.network.connectivity import check_connectivity
from sim.network.db import project_db, project_db_path
from sim.network.export import export_stable_network

logger = logging.getLogger(__name__)

_PRECLOSURE_COLS = frozenset({
    "_preclosure_capacity_ab", "_preclosure_capacity_ba",
    "_preclosure_speed_ab", "_preclosure_speed_ba",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _closure_overlaps_period(
    closure: Dict[str, Any],
    period_start: str,
    period_end: str,
) -> bool:
    """Check whether a closure's [start, end] overlaps [period_start, period_end]."""
    c_start = closure.get("start") or "1900-01-01"
    c_end = closure.get("end") or c_start
    return c_start <= period_end and c_end >= period_start


def _safe_float(val: Any, fallback: float = 50.0) -> float:
    v = float(val)
    return v if v == v else fallback  # NaN != NaN


def _strip_preclosure_columns(conn: sqlite3.Connection) -> int:
    """Restore pre-closure values and drop ``_preclosure_*`` columns.

    Returns the number of restored rows.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(links)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if not _PRECLOSURE_COLS.issubset(existing_cols):
        return -1

    cur.execute(
        "UPDATE links SET "
        "  capacity_ab = _preclosure_capacity_ab, "
        "  capacity_ba = _preclosure_capacity_ba, "
        "  speed_ab = _preclosure_speed_ab, "
        "  speed_ba = _preclosure_speed_ba, "
        "  travel_time_ab = CASE WHEN _preclosure_speed_ab > 0 "
        "    THEN distance * 3.6 / _preclosure_speed_ab ELSE travel_time_ab END, "
        "  travel_time_ba = CASE WHEN _preclosure_speed_ba > 0 "
        "    THEN distance * 3.6 / _preclosure_speed_ba ELSE travel_time_ba END "
        "WHERE _preclosure_capacity_ab IS NOT NULL"
    )
    restored = cur.rowcount
    for col in _PRECLOSURE_COLS:
        try:
            cur.execute(f"ALTER TABLE links DROP COLUMN {col}")
        except Exception:
            pass
    return restored


def _ensure_preclosure_columns(conn: sqlite3.Connection) -> None:
    """Add ``_preclosure_*`` columns if they do not exist yet."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(links)")
    existing = {row[1] for row in cur.fetchall()}
    for pcol in _PRECLOSURE_COLS:
        if pcol not in existing:
            cur.execute(f"ALTER TABLE links ADD COLUMN {pcol} REAL")


def _write_closure_state_to_db(
    conn: sqlite3.Connection,
    links_gdf: pd.DataFrame,
) -> None:
    """Persist closure-modified attributes and pre-closure backup to the DB."""
    _ensure_preclosure_columns(conn)
    cur = conn.cursor()
    for _, row in links_gdf.iterrows():
        cap_ab = _safe_float(row["capacity_ab"])
        cap_ba = _safe_float(row["capacity_ba"])
        spd_ab = _safe_float(row["speed_ab"], 5.0)
        spd_ba = _safe_float(row["speed_ba"], 5.0)
        tt_ab = _safe_float(row["travel_time_ab"], 0.01)
        tt_ba = _safe_float(row["travel_time_ba"], 0.01)
        pc_cap_ab = _safe_float(row.get("_preclosure_capacity_ab", cap_ab), cap_ab)
        pc_cap_ba = _safe_float(row.get("_preclosure_capacity_ba", cap_ba), cap_ba)
        pc_spd_ab = _safe_float(row.get("_preclosure_speed_ab", spd_ab), spd_ab)
        pc_spd_ba = _safe_float(row.get("_preclosure_speed_ba", spd_ba), spd_ba)
        cur.execute(
            "UPDATE links SET capacity_ab=?, capacity_ba=?, speed_ab=?, speed_ba=?, "
            "travel_time_ab=?, travel_time_ba=?, "
            "_preclosure_capacity_ab=?, _preclosure_capacity_ba=?, "
            "_preclosure_speed_ab=?, _preclosure_speed_ba=? "
            "WHERE link_id=?",
            (cap_ab, cap_ba, spd_ab, spd_ba, tt_ab, tt_ba,
             pc_cap_ab, pc_cap_ba, pc_spd_ab, pc_spd_ba,
             int(row["link_id"])),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_closures(
    source_path: Path,
    *,
    measurement_period: Optional[Dict[str, str]] = None,
    status_whitelist: Optional[list] = None,
) -> List[Dict[str, Any]]:
    """Load closures from parquet or legacy JSON.

    Returns a list of dicts, each with at least ``lon``, ``lat``,
    ``severity``, and optionally ``start``, ``end``, ``road_ref``.
    """
    if not source_path.exists():
        logger.info("Closures file not found: %s, skipping", source_path)
        return []

    suffix = source_path.suffix.lower()
    if suffix == ".parquet":
        try:
            gdf = gpd.read_parquet(source_path)
        except Exception:
            gdf = pd.read_parquet(source_path)
            gdf = gpd.GeoDataFrame(gdf)
        if gdf.empty:
            logger.info("Closures parquet is empty")
            return []
        closures = gdf.to_dict(orient="records")
    elif suffix == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        closures = data.get("closures", [])
    else:
        logger.warning("Unsupported closure file format: %s", suffix)
        return []

    if not closures:
        logger.info("Closures file is empty")
        return []

    total = len(closures)
    if measurement_period:
        p_start = measurement_period.get("start", "1900-01-01")
        p_end = measurement_period.get("end", "2099-12-31")
        closures = [c for c in closures if _closure_overlaps_period(c, p_start, p_end)]
        logger.info("Loaded %d closures, %d overlap measurement period %s..%s", total, len(closures), p_start, p_end)
    else:
        logger.info("Loaded %d closures (no temporal filter)", total)

    if status_whitelist:
        allowed = {str(s).strip().lower() for s in status_whitelist}
        before = len(closures)
        closures = [c for c in closures if str(c.get("status", "")).strip().lower() in allowed]
        logger.info("Status filter (%s): %d -> %d closures", ", ".join(sorted(allowed)), before, len(closures))

    return closures


def apply_baseline_closures(
    links: pd.DataFrame,
    closures: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Match closures to network links and reduce capacity/speed.

    Stores pre-closure values in ``_preclosure_*`` columns for later
    restoration by :func:`strip_closures`.
    """
    from shapely.geometry import Point

    bc_cfg = cfg.get("baseline_closures") or {}
    match_cfg = bc_cfg.get("matching") or {}
    severity_map = bc_cfg.get("severity_map") or {}
    max_dist_m = float(match_cfg.get("max_distance_m", 50))
    require_ref = bool(match_cfg.get("require_road_ref_match", False))
    metric_epsg = int(cfg.get("crs_epsg", 5514))

    if not closures:
        return links

    for col in ("_preclosure_capacity_ab", "_preclosure_capacity_ba",
                "_preclosure_speed_ab", "_preclosure_speed_ba"):
        src = col.replace("_preclosure_", "")
        if col not in links.columns:
            links[col] = links[src].copy()
        else:
            nan_mask = links[col].isna()
            if nan_mask.any():
                links.loc[nan_mask, col] = links.loc[nan_mask, src]

    closure_pts = gpd.GeoDataFrame(
        closures,
        geometry=[Point(float(c.get("lon", 0)), float(c.get("lat", 0))) for c in closures],
        crs="EPSG:4326",
    ).to_crs(epsg=metric_epsg)

    link_gdf = _prepare_link_gdf(links, metric_epsg)
    link_ref_col = "osm_ref_norm" if (link_gdf is not None and "osm_ref_norm" in link_gdf.columns) else None

    affected_link_ids: Dict[int, str] = {}

    if link_gdf is not None and not link_gdf.geometry.isna().all():
        sindex = link_gdf.sindex
        for _, cpt in closure_pts.iterrows():
            geom = cpt.geometry
            if geom is None or geom.is_empty:
                continue
            sev = str(cpt.get("severity", "lane_reduction"))
            closure_ref = str(cpt.get("road_ref", "") or "").strip()

            candidates = sindex.query(geom.buffer(max_dist_m), predicate="intersects")
            if len(candidates) == 0:
                nn = sindex.nearest(geom, max_distance=max_dist_m)
                candidates = nn[1] if nn.ndim == 2 and nn.shape[0] == 2 else nn.ravel()

            for cidx in candidates:
                row = link_gdf.iloc[int(cidx)]
                lid = int(row["link_id"])
                if closure_ref and link_ref_col:
                    link_ref = str(row.get(link_ref_col, "") or "").strip()
                    if require_ref and closure_ref and link_ref and closure_ref != link_ref:
                        continue
                if lid not in affected_link_ids or sev == "full":
                    affected_link_ids[lid] = sev
    else:
        logger.warning("No link geometries available for spatial closure matching")

    if not affected_link_ids:
        if link_gdf is not None and len(closure_pts) > 0:
            lb = link_gdf.total_bounds
            cb = closure_pts.total_bounds
            logger.debug(
                "No closures matched (link bounds [%.0f,%.0f,%.0f,%.0f], "
                "closure bounds [%.0f,%.0f,%.0f,%.0f], buffer %sm)",
                lb[0], lb[1], lb[2], lb[3], cb[0], cb[1], cb[2], cb[3], max_dist_m,
            )
        else:
            logger.info("No closures matched to network links")
        links.attrs["closure_manifest"] = {
            "n_closures_loaded": len(closures),
            "n_closures_matched": 0,
            "n_links_affected": 0,
            "affected_links": [],
        }
        return links

    for col in ("speed_ab", "speed_ba", "capacity_ab", "capacity_ba",
                "_preclosure_speed_ab", "_preclosure_speed_ba",
                "_preclosure_capacity_ab", "_preclosure_capacity_ba"):
        if col in links.columns:
            links[col] = links[col].astype(float)

    for lid, sev in affected_link_ids.items():
        sev_cfg = severity_map.get(sev, severity_map.get("lane_reduction", {}))
        cap_factor = float(sev_cfg.get("capacity_factor", 0.5))
        spd_factor = float(sev_cfg.get("speed_factor", 0.7))
        mask = links["link_id"] == lid
        if not mask.any():
            continue
        links.loc[mask, "capacity_ab"] = links.loc[mask, "_preclosure_capacity_ab"] * cap_factor
        links.loc[mask, "capacity_ba"] = links.loc[mask, "_preclosure_capacity_ba"] * cap_factor
        links.loc[mask, "speed_ab"] = links.loc[mask, "_preclosure_speed_ab"] * spd_factor
        links.loc[mask, "speed_ba"] = links.loc[mask, "_preclosure_speed_ba"] * spd_factor

        for tt_col, spd_col in [("travel_time_ab", "speed_ab"), ("travel_time_ba", "speed_ba")]:
            sub = mask & (links[spd_col] > 0)
            if sub.any():
                links.loc[sub, tt_col] = links.loc[sub, "distance"] * 3.6 / links.loc[sub, spd_col]

    min_cap, min_spd = 50.0, 5.0
    for col in ("capacity_ab", "capacity_ba"):
        n_nan = int(links[col].isna().sum())
        if n_nan:
            links[col] = links[col].fillna(min_cap)
            logger.warning("Filled %d NaN values in %s with %s", n_nan, col, min_cap)
    for col in ("speed_ab", "speed_ba"):
        n_nan = int(links[col].isna().sum())
        if n_nan:
            links[col] = links[col].fillna(min_spd)
            logger.warning("Filled %d NaN values in %s with %s", n_nan, col, min_spd)

    full_count = sum(1 for s in affected_link_ids.values() if s == "full")
    logger.info(
        "Applied %d baseline closures (%d full, %d partial)",
        len(affected_link_ids), full_count, len(affected_link_ids) - full_count,
    )

    manifest_links = []
    for lid, sev in affected_link_ids.items():
        row_mask = links["link_id"] == lid
        if not row_mask.any():
            continue
        row = links.loc[row_mask].iloc[0]
        entry: Dict[str, Any] = {"link_id": int(lid), "severity": sev}
        if "_preclosure_capacity_ab" in links.columns:
            entry["preclosure_cap_ab"] = round(float(row["_preclosure_capacity_ab"]), 1)
            entry["postclosure_cap_ab"] = round(float(row["capacity_ab"]), 1)
        manifest_links.append(entry)
    links.attrs["closure_manifest"] = {
        "n_closures_loaded": len(closures),
        "n_closures_matched": len(affected_link_ids),
        "n_links_affected": len(affected_link_ids),
        "n_full": full_count,
        "n_partial": len(affected_link_ids) - full_count,
        "affected_links": manifest_links,
    }

    return links


def strip_closures(
    config_path: str | Path = "config/brno/sim.yaml",
) -> None:
    """Restore pre-closure capacity/speed/travel_time in the project DB.

    This is the inverse of :func:`apply_baseline_closures`.  It reads
    ``_preclosure_*`` columns, writes back the original values, removes
    the columns, and re-exports the network.
    """
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    db_path = project_db_path(project_dir)
    if not db_path.is_file():
        logger.warning("Project DB not found at %s", db_path)
        return

    with project_db(project_dir) as conn:
        restored = _strip_preclosure_columns(conn)

    if restored == -1:
        logger.info("No _preclosure columns (closures not applied or already stripped)")
        return

    network_cfg = cfg.get("network") or {}
    outputs_dir = network_cfg.get("output_dir", "outputs/baseline/network")
    project = Project()
    project.open(str(project_dir))
    try:
        links_data = project.network.links.data
        crs_epsg = get_metric_epsg(cfg)
        connectivity_info = check_connectivity(project)
        export_stable_network(
            project, Path(outputs_dir), connectivity_info,
            normalized_links=links_data, output_crs_epsg=crs_epsg, cfg=cfg,
        )
    finally:
        project.close()

    logger.info("Stripped closures: restored %d links; re-exported network to %s", restored, outputs_dir)

    demand_cfg = cfg.get("demand") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    if matrix_path.is_file():
        logger.info("Re-running assignment after strip_closures")
        from sim.assignment import run_assignment
        run_assignment(config_path)
    else:
        logger.warning(
            "OD matrix not found at %s, skipping post-strip re-assignment; map may show stale volumes",
            matrix_path,
        )


def swap_db_closures(
    config_path: str | Path = "config/brno/sim.yaml",
    measurement_period: Optional[Dict[str, str]] = None,
) -> int:
    """Strip existing closures from the DB and optionally apply new ones.

    Returns the number of closures applied (0 when stripping only).
    """
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    db_path = project_db_path(project_dir)
    if not db_path.is_file():
        logger.warning("swap_db_closures: DB not found at %s", db_path)
        return 0

    with project_db(project_dir) as conn:
        restored = _strip_preclosure_columns(conn)
    if restored >= 0:
        logger.info("swap_db_closures: stripped closures from %d links", restored)
    else:
        logger.info("swap_db_closures: no existing closures to strip")

    if measurement_period is None:
        return 0

    bc_cfg = cfg.get("baseline_closures") or {}
    _cache = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))
    source_path = Path(bc_cfg.get("source_path", f"{_cache}/closures.parquet"))
    closures = load_closures(
        source_path,
        measurement_period=measurement_period,
        status_whitelist=bc_cfg.get("status_whitelist"),
    )
    if not closures:
        logger.info("swap_db_closures: no closures for period %s", measurement_period)
        return 0

    project = Project()
    project.open(str(project_dir))
    try:
        links = project.network.links.data.copy()
        crs_epsg = get_metric_epsg(cfg)
        links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=crs_epsg)
        links_gdf = apply_baseline_closures(links_gdf, closures, cfg)

        with project_db(project_dir) as conn:
            _write_closure_state_to_db(conn, links_gdf)
    finally:
        project.close()

    n = len(closures)
    n_affected = 0
    if "_preclosure_capacity_ab" in links_gdf.columns:
        n_affected = int((links_gdf["capacity_ab"] != links_gdf["_preclosure_capacity_ab"]).sum())
    logger.info(
        "swap_db_closures: applied %d closures for period %s (%d links affected)",
        n, measurement_period, n_affected,
    )
    return n


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_link_gdf(
    links: pd.DataFrame,
    metric_epsg: int,
) -> gpd.GeoDataFrame | None:
    """Reproject links to metric CRS, auto-detecting WGS84 mismatch."""
    if not (hasattr(links, "geometry") and links.geometry is not None and not links.geometry.isna().all()):
        return None

    link_gdf = gpd.GeoDataFrame(links, crs=metric_epsg if links.crs is None else links.crs)
    if link_gdf.crs is None:
        link_gdf = link_gdf.set_crs(epsg=metric_epsg)
    elif link_gdf.crs.to_epsg() != metric_epsg:
        link_gdf = link_gdf.to_crs(epsg=metric_epsg)

    sample_x = link_gdf.geometry.iloc[0].coords[0][0] if len(link_gdf) > 0 else 0
    sample_y = link_gdf.geometry.iloc[0].coords[0][1] if len(link_gdf) > 0 else 0
    if link_gdf.crs and link_gdf.crs.to_epsg() == metric_epsg and abs(sample_x) <= 180 and abs(sample_y) <= 90:
        logger.warning(
            "Link coords look like WGS84 (x=%.4f) but CRS is EPSG:%d; reprojecting from 4326",
            sample_x, metric_epsg,
        )
        link_gdf = link_gdf.set_crs(epsg=4326, allow_override=True).to_crs(epsg=metric_epsg)

    return link_gdf
