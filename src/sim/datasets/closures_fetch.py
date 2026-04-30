"""Closure data fetcher: PostgreSQL restrictions database."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from sim.defaults import SIM_DEFAULTS
from sim.datasets.utils import (
    coerce_object_columns_for_parquet,
    ensure_dir,
    load_aoi_bbox_wgs84,
)

logger = logging.getLogger(__name__)

_PG_FULL_CLOSURE_TYPES = frozenset({"road_closed", "roadClosed"})
_PG_LANE_REDUCTION_TYPES = frozenset({
    "laneClosures", "narrowLanes", "singleAlternateLineTraffic", "contraflow",
})


def _map_pg_severity(restriction_type: str, pg_severity: str) -> str:
    """Map PG ``restriction_type`` + ``severity`` to internal severity value."""
    if restriction_type in _PG_FULL_CLOSURE_TYPES:
        return "full"
    if restriction_type in _PG_LANE_REDUCTION_TYPES:
        return "lane_reduction"
    severity_map = {"standstill": "full", "serious": "lane_reduction", "moderate": "speed_limit"}
    return severity_map.get(pg_severity, "lane_reduction")


def fetch_postgres_closures(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    force: bool = False,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Fetch closure/restriction data from the PostgreSQL ``restrictions`` table.

    Produces a parquet file with the same schema consumed by
    :func:`sim.network.closures.load_closures` (``lon``, ``lat``,
    ``severity``, ``start``, ``end``, ``road_ref``, ``description_cs``).
    """
    import psycopg2

    if gpd is None:
        raise RuntimeError("geopandas is required for postgres_closures provider")

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    ensure_dir(cache_dir)
    cache_path = cache_dir / "closures.parquet"

    if cache_path.exists() and not force:
        logger.info("Using cached %s", cache_path)
        gdf = gpd.read_parquet(cache_path)
        return {
            "features": int(len(gdf)),
            "parquet": str(cache_path),
            "closures_parquet": str(cache_path),
            "closures_count": int(len(gdf)),
        }

    _db_defaults = SIM_DEFAULTS["datasets"]["closures_db"]
    db_cfg = cfg.get("closures_db") or {}
    conn = psycopg2.connect(
        host=db_cfg.get("host", _db_defaults["host"]),
        port=int(db_cfg.get("port", _db_defaults["port"])),
        dbname=db_cfg.get("dbname", _db_defaults["dbname"]),
        user=db_cfg.get("user", _db_defaults["user"]),
        password=db_cfg.get("password", _db_defaults["password"]),
    )

    table = source_cfg.get("table", "restrictions")

    query = f"""
        SELECT
            id,
            restriction_type,
            restriction_subtype,
            severity       AS pg_severity,
            status,
            road_number,
            street_name,
            city,
            description_cs,
            max_speed_kmh,
            valid_from,
            valid_to,
            first_seen,
            last_seen,
            ST_Y(location_point_geog::geometry) AS lat,
            ST_X(location_point_geog::geometry) AS lon
        FROM {table}
        WHERE location_point_geog IS NOT NULL
        ORDER BY id
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

    if df.empty:
        logger.info("PostgreSQL closures: no rows returned")
        return {"features": 0, "parquet": str(cache_path), "closures_count": 0}

    logger.info("PostgreSQL closures: fetched %d rows", len(df))

    bbox = load_aoi_bbox_wgs84(cfg)
    if bbox is not None:
        w, s, e, n = bbox
        margin = float(SIM_DEFAULTS["datasets"]["closures_db"]["bbox_margin_deg"])
        before = len(df)
        mask = (
            (df["lon"] >= w - margin) & (df["lon"] <= e + margin)
            & (df["lat"] >= s - margin) & (df["lat"] <= n + margin)
        )
        df = df[mask].copy()
        logger.info("PostgreSQL closures: %d total -> %d in model area", before, len(df))

    if df.empty:
        logger.info("PostgreSQL closures: no features in model area")
        return {"features": 0, "parquet": str(cache_path), "closures_count": 0}

    if "status" in df.columns:
        unique_statuses = sorted(df["status"].dropna().unique().tolist())
        logger.info("PostgreSQL closures: status values found: %s", unique_statuses)

    status_whitelist = source_cfg.get("status_whitelist")
    if status_whitelist and "status" in df.columns:
        allowed = {str(s).strip().lower() for s in status_whitelist}
        before_status = len(df)
        df = df[df["status"].fillna("").astype(str).str.strip().str.lower().isin(allowed)].copy()
        logger.info("PostgreSQL closures: status filter (%s): %d -> %d", ", ".join(sorted(allowed)), before_status, len(df))

    min_observed_days = source_cfg.get("min_observed_days")
    if min_observed_days is not None and "first_seen" in df.columns and "last_seen" in df.columns:
        min_days = float(min_observed_days)
        fs = pd.to_datetime(df["first_seen"], errors="coerce", utc=True)
        ls = pd.to_datetime(df["last_seen"], errors="coerce", utc=True)
        duration = (ls - fs).dt.total_seconds() / 86400.0
        before_dur = len(df)
        keep = duration.isna() | (duration >= min_days)
        df = df[keep].copy()
        logger.info("PostgreSQL closures: min_observed_days=%s: %d -> %d", min_days, before_dur, len(df))

    if df.empty:
        logger.info("PostgreSQL closures: no features after filtering")
        return {"features": 0, "parquet": str(cache_path), "closures_count": 0}

    df["severity"] = df.apply(
        lambda r: _map_pg_severity(str(r.get("restriction_type", "")), str(r.get("pg_severity", ""))),
        axis=1,
    )

    df["road_ref"] = df["road_number"].fillna("").astype(str).str.strip()

    for ts_col, fallback_col, out_col in [
        ("valid_from", "first_seen", "start"),
        ("valid_to", "last_seen", "end"),
    ]:
        primary = (
            pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            if ts_col in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        fallback = (
            pd.to_datetime(df[fallback_col], errors="coerce", utc=True)
            if fallback_col in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        merged = primary.fillna(fallback)
        df[out_col] = merged.dt.strftime("%Y-%m-%d").fillna("")

    if "description_cs" not in df.columns:
        df["description_cs"] = ""
    else:
        df["description_cs"] = df["description_cs"].fillna("")

    from shapely.geometry import Point

    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    full_count = int((gdf["severity"] == "full").sum())
    partial_count = int((gdf["severity"] == "lane_reduction").sum())
    speed_count = int((gdf["severity"] == "speed_limit").sum())
    logger.info(
        "PostgreSQL closures: %d events in model area (%d full, %d lane_reduction, %d speed_limit)",
        len(gdf), full_count, partial_count, speed_count,
    )

    gdf = coerce_object_columns_for_parquet(gdf)
    gdf.to_parquet(cache_path, index=False)

    return {
        "features": int(len(gdf)),
        "parquet": str(cache_path),
        "closures_parquet": str(cache_path),
        "closures_count": int(len(gdf)),
    }
