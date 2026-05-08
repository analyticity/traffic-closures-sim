"""Traffic jam data fetcher: PostgreSQL traffic_jams + road_segments tables.

Fetches observed jam records and joins them with road_segments to produce
a parquet with measured free-flow speeds (``speed_normal_kmh``) and
jam statistics per OSM segment -- directly usable for network speed
calibration and VDF validation.
"""
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


def fetch_postgres_jams(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    force: bool = False,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Fetch traffic jam records from the PostgreSQL ``traffic_jams`` table.

    Optionally joins with ``road_segments`` to attach OSM IDs and road
    metadata.  Produces two parquet files:

    * ``jams.parquet`` -- individual jam observations
    * ``jams_segment_stats.parquet`` -- per-segment aggregated statistics
      (median speed, free-flow speed, jam frequency, etc.)
    """
    import psycopg2

    if gpd is None:
        raise RuntimeError("geopandas is required for postgres_jams provider")

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    ensure_dir(cache_dir)
    jams_path = cache_dir / "jams.parquet"
    stats_path = cache_dir / "jams_segment_stats.parquet"

    if jams_path.exists() and stats_path.exists() and not force:
        logger.info("Using cached %s", jams_path)
        df = pd.read_parquet(jams_path)
        return {
            "jams_parquet": str(jams_path),
            "stats_parquet": str(stats_path),
            "jams_count": int(len(df)),
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

    jams_table = source_cfg.get("table", "traffic_jams")
    segments_table = source_cfg.get("segments_table", "road_segments")

    query = f"""
        SELECT
            j.id,
            j.first_seen,
            j.last_seen,
            j.city,
            j.street_name,
            j.road_number,
            j.road_type_code,
            j.delay_seconds,
            j.length_m,
            j.speed_kmh,
            j.speed_normal_kmh,
            j.severity,
            j.quality_score,
            j.segment_id,
            s.osm_id,
            s.road_ref   AS segment_road_ref,
            s.road_class  AS segment_road_class,
            s.max_speed   AS segment_max_speed,
            ST_Y(ST_Centroid(j.jam_line_geog::geometry)) AS lat,
            ST_X(ST_Centroid(j.jam_line_geog::geometry)) AS lon
        FROM {jams_table} j
        LEFT JOIN {segments_table} s ON j.segment_id = s.id
        WHERE j.jam_line_geog IS NOT NULL
        ORDER BY j.id
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

    if df.empty:
        logger.info("PostgreSQL jams: no rows returned")
        return {"jams_count": 0, "jams_parquet": str(jams_path), "stats_parquet": str(stats_path)}

    logger.info("PostgreSQL jams: fetched %d rows", len(df))

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
        logger.info("PostgreSQL jams: %d total -> %d in model area", before, len(df))

    if df.empty:
        logger.info("PostgreSQL jams: no features in model area")
        return {"jams_count": 0, "jams_parquet": str(jams_path), "stats_parquet": str(stats_path)}

    for col in ("delay_seconds", "length_m", "speed_kmh", "speed_normal_kmh"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "osm_id" in df.columns:
        df["osm_id"] = pd.to_numeric(df["osm_id"], errors="coerce").fillna(-1).astype(int)
    if "segment_id" in df.columns:
        df["segment_id"] = pd.to_numeric(df["segment_id"], errors="coerce").fillna(-1).astype(int)
    if "segment_max_speed" in df.columns:
        df["segment_max_speed"] = pd.to_numeric(df["segment_max_speed"], errors="coerce")
    if "quality_score" in df.columns:
        df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0).astype(int)

    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce", utc=True)
    df["last_seen"] = pd.to_datetime(df["last_seen"], errors="coerce", utc=True)
    df["hour"] = df["first_seen"].dt.hour
    df["weekday"] = df["first_seen"].dt.weekday  # 0=Mon

    out_df = coerce_object_columns_for_parquet(df)
    out_df.to_parquet(jams_path, index=False)
    logger.info("Wrote %d jam records to %s", len(out_df), jams_path)

    # --- Per-segment aggregated statistics ---
    valid = df[df["speed_normal_kmh"].notna() & (df["speed_normal_kmh"] > 0)].copy()
    if not valid.empty:
        agg = valid.groupby("segment_id").agg(
            osm_id=("osm_id", "first"),
            road_type_code=("road_type_code", "first"),
            segment_road_class=("segment_road_class", "first"),
            segment_max_speed=("segment_max_speed", "first"),
            segment_road_ref=("segment_road_ref", "first"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            jam_count=("id", "count"),
            speed_normal_median=("speed_normal_kmh", "median"),
            speed_normal_p25=("speed_normal_kmh", lambda x: x.quantile(0.25)),
            speed_normal_p75=("speed_normal_kmh", lambda x: x.quantile(0.75)),
            speed_jam_median=("speed_kmh", "median"),
            delay_median_s=("delay_seconds", "median"),
            delay_max_s=("delay_seconds", "max"),
            length_median_m=("length_m", "median"),
        ).reset_index()

        agg = coerce_object_columns_for_parquet(agg)
        agg.to_parquet(stats_path, index=False)
        logger.info(
            "Wrote segment stats for %d segments to %s "
            "(median free-flow speed: %.1f km/h)",
            len(agg), stats_path, float(agg["speed_normal_median"].median()),
        )
    else:
        logger.warning("No valid speed_normal_kmh values; segment stats empty")
        pd.DataFrame().to_parquet(stats_path)

    sev_counts = df["severity"].value_counts().to_dict()
    return {
        "jams_parquet": str(jams_path),
        "stats_parquet": str(stats_path),
        "jams_count": int(len(df)),
        "severity_counts": sev_counts,
        "segments_with_stats": int(len(valid["segment_id"].unique())) if not valid.empty else 0,
    }
