"""Event-links fetcher: causal relationships between traffic events.

The ``event_links`` table connects restrictions, accidents, and alerts
to the traffic jams they caused.  This module fetches those links and
produces per-restriction impact statistics -- how many jams, total delay,
etc. -- directly usable for scenario validation and experimental design.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sim.defaults import SIM_DEFAULTS
from sim.datasets.utils import (
    coerce_object_columns_for_parquet,
    ensure_dir,
    load_aoi_bbox_wgs84,
)

logger = logging.getLogger(__name__)


def fetch_postgres_event_links(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    force: bool = False,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Fetch event_links and compute per-restriction jam impact statistics.

    Produces two parquet files:

    * ``event_links.parquet`` -- raw causal links between events
    * ``restriction_impact.parquet`` -- per-restriction aggregate:
      jam count, total delay, max delay, mean jam speed, etc.
    """
    import psycopg2

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    ensure_dir(cache_dir)
    links_path = cache_dir / "event_links.parquet"
    impact_path = cache_dir / "restriction_impact.parquet"

    if links_path.exists() and impact_path.exists() and not force:
        logger.info("Using cached %s", links_path)
        df = pd.read_parquet(links_path)
        return {
            "event_links_parquet": str(links_path),
            "impact_parquet": str(impact_path),
            "event_links_count": int(len(df)),
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

    event_links_table = source_cfg.get("table", "event_links")

    # Fetch restriction -> traffic_jam causal links joined with jam details
    query = f"""
        SELECT
            el.id           AS link_id,
            el.source_type,
            el.source_id,
            el.target_type,
            el.target_id,
            el.link_type,
            el.confidence,
            el.description,
            j.first_seen    AS jam_first_seen,
            j.last_seen     AS jam_last_seen,
            j.delay_seconds AS jam_delay_s,
            j.length_m      AS jam_length_m,
            j.speed_kmh     AS jam_speed_kmh,
            j.speed_normal_kmh AS jam_speed_normal_kmh,
            j.severity      AS jam_severity,
            j.segment_id    AS jam_segment_id,
            j.city          AS jam_city,
            j.road_number   AS jam_road_number
        FROM {event_links_table} el
        LEFT JOIN traffic_jams j
            ON el.target_type = 'traffic_jam' AND el.target_id = j.id
        WHERE el.link_type = 'caused_by'
          AND el.source_type IN ('restriction', 'accident', 'alert')
        ORDER BY el.source_type, el.source_id
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

    if df.empty:
        logger.info("PostgreSQL event_links: no rows returned")
        return {
            "event_links_count": 0,
            "event_links_parquet": str(links_path),
            "impact_parquet": str(impact_path),
        }

    logger.info("PostgreSQL event_links: fetched %d causal links", len(df))

    for col in ("jam_delay_s", "jam_length_m", "jam_speed_kmh", "jam_speed_normal_kmh"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["jam_first_seen"] = pd.to_datetime(df["jam_first_seen"], errors="coerce", utc=True)
    df["jam_last_seen"] = pd.to_datetime(df["jam_last_seen"], errors="coerce", utc=True)
    df["jam_duration_min"] = (
        (df["jam_last_seen"] - df["jam_first_seen"]).dt.total_seconds() / 60.0
    )

    out_df = coerce_object_columns_for_parquet(df)
    out_df.to_parquet(links_path, index=False)
    logger.info("Wrote %d event links to %s", len(out_df), links_path)

    # --- Per-source impact statistics ---
    # Focus on restrictions -> jams (most relevant for closure experiments)
    restr = df[df["source_type"] == "restriction"].copy()
    if not restr.empty:
        impact = restr.groupby("source_id").agg(
            jams_caused=("target_id", "nunique"),
            total_delay_s=("jam_delay_s", "sum"),
            max_delay_s=("jam_delay_s", "max"),
            mean_delay_s=("jam_delay_s", "mean"),
            total_jam_length_m=("jam_length_m", "sum"),
            mean_jam_speed_kmh=("jam_speed_kmh", "mean"),
            mean_jam_duration_min=("jam_duration_min", "mean"),
            max_jam_duration_min=("jam_duration_min", "max"),
            jam_cities=("jam_city", lambda x: "; ".join(sorted(set(str(v) for v in x.dropna())))),
        ).reset_index()
        impact.rename(columns={"source_id": "restriction_id"}, inplace=True)

        impact = coerce_object_columns_for_parquet(impact)
        impact.to_parquet(impact_path, index=False)
        logger.info(
            "Wrote impact stats for %d restrictions to %s "
            "(median jams per restriction: %.0f, median total delay: %.0fs)",
            len(impact), impact_path,
            float(impact["jams_caused"].median()),
            float(impact["total_delay_s"].median()),
        )
    else:
        logger.info("No restriction -> jam links found")
        pd.DataFrame().to_parquet(impact_path)

    # Summary by source type
    by_type = df.groupby("source_type").agg(
        n_links=("link_id", "count"),
        unique_sources=("source_id", "nunique"),
        unique_targets=("target_id", "nunique"),
    ).to_dict(orient="index")

    return {
        "event_links_parquet": str(links_path),
        "impact_parquet": str(impact_path),
        "event_links_count": int(len(df)),
        "by_source_type": by_type,
    }
