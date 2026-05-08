"""Road-segment data fetcher: PostgreSQL road_segments table.

Fetches the ``road_segments`` table which maps internal ``segment_id``
(used as FK by restrictions, traffic_jams, accidents, alerts) to OSM way
IDs (``osm_id``).  This enables direct ID-based matching between DB
events and the AequilibraE network instead of spatial proximity search.
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


def fetch_postgres_segments(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    force: bool = False,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Fetch road_segments with geometry from PostgreSQL.

    Produces ``road_segments.parquet`` with columns:
    ``id``, ``osm_id``, ``name``, ``road_ref``, ``road_class``,
    ``city``, ``max_speed``, ``lat``, ``lon``.
    """
    import psycopg2

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    ensure_dir(cache_dir)
    cache_path = cache_dir / "road_segments.parquet"

    if cache_path.exists() and not force:
        logger.info("Using cached %s", cache_path)
        df = pd.read_parquet(cache_path)
        return {
            "segments_parquet": str(cache_path),
            "segments_count": int(len(df)),
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

    table = source_cfg.get("table", "road_segments")

    query = f"""
        SELECT
            id,
            osm_id,
            name,
            road_ref,
            road_class,
            city,
            max_speed,
            ST_Y(ST_Centroid(geog::geometry)) AS lat,
            ST_X(ST_Centroid(geog::geometry)) AS lon
        FROM {table}
        WHERE geog IS NOT NULL
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
        logger.info("PostgreSQL road_segments: no rows returned")
        return {"segments_count": 0, "segments_parquet": str(cache_path)}

    logger.info("PostgreSQL road_segments: fetched %d rows", len(df))

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
        logger.info("PostgreSQL road_segments: %d total -> %d in model area", before, len(df))

    for col in ("osm_id", "max_speed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = coerce_object_columns_for_parquet(df)
    df.to_parquet(cache_path, index=False)

    return {
        "segments_parquet": str(cache_path),
        "segments_count": int(len(df)),
    }


def build_segment_to_link_map(
    segments_parquet: Path,
    network_links: pd.DataFrame,
) -> Dict[int, int]:
    """Build a mapping from DB segment_id to AequilibraE link_id via OSM ID.

    The AequilibraE network stores OSM way IDs in the ``osm_id`` column.
    The ``road_segments`` table maps ``id`` (segment_id) -> ``osm_id``.
    This function joins them to produce ``{segment_id: link_id}``.

    When multiple links share the same ``osm_id`` (split ways), the first
    one is returned.
    """
    if not segments_parquet.exists():
        logger.warning("Segments parquet not found: %s", segments_parquet)
        return {}

    seg_df = pd.read_parquet(segments_parquet)
    if seg_df.empty or "osm_id" not in seg_df.columns:
        return {}

    if "osm_id" not in network_links.columns:
        logger.warning("Network links have no osm_id column; cannot build segment map")
        return {}

    seg_df["osm_id"] = pd.to_numeric(seg_df["osm_id"], errors="coerce")
    seg_df = seg_df.dropna(subset=["osm_id"])
    seg_df["osm_id"] = seg_df["osm_id"].astype(int)

    net = network_links[["link_id", "osm_id"]].copy()
    net["osm_id"] = pd.to_numeric(net["osm_id"], errors="coerce")
    net = net.dropna(subset=["osm_id"])
    net["osm_id"] = net["osm_id"].astype(int)
    osm_to_link = dict(zip(net["osm_id"], net["link_id"]))

    result: Dict[int, int] = {}
    for _, row in seg_df.iterrows():
        seg_id = int(row["id"])
        osm_id = int(row["osm_id"])
        lid = osm_to_link.get(osm_id)
        if lid is not None:
            result[seg_id] = int(lid)

    logger.info(
        "Segment-to-link map: %d/%d segments matched to network links",
        len(result), len(seg_df),
    )
    return result
