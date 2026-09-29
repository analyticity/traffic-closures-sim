"""Read the PostgreSQL-backed sources from CSV dumps instead of a live database.

The ``postgres_*`` fetchers in this package are each a thin wrapper around a
single ``SELECT``: they build a DataFrame, clip it to the model area and write
a parquet that everything downstream reads.  Only the first step needs the
database, so a plain ``COPY ... TO CSV`` dump of the same tables is enough to
run the whole evaluation offline (and inside Docker, where the DB is not
reachable at all).

The helpers here turn a raw table dump into **exactly the frame the SQL query
would have returned**: same column names, same derived columns.  Geography
columns arrive as PostGIS hex EWKB (``0101000020E6100000...``) and are decoded
to the ``lat``/``lon``/``*_wkt`` columns the fetchers expect.

Usage in a fetcher::

    csv_path = source_cfg.get("csv_path")
    if csv_path:
        df = load_jams_csv(csv_path, segments_csv_path=...)
    else:
        ...psycopg2 branch...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# psql writes SQL NULL as an unquoted empty field, but hand-made exports often
# use the literal string; treat both as missing.
_NULL_TOKENS = ("NULL", "null", "\\N")


def read_table_csv(
    path: str | Path,
    *,
    usecols: Optional[Sequence[str]] = None,
    required: Iterable[str] = (),
) -> pd.DataFrame:
    """Read a raw table dump, normalising SQL NULL tokens to ``NaN``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV source not found: {p}")

    usecols_arg = None
    if usecols is not None:
        header = pd.read_csv(p, nrows=0).columns
        usecols_arg = [c for c in usecols if c in header]
        missing_opt = [c for c in usecols if c not in header]
        if missing_opt:
            logger.debug("%s: columns not in dump, skipped: %s", p.name, missing_opt)

    df = pd.read_csv(
        p,
        usecols=usecols_arg,
        na_values=_NULL_TOKENS,
        keep_default_na=True,
        low_memory=False,
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name} is missing required columns: {missing}")

    logger.info("Read %d rows from %s", len(df), p)
    return df


def _decode_ewkb(series: pd.Series) -> pd.Series:
    """Decode a hex EWKB column to shapely geometries (``None`` where absent)."""
    from shapely import wkb

    def _one(value: Any):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text or text in _NULL_TOKENS:
            return None
        try:
            return wkb.loads(text, hex=True)
        except Exception:  # malformed row must not kill the whole import
            return None

    return series.map(_one)


def add_lonlat_from_geog(
    df: pd.DataFrame,
    geog_col: str,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    drop: bool = True,
) -> pd.DataFrame:
    """Add ``lon``/``lat`` from the **centroid** of a hex EWKB column.

    Mirrors ``ST_X/ST_Y(ST_Centroid(<geog>::geometry))`` in the SQL fetchers,
    so points keep their own coordinates and lines collapse to their midpoint.
    """
    if geog_col not in df.columns:
        raise ValueError(f"geography column {geog_col!r} not in dump")

    geoms = _decode_ewkb(df[geog_col])
    n_bad = int(geoms.isna().sum())
    if n_bad:
        logger.warning("%s: %d/%d rows have no decodable geometry", geog_col, n_bad, len(df))

    centroids = geoms.map(lambda g: g.centroid if g is not None else None)
    df[lon_col] = centroids.map(lambda p: p.x if p is not None else float("nan"))
    df[lat_col] = centroids.map(lambda p: p.y if p is not None else float("nan"))
    if drop:
        df = df.drop(columns=[geog_col])
    return df


def add_wkt_from_geog(
    df: pd.DataFrame,
    geog_col: str,
    *,
    wkt_col: str,
    drop: bool = True,
) -> pd.DataFrame:
    """Add a WKT column from hex EWKB, mirroring ``ST_AsText(<geog>::geometry)``."""
    if geog_col not in df.columns:
        df[wkt_col] = None
        return df

    geoms = _decode_ewkb(df[geog_col])
    df[wkt_col] = geoms.map(lambda g: g.wkt if g is not None else None)
    if drop:
        df = df.drop(columns=[geog_col])
    return df


# --------------------------------------------------------------------------
# Per-source loaders — each returns the frame its SQL query would have returned
# --------------------------------------------------------------------------

_SEGMENT_JOIN_COLS = {
    "osm_id": "osm_id",
    "road_ref": "segment_road_ref",
    "road_class": "segment_road_class",
    "max_speed": "segment_max_speed",
}


def load_segments_csv(path: str | Path) -> pd.DataFrame:
    """``road_segments``: id, osm_id, name, road_ref, road_class, city, max_speed, lat, lon."""
    df = read_table_csv(
        path,
        usecols=["id", "osm_id", "geog", "name", "road_ref", "road_class", "city", "max_speed"],
        required=["id", "osm_id", "geog"],
    )
    return add_lonlat_from_geog(df, "geog")


def load_jams_csv(
    path: str | Path,
    *,
    segments_csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """``traffic_jams`` left-joined to ``road_segments``, as in the SQL fetcher.

    ``raw`` and ``external_ids`` are skipped on read — they are the bulk of the
    dump (jams.csv is ~500 MB, mostly raw JSON) and nothing consumes them.
    """
    df = read_table_csv(
        path,
        usecols=[
            "id", "first_seen", "last_seen", "city", "street_name", "road_number",
            "road_type_code", "delay_seconds", "length_m", "speed_kmh",
            "speed_normal_kmh", "severity", "quality_score", "segment_id",
            "jam_line_geog",
        ],
        required=["id", "segment_id", "jam_line_geog"],
    )

    # The SQL has WHERE jam_line_geog IS NOT NULL.
    before = len(df)
    df = df[df["jam_line_geog"].notna()].copy()
    if len(df) < before:
        logger.info("jams: %d -> %d rows with geometry", before, len(df))

    df = add_lonlat_from_geog(df, "jam_line_geog")

    if segments_csv_path:
        seg = read_table_csv(
            segments_csv_path,
            usecols=["id", *_SEGMENT_JOIN_COLS.keys()],
            required=["id"],
        ).rename(columns=_SEGMENT_JOIN_COLS)
        df = df.merge(
            seg.rename(columns={"id": "segment_id"}),
            on="segment_id",
            how="left",
        )
        matched = int(df["osm_id"].notna().sum())
        logger.info(
            "jams: joined road_segments, %d/%d rows got an osm_id (%.1f %%)",
            matched, len(df), 100.0 * matched / max(len(df), 1),
        )
    else:
        logger.warning(
            "jams: no segments_csv_path given — osm_id will be empty and the "
            "jams cannot be mapped onto network links"
        )
        for col in _SEGMENT_JOIN_COLS.values():
            df[col] = None

    return df


def load_closures_csv(path: str | Path) -> pd.DataFrame:
    """``restrictions`` with the aliases the closures fetcher expects.

    ``external_ids`` is kept (the SQL drops it) because it is the only way to
    tell NDIC roadworks from short-lived Waze ``road_closed`` alerts, which
    matters for the closure before/after analysis.  Extra columns are harmless
    downstream — the parquet simply carries one more field.
    """
    df = read_table_csv(
        path,
        usecols=[
            "id", "external_ids", "restriction_type", "restriction_subtype", "severity",
            "status", "road_number", "street_name", "city", "description_cs",
            "max_speed_kmh", "valid_from", "valid_to", "first_seen", "last_seen",
            "direction", "km_from", "km_to", "road_type_code", "quality_score",
            "urgency", "probability", "segment_id",
            "location_point_geog", "location_line_geog",
        ],
        required=["id", "location_point_geog"],
    )

    # WHERE location_point_geog IS NOT NULL
    before = len(df)
    df = df[df["location_point_geog"].notna()].copy()
    if len(df) < before:
        logger.info("closures: %d -> %d rows with point geometry", before, len(df))

    df = df.rename(columns={"severity": "pg_severity", "direction": "closure_direction"})
    df = add_lonlat_from_geog(df, "location_point_geog")
    df = add_wkt_from_geog(df, "location_line_geog", wkt_col="line_wkt")
    return df


_EVENT_LINK_SOURCE_TYPES = ("restriction", "accident", "alert")


def load_event_links_csv(
    path: str | Path,
    *,
    jams_csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """``event_links`` filtered to ``caused_by`` and joined to jam details."""
    df = read_table_csv(
        path,
        usecols=[
            "id", "source_type", "source_id", "target_type", "target_id",
            "link_type", "confidence", "description",
        ],
        required=["id", "source_type", "source_id", "target_type", "target_id", "link_type"],
    ).rename(columns={"id": "link_id"})

    before = len(df)
    df = df[
        (df["link_type"] == "caused_by")
        & (df["source_type"].isin(_EVENT_LINK_SOURCE_TYPES))
    ].copy()
    logger.info("event_links: %d -> %d causal links", before, len(df))

    jam_cols = {
        "first_seen": "jam_first_seen",
        "last_seen": "jam_last_seen",
        "delay_seconds": "jam_delay_s",
        "length_m": "jam_length_m",
        "speed_kmh": "jam_speed_kmh",
        "speed_normal_kmh": "jam_speed_normal_kmh",
        "severity": "jam_severity",
        "segment_id": "jam_segment_id",
        "city": "jam_city",
        "road_number": "jam_road_number",
    }

    if jams_csv_path:
        jams = read_table_csv(
            jams_csv_path,
            usecols=["id", *jam_cols.keys()],
            required=["id"],
        ).rename(columns=jam_cols)
        df = df.merge(
            jams.rename(columns={"id": "target_id"}),
            on="target_id",
            how="left",
        )
    else:
        logger.warning("event_links: no jams_csv_path given — jam detail columns stay empty")
        for col in jam_cols.values():
            df[col] = None

    df = df.sort_values(["source_type", "source_id"]).reset_index(drop=True)
    return df


def resolve_csv_path(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    filename: str,
    *,
    key: str = "csv_path",
) -> Optional[str]:
    """Return ``source_cfg[key]`` or ``<datasets.csv_dir>/<filename>`` if it exists.

    Setting ``datasets.csv_dir`` is therefore enough to switch all four sources
    over at once, as long as the dumps keep their table names.  Returns ``None``
    when neither is available, which keeps the PostgreSQL branch in charge.
    """
    explicit = source_cfg.get(key)
    if explicit:
        return str(explicit)

    csv_dir = (cfg.get("datasets") or {}).get("csv_dir")
    if not csv_dir:
        return None

    candidate = Path(csv_dir) / filename
    return str(candidate) if candidate.exists() else None
