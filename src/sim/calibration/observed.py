"""Load and prepare observed traffic data (pentlogram, CSD) for calibration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae import Project

from sim.calibration.matching import (
    _bearing_diff,
    _bearing_from_geom,
    _LINK_TYPE_PRIORITY,
    _NON_CAR_LINK_TYPES,
)
from sim.datasets.csd import ensure_csd2025_validation_parquet, normalize_csd_count_columns
from sim.defaults import SIM_DEFAULTS as _SIM_DEFAULTS, LOCALE_DEFAULTS as _LOCALE_DEFAULTS
from sim.io_project import get_metric_epsg, get_nested, load_config

logger = logging.getLogger(__name__)


def load_pentlogram(cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    geojson_path = Path(get_nested(
        cfg,
        ["datasets", "sources", "calibration_brno_pentlogram_2024", "out_path"],
        "data/sources/brno/intensity/intenzita_dopravy_pentlogram_2024.geojson",
    ))
    if not geojson_path.exists():
        raise FileNotFoundError(f"Pentlogram not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    out_epsg = int(get_nested(
        cfg,
        ["datasets", "sources", "calibration_brno_pentlogram_2024", "output", "out_epsg"],
        get_metric_epsg(cfg),
    ))

    from sim.zoning.geo import force_to_target_crs
    gdf = force_to_target_crs(gdf, out_epsg, name="pentlogram")

    # Fail-fast geometry validation
    valid_geom = gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~valid_geom).sum())
    if n_invalid > 0:
        logger.warning(f"  WARNING: {n_invalid}/{len(gdf)} pentlogram features have invalid geometry")
        gdf = gdf[valid_geom].copy()

    pent_cfg = get_nested(cfg, [
        "datasets", "sources", "calibration_brno_pentlogram_2024",
    ], {}) or {}
    col_vol = str(pent_cfg.get("volume_column", "car_24"))
    col_truck = str(pent_cfg.get("truck_pct_column", "truc_24"))

    for col in (col_vol, col_truck):
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

    units_cfg = pent_cfg.get("units", {}) or {}
    car_mult = float(units_cfg.get("car_24_multiplier", 1000))

    total_vehicles = gdf.get(col_vol, pd.Series(0, index=gdf.index)) * car_mult
    truck_pct = gdf.get(col_truck, pd.Series(0, index=gdf.index)).clip(0, 100)

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

    Filtering modes (checked in order):
    1. Explicit *region_code* argument (e.g. ``"CZ064"``).
    2. ``locale.yaml`` → ``csd_region_filter`` with ``column`` / ``contains``.
    3. Automatic: when ``csd_region_filter`` is absent or empty, filter CSD
       to roads present in the built network (by matching ``sil`` values to
       ``osm_ref`` on network links). This requires no manual region code.
    """
    from sim.io_project import load_locale

    parquet_path = ensure_csd2025_validation_parquet(cfg)

    df = pd.read_parquet(parquet_path)
    df = normalize_csd_count_columns(df)

    if region_code is not None:
        if "kk" in df.columns:
            df = df[df["kk"].astype(str).str.contains(
                region_code.replace("CZ", ""), na=False,
            )].copy()
    else:
        locale = load_locale(cfg)
        csd_filter = locale.get("csd_region_filter") or {}
        filter_col = csd_filter.get("column", "kk")
        filter_val = csd_filter.get("contains")
        if filter_val and filter_col in df.columns:
            df = df[df[filter_col].astype(str).str.contains(
                str(filter_val), na=False,
            )].copy()
        else:
            df = _filter_csd_by_network_roads(df, cfg)

    if "sil" in df.columns:
        df["sil"] = df["sil"].astype(str)
    for col in ("sv", "o", "tv"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_csd_unfiltered(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load the full national CSD parquet without any region filtering.

    Used for gateway AADT lookup where the gateway road may span regions
    outside the model area (e.g. D55 absent from CZ071 but present in
    CZ064/CZ072).
    """
    parquet_path = ensure_csd2025_validation_parquet(cfg)
    df = pd.read_parquet(parquet_path)
    df = normalize_csd_count_columns(df)
    if "sil" in df.columns:
        df["sil"] = df["sil"].astype(str)
    for col in ("sv", "o", "tv"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _filter_csd_by_network_roads(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Keep only CSD sections whose road number appears in the model network.

    Also applies region filtering when the model area is available,
    so that national averages of trunk roads are not polluted by distant
    sections.  Falls back to returning all rows if the project is not
    yet built.
    """
    if "sil" not in df.columns:
        return df
    project_dir = Path(cfg.get("project_path", "project/model"))
    try:
        links_gdf = _load_network_links(project_dir)
    except Exception:
        logger.info("  CSD auto-filter: network not available, returning all CSD rows")
        return df

    # Step 1: region filter via model-area bounding box
    df = _filter_csd_by_model_area_region(df, cfg)

    # Step 2: road-number filter
    raw_refs = links_gdf["osm_ref"].fillna("").astype(str).str.strip()
    ref_parts = set()
    for r in raw_refs:
        if r:
            for part in r.split(";"):
                p = part.strip()
                if p:
                    ref_parts.add(p)
    if not ref_parts:
        return df

    before = len(df)
    df = df[df["sil"].astype(str).str.strip().isin(ref_parts)].copy()
    logger.info(
        f"  CSD auto-filter: {before} → {len(df)} sections "
        f"(matched {len(ref_parts)} network road refs)"
    )
    return df


def _filter_csd_by_model_area_region(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Filter CSD to the ``kk`` region matching the configured city.

    Uses the ``osm.place_name`` from config to look up the corresponding
    Czech region code.  Falls back to no filtering if the city is unknown.
    """
    if "kk" not in df.columns:
        return df

    kk_unique = df["kk"].astype(str).unique()
    if len(kk_unique) <= 1:
        return df

    place_name = str(cfg.get("osm", {}).get("place_name", "")).lower()

    region_cfg = _LOCALE_DEFAULTS.get("csd_region_map", {})
    locale_cfg = cfg.get("locale", {}).get("csd_region_map", {})
    _REGION_HINTS: dict[str, str] = {**region_cfg.get("region_hints", {}), **locale_cfg.get("region_hints", {})}
    _CITY_REGION: dict[str, str] = {**region_cfg.get("city_region", {}), **locale_cfg.get("city_region", {})}

    region_hint = None
    for key, code in _CITY_REGION.items():
        if key in place_name:
            region_hint = code
            break
    if region_hint is None:
        for key, code in _REGION_HINTS.items():
            if key in place_name:
                region_hint = code
                break

    if region_hint is None:
        return df

    matching_kk = [k for k in kk_unique if region_hint in k]
    if not matching_kk:
        return df

    before = len(df)
    df = df[df["kk"].astype(str).isin(matching_kk)].copy()
    logger.info(
        "  CSD region auto-filter: %d → %d sections (region=%s)",
        before, len(df), matching_kk[0],
    )
    return df


def _load_network_links(project_dir: Path) -> gpd.GeoDataFrame:
    project = Project()
    project.open(str(project_dir))
    try:
        links_df = project.network.links.data.copy()
        crs = getattr(project.network, "crs", "EPSG:4326")
    finally:
        project.close()
    return gpd.GeoDataFrame(links_df, geometry="geometry", crs=crs)


# --- Aggregate CSD helpers (for validation) ---

def _classify_csd_road(sil: str) -> str:
    """Delegate to canonical implementation in datasets.csd."""
    from sim.datasets.csd import classify_csd_road
    return classify_csd_road(sil)


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


def normalize_csd_sil_key(road: Any) -> str:
    """Normalize CSD ``sil`` / config road ids for comparisons.

    Matches the logic used for ``auto_csd_*`` screenline names: purely numeric
    refs become strings without leading zeros via ``int()``; otherwise the
    uppercased token is left-stripped of zeros.
    """
    road_norm = str(road).strip().upper().replace(" ", "")
    try:
        road_norm = str(int(float(road_norm)))
    except ValueError:
        road_norm = road_norm.lstrip("0") or road_norm
    return road_norm.lower()


# --- CSD split for calibration / validation ---

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
    corridor
        Entire roads (by ``sil``) are assigned as units to calibration or
        validation.  Roads are grouped by ``road_class`` and within each
        class sorted by section count (descending) then road number, then
        greedily assigned to whichever subset is furthest from its target
        section count.  This guarantees that the validation set contains
        roads the calibration has *never seen* -- a much stronger
        independence guarantee than ``alternating``.
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

    elif strategy == "corridor":
        rng = np.random.RandomState(random_seed)

        road_info = (
            csd.groupby("sil")
            .agg(n_sections=("sil", "size"), road_class=("road_class", "first"))
            .reset_index()
        )
        road_info = road_info.sample(frac=1.0, random_state=rng).reset_index(drop=True)

        target_calib = round(len(csd) * calib_share)
        calib_roads: set = set()
        valid_roads: set = set()
        calib_count = 0

        for rc, grp in road_info.groupby("road_class"):
            grp = grp.sort_values("n_sections", ascending=False)
            rc_target = round(grp["n_sections"].sum() * calib_share)
            rc_calib = 0
            for _, row in grp.iterrows():
                if rc_calib < rc_target:
                    calib_roads.add(row["sil"])
                    rc_calib += row["n_sections"]
                    calib_count += row["n_sections"]
                else:
                    valid_roads.add(row["sil"])

        if not valid_roads:
            logger.warning(
                "Corridor CSD split produced an empty validation "
                "subset. Falling back to alternating.",
            )
            return split_csd_for_calibration(
                csd, strategy="alternating",
                calib_share=calib_share, random_seed=random_seed,
            )

        is_calib = csd["sil"].isin(calib_roads)
        calib_df = csd[is_calib].copy()
        valid_df = csd[~is_calib].copy()

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


_CSD_COMPATIBLE_LINK_TYPES: dict = {
    "motorway": {"motorway", "motorway_link"},
    "trunk": {"trunk", "trunk_link", "primary", "primary_link"},
    "secondary": {"secondary", "secondary_link", "primary", "primary_link"},
    "tertiary": {"tertiary", "tertiary_link", "secondary", "secondary_link", "unclassified", "residential"},
}

_MIN_VOL_FOR_CSD_LW = _SIM_DEFAULTS["calibration"]["matching"]["min_vol_for_csd_lw"]


def resolve_csd_link_count_options(calib_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """CSD anchor overrides from ``calibration.matching`` config."""
    match_cfg = (calib_cfg or {}).get("matching") or {}
    raw_obs = match_cfg.get("csd_observed_overrides") or {}
    raw_selectors = match_cfg.get("csd_anchor_selectors") or {}
    # Deprecated: prefer ``csd_anchor_selectors`` (cut / screenline / near).
    raw_link_ids = match_cfg.get("csd_anchor_link_ids") or {}
    observed_overrides: Dict[str, Dict[str, float]] = {}
    for road, vals in raw_obs.items():
        key = normalize_csd_sil_key(road)
        if not isinstance(vals, dict):
            continue
        observed_overrides[key] = {
            k: float(v) for k, v in vals.items()
            if k in ("observed_car", "observed_motor_total", "observed_total", "observed_truck")
        }
    anchor_selectors: Dict[str, Dict[str, Any]] = {}
    for road, spec in raw_selectors.items():
        key = normalize_csd_sil_key(road)
        if isinstance(spec, dict):
            anchor_selectors[key] = dict(spec)
    anchor_link_ids = {
        normalize_csd_sil_key(road): int(lid)
        for road, lid in raw_link_ids.items()
    }
    return {
        "observed_overrides": observed_overrides,
        "anchor_selectors": anchor_selectors,
        "anchor_link_ids": anchor_link_ids,
    }


def load_screenlines_by_name(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load manual/auto screenline defs keyed by name (for CSD anchor selection)."""
    try:
        from sim.calibration.screenlines import load_screenlines_with_auto

        csd_for_auto = None
        try:
            csd_for_auto = load_csd(cfg)
        except Exception:
            pass
        sls = load_screenlines_with_auto(cfg, csd_df=csd_for_auto)
        return {sl.name: sl for sl in sls}
    except Exception:
        logger.debug("Screenlines not available for CSD anchor selection", exc_info=True)
        return {}


def _pick_csd_anchor_links(
    matched_links: gpd.GeoDataFrame,
    selector: Dict[str, Any],
    links_gdf: gpd.GeoDataFrame,
    *,
    screenlines_by_name: Optional[Dict[str, Any]] = None,
    metric_epsg: int = 5514,
) -> gpd.GeoDataFrame:
    """Narrow CSD anchor candidates using a stable selector (not link_id)."""
    from sim.calibration.screenlines import ScreenlineDef, resolve_screenline_links

    if "screenline" in selector:
        sl_name = str(selector["screenline"])
        sl = (screenlines_by_name or {}).get(sl_name)
        if sl is None:
            logger.warning(
                "CSD anchor screenline '%s' not found — using default link pick",
                sl_name,
            )
            return matched_links
        lids = {int(lid) for lid, _ in resolve_screenline_links(sl, links_gdf, metric_epsg)}
    elif "cut" in selector:
        cut = selector["cut"] or {}
        sl = ScreenlineDef(
            name="_csd_anchor_cut",
            geometry_wkt=cut.get("geometry_wkt"),
            attr_filter=dict(cut.get("filter") or {}),
            expected_links=cut.get("expected_links"),
        )
        pick_mode = str(cut.get("pick", "intersect")).strip().lower()
        geom_wkt = cut.get("geometry_wkt")
        if pick_mode == "nearest" and geom_wkt:
            from shapely import wkt as shapely_wkt
            import pyproj
            from shapely.ops import transform as shapely_transform

            geom_raw = shapely_wkt.loads(geom_wkt)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326", f"EPSG:{metric_epsg}", always_xy=True,
            )
            cut_metric = shapely_transform(transformer.transform, geom_raw)
            work = (
                matched_links.to_crs(epsg=metric_epsg)
                if matched_links.crs and matched_links.crs.to_epsg() != metric_epsg
                else matched_links
            )
            dists = work.geometry.distance(cut_metric)
            return matched_links.loc[[dists.idxmin()]]
        lids = {int(lid) for lid, _ in resolve_screenline_links(sl, links_gdf, metric_epsg)}
    elif "near" in selector:
        near = selector["near"] or {}
        lon = near.get("lon", near.get("lng"))
        lat = near.get("lat")
        if lon is None or lat is None:
            logger.warning("CSD anchor near: missing lon/lat — using default link pick")
            return matched_links
        pt = gpd.GeoSeries(
            [gpd.points_from_xy([float(lon)], [float(lat)])[0]],
            crs="EPSG:4326",
        ).to_crs(epsg=metric_epsg).iloc[0]
        work = matched_links.to_crs(epsg=metric_epsg) if matched_links.crs else matched_links
        dists = work.geometry.distance(pt)
        return matched_links.loc[[dists.idxmin()]]
    else:
        logger.warning("Unknown CSD anchor selector keys %s — using default pick", list(selector))
        return matched_links

    if not lids or "link_id" not in matched_links.columns:
        return matched_links
    picked = matched_links[matched_links["link_id"].astype(int).isin(lids)]
    if picked.empty:
        logger.warning(
            "CSD anchor selector %s matched no links among sil candidates — default pick",
            selector,
        )
        return matched_links
    tie_near = (selector.get("cut") or {}).get("tie_break_near")
    if len(picked) > 1 and isinstance(tie_near, dict):
        lon = tie_near.get("lon", tie_near.get("lng"))
        lat = tie_near.get("lat")
        if lon is not None and lat is not None:
            pt = gpd.GeoSeries(
                [gpd.points_from_xy([float(lon)], [float(lat)])[0]],
                crs="EPSG:4326",
            ).to_crs(epsg=metric_epsg).iloc[0]
            work = (
                picked.to_crs(epsg=metric_epsg)
                if picked.crs and picked.crs.to_epsg() != metric_epsg
                else picked
            )
            dists = work.geometry.distance(pt)
            picked = picked.loc[[dists.idxmin()]]
    return picked


def load_csd_as_link_counts(
    csd: pd.DataFrame,
    links_gdf: gpd.GeoDataFrame,
    *,
    observed_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    anchor_selectors: Optional[Dict[str, Dict[str, Any]]] = None,
    anchor_link_ids: Optional[Dict[str, int]] = None,
    screenlines_by_name: Optional[Dict[str, Any]] = None,
    metric_epsg: Optional[int] = None,
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
        road_key = normalize_csd_sil_key(road)
        ov = (observed_overrides or {}).get(road_key)
        if ov:
            if "observed_car" in ov:
                csd_mean_o = float(ov["observed_car"])
            if "observed_motor_total" in ov:
                csd_mean_sv = float(ov["observed_motor_total"])
            elif "observed_total" in ov:
                csd_mean_sv = float(ov["observed_total"])
            if "observed_truck" in ov:
                csd_mean_tv = float(ov["observed_truck"])
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

        selector = (anchor_selectors or {}).get(road_key)
        if selector:
            matched_links = _pick_csd_anchor_links(
                matched_links,
                selector,
                links_gdf,
                screenlines_by_name=screenlines_by_name,
                metric_epsg=int(metric_epsg or get_metric_epsg({})),
            )
        else:
            forced_lid = (anchor_link_ids or {}).get(road_key)
            if forced_lid is not None and "link_id" in matched_links.columns:
                forced = matched_links[matched_links["link_id"] == int(forced_lid)]
                if forced.empty:
                    logger.warning(
                        "CSD anchor link_id=%s not among matches for sil=%s — using default pick",
                        forced_lid, road,
                    )
                else:
                    matched_links = forced

        if "link_type" in matched_links.columns:
            matched_links = matched_links.sort_values(
                "link_type",
                key=lambda s: s.map(_LINK_TYPE_PRIORITY).fillna(99),
            )

        # Among same-priority links, prefer interior links (both nodes
        # shared with other matched links) over edge/dead-end links.
        first = matched_links.iloc[0]
        best_pri = _LINK_TYPE_PRIORITY.get(str(first.get("link_type", "")), 99)
        top_tier = matched_links[
            matched_links["link_type"].map(_LINK_TYPE_PRIORITY).fillna(99) == best_pri
        ]
        if len(top_tier) > 1 and "a_node" in top_tier.columns:
            all_a = top_tier["a_node"].tolist()
            all_b = top_tier["b_node"].tolist()
            node_set = set(all_a + all_b)
            centr = []
            for idx_r, row_r in top_tier.iterrows():
                other = top_tier.drop(index=idx_r)
                other_n = set(other["a_node"].tolist() + other["b_node"].tolist())
                c = (1 if row_r["a_node"] in other_n else 0) + (1 if row_r["b_node"] in other_n else 0)
                centr.append(c)
            top_tier = top_tier.copy()
            top_tier["_centr"] = centr
            top_tier = top_tier.sort_values(
                ["_centr", "distance"], ascending=[False, False],
            )
            first = top_tier.iloc[0]
        lid = first.get("link_id", matched_links.index[0])
        rows.append({
            "objectid": _SIM_DEFAULTS["calibration"]["matching"]["synthetic_objectid_base"] + len(rows),
            "link_id": lid,
            "geometry": first.geometry,
            "name": str(first.get("name", "") or ""),
            "osm_ref": str(first.get("osm_ref", "") or ""),
            "link_type": str(first.get("link_type", "") or ""),
            "observed_car": csd_mean_o,
            "observed_motor_total": csd_mean_sv,
            "observed_total": csd_mean_sv,
            "observed_truck": max(0.0, csd_mean_sv - csd_mean_o),
            "csd_road": road,
            "csd_road_class": road_class,
            "_n_matched_links": len(matched_links),
            "_csd_n_sections": len(csd_sub),
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
