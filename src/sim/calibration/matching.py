"""Count-to-network spatial matching for calibration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from sim._metrics import compute_geh
from sim.defaults import SIM_DEFAULTS as _SIM_DEFAULTS

logger = logging.getLogger(__name__)


from sim.calibration.gateway import _MAJOR_ROAD_TYPES, _MAJOR_ROAD_TYPES_STRICT  # noqa: F401


from sim._geo import bearing_from_line as _bearing_from_geom  # noqa: E402
from sim._geo import bearing_deg, bearing_diff as _bearing_diff  # noqa: E402,F401


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

    lid_to_idx: dict = {}
    for _i, _lid_val in enumerate(link_lid):
        lid_to_idx.setdefault(int(_lid_val), _i)

    for idx, row in joined.iterrows():
        if pd.isna(row.get("link_id")):
            continue
        matched_lid = int(row["link_id"])
        count_pt = row.geometry
        matched_lt = str(row.get("link_type", ""))

        mi = lid_to_idx.get(matched_lid)
        if mi is None:
            continue
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
            if _TYPE_FAMILY.get(lt) != _TYPE_FAMILY.get(matched_lt, matched_lt):
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

            _csw = _SIM_DEFAULTS["calibration"]["matching"]["corridor_score_weights"]
            score = _csw["distance"] * dist_score + _csw["bearing"] * bearing_score + _csw["name"] * name_score
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

_LINK_TYPE_PRIORITY: Dict[str, int] = {
    "motorway": 0, "trunk": 1, "primary": 2, "secondary": 3,
    "tertiary": 4, "unclassified": 5, "residential": 6,
    "motorway_link": 10, "trunk_link": 11, "primary_link": 12,
    "secondary_link": 13, "tertiary_link": 14,
}

_TYPE_FAMILY: Dict[str, str] = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
}


def _compute_match_confidence(
    vol: float, obs: float, link_type: str,
    n_sections: float = np.inf,
    match_quality: float = 1.0,
    *,
    threshold: float = 0.3,
) -> float:
    """Return 0.0–1.0 confidence score for a count↔link match.

    Combines volume-ratio plausibility, link-type compatibility,
    CSD section count, and geometric match quality. A match with
    confidence < *threshold* is flagged for exclusion.
    """
    score = 1.0

    if obs <= 0 or vol < 0:
        return 0.0

    ratio = vol / max(obs, 1.0)
    inv_ratio = obs / max(vol, 1.0) if vol > 0 else 0.0

    # Zero/near-zero model volume with high observed → certainly wrong link
    if vol <= 0 and obs >= 5000:
        return 0.0
    if obs >= 2000 and vol < obs * 0.01:
        return 0.0

    # Major-road reverse mismatch (obs << model)
    is_major_strict = link_type in _MAJOR_ROAD_TYPES_STRICT
    is_link = link_type.endswith("_link")

    if is_major_strict and vol > 15000 and inv_ratio < 0.30:
        score *= 0.1
    if is_major_strict and vol > 5000 and ratio > 4.0:
        score *= 0.15
    if is_link and obs > 2000 and vol > 0 and ratio < 0.20:
        score *= 0.1

    # Moderate ratio deviations penalise gradually
    if vol > 0 and obs > 0:
        if ratio > 2.5:
            score *= max(0.2, 1.0 - (ratio - 2.5) / 5.0)
        elif ratio < 0.4:
            score *= max(0.2, ratio / 0.4)

    # CSD low-confidence: few sections + large overestimation
    if n_sections < 3 and vol > 0 and obs > 0 and ratio > 2.0:
        score *= 0.2

    # Geometric match quality
    score *= max(match_quality, 0.0)

    return float(np.clip(score, 0.0, 1.0))


def _apply_exclusion_scoring(
    joined: gpd.GeoDataFrame,
    vol_col: Optional[str],
    match_quality_min: float,
    phase: str = "pre_corridor",
) -> gpd.GeoDataFrame:
    """Apply unified confidence scoring to flag unreliable matches."""
    if "_excluded" not in joined.columns:
        joined["_excluded"] = False

    if not vol_col or vol_col not in joined.columns:
        return joined

    vol_vals = pd.to_numeric(joined[vol_col], errors="coerce").fillna(0)
    obs_car = (
        pd.to_numeric(joined["observed_car"], errors="coerce").fillna(0)
        if "observed_car" in joined.columns
        else pd.Series(0, index=joined.index)
    )
    link_types = joined["link_type"].astype(str) if "link_type" in joined.columns else pd.Series("", index=joined.index)
    n_sections = (
        pd.to_numeric(joined["_csd_n_sections"], errors="coerce").fillna(np.inf)
        if "_csd_n_sections" in joined.columns
        else pd.Series(np.inf, index=joined.index)
    )
    mq = (
        pd.to_numeric(joined["_match_quality"], errors="coerce").fillna(1.0)
        if "_match_quality" in joined.columns
        else pd.Series(1.0, index=joined.index)
    )

    threshold = max(match_quality_min, 0.3) if phase == "pre_corridor" else 0.3

    confidences = np.array([
        _compute_match_confidence(v, o, lt, ns, q, threshold=threshold)
        for v, o, lt, ns, q in zip(vol_vals, obs_car, link_types, n_sections, mq)
    ])

    newly_excluded = (~joined["_excluded"]) & (confidences < threshold)
    n_new = int(newly_excluded.sum())
    if n_new > 0:
        joined.loc[newly_excluded, "_excluded"] = True
        joined.loc[newly_excluded, "_match_confidence"] = confidences[newly_excluded.values]
        logger.info(
            f"  Matching ({phase}): excluded {n_new} low-confidence matches "
            f"(threshold={threshold:.2f})"
        )

    _MAJOR_HW = {"motorway", "motorway_link", "trunk", "trunk_link"}
    obs_total = (
        pd.to_numeric(joined["observed_total"], errors="coerce").fillna(0)
        if "observed_total" in joined.columns
        else pd.Series(0, index=joined.index)
    )
    zero_major = (vol_vals <= 0) & (obs_total > 0) & link_types.isin(_MAJOR_HW)
    n_zero_major = int(zero_major.sum())
    if n_zero_major > 0:
        for idx in joined.index[zero_major]:
            row = joined.loc[idx]
            logger.warning(
                "  CONNECTIVITY WARNING: %s link %s has 0 modelled volume "
                "but %s observed — likely disconnected from graph",
                row.get("link_type", "?"),
                row.get("link_id", row.get("csd_road", "?")),
                f"{obs_total.loc[idx]:,.0f}",
            )

    return joined


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
    match_quality_min: float = 0.50,
    skip_exclusion: bool = False,
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
        "motorway": 1.0, "trunk": 0.875,
        "motorway_link": 0.45, "trunk_link": 0.35,
        "primary": 0.5, "primary_link": 0.25,
        "secondary": 0.25, "secondary_link": 0.12, "tertiary": 0.15,
        "tertiary_link": 0.07, "unclassified": 0.09, "road": 0.09,
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

        # Pre-scan: collect parent type families present among candidates
        # so we can penalize *_link types when their parent exists nearby.
        cand_families: set = set()
        for _ci in cand_idxs:
            _ct = str(links_sel.iloc[_ci].get("link_type", ""))
            if not _ct.endswith("_link"):
                fam = _TYPE_FAMILY.get(_ct)
                if fam:
                    cand_families.add(fam)

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

            q = 0.25 * (1.0 - dn) + 0.25 * (1.0 - bp) + 0.25 * rw + 0.25 * vb

            if lt.endswith("_link") and _TYPE_FAMILY.get(lt) in cand_families:
                q *= _SIM_DEFAULTS["calibration"]["matching"]["link_type_penalty"]

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
            is_major = (
                joined["link_type"].astype(str).isin(_MAJOR_ROAD_TYPES)
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

    if not skip_exclusion:
        joined = _apply_exclusion_scoring(joined, vol_col, match_quality_min)

    if aggregate_corridor:
        joined = _aggregate_corridor_volumes(joined, links_sel, buffer_m)

    corr_col = "_corridor_volume" if "_corridor_volume" in joined.columns else None
    if not skip_exclusion and corr_col:
        joined = _apply_exclusion_scoring(joined, corr_col, 0.0, phase="post_corridor")

    for col in ("_count_bearing", "_link_bearing"):
        if col in joined.columns:
            joined = joined.drop(columns=[col])

    # Diagnostic summary of matched vs unmatched/excluded
    n_total = len(joined)
    n_matched = int(joined["_matched"].sum()) if "_matched" in joined.columns else 0
    n_excluded = int(joined["_excluded"].sum()) if "_excluded" in joined.columns else 0
    n_usable = n_matched - n_excluded
    logger.info(
        f"  Match summary: {n_total} count stations -> "
        f"{n_matched} matched, {n_excluded} excluded, {n_usable} usable"
    )
    if n_usable < n_total * 0.5:
        # Log which stations failed matching for diagnosis
        if "_matched" in joined.columns and id_col in joined.columns:
            unmatched = joined[~joined["_matched"]]
            if not unmatched.empty:
                ids = unmatched[id_col].tolist()[:20]
                logger.warning(
                    f"  {len(unmatched)} count station(s) found no link match "
                    f"(buffer={buffer_m}m): {ids}"
                )
        if "_excluded" in joined.columns and id_col in joined.columns:
            excluded = joined[joined["_excluded"]]
            if not excluded.empty:
                ids = excluded[id_col].tolist()[:20]
                logger.warning(
                    f"  {len(excluded)} count station(s) excluded: {ids}"
                )

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

    # Store count-point coordinates so unmatched stations can be placed on map.
    if "geometry" in matched.columns:
        try:
            wgs = matched.to_crs(epsg=4326) if matched.crs and matched.crs.to_epsg() != 4326 else matched
            centroids = wgs.geometry.centroid
            diag["_count_lng"] = centroids.x.round(6)
            diag["_count_lat"] = centroids.y.round(6)
        except Exception:
            pass

    # Re-evaluate exclusion using final (post-calibration) volumes.
    final_vol_col = model_col if model_col and model_col in diag.columns else None
    if final_vol_col:
        diag_gdf = gpd.GeoDataFrame(diag) if not isinstance(diag, gpd.GeoDataFrame) else diag
        diag = _apply_exclusion_scoring(diag_gdf, final_vol_col, 0.0, phase="post_calibration")

    m_arr = diag[model_col].values.astype(float) if model_col and model_col in diag.columns else np.zeros(len(diag))
    o_arr = diag[obs_col].values.astype(float) if obs_col in diag.columns else np.zeros(len(diag))
    diag["GEH"] = compute_geh(m_arr, o_arr)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "matching_diagnostics.csv"
    diag.to_csv(out_path, index=False)
    logger.info(f"  Matching diagnostics: {out_path} ({len(diag)} rows)")

