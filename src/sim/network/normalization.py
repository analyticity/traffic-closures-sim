"""Link attribute normalization: speed, lanes, capacity, travel time.

Preserves AB/BA directional asymmetry -- for two-way links both values
are kept separately; only missing values are filled from defaults or
the opposite direction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from aequilibrae import Project

from sim.defaults import NETWORK_NORM_DEFAULTS
from sim.network.db import project_db, project_db_path

logger = logging.getLogger(__name__)

_NORM_DEFS = NETWORK_NORM_DEFAULTS["normalization"]
_DEFAULT_THRESHOLDS: Dict[str, float] = _NORM_DEFS["thresholds"]
_DEFAULT_SPEED_BY_LINK_TYPE: Dict[str, float] = _NORM_DEFS["defaults"]["speed_by_link_type"]
_DEFAULT_LANES_BY_LINK_TYPE: Dict[str, int] = _NORM_DEFS["defaults"]["lanes_by_link_type"]
_DEFAULT_CAPACITY_PER_LANE: Dict[str, int] = _NORM_DEFS["defaults"]["capacity_per_lane_by_link_type"]

_LINK_TYPE_IDS = {
    "motorway_link": "M",
    "trunk_link": "K",
    "primary_link": "D",
    "secondary_link": "L",
    "tertiary_link": "V",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolved_experiment_profile(network_cfg: dict, experiment_profile: str) -> dict:
    profiles = network_cfg.get("experiment_profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    prof = profiles.get(experiment_profile)
    if prof is None:
        prof = profiles.get("baseline")
    if prof is None:
        prof = {
            "speed_caps": {},
            "speed_floors": {},
            "capacity_factors": {},
            "time_penalties": {},
        }
    return prof


def _normalization_defaults(network_cfg: dict) -> tuple[dict, dict]:
    norm = network_cfg.get("normalization") or {}
    yaml_defaults = norm.get("defaults") if isinstance(norm.get("defaults"), dict) else {}

    defaults = {
        "speed_by_link_type": {**_DEFAULT_SPEED_BY_LINK_TYPE, **(yaml_defaults.get("speed_by_link_type") or {})},
        "lanes_by_link_type": {**_DEFAULT_LANES_BY_LINK_TYPE, **(yaml_defaults.get("lanes_by_link_type") or {})},
        "capacity_per_lane_by_link_type": {**_DEFAULT_CAPACITY_PER_LANE, **(yaml_defaults.get("capacity_per_lane_by_link_type") or {})},
    }
    yaml_thresh = norm.get("thresholds") if isinstance(norm.get("thresholds"), dict) else {}
    thresholds = {**_DEFAULT_THRESHOLDS, **yaml_thresh}
    return defaults, thresholds


def _threshold(thresholds: dict, key: str, default: float) -> float:
    if key not in thresholds:
        return float(default)
    return float(thresholds[key])


def _lt_mask(links: pd.DataFrame, link_type: str) -> pd.Series:
    return links["link_type"].astype(str).str.fullmatch(link_type, case=False, na=False)


def _apply_csd_capacity_hints(
    links: pd.DataFrame,
    csd_path: Optional[Path],
    peak_hour_factor: float = 0.10,
) -> int:
    """Refine capacity for links matchable to CSD data via road reference.

    Uses observed AADT from CSD to set a minimum effective capacity:
    capacity >= AADT * peak_hour_factor / lanes.  This prevents
    under-capacity on roads where CSD shows high traffic.

    Returns the number of links updated.
    """
    if csd_path is None or not csd_path.exists():
        return 0

    try:
        csd = pd.read_parquet(csd_path)
    except Exception:
        logger.debug("Cannot read CSD parquet at %s", csd_path)
        return 0

    if "sil" not in csd.columns or "sv" not in csd.columns:
        return 0
    if "osm_ref_norm" not in links.columns:
        return 0

    csd["sil_norm"] = csd["sil"].astype(str).str.strip().str.upper().str.replace("/", "", regex=False)
    csd["sv"] = pd.to_numeric(csd["sv"], errors="coerce").fillna(0)
    csd_agg = csd.groupby("sil_norm")["sv"].mean().reset_index()
    csd_agg.columns = ["road_ref", "aadt"]
    csd_agg = csd_agg[csd_agg["aadt"] > 0]

    if csd_agg.empty:
        return 0

    links["_ref_norm"] = links["osm_ref_norm"].astype(str).str.strip().str.upper().str.replace("/", "", regex=False)
    merged = links[["_ref_norm"]].merge(csd_agg, left_on="_ref_norm", right_on="road_ref", how="left")
    has_aadt = merged["aadt"].notna()

    n_updated = 0
    if has_aadt.any():
        idx = has_aadt[has_aadt].index
        min_cap = merged.loc[idx, "aadt"].values * peak_hour_factor
        for suffix, lanes_col in [("ab", "lanes_ab"), ("ba", "lanes_ba")]:
            cap_col = f"capacity_{suffix}"
            lanes_vals = links.loc[idx, lanes_col].clip(lower=1).values
            per_lane_min = min_cap / lanes_vals
            current = links.loc[idx, cap_col].values
            threshold = per_lane_min * lanes_vals
            upgrade = current < threshold
            if upgrade.any():
                new_cap = threshold[upgrade]
                up_idx = idx[upgrade]
                links.loc[up_idx, cap_col] = np.maximum(
                    links.loc[up_idx, cap_col].values,
                    new_cap,
                )
                n_updated += int(upgrade.sum())

    links.drop(columns=["_ref_norm"], inplace=True, errors="ignore")
    if n_updated > 0:
        logger.info("CSD capacity hints: upgraded %d link-directions", n_updated)
    return n_updated


def _apply_practical_speed_reduction(
    links: pd.DataFrame,
    project: Project,
    pffs_cfg: dict,
) -> None:
    """HCM-inspired practical free-flow speed reduction.

    ``practical_speed = posted_speed * base_factor - penalty * ipkm``

    where *ipkm* is the intersection density (nodes with degree >= threshold
    per kilometre of link length).  This captures the speed loss caused by
    signalisation, priority intersections, access points and general urban
    friction without any city-specific tuning.
    """
    if not pffs_cfg.get("enabled", False):
        return

    base_factor = float(pffs_cfg.get("base_factor", 0.85))
    penalty = float(pffs_cfg.get("intersection_penalty_per_km", 0.02))
    min_degree = int(pffs_cfg.get("min_intersection_degree", 3))
    min_speed = float(pffs_cfg.get("min_speed_kmh", 5.0))

    node_counts = (
        links["a_node"].value_counts().add(
            links["b_node"].value_counts(), fill_value=0
        )
    )
    high_degree_nodes = frozenset(node_counts[node_counts >= min_degree].index)

    a_is_int = links["a_node"].isin(high_degree_nodes).astype(int)
    b_is_int = links["b_node"].isin(high_degree_nodes).astype(int)
    n_intersections = a_is_int + b_is_int

    dist_km = (links["distance"] / 1000.0).clip(lower=0.01)
    ipkm = n_intersections / dist_km

    for col in ("speed_ab", "speed_ba"):
        links[col] = (links[col] * base_factor - penalty * ipkm).clip(lower=min_speed)

    logger.info(
        "Practical speed reduction (base=%.2f, penalty=%.3f/km): "
        "median ipkm=%.1f, mean speed change %.1f km/h",
        base_factor,
        penalty,
        float(ipkm.median()),
        float((links["speed_ab"] * (1 - base_factor)).mean()),
    )


def _apply_speed_floors(links: pd.DataFrame, speed_floors: dict) -> None:
    for lt, floor_val in speed_floors.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "speed_ab"] = np.maximum(links.loc[m, "speed_ab"], floor_val)
        links.loc[m, "speed_ba"] = np.maximum(links.loc[m, "speed_ba"], floor_val)


def _apply_speed_caps(links: pd.DataFrame, speed_caps: dict) -> None:
    for lt, cap in speed_caps.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        before_ab = links.loc[m, "speed_ab"].mean()
        links.loc[m, "speed_ab"] = np.minimum(links.loc[m, "speed_ab"], cap)
        links.loc[m, "speed_ba"] = np.minimum(links.loc[m, "speed_ba"], cap)
        after_ab = links.loc[m, "speed_ab"].mean()
        if before_ab != after_ab:
            logger.debug("speed_cap %s: %.1f -> %.1f km/h (cap=%s)", lt, before_ab, after_ab, cap)


def _apply_capacity_factors(links: pd.DataFrame, capacity_factors: dict) -> None:
    for lt, factor in capacity_factors.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "capacity_ab"] = links.loc[m, "capacity_ab"] * factor
        links.loc[m, "capacity_ba"] = links.loc[m, "capacity_ba"] * factor


def _apply_time_penalties(links: pd.DataFrame, time_penalties: dict) -> None:
    for lt, penalty_s in time_penalties.items():
        m = _lt_mask(links, lt)
        if not m.any():
            continue
        links.loc[m, "travel_time_ab"] = links.loc[m, "travel_time_ab"] + penalty_s
        links.loc[m, "travel_time_ba"] = links.loc[m, "travel_time_ba"] + penalty_s


def _apply_experiment_profile(
    links: pd.DataFrame,
    profile: dict,
    thresholds: dict,
) -> pd.DataFrame:
    _apply_speed_floors(links, profile.get("speed_floors") or {})
    _apply_speed_caps(links, profile.get("speed_caps") or {})
    _apply_capacity_factors(links, profile.get("capacity_factors") or {})

    min_spd = _threshold(thresholds, "min_speed_kmh", 5.0)
    min_cap = _threshold(thresholds, "min_capacity_vph", 50.0)
    links["speed_ab"] = links["speed_ab"].clip(lower=min_spd)
    links["speed_ba"] = links["speed_ba"].clip(lower=min_spd)
    links["capacity_ab"] = links["capacity_ab"].clip(lower=min_cap)
    links["capacity_ba"] = links["capacity_ba"].clip(lower=min_cap)
    return links


def ensure_link_types_registered(project: Project, link_types: set[str]) -> None:
    """Insert missing ``*_link`` types into AequilibraE's ``link_types`` table."""
    with project_db(project) as conn:
        existing = {r[0] for r in conn.execute("SELECT link_type FROM link_types").fetchall()}
        for lt in sorted(link_types):
            if lt in existing:
                continue
            lt_id = _LINK_TYPE_IDS.get(lt, lt[0].upper())
            used_ids = {r[0] for r in conn.execute("SELECT link_type_id FROM link_types").fetchall()}
            if lt_id in used_ids:
                for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
                    if c not in used_ids:
                        lt_id = c
                        break
            conn.execute(
                "INSERT INTO link_types (link_type, link_type_id, description) VALUES (?, ?, ?)",
                (lt, lt_id, f"Ramp/connector: {lt}"),
            )
            logger.debug("Registered link_type %r (id=%r) in link_types table", lt, lt_id)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_network_attributes(
    project: Project,
    network_cfg: dict,
    experiment_profile: str = "baseline",
    csd_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Normalize/compute link attributes, preserving AB/BA directional asymmetry.

    Writes back per-direction values (speed_ab, speed_ba, ...) to the SQLite DB.
    If *csd_path* is provided, CSD AADT data is used to set minimum capacities
    on matched roads.
    """
    defaults, thresholds = _normalization_defaults(network_cfg)
    default_speeds = defaults["speed_by_link_type"]
    default_lanes = defaults["lanes_by_link_type"]
    cap_per_lane = defaults["capacity_per_lane_by_link_type"]
    fallback_speed = _threshold(thresholds, "fallback_speed_kmh", 50.0)
    generic_cpl = _threshold(thresholds, "generic_capacity_per_lane", 900.0)
    min_tt = _threshold(thresholds, "min_travel_time_s", 0.01)

    profile = _resolved_experiment_profile(network_cfg, experiment_profile)
    logger.info("Applying network experiment profile: %s", experiment_profile)

    links = project.network.links.data.copy()

    # AequilibraE collapses *_link to parent; osm_highway restores fine type
    if "osm_highway" in links.columns:
        _RESTORABLE = {
            "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link",
        }
        osm_hw = links["osm_highway"].astype(str)
        restore_mask = osm_hw.isin(_RESTORABLE) & (links["link_type"] != osm_hw)
        n_restored = int(restore_mask.sum())
        if n_restored > 0:
            ensure_link_types_registered(project, _RESTORABLE)
            links.loc[restore_mask, "link_type"] = osm_hw[restore_mask]
            logger.info("Restored link_type from osm_highway for %d *_link links", n_restored)

    # Reclassify construction links
    constr_mask = links["link_type"].astype(str) == "construction"
    n_constr = int(constr_mask.sum())
    if n_constr > 0:
        spd = pd.to_numeric(links.loc[constr_mask, "speed_ab"], errors="coerce").fillna(50)
        lanes = pd.to_numeric(links.loc[constr_mask, "lanes_ab"], errors="coerce").fillna(1)
        dirn = links.loc[constr_mask, "direction"].fillna(0).astype(int)
        hw_mask = constr_mask.copy()
        hw_mask[:] = False
        hw_mask.loc[constr_mask[constr_mask].index] = (spd >= 80).values & (dirn == 1).values
        hw_1lane = hw_mask & (lanes <= 1)
        hw_multi = hw_mask & (lanes > 1)
        links.loc[hw_1lane, "link_type"] = "motorway_link"
        links.loc[hw_multi, "link_type"] = "trunk_link"
        still_constr = links["link_type"].astype(str) == "construction"
        links.loc[still_constr, "link_type"] = "residential"
        logger.info(
            "Reclassified %d construction links (%d motorway_link, %d trunk_link, %d residential)",
            n_constr, int(hw_1lane.sum()), int(hw_multi.sum()), int(still_constr.sum()),
        )

    # Per-road link_type overrides from experiment profile
    lt_overrides = profile.get("link_type_overrides") or []
    if lt_overrides and "osm_ref_norm" in links.columns:
        for ovr in lt_overrides:
            match_spec = ovr.get("match") or {}
            new_lt = ovr.get("set_link_type")
            if not match_spec or not new_lt:
                continue
            mask = pd.Series(True, index=links.index)
            for col, val in match_spec.items():
                if col in links.columns:
                    mask &= links[col].astype(str).str.lower() == str(val).lower()
                else:
                    mask[:] = False
            n_ovr = int(mask.sum())
            if n_ovr > 0:
                ensure_link_types_registered(project, {new_lt})
                links.loc[mask, "link_type"] = new_lt
                logger.info("Reclassified %d links matching %s to %s", n_ovr, match_spec, new_lt)
                min_spd = ovr.get("min_speed")
                if min_spd is not None:
                    min_spd = float(min_spd)
                    spd_ab = pd.to_numeric(links.loc[mask, "speed_ab"], errors="coerce")
                    spd_ba = pd.to_numeric(links.loc[mask, "speed_ba"], errors="coerce")
                    n_raised = int((spd_ab < min_spd).sum())
                    links.loc[mask, "speed_ab"] = np.where(spd_ab.isna(), min_spd, np.maximum(spd_ab, min_spd))
                    links.loc[mask, "speed_ba"] = np.where(spd_ba.isna(), min_spd, np.maximum(spd_ba, min_spd))
                    logger.info("Applied min_speed=%s km/h (%d links raised)", min_spd, n_raised)

    # Remove crossing links
    crossing_mask = links["link_type"].astype(str) == "crossing"
    n_crossing = int(crossing_mask.sum())
    if n_crossing > 0:
        crossing_ids = links.loc[crossing_mask, "link_id"].tolist()
        links = links[~crossing_mask].copy()
        with project_db(project) as conn:
            ph = ",".join("?" * len(crossing_ids))
            conn.execute(f"DELETE FROM links WHERE link_id IN ({ph})", crossing_ids)
        logger.info("Removed %d crossing links from network", n_crossing)

    links["estimated_distance"] = 0
    links["estimated_speed_ab"] = 0
    links["estimated_speed_ba"] = 0
    links["estimated_capacity_ab"] = 0
    links["estimated_capacity_ba"] = 0
    links["estimated_lanes_ab"] = 0
    links["estimated_lanes_ba"] = 0
    links["estimated_free_flow_time_ab"] = 0
    links["estimated_free_flow_time_ba"] = 0

    mask_twoway = links["direction"] == 0

    # --- Distance (from geometry if missing) ---
    if "distance" not in links.columns:
        links["distance"] = None
    missing_dist = links["distance"].isna() | (links["distance"] == 0)
    if missing_dist.any():
        links.loc[missing_dist, "distance"] = links.loc[missing_dist, "geometry"].length
        links.loc[missing_dist, "estimated_distance"] = 1

    def _resolve_directional(
        col_ab: str, col_ba: str, default_map: dict,
        fallback_val: float, est_ab: str, est_ba: str,
    ) -> None:
        if col_ab in links.columns:
            links[col_ab] = pd.to_numeric(links[col_ab], errors="coerce")
        else:
            links[col_ab] = None
        if col_ba in links.columns:
            links[col_ba] = pd.to_numeric(links[col_ba], errors="coerce")
        else:
            links[col_ba] = None

        ab_missing = links[col_ab].isna()
        ba_missing = links[col_ba].isna()
        tw_ab_only = mask_twoway & (~ab_missing) & ba_missing
        tw_ba_only = mask_twoway & ab_missing & (~ba_missing)
        links.loc[tw_ab_only, col_ba] = links.loc[tw_ab_only, col_ab]
        links.loc[tw_ab_only, est_ba] = 1
        links.loc[tw_ba_only, col_ab] = links.loc[tw_ba_only, col_ba]
        links.loc[tw_ba_only, est_ab] = 1

        still_ab_missing = links[col_ab].isna()
        still_ba_missing = links[col_ba].isna()
        if "link_type" in links.columns:
            for lt, dv in default_map.items():
                m_ab = still_ab_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                m_ba = still_ba_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                links.loc[m_ab, col_ab] = dv
                links.loc[m_ab, est_ab] = 1
                links.loc[m_ba, col_ba] = dv
                links.loc[m_ba, est_ba] = 1

        links.loc[links[col_ab].isna(), est_ab] = 1
        links.loc[links[col_ab].isna(), col_ab] = fallback_val
        links.loc[links[col_ba].isna(), est_ba] = 1
        links.loc[links[col_ba].isna(), col_ba] = fallback_val

    _resolve_directional("speed_ab", "speed_ba", default_speeds, fallback_speed, "estimated_speed_ab", "estimated_speed_ba")
    _resolve_directional("lanes_ab", "lanes_ba", default_lanes, 1, "estimated_lanes_ab", "estimated_lanes_ba")
    links["lanes_ab"] = links["lanes_ab"].clip(lower=1)
    links["lanes_ba"] = links["lanes_ba"].clip(lower=1)

    # --- Practical free-flow speed reduction (HCM-inspired) ---
    pffs_defaults = _NORM_DEFS.get("practical_speed") or {}
    pffs_profile = profile.get("practical_speed") or {}
    pffs_cfg = {**pffs_defaults, **pffs_profile}
    _apply_practical_speed_reduction(links, project, pffs_cfg)

    # --- Capacity (AB/BA) ---
    if "capacity_ab" in links.columns:
        links["capacity_ab"] = pd.to_numeric(links["capacity_ab"], errors="coerce")
    else:
        links["capacity_ab"] = None
    if "capacity_ba" in links.columns:
        links["capacity_ba"] = pd.to_numeric(links["capacity_ba"], errors="coerce")
    else:
        links["capacity_ba"] = None

    tw_cab = mask_twoway & links["capacity_ab"].notna() & links["capacity_ba"].isna()
    tw_cba = mask_twoway & links["capacity_ba"].notna() & links["capacity_ab"].isna()
    links.loc[tw_cab, "capacity_ba"] = links.loc[tw_cab, "capacity_ab"]
    links.loc[tw_cab, "estimated_capacity_ba"] = 1
    links.loc[tw_cba, "capacity_ab"] = links.loc[tw_cba, "capacity_ba"]
    links.loc[tw_cba, "estimated_capacity_ab"] = 1

    for suffix, est_col, lanes_col in [("ab", "estimated_capacity_ab", "lanes_ab"), ("ba", "estimated_capacity_ba", "lanes_ba")]:
        cap_col = f"capacity_{suffix}"
        cap_missing = links[cap_col].isna() | (links[cap_col] == 0)
        if cap_missing.any() and "link_type" in links.columns:
            for lt, cpl in cap_per_lane.items():
                m = cap_missing & links["link_type"].astype(str).str.fullmatch(lt, case=False)
                links.loc[m, cap_col] = float(cpl) * links.loc[m, lanes_col]
                links.loc[m, est_col] = 1
            remaining = links[cap_col].isna() | (links[cap_col] == 0)
            if remaining.any():
                links.loc[remaining, cap_col] = generic_cpl * links.loc[remaining, lanes_col]
                links.loc[remaining, est_col] = 1

    # --- CSD capacity hints (before experiment profile so caps/factors apply on top) ---
    if csd_path is not None:
        phf = float(network_cfg.get("csd_peak_hour_factor", 0.10))
        _apply_csd_capacity_hints(links, csd_path, peak_hour_factor=phf)

    links = _apply_experiment_profile(links, profile, thresholds)

    # --- Free-flow travel time (AB/BA) ---
    if "travel_time_ab" in links.columns:
        links["travel_time_ab"] = pd.to_numeric(links["travel_time_ab"], errors="coerce")
    else:
        links["travel_time_ab"] = None
    if "travel_time_ba" in links.columns:
        links["travel_time_ba"] = pd.to_numeric(links["travel_time_ba"], errors="coerce")
    else:
        links["travel_time_ba"] = None

    tw_tab = mask_twoway & links["travel_time_ab"].notna() & links["travel_time_ba"].isna()
    tw_tba = mask_twoway & links["travel_time_ba"].notna() & links["travel_time_ab"].isna()
    links.loc[tw_tab, "travel_time_ba"] = links.loc[tw_tab, "travel_time_ab"]
    links.loc[tw_tab, "estimated_free_flow_time_ba"] = 1
    links.loc[tw_tba, "travel_time_ab"] = links.loc[tw_tba, "travel_time_ba"]
    links.loc[tw_tba, "estimated_free_flow_time_ab"] = 1

    for tt_col, spd_col in [("travel_time_ab", "speed_ab"), ("travel_time_ba", "speed_ba")]:
        valid_spd = links[spd_col].notna() & (links[spd_col] > 0) & links["distance"].notna() & (links["distance"] > 0)
        links.loc[valid_spd, tt_col] = links.loc[valid_spd, "distance"] * 3.6 / links.loc[valid_spd, spd_col]

    _apply_time_penalties(links, profile.get("time_penalties") or {})
    for tt_col in ("travel_time_ab", "travel_time_ba"):
        links[tt_col] = links[tt_col].clip(lower=min_tt)

    # --- Write link attributes to DB ---
    db_path = project_db_path(project)
    if not db_path.is_file():
        logger.warning("Project database not found at %s, DB update skipped", db_path)
        return links

    _needed = ["link_type", "speed_ab", "speed_ba", "lanes_ab", "lanes_ba",
               "capacity_ab", "capacity_ba", "travel_time_ab", "travel_time_ba", "link_id"]
    updates = list(links[_needed].itertuples(index=False, name=None))

    with project_db(project) as conn:
        conn.executemany(
            "UPDATE links SET link_type=?, speed_ab=?, speed_ba=?, lanes_ab=?, lanes_ba=?, "
            "capacity_ab=?, capacity_ba=?, travel_time_ab=?, travel_time_ba=? "
            "WHERE link_id=?",
            updates,
        )

    n_asym_speed = int(((links["speed_ab"] - links["speed_ba"]).abs() > 0.1).sum())
    n_asym_lanes = int((links["lanes_ab"] != links["lanes_ba"]).sum())
    n_asym_cap = int(((links["capacity_ab"] - links["capacity_ba"]).abs() > 0.1).sum())
    n_est_speed = int((links["estimated_speed_ab"] | links["estimated_speed_ba"]).sum())
    n_est_cap = int((links["estimated_capacity_ab"] | links["estimated_capacity_ba"]).sum())

    logger.info("Updated %d links in DB (per-direction)", len(updates))
    logger.info(
        "Asymmetric: speed=%d, lanes=%d, capacity=%d; estimated: speed=%d, capacity=%d",
        n_asym_speed, n_asym_lanes, n_asym_cap, n_est_speed, n_est_cap,
    )
    return links
