from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from aequilibrae import Project

from sim.aequilibrae_paths import resolve_project_database_path
from sim.io_project import get_metric_epsg, load_config

from sim.defaults import NETWORK_NORM_DEFAULTS

_MAJOR_ROAD_TYPES = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})

# ---------------------------------------------------------------------------
# Czech-Republic normalization defaults (from centralized defaults.py).
# YAML values (if provided) merge on top of these.
# ---------------------------------------------------------------------------

_NORM_DEFS = NETWORK_NORM_DEFAULTS["normalization"]
_DEFAULT_THRESHOLDS: Dict[str, float] = _NORM_DEFS["thresholds"]
_DEFAULT_SPEED_BY_LINK_TYPE: Dict[str, float] = _NORM_DEFS["defaults"]["speed_by_link_type"]
_DEFAULT_LANES_BY_LINK_TYPE: Dict[str, int] = _NORM_DEFS["defaults"]["lanes_by_link_type"]
_DEFAULT_CAPACITY_PER_LANE: Dict[str, int] = _NORM_DEFS["defaults"]["capacity_per_lane_by_link_type"]


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
            print(f"  speed_cap {lt}: {before_ab:.1f} → {after_ab:.1f} km/h (cap={cap})")


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


_LINK_TYPE_IDS = {
    "motorway_link": "M",
    "trunk_link": "K",
    "primary_link": "D",
    "secondary_link": "L",
    "tertiary_link": "V",
}


def _ensure_link_types_registered(
    project: Project, link_types: set[str]
) -> None:
    """Insert missing *_link types into AequilibraE's link_types table."""
    db_path = resolve_project_database_path(project)
    conn = sqlite3.connect(str(db_path))
    existing = {
        r[0] for r in conn.execute("SELECT link_type FROM link_types").fetchall()
    }
    for lt in sorted(link_types):
        if lt in existing:
            continue
        lt_id = _LINK_TYPE_IDS.get(lt, lt[0].upper())
        used_ids = {
            r[0] for r in conn.execute("SELECT link_type_id FROM link_types").fetchall()
        }
        if lt_id in used_ids:
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
                if c not in used_ids:
                    lt_id = c
                    break
        conn.execute(
            "INSERT INTO link_types (link_type, link_type_id, description) VALUES (?, ?, ?)",
            (lt, lt_id, f"Ramp/connector: {lt}"),
        )
        print(f"  Registered link_type '{lt}' (id='{lt_id}') in link_types table")
    conn.commit()
    conn.close()


def normalize_network_attributes(
    project: Project,
    network_cfg: dict,
    experiment_profile: str = "baseline",
) -> pd.DataFrame:
    """Normalize/compute link attributes, preserving AB/BA directional asymmetry.

    For two-way links (direction=0), when both AB and BA values exist from OSM,
    they are kept separately -- never averaged. Only when a value is missing
    is a default or the opposite-direction value used as fallback.

    Writes back per-direction values (speed_ab, speed_ba, ...) to the SQLite DB.
    """
    defaults, thresholds = _normalization_defaults(network_cfg)
    default_speeds = defaults["speed_by_link_type"]
    default_lanes = defaults["lanes_by_link_type"]
    cap_per_lane = defaults["capacity_per_lane_by_link_type"]
    fallback_speed = _threshold(thresholds, "fallback_speed_kmh", 50.0)
    generic_cpl = _threshold(thresholds, "generic_capacity_per_lane", 900.0)
    min_tt = _threshold(thresholds, "min_travel_time_s", 0.01)

    profile = _resolved_experiment_profile(network_cfg, experiment_profile)
    print(f"Applying network experiment profile: {experiment_profile}")

    links = project.network.links.data.copy()

    # Restore fine-grained link_type from osm_highway where AequilibraE
    # collapsed *_link types into their parent (motorway_link → motorway).
    # This ensures normalization defaults (speed, capacity, lanes) are applied
    # correctly per actual OSM road class.
    if "osm_highway" in links.columns:
        _RESTORABLE = {
            "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link",
        }
        osm_hw = links["osm_highway"].astype(str)
        restore_mask = osm_hw.isin(_RESTORABLE) & (links["link_type"] != osm_hw)
        n_restored = int(restore_mask.sum())
        if n_restored > 0:
            _ensure_link_types_registered(project, _RESTORABLE)
            links.loc[restore_mask, "link_type"] = osm_hw[restore_mask]
            print(f"  Restored link_type from osm_highway for {n_restored} *_link links")

    # Reclassify construction links to their target road class so the
    # model represents the ideal network (all roads as designed, no closures).
    # osm_highway is None for all construction links, so we infer the target
    # class from speed / lanes / direction.
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
        print(f"  Reclassified {n_constr} construction links "
              f"({int(hw_1lane.sum())} motorway_link, {int(hw_multi.sum())} trunk_link, "
              f"{int(still_constr.sum())} residential)")

    # Apply per-road link_type overrides from experiment profile.
    # Fixes OSM misclassifications (e.g. trunk roads tagged as secondary).
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
                _ensure_link_types_registered(project, {new_lt})
                links.loc[mask, "link_type"] = new_lt
                print(f"  Reclassified {n_ovr} links matching {match_spec} → {new_lt}")
                min_spd = ovr.get("min_speed")
                if min_spd is not None:
                    min_spd = float(min_spd)
                    spd_ab = pd.to_numeric(links.loc[mask, "speed_ab"], errors="coerce")
                    spd_ba = pd.to_numeric(links.loc[mask, "speed_ba"], errors="coerce")
                    n_raised = int((spd_ab < min_spd).sum())
                    links.loc[mask, "speed_ab"] = np.where(
                        spd_ab.isna(), min_spd, np.maximum(spd_ab, min_spd)
                    )
                    links.loc[mask, "speed_ba"] = np.where(
                        spd_ba.isna(), min_spd, np.maximum(spd_ba, min_spd)
                    )
                    print(f"    Applied min_speed={min_spd} km/h ({n_raised} links raised)")

    # Remove crossing links (pedestrian crossing markup, not a road segment).
    crossing_mask = links["link_type"].astype(str) == "crossing"
    n_crossing = int(crossing_mask.sum())
    if n_crossing > 0:
        crossing_ids = links.loc[crossing_mask, "link_id"].tolist()
        links = links[~crossing_mask].copy()
        db_path = resolve_project_database_path(project)
        _conn = sqlite3.connect(str(db_path))
        ph = ",".join("?" * len(crossing_ids))
        _conn.execute(f"DELETE FROM links WHERE link_id IN ({ph})", crossing_ids)
        _conn.commit()
        _conn.close()
        print(f"  Removed {n_crossing} crossing links from network")

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

    # ------------------------------------------------------------------
    # 1. Distance (from geometry if missing)
    # ------------------------------------------------------------------
    if "distance" not in links.columns:
        links["distance"] = None
    missing_dist = links["distance"].isna() | (links["distance"] == 0)
    if missing_dist.any():
        links.loc[missing_dist, "distance"] = links.loc[missing_dist, "geometry"].length
        links.loc[missing_dist, "estimated_distance"] = 1

    # ------------------------------------------------------------------
    # Helper: resolve per-direction attribute from OSM columns
    # ------------------------------------------------------------------
    def _resolve_directional(
        col_ab: str,
        col_ba: str,
        default_map: dict,
        fallback_val: float,
        est_ab: str,
        est_ba: str,
    ) -> None:
        """Populate col_ab / col_ba keeping asymmetry, filling missing with defaults."""
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

        # Two-way: if one direction present but other missing, copy
        tw_ab_only = mask_twoway & (~ab_missing) & ba_missing
        tw_ba_only = mask_twoway & ab_missing & (~ba_missing)
        links.loc[tw_ab_only, col_ba] = links.loc[tw_ab_only, col_ab]
        links.loc[tw_ab_only, est_ba] = 1
        links.loc[tw_ba_only, col_ab] = links.loc[tw_ba_only, col_ba]
        links.loc[tw_ba_only, est_ab] = 1

        # Fill remaining missing from defaults by link_type
        still_ab_missing = links[col_ab].isna()
        still_ba_missing = links[col_ba].isna()
        if "link_type" in links.columns:
            for lt, dv in default_map.items():
                m_ab = still_ab_missing & links["link_type"].astype(str).str.fullmatch(
                    lt, case=False
                )
                m_ba = still_ba_missing & links["link_type"].astype(str).str.fullmatch(
                    lt, case=False
                )
                links.loc[m_ab, col_ab] = dv
                links.loc[m_ab, est_ab] = 1
                links.loc[m_ba, col_ba] = dv
                links.loc[m_ba, est_ba] = 1

        # Final fallback
        links.loc[links[col_ab].isna(), est_ab] = 1
        links.loc[links[col_ab].isna(), col_ab] = fallback_val
        links.loc[links[col_ba].isna(), est_ba] = 1
        links.loc[links[col_ba].isna(), col_ba] = fallback_val

    # ------------------------------------------------------------------
    # 2. Speed (preserve AB/BA)
    # ------------------------------------------------------------------
    _resolve_directional(
        "speed_ab",
        "speed_ba",
        default_speeds,
        fallback_speed,
        "estimated_speed_ab",
        "estimated_speed_ba",
    )

    # ------------------------------------------------------------------
    # 3. Lanes (preserve AB/BA)
    # ------------------------------------------------------------------
    _resolve_directional(
        "lanes_ab",
        "lanes_ba",
        default_lanes,
        1,
        "estimated_lanes_ab",
        "estimated_lanes_ba",
    )
    links["lanes_ab"] = links["lanes_ab"].clip(lower=1)
    links["lanes_ba"] = links["lanes_ba"].clip(lower=1)

    # ------------------------------------------------------------------
    # 4. Capacity (preserve AB/BA)
    # Total veh/h per direction: existing OSM/Aeq values kept; gaps filled as
    # capacity_* = capacity_per_lane[link_type] * lanes_* (see network_normalization.yaml).
    # ------------------------------------------------------------------
    if "capacity_ab" in links.columns:
        links["capacity_ab"] = pd.to_numeric(links["capacity_ab"], errors="coerce")
    else:
        links["capacity_ab"] = None

    if "capacity_ba" in links.columns:
        links["capacity_ba"] = pd.to_numeric(links["capacity_ba"], errors="coerce")
    else:
        links["capacity_ba"] = None

    # Two-way: copy present direction to missing one
    tw_cab = mask_twoway & links["capacity_ab"].notna() & links["capacity_ba"].isna()
    tw_cba = mask_twoway & links["capacity_ba"].notna() & links["capacity_ab"].isna()
    links.loc[tw_cab, "capacity_ba"] = links.loc[tw_cab, "capacity_ab"]
    links.loc[tw_cab, "estimated_capacity_ba"] = 1
    links.loc[tw_cba, "capacity_ab"] = links.loc[tw_cba, "capacity_ba"]
    links.loc[tw_cba, "estimated_capacity_ab"] = 1

    # Estimate missing from capacity_per_lane * lanes
    for suffix, est_col, lanes_col in [
        ("ab", "estimated_capacity_ab", "lanes_ab"),
        ("ba", "estimated_capacity_ba", "lanes_ba"),
    ]:
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

    # ------------------------------------------------------------------
    # 4.5 Experiment profile adjustments (speed/capacity hierarchy)
    # ------------------------------------------------------------------
    links = _apply_experiment_profile(links, profile, thresholds)

    # ------------------------------------------------------------------
    # 5. Free-flow travel time (preserve AB/BA)
    # ------------------------------------------------------------------
    if "travel_time_ab" in links.columns:
        links["travel_time_ab"] = pd.to_numeric(links["travel_time_ab"], errors="coerce")
    else:
        links["travel_time_ab"] = None

    if "travel_time_ba" in links.columns:
        links["travel_time_ba"] = pd.to_numeric(links["travel_time_ba"], errors="coerce")
    else:
        links["travel_time_ba"] = None

    # Two-way: copy present direction to missing
    tw_tab = mask_twoway & links["travel_time_ab"].notna() & links["travel_time_ba"].isna()
    tw_tba = mask_twoway & links["travel_time_ba"].notna() & links["travel_time_ab"].isna()
    links.loc[tw_tab, "travel_time_ba"] = links.loc[tw_tab, "travel_time_ab"]
    links.loc[tw_tab, "estimated_free_flow_time_ba"] = 1
    links.loc[tw_tba, "travel_time_ab"] = links.loc[tw_tba, "travel_time_ba"]
    links.loc[tw_tba, "estimated_free_flow_time_ab"] = 1

    # ALWAYS recompute travel time from distance/speed to ensure consistency.
    for tt_col, spd_col in [("travel_time_ab", "speed_ab"), ("travel_time_ba", "speed_ba")]:
        valid_spd = (
            links[spd_col].notna()
            & (links[spd_col] > 0)
            & links["distance"].notna()
            & (links["distance"] > 0)
        )
        links.loc[valid_spd, tt_col] = (
            links.loc[valid_spd, "distance"] * 3.6 / links.loc[valid_spd, spd_col]
        )

    # ------------------------------------------------------------------
    # 5.5 Experiment profile adjustments (time penalties)
    # ------------------------------------------------------------------
    _apply_time_penalties(links, profile.get("time_penalties") or {})

    # Floor travel time to prevent zero-cost links in BPR
    for tt_col in ("travel_time_ab", "travel_time_ba"):
        links[tt_col] = links[tt_col].clip(lower=min_tt)

    # ------------------------------------------------------------------
    # DB write-back (per-direction, preserving asymmetry)
    # ------------------------------------------------------------------
    db_path = resolve_project_database_path(project)
    if not db_path.is_file():
        print(f"Warning: project database not found at {db_path}, DB update skipped")
        return links

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    updates = []
    for _, row in links.iterrows():
        lid = int(row["link_id"])
        updates.append(
            (
                str(row["link_type"]),
                float(row["speed_ab"]),
                float(row["speed_ba"]),
                int(row["lanes_ab"]),
                int(row["lanes_ba"]),
                float(row["capacity_ab"]),
                float(row["capacity_ba"]),
                float(row["travel_time_ab"]),
                float(row["travel_time_ba"]),
                lid,
            )
        )

    cursor.executemany(
        "UPDATE links SET link_type=?, speed_ab=?, speed_ba=?, lanes_ab=?, lanes_ba=?, "
        "capacity_ab=?, capacity_ba=?, travel_time_ab=?, travel_time_ba=? "
        "WHERE link_id=?",
        updates,
    )
    conn.commit()
    conn.close()

    # Diagnostics
    n_asym_speed = int(((links["speed_ab"] - links["speed_ba"]).abs() > 0.1).sum())
    n_asym_lanes = int((links["lanes_ab"] != links["lanes_ba"]).sum())
    n_asym_cap = int(((links["capacity_ab"] - links["capacity_ba"]).abs() > 0.1).sum())
    n_est_speed = int((links["estimated_speed_ab"] | links["estimated_speed_ba"]).sum())
    n_est_cap = int((links["estimated_capacity_ab"] | links["estimated_capacity_ba"]).sum())

    print(f"Updated {len(updates)} links in DB (per-direction)")
    print(f"  Asymmetric: speed={n_asym_speed}, lanes={n_asym_lanes}, capacity={n_asym_cap}")
    print(f"  Estimated:  speed={n_est_speed}, capacity={n_est_cap}")

    return links


def repair_boundary_scc(project: Project) -> Dict[str, Any]:
    """Make one-way motorway/trunk links with nodes outside the largest directed
    SCC bidirectional, so that gateway nodes can participate in directed routing.

    At the model boundary, OSM one-way motorway links often lack the opposing
    carriageway (clipped or exits elsewhere).  This creates directed dead-ends
    that prevent through-traffic routing.  The fix changes ``direction`` from 1
    (or -1) to 0 and copies AB attributes to BA, giving the link a return path.
    Only links with at least one endpoint **outside** the largest directed SCC
    are touched.
    """
    links = project.network.links.data.copy()
    if links.empty:
        return {"repaired": 0, "scc_before": 0, "scc_after": 0,
                "major_outside_before": 0, "major_outside_after": 0,
                "repaired_ids": []}

    # ---- build directed graph and find largest SCC ----
    G = nx.DiGraph()
    for _, lk in links.iterrows():
        a, b = int(lk["a_node"]), int(lk["b_node"])
        d = int(lk.get("direction", 0) or 0)
        if d >= 0:
            G.add_edge(a, b)
        if d <= 0:
            G.add_edge(b, a)

    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return {"repaired": 0, "scc_before": 0, "scc_after": 0,
                "major_outside_before": 0, "major_outside_after": 0,
                "repaired_ids": []}

    largest_scc = max(sccs, key=len)
    scc_before = len(largest_scc)

    # ---- count major links outside SCC before repair ----
    major_outside_before = 0
    for _, lk in links.iterrows():
        if str(lk.get("link_type", "")) not in _MAJOR_ROAD_TYPES:
            continue
        a, b = int(lk["a_node"]), int(lk["b_node"])
        if a not in largest_scc or b not in largest_scc:
            major_outside_before += 1

    # ---- select one-way major links to repair ----
    repair_ids = []
    for _, lk in links.iterrows():
        lt = str(lk.get("link_type", ""))
        if lt not in _MAJOR_ROAD_TYPES:
            continue
        d = int(lk.get("direction", 0) or 0)
        if d == 0:
            continue
        a, b = int(lk["a_node"]), int(lk["b_node"])
        if a in largest_scc and b in largest_scc:
            continue
        repair_ids.append(int(lk["link_id"]))

    if not repair_ids:
        print(f"  Boundary SCC repair: nothing to fix (SCC={scc_before})")
        return {"repaired": 0, "scc_before": scc_before, "scc_after": scc_before,
                "major_outside_before": major_outside_before,
                "major_outside_after": major_outside_before,
                "repaired_ids": []}

    # ---- update DB: direction → 0, sync BA ← AB ----
    db_path = resolve_project_database_path(project)
    conn = sqlite3.connect(str(db_path))
    for lid in repair_ids:
        conn.execute(
            """UPDATE links
               SET direction = 0,
                   speed_ba     = speed_ab,
                   capacity_ba  = capacity_ab,
                   lanes_ba     = lanes_ab,
                   travel_time_ba = travel_time_ab
             WHERE link_id = ? AND direction != 0""",
            (lid,),
        )
    conn.commit()
    conn.close()

    try:
        project.network.refresh()
        project.network.links.refresh()
    except Exception:
        pass

    # ---- re-check SCC ----
    links2 = project.network.links.data
    G2 = nx.DiGraph()
    for _, lk in links2.iterrows():
        a, b = int(lk["a_node"]), int(lk["b_node"])
        d = int(lk.get("direction", 0) or 0)
        if d >= 0:
            G2.add_edge(a, b)
        if d <= 0:
            G2.add_edge(b, a)

    sccs2 = list(nx.strongly_connected_components(G2))
    largest_scc2 = max(sccs2, key=len) if sccs2 else set()
    scc_after = len(largest_scc2)

    major_outside_after = 0
    for _, lk in links2.iterrows():
        if str(lk.get("link_type", "")) not in _MAJOR_ROAD_TYPES:
            continue
        a, b = int(lk["a_node"]), int(lk["b_node"])
        if a not in largest_scc2 or b not in largest_scc2:
            major_outside_after += 1

    print(f"  Boundary SCC repair: made {len(repair_ids)} one-way major links bidirectional")
    print(f"  SCC: {scc_before} → {scc_after} nodes (+{scc_after - scc_before})")
    print(f"  Major links outside SCC: {major_outside_before} → {major_outside_after}")

    return {
        "repaired": len(repair_ids),
        "scc_before": scc_before,
        "scc_after": scc_after,
        "major_outside_before": major_outside_before,
        "major_outside_after": major_outside_after,
        "repaired_ids": repair_ids,
    }


def repair_divided_highway_dead_ends(
    project: Project,
    max_snap_distance_m: float = 600.0,
) -> Dict[str, Any]:
    """Connect dead-end carriageways of divided highways.

    Divided motorways/trunks imported from OSM often have two separate
    carriageways for each direction.  At the model boundary the two ways
    may terminate at different OSM nodes a few metres apart, creating
    dead-end branches that carry no traffic.

    This function detects such dead-ends and adds a short bidirectional
    link between pairs of nearby dead-end nodes that belong to the same
    road (matched by ``osm_ref_norm``).

    Parameters
    ----------
    project : Project
        Open AequilibraE project (links table is modified in-place).
    max_snap_distance_m : float
        Maximum distance (metres) between two dead-end nodes to connect.
    """
    from shapely.geometry import LineString, Point

    db_path = resolve_project_database_path(project)
    conn = sqlite3.connect(str(db_path))

    # 1) Collect motorway link info per node
    rows = conn.execute(
        "SELECT link_id, a_node, b_node, link_type, osm_ref_norm FROM links"
    ).fetchall()

    from collections import defaultdict
    node_major: Dict[int, list] = defaultdict(list)
    node_nonmajor: Dict[int, int] = defaultdict(int)
    link_ref: Dict[int, str] = {}
    link_lt: Dict[int, str] = {}

    _IGNORE = _MAJOR_ROAD_TYPES | {"centroid_connector"}

    for lid, a, b, lt, ref in rows:
        lt_str = str(lt or "")
        if lt_str in _MAJOR_ROAD_TYPES:
            node_major[a].append(lid)
            node_major[b].append(lid)
            link_lt[int(lid)] = lt_str
            if ref:
                link_ref[lid] = str(ref)
        elif lt_str not in _IGNORE:
            node_nonmajor[a] += 1
            node_nonmajor[b] += 1

    # 2) Find dead-end nodes: exactly 1 major link, no non-major/non-connector links
    dead_ends: Dict[int, int] = {}  # node_id -> the single major link_id
    for nid, major_lids in node_major.items():
        if len(major_lids) == 1 and node_nonmajor[nid] == 0:
            dead_ends[nid] = major_lids[0]

    if not dead_ends:
        print("  Divided highway repair: no dead-end carriageway nodes found")
        return {"connected_pairs": 0, "new_link_ids": []}

    # 3) Get node coordinates in a metric CRS for distance calculation
    node_ids = list(dead_ends.keys())
    placeholders = ",".join(str(n) for n in node_ids)
    try:
        node_geo_rows = conn.execute(
            f"SELECT node_id, ST_X(geometry), ST_Y(geometry) FROM nodes "
            f"WHERE node_id IN ({placeholders})"
        ).fetchall()
    except sqlite3.OperationalError:
        node_geo_rows = []

    if not node_geo_rows:
        # SpatiaLite functions may not be available; fall back to geometry blob parsing
        import struct

        def _parse_spatialite_point(blob):
            endian = '<' if blob[1] == 1 else '>'
            x = struct.unpack(endian + 'd', blob[43:51])[0]
            y = struct.unpack(endian + 'd', blob[51:59])[0]
            return x, y

        node_geo_rows = []
        for nid in node_ids:
            blob = conn.execute(
                "SELECT geometry FROM nodes WHERE node_id = ?", (nid,)
            ).fetchone()
            if blob and blob[0]:
                x, y = _parse_spatialite_point(blob[0])
                node_geo_rows.append((nid, x, y))

    if not node_geo_rows:
        conn.close()
        print("  Divided highway repair: could not read node coordinates")
        return {"connected_pairs": 0, "new_link_ids": []}

    # Build coordinate lookup (WGS-84)
    node_coords = {int(r[0]): (float(r[1]), float(r[2])) for r in node_geo_rows}

    # Geodesic distance on WGS84 ellipsoid (metres)
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    # 4) For each dead-end node, get its road ref and link_type (for new connector link)
    dead_ref: Dict[int, str] = {}
    dead_lt: Dict[int, str] = {}
    for nid, lid in dead_ends.items():
        dead_ref[nid] = link_ref.get(lid, "")
        dead_lt[nid] = link_lt.get(int(lid), "motorway")

    # 5) Greedy pair matching: match closest same-ref dead-end pairs
    remaining = set(dead_ends.keys()) & set(node_coords.keys())
    pairs_to_create: list = []

    while remaining:
        best_pair = None
        best_dist = max_snap_distance_m + 1

        remaining_list = sorted(remaining)
        for i, n1 in enumerate(remaining_list):
            ref1 = dead_ref.get(n1, "")
            if not ref1:
                continue
            c1 = node_coords[n1]
            for n2 in remaining_list[i + 1:]:
                ref2 = dead_ref.get(n2, "")
                if ref1 != ref2:
                    continue
                c2 = node_coords[n2]
                _, _, dist = geod.inv(c1[0], c1[1], c2[0], c2[1])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (n1, n2, dist, ref1)

        if best_pair is None or best_dist > max_snap_distance_m:
            break

        n1, n2, dist, ref = best_pair
        remaining.discard(n1)
        remaining.discard(n2)
        pairs_to_create.append((n1, n2, dist, ref))

    # Close read-only connection before writing via AequilibraE API
    conn.close()

    if not pairs_to_create:
        print("  Divided highway repair: no pairs within snap distance")
        return {"connected_pairs": 0, "new_link_ids": []}

    # 6) Create connecting links via AequilibraE API (needs exclusive DB access)
    lt_needed = {dead_lt.get(n1, "motorway") for n1, _, _, _ in pairs_to_create}
    lt_needed |= {dead_lt.get(n2, "motorway") for _, n2, _, _ in pairs_to_create}
    _ensure_link_types_registered(project, lt_needed)

    new_link_ids: list = []
    connected_pairs: list = []

    for n1, n2, dist, ref in pairs_to_create:
        c1, c2 = node_coords[n1], node_coords[n2]
        lt_use = dead_lt.get(n1, dead_lt.get(n2, "motorway"))
        if dead_lt.get(n1) and dead_lt.get(n2) and dead_lt[n1] != dead_lt[n2]:
            lt_use = dead_lt[n1]

        spd = 130.0 if lt_use in ("motorway", "motorway_link") else 90.0
        cap_lane = 2200.0 if lt_use in ("motorway", "motorway_link") else 1800.0
        lanes = 3 if lt_use in ("motorway", "motorway_link") else 2
        tt = (dist / 1000.0) / max(spd, 1.0) * 3600.0 if dist > 0 else 0.01

        links_api = project.network.links
        new_link = links_api.new()
        new_link.geometry = LineString([c1, c2])
        new_link.direction = 0
        new_link.distance = dist
        new_link.modes = "tc"
        new_link.link_type = lt_use
        new_link.speed_ab = spd
        new_link.speed_ba = spd
        new_link.capacity_ab = cap_lane * lanes
        new_link.capacity_ba = cap_lane * lanes
        new_link.lanes_ab = lanes
        new_link.lanes_ba = lanes
        new_link.travel_time_ab = tt
        new_link.travel_time_ba = tt
        new_link.save()

        new_link_ids.append(new_link.link_id)
        connected_pairs.append((n1, n2, round(dist, 1), ref))
        print(
            f"  Connected {ref} dead-end pair: {n1} ↔ {n2} ({dist:.0f}m) → link {new_link.link_id} "
            f"({lt_use})",
        )

    # 7) Update osm_ref and node IDs via SQL (AequilibraE API doesn't expose these)
    conn2 = sqlite3.connect(str(db_path))
    for lid, (n1, n2, _, ref) in zip(new_link_ids, pairs_to_create):
        lt_use = dead_lt.get(n1, dead_lt.get(n2, "motorway"))
        if dead_lt.get(n1) and dead_lt.get(n2) and dead_lt[n1] != dead_lt[n2]:
            lt_use = dead_lt[n1]
        conn2.execute(
            "UPDATE links SET a_node=?, b_node=?, osm_ref=?, osm_ref_norm=?, "
            "osm_highway=? WHERE link_id=?",
            (n1, n2, ref, ref, lt_use, lid),
        )
    conn2.commit()
    conn2.close()

    try:
        project.network.refresh()
        project.network.links.refresh()
    except Exception:
        pass

    print(f"  Divided highway repair: connected {len(connected_pairs)} carriageway pair(s)")
    return {"connected_pairs": len(connected_pairs), "new_link_ids": new_link_ids}


def check_connectivity(project: Project) -> Dict[str, Any]:
    """
    Check network connectivity and identify isolated components.
    Returns information about components and their sizes.
    Includes both undirected component analysis and directed (strongly
    connected component) analysis with warnings for major road links
    outside the largest SCC.
    """
    links = project.network.links.data
    nodes = project.network.nodes.data

    G = nx.DiGraph()

    for _, node in nodes.iterrows():
        G.add_node(node["node_id"])

    for _, link in links.iterrows():
        a_node = link["a_node"]
        b_node = link["b_node"]
        direction = link.get("direction", 1)
        lt = str(link.get("link_type", ""))
        osm_hw = str(link.get("osm_highway", "")) if "osm_highway" in link.index else lt

        attrs = {"link_id": link["link_id"], "link_type": lt, "osm_highway": osm_hw}
        if direction == 1:
            G.add_edge(a_node, b_node, **attrs)
        elif direction == -1:
            G.add_edge(b_node, a_node, **attrs)
        elif direction == 0:
            G.add_edge(a_node, b_node, **attrs)
            G.add_edge(b_node, a_node, **attrs)

    # --- Undirected analysis (original) ---
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    components_sorted = sorted(components, key=len, reverse=True)

    largest_component = components_sorted[0] if components_sorted else set()
    isolated_nodes = [node for comp in components_sorted[1:] for node in comp if len(comp) == 1]
    isolated_components = [comp for comp in components_sorted[1:] if len(comp) > 1]

    # --- Directed (SCC) analysis ---
    sccs = list(nx.strongly_connected_components(G))
    sccs_sorted = sorted(sccs, key=len, reverse=True)
    largest_scc = sccs_sorted[0] if sccs_sorted else set()

    major_outside_scc = 0
    for _, link in links.iterrows():
        lt = str(link.get("link_type", ""))
        osm_hw = str(link.get("osm_highway", "")) if "osm_highway" in link.index else ""
        if lt not in _MAJOR_ROAD_TYPES and osm_hw not in _MAJOR_ROAD_TYPES:
            continue
        a_node, b_node = link["a_node"], link["b_node"]
        if a_node not in largest_scc or b_node not in largest_scc:
            major_outside_scc += 1

    if major_outside_scc > 0:
        print(f"  WARNING: {major_outside_scc} motorway/trunk links are outside "
              f"the largest strongly-connected component ({len(largest_scc)} nodes)")

    result = {
        "total_components": len(components_sorted),
        "largest_component_size": len(largest_component),
        "isolated_nodes_count": len(isolated_nodes),
        "isolated_components_count": len(isolated_components),
        "directed_scc_count": len(sccs_sorted),
        "largest_scc_size": len(largest_scc),
        "major_links_outside_scc": major_outside_scc,
        "components": [
            {
                "component_id": i,
                "size": len(comp),
                "nodes": list(comp),
            }
            for i, comp in enumerate(components_sorted)
        ],
    }

    return result


def export_stable_network(
    project: Project,
    output_path: Path,
    connectivity_info: Dict[str, Any] | None = None,
    normalized_links: pd.DataFrame | None = None,
    output_crs_epsg: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export network with stable node/link IDs to GeoPackage (projected CRS),
    GeoJSON (WGS84 for maps), and Parquet (no geometry).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_crs_epsg is None:
        output_crs_epsg = get_metric_epsg(cfg or {})

    if normalized_links is not None:
        links = normalized_links.copy()
    else:
        links = project.network.links.data.copy()

    link_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    optional_cols = [
        "link_type",
        "name",
        "osm_id",
        "osm_ref",
        "osm_ref_norm",
        "osm_name_raw",
        "osm_highway",
        "speed_ab",
        "speed_ba",
        "lanes_ab",
        "lanes_ba",
        "capacity_ab",
        "capacity_ba",
        "travel_time_ab",
        "travel_time_ba",
        "speed",
        "lanes",
        "capacity",
        "free_flow_time",
    ]
    available_cols = [col for col in link_cols + optional_cols if col in links.columns]

    estimate_cols = [col for col in links.columns if col.startswith("estimated_")]
    available_cols.extend(estimate_cols)

    links_export = links[available_cols].copy()

    export_meta: Dict[str, Any] = {
        "distance_column_units": "meters",
        "projected_crs_epsg": int(output_crs_epsg),
    }

    if "geometry" in links_export.columns:
        links_gdf = gpd.GeoDataFrame(
            links_export,
            geometry="geometry",
            crs=getattr(links_export, "crs", None),
        )
        if links_gdf.crs is None:
            links_gdf = links_gdf.set_crs(epsg=int(output_crs_epsg), allow_override=True)

        gpkg_links = output_path / "network_links.gpkg"
        links_gdf.to_file(gpkg_links, driver="GPKG", layer="links")

        links_wgs84 = links_gdf.to_crs(epsg=4326)
        links_wgs84.to_file(output_path / "network_links.geojson", driver="GeoJSON")

        links_parquet = links_gdf.drop(columns=["geometry"])
        export_meta["network_links.gpkg"] = {
            "crs_epsg": int(output_crs_epsg),
            "aligned_with_distance": True,
        }
        export_meta["network_links.geojson"] = {
            "crs_epsg": 4326,
            "note": "WGS84 for web maps; lengths in degrees — use `distance` (m) or GPKG for metrics.",
        }
    else:
        links_parquet = links_export
        export_meta["network_links.geojson"] = {"skipped": True, "reason": "no geometry column"}

    parquet_path = output_path / "network_links.parquet"
    links_parquet.to_parquet(parquet_path, index=False)

    nodes = project.network.nodes.data.copy()
    node_cols = ["node_id", "osm_id", "is_centroid"]
    if "geometry" in nodes.columns:
        node_cols.append("geometry")
    available_node_cols = [col for col in node_cols if col in nodes.columns]

    nodes_export = nodes[available_node_cols].copy()

    if "geometry" in nodes_export.columns:
        nodes_gdf = gpd.GeoDataFrame(
            nodes_export,
            geometry="geometry",
            crs=getattr(nodes_export, "crs", None),
        )
        if nodes_gdf.crs is None:
            nodes_gdf = nodes_gdf.set_crs(epsg=int(output_crs_epsg), allow_override=True)
        nodes_gdf.to_file(output_path / "network_nodes.gpkg", driver="GPKG", layer="nodes")
        nodes_gdf.to_crs(epsg=4326).to_file(output_path / "network_nodes.geojson", driver="GeoJSON")
        nodes_parquet = nodes_gdf.drop(columns=["geometry"])
        export_meta["network_nodes.gpkg"] = {"crs_epsg": int(output_crs_epsg)}
        export_meta["network_nodes.geojson"] = {"crs_epsg": 4326}
    else:
        nodes_parquet = nodes_export

    nodes_parquet_path = output_path / "network_nodes.parquet"
    nodes_parquet.to_parquet(nodes_parquet_path, index=False)

    if connectivity_info:
        connectivity_path = output_path / "connectivity_report.json"
        connectivity_path.write_text(json.dumps(connectivity_info, indent=2), encoding="utf-8")

    summary = {
        "links_count": len(links),
        "nodes_count": len(nodes),
        "connectivity": connectivity_info,
        "exports": export_meta,
    }
    summary_path = output_path / "network_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Exported network to: {output_path}")
    print(f"  - Links: {len(links)}")
    print(f"  - Nodes: {len(nodes)}")
    if "geometry" in links_export.columns:
        print(f"  - network_links.gpkg (EPSG:{output_crs_epsg}, metres, aligns with distance)")
        print("  - network_links.geojson (EPSG:4326, for maps only)")
    if connectivity_info:
        print(f"  - Components: {connectivity_info['total_components']}")
        print(f"  - Largest component: {connectivity_info['largest_component_size']} nodes")
        if "directed_scc_count" in connectivity_info:
            print(f"  - Directed SCCs: {connectivity_info['directed_scc_count']}")
            print(f"  - Largest SCC: {connectivity_info['largest_scc_size']} nodes")
            if connectivity_info.get("major_links_outside_scc", 0) > 0:
                print(f"  - WARNING: {connectivity_info['major_links_outside_scc']} "
                      f"major links outside largest SCC")


# ---------------------------------------------------------------------------
# Baseline closures
# ---------------------------------------------------------------------------

def _closure_overlaps_period(
    closure: Dict[str, Any],
    period_start: str,
    period_end: str,
) -> bool:
    """Check whether a closure's [start, end] overlaps [period_start, period_end].

    Closures without an ``end`` date are treated as single-day events
    (active only on their ``start`` date).
    """
    c_start = closure.get("start") or "1900-01-01"
    c_end = closure.get("end") or c_start
    return c_start <= period_end and c_end >= period_start


def load_closures(
    source_path: Path,
    *,
    measurement_period: Optional[Dict[str, str]] = None,
    status_whitelist: Optional[list] = None,
) -> List[Dict[str, Any]]:
    """Load closures from parquet or legacy JSON.

    Returns a list of dicts, each with at least ``lon``, ``lat``,
    ``severity`` (``"full"`` / ``"lane_reduction"`` / ``"speed_limit"``),
    and optionally ``start``, ``end``, ``road_ref``.

    Parameters
    ----------
    status_whitelist : list, optional
        If provided, only closures whose ``status`` value (case-insensitive)
        is in this list are returned.
    """
    if not source_path.exists():
        print(f"  Closures file not found: {source_path} — skipping")
        return []

    suffix = source_path.suffix.lower()

    if suffix == ".parquet":
        try:
            gdf = gpd.read_parquet(source_path)
        except Exception:
            gdf = pd.read_parquet(source_path)
            gdf = gpd.GeoDataFrame(gdf)

        if gdf.empty:
            print("  Closures parquet is empty")
            return []

        closures = gdf.to_dict(orient="records")
    elif suffix == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        closures = data.get("closures", [])
    else:
        print(f"  Unsupported closure file format: {suffix}")
        return []

    if not closures:
        print("  Closures file is empty")
        return []

    total = len(closures)
    if measurement_period:
        p_start = measurement_period.get("start", "1900-01-01")
        p_end = measurement_period.get("end", "2099-12-31")
        closures = [c for c in closures if _closure_overlaps_period(c, p_start, p_end)]
        print(f"  Loaded {total} closures, {len(closures)} overlap measurement period {p_start}..{p_end}")
    else:
        print(f"  Loaded {total} closures (no temporal filter)")

    if status_whitelist:
        allowed = {str(s).strip().lower() for s in status_whitelist}
        before = len(closures)
        closures = [
            c for c in closures
            if str(c.get("status", "")).strip().lower() in allowed
        ]
        print(
            f"  Status filter ({', '.join(sorted(allowed))}): "
            f"{before} -> {len(closures)} closures"
        )

    return closures


def apply_baseline_closures(
    links: pd.DataFrame,
    closures: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Match closures to network links and reduce capacity/speed.

    Stores pre-closure values in ``_preclosure_*`` columns for later
    restoration by :func:`strip_closures`.

    Matching strategy:

    1. Spatial: find network links within ``max_distance_m`` of the
       closure point.
    2. If the closure carries a ``road_ref`` and the config option
       ``require_road_ref_match`` is true, only links whose
       ``osm_ref_norm`` matches the road reference are affected.
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
            # Fill NaN preclosure values (e.g. centroid connectors that were
            # never part of a previous closure cycle) from current values.
            nan_mask = links[col].isna()
            if nan_mask.any():
                links.loc[nan_mask, col] = links.loc[nan_mask, src]

    closure_pts = gpd.GeoDataFrame(
        closures,
        geometry=[Point(float(c.get("lon", 0)), float(c.get("lat", 0))) for c in closures],
        crs="EPSG:4326",
    ).to_crs(epsg=metric_epsg)

    if hasattr(links, "geometry") and links.geometry is not None and not links.geometry.isna().all():
        link_gdf = gpd.GeoDataFrame(links, crs=metric_epsg if links.crs is None else links.crs)
        if link_gdf.crs is None:
            link_gdf = link_gdf.set_crs(epsg=metric_epsg)
        elif link_gdf.crs.to_epsg() != metric_epsg:
            link_gdf = link_gdf.to_crs(epsg=metric_epsg)

        # Detect CRS mismatch: if link coordinates look like WGS84 degrees
        # (typical range 0–180) but CRS is metric, reproject from 4326.
        sample_x = link_gdf.geometry.iloc[0].coords[0][0] if len(link_gdf) > 0 else 0
        if link_gdf.crs and link_gdf.crs.to_epsg() == metric_epsg and abs(sample_x) < 360:
            print(f"  WARNING: link coords look like WGS84 (x={sample_x:.4f}) "
                  f"but CRS is EPSG:{metric_epsg} — reprojecting from 4326")
            link_gdf = link_gdf.set_crs(epsg=4326, allow_override=True)
            link_gdf = link_gdf.to_crs(epsg=metric_epsg)
    else:
        link_gdf = None

    link_ref_col = "osm_ref_norm" if (link_gdf is not None and "osm_ref_norm" in link_gdf.columns) else None

    affected_link_ids: Dict[int, str] = {}

    if link_gdf is not None and not link_gdf.geometry.isna().all():
        sindex = link_gdf.sindex
        for idx, cpt in closure_pts.iterrows():
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
        print("  WARNING: No link geometries available for spatial closure matching")

    if not affected_link_ids:
        # Diagnostic: report coordinate ranges to aid CRS debugging
        if link_gdf is not None and len(closure_pts) > 0:
            lb = link_gdf.total_bounds
            cb = closure_pts.total_bounds
            print(f"  No closures matched to network links "
                  f"(link bounds=[{lb[0]:.0f},{lb[1]:.0f},{lb[2]:.0f},{lb[3]:.0f}], "
                  f"closure bounds=[{cb[0]:.0f},{cb[1]:.0f},{cb[2]:.0f},{cb[3]:.0f}], "
                  f"buffer={max_dist_m}m)")
        else:
            print("  No closures matched to network links")
        return links

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
            valid = links.loc[mask, spd_col] > 0
            if valid.any():
                sub = mask & (links[spd_col] > 0)
                links.loc[sub, tt_col] = links.loc[sub, "distance"] * 3.6 / links.loc[sub, spd_col]

    # Ensure no NaN capacity/speed survives (connectors, missing data, etc.)
    min_cap = 50.0
    min_spd = 5.0
    for col in ("capacity_ab", "capacity_ba"):
        n_nan = int(links[col].isna().sum())
        if n_nan:
            links[col] = links[col].fillna(min_cap)
            print(f"  WARNING: filled {n_nan} NaN values in {col} with {min_cap}")
    for col in ("speed_ab", "speed_ba"):
        n_nan = int(links[col].isna().sum())
        if n_nan:
            links[col] = links[col].fillna(min_spd)
            print(f"  WARNING: filled {n_nan} NaN values in {col} with {min_spd}")

    full_count = sum(1 for s in affected_link_ids.values() if s == "full")
    partial_count = len(affected_link_ids) - full_count
    print(f"  Applied {len(affected_link_ids)} baseline closures ({full_count} full, {partial_count} partial)")
    return links


def strip_closures(
    config_path: str | Path = "config/brno/sim.yaml",
) -> None:
    """Restore pre-closure capacity/speed/travel_time in the project DB.

    This is the inverse of :func:`apply_baseline_closures`.  It reads
    ``_preclosure_*`` columns from the network link table, writes back the
    original values, and removes the ``_preclosure_*`` columns.
    """
    import sqlite3 as _sqlite3

    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    db_path = resolve_project_database_path(project_dir)
    if not db_path.is_file():
        print(f"Project DB not found at {db_path}")
        return

    conn = _sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Check if _preclosure columns exist
    cur.execute("PRAGMA table_info(links)")
    existing_cols = {row[1] for row in cur.fetchall()}
    preclosure_cols = {
        "_preclosure_capacity_ab", "_preclosure_capacity_ba",
        "_preclosure_speed_ab", "_preclosure_speed_ba",
    }
    if not preclosure_cols.issubset(existing_cols):
        print("  No _preclosure columns found — closures were not applied or already stripped")
        conn.close()
        return

    cur.execute(
        "UPDATE links SET "
        "  capacity_ab = _preclosure_capacity_ab, "
        "  capacity_ba = _preclosure_capacity_ba, "
        "  speed_ab = _preclosure_speed_ab, "
        "  speed_ba = _preclosure_speed_ba, "
        "  travel_time_ab = CASE WHEN speed_ab > 0 "
        "    THEN distance * 3.6 / _preclosure_speed_ab ELSE travel_time_ab END, "
        "  travel_time_ba = CASE WHEN speed_ba > 0 "
        "    THEN distance * 3.6 / _preclosure_speed_ba ELSE travel_time_ba END "
        "WHERE _preclosure_capacity_ab IS NOT NULL"
    )
    restored = cur.rowcount

    for col in preclosure_cols:
        try:
            cur.execute(f"ALTER TABLE links DROP COLUMN {col}")
        except Exception:
            pass

    conn.commit()
    conn.close()

    # Re-export the network with restored values
    network_cfg = cfg.get("network") or {}
    outputs_dir = network_cfg.get("output_dir", "outputs/baseline/network")
    project = Project()
    project.open(str(project_dir))
    try:
        links_data = project.network.links.data
        crs_epsg = get_metric_epsg(cfg)
        connectivity_info = check_connectivity(project)
        export_stable_network(
            project,
            Path(outputs_dir),
            connectivity_info,
            normalized_links=links_data,
            output_crs_epsg=crs_epsg,
            cfg=cfg,
        )
    finally:
        project.close()

    print(f"  Stripped closures: restored {restored} links to pre-closure values")
    print(f"  Re-exported network to {outputs_dir}")

    # Re-run assignment on the restored (closure-free) network so that
    # assignment_results.parquet reflects the clean state shown on the map.
    demand_cfg = cfg.get("demand") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    if matrix_path.is_file():
        print("\n=== RE-ASSIGNMENT (post strip-closures) ===")
        from sim.assignment import run_assignment
        run_assignment(config_path)
    else:
        print(f"  WARNING: OD matrix not found at {matrix_path} — skipping post-strip re-assignment")
        print("  The map will show stale volumes computed with closures still active.")


def swap_db_closures(
    config_path: str | Path = "config/brno/sim.yaml",
    measurement_period: Optional[Dict[str, str]] = None,
) -> int:
    """Strip existing closures from the DB and optionally apply new ones.

    This manipulates the AequilibraE SQLite ``links`` table directly:

    1. If ``_preclosure_*`` columns exist, restore original values (strip).
    2. If *measurement_period* is given, load closures filtered to that
       window and apply them (setting ``_preclosure_*`` for future revert).

    Returns the number of closures applied (0 when stripping only).
    """
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    db_path = resolve_project_database_path(project_dir)
    if not db_path.is_file():
        print(f"  swap_db_closures: DB not found at {db_path}")
        return 0

    # --- Step 1: strip existing closures ---
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(links)")
    existing_cols = {row[1] for row in cur.fetchall()}
    preclosure_cols = {
        "_preclosure_capacity_ab", "_preclosure_capacity_ba",
        "_preclosure_speed_ab", "_preclosure_speed_ba",
    }
    if preclosure_cols.issubset(existing_cols):
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
        for col in preclosure_cols:
            try:
                cur.execute(f"ALTER TABLE links DROP COLUMN {col}")
            except Exception:
                pass
        conn.commit()
        print(f"  swap_db_closures: stripped closures from {restored} links")
    else:
        print("  swap_db_closures: no existing closures to strip")
    conn.close()

    if measurement_period is None:
        return 0

    # --- Step 2: apply closures for the given period ---
    bc_cfg = cfg.get("baseline_closures") or {}
    _cache = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))
    source_path = Path(bc_cfg.get("source_path", f"{_cache}/closures.parquet"))
    status_wl = bc_cfg.get("status_whitelist")
    closures = load_closures(
        source_path,
        measurement_period=measurement_period,
        status_whitelist=status_wl,
    )
    if not closures:
        print(f"  swap_db_closures: no closures for period {measurement_period}")
        return 0

    project = Project()
    project.open(str(project_dir))
    try:
        links = project.network.links.data.copy()
        crs_epsg = get_metric_epsg(cfg)
        links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=crs_epsg)
        links_gdf = apply_baseline_closures(links_gdf, closures, cfg)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(links)")
        existing = {row[1] for row in cur.fetchall()}
        for pcol in preclosure_cols:
            if pcol not in existing:
                cur.execute(f"ALTER TABLE links ADD COLUMN {pcol} REAL")
        def _safe_float(val, fallback=50.0):
            v = float(val)
            return v if v == v else fallback  # NaN != NaN

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
        conn.commit()
        conn.close()
    finally:
        project.close()

    n = len(closures)
    # Count how many links actually had closure effects applied
    n_affected = sum(
        1 for _, row in links_gdf.iterrows()
        if "_preclosure_capacity_ab" in row.index
        and row.get("capacity_ab") != row.get("_preclosure_capacity_ab")
    ) if "_preclosure_capacity_ab" in links_gdf.columns else 0
    print(f"  swap_db_closures: applied {n} closures for period {measurement_period}"
          f" ({n_affected} links affected)")
    return n


def normalize_and_export_network(
    config_path: str | Path = "config/brno/sim.yaml",
    outputs_dir: str | Path | None = None,
) -> None:
    """
    Normalize link attributes, check connectivity, and export the network.
    """
    cfg = load_config(config_path)
    if outputs_dir is None:
        outputs_dir = cfg.get("network", {}).get("output_dir", "outputs/baseline/network")

    network_cfg = cfg.get("network") or {}
    experiment_profile = network_cfg.get("experiment_profile", "baseline")
    project_dir = Path(cfg["project_path"])

    project = Project()
    project.open(str(project_dir))

    try:
        print("=== NORMALIZE ATTRIBUTES ===")
        links = normalize_network_attributes(
            project,
            network_cfg,
            experiment_profile=experiment_profile,
        )
        print(f"Normalized {len(links)} links")

        print("\n=== REPAIR BOUNDARY SCC ===")
        repair_info = repair_boundary_scc(project)
        if repair_info.get("repaired", 0) > 0:
            repaired_set = set(repair_info["repaired_ids"])
            mask = links["link_id"].astype(int).isin(repaired_set)
            links.loc[mask, "direction"] = 0
            links.loc[mask, "speed_ba"] = links.loc[mask, "speed_ab"]
            links.loc[mask, "capacity_ba"] = links.loc[mask, "capacity_ab"]
            links.loc[mask, "lanes_ba"] = links.loc[mask, "lanes_ab"]
            links.loc[mask, "travel_time_ba"] = links.loc[mask, "travel_time_ab"]

        print("\n=== REPAIR DIVIDED HIGHWAYS ===")
        divided_info = repair_divided_highway_dead_ends(
            project,
            max_snap_distance_m=float(
                network_cfg.get("divided_highway_snap_m", 600)
            ),
        )
        if divided_info["new_link_ids"]:
            db_path = resolve_project_database_path(project)
            conn_tmp = sqlite3.connect(str(db_path))
            for new_lid in divided_info["new_link_ids"]:
                row = conn_tmp.execute(
                    "SELECT link_id, speed_ab, capacity_ab, lanes_ab, distance, link_type "
                    "FROM links WHERE link_id=?", (new_lid,)
                ).fetchone()
                if row:
                    spd = float(row[1]) if row[1] else 0.0
                    dist_m = float(row[4])
                    tt = (dist_m / 1000.0 / spd * 3600.0) if spd else 0.0
                    new_row = pd.DataFrame([{
                        "link_id": row[0], "link_type": str(row[5] or "motorway"),
                        "direction": 0, "speed_ab": row[1], "speed_ba": row[1],
                        "capacity_ab": row[2], "capacity_ba": row[2],
                        "lanes_ab": row[3], "lanes_ba": row[3],
                        "travel_time_ab": tt,
                        "travel_time_ba": tt,
                        "distance": row[4],
                    }])
                    links = pd.concat([links, new_row], ignore_index=True)
            conn_tmp.close()

        print("\n=== CHECK CONNECTIVITY ===")
        connectivity_info = check_connectivity(project)
        print(f"Components: {connectivity_info['total_components']}")
        print(f"Largest component: {connectivity_info['largest_component_size']} nodes")
        print(f"Isolated nodes: {connectivity_info['isolated_nodes_count']}")
        print(f"Isolated components: {connectivity_info['isolated_components_count']}")

        # ----- Baseline closures (optional) -----
        bc_cfg = cfg.get("baseline_closures") or {}
        if bc_cfg.get("enabled", False):
            print("\n=== BASELINE CLOSURES ===")
            _cache2 = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))
            source_path = Path(bc_cfg.get("source_path", f"{_cache2}/closures.parquet"))
            closures = load_closures(
                source_path,
                measurement_period=bc_cfg.get("measurement_period"),
                status_whitelist=bc_cfg.get("status_whitelist"),
            )
            if closures:
                links = apply_baseline_closures(links, closures, cfg)

                # Persist closure-affected values + _preclosure columns to DB
                db_path = resolve_project_database_path(project)
                if db_path.is_file():
                    conn = sqlite3.connect(str(db_path))
                    cur = conn.cursor()
                    cur.execute("PRAGMA table_info(links)")
                    existing = {row[1] for row in cur.fetchall()}
                    for pcol in ("_preclosure_capacity_ab", "_preclosure_capacity_ba",
                                 "_preclosure_speed_ab", "_preclosure_speed_ba"):
                        if pcol not in existing:
                            cur.execute(f"ALTER TABLE links ADD COLUMN {pcol} REAL")
                    def _sf(val, fb=50.0):
                        v = float(val)
                        return v if v == v else fb

                    for _, row in links.iterrows():
                        ca = _sf(row["capacity_ab"])
                        cb = _sf(row["capacity_ba"])
                        sa = _sf(row["speed_ab"], 5.0)
                        sb = _sf(row["speed_ba"], 5.0)
                        ta = _sf(row["travel_time_ab"], 0.01)
                        tb = _sf(row["travel_time_ba"], 0.01)
                        cur.execute(
                            "UPDATE links SET capacity_ab=?, capacity_ba=?, speed_ab=?, speed_ba=?, "
                            "travel_time_ab=?, travel_time_ba=?, "
                            "_preclosure_capacity_ab=?, _preclosure_capacity_ba=?, "
                            "_preclosure_speed_ab=?, _preclosure_speed_ba=? "
                            "WHERE link_id=?",
                            (ca, cb, sa, sb, ta, tb,
                             _sf(row.get("_preclosure_capacity_ab", ca), ca),
                             _sf(row.get("_preclosure_capacity_ba", cb), cb),
                             _sf(row.get("_preclosure_speed_ab", sa), sa),
                             _sf(row.get("_preclosure_speed_ba", sb), sb),
                             int(row["link_id"])),
                        )
                    conn.commit()
                    conn.close()
            else:
                print("  No closures to apply (empty or file missing)")

        print("\n=== EXPORT NETWORK ===")
        out_dir = Path(outputs_dir)
        crs_epsg = get_metric_epsg(cfg)
        export_stable_network(
            project,
            out_dir,
            connectivity_info,
            normalized_links=links,
            output_crs_epsg=crs_epsg,
            cfg=cfg,
        )

        print("\n=== NORMALIZATION COMPLETE ===")

    finally:
        project.close()
