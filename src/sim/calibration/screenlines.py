"""Screenline, cordon, and radial definitions for localized calibration.

Supports three specification methods:
- Geometry cut-line (WKT linestring, auto-detects crossing links) -- preferred
- Boundary polygon (GeoJSON file, for cordons)
- Explicit link IDs (fragile fallback, breaks on network re-import)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml

from sim._metrics import compute_geh

logger = logging.getLogger(__name__)


# --- Data structures ---

@dataclass
class ScreenlineDef:
    name: str
    description: str = ""
    sl_type: str = "screenline"
    links: List[Tuple[int, int]] = field(default_factory=list)
    has_explicit_links: bool = False
    geometry_wkt: Optional[str] = None
    boundary_geojson: Optional[str] = None
    observed_aadt_cars: Optional[float] = None
    observed_aadt_all: Optional[float] = None
    attr_filter: Dict[str, str] = field(default_factory=dict)
    expected_links: Optional[int] = None


@dataclass
class ScreenlineResult:
    name: str
    sl_type: str
    modeled_total: float
    observed_total: float
    ratio: float
    geh: float
    n_links: int
    obs_source: str = "pentlogram"
    per_link: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.sl_type,
            "modeled_total": round(self.modeled_total, 0),
            "observed_total": round(self.observed_total, 0),
            "ratio": round(self.ratio, 3) if np.isfinite(self.ratio) else None,
            "geh": round(self.geh, 2) if np.isfinite(self.geh) else None,
            "n_links": self.n_links,
            "obs_source": self.obs_source,
            "per_link": self.per_link,
        }


# --- Loader ---

def load_screenlines(config_path: str | Path) -> List[ScreenlineDef]:
    """Load screenline definitions from YAML."""
    p = Path(config_path)
    if not p.exists():
        return []

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("screenlines", [])
    if not entries:
        return []

    result: List[ScreenlineDef] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "unnamed"))

        links: List[Tuple[int, int]] = []
        for lk in (e.get("links") or []):
            if isinstance(lk, dict):
                lid = int(lk.get("link_id", 0))
                d = int(lk.get("direction", 0))
                if lid > 0:
                    links.append((lid, d))

        obs_cars = e.get("observed_aadt_cars")
        obs_all = e.get("observed_aadt_all")

        raw_filter = e.get("filter") or {}
        attr_filter = {str(k): str(v) for k, v in raw_filter.items()} if isinstance(raw_filter, dict) else {}

        raw_expected = e.get("expected_links")
        expected_links = int(raw_expected) if raw_expected is not None else None

        result.append(ScreenlineDef(
            name=name,
            description=str(e.get("description", "")),
            sl_type=str(e.get("type", "screenline")),
            links=links,
            has_explicit_links=bool(links),
            geometry_wkt=e.get("geometry_wkt"),
            boundary_geojson=e.get("boundary_geojson"),
            observed_aadt_cars=float(obs_cars) if obs_cars is not None else None,
            observed_aadt_all=float(obs_all) if obs_all is not None else None,
            attr_filter=attr_filter,
            expected_links=expected_links,
        ))
    return result


# --- Link resolver ---

def _match_attr_filter(row: pd.Series, attr_filter: Dict[str, str]) -> bool:
    """Check if a link row matches ALL attribute filter constraints.

    Each filter value may be comma-separated to allow multiple alternatives
    (OR within a key, AND across keys).  Link attribute values containing
    semicolons (e.g. ``osm_ref_norm = "13;27"``) are split into components
    so that a filter ``"13"`` matches any component.
    """
    for key, pattern in attr_filter.items():
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        val_str = str(val).strip().lower()
        alternatives = {a.strip().lower() for a in pattern.split(",")}
        val_parts = {p.strip() for p in val_str.split(";")} if ";" in val_str else {val_str}
        if not val_parts & alternatives:
            return False
    return True


_RAMP_LINK_TYPES = frozenset({
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
})


def _resolve_attr_only_screenline(
    sl: "ScreenlineDef",
    links_gdf: gpd.GeoDataFrame,
) -> List[Tuple[int, int]]:
    """Resolve a screenline that has only ``attr_filter`` and no geometry.

    For CSD-based screenlines the "screenline" is conceptually a single
    cross-section of road X.  We pick a representative link that best
    captures the corridor's volume by:

    1. Filtering out ramp/link types (e.g. ``trunk_link``) to keep only
       mainline segments.
    2. Sorting remaining links by geometric length (longest first) -- the
       longest segment is most likely a straight mainline section.
    3. Among the top candidates, selecting the one closest to the
       geographic centroid of all candidates (to avoid boundary artifacts).

    Returns a single bidirectional link or a pair of one-way links.
    """
    candidates = []
    for idx in range(len(links_gdf)):
        row = links_gdf.iloc[idx]
        if row.geometry is None:
            continue
        if _match_attr_filter(row, sl.attr_filter):
            candidates.append(row)

    if not candidates:
        return []

    cand_df = gpd.GeoDataFrame(candidates)
    has_lt = "link_type" in cand_df.columns

    mainline = (
        cand_df[~cand_df["link_type"].isin(_RAMP_LINK_TYPES)]
        if has_lt
        else cand_df
    )
    if mainline.empty:
        mainline = cand_df

    mainline = mainline.copy()
    mainline["_length"] = mainline.geometry.length

    centroid_all = mainline.geometry.unary_union.centroid

    top_n = min(10, len(mainline))
    longest = mainline.nlargest(top_n, "_length")

    dists = longest.geometry.centroid.distance(centroid_all)
    best_idx = dists.idxmin()
    chosen = longest.loc[best_idx]

    bidir = (
        int(chosen.get("direction", 0)) == 0
        if "direction" in chosen.index
        else True
    )

    if bidir:
        return [(int(chosen["link_id"]), 0)]

    pair: List[Tuple[int, int]] = [(int(chosen["link_id"]), 0)]
    pt = chosen.geometry.centroid
    others = mainline[
        (mainline["link_id"] != int(chosen["link_id"]))
        & (mainline.get("direction", pd.Series(0, index=mainline.index)) != 0)
    ]
    if not others.empty:
        odists = others.geometry.centroid.distance(pt)
        nearest_idx = odists.idxmin()
        pair.append((int(others.loc[nearest_idx, "link_id"]), 0))

    return pair


def _dedup_link_ids(
    sl_name: str,
    links: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Remove duplicate link_id entries within a single screenline."""
    seen: set = set()
    out: List[Tuple[int, int]] = []
    for lid, d in links:
        if lid in seen:
            continue
        seen.add(lid)
        out.append((lid, d))
    n_dropped = len(links) - len(out)
    if n_dropped:
        logger.warning(
            "Screenline '%s': removed %d duplicate link_id(s) (kept %d unique)",
            sl_name, n_dropped, len(out),
        )
    return out


def _dedup_cross_screenline_links(
    sl_query: Dict[str, List[Tuple[int, int]]],
) -> Tuple[Dict[str, List[Tuple[int, int]]], List[str]]:
    """Remove link_ids that appear in more than one screenline.

    For each duplicated link_id, the link is kept in the screenline that
    has the **fewest** total links (tie-break: alphabetical screenline
    name).  This preserves sparse screenlines that would otherwise lose
    their only control point.

    Returns ``(deduped_sl_query, log_lines)`` where *log_lines* is a
    human-readable list of removals for logging.
    """
    from collections import defaultdict

    link_to_sls: Dict[int, List[str]] = defaultdict(list)
    for sl_name, link_tuples in sl_query.items():
        for lid, _d in link_tuples:
            link_to_sls[lid].append(sl_name)

    collisions = {
        lid: sls for lid, sls in link_to_sls.items() if len(sls) > 1
    }
    if not collisions:
        return sl_query, []

    sl_size = {name: len(links) for name, links in sl_query.items()}

    keep_in: Dict[int, str] = {}
    for lid, sls in collisions.items():
        best = min(sls, key=lambda s: (sl_size[s], s))
        keep_in[lid] = best

    log_lines: List[str] = []
    out: Dict[str, List[Tuple[int, int]]] = {}
    for sl_name, link_tuples in sl_query.items():
        filtered = []
        for lid, d in link_tuples:
            if lid in keep_in and keep_in[lid] != sl_name:
                log_lines.append(
                    f"  link {lid}: removed from '{sl_name}', "
                    f"kept in '{keep_in[lid]}'"
                )
                continue
            filtered.append((lid, d))
        out[sl_name] = filtered

    return out, log_lines


def resolve_screenline_links(
    sl: ScreenlineDef,
    links_gdf: gpd.GeoDataFrame,
    metric_epsg: int = 5514,
) -> List[Tuple[int, int]]:
    """Resolve a screenline definition to a list of (link_id, direction) tuples.

    If the screenline has explicit link IDs (``has_explicit_links``), returns
    those directly.  Otherwise resolves from ``geometry_wkt`` (preferred) or
    ``boundary_geojson`` by intersecting link geometries, optionally filtered
    by ``attr_filter``.
    """
    if sl.has_explicit_links and sl.links:
        return _dedup_link_ids(sl.name, sl.links)

    from shapely import wkt as shapely_wkt
    from shapely.ops import transform as shapely_transform
    import pyproj

    # --- build the cut-line geometry in metric CRS --------------------------
    geom_raw = None
    geom_crs: Optional[int] = None
    lm = links_gdf

    if sl.geometry_wkt:
        geom_raw = shapely_wkt.loads(sl.geometry_wkt)
        geom_crs = 4326  # WKT in YAML is always WGS84
    elif sl.boundary_geojson:
        bp = Path(sl.boundary_geojson)
        if bp.exists():
            bg = gpd.read_file(bp)
            if bg.crs is not None and bg.crs.to_epsg() != metric_epsg:
                bg = bg.to_crs(epsg=metric_epsg)
            geom_raw = bg.geometry.unary_union.boundary
            geom_crs = metric_epsg

    # --- attr-only screenlines (no geometry) ---------------------------------
    # CSD screenlines are defined purely by attribute filter (e.g. road number)
    # with no cut-line geometry.  Resolve by scanning all matching links.
    if geom_raw is None:
        if sl.attr_filter:
            resolved = _resolve_attr_only_screenline(sl, links_gdf)
        else:
            return []
    else:
        if geom_crs is not None and geom_crs != metric_epsg:
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{geom_crs}", f"EPSG:{metric_epsg}", always_xy=True,
            )
            geom_metric = shapely_transform(transformer.transform, geom_raw)
        else:
            geom_metric = geom_raw

        # --- reproject links if needed --------------------------------------
        if links_gdf.crs is not None and links_gdf.crs.to_epsg() != metric_epsg:
            lm = links_gdf.to_crs(epsg=metric_epsg)
        else:
            lm = links_gdf

        # --- spatial intersection via sindex --------------------------------
        candidates = lm.sindex.query(geom_metric, predicate="intersects")
        resolved: List[Tuple[int, int]] = []
        for idx in candidates:
            row = lm.iloc[idx]
            if row.geometry is None:
                continue
            if sl.attr_filter and not _match_attr_filter(row, sl.attr_filter):
                continue
            lid = int(row["link_id"])
            resolved.append((lid, 0))

    if not resolved and sl.attr_filter and geom_raw is not None:
        resolved = _resolve_attr_only_screenline(sl, links_gdf)

    resolved = _dedup_link_ids(sl.name, resolved)

    # --- expected_links validation ------------------------------------------
    if sl.expected_links is not None and len(resolved) != sl.expected_links:
        logger.warning(
            "Screenline '%s': resolved %d link(s) but expected %d — check "
            "cut-line geometry and filter (filter=%s)",
            sl.name, len(resolved), sl.expected_links, sl.attr_filter or "none",
        )

    # --- log results with link_type audit ------------------------------------
    _NON_ROUTABLE_TYPES = frozenset({
        "living_street", "pedestrian", "footway", "service",
        "cycleway", "steps", "path", "track",
    })
    link_descs = []
    non_routable_lids = []
    has_link_type = "link_type" in lm.columns
    for lid, _ in resolved:
        match = lm[lm["link_id"] == lid]
        lname = str(match["name"].iloc[0]) if (not match.empty and "name" in match.columns) else "?"
        lt = str(match["link_type"].iloc[0]) if (not match.empty and has_link_type) else "?"
        link_descs.append(f"{lid} ({lname}, {lt})")
        if has_link_type and not match.empty and lt in _NON_ROUTABLE_TYPES:
            non_routable_lids.append(lid)

    if resolved:
        logger.info(
            "Screenline '%s' resolved to %d link(s): %s",
            sl.name, len(resolved), ", ".join(link_descs),
        )
    else:
        logger.warning(
            "Screenline '%s' resolved to 0 links (filter=%s)",
            sl.name, sl.attr_filter or "none",
        )

    if non_routable_lids and len(non_routable_lids) == len(resolved):
        logger.warning(
            "Screenline '%s': ALL %d resolved link(s) are non-routable types "
            "(%s) — this screenline will likely show zero modeled flow",
            sl.name, len(resolved),
            ", ".join(f"{lid}" for lid in non_routable_lids),
        )
        resolved = []
    elif non_routable_lids:
        logger.warning(
            "Screenline '%s': %d of %d resolved link(s) are non-routable — "
            "filtering them out: %s",
            sl.name, len(non_routable_lids), len(resolved),
            non_routable_lids,
        )
        nr_set = set(non_routable_lids)
        resolved = [(lid, d) for lid, d in resolved if lid not in nr_set]

    return resolved


# --- Evaluator ---

def _get_link_volume(
    vol_df: pd.DataFrame,
    link_id: int,
    direction: int,
    vol_col: str,
    _vol_idx: Optional[pd.DataFrame] = None,
) -> float:
    """Get modeled volume for a link, respecting direction.

    Pass *_vol_idx* (vol_df indexed by link_id) to avoid repeated O(N) scans.
    """
    idx = _vol_idx if _vol_idx is not None else vol_df
    if _vol_idx is None:
        rows = vol_df[vol_df["link_id"] == link_id]
    else:
        try:
            rows = idx.loc[[link_id]]
        except KeyError:
            return 0.0

    if rows.empty:
        return 0.0

    if vol_col not in rows.columns:
        return 0.0

    vol = float(rows[vol_col].iloc[0])

    ab_col = vol_col.replace("_tot", "_ab") if "_tot" in vol_col else None
    ba_col = vol_col.replace("_tot", "_ba") if "_tot" in vol_col else None

    if direction == 1 and ab_col and ab_col in rows.columns:
        return float(rows[ab_col].iloc[0])
    if direction == -1 and ba_col and ba_col in rows.columns:
        return float(rows[ba_col].iloc[0])
    return vol


def _get_link_observed(
    matched: gpd.GeoDataFrame,
    link_id: int,
    obs_col: str,
    _obs_idx: Optional[pd.DataFrame] = None,
) -> float:
    """Get observed count for a link from the matched counts DataFrame.

    Pass *_obs_idx* (matched indexed by link_id) to avoid repeated O(N) scans.
    """
    if _obs_idx is not None:
        try:
            rows = _obs_idx.loc[[link_id]]
        except KeyError:
            return 0.0
    else:
        rows = matched[matched["link_id"] == link_id] if "link_id" in matched.columns else pd.DataFrame()

    if rows.empty:
        return 0.0
    if obs_col not in rows.columns:
        return 0.0
    return float(rows[obs_col].sum())


def evaluate_screenline(
    sl: ScreenlineDef,
    vol_df: pd.DataFrame,
    matched_counts: gpd.GeoDataFrame,
    vol_col: str,
    obs_col: str = "observed_car",
    links_gdf: Optional[gpd.GeoDataFrame] = None,
) -> ScreenlineResult:
    """Evaluate a single screenline: sum modeled and observed volumes, compute GEH."""
    resolved = sl.links
    if not resolved and links_gdf is not None:
        resolved = resolve_screenline_links(sl, links_gdf)

    mod_total = 0.0
    obs_total = 0.0
    per_link: List[Dict[str, Any]] = []

    vol_idx = vol_df.set_index("link_id", drop=False) if "link_id" in vol_df.columns else vol_df
    obs_idx = (
        matched_counts.set_index("link_id", drop=False)
        if "link_id" in matched_counts.columns
        else matched_counts
    )

    for link_id, direction in resolved:
        mv = _get_link_volume(vol_df, link_id, direction, vol_col, _vol_idx=vol_idx)
        ov = _get_link_observed(matched_counts, link_id, obs_col, _obs_idx=obs_idx)
        mod_total += mv
        obs_total += ov
        per_link.append({
            "link_id": link_id,
            "direction": direction,
            "modeled": round(mv, 0),
            "observed": round(ov, 0),
        })

    # Use CSD AADT from config when pentlogram gives no observed data,
    # or when matched counts are suspiciously low (< 20% of known AADT).
    # The latter catches cases where CSD-split puts the main road section
    # into the validation set and a nearby minor road gets matched instead.
    obs_source = "pentlogram"
    config_aadt = sl.observed_aadt_all or sl.observed_aadt_cars
    if config_aadt is not None and config_aadt > 0:
        if obs_total <= 0:
            obs_total = config_aadt
            obs_source = "csd_config"
        elif obs_total < 0.20 * config_aadt:
            logger.warning(
                "Screenline '%s': matched obs=%,.0f is only %.0f%% of "
                "config AADT=%,.0f — overriding with config AADT",
                sl.name, obs_total,
                100.0 * obs_total / config_aadt,
                config_aadt,
            )
            obs_total = config_aadt
            obs_source = "csd_config_override"

    if obs_source != "pentlogram" and per_link:
        orig_obs = sum(pl["observed"] for pl in per_link)
        if orig_obs > 0:
            scale = obs_total / orig_obs
            for pl in per_link:
                pl["observed"] = round(pl["observed"] * scale, 0)
        else:
            share = obs_total / len(per_link)
            for pl in per_link:
                pl["observed"] = round(share, 0)

    ratio = mod_total / max(obs_total, 1.0)
    geh_arr = compute_geh(
        np.array([mod_total], dtype=float),
        np.array([obs_total], dtype=float),
    )
    geh_val = float(geh_arr[0]) if len(geh_arr) > 0 and np.isfinite(geh_arr[0]) else float("nan")

    return ScreenlineResult(
        name=sl.name,
        sl_type=sl.sl_type,
        modeled_total=mod_total,
        observed_total=obs_total,
        ratio=ratio,
        geh=geh_val,
        n_links=len(resolved),
        obs_source=obs_source,
        per_link=per_link,
    )


def evaluate_all_screenlines(
    screenlines: List[ScreenlineDef],
    vol_df: pd.DataFrame,
    matched_counts: gpd.GeoDataFrame,
    vol_col: str,
    obs_col: str = "observed_car",
    links_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Dict[str, ScreenlineResult]:
    """Evaluate all screenlines and return {name: result}."""
    seen_names: set = set()
    unique_screenlines: List[ScreenlineDef] = []
    for sl in screenlines:
        if sl.name in seen_names:
            logger.warning("Duplicate screenline name '%s' — skipping duplicate", sl.name)
            continue
        seen_names.add(sl.name)
        unique_screenlines.append(sl)

    results: Dict[str, ScreenlineResult] = {}
    for sl in unique_screenlines:
        r = evaluate_screenline(sl, vol_df, matched_counts, vol_col, obs_col, links_gdf)
        if r.observed_total and r.observed_total > 0:
            if r.modeled_total == 0:
                logger.warning(
                    "Screenline '%s': modeled=0 vs observed=%,.0f — "
                    "resolved links may be disconnected or missing demand",
                    sl.name, r.observed_total,
                )
            elif r.ratio < 0.1:
                logger.warning(
                    "Screenline '%s': ratio=%.3f (modeled=%,.0f vs observed=%,.0f) — "
                    "severe under-assignment, check gateway demand",
                    sl.name, r.ratio, r.modeled_total, r.observed_total,
                )
        results[sl.name] = r
    return results


# --- Auto-generation from gateway metadata + CSD data ---

def auto_generate_screenlines(
    cfg: Dict[str, Any],
    *,
    csd_df: Optional[pd.DataFrame] = None,
    csd_df_full: Optional[pd.DataFrame] = None,
) -> List[ScreenlineDef]:
    """Generate screenlines automatically from gateway diagnostics and CSD.

    Two sources:
    1. **Gateway screenlines** -- one per external gateway, using boundary
       crossing point and road ref from ``gateway_diagnostics.csv``.
    2. **CSD screenlines** -- for major CSD road sections inside the model
       area (by matching ``sil`` to network ``osm_ref``), above a
       configurable AADT threshold.

    Parameters
    ----------
    csd_df : DataFrame, optional
        Region-filtered CSD data, used for CSD screenlines inside the model.
    csd_df_full : DataFrame, optional
        Unfiltered (national) CSD data, used for gateway AADT lookup.
        Gateway roads may span multiple regions, so using the full CSD
        prevents gaps like D55 being missing when the model is in CZ071 but
        D55 only has sections in CZ064/CZ072.  Falls back to *csd_df* if
        not provided.

    Returns a list of :class:`ScreenlineDef` ready for evaluation.
    Manual screenlines from YAML take precedence (caller merges).
    """
    import pyproj

    auto_cfg = cfg.get("calibration", {}).get("auto_screenlines") or {}
    if not bool(auto_cfg.get("enabled", False)):
        return []

    crs_epsg = int(cfg.get("crs_epsg", 5514))
    zoning_out = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))

    result: List[ScreenlineDef] = []

    # --- 1) Gateway-based screenlines ---------------------------------------
    if bool(auto_cfg.get("gateway_screenlines", True)):
        gw_diag_path = zoning_out / "gateway_diagnostics.csv"
        if gw_diag_path.exists():
            gw_df = pd.read_csv(gw_diag_path)
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True,
            )

            # Build per-road CSD lookup for gateway AADT.  We use the
            # unfiltered (national) CSD when available, because gateway
            # roads can span multiple regions (e.g. D55 is absent from
            # CZ071 but present in CZ064).
            csd_mean_by_road: Dict[str, Dict[str, float]] = {}
            _gw_csd = csd_df_full if csd_df_full is not None else csd_df
            if _gw_csd is not None and not _gw_csd.empty and "sil" in _gw_csd.columns and "sv" in _gw_csd.columns:
                _gw_csd = _gw_csd.copy()
                for col in ("sv", "o"):
                    if col in _gw_csd.columns:
                        _gw_csd[col] = pd.to_numeric(_gw_csd[col], errors="coerce").fillna(0)

                for road_sil, grp in _gw_csd.groupby("sil"):
                    road_raw = str(road_sil).strip().upper().replace(" ", "")
                    try:
                        road_key = str(int(road_raw))
                    except ValueError:
                        road_key = road_raw.lstrip("0") or road_raw

                    sv_vals = grp["sv"].values
                    o_vals = grp["o"].values if "o" in grp.columns else sv_vals

                    csd_mean_by_road[road_key] = {
                        "sv": float(sv_vals.mean()),
                        "o": float(o_vals.mean()),
                    }

            # Pre-load network links for anchor-node matching.
            links_gdf: Optional[gpd.GeoDataFrame] = None
            network_gpkg = Path(cfg.get("demand", {}).get(
                "network_dir", "outputs/baseline/network",
            )) / "network_links.gpkg"
            network_geojson = network_gpkg.with_suffix(".geojson")
            for net_path in (network_gpkg, network_geojson):
                if net_path.exists():
                    try:
                        links_gdf = gpd.read_file(net_path)
                    except Exception:
                        pass
                    break

            node_to_link_rows: dict = {}
            if links_gdf is not None:
                for i, (a, b) in enumerate(zip(links_gdf["a_node"].values, links_gdf["b_node"].values)):
                    node_to_link_rows.setdefault(int(a), []).append(i)
                    node_to_link_rows.setdefault(int(b), []).append(i)

            for _, row in gw_df.iterrows():
                gw_name = str(row.get("gateway_name", ""))
                if not gw_name:
                    continue
                bx = row.get("boundary_x")
                by = row.get("boundary_y")
                if pd.isna(bx) or pd.isna(by):
                    continue

                ref = str(row.get("matched_ref", "")).strip()
                anchor_raw = row.get("anchor_node_id")
                target_nodes_raw = str(row.get("target_node_ids", ""))

                # --- resolve explicit links from anchor/target nodes ------
                explicit_links: List[Tuple[int, int]] = []
                if links_gdf is not None and target_nodes_raw:
                    try:
                        tnodes = {int(float(x.strip()))
                                  for x in target_nodes_raw.split(",")
                                  if x.strip()}
                    except (ValueError, TypeError):
                        tnodes = set()

                    if tnodes:
                        for nid in tnodes:
                            adj_idxs = node_to_link_rows.get(nid, [])
                            adj = links_gdf.iloc[adj_idxs]
                            if ref:
                                ref_lower = ref.lower()
                                adj = adj[adj["osm_ref_norm"].apply(
                                    lambda v, rl=ref_lower: (
                                        rl in str(v).lower().split(";")
                                        if pd.notna(v) else False
                                    )
                                )]
                            for _, lrow in adj.iterrows():
                                lid = int(lrow["link_id"])
                                if lid not in {l[0] for l in explicit_links}:
                                    explicit_links.append((lid, 0))

                # --- CSD observed AADT ------------------------------------
                obs_all = 0.0
                obs_cars = 0.0
                ref_norm = ""
                if ref:
                    ref_norm = ref.upper().replace(" ", "")
                    try:
                        ref_norm = str(int(ref_norm))
                    except ValueError:
                        ref_norm = ref_norm.lstrip("0") or ref_norm

                    if csd_mean_by_road:
                        csd_hit = csd_mean_by_road.get(ref_norm)
                        if csd_hit:
                            obs_all = csd_hit["sv"]
                            obs_cars = csd_hit["o"]

                # --- build fallback WKT for resolution when no network ----
                odx = row.get("outward_dx", 0)
                ody = row.get("outward_dy", 0)
                bx, by = float(bx), float(by)
                perp_dx, perp_dy = -float(ody), float(odx)
                norm = (perp_dx ** 2 + perp_dy ** 2) ** 0.5
                if norm < 1e-9:
                    perp_dx, perp_dy = 100.0, 0.0
                else:
                    perp_dx = perp_dx / norm * 200
                    perp_dy = perp_dy / norm * 200
                p1 = transformer.transform(bx - perp_dx, by - perp_dy)
                p2 = transformer.transform(bx + perp_dx, by + perp_dy)
                wkt = f"LINESTRING({p1[0]:.6f} {p1[1]:.6f}, {p2[0]:.6f} {p2[1]:.6f})"

                attr_filter: Dict[str, str] = {}
                if ref:
                    attr_filter["osm_ref_norm"] = ref

                if explicit_links:
                    result.append(ScreenlineDef(
                        name=f"auto_gw_{gw_name}",
                        description=f"Auto-generated gateway screenline for {gw_name}",
                        sl_type="radial",
                        links=explicit_links,
                        has_explicit_links=True,
                        observed_aadt_all=obs_all if obs_all > 0 else None,
                        observed_aadt_cars=obs_cars if obs_cars > 0 else None,
                    ))
                else:
                    result.append(ScreenlineDef(
                        name=f"auto_gw_{gw_name}",
                        description=f"Auto-generated gateway screenline for {gw_name}",
                        sl_type="radial",
                        geometry_wkt=wkt,
                        attr_filter=attr_filter,
                        expected_links=2,
                        observed_aadt_all=obs_all if obs_all > 0 else None,
                        observed_aadt_cars=obs_cars if obs_cars > 0 else None,
                    ))
            n_with_obs = sum(1 for s in result if getattr(s, "observed_aadt_all", None))
            logger.info(
                "Auto-generated %d gateway screenlines (%d with CSD observed AADT)",
                len(result), n_with_obs,
            )

    # --- 2) CSD-based screenlines -------------------------------------------
    csd_min_aadt = float(auto_cfg.get("csd_min_aadt", 5000))
    if csd_df is not None and not csd_df.empty and bool(auto_cfg.get("csd_screenlines", True)):
        for col in ("sv", "o", "tv"):
            if col in csd_df.columns:
                csd_df[col] = pd.to_numeric(csd_df[col], errors="coerce").fillna(0)

        # Collect all road refs present in the model network so we only
        # generate screenlines for roads that the model can actually resolve.
        # Also compute per-ref total link length for coverage filtering.
        network_refs: set[str] = set()
        network_ref_km: Dict[str, float] = {}
        project_dir = Path(cfg.get("project_path", ""))
        if project_dir.exists():
            try:
                from aequilibrae import Project as _AeqProject
                _prj = _AeqProject()
                _prj.open(str(project_dir))
                try:
                    _ldf = _prj.network.links.data
                    if "osm_ref_norm" in _ldf.columns:
                        _has_dist = "distance" in _ldf.columns
                        for _, _lrow in _ldf.iterrows():
                            _ref_raw = _lrow.get("osm_ref_norm")
                            if pd.isna(_ref_raw):
                                continue
                            _link_km = float(_lrow["distance"]) / 1000.0 if _has_dist else 0.0
                            for part in str(_ref_raw).split(";"):
                                part = part.strip().lower()
                                if part:
                                    network_refs.add(part)
                                    network_ref_km[part] = network_ref_km.get(part, 0.0) + _link_km
                finally:
                    _prj.close()
                logger.info(
                    "CSD screenline filter: %d unique road refs in model network",
                    len(network_refs),
                )
            except Exception as exc:
                logger.debug("Could not load network refs for CSD filter: %s", exc)

        csd_for_coverage = csd_df_full if csd_df_full is not None else csd_df
        csd_road_km: Dict[str, float] = {}
        if (csd_for_coverage is not None and not csd_for_coverage.empty
                and "sil" in csd_for_coverage.columns and "delka" in csd_for_coverage.columns):
            for _sil, _grp in csd_for_coverage.groupby("sil"):
                _rn = str(_sil).strip().upper().replace(" ", "")
                try:
                    _rn = str(int(_rn))
                except ValueError:
                    _rn = _rn.lstrip("0") or _rn
                csd_road_km[_rn.lower()] = float(pd.to_numeric(_grp["delka"], errors="coerce").sum())

        if "sil" in csd_df.columns and "sv" in csd_df.columns:
            grouped = csd_df.groupby("sil", as_index=False).agg(
                sv_mean=("sv", "mean"),
                o_mean=("o", "mean") if "o" in csd_df.columns else ("sv", "mean"),
            )
            major = grouped[grouped["sv_mean"] >= csd_min_aadt]

            min_coverage = float(auto_cfg.get("csd_min_coverage", 0.5))
            n_skipped_no_network = 0
            n_skipped_partial = 0
            for _, row in major.iterrows():
                road = str(row["sil"]).strip()

                road_norm = road.upper().replace(" ", "")
                try:
                    road_norm = str(int(road_norm))
                except ValueError:
                    road_norm = road_norm.lstrip("0") or road_norm

                name = f"auto_csd_{road_norm}"
                if network_refs and road_norm.lower() not in network_refs:
                    n_skipped_no_network += 1
                    continue

                model_km = network_ref_km.get(road_norm.lower(), 0.0)
                csd_km = csd_road_km.get(road_norm.lower(), 0.0)
                coverage = model_km / csd_km if csd_km > 0 else 1.0
                if coverage < min_coverage:
                    n_skipped_partial += 1
                    logger.debug(
                        "Skipping CSD screenline %s: coverage=%.2f "
                        "(model=%.1f km, csd=%.1f km) < %.2f",
                        name, coverage, model_km, csd_km, min_coverage,
                    )
                    continue

                result.append(ScreenlineDef(
                    name=name,
                    description=f"Auto-generated CSD screenline for road {road_norm}",
                    sl_type="radial",
                    observed_aadt_cars=float(row.get("o_mean", 0)),
                    observed_aadt_all=float(row["sv_mean"]),
                    attr_filter={"osm_ref_norm": road_norm},
                ))
            n_csd = len([s for s in result if s.name.startswith("auto_csd_")])
            logger.info(
                "Auto-generated %d CSD screenlines (min AADT=%d, "
                "skipped %d no-network, %d partial-coverage)",
                n_csd, int(csd_min_aadt), n_skipped_no_network,
                n_skipped_partial,
            )

    return result


def load_screenlines_with_auto(
    cfg: Dict[str, Any],
    csd_df: Optional[pd.DataFrame] = None,
    csd_df_full: Optional[pd.DataFrame] = None,
) -> List[ScreenlineDef]:
    """Load manual screenlines from YAML, then merge auto-generated ones.

    Manual definitions take precedence: auto-generated screenlines whose
    name collides with a manual one are dropped.

    Parameters
    ----------
    csd_df : region-filtered CSD (for CSD screenlines inside the model)
    csd_df_full : unfiltered national CSD (for gateway AADT lookup)
    """
    config_dir = cfg.get("_meta", {}).get("base_dir", "")
    default_sl = str(Path(config_dir) / "screenlines.yaml") if config_dir else "config/brno/screenlines.yaml"
    sl_path = str(cfg.get("calibration", {}).get(
        "screenlines_path", default_sl,
    ))
    manual = load_screenlines(sl_path)
    manual_names = {sl.name for sl in manual}

    auto = auto_generate_screenlines(cfg, csd_df=csd_df, csd_df_full=csd_df_full)
    for sl in auto:
        if sl.name not in manual_names:
            manual.append(sl)

    return manual
