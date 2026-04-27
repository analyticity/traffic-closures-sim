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
        return sl.links

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

    # --- expected_links validation ------------------------------------------
    if sl.expected_links is not None and len(resolved) != sl.expected_links:
        logger.warning(
            "Screenline '%s': resolved %d link(s) but expected %d — check "
            "cut-line geometry and filter (filter=%s)",
            sl.name, len(resolved), sl.expected_links, sl.attr_filter or "none",
        )

    # --- log results --------------------------------------------------------
    link_names = []
    for lid, _ in resolved:
        match = lm[lm["link_id"] == lid]
        lname = str(match["name"].iloc[0]) if (not match.empty and "name" in match.columns) else "?"
        link_names.append(f"{lid} ({lname})")
    if resolved:
        logger.info(
            "Screenline '%s' resolved to %d link(s): %s",
            sl.name, len(resolved), ", ".join(link_names),
        )
    else:
        logger.warning(
            "Screenline '%s' resolved to 0 links (filter=%s)",
            sl.name, sl.attr_filter or "none",
        )

    return resolved


# --- Evaluator ---

def _get_link_volume(
    vol_df: pd.DataFrame,
    link_id: int,
    direction: int,
    vol_col: str,
) -> float:
    """Get modeled volume for a link, respecting direction."""
    rows = vol_df[vol_df["link_id"] == link_id]
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
) -> float:
    """Get observed count for a link from the matched counts DataFrame."""
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

    for link_id, direction in resolved:
        mv = _get_link_volume(vol_df, link_id, direction, vol_col)
        ov = _get_link_observed(matched_counts, link_id, obs_col)
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
    results: Dict[str, ScreenlineResult] = {}
    for sl in screenlines:
        r = evaluate_screenline(sl, vol_df, matched_counts, vol_col, obs_col, links_gdf)
        results[sl.name] = r
    return results


# --- Auto-generation from gateway metadata + CSD data ---

def auto_generate_screenlines(
    cfg: Dict[str, Any],
    *,
    csd_df: Optional[pd.DataFrame] = None,
) -> List[ScreenlineDef]:
    """Generate screenlines automatically from gateway diagnostics and CSD.

    Two sources:
    1. **Gateway screenlines** -- one per external gateway, using boundary
       crossing point and road ref from ``gateway_diagnostics.csv``.
    2. **CSD screenlines** -- for major CSD road sections inside the model
       area (by matching ``sil`` to network ``osm_ref``), above a
       configurable AADT threshold.

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
    if bool(auto_cfg.get("gateway_screenlines", False)):
        gw_diag_path = zoning_out / "gateway_diagnostics.csv"
        if gw_diag_path.exists():
            gw_df = pd.read_csv(gw_diag_path)
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True,
            )
            for _, row in gw_df.iterrows():
                gw_name = str(row.get("gateway_name", ""))
                if not gw_name:
                    continue
                bx = row.get("boundary_x")
                by = row.get("boundary_y")
                odx = row.get("outward_dx", 0)
                ody = row.get("outward_dy", 0)
                if pd.isna(bx) or pd.isna(by):
                    continue

                bx, by = float(bx), float(by)
                # Perpendicular to outward direction, ~200m cut-line
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

                ref = str(row.get("matched_ref", "")).strip()
                attr_filter: Dict[str, str] = {}
                if ref:
                    attr_filter["osm_ref_norm"] = ref

                result.append(ScreenlineDef(
                    name=f"auto_gw_{gw_name}",
                    description=f"Auto-generated gateway screenline for {gw_name}",
                    sl_type="radial",
                    geometry_wkt=wkt,
                    attr_filter=attr_filter,
                    expected_links=2,
                ))
            logger.info("Auto-generated %d gateway screenlines", len(result))

    # --- 2) CSD-based screenlines -------------------------------------------
    csd_min_aadt = float(auto_cfg.get("csd_min_aadt", 5000))
    if csd_df is not None and not csd_df.empty and bool(auto_cfg.get("csd_screenlines", True)):
        gw_names_generated = {sl.name for sl in result}
        for col in ("sv", "o", "tv"):
            if col in csd_df.columns:
                csd_df[col] = pd.to_numeric(csd_df[col], errors="coerce").fillna(0)

        # Collect all road refs present in the model network so we only
        # generate screenlines for roads that the model can actually resolve.
        network_refs: set[str] = set()
        project_dir = Path(cfg.get("project_path", ""))
        if project_dir.exists():
            try:
                from aequilibrae import Project as _AeqProject
                _prj = _AeqProject()
                _prj.open(str(project_dir))
                try:
                    _ldf = _prj.network.links.data
                    if "osm_ref_norm" in _ldf.columns:
                        for ref_val in _ldf["osm_ref_norm"].dropna().unique():
                            for part in str(ref_val).split(";"):
                                part = part.strip().lower()
                                if part:
                                    network_refs.add(part)
                finally:
                    _prj.close()
                logger.info(
                    "CSD screenline filter: %d unique road refs in model network",
                    len(network_refs),
                )
            except Exception as exc:
                logger.debug("Could not load network refs for CSD filter: %s", exc)

        if "sil" in csd_df.columns and "sv" in csd_df.columns:
            grouped = csd_df.groupby("sil", as_index=False).agg(
                sv_mean=("sv", "mean"),
                o_mean=("o", "mean") if "o" in csd_df.columns else ("sv", "mean"),
            )
            major = grouped[grouped["sv_mean"] >= csd_min_aadt]

            n_skipped_no_network = 0
            for _, row in major.iterrows():
                road = str(row["sil"]).strip()

                # Normalize CSD road id to match osm_ref_norm format:
                # strip leading zeros from numeric road IDs (CSD uses "00732"
                # but OSM ref is just "732").
                road_norm = road.upper().replace(" ", "")
                try:
                    road_norm = str(int(road_norm))
                except ValueError:
                    road_norm = road_norm.lstrip("0") or road_norm

                name = f"auto_csd_{road_norm}"
                if name in gw_names_generated:
                    continue
                if network_refs and road_norm.lower() not in network_refs:
                    n_skipped_no_network += 1
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
                "skipped %d roads not in model network)",
                n_csd, int(csd_min_aadt), n_skipped_no_network,
            )

    return result


def load_screenlines_with_auto(
    cfg: Dict[str, Any],
    csd_df: Optional[pd.DataFrame] = None,
) -> List[ScreenlineDef]:
    """Load manual screenlines from YAML, then merge auto-generated ones.

    Manual definitions take precedence: auto-generated screenlines whose
    name collides with a manual one are dropped.
    """
    config_dir = cfg.get("_meta", {}).get("base_dir", "")
    default_sl = str(Path(config_dir) / "screenlines.yaml") if config_dir else "config/brno/screenlines.yaml"
    sl_path = str(cfg.get("calibration", {}).get(
        "screenlines_path", default_sl,
    ))
    manual = load_screenlines(sl_path)
    manual_names = {sl.name for sl in manual}

    auto = auto_generate_screenlines(cfg, csd_df=csd_df)
    for sl in auto:
        if sl.name not in manual_names:
            manual.append(sl)

    return manual
