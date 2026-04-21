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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Link resolver (explicit + geometry)
# ---------------------------------------------------------------------------

def _match_attr_filter(row: pd.Series, attr_filter: Dict[str, str]) -> bool:
    """Check if a link row matches ALL attribute filter constraints.

    Each filter value may be comma-separated to allow multiple alternatives
    (OR within a key, AND across keys).
    """
    for key, pattern in attr_filter.items():
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        val_str = str(val).strip().lower()
        alternatives = [a.strip().lower() for a in pattern.split(",")]
        if val_str not in alternatives:
            return False
    return True


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

    if geom_raw is None:
        return []

    if geom_crs is not None and geom_crs != metric_epsg:
        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{geom_crs}", f"EPSG:{metric_epsg}", always_xy=True,
        )
        geom_metric = shapely_transform(transformer.transform, geom_raw)
    else:
        geom_metric = geom_raw

    # --- reproject links if needed ------------------------------------------
    if links_gdf.crs is not None and links_gdf.crs.to_epsg() != metric_epsg:
        lm = links_gdf.to_crs(epsg=metric_epsg)
    else:
        lm = links_gdf

    # --- spatial intersection via sindex ------------------------------------
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


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

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

    # Use CSD AADT from config when pentlogram gives no observed data
    obs_source = "pentlogram"
    if obs_total <= 0 and sl.observed_aadt_cars is not None and sl.observed_aadt_cars > 0:
        obs_total = sl.observed_aadt_cars
        obs_source = "csd_config"

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
