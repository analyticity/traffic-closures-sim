"""Extract closures active on a given date and match them to network links.

Produces a list of scenario-compatible link modifications that can be fed
directly into :func:`sim.scenarios.submit_scenario`.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from sim.network.closures import load_closures

logger = logging.getLogger(__name__)


def _match_closures_to_links(
    closures: List[Dict[str, Any]],
    link_gdf: gpd.GeoDataFrame,
    *,
    max_distance_m: float = 50.0,
    require_road_ref_match: bool = False,
    metric_epsg: int = 5514,
) -> Dict[int, List[Dict[str, Any]]]:
    """Spatially match closure points to the nearest network links.

    Returns ``{link_id: [closure_dict, ...]}``.  When multiple closures
    hit the same link the most severe one wins during scenario generation,
    but all are kept here for the API response.
    """
    if not closures:
        return {}

    closure_pts = gpd.GeoDataFrame(
        closures,
        geometry=[Point(float(c.get("lon", 0)), float(c.get("lat", 0))) for c in closures],
        crs="EPSG:4326",
    ).to_crs(epsg=metric_epsg)

    gdf = link_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=metric_epsg)
    elif gdf.crs.to_epsg() != metric_epsg:
        gdf = gdf.to_crs(epsg=metric_epsg)

    link_ref_col = "osm_ref_norm" if "osm_ref_norm" in gdf.columns else None

    link_closures: Dict[int, List[Dict[str, Any]]] = {}
    sindex = gdf.sindex

    for idx, cpt in closure_pts.iterrows():
        geom = cpt.geometry
        if geom is None or geom.is_empty:
            continue

        closure_dict = closures[int(idx)]
        closure_ref = str(closure_dict.get("road_ref", "") or "").strip()

        candidates = sindex.query(geom.buffer(max_distance_m), predicate="intersects")
        if len(candidates) == 0:
            nn = sindex.nearest(geom, max_distance=max_distance_m)
            candidates = nn[1] if nn.ndim == 2 and nn.shape[0] == 2 else nn.ravel()

        for cidx in candidates:
            row = gdf.iloc[int(cidx)]
            lid = int(row["link_id"])

            if closure_ref and link_ref_col:
                link_ref = str(row.get(link_ref_col, "") or "").strip()
                if require_road_ref_match and closure_ref and link_ref and closure_ref != link_ref:
                    continue

            link_closures.setdefault(lid, []).append(closure_dict)

    return link_closures


_SEVERITY_PRIORITY = {"full": 0, "lane_reduction": 1, "speed_limit": 2}


def _pick_dominant_severity(closure_list: List[Dict[str, Any]]) -> str:
    """Return the most severe severity from a list of closures hitting one link."""
    best = "speed_limit"
    for c in closure_list:
        sev = str(c.get("severity", "lane_reduction"))
        if _SEVERITY_PRIORITY.get(sev, 99) < _SEVERITY_PRIORITY.get(best, 99):
            best = sev
    return best


def _closure_description(closure_list: List[Dict[str, Any]]) -> str:
    """Build a human-readable description from closure attributes."""
    texts = []
    for c in closure_list:
        for key in ("description_cs", "txt", "otxt", "txpl_text", "event_popis1"):
            v = c.get(key)
            if v and str(v).strip():
                texts.append(str(v).strip())
                break
    return "; ".join(dict.fromkeys(texts)) if texts else ""


def closures_for_date(
    date: str,
    cfg: Dict[str, Any],
    link_gdf: Optional[gpd.GeoDataFrame] = None,
) -> List[Dict[str, Any]]:
    """Return scenario-ready link modifications from closures active on *date*.

    Parameters
    ----------
    date:
        ISO date string ``YYYY-MM-DD``.
    cfg:
        Full simulation config dict (as from ``load_config``).
    link_gdf:
        Pre-loaded network links GeoDataFrame.  When ``None`` the function
        loads the exported network from disk (``outputs/baseline/network``).

    Returns
    -------
    List of dicts, each with keys compatible with ``ScenarioLinkInput`` plus
    extra metadata (``name``, ``link_type``, ``severity_source``,
    ``closure_text``, ``start``, ``end``, ``lon``, ``lat``).
    """
    bc_cfg = cfg.get("baseline_closures") or {}
    _cache = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))
    source_path = Path(bc_cfg.get("source_path", f"{_cache}/closures.parquet"))
    match_cfg = bc_cfg.get("matching") or {}
    severity_map = bc_cfg.get("severity_map") or {}
    max_dist = float(match_cfg.get("max_distance_m", 50))
    require_ref = bool(match_cfg.get("require_road_ref_match", False))
    metric_epsg = int(cfg.get("crs_epsg", 5514))

    closures = load_closures(
        source_path,
        measurement_period={"start": date, "end": date},
        status_whitelist=bc_cfg.get("status_whitelist"),
    )
    if not closures:
        logger.info("No closures active on %s", date)
        return []

    if link_gdf is None:
        net_dir = Path(cfg.get("network", {}).get("output_dir", "outputs/baseline/network"))
        gpkg = net_dir / "network_links.gpkg"
        parquet = net_dir / "network_links.parquet"
        geojson = net_dir / "network_links.geojson"
        if gpkg.exists():
            link_gdf = gpd.read_file(gpkg)
        elif geojson.exists():
            link_gdf = gpd.read_file(geojson)
        elif parquet.exists():
            try:
                link_gdf = gpd.read_parquet(str(parquet))
            except Exception:
                link_gdf = gpd.GeoDataFrame(pd.read_parquet(str(parquet)))
        else:
            logger.error("No network link file found in %s", net_dir)
            return []

    if link_gdf.crs is None:
        link_gdf = link_gdf.set_crs(epsg=metric_epsg, allow_override=True)

    has_dir = "direction" in link_gdf.columns
    for attr in ("speed", "capacity", "lanes"):
        ab, ba = f"{attr}_ab", f"{attr}_ba"
        if ab in link_gdf.columns and ba in link_gdf.columns and attr not in link_gdf.columns:
            if has_dir:
                link_gdf[attr] = np.where(
                    link_gdf["direction"] == 1,
                    link_gdf[ab],
                    link_gdf[[ab, ba]].max(axis=1),
                )
            else:
                link_gdf[attr] = link_gdf[[ab, ba]].max(axis=1)

    link_closures = _match_closures_to_links(
        closures, link_gdf,
        max_distance_m=max_dist,
        require_road_ref_match=require_ref,
        metric_epsg=metric_epsg,
    )

    if not link_closures:
        logger.info("No closures matched to network links for date %s", date)
        return []

    link_lookup = link_gdf.drop_duplicates(subset="link_id").set_index("link_id")

    results: List[Dict[str, Any]] = []
    for lid, cl_list in link_closures.items():
        if lid not in link_lookup.index:
            continue
        row = link_lookup.loc[lid]
        severity = _pick_dominant_severity(cl_list)
        lanes = max(int(row.get("lanes", 2) or 2), 1)

        if severity == "full":
            closure_type = "full"
            lanes_remaining = 1
        else:
            cap_factor = float(
                severity_map.get(severity, {}).get("capacity_factor", 0.5)
            )
            closure_type = "lanes"
            lanes_remaining = max(1, int(math.ceil(lanes * cap_factor)))
            if lanes_remaining >= lanes:
                lanes_remaining = max(lanes - 1, 1)

        first = cl_list[0]
        results.append({
            "link_id": lid,
            "direction": "both",
            "closure_type": closure_type,
            "lanes": lanes,
            "lanes_remaining": lanes_remaining,
            "name": str(row.get("name", "") or ""),
            "link_type": str(row.get("link_type", "") or ""),
            "severity_source": severity,
            "closure_text": _closure_description(cl_list),
            "start": str(first.get("start", "") or ""),
            "end": str(first.get("end", "") or ""),
            "lon": float(first.get("lon", 0)),
            "lat": float(first.get("lat", 0)),
        })

    logger.info(
        "Date %s: %d closures -> %d affected links",
        date, len(closures), len(results),
    )
    return results


def closures_geojson_for_date(
    date: str,
    cfg: Dict[str, Any],
    link_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Dict[str, Any]:
    """Return a GeoJSON FeatureCollection of closure points for a given date.

    Each feature is a Point at the closure location, with properties
    including the matched ``link_id`` and scenario parameters.
    """
    items = closures_for_date(date, cfg, link_gdf)
    seen_coords: set[tuple[float, float]] = set()
    features = []
    for item in items:
        key = (round(item["lon"], 6), round(item["lat"], 6))
        if key in seen_coords:
            continue
        seen_coords.add(key)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item["lon"], item["lat"]],
            },
            "properties": {
                "link_id": item["link_id"],
                "direction": item["direction"],
                "closure_type": item["closure_type"],
                "lanes": item["lanes"],
                "lanes_remaining": item["lanes_remaining"],
                "name": item["name"],
                "link_type": item["link_type"],
                "severity": item["severity_source"],
                "closure_text": item["closure_text"],
                "start": item["start"],
                "end": item["end"],
            },
        })
    return {"type": "FeatureCollection", "features": features}
