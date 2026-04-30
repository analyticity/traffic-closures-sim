"""OSM data download, attribute aggregation, and link enrichment."""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
from aequilibrae import Project
from shapely.geometry import box

from sim._text import clean_text, normalize_name_upper
from sim.network.crs import network_bbox_wgs84_from_project
from sim.network.db import project_db, refresh_network

logger = logging.getLogger(__name__)

_MAJOR_REF_PROPAGATION_TYPES = frozenset({
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
})


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def normalize_ref(value: Any) -> str:
    """Upper-case, strip spaces / dashes / slashes."""
    text = str(value or "").upper().strip()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("\\", "/")
    text = text.replace("/", "")
    return text


def listify(value: Any) -> List[Any]:
    """Flatten *value* into a flat list; ``None`` / ``NaN`` become ``[]``."""
    if value is None:
        return []
    try:
        if isinstance(value, float) and math.isnan(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        out: List[Any] = []
        for item in value:
            out.extend(listify(item))
        return out
    return [value]


def extract_osm_ids(value: Any) -> List[int]:
    """Robust parser for ``osm_id`` values stored in the links table.

    Handles int, float, stringified ints, list-like strings, tuples, lists, sets.
    """
    out: List[int] = []
    if value is None:
        return out
    try:
        if isinstance(value, float) and math.isnan(value):
            return out
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(extract_osm_ids(item))
        return list(dict.fromkeys(out))

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, float):
        try:
            return [int(value)]
        except Exception:
            return out

    text = str(value).strip()
    if not text:
        return out

    for n in re.findall(r"-?\d+", text):
        try:
            out.append(int(n))
        except Exception:
            continue
    return list(dict.fromkeys(out))


def choose_best(counter: Counter) -> Optional[str]:
    """Return the most common element or ``None``."""
    if not counter:
        return None
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Geocoding / polygon buffering
# ---------------------------------------------------------------------------

def geocode_place(place_name: str) -> Any:
    """Return the WGS-84 boundary polygon for *place_name* via osmnx."""
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("osmnx is required for place buffering: pip install osmnx") from e
    gdf = ox.geocode_to_gdf(place_name)
    return gdf.geometry.iloc[0]


def buffer_polygon_km(poly: Any, buffer_km: float, crs_epsg: int) -> Any:
    """Buffer a WGS-84 polygon by *buffer_km* kilometres via a metric CRS round-trip."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True).transform
    to_wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True).transform

    poly_m = shp_transform(to_metric, poly)
    poly_buf = poly_m.buffer(buffer_km * 1000.0)
    return shp_transform(to_wgs84, poly_buf)


# ---------------------------------------------------------------------------
# OSM download + aggregation
# ---------------------------------------------------------------------------

def download_osm_drive_edges(
    *,
    place_name: Optional[str] = None,
    bbox_cfg: Optional[Iterable[float]] = None,
    polygon: Optional[Any] = None,
) -> gpd.GeoDataFrame:
    """Download the ``drive`` network from OSM and return edges as a GeoDataFrame."""
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("Missing osmnx. Install: pip install osmnx") from e

    common = dict(network_type="drive", simplify=False, retain_all=True, truncate_by_edge=True)

    if bbox_cfg:
        west, south, east, north = [float(x) for x in bbox_cfg]
        G = ox.graph_from_polygon(box(west, south, east, north), **common)
    elif polygon is not None:
        G = ox.graph_from_polygon(polygon, **common)
    elif place_name:
        G = ox.graph_from_place(place_name, **common)
    else:
        raise ValueError("Need either bbox_cfg, polygon, or place_name to download OSM edges")

    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)
    return edges.reset_index()


def aggregate_osm_edge_attributes(
    edges: gpd.GeoDataFrame,
) -> Dict[int, Dict[str, Optional[str]]]:
    """Group OSMnx edges by ``osmid`` and pick dominant ref / name / highway."""
    grouped: Dict[int, Dict[str, Counter]] = defaultdict(
        lambda: {"ref": Counter(), "name": Counter(), "highway": Counter()}
    )

    for _, row in edges.iterrows():
        osmids: list[int] = []
        for raw_id in listify(row.get("osmid")):
            try:
                osmids.append(int(raw_id))
            except Exception:
                continue
        if not osmids:
            continue

        refs = [v for v in (clean_text(v) for v in listify(row.get("ref"))) if v]
        names = [v for v in (clean_text(v) for v in listify(row.get("name"))) if v]
        highways = [v for v in (clean_text(v) for v in listify(row.get("highway"))) if v]

        for oid in osmids:
            if refs:
                grouped[oid]["ref"].update(refs)
            if names:
                grouped[oid]["name"].update(names)
            if highways:
                grouped[oid]["highway"].update(highways)

    out: Dict[int, Dict[str, Optional[str]]] = {}
    for oid, counters in grouped.items():
        ref = choose_best(counters["ref"])
        name = choose_best(counters["name"])
        highway = choose_best(counters["highway"])
        out[oid] = {
            "osm_ref": ref,
            "osm_ref_norm": normalize_ref(ref) if ref else None,
            "osm_name_raw": name,
            "osm_highway": highway,
        }
    return out


# ---------------------------------------------------------------------------
# Enrichment schema
# ---------------------------------------------------------------------------

_ENRICHMENT_FIELDS = [
    ("osm_ref", "OSM ref tag", "TEXT"),
    ("osm_ref_norm", "Normalized OSM ref tag", "TEXT"),
    ("osm_name_raw", "Original OSM name tag", "TEXT"),
    ("osm_highway", "Original OSM highway tag", "TEXT"),
]


def ensure_link_enrichment_fields(project: Project) -> None:
    """Add OSM enrichment columns to the links table if they don't exist yet."""
    for field_name, description, data_type in _ENRICHMENT_FIELDS:
        try:
            project.network.links.fields.add(field_name, description, data_type)
        except Exception:
            pass
    try:
        project.network.links.refresh_fields()
    except Exception:
        pass
    try:
        project.network.links.refresh()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Corridor ref-gap filling (second pass)
# ---------------------------------------------------------------------------

def fill_missing_refs_from_named_corridors(
    project: Project,
    project_dir: Path,
) -> Dict[str, int]:
    """Propagate the dominant ``ref`` along named major-road corridors with gaps.

    If a corridor shares the same name and most connected links already carry
    the same ref, short missing segments are filled in.
    """
    df = project.network.links.data.copy()
    if df.empty:
        return {"filled": 0, "components_used": 0}

    required = {"link_id", "a_node", "b_node", "modes", "link_type"}
    if not required.issubset(df.columns):
        return {"filled": 0, "components_used": 0}
    if "osm_ref_norm" not in df.columns:
        return {"filled": 0, "components_used": 0}

    work = df[
        df["modes"].astype(str).str.contains("c", na=False)
        & df["link_type"].astype(str).isin(_MAJOR_REF_PROPAGATION_TYPES)
    ].copy()
    if work.empty:
        return {"filled": 0, "components_used": 0}

    if "osm_name_raw" in work.columns:
        work["_name_src"] = work["osm_name_raw"]
    else:
        work["_name_src"] = None
    if "name" in work.columns:
        work["_name_src"] = work["_name_src"].where(work["_name_src"].notna(), work["name"])

    work["_name_norm"] = work["_name_src"].apply(normalize_name_upper)
    work["_ref_norm"] = work["osm_ref_norm"].fillna("").astype(str).str.strip()
    work["_ref_raw"] = (
        work["osm_ref"].fillna("").astype(str).str.strip()
        if "osm_ref" in work.columns
        else ""
    )

    work = work[work["_name_norm"] != ""].copy()
    if work.empty:
        return {"filled": 0, "components_used": 0}

    updates: List[Tuple[str, str, int]] = []
    components_used = 0

    for _, group in work.groupby("_name_norm"):
        by_link = {int(r["link_id"]): r for _, r in group.iterrows()}
        node_to_links: Dict[int, List[int]] = defaultdict(list)
        for _, r in group.iterrows():
            lid = int(r["link_id"])
            node_to_links[int(r["a_node"])].append(lid)
            node_to_links[int(r["b_node"])].append(lid)

        seen: set[int] = set()
        for lid0 in by_link:
            if lid0 in seen:
                continue
            stack = [lid0]
            component_link_ids: List[int] = []
            seen.add(lid0)
            while stack:
                lid = stack.pop()
                component_link_ids.append(lid)
                row = by_link[lid]
                for other in node_to_links[int(row["a_node"])] + node_to_links[int(row["b_node"])]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)

            comp = group[group["link_id"].astype(int).isin(component_link_ids)].copy()
            known = comp[comp["_ref_norm"] != ""].copy()
            missing = comp[comp["_ref_norm"] == ""].copy()
            if known.empty or missing.empty:
                continue

            ref_counts = Counter(known["_ref_norm"].tolist())
            raw_counts = Counter([x for x in known["_ref_raw"].tolist() if x])
            dominant_ref_norm, dominant_ref_count = ref_counts.most_common(1)[0]
            dominant_ref_raw = raw_counts.most_common(1)[0][0] if raw_counts else dominant_ref_norm

            if len(ref_counts) > 1 and (dominant_ref_count / len(known)) < 0.8:
                continue
            if len(known) < 2 and len(missing) > 2:
                continue

            components_used += 1
            for lid in missing["link_id"].astype(int).tolist():
                updates.append((dominant_ref_raw, dominant_ref_norm, lid))

    if not updates:
        return {"filled": 0, "components_used": 0}

    with project_db(project_dir, timeout=30.0) as conn:
        conn.executemany(
            """
            UPDATE links
               SET osm_ref = ?,
                   osm_ref_norm = ?
             WHERE link_id = ?
               AND (osm_ref IS NULL OR TRIM(osm_ref) = '')
            """,
            updates,
        )

    try:
        project.network.links.refresh()
    except Exception:
        pass

    return {"filled": len(updates), "components_used": components_used}


# ---------------------------------------------------------------------------
# Main enrichment entry point
# ---------------------------------------------------------------------------

def enrich_links_from_osm(
    project: Project,
    project_dir: Path,
    *,
    crs_epsg_hint: int,
    place_name: Optional[str] = None,
    bbox_cfg: Optional[Iterable[float]] = None,
    buffered_polygon: Optional[Any] = None,
) -> Dict[str, Any]:
    """Enrich AequilibraE links with OSM tags (ref, name, highway) via ``osm_id`` join.

    Performs two passes:
    1. Direct match via ``osm_id`` against a fresh OSMnx download.
    2. Fill remaining ref gaps along named major-road corridors.
    """
    _empty_stats = {
        "total_links": 0,
        "matched_osm_id": 0,
        "updated": 0,
        "filled_osm_ref": 0,
        "filled_osm_name_raw": 0,
        "filled_from_named_corridors": 0,
        "download_bbox_wgs84": None,
        "download_source": None,
    }

    ensure_link_enrichment_fields(project)

    links_df = project.network.links.data.copy()
    if links_df.empty:
        return _empty_stats

    if "osm_id" not in links_df.columns:
        logger.warning("links table has no osm_id column; OSM enrichment skipped")
        return {**_empty_stats, "total_links": len(links_df), "download_source": "missing_osm_id_column"}

    # --- Resolve download area ---
    effective_bbox: Optional[tuple] = None
    download_source: Optional[str] = None
    enrich_polygon = None

    try:
        effective_bbox = network_bbox_wgs84_from_project(project, crs_epsg_hint)
        download_source = "project_extent"
    except Exception as e:
        logger.warning("could not derive enrichment bbox from current project extent: %s", e)
        if bbox_cfg:
            effective_bbox = tuple(float(x) for x in bbox_cfg)
            download_source = "config_bbox"
        elif buffered_polygon is not None:
            enrich_polygon = buffered_polygon
            download_source = "buffered_polygon"
        elif place_name:
            download_source = "place_name"
        else:
            raise RuntimeError("Could not determine any OSM download area for enrichment")

    if effective_bbox is not None:
        logger.info("OSM enrichment download area (WGS84): %s", effective_bbox)
    elif enrich_polygon is not None:
        logger.info("OSM enrichment download area (buffered polygon): %s", enrich_polygon.bounds)
    else:
        logger.info("OSM enrichment download area: place_name=%s", place_name)

    edges = download_osm_drive_edges(
        place_name=place_name if (effective_bbox is None and enrich_polygon is None) else None,
        bbox_cfg=effective_bbox,
        polygon=enrich_polygon,
    )
    osm_map = aggregate_osm_edge_attributes(edges)

    # --- Match links to OSM attributes ---
    updates: list[tuple] = []
    matched_links = 0
    for _, row in links_df.iterrows():
        candidate_ids = extract_osm_ids(row.get("osm_id"))
        if not candidate_ids:
            continue
        attrs = None
        for oid in candidate_ids:
            attrs = osm_map.get(oid)
            if attrs:
                break
        if not attrs:
            continue
        matched_links += 1
        updates.append((
            attrs.get("osm_ref"),
            attrs.get("osm_ref_norm"),
            attrs.get("osm_name_raw"),
            attrs.get("osm_highway"),
            int(row["link_id"]),
        ))

    # --- Write to DB ---
    with project_db(project_dir, timeout=30.0) as conn:
        conn.execute(
            "UPDATE links SET osm_ref = NULL, osm_ref_norm = NULL, osm_name_raw = NULL, osm_highway = NULL"
        )
        if updates:
            conn.executemany(
                """
                UPDATE links
                   SET osm_ref = ?, osm_ref_norm = ?, osm_name_raw = ?, osm_highway = ?
                 WHERE link_id = ?
                """,
                updates,
            )

    try:
        project.network.links.refresh()
    except Exception:
        pass

    if not updates:
        logger.warning("no link matched OSM attributes by osm_id")

    corridor_fill = fill_missing_refs_from_named_corridors(project, project_dir)

    # --- Compute summary ---
    refreshed = project.network.links.data.copy()
    filled_ref = int(refreshed["osm_ref"].notna().sum()) if "osm_ref" in refreshed.columns else 0
    filled_name = int(refreshed["osm_name_raw"].notna().sum()) if "osm_name_raw" in refreshed.columns else 0

    return {
        "total_links": len(links_df),
        "matched_osm_id": matched_links,
        "updated": len(updates),
        "filled_osm_ref": filled_ref,
        "filled_osm_name_raw": filled_name,
        "filled_from_named_corridors": corridor_fill["filled"],
        "download_bbox_wgs84": list(effective_bbox) if effective_bbox is not None else None,
        "download_source": download_source,
    }
