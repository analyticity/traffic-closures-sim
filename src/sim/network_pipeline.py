from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from aequilibrae import Project
from shapely.geometry import box

from sim.aequilibrae_paths import resolve_project_database_path
from sim.defaults import SIM_DEFAULTS
from sim.io_project import get_metric_epsg, load_config

logger = logging.getLogger(__name__)

_MAP_CFG = SIM_DEFAULTS["network"]["map_export"]
NETWORK_MAP_PALETTE: Dict[str, str] = _MAP_CFG["palette"]
NETWORK_MAP_EXPORT_DPI: int = _MAP_CFG["dpi"]
NETWORK_MAP_EXPORT_FIGSIZE: Tuple[float, float] = tuple(_MAP_CFG["figsize"])


def _save_network_links_map_png(
    links_gdf: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
    png_path: Path,
    *,
    title: str,
    links_color: str,
    dpi: int = NETWORK_MAP_EXPORT_DPI,
    figsize: Tuple[float, float] = NETWORK_MAP_EXPORT_FIGSIZE,
) -> None:
    """Render links + bbox frame; used for maps_dir before/after snapshots."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=NETWORK_MAP_PALETTE["figure"])
    ax.set_facecolor(NETWORK_MAP_PALETTE["figure"])
    title_pt = max(14.0, min(24.0, figsize[0] * 1.05))
    links_gdf.plot(ax=ax, color=links_color, linewidth=0.25, zorder=1)
    bbox_gdf.boundary.plot(ax=ax, color=NETWORK_MAP_PALETTE["bbox"], linewidth=3.5, zorder=3)
    ax.set_title(title, color=NETWORK_MAP_PALETTE["title"], fontsize=title_pt)
    ax.set_axis_off()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=NETWORK_MAP_PALETTE["figure"],
        pad_inches=0.05,
    )
    plt.close(fig)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _project_db_path(project_dir: Path) -> Path:
    return resolve_project_database_path(project_dir)

_MAJOR_REF_PROPAGATION_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}


def _normalize_name(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("-", "")
    text = text.replace("/", "")
    text = text.replace("\\", "")
    return text


def _fill_missing_refs_from_named_corridors(
    project: Project,
    project_dir: Path,
) -> Dict[str, int]:
    """
    Second pass:
    If a major-road corridor has the same name and most of its connected links
    already have the same ref, propagate that ref to short missing gaps.
    This helps when OSM ``ref`` is missing on short segments along a named major corridor.
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

    work["_name_norm"] = work["_name_src"].apply(_normalize_name)
    work["_ref_norm"] = work["osm_ref_norm"].fillna("").astype(str).str.strip()
    work["_ref_raw"] = work["osm_ref"].fillna("").astype(str).str.strip() if "osm_ref" in work.columns else ""

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

        for lid0 in by_link.keys():
            if lid0 in seen:
                continue

            stack = [lid0]
            component_link_ids: List[int] = []
            seen.add(lid0)

            while stack:
                lid = stack.pop()
                component_link_ids.append(lid)
                row = by_link[lid]
                neigh = node_to_links[int(row["a_node"])] + node_to_links[int(row["b_node"])]

                for other in neigh:
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

            # Be conservative: fill only when the component is clearly dominated by one ref
            if len(ref_counts) > 1 and (dominant_ref_count / len(known)) < 0.8:
                continue

            # Also avoid filling huge ambiguous components from just 1 known segment
            if len(known) < 2 and len(missing) > 2:
                continue

            components_used += 1
            for lid in missing["link_id"].astype(int).tolist():
                updates.append((dominant_ref_raw, dominant_ref_norm, lid))

    if not updates:
        return {"filled": 0, "components_used": 0}

    db_path = _project_db_path(project_dir)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
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
        conn.commit()
    finally:
        conn.close()

    try:
        project.network.links.refresh()
    except Exception:
        pass

    return {"filled": int(len(updates)), "components_used": int(components_used)}


def create_or_open_project(project_dir: Path) -> Project:
    """
    Opens an existing AequilibraE project if it exists, otherwise creates a new one.
    """
    project = Project()
    has_db = any(project_dir.glob("*.sqlite")) or any(project_dir.glob("*.db")) or any(project_dir.glob("*.sqlite3"))
    if has_db:
        project.open(str(project_dir))
    else:
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project.new(str(project_dir))
    return project


# --- CRS helpers ---

def _guess_crs_from_coords(geoms: gpd.GeoSeries, fallback_epsg: int = 5514) -> str:
    """
    Heuristic:
    - If coords look like degrees (|x| <= 180 and |y| <= 90-ish) -> EPSG:4326
    - else -> EPSG:{fallback_epsg}
    """
    s = geoms.dropna()
    if s.empty:
        return f"EPSG:{fallback_epsg}"

    s = s.iloc[:200]
    xs = []
    ys = []
    for g in s:
        try:
            c = g.centroid
            xs.append(float(c.x))
            ys.append(float(c.y))
        except Exception:
            continue

    if not xs or not ys:
        return f"EPSG:{fallback_epsg}"

    max_abs_x = max(abs(x) for x in xs)
    max_abs_y = max(abs(y) for y in ys)

    if max_abs_x <= 180.0 and max_abs_y <= 90.0:
        return "EPSG:4326"
    return f"EPSG:{fallback_epsg}"


def _as_gdf(df, crs_hint_epsg: int) -> gpd.GeoDataFrame:
    g = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None))
    if g.crs is None:
        guessed = _guess_crs_from_coords(g.geometry, fallback_epsg=crs_hint_epsg)
        g = g.set_crs(guessed, allow_override=True)
    return g


def _network_links_gdf_with_crs(project: Project, crs_epsg_hint: int) -> gpd.GeoDataFrame:
    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        raise RuntimeError("Network links missing geometry/empty")
    return _as_gdf(links, crs_epsg_hint)


def _network_bbox_wgs84_from_project(
    project: Project,
    crs_epsg_hint: int,
    *,
    pad_ratio: float = 0.005,
    min_pad_deg: float = 0.0015,
) -> Tuple[float, float, float, float]:
    """
    Computes current network extent from ACTUAL project links and converts it to WGS84.
    Smaller padding than before, so enrichment area stays closer to the real network.
    """
    gdf = _network_links_gdf_with_crs(project, crs_epsg_hint)

    if gdf.crs is None:
        raise RuntimeError("Could not determine CRS of project links")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    minx, miny, maxx, maxy = map(float, gdf.total_bounds)

    dx = max(maxx - minx, 0.0)
    dy = max(maxy - miny, 0.0)

    padx = max(dx * pad_ratio, min_pad_deg)
    pady = max(dy * pad_ratio, min_pad_deg)

    return (minx - padx, miny - pady, maxx + padx, maxy + pady)


def _native_bbox_from_wgs84_bbox(
    project: Project,
    bbox_wgs84: Iterable[float],
    crs_epsg_hint: int,
) -> Tuple[float, float, float, float]:
    """
    Converts config bbox in WGS84 into native link CRS used by the current project.
    This makes trimming safe even when DB coordinates are not EPSG:4326.
    """
    west, south, east, north = [float(x) for x in bbox_wgs84]
    bbox_geom = box(west, south, east, north)

    links_gdf = _network_links_gdf_with_crs(project, crs_epsg_hint)
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs="EPSG:4326")

    if links_gdf.crs is not None and links_gdf.crs.to_epsg() != 4326:
        bbox_gdf = bbox_gdf.to_crs(links_gdf.crs)

    minx, miny, maxx, maxy = map(float, bbox_gdf.total_bounds)
    if not (minx < maxx and miny < maxy):
        raise RuntimeError(f"Invalid converted native bbox: {(minx, miny, maxx, maxy)}")

    return minx, miny, maxx, maxy


# --- Deterministic trimming ---

def compute_bbox_from_links_raw(project: Project) -> Tuple[float, float, float, float]:
    """
    Returns bbox from link geometries exactly as stored in the DB (raw/native coordinates).
    """
    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        raise RuntimeError("Network links missing geometry/empty")

    minx, miny, maxx, maxy = map(float, links.total_bounds)
    if not (minx < maxx and miny < maxy):
        raise RuntimeError(f"Invalid link bounds: {(minx, miny, maxx, maxy)}")
    return minx, miny, maxx, maxy


_CORRIDOR_LINK_TYPES = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link",
})


def _compute_urban_trim_bbox(
    project: Project,
    *,
    pad_ratio: float = 0.02,
    quantile: float = 0.01,
) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box of the "urban core" -- nodes that touch at least
    one non-highway/non-primary link.  Long corridor-only segments are excluded
    so they don't inflate the trim area.  A quantile trim + padding is applied.
    """
    import numpy as np

    links = project.network.links.data
    nodes = project.network.nodes.data
    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))

    if "link_type" in links_gdf.columns:
        local = links_gdf[
            ~links_gdf["link_type"].astype(str).str.strip().str.lower().isin(_CORRIDOR_LINK_TYPES)
        ]
        urban_ids = set(local["a_node"].astype(int)) | set(local["b_node"].astype(int))
    else:
        urban_ids = set(nodes_gdf["node_id"].astype(int))

    urban = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(urban_ids)]
    if len(urban) < 50:
        urban = nodes_gdf

    xs = urban.geometry.x.to_numpy(dtype=float)
    ys = urban.geometry.y.to_numpy(dtype=float)

    if quantile > 0:
        lx, hx = np.quantile(xs, quantile), np.quantile(xs, 1 - quantile)
        ly, hy = np.quantile(ys, quantile), np.quantile(ys, 1 - quantile)
    else:
        lx, hx = xs.min(), xs.max()
        ly, hy = ys.min(), ys.max()

    dx = (hx - lx) * pad_ratio
    dy = (hy - ly) * pad_ratio
    return (lx - dx, ly - dy, hx + dx, hy + dy)


def trim_network_to_bbox_raw(
    project: Project,
    bbox: Tuple[float, float, float, float],
    project_dir: Path,
) -> Dict[str, int]:
    """
    Trim **non-corridor** links to *bbox* (native CRS).

    Corridor links (motorway/trunk/primary) are always kept so that highway
    corridors extend beyond the urban core to where gateways are placed.
    Only secondary-and-below suburban links outside the bbox are removed.
    Orphan nodes no longer referenced by any kept link are cleaned up.
    """
    west, south, east, north = bbox
    rect = box(west, south, east, north)

    nodes = project.network.nodes.data
    links = project.network.links.data

    if "geometry" not in nodes.columns or len(nodes) == 0:
        raise RuntimeError("Network nodes missing geometry/empty")
    if "geometry" not in links.columns or len(links) == 0:
        raise RuntimeError("Network links missing geometry/empty")
    if "node_id" not in nodes.columns:
        raise RuntimeError("Network nodes missing node_id")
    if "a_node" not in links.columns or "b_node" not in links.columns:
        raise RuntimeError("Network links missing a_node/b_node")
    if "link_id" not in links.columns:
        raise RuntimeError("Network links missing link_id")

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=getattr(nodes, "crs", None))
    links_gdf = gpd.GeoDataFrame(links, geometry="geometry", crs=getattr(links, "crs", None))

    inside_node_ids = set(
        nodes_gdf.loc[nodes_gdf.geometry.within(rect), "node_id"].astype(int)
    )

    a_in = links_gdf["a_node"].astype(int).isin(inside_node_ids)
    b_in = links_gdf["b_node"].astype(int).isin(inside_node_ids)

    is_corridor = links_gdf["link_type"].astype(str).str.strip().str.lower().isin(
        _CORRIDOR_LINK_TYPES
    ) if "link_type" in links_gdf.columns else pd.Series(False, index=links_gdf.index)

    keep_link = is_corridor | (a_in & b_in)

    kept_link_ids = set(links_gdf.loc[keep_link, "link_id"].astype(int))
    kept_links = links_gdf.loc[keep_link]
    referenced_node_ids = (
        set(kept_links["a_node"].astype(int)) | set(kept_links["b_node"].astype(int))
    )

    link_ids_to_remove = [
        int(x) for x in links_gdf["link_id"] if int(x) not in kept_link_ids
    ]
    node_ids_to_remove = [
        int(x) for x in nodes_gdf["node_id"] if int(x) not in referenced_node_ids
    ]

    links_deleted = 0
    for link_id in link_ids_to_remove:
        try:
            project.network.links.delete(link_id)
            links_deleted += 1
        except Exception:
            pass

    if node_ids_to_remove:
        db_path = _project_db_path(project_dir)
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        placeholders = ",".join("?" * len(node_ids_to_remove))
        conn.execute(f"DELETE FROM nodes WHERE node_id IN ({placeholders})", node_ids_to_remove)
        conn.commit()
        conn.close()

    nodes_deleted = len(node_ids_to_remove)

    try:
        project.network.refresh()
        project.network.nodes.refresh()
        project.network.links.refresh()
    except Exception:
        pass

    return {"nodes_deleted": nodes_deleted, "links_deleted": links_deleted}


# --- Non-drivable link removal (OSM highway → link_type) ---

# link_type values that are not motor-vehicle roads; keep configurable via sim.yaml
DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES = frozenset(
    {
        "footway",
        "path",
        "pedestrian",
        "track",
        "steps",
        "cycleway",
        "bridleway",
        "elevator",
        "escalator",
        "corridor",
        "platform",
        "proposed",
        "crossing",
    }
)


def prune_orphan_nodes(project: Project, project_dir: Path) -> int:
    """Delete nodes not referenced by any link (raw SQLite; then refresh project)."""
    db_path = _project_db_path(project_dir)
    conn = sqlite3.connect(str(db_path), timeout=120.0)
    try:
        cur = conn.execute(
            """
            DELETE FROM nodes
             WHERE node_id NOT IN (
                 SELECT a_node FROM links
                 UNION
                 SELECT b_node FROM links
             )
            """
        )
        deleted = int(cur.rowcount or 0)
        conn.commit()
    finally:
        conn.close()

    try:
        project.network.refresh()
        project.network.nodes.refresh()
        project.network.links.refresh()
    except Exception:
        pass

    return deleted


def remove_non_drivable_links(
    project: Project,
    project_dir: Path,
    *,
    excluded_link_types: Set[str],
    require_mode_car: bool = True,
) -> Dict[str, int]:
    """Drop pedestrian / cycle-only OSM ways and optionally any link without car mode ``c``."""
    try:
        project.network.links.refresh()
    except Exception:
        pass

    links = project.network.links.data
    if links.empty or "link_id" not in links.columns:
        return {"links_removed": 0, "nodes_pruned": 0}

    if "link_type" in links.columns:
        lt = links["link_type"].astype(str).str.lower().str.strip()
        mask_excluded = lt.isin({x.lower() for x in excluded_link_types})
    else:
        mask_excluded = pd.Series(False, index=links.index)

    if require_mode_car and "modes" in links.columns:
        has_car = links["modes"].astype(str).str.contains("c", na=False, regex=False)
        mask_remove = mask_excluded | (~has_car)
    else:
        mask_remove = mask_excluded

    ids_to_remove = sorted({int(x) for x in links.loc[mask_remove, "link_id"].tolist()})
    if not ids_to_remove:
        return {"links_removed": 0, "nodes_pruned": 0}

    db_path = _project_db_path(project_dir)
    conn = sqlite3.connect(str(db_path), timeout=120.0)
    chunk = 450
    try:
        for i in range(0, len(ids_to_remove), chunk):
            part = ids_to_remove[i : i + chunk]
            ph = ",".join("?" * len(part))
            conn.execute(f"DELETE FROM links WHERE link_id IN ({ph})", part)
        conn.commit()
    finally:
        conn.close()

    removed = len(ids_to_remove)

    try:
        project.network.refresh()
        project.network.links.refresh()
        project.network.nodes.refresh()
    except Exception:
        pass

    pruned = prune_orphan_nodes(project, project_dir)

    return {"links_removed": removed, "nodes_pruned": pruned}


def remove_disconnected_components_keep_largest(
    project: Project,
    project_dir: Path,
) -> Dict[str, Any]:
    """
    Delete links that are not in the largest undirected connected component
    (component size = number of links). Isolated nodes are removed via ``prune_orphan_nodes``.
    Parallel links (same endpoints) are kept or dropped with their component.
    """
    try:
        project.network.links.refresh()
    except Exception:
        pass

    links = project.network.links.data
    empty = {
        "links_removed": 0,
        "nodes_pruned": 0,
        "components": 0,
        "kept_links": 0,
    }
    if links.empty or "link_id" not in links.columns:
        return empty
    if "a_node" not in links.columns or "b_node" not in links.columns:
        raise RuntimeError("Network links missing a_node/b_node")

    G = nx.MultiGraph()
    for _, row in links.iterrows():
        G.add_edge(int(row["a_node"]), int(row["b_node"]), key=int(row["link_id"]))

    n_links = int(len(links))
    if G.number_of_nodes() == 0:
        return {**empty, "components": 0, "kept_links": n_links}

    components = list(nx.connected_components(G))
    n_comp = len(components)
    if n_comp <= 1:
        return {
            "links_removed": 0,
            "nodes_pruned": 0,
            "components": n_comp,
            "kept_links": n_links,
        }

    def _edge_count(nodes: Set[Any]) -> int:
        return int(G.subgraph(nodes).number_of_edges())

    best_nodes = max(components, key=_edge_count)
    subgraph = G.subgraph(best_nodes)
    kept_link_ids = {int(k) for _u, _v, k in subgraph.edges(keys=True)}
    all_link_ids = {int(x) for x in links["link_id"].tolist()}
    ids_to_remove = sorted(all_link_ids - kept_link_ids)
    if not ids_to_remove:
        return {
            "links_removed": 0,
            "nodes_pruned": 0,
            "components": n_comp,
            "kept_links": n_links,
        }

    db_path = _project_db_path(project_dir)
    conn = sqlite3.connect(str(db_path), timeout=120.0)
    chunk = 450
    try:
        for i in range(0, len(ids_to_remove), chunk):
            part = ids_to_remove[i : i + chunk]
            ph = ",".join("?" * len(part))
            conn.execute(f"DELETE FROM links WHERE link_id IN ({ph})", part)
        conn.commit()
    finally:
        conn.close()

    try:
        project.network.refresh()
        project.network.links.refresh()
        project.network.nodes.refresh()
    except Exception:
        pass

    pruned = prune_orphan_nodes(project, project_dir)
    return {
        "links_removed": len(ids_to_remove),
        "nodes_pruned": pruned,
        "components": n_comp,
        "kept_links": len(kept_link_ids),
    }


# --- OSM enrichment helpers ---

def _normalize_ref(value: Any) -> str:
    text = str(value or "").upper().strip()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("\\", "/")
    text = text.replace("/", "")
    return text


def _listify(value: Any) -> List[Any]:
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
            out.extend(_listify(item))
        return out
    return [value]


def _extract_osm_ids(value: Any) -> List[int]:
    """
    Robust parser for osm_id values stored in the links table.
    Handles:
    - int / float
    - stringified ints
    - list-like strings: "[123, 456]"
    - tuples / lists / sets
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
            out.extend(_extract_osm_ids(item))
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

    nums = re.findall(r"-?\d+", text)
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            continue

    return list(dict.fromkeys(out))


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _choose_best(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _geocode_place(place_name: str) -> "Polygon":
    """Return the WGS84 boundary polygon for *place_name* via osmnx."""
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("osmnx is required for place buffering: pip install osmnx") from e
    gdf = ox.geocode_to_gdf(place_name)
    return gdf.geometry.iloc[0]


def _buffer_polygon_km(poly: "Polygon", buffer_km: float, crs_epsg: int) -> "Polygon":
    """Buffer a WGS84 polygon by *buffer_km* kilometres via a metric CRS round-trip."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    to_metric = Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True).transform
    to_wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True).transform

    poly_m = shp_transform(to_metric, poly)
    poly_buf = poly_m.buffer(buffer_km * 1000.0)
    return shp_transform(to_wgs84, poly_buf)



def _download_osm_drive_edges(
    *,
    place_name: Optional[str] = None,
    bbox_cfg: Optional[Iterable[float]] = None,
    polygon: Optional[Any] = None,
) -> gpd.GeoDataFrame:
    try:
        import osmnx as ox
    except ImportError as e:
        raise RuntimeError("Missing osmnx. Install: pip install osmnx") from e

    if bbox_cfg:
        west, south, east, north = [float(x) for x in bbox_cfg]
        dl_polygon = box(west, south, east, north)
        G = ox.graph_from_polygon(
            dl_polygon,
            network_type="drive",
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
        )
    elif polygon is not None:
        G = ox.graph_from_polygon(
            polygon,
            network_type="drive",
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
        )
    elif place_name:
        G = ox.graph_from_place(
            place_name,
            network_type="drive",
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
        )
    else:
        raise ValueError("Need either bbox_cfg, polygon, or place_name to download OSM edges")

    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)
    edges = edges.reset_index()
    return edges


def _aggregate_osm_edge_attributes(edges: gpd.GeoDataFrame) -> Dict[int, Dict[str, Optional[str]]]:
    """
    Group OSMnx edges by osmid and aggregate attributes like ref/name/highway.
    AequilibraE links often preserve the OSM way id as `osm_id`, so this is the cleanest join key.
    """
    grouped: Dict[int, Dict[str, Counter]] = defaultdict(lambda: {
        "ref": Counter(),
        "name": Counter(),
        "highway": Counter(),
    })

    for _, row in edges.iterrows():
        osmids = []
        for raw_id in _listify(row.get("osmid")):
            try:
                osmids.append(int(raw_id))
            except Exception:
                continue

        if not osmids:
            continue

        refs = [_clean_text(v) for v in _listify(row.get("ref"))]
        refs = [v for v in refs if v]
        names = [_clean_text(v) for v in _listify(row.get("name"))]
        names = [v for v in names if v]
        highways = [_clean_text(v) for v in _listify(row.get("highway"))]
        highways = [v for v in highways if v]

        for oid in osmids:
            if refs:
                grouped[oid]["ref"].update(refs)
            if names:
                grouped[oid]["name"].update(names)
            if highways:
                grouped[oid]["highway"].update(highways)

    out: Dict[int, Dict[str, Optional[str]]] = {}
    for oid, counters in grouped.items():
        ref = _choose_best(counters["ref"])
        name = _choose_best(counters["name"])
        highway = _choose_best(counters["highway"])
        out[oid] = {
            "osm_ref": ref,
            "osm_ref_norm": _normalize_ref(ref) if ref else None,
            "osm_name_raw": name,
            "osm_highway": highway,
        }
    return out


def _ensure_link_enrichment_fields(project: Project) -> None:
    fields_to_add = [
        ("osm_ref", "OSM ref tag", "TEXT"),
        ("osm_ref_norm", "Normalized OSM ref tag", "TEXT"),
        ("osm_name_raw", "Original OSM name tag", "TEXT"),
        ("osm_highway", "Original OSM highway tag", "TEXT"),
    ]

    for field_name, description, data_type in fields_to_add:
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


def enrich_links_from_osm(
    project: Project,
    project_dir: Path,
    *,
    crs_epsg_hint: int,
    place_name: Optional[str] = None,
    bbox_cfg: Optional[Iterable[float]] = None,
    buffered_polygon: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Enrich AequilibraE links with OSM tags using existing `osm_id` in the links table.

    Important:
    - Prefer ACTUAL current project extent for OSM download.
    - Use a smaller padding than before.
    - Use unsimplified OSM graph to preserve original way refs better.
    - Do a second pass to fill short missing ref gaps inside the same named corridor.
    """
    _ensure_link_enrichment_fields(project)

    links_df = project.network.links.data.copy()
    if links_df.empty:
        return {
            "total_links": 0,
            "matched_osm_id": 0,
            "updated": 0,
            "filled_osm_ref": 0,
            "filled_osm_name_raw": 0,
            "filled_from_named_corridors": 0,
            "download_bbox_wgs84": None,
            "download_source": None,
        }

    if "osm_id" not in links_df.columns:
        logger.warning("links table has no osm_id column; OSM enrichment skipped")
        return {
            "total_links": int(len(links_df)),
            "matched_osm_id": 0,
            "updated": 0,
            "filled_osm_ref": 0,
            "filled_osm_name_raw": 0,
            "filled_from_named_corridors": 0,
            "download_bbox_wgs84": None,
            "download_source": "missing_osm_id_column",
        }

    effective_bbox = None
    download_source = None

    enrich_polygon = None
    try:
        effective_bbox = _network_bbox_wgs84_from_project(project, crs_epsg_hint)
        download_source = "project_extent"
    except Exception as e:
        logger.warning("could not derive enrichment bbox from current project extent: %s", e)
        if bbox_cfg:
            effective_bbox = tuple(float(x) for x in bbox_cfg)
            download_source = "config_bbox"
        elif buffered_polygon is not None:
            effective_bbox = None
            enrich_polygon = buffered_polygon
            download_source = "buffered_polygon"
        elif place_name:
            effective_bbox = None
            download_source = "place_name"
        else:
            raise RuntimeError("Could not determine any OSM download area for enrichment")

    if effective_bbox is not None:
        logger.info("OSM enrichment download area (WGS84): %s", effective_bbox)
    elif enrich_polygon is not None:
        logger.info("OSM enrichment download area (buffered polygon): %s", enrich_polygon.bounds)
    else:
        logger.info("OSM enrichment download area: place_name=%s", place_name)

    edges = _download_osm_drive_edges(
        place_name=place_name if (effective_bbox is None and enrich_polygon is None) else None,
        bbox_cfg=effective_bbox,
        polygon=enrich_polygon,
    )
    osm_map = _aggregate_osm_edge_attributes(edges)

    updates = []
    matched_links = 0

    for _, row in links_df.iterrows():
        candidate_ids = _extract_osm_ids(row.get("osm_id"))
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

    db_path = _project_db_path(project_dir)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        conn.execute(
            """
            UPDATE links
               SET osm_ref = NULL,
                   osm_ref_norm = NULL,
                   osm_name_raw = NULL,
                   osm_highway = NULL
            """
        )

        if updates:
            conn.executemany(
                """
                UPDATE links
                   SET osm_ref = ?,
                       osm_ref_norm = ?,
                       osm_name_raw = ?,
                       osm_highway = ?
                 WHERE link_id = ?
                """,
                updates,
            )
        conn.commit()
    finally:
        conn.close()

    try:
        project.network.links.refresh()
    except Exception:
        pass

    if not updates:
        logger.warning("no link matched OSM attributes by osm_id")

    corridor_fill_stats = _fill_missing_refs_from_named_corridors(project, project_dir)

    refreshed = project.network.links.data.copy()

    filled_ref = 0
    filled_name = 0
    if not refreshed.empty:
        if "osm_ref" in refreshed.columns:
            filled_ref = int(refreshed["osm_ref"].notna().sum())
        if "osm_name_raw" in refreshed.columns:
            filled_name = int(refreshed["osm_name_raw"].notna().sum())

    return {
        "total_links": int(len(links_df)),
        "matched_osm_id": int(matched_links),
        "updated": int(len(updates)),
        "filled_osm_ref": int(filled_ref),
        "filled_osm_name_raw": int(filled_name),
        "filled_from_named_corridors": int(corridor_fill_stats["filled"]),
        "download_bbox_wgs84": list(effective_bbox) if effective_bbox is not None else None,
        "download_source": download_source,
    }


# --- Main ---

def build_network_from_osm(
    config_path: str | Path = "config/brno/sim.yaml",
    outputs_dir: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    if outputs_dir is None:
        outputs_dir = cfg.get("network", {}).get("maps_dir", "outputs/baseline/maps")

    project_dir = Path(cfg["project_path"])
    crs_epsg_hint = get_metric_epsg(cfg)

    osm_cfg = cfg.get("osm", {}) or {}
    place_name: Optional[str] = osm_cfg.get("place_name")
    bbox_cfg = cfg.get("model_bbox") or osm_cfg.get("bbox")
    buffer_km = float(osm_cfg.get("buffer_km", 0))

    if not bbox_cfg and not place_name:
        raise ValueError("Missing config model_bbox or osm.place_name in sim.yaml")

    # Two bboxes when buffer_km is set:
    #   model_bbox_polygon  – buffer_km around the city (desired model extent, used for trim)
    #   buffered_polygon    – buffer_km + margin (larger download area for connectivity)
    # After download + isolated-component filter we trim back to model_bbox_polygon,
    # so the network doesn't have long highway tentacles.
    _CONNECTIVITY_MARGIN_KM = 5.0
    buffered_polygon = None
    model_bbox_polygon = None
    if place_name and buffer_km > 0 and not bbox_cfg:
        download_km = buffer_km + _CONNECTIVITY_MARGIN_KM
        logger.info(
            "Geocoding %r, buffer %s km (+ %s km connectivity margin -> %s km download)",
            place_name,
            buffer_km,
            _CONNECTIVITY_MARGIN_KM,
            download_km,
        )
        place_poly = _geocode_place(place_name)
        model_bbox_polygon = box(*_buffer_polygon_km(place_poly, buffer_km, crs_epsg_hint).bounds)
        buffered_polygon = box(*_buffer_polygon_km(place_poly, download_km, crs_epsg_hint).bounds)
        logger.info("model bbox (WGS84): %s", model_bbox_polygon.bounds)
        logger.info("download bbox (WGS84): %s", buffered_polygon.bounds)

    project = create_or_open_project(project_dir)

    links_before = project.network.count_links()
    nodes_before = project.network.count_nodes()

    # --- Detect stale network when buffer_km changed ---
    if links_before > 0 and buffered_polygon is not None:
        try:
            net_bbox = _network_bbox_wgs84_from_project(project, crs_epsg_hint, pad_ratio=0, min_pad_deg=0)
            net_box = box(*net_bbox)
            buf_box = box(*buffered_polygon.bounds)
            coverage = net_box.area / buf_box.area if buf_box.area > 0 else 1.0
            if coverage < 0.95:
                logger.info(
                    "Existing network covers only %s of buffered area; rebuilding project from scratch",
                    format(coverage, ".0%"),
                )
                project.close()
                shutil.rmtree(project_dir)
                project = create_or_open_project(project_dir)
                links_before = 0
                nodes_before = 0
            else:
                logger.info("Existing network already covers the buffered area; skipping rebuild")
        except Exception as e:
            logger.warning(
                "could not check network extent vs buffer (%s), keeping existing network",
                e,
            )

    # --- Build network (only if empty) ---
    osm_imported_this_run = False
    if links_before == 0 or nodes_before == 0:
        osm_imported_this_run = True
        if bbox_cfg:
            west, south, east, north = [float(x) for x in bbox_cfg]
            model_area = box(west, south, east, north)
            project.network.create_from_osm(model_area=model_area)
            logger.info("Network created from model_bbox (WGS84): %s", bbox_cfg)
        elif buffered_polygon is not None:
            project.network.create_from_osm(model_area=buffered_polygon)
            logger.info("Network from place_name %r + %s km buffer", place_name, buffer_km)
        else:
            project.network.create_from_osm(place_name=place_name)
            logger.info("Network created from place_name: %s", place_name)

    # --- Bbox for maps / metadata only (no link deletion) ---
    if bbox_cfg:
        bbox_native = _native_bbox_from_wgs84_bbox(project, bbox_cfg, crs_epsg_hint)
        bbox_wgs84_used = tuple(float(x) for x in bbox_cfg)
        logger.info("Reference bbox from config (WGS84): %s", bbox_wgs84_used)
        logger.info("Reference bbox (native): %s", bbox_native)
    else:
        bbox_native = compute_bbox_from_links_raw(project)
        bbox_wgs84_used = None
        logger.info("Network extent bbox (native): %s", bbox_native)

    out_dir = Path(outputs_dir)
    _ensure_dir(out_dir)

    if osm_imported_this_run:
        try:
            links_df_pre = project.network.links.data
            if len(links_df_pre) > 0 and "geometry" in links_df_pre.columns:
                links_gdf_pre = gpd.GeoDataFrame(
                    links_df_pre, geometry="geometry", crs=getattr(links_df_pre, "crs", None)
                )
                bbox_pre = compute_bbox_from_links_raw(project)
                bbox_poly_pre = box(*bbox_pre)
                bbox_gdf_pre = gpd.GeoDataFrame({"geometry": [bbox_poly_pre]}, crs=links_gdf_pre.crs)
                png_before = out_dir / "links_native_before_processing.png"
                _save_network_links_map_png(
                    links_gdf_pre,
                    bbox_gdf_pre,
                    png_before,
                    title="Network after OSM import (before filters & trim)",
                    links_color=NETWORK_MAP_PALETTE["links_before"],
                )
                logger.info("Wrote (pre-processing map): %s", png_before)
        except Exception as e:
            logger.warning("could not write pre-processing map (%s)", e)
    else:
        logger.info(
            "Skipping %s (loaded existing project; delete project DB to regenerate)",
            "links_native_before_processing.png",
        )

    trim_stats = {"nodes_deleted": 0, "links_deleted": 0}

    # --- Keep motor-vehicle network only (pedestrian / cycle OSM ways, non-car modes) ---
    dn = (cfg.get("network") or {}).get("drivable_network") or {}
    drivable_stats: Dict[str, Any] = {}
    if dn.get("enabled", True):
        raw_excl = dn.get("excluded_link_types")
        if raw_excl:
            excluded = {str(x).strip().lower() for x in raw_excl if str(x).strip()}
        else:
            excluded = set(DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES)
        require_car = bool(dn.get("require_mode_car", True))
        logger.info(
            "Drivable-network filter: %s excluded link_type values, require_mode_car=%s",
            len(excluded),
            require_car,
        )
        drivable_stats = remove_non_drivable_links(
            project,
            project_dir,
            excluded_link_types=excluded,
            require_mode_car=require_car,
        )
        logger.info("  removed links: %s", drivable_stats.get("links_removed", 0))
        logger.info("  pruned orphan nodes: %s", drivable_stats.get("nodes_pruned", 0))

    iso = (cfg.get("network") or {}).get("isolated_components") or {}
    isolated_stats: Dict[str, Any] = {}
    if iso.get("enabled", True):
        logger.info("Largest-component filter: removing disconnected subgraphs")
        isolated_stats = remove_disconnected_components_keep_largest(project, project_dir)
        logger.info(
            "  components (before): %s | removed links: %s | pruned orphan nodes: %s",
            isolated_stats.get("components", 0),
            isolated_stats.get("links_removed", 0),
            isolated_stats.get("nodes_pruned", 0),
        )

    # --- Trim: compute urban-core bbox, then cut highway tentacles ---
    # Even with buffer_km=0, place_name downloads include highways that extend
    # to the admin boundary edge.  We compute the extent of "urban" nodes
    # (those touching at least one non-highway link), pad it, and trim.
    trim_bbox = _compute_urban_trim_bbox(project, pad_ratio=0.02)
    if model_bbox_polygon is not None:
        model_trim = _native_bbox_from_wgs84_bbox(
            project, list(model_bbox_polygon.bounds), crs_epsg_hint,
        )
        trim_bbox = (
            min(trim_bbox[0], model_trim[0]),
            min(trim_bbox[1], model_trim[1]),
            max(trim_bbox[2], model_trim[2]),
            max(trim_bbox[3], model_trim[3]),
        )
    logger.info("Trimming network to urban-core bbox (native): %s", trim_bbox)
    trim_stats = trim_network_to_bbox_raw(project, trim_bbox, project_dir)
    logger.info("  trimmed: %s links, %s nodes", trim_stats["links_deleted"], trim_stats["nodes_deleted"])
    if trim_stats["links_deleted"] > 0 and iso.get("enabled", True):
        logger.info("  re-running isolated-component filter after trim")
        iso2 = remove_disconnected_components_keep_largest(project, project_dir)
        trim_stats["post_trim_iso_links_removed"] = iso2.get("links_removed", 0)
        trim_stats["post_trim_iso_nodes_pruned"] = iso2.get("nodes_pruned", 0)
        logger.info(
            "  post-trim components: %s | removed links: %s",
            iso2.get("components", 0),
            iso2.get("links_removed", 0),
        )
    bbox_native = compute_bbox_from_links_raw(project)
    logger.info("  final network extent (native): %s", bbox_native)

    # --- OSM enrichment of links ---
    enrich_stats = enrich_links_from_osm(
        project,
        project_dir,
        crs_epsg_hint=crs_epsg_hint,
        place_name=place_name,
        bbox_cfg=bbox_cfg,
        buffered_polygon=buffered_polygon,
    )
    logger.info("OSM enrichment: %s", enrich_stats)

    # --- Visual verification artifacts ---
    links_n = project.network.count_links()
    nodes_n = project.network.count_nodes()

    counts_path = out_dir / "network_counts.json"
    counts_path.write_text(
        json.dumps(
            {
                "links": int(links_n),
                "nodes": int(nodes_n),
                "place_name": place_name or "",
                "buffer_km": buffer_km,
                "osm_bbox_used_for_import_wgs84": bbox_cfg or None,
                "trim_bbox_wgs84_requested": list(bbox_wgs84_used) if bbox_wgs84_used is not None else None,
                "trim_bbox_native_used": {
                    "minx": bbox_native[0],
                    "miny": bbox_native[1],
                    "maxx": bbox_native[2],
                    "maxy": bbox_native[3],
                },
                "trim_stats": trim_stats,
                "drivable_network_stats": drivable_stats,
                "isolated_components_stats": isolated_stats,
                "enrich_stats": enrich_stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    links_df = project.network.links.data
    links_gdf_native = gpd.GeoDataFrame(links_df, geometry="geometry", crs=getattr(links_df, "crs", None))
    bbox_poly_native = box(*bbox_native)
    bbox_gdf_native = gpd.GeoDataFrame({"geometry": [bbox_poly_native]}, crs=links_gdf_native.crs)

    png_path_native = out_dir / "links_native.png"
    _save_network_links_map_png(
        links_gdf_native,
        bbox_gdf_native,
        png_path_native,
        title="AequilibraE links + reference bbox (native)",
        links_color=NETWORK_MAP_PALETTE["links_after"],
    )

    geojson_path_native = out_dir / "links_native.geojson"
    bbox_geojson_path_native = out_dir / "model_bbox_native.geojson"

    base_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    osm_attr_cols = ["speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "travel_time_ab", "travel_time_ba"]
    metadata_cols = ["link_type", "name", "osm_id", "osm_ref", "osm_ref_norm", "osm_name_raw", "osm_highway"]
    active_transport_cols = ["cycleway", "cycleway_left", "cycleway_right", "busway", "busway_left", "busway_right"]
    capacity_cols = ["capacity_ab", "capacity_ba"]

    desired_cols = base_cols + osm_attr_cols + metadata_cols + active_transport_cols + capacity_cols
    available_cols = [col for col in desired_cols if col in links_gdf_native.columns]

    links_gdf_native[available_cols].to_file(geojson_path_native, driver="GeoJSON")
    bbox_gdf_native.to_file(bbox_geojson_path_native, driver="GeoJSON")

    links_gdf_plot = links_gdf_native.copy()
    if links_gdf_plot.crs is None:
        links_gdf_plot = _as_gdf(links_gdf_plot, crs_epsg_hint)

    wgs_ok = False
    try:
        links_wgs84 = links_gdf_plot.to_crs(epsg=4326)
        bbox_wgs84 = bbox_gdf_native.copy()
        if bbox_wgs84.crs is None:
            bbox_wgs84 = bbox_wgs84.set_crs(links_gdf_plot.crs, allow_override=True)
        bbox_wgs84 = bbox_wgs84.to_crs(epsg=4326)
        wgs_ok = True
    except Exception:
        links_wgs84 = None
        bbox_wgs84 = None

    if wgs_ok and links_wgs84 is not None and bbox_wgs84 is not None:
        png_path_wgs84 = out_dir / "links_wgs84.png"
        _save_network_links_map_png(
            links_wgs84,
            bbox_wgs84,
            png_path_wgs84,
            title="AequilibraE links + bbox (WGS84, best-effort)",
            links_color=NETWORK_MAP_PALETTE["links_after"],
        )

        geojson_path_wgs84 = out_dir / "links_wgs84.geojson"
        bbox_geojson_path_wgs84 = out_dir / "model_bbox_wgs84.geojson"
        links_wgs84[available_cols].to_file(geojson_path_wgs84, driver="GeoJSON")
        bbox_wgs84.to_file(bbox_geojson_path_wgs84, driver="GeoJSON")
        wgs_files = [
            out_dir / "links_wgs84.png",
            out_dir / "links_wgs84.geojson",
            out_dir / "model_bbox_wgs84.geojson",
        ]
    else:
        wgs_files = []

    logger.info(
        "Network build done: project=%s nodes=%s links=%s bbox_native=%s",
        project_dir.resolve(),
        nodes_n,
        links_n,
        bbox_native,
    )
    if bbox_wgs84_used is not None:
        logger.info("Config reference bbox (WGS84): %s", bbox_wgs84_used)
    logger.info("Trim stats: %s", trim_stats)
    logger.info("Enrich stats: %s", enrich_stats)
    logger.info(
        "Wrote: %s, %s, %s, %s%s",
        counts_path,
        png_path_native,
        geojson_path_native,
        bbox_geojson_path_native,
        (", " + ", ".join(str(p) for p in wgs_files)) if wgs_files else "",
    )

    project.close()


if __name__ == "__main__":
    build_network_from_osm()
