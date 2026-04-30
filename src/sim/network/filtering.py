"""Network filtering: drivable-only, connected-component, and spatial trim."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Set

import geopandas as gpd
import networkx as nx
import pandas as pd
from aequilibrae import Project
from shapely.geometry import box

from sim.network.crs import _CORRIDOR_LINK_TYPES
from sim.network.db import bulk_delete_by_ids, project_db, refresh_network

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_HIGHWAY_LINK_TYPES = frozenset({
    "footway", "path", "pedestrian", "track", "steps",
    "cycleway", "bridleway", "elevator", "escalator",
    "corridor", "platform", "proposed", "crossing",
})


# ---------------------------------------------------------------------------
# Orphan-node cleanup
# ---------------------------------------------------------------------------

def prune_orphan_nodes(project: Project, project_dir: Path) -> int:
    """Delete nodes not referenced by any link, then refresh the project cache."""
    with project_db(project_dir) as conn:
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

    refresh_network(project)
    return deleted


# ---------------------------------------------------------------------------
# Drivable-network filter
# ---------------------------------------------------------------------------

def remove_non_drivable_links(
    project: Project,
    project_dir: Path,
    *,
    excluded_link_types: Set[str],
    require_mode_car: bool = True,
) -> Dict[str, int]:
    """Drop pedestrian / cycle-only OSM ways and optionally links without car mode ``c``."""
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

    with project_db(project_dir) as conn:
        bulk_delete_by_ids(conn, "links", "link_id", ids_to_remove)

    refresh_network(project)
    pruned = prune_orphan_nodes(project, project_dir)
    return {"links_removed": len(ids_to_remove), "nodes_pruned": pruned}


# ---------------------------------------------------------------------------
# Connected-component filter
# ---------------------------------------------------------------------------

def remove_disconnected_components_keep_largest(
    project: Project,
    project_dir: Path,
) -> Dict[str, Any]:
    """Keep only the largest undirected connected component (by edge count).

    Isolated nodes are cleaned up via :func:`prune_orphan_nodes`.
    """
    try:
        project.network.links.refresh()
    except Exception:
        pass

    links = project.network.links.data
    empty: Dict[str, Any] = {"links_removed": 0, "nodes_pruned": 0, "components": 0, "kept_links": 0}
    if links.empty or "link_id" not in links.columns:
        return empty
    if "a_node" not in links.columns or "b_node" not in links.columns:
        raise RuntimeError("Network links missing a_node/b_node")

    G = nx.MultiGraph()
    for _, row in links.iterrows():
        G.add_edge(int(row["a_node"]), int(row["b_node"]), key=int(row["link_id"]))

    n_links = len(links)
    if G.number_of_nodes() == 0:
        return {**empty, "kept_links": n_links}

    components = list(nx.connected_components(G))
    n_comp = len(components)
    if n_comp <= 1:
        return {"links_removed": 0, "nodes_pruned": 0, "components": n_comp, "kept_links": n_links}

    best_nodes = max(components, key=lambda ns: G.subgraph(ns).number_of_edges())
    kept_link_ids = {int(k) for _u, _v, k in G.subgraph(best_nodes).edges(keys=True)}
    all_link_ids = {int(x) for x in links["link_id"].tolist()}
    ids_to_remove = sorted(all_link_ids - kept_link_ids)
    if not ids_to_remove:
        return {"links_removed": 0, "nodes_pruned": 0, "components": n_comp, "kept_links": n_links}

    with project_db(project_dir) as conn:
        bulk_delete_by_ids(conn, "links", "link_id", ids_to_remove)

    refresh_network(project)
    pruned = prune_orphan_nodes(project, project_dir)
    return {
        "links_removed": len(ids_to_remove),
        "nodes_pruned": pruned,
        "components": n_comp,
        "kept_links": len(kept_link_ids),
    }


# ---------------------------------------------------------------------------
# Spatial trim
# ---------------------------------------------------------------------------

def trim_network_to_bbox_raw(
    project: Project,
    bbox: tuple[float, float, float, float],
    project_dir: Path,
) -> Dict[str, int]:
    """Trim **non-corridor** links to *bbox* (native CRS).

    Corridor links (motorway / trunk / primary) are always kept so highway
    corridors extend beyond the urban core to where gateways are placed.
    Orphan nodes are cleaned up afterwards.
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

    is_corridor = (
        links_gdf["link_type"].astype(str).str.strip().str.lower().isin(_CORRIDOR_LINK_TYPES)
        if "link_type" in links_gdf.columns
        else pd.Series(False, index=links_gdf.index)
    )

    keep_link = is_corridor | (a_in & b_in)
    kept_link_ids = set(links_gdf.loc[keep_link, "link_id"].astype(int))
    kept_links = links_gdf.loc[keep_link]
    referenced_node_ids = set(kept_links["a_node"].astype(int)) | set(kept_links["b_node"].astype(int))

    link_ids_to_remove = [int(x) for x in links_gdf["link_id"] if int(x) not in kept_link_ids]
    node_ids_to_remove = [int(x) for x in nodes_gdf["node_id"] if int(x) not in referenced_node_ids]

    links_deleted = 0
    for link_id in link_ids_to_remove:
        try:
            project.network.links.delete(link_id)
            links_deleted += 1
        except Exception:
            pass

    if node_ids_to_remove:
        with project_db(project_dir, timeout=30.0) as conn:
            placeholders = ",".join("?" * len(node_ids_to_remove))
            conn.execute(f"DELETE FROM nodes WHERE node_id IN ({placeholders})", node_ids_to_remove)

    refresh_network(project)
    return {"nodes_deleted": len(node_ids_to_remove), "links_deleted": links_deleted}
