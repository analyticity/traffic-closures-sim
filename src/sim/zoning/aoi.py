"""Model area (AOI) construction and zone filtering."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

import geopandas as gpd
import numpy as np
from aequilibrae import Project
from shapely.affinity import translate
from shapely.geometry import box

from sim.zoning.geo import axis_aligned_square_from_points, force_to_target_crs
from sim.zoning.network import network_ref

logger = logging.getLogger(__name__)

_HIGHWAY_ONLY_TYPES = frozenset({
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link",
})


def build_model_area(
    project: Project,
    target_epsg: int,
    *,
    quantile: float = 0.0,
    pad_ratio: float = 0.02,
    extra_margin_m: float = 0.0,
    shift_x_m: float = 0.0,
    shift_y_m: float = 0.0,
) -> Any:
    """Compute the analysis-area polygon from the largest network component.

    The bbox is built in WGS-84 to avoid the rotation Krovak introduces,
    then reprojected back to *target_epsg*.
    """
    logger.info("Zoning: building model area (WGS84-aligned bbox of largest component)")

    links_gdf = network_ref(project, "links", target_epsg)
    nodes_gdf = network_ref(project, "nodes", target_epsg)

    if "a_node" not in links_gdf.columns or "b_node" not in links_gdf.columns:
        raise RuntimeError("Links are missing a_node/b_node columns")
    if "node_id" not in nodes_gdf.columns:
        raise RuntimeError("Nodes are missing node_id column")

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in zip(links_gdf["a_node"].astype(int).values, links_gdf["b_node"].astype(int).values):
        adj[a].append(b)
        adj[b].append(a)

    visited: set[int] = set()
    largest_comp: list[int] = []
    for start in adj.keys():
        if start in visited:
            continue
        qd: deque[int] = deque([start])
        comp: list[int] = []
        visited.add(start)
        while qd:
            u = qd.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    qd.append(v)
        if len(comp) > len(largest_comp):
            largest_comp = comp

    if not largest_comp:
        minx, miny, maxx, maxy = nodes_gdf.total_bounds
        aoi = box(minx, miny, maxx, maxy)
        logger.warning(
            "Could not compute components; using bbox of ALL nodes (axis-aligned fallback)"
        )
        return aoi

    comp_set = set(largest_comp)
    core_nodes = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(comp_set)].copy()

    if "link_type" in links_gdf.columns:
        local_links = links_gdf[
            ~links_gdf["link_type"].astype(str).str.strip().str.lower().isin(_HIGHWAY_ONLY_TYPES)
        ]
        node_has_local = set(local_links["a_node"].astype(int)) | set(local_links["b_node"].astype(int))
        urban_mask = core_nodes["node_id"].astype(int).isin(node_has_local)
        n_excluded = int((~urban_mask).sum())
        if urban_mask.sum() > 100:
            core_nodes = core_nodes[urban_mask].copy()
            logger.debug("AOI point cloud: excluded %d highway-only nodes", n_excluded)

    core_wgs84 = core_nodes.to_crs(epsg=4326)
    lons = core_wgs84.geometry.x.to_numpy(dtype=float)
    lats = core_wgs84.geometry.y.to_numpy(dtype=float)

    aoi_wgs84 = axis_aligned_square_from_points(
        lons,
        lats,
        quantile=float(quantile),
        pad_ratio=float(pad_ratio),
        make_square=False,
    )

    aoi_gdf = gpd.GeoDataFrame({"geometry": [aoi_wgs84]}, crs="EPSG:4326")
    aoi_gdf = aoi_gdf.to_crs(epsg=target_epsg)
    aoi = aoi_gdf.geometry.iloc[0]

    if float(extra_margin_m) > 0.0:
        aoi = aoi.buffer(float(extra_margin_m), join_style=2)

    if float(shift_x_m) != 0.0 or float(shift_y_m) != 0.0:
        aoi = translate(aoi, xoff=float(shift_x_m), yoff=float(shift_y_m))

    bminx, bminy, bmaxx, bmaxy = map(float, aoi.bounds)
    logger.info(
        "AOI WGS84-aligned (reprojected to EPSG:%d) core_nodes=%d q=%s pad=%s "
        "extra_margin_m=%s shift=(%s, %s): bounds(minx=%.2f, miny=%.2f, maxx=%.2f, maxy=%.2f)",
        target_epsg,
        len(core_nodes),
        float(quantile),
        float(pad_ratio),
        float(extra_margin_m),
        float(shift_x_m),
        float(shift_y_m),
        bminx,
        bminy,
        bmaxx,
        bmaxy,
    )

    return aoi


def filter_zones_centroid_in_bbox(
    zones: gpd.GeoDataFrame,
    model_area_bbox: Any,
    crs_epsg: int,
    *,
    min_intersection_share: float = 0.05,
) -> gpd.GeoDataFrame:
    """Keep zones whose representative point is inside *model_area_bbox*
    or that have at least *min_intersection_share* overlap."""
    if zones.empty:
        return zones

    z = zones.copy()
    if z.crs is None:
        z = z.set_crs(epsg=crs_epsg, allow_override=True)
    if z.crs.to_epsg() != crs_epsg:
        z = z.to_crs(epsg=crs_epsg)

    rep_inside = z.geometry.representative_point().within(model_area_bbox)

    inter = z.geometry.intersection(model_area_bbox)
    inter_area = inter.area
    zone_area = z.geometry.area.replace(0, np.nan)
    share = (inter_area / zone_area).fillna(0.0)

    keep = (share >= min_intersection_share) | rep_inside

    n_kept = int(keep.sum())
    n_share_only = int((keep & ~rep_inside).sum())
    logger.info(
        "Zoning: AOI filter kept=%d dropped=%d (by intersection share: %d, min_share=%s)",
        n_kept,
        len(z) - n_kept,
        n_share_only,
        min_intersection_share,
    )
    return z.loc[keep].reset_index(drop=True)
