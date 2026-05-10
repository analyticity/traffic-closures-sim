"""Boundary-gateway discovery, whitelist matching, merging, and external-zone creation."""
from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae import Project
from shapely.affinity import translate
from shapely.geometry import Point, box
from shapely.ops import nearest_points

from sim.zoning.geo import (
    bearing_deg,
    circular_mean_ring_pos,
    normalize_vector,
    safe_line_midpoint,
)
from sim.zoning.network import (
    ROAD_CLASS_WEIGHT,
    eligible_road_nodes,
    merge_class_group,
    network_ref,
    pick_nodes_near_boundary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation helpers (road refs / gateway names)
# ---------------------------------------------------------------------------

def norm_text(value: Any) -> str:
    """Normalise a road reference for fuzzy matching (strip diacritics, spaces, punctuation)."""
    text = str(value or "").strip().upper()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for ch in (" ", "/", "\\", "-", "."):
        text = text.replace(ch, "")
    return text


def slug_token(value: Any) -> str:
    """Short slug for gateway naming; falls back to ``'GW'``."""
    return norm_text(value) or "GW"


def compass8(angle: float) -> str:
    """8-wind compass label from bearing angle (0 = N, 90 = E)."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((angle % 360.0) + 22.5) // 45.0) % 8
    return labels[idx]


# ---------------------------------------------------------------------------
# Whitelist resolution
# ---------------------------------------------------------------------------

def resolve_whitelist(ext_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse the ``external_gateways.whitelist`` config section into spec dicts."""
    raw = ext_cfg.get("whitelist") or []
    results: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            ref = str(item.get("ref", "")).strip()
            if not ref:
                continue
            entry: Dict[str, Any] = {
                "raw": ref,
                "norm": norm_text(ref),
                "slug": slug_token(ref),
            }
            if "anchor_latlon" in item:
                ll = item["anchor_latlon"]
                if isinstance(ll, (list, tuple)) and len(ll) == 2:
                    entry["anchor_latlon"] = (float(ll[0]), float(ll[1]))
            if "anchor_node_id" in item:
                entry["anchor_node_id"] = int(item["anchor_node_id"])
            results.append(entry)
        else:
            s = str(item).strip()
            if not s:
                continue
            results.append({
                "raw": s,
                "norm": norm_text(s),
                "slug": slug_token(s),
            })
    return results


# ---------------------------------------------------------------------------
# Ring-based clustering
# ---------------------------------------------------------------------------

def cluster_positions_on_ring(
    df: pd.DataFrame,
    pos_col: str,
    threshold_m: float,
    ring_length: float,
) -> List[pd.DataFrame]:
    """Cluster rows by their position on a closed ring (model-area boundary)."""
    if df.empty:
        return []

    s = df.sort_values(pos_col).copy()
    idxs = s.index.tolist()
    poss = s[pos_col].astype(float).tolist()

    clusters: List[List[Any]] = [[idxs[0]]]
    prev = poss[0]

    for idx, pos in zip(idxs[1:], poss[1:]):
        if (pos - prev) <= float(threshold_m):
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev = pos

    if len(clusters) > 1 and ring_length > 0:
        first_pos = float(s.loc[clusters[0][0], pos_col])
        last_pos = float(s.loc[clusters[-1][-1], pos_col])
        wrap_gap = ring_length - last_pos + first_pos
        if wrap_gap <= float(threshold_m):
            merged = clusters[-1] + clusters[0]
            clusters = [merged] + clusters[1:-1]

    return [df.loc[c].copy() for c in clusters]


# ---------------------------------------------------------------------------
# Automatic boundary-road discovery
# ---------------------------------------------------------------------------

def auto_discover_boundary_roads(
    project: Project,
    target_epsg: int,
    model_area: Any,
    existing_refs: set[str],
    *,
    boundary_buffer_m: float = 1000.0,
    min_link_types: Optional[List[str]] = None,
    min_lanes: int = 1,
) -> List[Dict[str, Any]]:
    """Find named roads crossing the model boundary not already in the whitelist."""
    if min_link_types is None:
        min_link_types = [
            "motorway", "motorway_link", "trunk", "trunk_link",
            "primary", "primary_link", "secondary", "secondary_link",
        ]

    allowed = {str(x).strip() for x in min_link_types}
    links_gdf = network_ref(project, "links", target_epsg).copy()
    if links_gdf.empty:
        return []

    links_gdf = links_gdf[
        links_gdf["modes"].astype(str).str.contains("c", na=False)
        & (links_gdf["link_type"].astype(str) != "centroid_connector")
    ].copy()

    boundary = model_area.boundary
    links_gdf["_dist_boundary"] = links_gdf.geometry.distance(boundary)
    near = links_gdf[links_gdf["_dist_boundary"] <= float(boundary_buffer_m)].copy()
    near = near[near["link_type"].astype(str).isin(allowed)].copy()

    if near.empty:
        return []

    has_ref = near[near["osm_ref"].notna() & (near["osm_ref"].astype(str).str.strip() != "")].copy()
    if has_ref.empty:
        return []

    existing_norms = {norm_text(r) for r in existing_refs}

    results: List[Dict[str, Any]] = []
    for ref_val, group in has_ref.groupby("osm_ref"):
        ref_str = str(ref_val).strip()
        if not ref_str:
            continue
        ref_norm = norm_text(ref_str)
        if ref_norm in existing_norms:
            continue
        parts = ref_str.split(";")
        if any(norm_text(p) in existing_norms for p in parts):
            continue

        best_type = group["link_type"].map(ROAD_CLASS_WEIGHT).max()
        max_lanes = 1
        for col in ("lanes_ab", "lanes_ba"):
            if col in group.columns:
                v = pd.to_numeric(group[col], errors="coerce").max()
                if pd.notna(v):
                    max_lanes = max(max_lanes, int(v))

        if max_lanes < min_lanes:
            continue

        results.append({
            "raw": ref_str,
            "norm": ref_norm,
            "slug": slug_token(ref_str),
            "_class_weight": float(best_type) if pd.notna(best_type) else 0.0,
            "_n_boundary_links": len(group),
        })

    results.sort(key=lambda r: (-r["_class_weight"], -r["_n_boundary_links"]))
    return results


# ---------------------------------------------------------------------------
# Gateway target-node selection (boundary + whitelist driven)
# ---------------------------------------------------------------------------

def select_gateway_target_nodes(
    project: Project,
    target_epsg: int,
    model_area: Any,
    whitelist_specs: List[Dict[str, Any]],
    *,
    nodes_per_gateway: int = 2,
    boundary_buffer_m: float = 1000.0,
    min_gateway_separation_m: float = 800.0,
    allowed_link_types: Optional[List[str]] = None,
    max_anchor_distance_m: float = 2000.0,
) -> Tuple[
    Dict[str, List[int]],
    Dict[str, Dict[str, Any]],
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Match whitelist road-refs to boundary links and select gateway target nodes.

    Returns ``(gateway_targets, gateway_meta, debug_corridors, debug_points)``.
    """
    def _token_variants(raw_token: str, norm_token: str) -> set[str]:
        vals = set()
        for v in (raw_token, norm_token):
            t = norm_text(v)
            if t:
                vals.add(t)
                if t.startswith("I") and t[1:].isdigit():
                    num = t[1:]
                    vals.add(num)
                    vals.add("D" + num)
                elif t.startswith("D") and t[1:].isdigit():
                    num = t[1:]
                    vals.add(num)
                    vals.add("I" + num)
        return vals

    def _match_ref_variants(row: pd.Series, variants: set[str]) -> int:
        for col in ("osm_ref_norm", "osm_ref", "ref"):
            if col not in row or pd.isna(row.get(col)):
                continue
            raw_val = str(row.get(col))
            parts = raw_val.split(";") if ";" in raw_val else [raw_val]
            for part in parts:
                rv = norm_text(part)
                if rv in variants:
                    return 1
                if rv.startswith(("I", "D")) and rv[1:].isdigit() and rv[1:] in variants:
                    return 1
        return 0

    road_nids, node_weight = eligible_road_nodes(project, use_directed_scc=False)
    scc_nids, _ = eligible_road_nodes(project, use_directed_scc=True)

    _empty_crs = f"EPSG:{target_epsg}"
    if not road_nids:
        return {}, {}, gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs), gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs)

    nodes_gdf = network_ref(project, "nodes", target_epsg)[["node_id", "geometry"]].copy()
    nodes_gdf = nodes_gdf[nodes_gdf["node_id"].astype(int).isin(road_nids)].copy()
    if nodes_gdf.empty:
        return {}, {}, gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs), gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs)

    node_geom: Dict[int, Point] = {
        int(r["node_id"]): r.geometry
        for _, r in nodes_gdf.iterrows()
    }

    center = model_area.representative_point()
    boundary = model_area.boundary
    ring_length = float(boundary.length)
    cx, cy = float(center.x), float(center.y)

    links_gdf = network_ref(project, "links", target_epsg).copy()
    if links_gdf.empty:
        return {}, {}, gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs), gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs)

    links_gdf = links_gdf[
        links_gdf["modes"].astype(str).str.contains("c", na=False)
        & (links_gdf["link_type"].astype(str) != "centroid_connector")
        & links_gdf["a_node"].astype(int).isin(road_nids)
        & links_gdf["b_node"].astype(int).isin(road_nids)
    ].copy()

    if allowed_link_types:
        allowed = {str(x).strip() for x in allowed_link_types}
        if allowed:
            whitelist_refs: set[str] = set()
            for s in whitelist_specs:
                for k in ("norm", "raw"):
                    v = str(s.get(k, "")).strip()
                    if v:
                        whitelist_refs.add(v.lower())
            ref_col = links_gdf.get("osm_ref", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
            ref_norm = links_gdf.get("osm_ref_norm", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
            is_whitelisted = ref_col.isin(whitelist_refs) | ref_norm.isin(whitelist_refs)
            is_allowed_type = links_gdf["link_type"].astype(str).isin(allowed)
            links_gdf = links_gdf[is_allowed_type | is_whitelisted].copy()

    if links_gdf.empty:
        return {}, {}, gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs), gpd.GeoDataFrame({"geometry": []}, crs=_empty_crs)

    def _nearest_boundary_point(geom: Any) -> Point:
        try:
            _, bp = nearest_points(geom, boundary)
            return bp
        except Exception:
            mid = safe_line_midpoint(geom)
            try:
                _, bp = nearest_points(mid, boundary)
                return bp
            except Exception:
                return mid

    links_gdf["_dist_boundary"] = links_gdf.geometry.distance(boundary)
    links_gdf["_boundary_pt"] = links_gdf.geometry.apply(_nearest_boundary_point)
    links_gdf["_boundary_pos"] = links_gdf["_boundary_pt"].apply(lambda p: float(boundary.project(p)))
    links_gdf["_boundary_angle"] = links_gdf["_boundary_pt"].apply(
        lambda p: bearing_deg(cx, cy, float(p.x), float(p.y))
    )
    links_gdf["_road_class_weight"] = links_gdf["link_type"].astype(str).map(ROAD_CLASS_WEIGHT).fillna(0.5)

    gateway_targets: Dict[str, List[int]] = {}
    gateway_meta: Dict[str, Dict[str, Any]] = {}
    used_names: Dict[str, int] = {}

    debug_corridor_parts: List[gpd.GeoDataFrame] = []
    debug_point_rows: List[Dict[str, Any]] = []

    for spec_priority, spec in enumerate(whitelist_specs):
        token_raw = spec["raw"]
        token_norm = spec["norm"]
        token_slug = spec["slug"]

        ref_variants = _token_variants(token_raw, token_norm)

        logger.info("Discover whitelist token %s", token_raw)
        logger.debug("ref variants: %s", sorted(ref_variants))

        cand = links_gdf.copy()
        cand["_ref_match"] = cand.apply(lambda r: _match_ref_variants(r, ref_variants), axis=1).astype(int)
        matched = cand[cand["_ref_match"] == 1].copy()

        if matched.empty:
            logger.warning("token %s: no ref match in whole network", token_raw)
            continue

        logger.debug("matched in whole network: %d", len(matched))

        matched["debug_token"] = token_raw
        matched["debug_token_slug"] = token_slug
        debug_corridor_parts.append(matched.copy())

        boundary_near = matched[matched["_dist_boundary"] <= float(boundary_buffer_m)].copy()
        if boundary_near.empty:
            boundary_near = matched.nsmallest(min(20, len(matched)), "_dist_boundary").copy()
            logger.warning(
                "token %s: no boundary-near matched links within %.0f m, using nearest %d",
                token_raw,
                boundary_buffer_m,
                len(boundary_near),
            )

        clusters = cluster_positions_on_ring(
            boundary_near,
            pos_col="_boundary_pos",
            threshold_m=float(min_gateway_separation_m),
            ring_length=ring_length,
        )
        logger.debug("boundary clusters: %d", len(clusters))

        manual_anchor_pt: Optional[Point] = None
        if "anchor_latlon" in spec:
            lat, lon = spec["anchor_latlon"]
            anchor_gs = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=target_epsg)
            manual_anchor_pt = anchor_gs.iloc[0]
            logger.debug(
                "manual anchor_latlon -> projected (%.1f, %.1f)",
                manual_anchor_pt.x,
                manual_anchor_pt.y,
            )

        manual_anchor_nid: Optional[int] = spec.get("anchor_node_id")
        if manual_anchor_nid is not None:
            logger.debug("manual anchor_node_id = %s", manual_anchor_nid)

        best_manual_cluster: Optional[int] = None
        if (manual_anchor_pt is not None or manual_anchor_nid is not None) and len(clusters) > 1:
            ref_pt = manual_anchor_pt
            if ref_pt is None and manual_anchor_nid is not None and manual_anchor_nid in node_geom:
                ref_pt = node_geom[manual_anchor_nid]
            if ref_pt is not None:
                best_dist = float("inf")
                for ci, cdf in enumerate(clusters):
                    cd = float(cdf.geometry.distance(ref_pt).min())
                    if cd < best_dist:
                        best_dist = cd
                        best_manual_cluster = ci

        for cl_i, cl in enumerate(clusters, start=1):
            local_nodes = sorted(
                {
                    int(x)
                    for x in pd.concat([cl["a_node"], cl["b_node"]], ignore_index=True).astype(int).tolist()
                    if int(x) in node_geom
                }
            )
            if not local_nodes:
                continue

            use_manual = (
                (manual_anchor_nid is not None or manual_anchor_pt is not None)
                and (best_manual_cluster is None or best_manual_cluster == cl_i - 1)
            )

            if use_manual and manual_anchor_nid is not None and manual_anchor_nid in node_geom:
                chosen = [manual_anchor_nid]
                extra = [n for n in local_nodes if n != manual_anchor_nid and n in node_geom]
                if extra:
                    extra.sort(key=lambda n: float(node_geom[n].distance(node_geom[manual_anchor_nid])))
                    for n in extra[: nodes_per_gateway - 1]:
                        chosen.append(n)
                logger.debug(
                    "cluster %d: using manual anchor_node_id %s, targets=%s",
                    cl_i,
                    manual_anchor_nid,
                    chosen,
                )
            elif use_manual and manual_anchor_pt is not None:
                ranked = sorted(local_nodes, key=lambda n: float(node_geom[n].distance(manual_anchor_pt)))
                chosen = ranked[: nodes_per_gateway]
                logger.debug(
                    "cluster %d: using manual anchor_latlon, nearest nodes=%s",
                    cl_i,
                    chosen,
                )
            else:
                boundary_local_nodes = [
                    nid for nid in local_nodes
                    if float(node_geom[nid].distance(boundary)) <= float(boundary_buffer_m) * 1.25
                ]
                target_pool = boundary_local_nodes if boundary_local_nodes else local_nodes

                chosen = pick_nodes_near_boundary(
                    target_pool,
                    node_geom,
                    boundary,
                    node_weight,
                    max_nodes=int(nodes_per_gateway),
                    min_node_sep_m=25.0,
                    scc_nodes=scc_nids,
                )

                if len(chosen) < nodes_per_gateway and len(target_pool) < len(local_nodes):
                    chosen = pick_nodes_near_boundary(
                        local_nodes,
                        node_geom,
                        boundary,
                        node_weight,
                        max_nodes=int(nodes_per_gateway),
                        min_node_sep_m=25.0,
                        scc_nodes=scc_nids,
                    )

            if not chosen:
                logger.warning(
                    "token %s cluster %d: no chosen boundary nodes",
                    token_raw,
                    cl_i,
                )
                continue

            non_scc = [n for n in chosen if n not in scc_nids]
            if non_scc:
                logger.warning(
                    "token %s cluster %d: %d of %d target node(s) outside directed SCC: %s",
                    token_raw,
                    cl_i,
                    len(non_scc),
                    len(chosen),
                    non_scc,
                )

            if len(chosen) < nodes_per_gateway and not use_manual:
                logger.warning(
                    "token %s cluster %d: only %d of %d target nodes found "
                    "(boundary_buffer=%.0f m). "
                    "Consider increasing boundary_buffer_m or adding manual gateway nodes.",
                    token_raw,
                    cl_i,
                    len(chosen),
                    nodes_per_gateway,
                    boundary_buffer_m,
                )

            anchor_id = int(chosen[0])
            anchor_geom = node_geom[anchor_id]

            cluster_boundary_pos = circular_mean_ring_pos(
                cl["_boundary_pos"].astype(float).tolist(),
                ring_length,
            )
            cluster_boundary_pt = boundary.interpolate(cluster_boundary_pos)

            ux, uy = normalize_vector(
                float(cluster_boundary_pt.x) - cx,
                float(cluster_boundary_pt.y) - cy,
            )
            cluster_angle = bearing_deg(cx, cy, float(cluster_boundary_pt.x), float(cluster_boundary_pt.y))
            compass_label = compass8(cluster_angle)

            gw_name = f"{token_slug}_{compass_label}"
            if gw_name in used_names:
                used_names[gw_name] += 1
                gw_name = f"{gw_name}_{used_names[gw_name]}"
            else:
                used_names[gw_name] = 1

            best_link = cl.sort_values(
                ["_road_class_weight", "_dist_boundary"],
                ascending=[False, True],
            ).iloc[0]

            matched_ref = str(
                best_link.get("osm_ref_norm", "")
                or best_link.get("osm_ref", "")
                or best_link.get("ref", "")
                or ""
            )
            matched_name = str(
                best_link.get("osm_name_raw", "")
                or best_link.get("name", "")
                or ""
            )

            dist_to_boundary = float(anchor_geom.distance(boundary))
            if dist_to_boundary > max_anchor_distance_m:
                logger.debug(
                    "%s: anchor %.0fm from boundary > max %.0fm",
                    gw_name,
                    dist_to_boundary,
                    max_anchor_distance_m,
                )
                continue

            best_link_type = str(best_link.get("link_type", "") or "")
            predominant_type = (
                cl["link_type"]
                .astype(str)
                .mode()
                .iloc[0]
                if "link_type" in cl.columns and not cl.empty
                else best_link_type
            )

            gateway_targets[gw_name] = chosen
            gateway_meta[gw_name] = {
                "gateway_name": gw_name,
                "whitelist_token": token_raw,
                "whitelist_priority": int(spec_priority),
                "cluster_index": cl_i,
                "anchor_node_id": int(anchor_id),
                "anchor_x": float(anchor_geom.x),
                "anchor_y": float(anchor_geom.y),
                "boundary_x": float(cluster_boundary_pt.x),
                "boundary_y": float(cluster_boundary_pt.y),
                "boundary_pos": float(cluster_boundary_pos),
                "outward_dx": float(ux),
                "outward_dy": float(uy),
                "target_node_ids": chosen,
                "matched_ref": matched_ref,
                "matched_name": matched_name,
                "link_type": best_link_type,
                "predominant_link_type": predominant_type,
                "auto_discovered": bool(spec.get("auto_discovered", False)),
                "boundary_angle": float(cluster_angle),
                "dist_boundary_m": dist_to_boundary,
                "merged_from": "",
            }

            logger.debug(
                "gateway %s: boundary=(%.1f, %.1f) anchor=(%.1f, %.1f) targets=%s "
                "matched_ref=%r matched_name=%r type=%s",
                gw_name,
                float(cluster_boundary_pt.x),
                float(cluster_boundary_pt.y),
                float(anchor_geom.x),
                float(anchor_geom.y),
                chosen,
                matched_ref,
                matched_name,
                gateway_meta[gw_name]["link_type"],
            )

            debug_point_rows.append({
                "kind": "anchor",
                "token": token_raw,
                "cluster": cl_i,
                "node_id": int(anchor_id),
                "geometry": anchor_geom,
            })
            for nid in chosen:
                debug_point_rows.append({
                    "kind": "target",
                    "token": token_raw,
                    "cluster": cl_i,
                    "node_id": int(nid),
                    "geometry": node_geom[int(nid)],
                })

    n_before_merge = len(gateway_meta)
    gateway_targets, gateway_meta = _merge_gateway_candidates_on_boundary(
        gateway_targets=gateway_targets,
        gateway_meta=gateway_meta,
        node_geom=node_geom,
        node_weight=node_weight,
        model_area=model_area,
        merge_distance_m=float(min_gateway_separation_m),
        nodes_per_gateway=int(nodes_per_gateway),
        scc_nodes=scc_nids,
    )
    n_merged = n_before_merge - len(gateway_meta)
    if n_merged > 0:
        logger.info(
            "cross-token dedup: merged %d co-located gateway(s) (%d -> %d)",
            n_merged,
            n_before_merge,
            len(gateway_meta),
        )

    _ntl: Dict[int, List[int]] = {}
    if "link_id" in links_gdf.columns and "a_node" in links_gdf.columns:
        for _, lrow in links_gdf.iterrows():
            lid = int(lrow["link_id"])
            for nid in (int(lrow["a_node"]), int(lrow["b_node"])):
                _ntl.setdefault(nid, []).append(lid)

    n_before_target_dedup = len(gateway_meta)
    gateway_targets, gateway_meta = _merge_gateways_by_shared_targets(
        gateway_targets=gateway_targets,
        gateway_meta=gateway_meta,
        node_geom=node_geom,
        node_weight=node_weight,
        model_area=model_area,
        nodes_per_gateway=int(nodes_per_gateway),
        scc_nodes=scc_nids,
        node_to_links=_ntl if _ntl else None,
    )
    n_target_merged = n_before_target_dedup - len(gateway_meta)
    if n_target_merged > 0:
        logger.info(
            "target-node dedup: merged %d gateway(s) sharing target nodes/links (%d -> %d)",
            n_target_merged,
            n_before_target_dedup,
            len(gateway_meta),
        )

    debug_corridors_gdf = (
        gpd.GeoDataFrame(pd.concat(debug_corridor_parts, ignore_index=True), crs=links_gdf.crs)
        if debug_corridor_parts
        else gpd.GeoDataFrame({"geometry": []}, crs=links_gdf.crs)
    )

    debug_points_gdf = (
        gpd.GeoDataFrame(debug_point_rows, geometry="geometry", crs=nodes_gdf.crs)
        if debug_point_rows
        else gpd.GeoDataFrame({"geometry": []}, crs=nodes_gdf.crs)
    )

    return gateway_targets, gateway_meta, debug_corridors_gdf, debug_points_gdf


# ---------------------------------------------------------------------------
# Gateway merging (cross-token, boundary-proximity)
# ---------------------------------------------------------------------------

def _merge_gateway_candidates_on_boundary(
    gateway_targets: Dict[str, List[int]],
    gateway_meta: Dict[str, Dict[str, Any]],
    node_geom: Dict[int, Point],
    node_weight: Dict[int, float],
    model_area: Any,
    *,
    merge_distance_m: float,
    nodes_per_gateway: int,
    scc_nodes: Optional[set] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, Any]]]:
    if len(gateway_meta) <= 1:
        return gateway_targets, gateway_meta

    boundary = model_area.boundary
    ring_length = float(boundary.length)

    df = pd.DataFrame(
        [
            {
                "gateway_name": gw_name,
                "boundary_pos": float(meta.get("boundary_pos", 0.0)),
            }
            for gw_name, meta in gateway_meta.items()
        ]
    )

    clusters = cluster_positions_on_ring(
        df=df,
        pos_col="boundary_pos",
        threshold_m=float(merge_distance_m),
        ring_length=ring_length,
    )

    merged_targets: Dict[str, List[int]] = {}
    merged_meta: Dict[str, Dict[str, Any]] = {}

    for cl in clusters:
        names = cl["gateway_name"].tolist()
        metas = [gateway_meta[n] for n in names]

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for meta_item in metas:
            lt = str(meta_item.get("predominant_link_type", "") or meta_item.get("link_type", ""))
            grp = merge_class_group(lt)
            groups.setdefault(grp, []).append(meta_item)

        for _grp_key, grp_metas in groups.items():
            grp_names = [m["gateway_name"] for m in grp_metas]

            rep = sorted(
                grp_metas,
                key=lambda m: (
                    int(m.get("whitelist_priority", 9999)),
                    float(m.get("dist_boundary_m", 1e9)),
                    -float(ROAD_CLASS_WEIGHT.get(str(m.get("link_type", "")), 0.0)),
                ),
            )[0]

            cluster_boundary_pos = circular_mean_ring_pos(
                [float(m.get("boundary_pos", 0.0)) for m in grp_metas],
                ring_length,
            )
            cluster_boundary_pt = boundary.interpolate(cluster_boundary_pos)

            union_nodes: List[int] = []
            for m in grp_metas:
                for nid in m.get("target_node_ids", []):
                    nid = int(nid)
                    if nid in node_geom and nid not in union_nodes:
                        union_nodes.append(nid)

            chosen = pick_nodes_near_boundary(
                union_nodes,
                node_geom,
                boundary,
                node_weight,
                max_nodes=int(nodes_per_gateway),
                min_node_sep_m=25.0,
                scc_nodes=scc_nodes,
            )

            if not chosen:
                anchor_id = int(rep["anchor_node_id"])
                chosen = [anchor_id] if anchor_id in node_geom else []

            if not chosen:
                continue

            anchor_id = int(chosen[0])
            anchor_geom = node_geom[anchor_id]

            new_meta = dict(rep)
            new_meta["target_node_ids"] = chosen
            new_meta["anchor_node_id"] = int(anchor_id)
            new_meta["anchor_x"] = float(anchor_geom.x)
            new_meta["anchor_y"] = float(anchor_geom.y)
            new_meta["boundary_x"] = float(cluster_boundary_pt.x)
            new_meta["boundary_y"] = float(cluster_boundary_pt.y)
            new_meta["boundary_pos"] = float(cluster_boundary_pos)
            new_meta["merged_from"] = "|".join(grp_names)

            merged_targets[new_meta["gateway_name"]] = chosen
            merged_meta[new_meta["gateway_name"]] = new_meta

            if len(grp_names) > 1:
                logger.debug(
                    "merged boundary-near gateways %s -> %s (class group %s)",
                    grp_names,
                    new_meta["gateway_name"],
                    _grp_key,
                )

    return merged_targets, merged_meta


# ---------------------------------------------------------------------------
# Gateway merging (target-node overlap)
# ---------------------------------------------------------------------------

def _merge_gateways_by_shared_targets(
    gateway_targets: Dict[str, List[int]],
    gateway_meta: Dict[str, Dict[str, Any]],
    node_geom: Dict[int, Point],
    node_weight: Dict[int, float],
    model_area: Any,
    *,
    nodes_per_gateway: int,
    scc_nodes: Optional[set] = None,
    node_to_links: Optional[Dict[int, List[int]]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, Any]]]:
    """Merge gateways whose target nodes or adjacent links overlap.

    Multiple boundary clusters for the same road (e.g. 379_NE, 379_NE_2,
    379_NE_3) can be far apart on the boundary ring but still route through
    the same physical link.  Each gets its own OD demand share, inflating
    the total.  This pass uses union-find to group overlapping gateways
    and keeps a single representative.

    Two merge passes:
      1. Shared target node IDs (original logic).
      2. Shared adjacent link IDs -- gateways whose target nodes touch
         the same network link are merged even if the nodes differ.
    """
    if len(gateway_meta) <= 1:
        return gateway_targets, gateway_meta

    from collections import defaultdict

    node_to_gws: Dict[int, List[str]] = defaultdict(list)
    for gw_name, nodes in gateway_targets.items():
        for nid in nodes:
            node_to_gws[nid].append(gw_name)

    parent: Dict[str, str] = {n: n for n in gateway_targets}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    # Pass 1: shared target node IDs
    for _nid, gw_names in node_to_gws.items():
        for i in range(1, len(gw_names)):
            _union(gw_names[0], gw_names[i])

    # Pass 2: shared adjacent link IDs
    if node_to_links:
        link_to_gws: Dict[int, List[str]] = defaultdict(list)
        for gw_name, nodes in gateway_targets.items():
            for nid in nodes:
                for lid in node_to_links.get(nid, []):
                    link_to_gws[lid].append(gw_name)
        for lid, gw_names in link_to_gws.items():
            if len(gw_names) > 1:
                for i in range(1, len(gw_names)):
                    _union(gw_names[0], gw_names[i])

    groups: Dict[str, List[str]] = defaultdict(list)
    for name in gateway_targets:
        groups[_find(name)].append(name)

    merged_targets: Dict[str, List[int]] = {}
    merged_meta: Dict[str, Dict[str, Any]] = {}

    boundary = model_area.boundary

    for _root, group_names in groups.items():
        if len(group_names) == 1:
            name = group_names[0]
            merged_targets[name] = gateway_targets[name]
            merged_meta[name] = gateway_meta[name]
            continue

        group_metas = [gateway_meta[n] for n in group_names]
        rep = sorted(
            group_metas,
            key=lambda m: (
                int(m.get("whitelist_priority", 9999)),
                float(m.get("dist_boundary_m", 1e9)),
                -float(ROAD_CLASS_WEIGHT.get(str(m.get("link_type", "")), 0.0)),
            ),
        )[0]

        union_nodes: List[int] = []
        for name in group_names:
            for nid in gateway_targets[name]:
                if nid not in union_nodes and nid in node_geom:
                    union_nodes.append(nid)

        chosen = pick_nodes_near_boundary(
            union_nodes,
            node_geom,
            boundary,
            node_weight,
            max_nodes=int(nodes_per_gateway),
            min_node_sep_m=25.0,
            scc_nodes=scc_nodes,
        )

        if not chosen:
            chosen = gateway_targets[rep["gateway_name"]]

        rep_name = rep["gateway_name"]
        new_meta = dict(rep)
        new_meta["target_node_ids"] = chosen
        new_meta["anchor_node_id"] = int(chosen[0])
        if chosen[0] in node_geom:
            new_meta["anchor_x"] = float(node_geom[chosen[0]].x)
            new_meta["anchor_y"] = float(node_geom[chosen[0]].y)
        new_meta["merged_from"] = "|".join(group_names)

        merged_targets[rep_name] = chosen
        merged_meta[rep_name] = new_meta

        logger.info(
            "target-dedup: merged gateways %s -> %s (shared target nodes/links)",
            group_names, rep_name,
        )

    return merged_targets, merged_meta


# ---------------------------------------------------------------------------
# External gateway zone geometry
# ---------------------------------------------------------------------------

def build_external_gateway_zones(
    gateway_meta: Dict[str, Dict[str, Any]],
    target_epsg: int,
    *,
    zone_offset_m: float = 10.0,
    zone_size_m: float = 40.0,
    start_id: int = 8_000_000_000,
) -> gpd.GeoDataFrame:
    """Build synthetic polygons for external gateway zones.

    Zone centroid is placed just outside the AOI:
    ``centroid = boundary_point + outward * (half_size + zone_offset_m)``
    """
    if not gateway_meta:
        return gpd.GeoDataFrame({"geometry": []}, crs=f"EPSG:{target_epsg}")

    rows: List[Dict[str, Any]] = []
    half = float(zone_size_m) / 2.0

    for i, gw_name in enumerate(sorted(gateway_meta.keys())):
        meta = gateway_meta[gw_name]

        bx = float(meta["boundary_x"])
        by = float(meta["boundary_y"])
        ux = float(meta["outward_dx"])
        uy = float(meta["outward_dy"])

        centroid_dist = half + float(zone_offset_m)
        cx = bx + ux * centroid_dist
        cy = by + uy * centroid_dist

        geom = translate(
            box(-half, -half, half, half),
            xoff=cx,
            yoff=cy,
        )

        rows.append({
            "zone_id": start_id + i,
            "name": f"EXT_{gw_name}",
            "source_rank": 999,
            "is_external": 1,
            "gateway_name": gw_name,
            "anchor_node_id": int(meta["anchor_node_id"]),
            "anchor_x": float(meta["anchor_x"]),
            "anchor_y": float(meta["anchor_y"]),
            "boundary_x": bx,
            "boundary_y": by,
            "boundary_pos": float(meta.get("boundary_pos", 0.0)),
            "centroid_x": cx,
            "centroid_y": cy,
            "outward_dx": ux,
            "outward_dy": uy,
            "matched_ref": str(meta.get("matched_ref", "")),
            "matched_name": str(meta.get("matched_name", "")),
            "link_type": str(meta.get("link_type", "")),
            "whitelist_token": str(meta.get("whitelist_token", "")),
            "merged_from": str(meta.get("merged_from", "")),
            "geometry": geom,
        })

        logger.debug(
            "gateway zone %s: boundary=(%.1f, %.1f) centroid=(%.1f, %.1f) anchor_node=%d",
            gw_name,
            bx,
            by,
            cx,
            cy,
            int(meta["anchor_node_id"]),
        )

    return gpd.GeoDataFrame(rows, crs=f"EPSG:{target_epsg}")
