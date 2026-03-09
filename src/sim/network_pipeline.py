from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from aequilibrae import Project
from shapely.geometry import box

from sim.io_project import load_config


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


# ----------------------------
# CRS helpers (ONLY for plotting/export)
# ----------------------------

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


# ----------------------------
# Deterministic trimming (NO CRS transforms)
# ----------------------------

def compute_bbox_from_links_raw(project: Project) -> Tuple[float, float, float, float]:
    """
    Returns bbox from link geometries exactly as stored in the DB (raw coordinates).
    NO CRS guessing, NO to_crs. This is what you want for 'make bbox match the network'.
    """
    links = project.network.links.data
    if "geometry" not in links.columns or len(links) == 0:
        raise RuntimeError("Network links missing geometry/empty")

    minx, miny, maxx, maxy = map(float, links.total_bounds)
    if not (minx < maxx and miny < maxy):
        raise RuntimeError(f"Invalid link bounds: {(minx, miny, maxx, maxy)}")
    return minx, miny, maxx, maxy


def trim_network_to_bbox_raw(
    project: Project,
    bbox: Tuple[float, float, float, float],
    project_dir: Path,
) -> Dict[str, int]:
    """
    HARD trim in RAW coordinate space (whatever the DB uses):
    - keep nodes within bbox (by node geometry point)
    - keep links where both endpoints are kept AND link geometry intersects bbox
    Uses links.delete(link_id) API and direct SQLite for nodes (.data has no setter).
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

    # RAW: no transforms
    inside_nodes = nodes_gdf.geometry.within(rect)
    kept_nodes = nodes_gdf.loc[inside_nodes].copy()
    kept_node_ids = set(kept_nodes["node_id"].astype(int).tolist())

    # Keep only links fully connected inside the kept node set
    a_inside = links_gdf["a_node"].astype(int).isin(kept_node_ids)
    b_inside = links_gdf["b_node"].astype(int).isin(kept_node_ids)
    geom_inside = links_gdf.geometry.intersects(rect)

    kept_links = links_gdf.loc[a_inside & b_inside & geom_inside].copy()
    kept_link_ids = set(kept_links["link_id"].astype(int).tolist())

    link_ids_to_remove = [int(x) for x in links_gdf["link_id"] if int(x) not in kept_link_ids]
    node_ids_to_remove = [int(x) for x in nodes_gdf["node_id"] if int(x) not in kept_node_ids]

    links_deleted = 0
    for link_id in link_ids_to_remove:
        try:
            project.network.links.delete(link_id)
            links_deleted += 1
        except Exception:
            pass

    # Nodes: API has no delete; use SQLite (project holds DB open, use short timeout)
    if node_ids_to_remove:
        db_files = list(project_dir.glob("*.sqlite")) + list(project_dir.glob("*.db")) + list(project_dir.glob("*.sqlite3"))
        db_path = db_files[0] if db_files else project_dir / "project_database.sqlite"
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


# ----------------------------
# Main
# ----------------------------

def build_network_from_osm(
    config_path: str | Path = "config/sim.yaml",
    outputs_dir: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    if outputs_dir is None:
        outputs_dir = cfg.get("network", {}).get("maps_dir", "outputs/baseline/maps")

    project_dir = Path(cfg["project_path"])

    crs_epsg_hint = int(cfg.get("crs_epsg", 5514))

    osm_cfg = cfg.get("osm", {}) or {}
    place_name: Optional[str] = osm_cfg.get("place_name")
    bbox_cfg = cfg.get("model_bbox") or osm_cfg.get("bbox")  # [west, south, east, north] in WGS84

    if not bbox_cfg and not place_name:
        raise ValueError("Missing config model_bbox or osm.place_name in config/sim.yaml")

    project = create_or_open_project(project_dir)

    links_before = project.network.count_links()
    nodes_before = project.network.count_nodes()

    # --- Build network (only if empty) ---
    if links_before == 0 or nodes_before == 0:
        # Priority: explicit bbox (OSM import area), else place_name.
        if bbox_cfg:
            west, south, east, north = bbox_cfg
            model_area = box(west, south, east, north)
            project.network.create_from_osm(model_area=model_area)
            print(f"Network created from model_bbox: {bbox_cfg}")
        else:
            project.network.create_from_osm(place_name=place_name)
            print(f"Network created from place_name: {place_name}")

    # --- Trim to config bbox (the area we actually want) ---
    if bbox_cfg:
        bbox_raw = tuple(float(x) for x in bbox_cfg)
    else:
        bbox_raw = compute_bbox_from_links_raw(project)
    print("Trim bbox:", bbox_raw)

    trim_stats = trim_network_to_bbox_raw(project, bbox_raw, project_dir)
    print("Trimmed network:", trim_stats)

    # --- Visual verification artifacts ---
    out_dir = Path(outputs_dir)
    _ensure_dir(out_dir)

    links_n = project.network.count_links()
    nodes_n = project.network.count_nodes()

    counts_path = out_dir / "network_counts.json"
    counts_path.write_text(
        json.dumps(
            {
                "links": int(links_n),
                "nodes": int(nodes_n),
                "place_name": place_name or "",
                "osm_bbox_used_for_import_wgs84": bbox_cfg or None,
                "raw_bbox_from_links_db_coords": {
                    "minx": bbox_raw[0],
                    "miny": bbox_raw[1],
                    "maxx": bbox_raw[2],
                    "maxy": bbox_raw[3],
                },
                "trim_stats": trim_stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Prepare GDFs for plotting/export
    links_df = project.network.links.data
    links_gdf_native = gpd.GeoDataFrame(links_df, geometry="geometry", crs=getattr(links_df, "crs", None))
    bbox_poly_native = box(*bbox_raw)
    bbox_gdf_native = gpd.GeoDataFrame({"geometry": [bbox_poly_native]}, crs=links_gdf_native.crs)

    # Plot in "native" coords (always consistent with bbox)
    fig, ax = plt.subplots(figsize=(10, 10))
    links_gdf_native.plot(ax=ax, linewidth=0.2, zorder=1)
    bbox_gdf_native.boundary.plot(ax=ax, linewidth=3.0, zorder=3)
    ax.set_title("AequilibraE links + RAW bbox (native DB coords)")
    ax.set_axis_off()
    png_path_native = out_dir / "links_native.png"
    fig.savefig(png_path_native, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also export GeoJSON (native coords)
    geojson_path_native = out_dir / "links_native.geojson"
    bbox_geojson_path_native = out_dir / "model_bbox_native.geojson"

    base_cols = ["link_id", "a_node", "b_node", "direction", "modes", "distance", "geometry"]
    osm_attr_cols = ["speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "travel_time_ab", "travel_time_ba"]
    metadata_cols = ["link_type", "name", "osm_id"]
    active_transport_cols = ["cycleway", "cycleway_left", "cycleway_right", "busway", "busway_left", "busway_right"]
    capacity_cols = ["capacity_ab", "capacity_ba"]

    desired_cols = base_cols + osm_attr_cols + metadata_cols + active_transport_cols + capacity_cols
    available_cols = [col for col in desired_cols if col in links_gdf_native.columns]

    links_gdf_native[available_cols].to_file(geojson_path_native, driver="GeoJSON")
    bbox_gdf_native.to_file(bbox_geojson_path_native, driver="GeoJSON")

    # Optional: WGS84 export if CRS can be guessed (for easy QGIS)
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
        fig, ax = plt.subplots(figsize=(10, 10))
        links_wgs84.plot(ax=ax, linewidth=0.2, zorder=1)
        bbox_wgs84.boundary.plot(ax=ax, linewidth=3.0, zorder=3)
        ax.set_title("AequilibraE links + bbox (WGS84, best-effort)")
        ax.set_axis_off()
        png_path_wgs84 = out_dir / "links_wgs84.png"
        fig.savefig(png_path_wgs84, dpi=200, bbox_inches="tight")
        plt.close(fig)

        geojson_path_wgs84 = out_dir / "links_wgs84.geojson"
        bbox_geojson_path_wgs84 = out_dir / "model_bbox_wgs84.geojson"
        links_wgs84[available_cols].to_file(geojson_path_wgs84, driver="GeoJSON")
        bbox_wgs84.to_file(bbox_geojson_path_wgs84, driver="GeoJSON")

    print("=== NETWORK BUILD DONE ===")
    print("Project:", project_dir.resolve())
    print("Nodes:", nodes_n, "Links:", links_n)
    print("RAW bbox (native coords):", bbox_raw)
    print("Trim stats:", trim_stats)
    print("Wrote:", counts_path)
    print("Wrote:", png_path_native)
    print("Wrote:", geojson_path_native)
    print("Wrote:", bbox_geojson_path_native)
    if wgs_ok:
        print("Wrote:", out_dir / "links_wgs84.png")
        print("Wrote:", out_dir / "links_wgs84.geojson")
        print("Wrote:", out_dir / "model_bbox_wgs84.geojson")

    project.close()


if __name__ == "__main__":
    build_network_from_osm()
