"""FastAPI REST API for reading simulation results (links, zones, reports, maps)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from sim.io_project import load_config

# ---------------------------------------------------------------------------
# Data layer (lazy-loaded singletons)
# ---------------------------------------------------------------------------

_cfg: Dict[str, Any] = {}
_links_gdf: Optional[gpd.GeoDataFrame] = None
_nodes_gdf: Optional[gpd.GeoDataFrame] = None


def _out(section: str) -> Path:
    """Resolve an output directory from config."""
    if section == "network":
        return Path(_cfg.get("network", {}).get("output_dir", "outputs/baseline/network"))
    if section == "maps":
        return Path(_cfg.get("network", {}).get("maps_dir", "outputs/baseline/maps"))
    if section == "zones":
        return Path(_cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))
    if section == "demand":
        return Path(_cfg.get("demand", {}).get("output_dir", "outputs/baseline/demand"))
    return Path(f"outputs/baseline/{section}")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(404, f"File not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_links() -> gpd.GeoDataFrame:
    """Load network links merged with assignment volumes (cached)."""
    global _links_gdf
    if _links_gdf is not None:
        return _links_gdf

    net_path = _out("network") / "network_links.parquet"
    geojson_path = _out("network") / "network_links.geojson"
    if not net_path.exists() and not geojson_path.exists():
        raise HTTPException(404, "network_links not found. Run normalize-network first.")

    try:
        links = gpd.read_parquet(str(net_path))
    except (ValueError, Exception):
        if geojson_path.exists():
            links = gpd.read_file(geojson_path)
        else:
            links = gpd.GeoDataFrame(pd.read_parquet(str(net_path)))
            if links.crs is None:
                links = links.set_crs(epsg=4326, allow_override=True)

    vol_path = _out("demand") / "assignment_results.parquet"
    if vol_path.exists():
        vols = pd.read_parquet(str(vol_path))
        vol_cols = ["link_id"] + [c for c in vols.columns if c != "link_id"]
        links = links.merge(vols[vol_cols], on="link_id", how="left")

    if links.crs is None:
        links = links.set_crs(epsg=4326, allow_override=True)

    _links_gdf = links
    return _links_gdf


def _get_nodes() -> gpd.GeoDataFrame:
    global _nodes_gdf
    if _nodes_gdf is not None:
        return _nodes_gdf

    path = _out("network") / "network_nodes.parquet"
    if not path.exists():
        raise HTTPException(404, "network_nodes.parquet not found.")
    _nodes_gdf = gpd.read_parquet(str(path))
    return _nodes_gdf


def _gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Convert GeoDataFrame to a dict suitable for JSONResponse."""
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return json.loads(gdf.to_json())


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Brno Traffic Simulation API",
    version="0.1.0",
    description="Read-only API for simulation results (links with volumes, zones, reports).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Links & nodes
# ---------------------------------------------------------------------------

@app.get("/api/links")
def get_links(
    min_volume: float = Query(0, description="Minimum wd_daily_tot to include"),
    link_types: Optional[str] = Query(None, description="Comma-separated link types (e.g. motorway,primary)"),
):
    gdf = _get_links().copy()

    if "wd_daily_tot" in gdf.columns and min_volume > 0:
        gdf = gdf[gdf["wd_daily_tot"].fillna(0) >= min_volume]

    if link_types:
        types = [t.strip() for t in link_types.split(",")]
        gdf = gdf[gdf["link_type"].isin(types)]

    keep = ["link_id", "link_type", "name", "speed", "capacity",
            "wd_daily_tot", "wd_daily_ab", "wd_daily_ba",
            "VOC_max", "Congested_Time_Max", "geometry"]
    keep = [c for c in keep if c in gdf.columns]
    return JSONResponse(_gdf_to_geojson(gdf[keep]))


@app.get("/api/links/{link_id}")
def get_link(link_id: int):
    gdf = _get_links()
    row = gdf[gdf["link_id"] == link_id]
    if row.empty:
        raise HTTPException(404, f"Link {link_id} not found")
    return JSONResponse(_gdf_to_geojson(row))


@app.get("/api/nodes")
def get_nodes():
    gdf = _get_nodes()
    return JSONResponse(_gdf_to_geojson(gdf))


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

@app.get("/api/zones")
def get_zones():
    path = _out("zones") / "zones.geojson"
    if not path.exists():
        raise HTTPException(404, "zones.geojson not found")
    gdf = gpd.read_file(path)
    return JSONResponse(_gdf_to_geojson(gdf))


@app.get("/api/centroids")
def get_centroids():
    path = _out("zones") / "centroids.geojson"
    if not path.exists():
        raise HTTPException(404, "centroids.geojson not found")
    gdf = gpd.read_file(path)
    return JSONResponse(_gdf_to_geojson(gdf))


@app.get("/api/model-area")
def get_model_area():
    path = _out("zones") / "model_area.geojson"
    if not path.exists():
        raise HTTPException(404, "model_area.geojson not found")
    gdf = gpd.read_file(path)
    return JSONResponse(_gdf_to_geojson(gdf))


# ---------------------------------------------------------------------------
# Reports (JSON)
# ---------------------------------------------------------------------------

@app.get("/api/reports/calibration")
def report_calibration():
    return JSONResponse(_read_json(_out("demand") / "calibration_report.json"))


@app.get("/api/reports/validation")
def report_validation():
    return JSONResponse(_read_json(_out("demand") / "validation_report.json"))


@app.get("/api/reports/od-summary")
def report_od_summary():
    return JSONResponse(_read_json(_out("demand") / "od_summary.json"))


@app.get("/api/reports/network")
def report_network():
    return JSONResponse(_read_json(_out("network") / "network_summary.json"))


# ---------------------------------------------------------------------------
# Maps (PNG)
# ---------------------------------------------------------------------------

@app.get("/api/maps/zones")
def map_zones():
    path = _out("zones") / "zones_map.png"
    if not path.exists():
        raise HTTPException(404, "zones_map.png not found")
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/maps/network")
def map_network():
    path = _out("maps") / "links_wgs84.png"
    if not path.exists():
        raise HTTPException(404, "links_wgs84.png not found")
    return FileResponse(str(path), media_type="image/png")


# ---------------------------------------------------------------------------
# Server start
# ---------------------------------------------------------------------------

def start_server(config_path: str = "config/sim.yaml") -> None:
    """Load config and start uvicorn."""
    import uvicorn

    global _cfg
    _cfg = load_config(config_path)

    api_cfg = _cfg.get("api", {}) or {}
    host = str(api_cfg.get("host", "0.0.0.0"))
    port = int(api_cfg.get("port", 8000))

    print(f"Starting API server on {host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)
