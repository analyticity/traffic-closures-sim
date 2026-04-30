"""FastAPI REST API for simulation results and scenario analysis."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from enum import Enum as PyEnum
from pydantic import BaseModel, Field, model_validator
from sim.defaults import SIM_DEFAULTS
from sim.io_project import get_metric_epsg, load_config, resolve_project_database_path
from sim._metrics import aggregate_daily_volumes

logger = logging.getLogger(__name__)

# --- Data layer (lazy-loaded singletons) ---

_cfg: Dict[str, Any] = {}
_links_gdf: Optional[gpd.GeoDataFrame] = None
_links_gdf_mtime: float = 0.0
_nodes_gdf: Optional[gpd.GeoDataFrame] = None
_nodes_gdf_mtime: float = 0.0


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


def _sanitize_nan(obj: Any) -> Any:
    """Replace NaN/Inf floats with None so JSON serialization succeeds."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(404, f"File not found: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _sanitize_nan(data)


def _map_center_lat_lng() -> Optional[tuple[float, float]]:
    """Centroid of model area (or centroids) in WGS84 for Leaflet ``[lat, lng]``."""
    zones_out = _out("zones")
    metric_epsg = int(get_metric_epsg(_cfg)) if _cfg else 5514

    def _centroid_from_gdf(gdf: gpd.GeoDataFrame) -> Optional[tuple[float, float]]:
        if gdf is None or gdf.empty:
            return None
        g = gdf.copy()
        if g.crs is None:
            g = g.set_crs(epsg=metric_epsg, allow_override=True)
        g = g.to_crs(epsg=4326)
        union = g.geometry.unary_union
        if union is None or union.is_empty:
            return None
        c = union.centroid
        return (float(c.y), float(c.x))

    for name in ("model_area.geojson", "centroids.geojson"):
        path = zones_out / name
        if not path.exists():
            continue
        try:
            gdf = gpd.read_file(path)
            out = _centroid_from_gdf(gdf)
            if out:
                return out
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not derive map center from %s: %s", path, exc)
    return None


def _model_meta_payload() -> Dict[str, Any]:
    """Labels and default map view for the UI (multi-city aware)."""
    meta_cfg = (_cfg.get("_meta") or {}) if _cfg else {}
    city_slug = str(meta_cfg.get("city_slug", "") or "").strip()
    osm = (_cfg.get("osm") or {}) if _cfg else {}
    place_name = str(osm.get("place_name", "") or "").strip()
    if place_name:
        title_short = place_name.split(",")[0].strip()
    elif city_slug:
        title_short = city_slug.replace("_", " ").replace("-", " ").title()
    else:
        title_short = "Model"

    center = _map_center_lat_lng()
    if center is None:
        fb = SIM_DEFAULTS["api"]["fallback_map_center"]
        center = (fb[0], fb[1])

    bc_cfg = (_cfg.get("baseline_closures") or {}) if _cfg else {}
    cache_dir = str(Path((_cfg or {}).get("datasets", {}).get("cache_dir", "data/cache")))
    closures_path = Path(bc_cfg.get("source_path", f"{cache_dir}/closures.parquet"))

    return {
        "place_name": place_name,
        "city_slug": city_slug,
        "title_short": title_short,
        "map_center": {"lat": center[0], "lng": center[1]},
        "features": {
            "has_closures": closures_path.exists(),
        },
    }


def invalidate_cache() -> None:
    """Drop cached data so the next request reloads from disk."""
    global _links_gdf, _links_gdf_mtime, _nodes_gdf, _nodes_gdf_mtime
    _links_gdf = None
    _links_gdf_mtime = 0.0
    _nodes_gdf = None
    _nodes_gdf_mtime = 0.0
    logger.info("API cache invalidated")


def _get_links() -> gpd.GeoDataFrame:
    """Load network links merged with assignment volumes (cached).

    The cache is invalidated automatically when the assignment results
    file on disk is newer than the cached version.
    """
    global _links_gdf, _links_gdf_mtime

    vol_path = _out("demand") / "assignment_results.parquet"
    current_mtime = vol_path.stat().st_mtime if vol_path.exists() else 0.0
    if _links_gdf is not None and current_mtime <= _links_gdf_mtime:
        return _links_gdf

    gpkg_path = _out("network") / "network_links.gpkg"
    net_path = _out("network") / "network_links.parquet"
    geojson_path = _out("network") / "network_links.geojson"
    if not gpkg_path.exists() and not net_path.exists() and not geojson_path.exists():
        net_dir = _out("network")
        raise HTTPException(
            404,
            f"network_links not found under {net_dir}. "
            "Run normalize-network for this model, and start the API with the same city config "
            "(e.g. `python run.py --config config/most/sim.yaml serve`). "
            "If you use `uvicorn sim.api:app`, set `SIM_CONFIG=config/most/sim.yaml` "
            "and run from the project root.",
        )

    if gpkg_path.exists():
        links = gpd.read_file(gpkg_path)
    elif geojson_path.exists():
        links = gpd.read_file(geojson_path)
    else:
        try:
            links = gpd.read_parquet(str(net_path))
        except (ValueError, Exception):
            links = gpd.GeoDataFrame(pd.read_parquet(str(net_path)))
        if links.crs is None:
            links = links.set_crs(epsg=4326, allow_override=True)

    # Derive scalar summary columns from directional network attributes.
    # For one-way links (direction=1) use the AB value only; BA columns
    # may contain synthetic defaults from normalization.
    has_dir = "direction" in links.columns
    for attr in ("speed", "capacity", "lanes"):
        ab, ba = f"{attr}_ab", f"{attr}_ba"
        if ab in links.columns and ba in links.columns and attr not in links.columns:
            if has_dir:
                links[attr] = np.where(
                    links["direction"] == 1,
                    links[ab],
                    links[[ab, ba]].max(axis=1),
                )
            else:
                links[attr] = links[[ab, ba]].max(axis=1)

    if vol_path.exists():
        vols = pd.read_parquet(str(vol_path))
        vol_cols = ["link_id"] + [c for c in vols.columns if c != "link_id"]
        links = links.merge(vols[vol_cols], on="link_id", how="left")

    aggregate_daily_volumes(links)

    if links.crs is None:
        links = links.set_crs(epsg=4326, allow_override=True)

    _links_gdf = links
    _links_gdf_mtime = current_mtime
    return _links_gdf


def _get_nodes() -> gpd.GeoDataFrame:
    global _nodes_gdf, _nodes_gdf_mtime

    path = _out("network") / "network_nodes.parquet"
    if not path.exists():
        raise HTTPException(404, "network_nodes.parquet not found.")
    current_mtime = path.stat().st_mtime
    if _nodes_gdf is not None and current_mtime <= _nodes_gdf_mtime:
        return _nodes_gdf

    _nodes_gdf = gpd.read_parquet(str(path))
    _nodes_gdf_mtime = current_mtime
    return _nodes_gdf


def _gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Convert GeoDataFrame to a dict suitable for JSONResponse."""
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return json.loads(gdf.to_json())


# --- FastAPI app ---

_api_title = "Traffic Simulation API"
_api_version = "0.2.0"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load ``sim.yaml`` when the app is started via ``uvicorn`` (not via ``run.py serve``).

    ``run.py … serve`` calls :func:`start_server`, which sets ``_cfg`` before uvicorn starts.
    If ``_cfg`` is still empty here, read ``SIM_CONFIG`` (default ``config/brno/sim.yaml``)
    so paths like ``network.output_dir`` resolve to the correct city.
    """
    global _cfg
    if _cfg:
        logger.info("API using config already set by start_server (%s)", _cfg.get("_meta", {}).get("config_path"))
    else:
        config_path = os.environ.get("SIM_CONFIG", "config/brno/sim.yaml")
        logger.info("API loading config from %s (override with SIM_CONFIG=…)", config_path)
        _cfg = load_config(config_path)
        osm = _cfg.get("osm") or {}
        place = str(osm.get("place_name", "") or "").strip()
        if place:
            short = place.split(",")[0].strip()
            app.title = f"{short} — API simulace dopravy"
    yield


app = FastAPI(
    title=_api_title,
    version=_api_version,
    description="API for simulation results and scenario what-if analysis.",
    lifespan=_lifespan,
)

_cors_origins = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Pydantic models for scenario API ---

class Direction(str, PyEnum):
    AB = "ab"
    BA = "ba"
    BOTH = "both"

class ClosureType(str, PyEnum):
    FULL = "full"
    LANES = "lanes"

class ScenarioLinkInput(BaseModel):
    link_id: int = Field(gt=0)
    direction: Direction = Direction.BOTH
    closure_type: ClosureType = ClosureType.FULL
    lanes: int = Field(ge=1, default=2)
    lanes_remaining: int = Field(ge=0, default=1)

    @model_validator(mode="after")
    def validate_lanes(self):
        if self.lanes_remaining > self.lanes:
            raise ValueError(
                f"lanes_remaining ({self.lanes_remaining}) cannot exceed "
                f"lanes ({self.lanes})"
            )
        return self

class ScenarioRunRequest(BaseModel):
    links: List[ScenarioLinkInput]

    @model_validator(mode="after")
    def deduplicate_links(self):
        seen = set()
        unique = []
        for link in self.links:
            key = (link.link_id, link.direction)
            if key in seen:
                continue
            seen.add(key)
            unique.append(link)
        self.links = unique
        return self


# --- Links & nodes ---

@app.get("/api/links")
def get_links(
    min_volume: float = Query(0, description="Minimum wd_daily_tot to include"),
    link_types: Optional[str] = Query(None, description="Comma-separated link types"),
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD) for temporal scaling"),
    period: str = Query("daily", description="Period: daily, day, evening, night"),
):
    gdf = _get_links().copy()

    # Apply temporal scaling if date is provided
    if date and "wd_daily_tot" in gdf.columns:
        try:
            from sim.demand.temporal import load_profile, get_combined_factor
            profile = load_profile(_cfg)
            factor = get_combined_factor(date, period, profile)
            for col in ("wd_daily_tot", "wd_daily_ab", "wd_daily_ba"):
                if col in gdf.columns:
                    gdf[col] = gdf[col].fillna(0) * factor
        except FileNotFoundError:
            pass

    if "wd_daily_tot" in gdf.columns and min_volume > 0:
        gdf = gdf[gdf["wd_daily_tot"].fillna(0) >= min_volume]

    if link_types:
        types = [t.strip() for t in link_types.split(",")]
        gdf = gdf[gdf["link_type"].isin(types)]

    keep = ["link_id", "link_type", "name", "osm_ref", "speed", "capacity", "lanes", "distance",
            "direction", "a_node", "b_node",
            "wd_daily_tot", "wd_daily_ab", "wd_daily_ba",
            "VOC_max", "VOC_AB", "VOC_BA",
            "peak_hour_vol_AB", "peak_hour_vol_BA", "K_factor", "LOS_max",
            "Congested_Time_Max", "Congested_Time_AB", "Congested_Time_BA",
            "Delay_factor_Max", "Delay_factor_AB", "Delay_factor_BA",
            "geometry"]
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


# --- Zones ---

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


@app.get("/api/meta")
def get_model_meta():
    """City name / slug and default map center from the loaded sim config and zones."""
    return JSONResponse(_model_meta_payload())


# --- Temporal ---

@app.get("/api/temporal/profile")
def temporal_profile():
    return JSONResponse(_read_json(_out("demand") / "temporal_profile.json"))


@app.get("/api/temporal/day-info")
def temporal_day_info(date: str = Query(..., description="Date YYYY-MM-DD")):
    try:
        from sim.demand.temporal import load_profile, day_info
        profile = load_profile(_cfg)
        return JSONResponse(day_info(date, profile))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


# --- Reports (JSON) ---

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


# --- Maps (PNG) ---

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


# --- Scenarios ---

@app.post("/api/scenarios/run")
def run_scenario(req: ScenarioRunRequest):
    from sim.scenarios import submit_scenario

    if not req.links:
        raise HTTPException(400, "No links provided for scenario.")

    baseline_gdf = _get_links()
    job = submit_scenario(
        scenario_links=[sl.model_dump() for sl in req.links],
        cfg=_cfg,
        baseline_links_gdf=baseline_gdf,
    )
    return JSONResponse(job.to_status_dict(), status_code=202)


@app.get("/api/scenarios/{scenario_id}/status")
def scenario_status(scenario_id: str):
    from sim.scenarios import get_job

    job = get_job(scenario_id)
    if job is None:
        raise HTTPException(404, f"Scenario job {scenario_id} not found.")
    return JSONResponse(job.to_status_dict())


@app.get("/api/scenarios/{scenario_id}/results")
def scenario_results(scenario_id: str):
    from sim.scenarios import get_job
    from sim.scenarios.state import JobStatus

    job = get_job(scenario_id)
    if job is None:
        raise HTTPException(404, f"Scenario job {scenario_id} not found.")
    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(409, "Scenario is still running.")
    if job.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT):
        raise HTTPException(500, f"Scenario failed: {job.error or job.status.name}")
    geojson = job.load_geojson()
    if geojson is None:
        raise HTTPException(500, "Scenario results not found on disk.")
    return JSONResponse(geojson)


@app.get("/api/scenarios/{scenario_id}/delta-summary")
def scenario_delta_summary(scenario_id: str):
    """Lightweight summary of delta (difference) between scenario and baseline."""
    from sim.scenarios import get_job
    from sim.scenarios.state import JobStatus

    job = get_job(scenario_id)
    if job is None:
        raise HTTPException(404, f"Scenario job {scenario_id} not found.")
    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(409, "Scenario is still running.")
    if job.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT):
        raise HTTPException(500, f"Scenario failed: {job.error or job.status.name}")

    geojson = job.load_geojson()
    if geojson is None:
        raise HTTPException(500, "Scenario results not found on disk.")
    features = geojson.get("features", [])
    affected = []
    for f in features:
        p = f.get("properties", {})
        dv = p.get("delta_vol")
        if dv is None:
            continue
        adv = abs(dv)
        if adv < 50:
            continue
        affected.append({
            "link_id": p.get("link_id"),
            "name": p.get("name", ""),
            "link_type": p.get("link_type", ""),
            "delta_vol": round(dv, 0),
            "delta_pct": round(p.get("delta_pct", 0), 1),
            "baseline_vol": round(p.get("baseline_vol", 0), 0),
            "scenario_vol": round(p.get("wd_daily_tot", 0), 0),
            "delta_voc": round(p.get("delta_voc", 0), 3),
        })

    affected.sort(key=lambda x: abs(x["delta_vol"]), reverse=True)
    top = affected[:20]

    abs_deltas = [abs(a["delta_vol"]) for a in affected]
    pct_deltas = [abs(a["delta_pct"]) for a in affected]
    increase = [a for a in affected if a["delta_vol"] > 0]
    decrease = [a for a in affected if a["delta_vol"] < 0]

    return JSONResponse({
        "total_links_affected": len(affected),
        "mean_abs_delta_vol": round(sum(abs_deltas) / max(len(abs_deltas), 1), 0),
        "mean_abs_delta_pct": round(sum(pct_deltas) / max(len(pct_deltas), 1), 1),
        "max_increase": max(increase, key=lambda x: x["delta_vol"]) if increase else None,
        "max_decrease": min(decrease, key=lambda x: x["delta_vol"]) if decrease else None,
        "top_affected": top,
    })


# --- Closures (date-based) ---


class ClosureScenarioRequest(BaseModel):
    date: str = Field(..., description="ISO date YYYY-MM-DD")


@app.get("/api/closures")
def get_closures(date: str = Query(..., description="ISO date YYYY-MM-DD")):
    """Return closures active on a given date, matched to network links."""
    from sim.scenarios.closures import closures_geojson_for_date

    try:
        link_gdf = _get_links()
    except HTTPException:
        link_gdf = None
    geojson = closures_geojson_for_date(date, _cfg, link_gdf)
    return JSONResponse(geojson)


@app.post("/api/closures/run-scenario")
def run_closure_scenario(req: ClosureScenarioRequest):
    """Create and run a scenario from all closures active on a date.

    Calls the standard scenario pipeline (full equilibrium assignment) so
    the resulting link volumes are valid.
    """
    from sim.scenarios.closures import closures_for_date
    from sim.scenarios import submit_scenario

    try:
        baseline_gdf = _get_links()
    except HTTPException:
        raise HTTPException(500, "Baseline network not available.")

    items = closures_for_date(req.date, _cfg, baseline_gdf)
    if not items:
        raise HTTPException(404, f"No closures found for date {req.date}.")

    scenario_links = [
        {
            "link_id": it["link_id"],
            "direction": it["direction"],
            "closure_type": it["closure_type"],
            "lanes": it["lanes"],
            "lanes_remaining": it["lanes_remaining"],
        }
        for it in items
    ]

    job = submit_scenario(
        scenario_links=scenario_links,
        cfg=_cfg,
        baseline_links_gdf=baseline_gdf,
    )
    return JSONResponse(job.to_status_dict(), status_code=202)


# --- Diagnostics ---

_MATCHING_DIAG_BIAS_COLS = ("link_id", "observed_car", "_corridor_volume")


def _diag_truthy(s: pd.Series) -> pd.Series:
    """Treat CSV booleans / 0-1 / strings as truth values."""
    if pd.api.types.is_bool_dtype(s):
        return s
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int) != 0
    return s.astype(str).str.lower().isin(("true", "1", "yes", "t"))


def _diag_falsy(s: pd.Series) -> pd.Series:
    """Opposite of truthy (for ``_excluded``: keep rows that are not excluded)."""
    if pd.api.types.is_bool_dtype(s):
        return ~s
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int) == 0
    return ~s.astype(str).str.lower().isin(("true", "1", "yes", "t"))


def _classify_station_status(row: pd.Series) -> str:
    """Return 'usable', 'excluded', or 'unmatched' for a diagnostics row."""
    if "_matched" in row.index:
        matched = row["_matched"]
        if isinstance(matched, (bool, np.bool_)):
            is_matched = bool(matched)
        elif pd.isna(matched):
            is_matched = False
        else:
            is_matched = str(matched).lower() in ("true", "1", "yes", "t")
        if not is_matched:
            return "unmatched"
    if "_excluded" in row.index:
        excluded = row["_excluded"]
        if isinstance(excluded, (bool, np.bool_)):
            is_excluded = bool(excluded)
        elif pd.isna(excluded):
            is_excluded = False
        else:
            is_excluded = str(excluded).lower() in ("true", "1", "yes", "t")
        if is_excluded:
            return "excluded"
    return "usable"


def _station_status_column(df: pd.DataFrame) -> pd.Series:
    """Vectorised version of _classify_station_status."""
    status = pd.Series("usable", index=df.index)
    if "_matched" in df.columns:
        status[~_diag_truthy(df["_matched"])] = "unmatched"
    if "_excluded" in df.columns:
        status[(status == "usable") & _diag_truthy(df["_excluded"])] = "excluded"
    return status


def _filter_matching_diag_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with matched link + observed + modeled corridor volume.

    ``matching_diagnostics.csv`` from a city without pentlogram matches (or
    before any link matched) may omit ``link_id`` / ``_corridor_volume``; return
    an empty frame instead of raising ``KeyError`` in callers.
    """
    if not all(c in df.columns for c in _MATCHING_DIAG_BIAS_COLS):
        return pd.DataFrame(columns=list(_MATCHING_DIAG_BIAS_COLS))

    out = df.copy()
    if "_matched" in out.columns:
        out = out[_diag_truthy(out["_matched"])]
    if "_excluded" in out.columns:
        out = out[_diag_falsy(out["_excluded"])]
    out = out.dropna(subset=list(_MATCHING_DIAG_BIAS_COLS))
    out["link_id"] = out["link_id"].astype(int)
    return out


def _bias_all_stations(raw_df: pd.DataFrame) -> JSONResponse:
    """Return *all* count stations with a ``status`` field (usable/excluded/unmatched)."""
    df = raw_df.copy()
    df["_status"] = _station_status_column(df)

    links = _get_links()
    link_geom = links[["link_id", "geometry"]].drop_duplicates("link_id")
    link_geom_map = dict(zip(link_geom["link_id"], link_geom["geometry"]))

    has_coords = "_count_lat" in df.columns and "_count_lng" in df.columns
    features = []

    for _, row in df.iterrows():
        status = str(row["_status"])
        lid = row.get("link_id")
        has_lid = lid is not None and not pd.isna(lid)

        # Determine point coordinates: prefer link midpoint, fall back to count coords.
        lng, lat = None, None
        if has_lid:
            lid_int = int(lid)
            geom = link_geom_map.get(lid_int)
            if geom is not None and not geom.is_empty:
                midpoint = geom.interpolate(0.5, normalized=True)
                lng, lat = midpoint.x, midpoint.y

        if lng is None and has_coords:
            _lat = row.get("_count_lat")
            _lng = row.get("_count_lng")
            if _lat is not None and not pd.isna(_lat) and _lng is not None and not pd.isna(_lng):
                lng, lat = float(_lng), float(_lat)

        if lng is None:
            continue

        observed = float(row.get("observed_car", 0) or 0)
        mod_val = row.get("_corridor_volume", None)
        modeled = float(mod_val) if mod_val is not None and not pd.isna(mod_val) else 0.0
        ratio = modeled / observed if observed > 0 else 0.0
        error = modeled - observed

        name_val = row.get("name", "")
        if pd.isna(name_val):
            name_val = ""
        lt_val = row.get("link_type", "")
        if pd.isna(lt_val):
            lt_val = ""
        geh_val = row.get("GEH", 0)
        if pd.isna(geh_val):
            geh_val = 0.0
        cn_val = row.get("_corridor_n_links", 1)
        if pd.isna(cn_val):
            cn_val = 1

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
            "properties": {
                "link_id": int(lid) if has_lid else None,
                "name": str(name_val),
                "link_type": str(lt_val),
                "observed": round(observed),
                "modeled": round(modeled),
                "ratio": round(ratio, 3),
                "error": round(error),
                "geh": round(float(geh_val), 1),
                "corridor_n_links": int(cn_val),
                "status": status,
            },
        })

    summary = {"usable": 0, "excluded": 0, "unmatched": 0}
    for f in features:
        s = f["properties"]["status"]
        if s in summary:
            summary[s] += 1

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features,
        "summary": summary,
    })


@app.get("/api/diagnostics/bias")
def diagnostics_bias(
    cluster: bool = Query(False, description="Aggregate nearby stations into spatial clusters"),
    eps: float = Query(100.0, description="Cluster radius in metres (DBSCAN eps)"),
    show_all: bool = Query(False, description="Include excluded and unmatched stations"),
):
    """Count-station bias map: observed vs modeled at each pentlogram station."""
    diag_path = _out("demand") / "matching_diagnostics.csv"
    if not diag_path.exists():
        raise HTTPException(404, "matching_diagnostics.csv not found. Run calibrate first.")

    raw_df = pd.read_csv(diag_path)

    if show_all and not cluster:
        return _bias_all_stations(raw_df)

    df = _filter_matching_diag_bias(raw_df)
    if df.empty:
        return JSONResponse({
            "type": "FeatureCollection",
            "features": [],
            "warning": "No link-matched count stations (missing link_id / _corridor_volume in "
            "matching_diagnostics.csv, or no rows passed filters). "
            "Bias map needs calibration with matched observations (CSD or pentlogram, linked to network).",
        })

    links = _get_links()
    link_geom = links[["link_id", "geometry"]].drop_duplicates("link_id")
    merged = df.merge(link_geom, on="link_id", how="inner")

    if not cluster:
        features = []
        for _, row in merged.iterrows():
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                continue
            midpoint = geom.interpolate(0.5, normalized=True)

            observed = float(row["observed_car"])
            modeled = float(row["_corridor_volume"])
            ratio = modeled / observed if observed > 0 else 0.0
            error = modeled - observed

            name_val = row.get("name", "")
            if pd.isna(name_val):
                name_val = ""
            lt_val = row.get("link_type", "")
            if pd.isna(lt_val):
                lt_val = ""
            geh_val = row.get("GEH", 0)
            if pd.isna(geh_val):
                geh_val = 0.0

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [midpoint.x, midpoint.y]},
                "properties": {
                    "link_id": int(row["link_id"]),
                    "name": str(name_val),
                    "link_type": str(lt_val),
                    "observed": round(observed),
                    "modeled": round(modeled),
                    "ratio": round(ratio, 3),
                    "error": round(error),
                    "geh": round(float(geh_val), 1),
                    "corridor_n_links": int(row.get("_corridor_n_links", 1)),
                    "status": "usable",
                },
            })

        return JSONResponse({
            "type": "FeatureCollection",
            "features": features,
        })

    # --- Clustered mode ---
    from sklearn.cluster import DBSCAN
    from sim._metrics import compute_geh as _geh

    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=links.crs)
    epsg = get_metric_epsg(_cfg)
    if merged_gdf.crs and merged_gdf.crs.to_epsg() != epsg:
        metric_gdf = merged_gdf.to_crs(epsg=epsg)
    else:
        metric_gdf = merged_gdf

    midpoints = metric_gdf.geometry.apply(lambda g: g.interpolate(0.5, normalized=True) if g and not g.is_empty else None)
    valid = midpoints.notna()
    metric_gdf = metric_gdf[valid].copy()
    midpoints = midpoints[valid]

    coords_arr = np.array([[p.x, p.y] for p in midpoints])
    labels = DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit_predict(coords_arr)
    metric_gdf["_cluster"] = labels

    if merged_gdf.crs and merged_gdf.crs.to_epsg() != 4326:
        wgs_gdf = merged_gdf[valid].copy()
        wgs_gdf = wgs_gdf.to_crs(epsg=4326)
    else:
        wgs_gdf = merged_gdf[valid].copy()
    wgs_gdf["_cluster"] = labels
    wgs_midpoints = wgs_gdf.geometry.apply(lambda g: g.interpolate(0.5, normalized=True) if g and not g.is_empty else None)

    features = []
    for cid in sorted(metric_gdf["_cluster"].unique()):
        mask = metric_gdf["_cluster"] == cid
        grp = metric_gdf[mask]
        wgs_grp_mid = wgs_midpoints[mask].dropna()

        sum_obs = float(grp["observed_car"].sum())
        sum_mod = float(grp["_corridor_volume"].sum())
        if sum_obs <= 0:
            continue

        ratio = sum_mod / sum_obs
        error = sum_mod - sum_obs
        geh_val = float(_geh(np.array([sum_mod]), np.array([sum_obs]))[0])

        cx = float(wgs_grp_mid.apply(lambda p: p.x).mean())
        cy = float(wgs_grp_mid.apply(lambda p: p.y).mean())

        names = sorted(set(
            str(n) for n in grp["name"].dropna().unique() if str(n).strip()
        ))
        link_ids = grp["link_id"].tolist()

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": {
                "cluster_id": int(cid),
                "n_stations": len(grp),
                "names": names,
                "link_ids": link_ids,
                "observed": round(sum_obs),
                "modeled": round(sum_mod),
                "ratio": round(ratio, 3),
                "error": round(error),
                "geh": round(geh_val, 1) if np.isfinite(geh_val) else 0.0,
            },
        })

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features,
        "clustered": True,
        "eps_m": eps,
        "n_clusters": len(features),
    })


@app.get("/api/diagnostics/through-traffic")
def diagnostics_through_traffic():
    """Through-traffic corridor map: links colored by external-through share."""
    vol_path = _out("demand") / "assignment_results.parquet"
    if not vol_path.exists():
        raise HTTPException(404, "assignment_results.parquet not found.")

    vols = pd.read_parquet(str(vol_path))
    if "wd_daily_external_through_tot" not in vols.columns:
        return JSONResponse({
            "links": {"type": "FeatureCollection", "features": []},
            "gateways": {"type": "FeatureCollection", "features": []},
            "screenlines": {"type": "FeatureCollection", "features": []},
            "warning": "No external-through volume column in assignment results.",
        })
    if "total_vehicles_tot" not in vols.columns:
        local_col = next((c for c in ("wd_daily_local_tot", "wd_daily_tot") if c in vols.columns), None)
        if local_col:
            vols["total_vehicles_tot"] = vols[local_col] + vols["wd_daily_external_through_tot"]
        else:
            vols["total_vehicles_tot"] = vols["wd_daily_external_through_tot"]
    vols = vols[["link_id", "wd_daily_external_through_tot", "total_vehicles_tot"]].copy()
    vols["through_share"] = np.where(
        vols["total_vehicles_tot"] > 0,
        vols["wd_daily_external_through_tot"] / vols["total_vehicles_tot"],
        0.0,
    )
    vols = vols[vols["through_share"] > 0.01]

    links_gdf = _get_links()
    link_geom = links_gdf[["link_id", "link_type", "geometry"]].drop_duplicates("link_id")
    merged = link_geom.merge(vols, on="link_id", how="inner")
    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=links_gdf.crs)

    if merged_gdf.crs and merged_gdf.crs.to_epsg() != 4326:
        merged_gdf = merged_gdf.to_crs(epsg=4326)

    link_features = []
    for _, row in merged_gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue
        link_features.append({
            "type": "Feature",
            "geometry": json.loads(gpd.GeoSeries([geom], crs=4326).to_json())["features"][0]["geometry"],
            "properties": {
                "link_id": int(row["link_id"]),
                "link_type": row.get("link_type", ""),
                "through_volume": round(float(row["wd_daily_external_through_tot"])),
                "total_volume": round(float(row["total_vehicles_tot"])),
                "through_share": round(float(row["through_share"]), 3),
            },
        })

    _sn_out = Path(_cfg.get("supernetwork", {}).get("output_dir", "outputs/baseline/supernetwork"))
    gw_path = _sn_out / "gateway_points.geojson"
    gateways_geojson = None
    if gw_path.exists():
        gw_gdf = gpd.read_file(gw_path)
        if gw_gdf.crs and gw_gdf.crs.to_epsg() != 4326:
            gw_gdf = gw_gdf.to_crs(epsg=4326)
        gateways_geojson = json.loads(gw_gdf[["gateway_name", "graph_node", "geometry"]].to_json())

    from sim.calibration.screenlines import load_screenlines, resolve_screenline_links

    _config_dir = _cfg.get("_meta", {}).get("base_dir", "")
    _default_sl = str(Path(_config_dir) / "screenlines.yaml") if _config_dir else "config/brno/screenlines.yaml"
    sl_path = str((_cfg.get("calibration") or {}).get("screenlines_path", _default_sl))
    screenline_defs = load_screenlines(sl_path)
    screenline_ids: List[int] = []
    for sl_def in screenline_defs:
        resolved = resolve_screenline_links(sl_def, links_gdf)
        screenline_ids.extend(lid for lid, _ in resolved)

    sl_features = []
    if screenline_ids:
        sl_gdf = links_gdf[links_gdf["link_id"].isin(screenline_ids)][["link_id", "name", "geometry"]].copy()
        if sl_gdf.crs and sl_gdf.crs.to_epsg() != 4326:
            sl_gdf = sl_gdf.to_crs(epsg=4326)
        sl_features = json.loads(sl_gdf.to_json())["features"]

    return JSONResponse({
        "links": {"type": "FeatureCollection", "features": link_features},
        "gateways": gateways_geojson or {"type": "FeatureCollection", "features": []},
        "screenlines": {"type": "FeatureCollection", "features": sl_features},
    })


@app.get("/api/diagnostics/corridors")
def diagnostics_corridors():
    """Corridor-level bias: aggregate observed/modeled by street name."""
    diag_path = _out("demand") / "matching_diagnostics.csv"
    if not diag_path.exists():
        raise HTTPException(404, "matching_diagnostics.csv not found. Run calibrate first.")

    df = _filter_matching_diag_bias(pd.read_csv(diag_path))
    if df.empty:
        return JSONResponse({
            "corridors": [],
            "warning": "No link-matched count stations — see matching_diagnostics.csv "
            "(needs link_id, _corridor_volume, matched CSD or pentlogram).",
        })

    if "name" not in df.columns:
        df = df.copy()
        df["name"] = ""
    else:
        df["name"] = df["name"].fillna("")

    links = _get_links()
    link_cols = ["link_id", "geometry"]
    if "osm_ref" in links.columns:
        link_cols.append("osm_ref")
    link_info = links[link_cols].drop_duplicates("link_id")
    merged = df.merge(link_info, on="link_id", how="inner")

    unnamed = merged["name"].str.strip() == ""
    if unnamed.any() and "osm_ref" in merged.columns:
        ref = merged.loc[unnamed, "osm_ref"].fillna("")
        lt = merged.loc[unnamed, "link_type"].fillna("road") if "link_type" in merged.columns else "road"
        merged.loc[unnamed, "name"] = lt.astype(str) + " " + ref.astype(str)
    merged = merged[merged["name"].str.strip() != ""]

    from sim._metrics import compute_geh as _geh
    from shapely.geometry import mapping

    corridors = []
    for cname, grp in merged.groupby("name"):
        sum_obs = float(grp["observed_car"].sum())
        sum_mod = float(grp["_corridor_volume"].sum())
        if sum_obs <= 0:
            continue
        ratio = sum_mod / sum_obs
        bias_pct = (sum_mod - sum_obs) / sum_obs * 100.0
        geh_val = float(_geh(np.array([sum_mod]), np.array([sum_obs]))[0])

        geoms = [g for g in grp["geometry"] if g is not None and not g.is_empty]
        if not geoms:
            continue

        merged_geom = geoms[0] if len(geoms) == 1 else gpd.GeoSeries(geoms, crs=links.crs).unary_union
        if links.crs and links.crs.to_epsg() != 4326:
            merged_geom = gpd.GeoSeries([merged_geom], crs=links.crs).to_crs(epsg=4326).iloc[0]

        lt_col = "link_type" if "link_type" in grp.columns else None
        link_types = sorted(grp[lt_col].dropna().unique().tolist()) if lt_col else []

        corridors.append({
            "name": str(cname),
            "n_stations": len(grp),
            "sum_observed": round(sum_obs),
            "sum_modeled": round(sum_mod),
            "ratio": round(ratio, 3),
            "geh": round(geh_val, 1) if np.isfinite(geh_val) else None,
            "bias_pct": round(bias_pct, 1),
            "link_types": link_types,
            "link_ids": grp["link_id"].tolist(),
            "geometry": mapping(merged_geom),
        })

    corridors.sort(key=lambda c: abs(c["bias_pct"]), reverse=True)
    return JSONResponse({"corridors": corridors})


# --- Intersection delay heuristic ---

_SIGNALIZED_TYPES = frozenset({
    "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link",
})
_MINOR_TYPES = frozenset({
    "tertiary", "tertiary_link", "unclassified", "road",
    "residential", "living_street", "service",
})


def _estimate_intersection_delays(G, path_nodes: list) -> float:
    """Heuristic per-node delay along a path (seconds).

    Only nodes that are real intersections (degree >= 3 considering both
    in- and out-edges with distinct neighbours) receive a delay.  Simple
    pass-through nodes (degree 2 = continuation on same road) get nothing.

    Delay per real intersection depends on the highest road class present:
      - signalized (primary/secondary/trunk junction): 20 s
      - minor (tertiary/residential): 6 s
      - motorway merge/diverge: 2 s
    """
    if len(path_nodes) < 3:
        return 0.0

    total_delay = 0.0
    for i in range(1, len(path_nodes) - 1):
        node = path_nodes[i]

        neighbours = set()
        for _, target in G.out_edges(node):
            neighbours.add(target)
        for source, _ in G.in_edges(node):
            neighbours.add(source)

        if len(neighbours) <= 2:
            continue

        in_type = G[path_nodes[i - 1]][node].get("link_type", "")
        out_type = G[node][path_nodes[i + 1]].get("link_type", "")

        if in_type == "motorway" and out_type == "motorway":
            continue

        incident_types = {in_type, out_type}
        for _, _, d in G.edges(node, data=True):
            incident_types.add(d.get("link_type", ""))
        for _, _, d in G.in_edges(node, data=True):
            incident_types.add(d.get("link_type", ""))

        if incident_types & _SIGNALIZED_TYPES:
            total_delay += 20.0
        elif incident_types & _MINOR_TYPES:
            total_delay += 6.0
        else:
            total_delay += 2.0

    return total_delay


def _build_network_graph(links_gdf: gpd.GeoDataFrame):
    """Build a networkx DiGraph from network links (shared helper).

    Edge weights prefer BPR congested times from assignment results
    (``Congested_Time_AB/BA``) when available, falling back to free-flow
    ``travel_time_ab/ba``.  Both values are stored on every edge so that
    callers can report free-flow vs. congested separately.
    """
    import networkx as nx

    has_congested = (
        "Congested_Time_AB" in links_gdf.columns
        and "Congested_Time_BA" in links_gdf.columns
    )

    has_direction = "direction" in links_gdf.columns

    G = nx.DiGraph()
    for _, row in links_gdf.iterrows():
        if pd.isna(row.get("a_node")) or pd.isna(row.get("b_node")):
            continue
        a, b = int(row["a_node"]), int(row["b_node"])
        lid = int(row["link_id"])
        geom = row["geometry"]
        link_type = str(row.get("link_type", "") or "")

        # direction: 0 = two-way, 1 = a→b only
        direction = int(row.get("direction", 0)) if has_direction else 0
        allow_ab = direction in (0, 1)
        allow_ba = direction == 0

        ff_ab = float(row.get("travel_time_ab") or 1e6)
        ff_ba = float(row.get("travel_time_ba") or 1e6)

        if has_congested and allow_ab:
            ct_ab = row.get("Congested_Time_AB")
            tt_ab = float(ct_ab) if pd.notna(ct_ab) and float(ct_ab) > 0 else ff_ab
        else:
            tt_ab = ff_ab

        if has_congested and allow_ba:
            ct_ba = row.get("Congested_Time_BA")
            tt_ba = float(ct_ba) if pd.notna(ct_ba) and float(ct_ba) > 0 else ff_ba
        else:
            tt_ba = ff_ba

        through_vol = float(row.get("wd_daily_external_through_tot") or 0)
        total_vol = float(row.get("total_vehicles_tot") or 0)
        link_name = str(row.get("name") or "")
        dist = float(row.get("distance") or 0)

        if allow_ab and not np.isnan(tt_ab) and tt_ab < 1e5:
            G.add_edge(a, b, weight=tt_ab, free_flow_time=ff_ab,
                       link_id=lid, geom=geom, reverse_geom=False,
                       link_type=link_type,
                       through_vol=through_vol, total_vol=total_vol,
                       link_name=link_name, distance=dist)
        if allow_ba and not np.isnan(tt_ba) and tt_ba < 1e5:
            G.add_edge(b, a, weight=tt_ba, free_flow_time=ff_ba,
                       link_id=lid, geom=geom, reverse_geom=True,
                       link_type=link_type,
                       through_vol=through_vol, total_vol=total_vol,
                       link_name=link_name, distance=dist)
    return G


def _edge_coords(edge_data: dict, prev_end: tuple = None) -> list:
    """Return geometry coordinates oriented in the traversal direction.

    Uses the ``reverse_geom`` flag as primary hint, then verifies against
    *prev_end* (last point of the path so far) and flips if the other end
    is closer -- handles any edge cases in geometry digitisation order.
    """
    geom = edge_data.get("geom")
    if geom is None or geom.is_empty:
        return []
    c = list(geom.coords)
    if edge_data.get("reverse_geom"):
        c = c[::-1]
    if prev_end is not None and len(c) >= 2:
        d_fwd = (c[0][0] - prev_end[0]) ** 2 + (c[0][1] - prev_end[1]) ** 2
        d_rev = (c[-1][0] - prev_end[0]) ** 2 + (c[-1][1] - prev_end[1]) ** 2
        if d_rev < d_fwd:
            c = c[::-1]
    return c


@app.get("/api/diagnostics/corridor-diagnosis")
def diagnostics_corridor_diagnosis(
    name: str = Query(..., description="Corridor street name to diagnose"),
):
    """Diagnose why a corridor is under/over-estimated: compare via-corridor vs model-preferred path."""
    import networkx as nx

    links_gdf = _get_links()
    if links_gdf.crs and links_gdf.crs.to_epsg() != 4326:
        links_gdf = links_gdf.to_crs(epsg=4326)

    corridor_links = links_gdf[links_gdf["name"].fillna("") == name]
    if corridor_links.empty:
        raise HTTPException(404, f"No links found with name '{name}'")

    corridor_link_ids = set(corridor_links["link_id"].astype(int).tolist())

    corridor_nodes: set = set()
    for _, row in corridor_links.iterrows():
        corridor_nodes.add(int(row["a_node"]))
        corridor_nodes.add(int(row["b_node"]))

    G = _build_network_graph(links_gdf)
    graph_nodes = set(G.nodes())

    corridor_nodes_in_graph = corridor_nodes & graph_nodes
    if len(corridor_nodes_in_graph) < 2:
        raise HTTPException(400, f"Corridor '{name}' has fewer than 2 nodes in the graph")

    # Pick corridor endpoints: the two geographically most distant nodes.
    # This ensures we span the full corridor length, not just a short fragment.
    nodes_path = _out("network") / "network_nodes.geojson"
    node_coords_local: Dict[int, tuple] = {}
    if nodes_path.exists():
        _nodes_gdf = gpd.read_file(nodes_path)
        if _nodes_gdf.crs and _nodes_gdf.crs.to_epsg() != 4326:
            _nodes_gdf = _nodes_gdf.to_crs(epsg=4326)
        for _, _r in _nodes_gdf.iterrows():
            node_coords_local[int(_r["node_id"])] = (_r.geometry.x, _r.geometry.y)

    cn_list = [n for n in corridor_nodes_in_graph if n in node_coords_local]
    if len(cn_list) >= 2:
        best_dist, ep_a, ep_b = 0, cn_list[0], cn_list[1]
        for i, na in enumerate(cn_list):
            for nb in cn_list[i + 1:]:
                ca, cb = node_coords_local[na], node_coords_local[nb]
                d = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
                if d > best_dist:
                    best_dist, ep_a, ep_b = d, na, nb
    else:
        cn_sorted = sorted(corridor_nodes_in_graph)
        ep_a, ep_b = cn_sorted[0], cn_sorted[-1]

    # Diag CSV for corridor stats
    diag_path = _out("demand") / "matching_diagnostics.csv"
    corridor_stats = None
    if diag_path.exists():
        df = _filter_matching_diag_bias(pd.read_csv(diag_path))
        cdf = df[df["name"].fillna("") == name] if not df.empty and "name" in df.columns else pd.DataFrame()
        if not cdf.empty:
            s_obs = float(cdf["observed_car"].sum())
            s_mod = float(cdf["_corridor_volume"].sum())
            corridor_stats = {
                "sum_observed": round(s_obs),
                "sum_modeled": round(s_mod),
                "ratio": round(s_mod / max(s_obs, 1), 3),
                "n_stations": len(cdf),
            }

    result: Dict[str, Any] = {
        "corridor_name": name,
        "corridor_stats": corridor_stats,
    }

    # Corridor link features
    from shapely.geometry import mapping as _mapping
    corr_features = []
    for _, row in corridor_links.iterrows():
        g = row["geometry"]
        if g is None or g.is_empty:
            continue
        vol = float(row.get("total_vehicles_tot") or 0)
        corr_features.append({
            "type": "Feature",
            "geometry": _mapping(g),
            "properties": {"link_id": int(row["link_id"]), "total_volume": round(vol)},
        })
    result["corridor_links"] = {"type": "FeatureCollection", "features": corr_features}

    def _path_to_geojson(path_nodes, label):
        coords, link_ids = [], []
        total_tt, total_ff, total_dist = 0.0, 0.0, 0.0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            ed = G[u][v]
            total_tt += ed["weight"]
            total_ff += ed.get("free_flow_time", ed["weight"])
            total_dist += ed.get("distance", 0)
            link_ids.append(ed["link_id"])
            prev = coords[-1] if coords else None
            c = _edge_coords(ed, prev)
            if c:
                if coords:
                    coords.extend(c[1:])
                else:
                    coords.extend(c)
        intersection_delay = _estimate_intersection_delays(G, path_nodes)
        return {
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in coords]} if coords else None,
            "travel_time": round(total_tt + intersection_delay, 1),
            "free_flow_time": round(total_ff, 1),
            "link_time": round(total_tt, 1),
            "intersection_delay": round(intersection_delay, 1),
            "distance": round(total_dist, 1),
            "n_links": len(link_ids),
            "link_ids": link_ids,
            "label": label,
        }

    # Free-flow shortest path (unconstrained)
    try:
        free_path = nx.shortest_path(G, ep_a, ep_b, weight="weight")
        result["free_flow_path"] = _path_to_geojson(free_path, "Shortest path (free flow)")
    except nx.NetworkXNoPath:
        result["free_flow_path"] = None

    # Via-corridor path: heavy penalty on edges NOT in the corridor
    G_via = G.copy()
    for u, v, d in G_via.edges(data=True):
        if d["link_id"] not in corridor_link_ids:
            G_via[u][v]["weight"] = d["weight"] * 10
    try:
        via_path = nx.shortest_path(G_via, ep_a, ep_b, weight="weight")
        result["via_corridor_path"] = _path_to_geojson(via_path, f"Route via {name}")
    except nx.NetworkXNoPath:
        result["via_corridor_path"] = None

    # Generate human-readable diagnosis
    if result.get("free_flow_path") and result.get("via_corridor_path"):
        ff = result["free_flow_path"]
        vc = result["via_corridor_path"]
        tt_diff_pct = (vc["travel_time"] - ff["travel_time"]) / max(ff["travel_time"], 0.1) * 100
        dist_diff_pct = (vc["distance"] - ff["distance"]) / max(ff["distance"], 0.1) * 100

        # Find dominant street names on the free-flow path
        lid_to_edge: Dict[int, dict] = {}
        for u, v, d in G.edges(data=True):
            lid_to_edge[d["link_id"]] = d
        ff_names: Dict[str, float] = {}
        for lid in ff.get("link_ids", []):
            ed = lid_to_edge.get(lid)
            if ed:
                n = ed.get("link_name", "")
                if n:
                    ff_names[n] = ff_names.get(n, 0) + ed.get("distance", 0)
        top_streets = sorted(ff_names.items(), key=lambda x: x[1], reverse=True)[:3]
        alt_names = ", ".join(s[0] for s in top_streets if s[0] != name) or "other streets"

        if abs(tt_diff_pct) < 3:
            result["diagnosis"] = (
                f"Route via {name} has similar travel time to alternative ({alt_names}). "
                f"Under-estimation may be caused by OD matrix or capacity parameter errors."
            )
        else:
            result["diagnosis"] = (
                f"Model prefers {alt_names} (travel time {abs(tt_diff_pct):.0f}% "
                f"{'lower' if tt_diff_pct > 0 else 'higher'}, "
                f"distance {abs(dist_diff_pct):.0f}% "
                f"{'longer' if dist_diff_pct > 0 else 'shorter'})."
            )
        result["time_diff_pct"] = round(tt_diff_pct, 1)
        result["dist_diff_pct"] = round(dist_diff_pct, 1)
    else:
        result["diagnosis"] = "Cannot compare routes (path not found)."

    return JSONResponse(result)


@app.get("/api/diagnostics/zone-route")
def diagnostics_zone_route(
    origin: int = Query(..., description="Origin zone_id"),
    destination: int = Query(..., description="Destination zone_id"),
):
    """Compute shortest-path route between two zones, annotated with bias data."""
    import networkx as nx

    centroids_path = _out("zones") / "centroids.geojson"
    if not centroids_path.exists():
        raise HTTPException(404, "centroids.geojson not found. Run build-zones first.")
    centroids_gdf = gpd.read_file(centroids_path)

    def _zone_row(zone_id: int):
        row = centroids_gdf[centroids_gdf["zone_id"] == zone_id]
        if row.empty:
            return None
        return row.iloc[0]

    o_row = _zone_row(origin)
    d_row = _zone_row(destination)
    if o_row is None:
        raise HTTPException(404, f"Origin zone {origin} not found")
    if d_row is None:
        raise HTTPException(404, f"Destination zone {destination} not found")

    links_gdf = _get_links()
    if links_gdf.crs and links_gdf.crs.to_epsg() != 4326:
        links_gdf = links_gdf.to_crs(epsg=4326)

    G = _build_network_graph(links_gdf)
    graph_nodes = set(G.nodes())

    def _resolve_zone_node(row) -> Optional[int]:
        cnode = int(row["centroid_node_id"])
        if cnode in graph_nodes:
            return cnode
        db_path = resolve_project_database_path(_cfg)
        if not db_path.exists():
            return None
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT a_node, b_node FROM links WHERE a_node=? OR b_node=?",
            (cnode, cnode),
        )
        for a, b in cur.fetchall():
            other = b if a == cnode else a
            if other in graph_nodes:
                conn.close()
                return other
        conn.close()
        return None

    o_node = _resolve_zone_node(o_row)
    d_node = _resolve_zone_node(d_row)
    if o_node is None:
        raise HTTPException(400, f"Cannot resolve origin zone {origin} to graph node")
    if d_node is None:
        raise HTTPException(400, f"Cannot resolve destination zone {destination} to graph node")

    # Node coordinates for markers
    nodes_path = _out("network") / "network_nodes.geojson"
    node_coords: Dict[int, tuple] = {}
    if nodes_path.exists():
        nodes_gdf = gpd.read_file(nodes_path)
        if nodes_gdf.crs and nodes_gdf.crs.to_epsg() != 4326:
            nodes_gdf = nodes_gdf.to_crs(epsg=4326)
        for _, row in nodes_gdf.iterrows():
            node_coords[int(row["node_id"])] = (row.geometry.x, row.geometry.y)

    try:
        path_nodes = nx.shortest_path(G, o_node, d_node, weight="weight")
    except nx.NetworkXNoPath:
        raise HTTPException(400, f"No path between zones {origin} → {destination}")

    # Build path geometry and collect link info
    path_coords: list = []
    path_link_ids: list = []
    total_tt, total_ff, total_dist = 0.0, 0.0, 0.0
    street_dist: Dict[str, float] = {}
    for i in range(len(path_nodes) - 1):
        edge = G[path_nodes[i]][path_nodes[i + 1]]
        lid = edge["link_id"]
        path_link_ids.append(lid)
        total_tt += edge["weight"]
        total_ff += edge.get("free_flow_time", edge["weight"])
        total_dist += edge.get("distance", 0)
        sn = edge.get("link_name", "")
        if sn:
            street_dist[sn] = street_dist.get(sn, 0) + edge.get("distance", 0)
        prev = path_coords[-1] if path_coords else None
        c = _edge_coords(edge, prev)
        if c:
            if path_coords:
                path_coords.extend(c[1:])
            else:
                path_coords.extend(c)

    intersection_delay = _estimate_intersection_delays(G, path_nodes)

    top_streets = sorted(street_dist.items(), key=lambda x: x[1], reverse=True)[:8]

    # Bias data along the route
    diag_path = _out("demand") / "matching_diagnostics.csv"
    bias_on_route: list = []
    if diag_path.exists():
        df = _filter_matching_diag_bias(pd.read_csv(diag_path))
        route_diag = df[df["link_id"].isin(set(path_link_ids))] if not df.empty else pd.DataFrame()
        for _, r in route_diag.iterrows():
            obs = float(r["observed_car"])
            mod = float(r["_corridor_volume"])
            link_row = links_gdf[links_gdf["link_id"] == int(r["link_id"])]
            mid = None
            if not link_row.empty:
                g = link_row.iloc[0].geometry
                if g and not g.is_empty:
                    mp = g.interpolate(0.5, normalized=True)
                    mid = [mp.x, mp.y]
            bias_on_route.append({
                "link_id": int(r["link_id"]),
                "name": str(r.get("name", "") or ""),
                "observed": round(obs),
                "modeled": round(mod),
                "ratio": round(mod / max(obs, 1), 3),
                "coords": mid,
            })

    o_coord = node_coords.get(o_node)
    d_coord = node_coords.get(d_node)

    return JSONResponse({
        "origin": {
            "zone_id": origin,
            "name": str(o_row["name"]),
            "node": o_node,
            "coords": list(o_coord) if o_coord else None,
        },
        "destination": {
            "zone_id": destination,
            "name": str(d_row["name"]),
            "node": d_node,
            "coords": list(d_coord) if d_coord else None,
        },
        "path": {
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in path_coords]} if path_coords else None,
            "travel_time": round(total_tt + intersection_delay, 1),
            "free_flow_time": round(total_ff, 1),
            "link_time": round(total_tt, 1),
            "intersection_delay": round(intersection_delay, 1),
            "distance": round(total_dist, 1),
            "n_links": len(path_link_ids),
        },
        "streets": [{"name": s, "distance": round(d, 0)} for s, d in top_streets],
        "bias_stations": bias_on_route,
    })


@app.get("/api/diagnostics/zones-list")
def diagnostics_zones_list():
    """Quick zone list for route selectors."""
    centroids_path = _out("zones") / "centroids.geojson"
    if not centroids_path.exists():
        raise HTTPException(404, "centroids.geojson not found.")
    gdf = gpd.read_file(centroids_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    zones = []
    for _, row in gdf.iterrows():
        zones.append({
            "zone_id": int(row["zone_id"]),
            "name": str(row["name"]),
            "is_external": bool(row.get("is_external", False)) if pd.notna(row.get("is_external")) else False,
            "coords": [row.geometry.x, row.geometry.y],
        })
    zones.sort(key=lambda z: (z["is_external"], z["name"]))
    return JSONResponse({"zones": zones})


@app.get("/api/diagnostics/routes")
def diagnostics_routes():
    """Route comparison: free-flow shortest path vs high-volume corridor for key OD pairs."""
    import networkx as nx

    links_gdf = _get_links()
    if links_gdf.crs and links_gdf.crs.to_epsg() != 4326:
        links_gdf = links_gdf.to_crs(epsg=4326)

    G = _build_network_graph(links_gdf)

    nodes_path = _out("network") / "network_nodes.geojson"
    node_coords: Dict[int, tuple] = {}
    if nodes_path.exists():
        nodes_gdf = gpd.read_file(nodes_path)
        if nodes_gdf.crs and nodes_gdf.crs.to_epsg() != 4326:
            nodes_gdf = nodes_gdf.to_crs(epsg=4326)
        for _, row in nodes_gdf.iterrows():
            nid = int(row["node_id"])
            node_coords[nid] = (row.geometry.x, row.geometry.y)

    graph_nodes = set(G.nodes())

    # Gateway anchor nodes (the real network node the gateway snaps to).
    # The graph_node from supernetwork_summary is a virtual node not in the
    # exported geojson -- use anchor_node_id from gateway_points instead.
    gw_anchor: Dict[str, int] = {}
    _sn_out2 = Path(_cfg.get("supernetwork", {}).get("output_dir", "outputs/baseline/supernetwork"))
    gw_path = _sn_out2 / "gateway_points.geojson"
    if gw_path.exists():
        gw_gdf = gpd.read_file(gw_path)
        for _, row in gw_gdf.iterrows():
            aid = int(row["anchor_node_id"])
            if aid in graph_nodes:
                gw_anchor[row["gateway_name"]] = aid

    # Centroid connector neighbour: for virtual centroid nodes not in the
    # exported network, find the first real-network node they connect to
    # via the project database.
    def _resolve_centroid_node(centroid_node_id: int) -> Optional[int]:
        if centroid_node_id in graph_nodes:
            return centroid_node_id
        db_path = resolve_project_database_path(_cfg)
        if not db_path.exists():
            return None
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute(
            "SELECT a_node, b_node FROM links WHERE a_node=? OR b_node=?",
            (centroid_node_id, centroid_node_id),
        )
        for a, b in cur.fetchall():
            other = b if a == centroid_node_id else a
            if other in graph_nodes:
                conn.close()
                return other
        conn.close()
        return None

    diagnostic_routes = (_cfg.get("api", {}) or {}).get("diagnostic_routes") or []
    od_pairs = []
    for route in diagnostic_routes:
        o_gw = route.get("origin_gateway")
        d_gw = route.get("destination_gateway")
        o_zone = route.get("origin_zone_id")
        d_zone = route.get("destination_zone_id")
        origin = gw_anchor.get(o_gw) if o_gw else (_resolve_centroid_node(o_zone) if o_zone else None)
        destination = gw_anchor.get(d_gw) if d_gw else (_resolve_centroid_node(d_zone) if d_zone else None)
        od_pairs.append({"name": route.get("name", f"{o_gw or o_zone} → {d_gw or d_zone}"),
                         "origin": origin, "destination": destination})

    # Pre-compute top high-volume through-traffic links (top 200 by volume)
    all_through_edges = [
        (u, v, d) for u, v, d in G.edges(data=True) if d.get("through_vol", 0) > 100
    ]
    all_through_edges.sort(key=lambda x: x[2]["through_vol"], reverse=True)
    top_through_ids = {d["link_id"] for _, _, d in all_through_edges[:300]}

    results = []
    for pair in od_pairs:
        o, d = pair["origin"], pair["destination"]
        if o is None or d is None or o not in G or d not in G:
            continue

        try:
            path_nodes = nx.shortest_path(G, o, d, weight="weight")
        except nx.NetworkXNoPath:
            continue

        path_coords: list = []
        path_link_ids: list = []
        for i in range(len(path_nodes) - 1):
            edge = G[path_nodes[i]][path_nodes[i + 1]]
            path_link_ids.append(edge["link_id"])
            prev = path_coords[-1] if path_coords else None
            c = _edge_coords(edge, prev)
            if c:
                if path_coords:
                    path_coords.extend(c[1:])
                else:
                    path_coords.extend(c)

        path_id_set = set(path_link_ids)
        non_path_high_vol = []
        for _, _, edata in all_through_edges:
            lid = edata["link_id"]
            if lid in top_through_ids and lid not in path_id_set:
                geom = edata.get("geom")
                if geom and not geom.is_empty:
                    non_path_high_vol.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": [list(c) for c in geom.coords]},
                        "properties": {
                            "link_id": lid,
                            "through_volume": round(edata["through_vol"]),
                            "total_volume": round(edata["total_vol"]),
                        },
                    })
                    if len(non_path_high_vol) >= 100:
                        break

        o_coord = node_coords.get(o)
        d_coord = node_coords.get(d)

        results.append({
            "name": pair["name"],
            "origin": {"node": o, "coords": list(o_coord) if o_coord else None},
            "destination": {"node": d, "coords": list(d_coord) if d_coord else None},
            "free_flow_path": {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(c) for c in path_coords]} if path_coords else None,
                "properties": {"link_ids": path_link_ids, "n_links": len(path_link_ids)},
            },
            "high_volume_links": {
                "type": "FeatureCollection",
                "features": non_path_high_vol,
            },
        })

    return JSONResponse({"routes": results})


# --- Server start ---

def start_server(config_path: str = "config/brno/sim.yaml") -> None:
    """Load config and start uvicorn."""
    import uvicorn

    global _cfg
    _cfg = load_config(config_path)

    api_cfg = _cfg.get("api", {}) or {}
    host = str(api_cfg.get("host", "0.0.0.0"))
    port = int(api_cfg.get("port", 8000))

    osm = _cfg.get("osm") or {}
    place = str(osm.get("place_name", "") or "").strip()
    if place:
        short = place.split(",")[0].strip()
        app.title = f"{short} — API simulace dopravy"

    logger.info("Starting API server on %s:%s", host, port)
    logger.info("Docs: http://%s:%s/docs", host, port)
    logger.info("Loaded sim config: %s", _cfg.get("_meta", {}).get("config_path"))
    logger.info("Network link files expected under: %s", _out("network"))
    uvicorn.run(app, host=host, port=port)
