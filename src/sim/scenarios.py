"""Scenario engine: apply network modifications and run assignment.

Supports two modification types per link:
- **full closure**: sets capacity to near-zero and travel time to a penalty
  value so BPR makes the link prohibitively expensive.
- **lane reduction**: scales capacity proportionally by
  ``lanes_remaining / lanes``.

Modifications are applied *in-memory* on the AequilibraE graph DataFrame so
the on-disk project and baseline results remain untouched.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix

from sim.assignment import build_graph, execute_assignment, fix_node_ids
from sim.io_project import load_config
from sim._metrics import aggregate_daily_volumes
from sim.scenario_state import JobStatus, transition_job

logger = logging.getLogger(__name__)

CLOSURE_CAPACITY = 0.001
CLOSURE_TRAVEL_TIME = 99_999.0

_SCENARIO_OUTPUT_DIR = Path("outputs/scenarios")


# ---------------------------------------------------------------------------
# Graph modification
# ---------------------------------------------------------------------------

def apply_scenario_to_graph(graph, scenario_links: List[Dict[str, Any]]) -> None:
    """Modify *graph* in-place for a list of scenario link changes.

    Each dict in *scenario_links* must contain:
      link_id        – int
      direction      – "both" | "ab" | "ba"
      closure_type   – "full" | "lanes"
      lanes          – int (original lane count)
      lanes_remaining – int (only used when closure_type == "lanes")
    """
    gdf = graph.graph

    time_col = "free_flow_time" if "free_flow_time" in gdf.columns else "travel_time"

    for sl in scenario_links:
        link_id = int(sl["link_id"])
        direction = sl.get("direction", "both")
        closure_type = sl.get("closure_type", "full")
        lanes_orig = max(int(sl.get("lanes", 1)), 1)
        lanes_remaining = max(int(sl.get("lanes_remaining", 1)), 1)

        mask = gdf["link_id"] == link_id

        if direction == "ab":
            mask = mask & (gdf["direction"] == 1)
        elif direction == "ba":
            mask = mask & (gdf["direction"] == -1)

        if not mask.any():
            continue

        if closure_type == "full":
            gdf.loc[mask, "capacity"] = CLOSURE_CAPACITY
            gdf.loc[mask, time_col] = CLOSURE_TRAVEL_TIME
        else:
            ratio = lanes_remaining / lanes_orig
            gdf.loc[mask, "capacity"] = gdf.loc[mask, "capacity"] * ratio


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

@dataclass
class ScenarioJob:
    id: str
    status: JobStatus = JobStatus.QUEUED
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    geojson_path: Optional[Path] = None
    scenario_links: List[Dict[str, Any]] = field(default_factory=list)
    _future: Optional[Future] = field(default=None, repr=False)

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)

    def to_status_dict(self) -> dict:
        # API contract: frontend expects "done" | "running" | "error" (not enum names).
        if self.status == JobStatus.SUCCEEDED:
            api_status = "done"
        elif self.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT):
            api_status = "error"
        elif self.status == JobStatus.RUNNING:
            api_status = "running"
        else:
            api_status = "running"  # QUEUED — keep polling / same UX as running
        return {
            "id": self.id,
            "status": api_status,
            "error": self.error,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed(),
        }

    def load_geojson(self) -> Optional[dict]:
        """Load result GeoJSON from disk (not kept in RAM)."""
        if self.geojson_path and self.geojson_path.exists():
            return json.loads(self.geojson_path.read_text(encoding="utf-8"))
        return None

    def _set_status(self, target: JobStatus) -> None:
        self.status = transition_job(self.status, target)


_jobs: Dict[str, ScenarioJob] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)

JOB_TTL_SECONDS = 3600
JOB_TIMEOUT_SECONDS = 1800


def _prune_old_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        expired = [
            jid for jid, j in _jobs.items()
            if j.finished_at and (now - j.finished_at) > JOB_TTL_SECONDS
        ]
        for jid in expired:
            job = _jobs.pop(jid)
            if job.geojson_path and job.geojson_path.exists():
                try:
                    job.geojson_path.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_scenario_worker(
    job: ScenarioJob,
    cfg: Dict[str, Any],
    baseline_links_gdf: gpd.GeoDataFrame,
) -> None:
    """Execute a scenario assignment in a background thread."""
    try:
        with _jobs_lock:
            job._set_status(JobStatus.RUNNING)

        project_dir = Path(cfg["project_path"])
        demand_cfg = cfg.get("demand") or {}
        calib_cfg = cfg.get("calibration") or {}
        assign_cfg = cfg.get("assignment") or {}
        matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
        core_name = str(calib_cfg.get("core_name", "wd_daily"))
        bpr_params = dict(assign_cfg.get("bpr") or {}) or None
        mc_cfg = assign_cfg.get("multi_class") or {}
        multi_classes = (
            list(mc_cfg["classes"])
            if mc_cfg.get("enabled") and "classes" in mc_cfg
            else None
        )

        algorithm = str(calib_cfg.get("algorithm", "bfw"))
        max_iter = int(assign_cfg.get("scenario_max_iter", calib_cfg.get("max_iter", 100)))
        rgap = float(assign_cfg.get("scenario_rgap", calib_cfg.get("rgap_target", 0.001)))

        fix_node_ids(project_dir)

        mat = AequilibraeMatrix()
        mat.load(str(matrix_path))
        mat.computational_view([core_name])

        project = Project()
        project.open(str(project_dir))
        try:
            graph = build_graph(project, mat, bpr_parameters=bpr_params)

            apply_scenario_to_graph(graph, job.scenario_links)

            gc_cfg = assign_cfg.get("generalized_cost") or {}
            gc_enabled = bool(gc_cfg.get("enabled", False))
            gc_field = (
                str(gc_cfg["fixed_cost_field"])
                if gc_enabled and "fixed_cost_field" in gc_cfg
                else None
            )
            gc_mult = float(gc_cfg.get("fixed_cost_multiplier", 0.0)) if gc_enabled else 0.0
            gc_vot = float(gc_cfg.get("vot", 1.0))

            df, _skims, _sl = execute_assignment(
                project,
                mat,
                algorithm=algorithm,
                max_iter=max_iter,
                rgap_target=rgap,
                bpr_parameters=bpr_params,
                multi_class=multi_classes,
                fixed_cost_field=gc_field,
                fixed_cost_multiplier=gc_mult,
                vot=gc_vot,
                graph=graph,
            )
        finally:
            project.close()
            mat.close()

        geojson = _build_scenario_geojson(baseline_links_gdf, df)

        _SCENARIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _SCENARIO_OUTPUT_DIR / f"{job.id}.geojson"
        out_path.write_text(json.dumps(geojson), encoding="utf-8")
        job.geojson_path = out_path

        with _jobs_lock:
            job._set_status(JobStatus.SUCCEEDED)

    except Exception:
        logger.exception("Scenario job %s failed", job.id)
        job.error = "Scenario computation failed. Check server logs for details."
        with _jobs_lock:
            job._set_status(JobStatus.FAILED)
    finally:
        job.finished_at = time.time()


def _build_scenario_geojson(
    baseline_gdf: gpd.GeoDataFrame,
    assignment_df: pd.DataFrame,
) -> dict:
    """Merge assignment results with baseline geometry, return GeoJSON dict.

    Preserves baseline volume/V-C columns and computes delta fields so the
    frontend can render a difference map.
    """
    gdf = baseline_gdf.copy()

    # Preserve baseline values before they get overwritten by the merge
    if "wd_daily_tot" in gdf.columns:
        gdf["baseline_vol"] = gdf["wd_daily_tot"].fillna(0)
    if "VOC_max" in gdf.columns:
        gdf["baseline_voc"] = gdf["VOC_max"].fillna(0)
    if "Congested_Time_Max" in gdf.columns:
        gdf["baseline_ct"] = gdf["Congested_Time_Max"].fillna(0)

    _baseline_cols = {"baseline_vol", "baseline_voc", "baseline_ct"}
    vol_cols = [c for c in assignment_df.columns if c != "link_id"]

    # Drop synthesized daily totals so aggregate_daily_volumes can recompute
    # them from the scenario's class-specific volumes after merge.
    _synth_daily = [c for c in ("wd_daily_tot", "wd_daily_ab", "wd_daily_ba") if c in gdf.columns]
    existing_overlap = [c for c in vol_cols if c in gdf.columns and c not in _baseline_cols]
    drop_cols = list(set(existing_overlap + _synth_daily))
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)

    gdf = gdf.merge(
        assignment_df[["link_id"] + vol_cols],
        on="link_id",
        how="left",
    )

    aggregate_daily_volumes(gdf)

    # --- Delta computation ---
    scen_vol = gdf["wd_daily_tot"].fillna(0) if "wd_daily_tot" in gdf.columns else 0
    base_vol = gdf["baseline_vol"].fillna(0)
    gdf["delta_vol"] = scen_vol - base_vol
    gdf["delta_pct"] = (gdf["delta_vol"] / base_vol.clip(lower=1)) * 100
    gdf["abs_delta_vol"] = gdf["delta_vol"].abs()

    if "VOC_max" in gdf.columns and "baseline_voc" in gdf.columns:
        gdf["delta_voc"] = gdf["VOC_max"].fillna(0) - gdf["baseline_voc"].fillna(0)

    if "Congested_Time_Max" in gdf.columns and "baseline_ct" in gdf.columns:
        gdf["delta_ct"] = gdf["Congested_Time_Max"].fillna(0) - gdf["baseline_ct"].fillna(0)

    keep = [
        "link_id", "link_type", "name", "osm_ref", "speed", "capacity", "lanes", "distance",
        "wd_daily_tot", "wd_daily_ab", "wd_daily_ba",
        "VOC_max", "VOC_AB", "VOC_BA",
        "peak_hour_vol_AB", "peak_hour_vol_BA", "K_factor", "LOS_max",
        "Congested_Time_Max", "Congested_Time_AB", "Congested_Time_BA",
        "Delay_factor_Max", "Delay_factor_AB", "Delay_factor_BA",
        # Delta fields
        "baseline_vol", "delta_vol", "delta_pct", "abs_delta_vol",
        "baseline_voc", "delta_voc",
        "baseline_ct", "delta_ct",
        "geometry",
    ]
    keep = [c for c in keep if c in gdf.columns]
    gdf = gdf[keep]

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return json.loads(gdf.to_json())


# ---------------------------------------------------------------------------
# Public helpers (called by api.py)
# ---------------------------------------------------------------------------

def submit_scenario(
    scenario_links: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    baseline_links_gdf: gpd.GeoDataFrame,
) -> ScenarioJob:
    """Create a new scenario job and submit it to the thread pool."""
    _prune_old_jobs()

    job = ScenarioJob(
        id=str(uuid.uuid4()),
        scenario_links=scenario_links,
    )
    with _jobs_lock:
        _jobs[job.id] = job
    future = _executor.submit(_run_scenario_worker, job, cfg, baseline_links_gdf)
    job._future = future
    return job


def get_job(job_id: str) -> Optional[ScenarioJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def cancel_job(job_id: str) -> bool:
    """Attempt to cancel a queued job. Returns True if cancelled."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status != JobStatus.QUEUED:
            return False
        if job._future and job._future.cancel():
            job._set_status(JobStatus.CANCELLED)
            job.finished_at = time.time()
            return True
    return False
