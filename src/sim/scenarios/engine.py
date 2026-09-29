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
from sim._metrics import aggregate_daily_volumes
from sim.scenarios.state import JobStatus, transition_job

logger = logging.getLogger(__name__)

CLOSURE_CAPACITY = 0.001
CLOSURE_TRAVEL_TIME = 99_999.0

_SCENARIO_OUTPUT_DIR = Path("outputs/scenarios")


# --- Graph modification ---

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

    if not scenario_links:
        return

    has_dir = "direction" in gdf.columns

    link_id_to_rows: dict = {}
    for i, lid in enumerate(gdf["link_id"].values):
        link_id_to_rows.setdefault(int(lid), []).append(i)

    for sl in scenario_links:
        link_id = int(sl["link_id"])
        direction = sl.get("direction", "both")
        closure_type = sl.get("closure_type", "full")
        lanes_orig = max(int(sl.get("lanes", 1)), 1)
        # Allow partial lane reduction (ratio < 1). Previously min(..., 1) on
        # lanes_remaining forced at least one lane open, so 10%/30%/50%
        # reductions on 2-lane links all collapsed to the same capacity.
        _rem = int(sl.get("lanes_remaining", lanes_orig))
        lanes_remaining = max(0, min(_rem, lanes_orig))

        row_indices = link_id_to_rows.get(link_id)
        if not row_indices:
            continue

        if has_dir and direction in ("ab", "ba"):
            dir_val = 1 if direction == "ab" else -1
            row_indices = [i for i in row_indices if gdf["direction"].iat[i] == dir_val]

        if not row_indices:
            continue

        if closure_type == "lanes" and lanes_remaining <= 0:
            closure_type = "full"

        if closure_type == "full":
            for i in row_indices:
                gdf.iat[i, gdf.columns.get_loc("capacity")] = CLOSURE_CAPACITY
                gdf.iat[i, gdf.columns.get_loc(time_col)] = CLOSURE_TRAVEL_TIME
        else:
            ratio = lanes_remaining / lanes_orig
            cap_col_idx = gdf.columns.get_loc("capacity")
            for i in row_indices:
                gdf.iat[i, cap_col_idx] = gdf.iat[i, cap_col_idx] * ratio


# --- Job management ---

@dataclass
class ScenarioJob:
    id: str
    status: JobStatus = JobStatus.QUEUED
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    geojson_path: Optional[Path] = None
    scenario_links: List[Dict[str, Any]] = field(default_factory=list)
    delta_summary: Dict[str, Any] = field(default_factory=dict)
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
        status = {
            "id": self.id,
            "status": api_status,
            "error": self.error,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed(),
        }
        if self.delta_summary:
            status["delta_summary"] = self.delta_summary
        return status

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

# Null-run baseline cache: one unmodified assignment per (project, matrix,
# assignment settings).  See `_null_baseline_results`.
_null_baseline_cache: Dict[tuple, pd.DataFrame] = {}
_null_baseline_lock = threading.Lock()


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


# --- Assignment execution ---

def _assignment_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve every knob that affects an assignment run, in one place.

    Both the scenario run and the null-run baseline read from this, so the two
    cannot silently drift apart -- which is the whole point of the null run.
    """
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    assign_cfg = cfg.get("assignment") or {}
    mc_cfg = assign_cfg.get("multi_class") or {}
    gc_cfg = assign_cfg.get("generalized_cost") or {}
    gc_enabled = bool(gc_cfg.get("enabled", False))

    return {
        "project_dir": Path(cfg["project_path"]),
        "matrix_path": Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem")),
        "core_name": str(calib_cfg.get("core_name", "wd_daily")),
        "bpr_params": dict(assign_cfg.get("bpr") or {}) or None,
        "assign_cfg": assign_cfg,
        "multi_classes": (
            list(mc_cfg["classes"])
            if mc_cfg.get("enabled") and "classes" in mc_cfg
            else None
        ),
        "algorithm": str(calib_cfg.get("algorithm", "bfw")),
        "cores": int(assign_cfg.get("cores", 0)),
        "max_iter": int(assign_cfg.get("scenario_max_iter", calib_cfg.get("max_iter", 100))),
        "rgap": float(assign_cfg.get("scenario_rgap", calib_cfg.get("rgap_target", 0.001))),
        "gc_field": (
            str(gc_cfg["fixed_cost_field"])
            if gc_enabled and "fixed_cost_field" in gc_cfg
            else None
        ),
        "gc_mult": float(gc_cfg.get("fixed_cost_multiplier", 0.0)) if gc_enabled else 0.0,
        "gc_vot": float(gc_cfg.get("vot", 1.0)),
    }


def run_assignment_with_scenario(
    cfg: Dict[str, Any],
    scenario_links: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Run one assignment with *scenario_links* applied in memory.

    Passing an empty list produces the **null run**: the untouched network,
    solved by exactly the same code path with exactly the same settings.
    """
    s = _assignment_settings(cfg)

    fix_node_ids(s["project_dir"])

    mat = AequilibraeMatrix()
    mat.load(str(s["matrix_path"]))
    mat.computational_view([s["core_name"]])

    project = Project()
    project.open(str(s["project_dir"]))
    try:
        graph = build_graph(
            project, mat,
            bpr_parameters=s["bpr_params"],
            assignment_cfg=s["assign_cfg"],
        )

        apply_scenario_to_graph(graph, scenario_links)

        df, _skims, _sl, _conv = execute_assignment(
            project,
            mat,
            algorithm=s["algorithm"],
            cores=s["cores"],
            max_iter=s["max_iter"],
            rgap_target=s["rgap"],
            bpr_parameters=s["bpr_params"],
            multi_class=s["multi_classes"],
            fixed_cost_field=s["gc_field"],
            fixed_cost_multiplier=s["gc_mult"],
            vot=s["gc_vot"],
            graph=graph,
        )
    finally:
        project.close()
        mat.close()

    return df


def _null_baseline_key(cfg: Dict[str, Any]) -> tuple:
    """Cache key: everything that would make the null run come out different."""
    s = _assignment_settings(cfg)

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    project_db = s["project_dir"] / "project_database.sqlite"
    return (
        str(s["project_dir"]), _mtime(project_db),
        str(s["matrix_path"]), _mtime(s["matrix_path"]),
        s["core_name"], s["algorithm"], s["max_iter"], s["rgap"], s["cores"],
        json.dumps(s["bpr_params"], sort_keys=True, default=str),
        json.dumps(s["multi_classes"], sort_keys=True, default=str),
        s["gc_field"], s["gc_mult"], s["gc_vot"],
    )


def null_baseline_results(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Assignment of the untouched network, computed once and cached.

    Why this exists: scenario deltas used to be taken against the stored
    ``assignment_results.parquet`` from the ``assign`` step.  That is a
    *different run* -- different iteration budget, possibly a different config
    revision -- so subtracting the two mixed the scenario effect with the gap
    between two independent approximations of the same equilibrium.  Measured
    on Brno v6: closing a single link carrying **zero** vehicles still moved
    753 links by >500 veh/day and shifted VHT by -1 424 veh*h, which is larger
    than the effect of closing a real arterial.

    Diffing against a null run computed here removes that asymmetry by
    construction.  What remains is genuine effect plus BFW path-dependence,
    which shrinks as ``assignment.scenario_rgap`` is tightened -- measure it
    with a placebo scenario (close a link with no traffic) rather than
    assuming it is negligible.
    """
    key = _null_baseline_key(cfg)
    with _null_baseline_lock:
        cached = _null_baseline_cache.get(key)
    if cached is not None:
        logger.info("Null-run baseline: using cached result")
        return cached

    logger.info("Null-run baseline: running unmodified assignment (cached afterwards)")
    started = time.time()
    df = run_assignment_with_scenario(cfg, [])
    logger.info("Null-run baseline: done in %.1f s", time.time() - started)

    with _null_baseline_lock:
        _null_baseline_cache[key] = df
    return df


def clear_null_baseline_cache() -> None:
    """Drop the cached null run (call after re-running ``assign``)."""
    with _null_baseline_lock:
        _null_baseline_cache.clear()


def _overlay_assignment(
    gdf: gpd.GeoDataFrame,
    assignment_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Put the volumes of *assignment_df* onto the geometry frame *gdf*."""
    vol_cols = [c for c in assignment_df.columns if c != "link_id"]
    synth = [c for c in ("wd_daily_tot", "wd_daily_ab", "wd_daily_ba") if c in gdf.columns]
    drop_cols = list({*(c for c in vol_cols if c in gdf.columns), *synth})

    out = gdf.drop(columns=drop_cols) if drop_cols else gdf.copy()
    out = out.merge(assignment_df[["link_id"] + vol_cols], on="link_id", how="left")
    aggregate_daily_volumes(out)
    return out


def _delta_summary(geojson: dict) -> Dict[str, float]:
    """Headline numbers for the scenario, for logs and the job status."""
    deltas = [
        float(f["properties"].get("delta_vol") or 0.0)
        for f in geojson.get("features", [])
    ]
    if not deltas:
        return {}
    abs_deltas = [abs(d) for d in deltas]
    return {
        "sum_abs_delta_vol": round(sum(abs_deltas), 1),
        "links_over_500": int(sum(1 for d in abs_deltas if d > 500)),
        "links_over_100": int(sum(1 for d in abs_deltas if d > 100)),
        "max_increase": round(max(deltas), 1),
        "max_decrease": round(min(deltas), 1),
    }


# --- Background worker ---

def _run_scenario_worker(
    job: ScenarioJob,
    cfg: Dict[str, Any],
    baseline_links_gdf: gpd.GeoDataFrame,
) -> None:
    """Execute a scenario assignment in a background thread."""
    try:
        with _jobs_lock:
            job._set_status(JobStatus.RUNNING)

        assign_cfg = cfg.get("assignment") or {}
        baseline_mode = str(assign_cfg.get("scenario_baseline", "null_run")).strip().lower()

        # The null run must be computed *before* the scenario so a failure here
        # is reported as a scenario failure rather than a silent fallback to
        # the stored (non-comparable) baseline.
        if baseline_mode == "null_run":
            null_df = null_baseline_results(cfg)
            baseline_for_delta = _overlay_assignment(baseline_links_gdf, null_df)
        elif baseline_mode == "stored":
            logger.warning(
                "scenario_baseline='stored': deltas are taken against "
                "assignment_results.parquet, i.e. against a different run. "
                "Expect a large noise floor -- see null_baseline_results()."
            )
            baseline_for_delta = baseline_links_gdf
        else:
            raise ValueError(
                f"assignment.scenario_baseline must be 'null_run' or 'stored', "
                f"got {baseline_mode!r}"
            )

        df = run_assignment_with_scenario(cfg, job.scenario_links)

        geojson = _build_scenario_geojson(baseline_for_delta, df)

        job.delta_summary = _delta_summary(geojson)
        job.delta_summary["baseline_mode"] = baseline_mode
        logger.info("Scenario job %s deltas: %s", job.id, job.delta_summary)

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
    _MIN_BASELINE_VOL = 200

    scen_vol = gdf["wd_daily_tot"].fillna(0) if "wd_daily_tot" in gdf.columns else 0
    base_vol = gdf["baseline_vol"].fillna(0)
    gdf["delta_vol"] = scen_vol - base_vol
    gdf["delta_pct"] = (gdf["delta_vol"] / base_vol.clip(lower=1)) * 100
    gdf["abs_delta_vol"] = gdf["delta_vol"].abs()

    if "VOC_max" in gdf.columns and "baseline_voc" in gdf.columns:
        gdf["delta_voc"] = gdf["VOC_max"].fillna(0) - gdf["baseline_voc"].fillna(0)

    if "Congested_Time_Max" in gdf.columns and "baseline_ct" in gdf.columns:
        gdf["delta_ct"] = gdf["Congested_Time_Max"].fillna(0) - gdf["baseline_ct"].fillna(0)

    max_vol = pd.concat([base_vol, scen_vol], axis=1).max(axis=1)
    gdf["low_volume_flag"] = (max_vol < _MIN_BASELINE_VOL).astype(int)

    network_mean_vol = base_vol[base_vol > 0].mean()
    if pd.isna(network_mean_vol) or network_mean_vol == 0:
        network_mean_vol = 1.0
    gdf["significance"] = (
        gdf["abs_delta_vol"] * (base_vol.clip(lower=1) / network_mean_vol)
    )

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
        "low_volume_flag", "significance",
        "geometry",
    ]
    keep = [c for c in keep if c in gdf.columns]
    gdf = gdf[keep]

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return json.loads(gdf.to_json())


# --- Public helpers ---

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


