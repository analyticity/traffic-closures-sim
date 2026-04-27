"""Traffic assignment: load OD matrix, build graph, run AequilibraE assignment.

Public API
----------
``run_assignment(config_path)``
    CLI entry-point – loads config, runs a single assignment, saves results.

``execute_assignment(project, mat, *, algorithm, …) -> pd.DataFrame``
    Low-level reusable function.  Takes an *open* AequilibraE Project and a
    *loaded* AequilibraeMatrix and returns a link-volume DataFrame.  Used by
    the iterative calibration loop so it can swap matrices between iterations
    without re-opening the project every time.

``build_graph(project, mat, *, bpr_parameters) -> Graph``
    Builds and configures the car graph once.  Returned object can be passed
    to ``execute_assignment(graph=…)`` to avoid rebuilding every iteration.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TrafficAssignment, TrafficClass

from sim.defaults import SIM_DEFAULTS
from sim.io_project import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Methodology defaults – sourced from centralized defaults.py.
# YAML overrides merge on top via load_config.
# ---------------------------------------------------------------------------

_BPR_DEFAULTS = SIM_DEFAULTS["assignment"]["bpr"]
_DEFAULT_BPR_BY_LINK_TYPE: Dict[str, Dict[str, float]] = _BPR_DEFAULTS["by_link_type"]
_DEFAULT_DAILY_CAP_FACTOR: Dict[str, Any] = _BPR_DEFAULTS["daily_capacity_factor"]
_DEFAULT_MULTI_CLASS = SIM_DEFAULTS["assignment"]["multi_class"]["classes"]


def _apply_bpr_defaults(bpr_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge YAML BPR overrides on top of built-in defaults."""
    base: Dict[str, Any] = {
        "vdf_function": _BPR_DEFAULTS["vdf_function"],
        "per_link": _BPR_DEFAULTS["per_link"],
        "alpha_default": _BPR_DEFAULTS["alpha_default"],
        "beta_default": _BPR_DEFAULTS["beta_default"],
        "daily_capacity_factor": dict(_DEFAULT_DAILY_CAP_FACTOR),
        "by_link_type": {k: dict(v) for k, v in _DEFAULT_BPR_BY_LINK_TYPE.items()},
    }
    if not bpr_params:
        return base
    merged = dict(base)
    for key, val in bpr_params.items():
        if key == "by_link_type" and isinstance(val, dict):
            merged_lt = {k: dict(v) for k, v in _DEFAULT_BPR_BY_LINK_TYPE.items()}
            for lt, lt_val in val.items():
                if lt in merged_lt and isinstance(lt_val, dict):
                    merged_lt[lt].update(lt_val)
                else:
                    merged_lt[lt] = lt_val
            merged["by_link_type"] = merged_lt
        elif key == "daily_capacity_factor" and isinstance(val, dict):
            merged_dcf = dict(_DEFAULT_DAILY_CAP_FACTOR)
            for dk, dv in val.items():
                if dk == "by_link_type" and isinstance(dv, dict):
                    merged_dcf_lt = dict(_DEFAULT_DAILY_CAP_FACTOR.get("by_link_type", {}))
                    merged_dcf_lt.update(dv)
                    merged_dcf["by_link_type"] = merged_dcf_lt
                else:
                    merged_dcf[dk] = dv
            merged["daily_capacity_factor"] = merged_dcf
        else:
            merged[key] = val
    return merged


def _resolve_multi_class(mc_cfg: Optional[Dict[str, Any]]) -> Optional[list]:
    """Return multi-class list, using built-in defaults when YAML omits classes."""
    if mc_cfg is None:
        mc_cfg = {}
    if not mc_cfg.get("enabled", True):
        return None
    return list(mc_cfg.get("classes", _DEFAULT_MULTI_CLASS))


def resolve_daily_cap_factor_default(bpr_cfg: dict) -> float:
    """Extract the scalar default from ``daily_capacity_factor`` (which may be a dict or float)."""
    dcf_raw = bpr_cfg.get("daily_capacity_factor", 10.0)
    if isinstance(dcf_raw, dict):
        return float(dcf_raw.get("default", 10.0))
    return float(dcf_raw)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_connectors(project_dir: Path) -> Dict[str, Any]:
    """Verify that centroid connectors actually reach the road network."""
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    try:
        total_conn = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='centroid_connector'"
        ).fetchone()[0]
        total_cents = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE is_centroid=1"
        ).fetchone()[0]
        good = conn.execute("""
            SELECT COUNT(DISTINCT l.a_node) FROM links l
            WHERE l.link_type='centroid_connector'
            AND EXISTS (
                SELECT 1 FROM links l2
                WHERE (l2.a_node=l.b_node OR l2.b_node=l.b_node)
                AND l2.link_type != 'centroid_connector'
            )
        """).fetchone()[0]
        return {
            "connectors": total_conn,
            "centroids": total_cents,
            "centroids_connected": good,
            "ok": good == total_cents and total_cents > 0,
        }
    finally:
        conn.close()


def fix_node_ids(project_dir: Path) -> int:
    """Renumber nodes > uint32 to small IDs (AequilibraE graph requires uint32)."""
    uint32_max = int(np.iinfo(np.uint32).max)
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    try:
        big = conn.execute(
            "SELECT node_id FROM nodes WHERE node_id > ?", (uint32_max,)
        ).fetchall()
        if not big:
            return 0
        existing = set(r[0] for r in conn.execute(
            "SELECT node_id FROM nodes WHERE node_id <= ?", (uint32_max,)
        ))
        for (old_id,) in big:
            new_id = 1
            while new_id in existing:
                new_id += 1
            conn.execute("UPDATE nodes SET node_id=? WHERE node_id=?", (-new_id, old_id))
            conn.execute("UPDATE links SET a_node=? WHERE a_node=?", (-new_id, old_id))
            conn.execute("UPDATE links SET b_node=? WHERE b_node=?", (-new_id, old_id))
            existing.add(new_id)
        neg = conn.execute("SELECT node_id FROM nodes WHERE node_id < 0").fetchall()
        for (nid,) in neg:
            conn.execute("UPDATE nodes SET node_id=? WHERE node_id=?", (-nid, nid))
            conn.execute("UPDATE links SET a_node=? WHERE a_node=?", (-nid, nid))
            conn.execute("UPDATE links SET b_node=? WHERE b_node=?", (-nid, nid))
        conn.commit()
        return len(big)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Graph builder (cacheable across iterations)
# ---------------------------------------------------------------------------

def build_graph(
    project: Project,
    mat: AequilibraeMatrix,
    *,
    bpr_parameters: Optional[Dict[str, object]] = None,
):
    """Build and configure the car-mode graph once.

    The returned graph can be passed to ``execute_assignment(graph=...)`` to
    skip the expensive ``build_graphs`` / ``prepare_graph`` cycle on every
    calibration iteration.  Only the OD matrix changes between iterations;
    the network (and therefore the graph) stays constant.
    """
    project.network.build_graphs(modes=["c"])
    graph = project.network.graphs["c"]
    # keep dead ends so centroid-connector stubs are not pruned
    graph.prepare_graph(np.asarray(mat.index[:], dtype=np.int64), remove_dead_ends=False)

    gcols = list(graph.graph.columns)
    time_field = next(
        (c for c in ("free_flow_time", "travel_time") if c in gcols), None
    )
    if time_field is None:
        raise RuntimeError(f"No time field in graph. Columns: {gcols}")

    gdf = graph.graph
    if "link_type" not in gdf.columns and "link_id" in gdf.columns:
        links_data = project.network.links.data
        if "link_type" in links_data.columns:
            lt_map = dict(zip(links_data["link_id"], links_data["link_type"]))
            gdf["link_type"] = gdf["link_id"].map(lt_map)

    graph.set_graph(time_field)
    graph.set_skimming([time_field])
    graph.set_blocked_centroid_flows(True)

    if bpr_parameters and bpr_parameters.get("per_link") and "link_type" in gdf.columns:
        by_lt = bpr_parameters.get("by_link_type") or {}
        a_default = float(bpr_parameters.get("alpha_default", 0.85))
        b_default = float(bpr_parameters.get("beta_default", 4.0))
        lt_series = gdf["link_type"].astype(str)
        gdf["alpha"] = lt_series.map(
            {lt: float(v["alpha"]) for lt, v in by_lt.items() if "alpha" in v}
        ).fillna(a_default)
        gdf["beta"] = lt_series.map(
            {lt: float(v["beta"]) for lt, v in by_lt.items() if "beta" in v}
        ).fillna(b_default)

    # Scale hourly capacity to daily-equivalent so BPR sees a meaningful V/C
    # when the OD matrix represents full-day demand.  Factor = 1/K where K is
    # the peak-hour share of daily traffic.  K varies by road type (FHWA/HCM).
    dcf_raw = (bpr_parameters or {}).get("daily_capacity_factor", 10.0)
    if isinstance(dcf_raw, dict):
        dcf_default = float(dcf_raw.get("default", 10.0))
        dcf_by_lt = dcf_raw.get("by_link_type") or {}
    else:
        dcf_default = float(dcf_raw)
        dcf_by_lt = {}

    if "capacity" in gdf.columns:
        n_nan = int(gdf["capacity"].isna().sum())
        if n_nan:
            import warnings
            warnings.warn(f"Graph has {n_nan} links with NaN capacity — filling with 50*dcf")
            gdf["capacity"] = gdf["capacity"].fillna(50.0)

        if dcf_by_lt and "link_type" in gdf.columns:
            lt_s = gdf["link_type"].astype(str)
            dcf_series = lt_s.map(
                {lt: float(v) for lt, v in dcf_by_lt.items()}
            ).fillna(dcf_default)
            gdf["_k_factor"] = 1.0 / dcf_series
            gdf["capacity"] = gdf["capacity"] * dcf_series
        else:
            gdf["_k_factor"] = 1.0 / dcf_default
            if dcf_default != 1.0:
                gdf["capacity"] = gdf["capacity"] * dcf_default

    return graph


def _resolve_vdf_params(
    bpr_parameters: Optional[Dict[str, object]],
    gdf_columns: list,
) -> Dict[str, object]:
    """Return the ``set_vdf_parameters`` dict for the given BPR config."""
    if bpr_parameters and bpr_parameters.get("per_link") and "link_type" in gdf_columns:
        return {"alpha": "alpha", "beta": "beta"}
    if bpr_parameters and not bpr_parameters.get("per_link"):
        return {
            "alpha": float(bpr_parameters.get("alpha_default", bpr_parameters.get("alpha", 0.85))),
            "beta": float(bpr_parameters.get("beta_default", bpr_parameters.get("beta", 4.0))),
        }
    return {"alpha": 0.85, "beta": 4.0}


def _resolve_time_field(graph) -> str:
    gcols = list(graph.graph.columns)
    time_field = next(
        (c for c in ("free_flow_time", "travel_time") if c in gcols), None
    )
    if time_field is None:
        raise RuntimeError(f"No time field in graph. Columns: {gcols}")
    return time_field


# ---------------------------------------------------------------------------
# Low-level assignment (reusable by calibration loop)
# ---------------------------------------------------------------------------

def execute_assignment(
    project: Project,
    mat: AequilibraeMatrix,
    *,
    algorithm: str = "bfw",
    max_iter: int = 100,
    rgap_target: float = 0.001,
    save_skims: bool = False,
    select_links: Optional[Dict[str, list]] = None,
    fixed_cost_field: Optional[str] = None,
    fixed_cost_multiplier: float = 0.0,
    vot: float = 1.0,
    bpr_parameters: Optional[Dict[str, object]] = None,
    multi_class: Optional[list] = None,
    graph=None,
    cores: int = 0,
    skim_method: str = "blended",
) -> Tuple[pd.DataFrame, Optional[AequilibraeMatrix], Dict[str, np.ndarray]]:
    """Run assignment on an already-open project with a loaded matrix.

    Returns ``(link_volume_df, skims, select_link_matrices)`` where
    *skims* is an ``AequilibraeMatrix`` when ``save_skims=True``,
    and *select_link_matrices* is ``{screenline_name: OD_array}``
    when ``select_links`` is provided.

    Parameters
    ----------
    graph : optional
        Pre-built graph from ``build_graph()``.  When *None* the graph is
        built from scratch (backwards-compatible).  Passing a cached graph
        avoids the ``build_graphs`` / ``prepare_graph`` overhead inside
        calibration iterations.
    cores : int
        Number of CPU threads for the assignment solver.  ``0`` (default)
        uses all available cores.
    skim_method : str
        ``"blended"`` (default) returns iteration-averaged skims;
        ``"final"`` returns the last-iteration skims which better
        reflect the equilibrium state.
    """
    if graph is None:
        graph = build_graph(project, mat, bpr_parameters=bpr_parameters)

    time_field = _resolve_time_field(graph)
    gdf = graph.graph
    vdf_params = _resolve_vdf_params(bpr_parameters, list(gdf.columns))

    # Build traffic classes.
    use_gc = fixed_cost_field and fixed_cost_multiplier > 0 and fixed_cost_field in gdf.columns
    mat_cores = list(mat.names) if hasattr(mat, "names") else []
    use_multi = bool(multi_class) and all(c.get("core") in mat_cores for c in (multi_class or []))

    traffic_classes: list = []
    primary_tc = None
    _aux_matrices: list = []

    if use_multi:
        # Per-cell proportional scaling: calibration adjusts wd_daily
        # cell-by-cell (sector OD scaling), so we propagate that spatial
        # pattern to each sub-core while preserving the local/through split.
        daily_view = mat.matrix_view[:, :] if mat.matrix_view is not None else None
        sub_core_names = [str(c["core"]) for c in multi_class]
        original_sum = sum(
            np.nan_to_num(mat.matrix[cn][:, :]) for cn in sub_core_names
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            cell_ratio: np.ndarray = np.where(
                original_sum > 1e-9,
                np.nan_to_num(daily_view) / original_sum,
                1.0,
            ) if daily_view is not None else np.ones_like(original_sum)

        for cls_cfg in multi_class:
            cls_name = str(cls_cfg["name"])
            cls_core = str(cls_cfg["core"])
            cls_vot = float(cls_cfg.get("vot", vot))
            cls_pce = float(cls_cfg.get("pce", 1.0))

            cls_mat = AequilibraeMatrix()
            cls_mat.create_empty(
                zones=int(len(mat.index)),
                matrix_names=[cls_core],
                memory_only=True,
            )
            cls_mat.index[:] = mat.index[:]
            cls_mat.matrix[cls_core][:, :] = mat.matrix[cls_core][:, :] * cell_ratio
            cls_mat.computational_view([cls_core])
            _aux_matrices.append(cls_mat)

            tc = TrafficClass(name=cls_name, graph=graph, matrix=cls_mat)
            if use_gc:
                tc.set_fixed_cost(fixed_cost_field, multiplier=fixed_cost_multiplier)
                tc.set_vot(cls_vot)
            if cls_pce != 1.0:
                tc.set_pce(cls_pce)
            traffic_classes.append(tc)
            if primary_tc is None:
                primary_tc = tc
    else:
        tc = TrafficClass(name="car", graph=graph, matrix=mat)
        if use_gc:
            tc.set_fixed_cost(fixed_cost_field, multiplier=fixed_cost_multiplier)
            tc.set_vot(vot)
        traffic_classes.append(tc)
        primary_tc = tc

    valid_sl: Dict[str, list] = {}
    if select_links and traffic_classes:
        graph_link_ids = set(traffic_classes[0].graph.graph["link_id"].values)
        for sl_name, link_list in select_links.items():
            valid_pairs = [(lid, d) for lid, d in link_list if lid in graph_link_ids]
            missing = [(lid, d) for lid, d in link_list if lid not in graph_link_ids]
            if missing:
                warnings.warn(
                    f"Select-link '{sl_name}': link(s) {missing} not in graph — skipped",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if valid_pairs:
                valid_sl[sl_name] = valid_pairs
        if valid_sl:
            for tc in traffic_classes:
                try:
                    tc.set_select_links(valid_sl)
                except Exception as exc:
                    warnings.warn(
                        f"set_select_links failed for class '{tc.name}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        elif select_links:
            warnings.warn(
                "All select-link entries have missing links — select-link analysis disabled",
                RuntimeWarning,
                stacklevel=2,
            )

    assig = TrafficAssignment()
    assig.set_classes(traffic_classes)
    vdf_name = str((bpr_parameters or {}).get("vdf_function", "BPR")).upper()
    assig.set_vdf(vdf_name)
    assig.set_vdf_parameters(vdf_params)
    assig.set_capacity_field("capacity")
    assig.set_time_field(time_field)
    assig.set_algorithm(algorithm)
    assig.max_iter = max_iter
    assig.rgap_target = rgap_target
    assig.set_cores(cores if cores > 0 else os.cpu_count() or 1)

    assig.execute()

    # --- Convergence diagnostics ---
    try:
        report_df = assig.report()
        if report_df is not None and not report_df.empty:
            last_row = report_df.iloc[-1]
            final_rgap = float(last_row.get("rgap", last_row.get("Relative Gap", float("nan"))))
            n_iters = len(report_df)
            if final_rgap > rgap_target:
                warnings.warn(
                    f"Assignment did NOT converge: rgap={final_rgap:.6f} > "
                    f"target={rgap_target:.6f} after {n_iters}/{max_iter} iterations",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                logger.info(f"  Assignment converged: rgap={final_rgap:.6f} in {n_iters} iterations")
    except Exception as exc:
        warnings.warn(f"Could not read assignment convergence report: {exc}", RuntimeWarning, stacklevel=2)

    result = assig.results()
    df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    if "link_id" not in df.columns:
        if df.index.name == "link_id":
            df = df.reset_index()
        else:
            df["link_id"] = graph.graph["link_id"].values[: len(df)]
    elif df.index.name == "link_id":
        df = df.reset_index(drop=True)

    skims = None
    if save_skims and primary_tc is not None:
        try:
            if skim_method == "final":
                skims = primary_tc._aon_results.skims
            else:
                skims = primary_tc.results.skims
        except Exception as exc:
            warnings.warn(f"Could not retrieve skims ({skim_method}): {exc}", RuntimeWarning, stacklevel=2)

    sl_matrices: Dict[str, np.ndarray] = {}
    if valid_sl and traffic_classes:
        for sl_name in valid_sl:
            combined: Optional[np.ndarray] = None
            for tc in traffic_classes:
                try:
                    m = tc.results.select_link_od.matrix[sl_name][:, :].copy()
                    combined = m if combined is None else combined + m
                except Exception as exc:
                    warnings.warn(
                        f"Select-link OD extraction failed for '{sl_name}' "
                        f"class '{tc.name}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if combined is not None:
                sl_matrices[sl_name] = combined

    # --- Peak-hour volumes and LOS (for display) ---
    _enrich_with_peak_hour_and_los(df, graph)

    return df, skims, sl_matrices


def _voc_to_los(voc: float) -> str:
    """Map V/C ratio to HCM Level-of-Service grade."""
    if voc <= 0.35:
        return "A"
    if voc <= 0.55:
        return "B"
    if voc <= 0.75:
        return "C"
    if voc <= 0.90:
        return "D"
    if voc <= 1.00:
        return "E"
    return "F"


def _enrich_with_peak_hour_and_los(df: pd.DataFrame, graph) -> None:
    """Add ``peak_hour_vol_AB/BA``, ``K_factor``, and ``LOS_max`` to *df* in-place."""
    gdf = graph.graph
    k_map = None
    if "_k_factor" in gdf.columns and "link_id" in gdf.columns:
        k_map = dict(zip(gdf["link_id"], gdf["_k_factor"]))

    if k_map and "link_id" in df.columns:
        df["K_factor"] = df["link_id"].map(k_map).fillna(0.10)
    else:
        df["K_factor"] = 0.10

    for d in ("AB", "BA"):
        pce_col = f"PCE_{d}"
        if pce_col in df.columns:
            df[f"peak_hour_vol_{d}"] = (df[pce_col] * df["K_factor"]).round(0)
        else:
            df[f"peak_hour_vol_{d}"] = 0.0

    if "VOC_max" in df.columns:
        df["LOS_max"] = df["VOC_max"].apply(_voc_to_los)
    else:
        df["LOS_max"] = "A"


def _detect_volume_col(df: pd.DataFrame) -> str | None:
    """Find the first ``*_tot`` column with non-zero sum."""
    return next(
        (c for c in df.columns if c.endswith("_tot") and df[c].sum() > 0),
        next((c for c in df.columns if "tot" in c.lower()), None),
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _run_assignment_pass(
    config_path: str | Path,
    *,
    algorithm: str,
    max_iter: int,
    rgap: float,
    save_skims: bool,
    banner: str,
    log_volume_note: str = "",
    skim_method: str = "blended",
) -> None:
    """Shared body for ``assign`` and ``assign-warm-skims``."""
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    assign_cfg = cfg.get("assignment") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)
    core_name = str(calib_cfg.get("core_name", "wd_daily"))
    cores = int(assign_cfg.get("cores", 0))

    logger.info(banner)

    logger.info("\n1) Pre-flight checks ...")
    renamed = fix_node_ids(project_dir)
    if renamed:
        logger.info(f"   Fixed {renamed} overflow node IDs")
    conn_info = _check_connectors(project_dir)
    logger.info(f"   Connectors: {conn_info['connectors']}, "
          f"centroids: {conn_info['centroids']}, "
          f"connected: {conn_info['centroids_connected']}")
    if not conn_info["ok"]:
        logger.warning("   WARNING: not all centroids reach the road network!")

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}")

    logger.info(f"\n2) Loading matrix: {matrix_path}")
    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])
    total_demand = float(mat.matrix_view.sum())
    logger.info(f"   Core '{core_name}': {mat.zones} zones, demand={total_demand:,.0f}")

    gc_cfg = assign_cfg.get("generalized_cost") or {}
    gc_enabled = bool(gc_cfg.get("enabled", True))
    gc_field = str(gc_cfg.get("fixed_cost_field", "distance")) if gc_enabled else None
    gc_mult = float(gc_cfg.get("fixed_cost_multiplier", 0.006)) if gc_enabled else 0.0
    gc_vot = float(gc_cfg.get("vot", 1.0))
    bpr_params = _apply_bpr_defaults(assign_cfg.get("bpr") or {})
    mc_cfg = assign_cfg.get("multi_class") or {}
    multi_classes = _resolve_multi_class(mc_cfg)

    logger.info(f"\n3) Running {algorithm.upper()} ...")
    project = Project()
    project.open(str(project_dir))
    try:
        df, skims, _sl = execute_assignment(
            project,
            mat,
            algorithm=algorithm,
            max_iter=max_iter,
            rgap_target=rgap,
            save_skims=save_skims,
            fixed_cost_field=gc_field,
            fixed_cost_multiplier=gc_mult,
            vot=gc_vot,
            bpr_parameters=bpr_params,
            multi_class=multi_classes,
            cores=cores,
            skim_method=skim_method,
        )
    finally:
        project.close()
        mat.close()

    tot_col = _detect_volume_col(df)
    total_vol = float(df[tot_col].sum()) if tot_col else 0.0
    note = log_volume_note or "Results"
    logger.info(f"\n4) {note}: {len(df)} links, vol_col={tot_col}, total={total_vol:,.0f}")

    if total_vol <= 0:
        import sqlite3 as _sq
        logger.warning("\n   *** ZERO VOLUME — diagnostics ***")
        logger.info(f"   Matrix demand: {total_demand:,.0f}")
        logger.info(f"   Connectors OK: {conn_info['ok']}  "
              f"(connected={conn_info['centroids_connected']}/{conn_info['centroids']})")
        db = str(project_dir / "project_database.sqlite")
        cn = _sq.connect(db)
        modes = cn.execute(
            "SELECT modes, COUNT(*) FROM links WHERE link_type='centroid_connector' GROUP BY modes"
        ).fetchall()
        logger.info(f"   Connector modes: {modes}")
        bad_tt = cn.execute(
            "SELECT SUM(CASE WHEN travel_time_ab<=0 OR travel_time_ab IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM links"
        ).fetchone()
        logger.info(f"   Links with bad travel_time: {bad_tt[0]}/{bad_tt[1]}")
        cn.close()
        logger.info("   Likely causes: connectors not reaching road network, or centroid IDs mismatch.")

    out_path = output_dir / "assignment_results.parquet"
    df.to_parquet(str(out_path), index=False)
    logger.info(f"   Saved: {out_path}")

    if skims is not None:
        skim_path = output_dir / "skims.aem"
        try:
            skims.export(str(skim_path))
            logger.info(f"   Skims saved: {skim_path}")
        except Exception as exc:
            warnings.warn(
                f"Skim export failed ({skim_path}): {exc}. "
                "Trip distribution may fall back to Euclidean distance.",
                RuntimeWarning,
                stacklevel=2,
            )


def run_assignment(config_path: str | Path = "config/brno/sim.yaml") -> None:
    cfg = load_config(config_path)
    calib_cfg = cfg.get("calibration") or {}
    algorithm = str(calib_cfg.get("algorithm", "bfw"))
    max_iter = int(calib_cfg.get("max_iter", 100))
    rgap = float(calib_cfg.get("rgap_target", 0.001))
    save_skims = bool(calib_cfg.get("save_skims", False))
    _run_assignment_pass(
        config_path,
        algorithm=algorithm,
        max_iter=max_iter,
        rgap=rgap,
        save_skims=save_skims,
        banner="=== TRAFFIC ASSIGNMENT ===",
    )


def run_warm_skim_assignment(config_path: str | Path = "config/brno/sim.yaml") -> None:
    """Shorter assignment pass with skims always saved, for impedance before ``distribute``."""
    cfg = load_config(config_path)
    warm = (cfg.get("assignment") or {}).get("warm_skim_pass") or {}
    calib_cfg = cfg.get("calibration") or {}
    algorithm = str(warm.get("algorithm") or calib_cfg.get("algorithm", "bfw"))
    max_iter = int(warm.get("max_iter", 30))
    rgap = float(warm.get("rgap_target", 0.01))
    with warnings.catch_warnings():
        warnings.filterwarnings("default", message="Assignment did NOT converge", category=RuntimeWarning)
        warnings.simplefilter("always", RuntimeWarning)

        def _warn_to_print(message, category, filename, lineno, file=None, line=None):
            if "Assignment did NOT converge" in str(message):
                logger.info(f"  [info] Warm skim: {message} (acceptable for preliminary pass)")
            else:
                warnings.showwarning(message, category, filename, lineno, file, line)

        old_showwarning = warnings.showwarning
        warnings.showwarning = _warn_to_print
        try:
            _run_assignment_pass(
                config_path,
                algorithm=algorithm,
                max_iter=max_iter,
                rgap=rgap,
                save_skims=True,
                banner="=== WARM SKIM ASSIGNMENT (for trip distribution) ===",
                log_volume_note="Warm-pass results (intermediate; re-run assign after distribute if needed)",
                skim_method="final",
            )
        finally:
            warnings.showwarning = old_showwarning


# ---------------------------------------------------------------------------
# Temporal assignment (date + period)
# ---------------------------------------------------------------------------

def run_temporal_assignment(
    config_path: str | Path,
    date_str: str,
    period: str = "daily",
) -> pd.DataFrame:
    """Run assignment for a specific date, scaling the base OD by the learned
    day factor and optional period share.

    Returns the link-volume DataFrame (same format as baseline assignment).
    """
    from sim.temporal import load_profile, get_combined_factor, classify_day

    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))

    algorithm = str(calib_cfg.get("algorithm", "bfw"))
    max_iter = int(calib_cfg.get("max_iter", 100))
    rgap = float(calib_cfg.get("rgap_target", 0.001))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))

    profile = load_profile(cfg)
    factor = get_combined_factor(date_str, period, profile)
    day_type = classify_day(date_str)

    logger.info(f"=== TEMPORAL ASSIGNMENT: {date_str} ({day_type}), period={period}, factor={factor:.3f} ===")

    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])

    # Scale matrix data in memory (do NOT save — keep original on disk)
    mat.matrix_view[:, :] = mat.matrix_view[:, :] * factor
    total_demand = float(mat.matrix_view.sum())
    logger.info(f"  Scaled demand: {total_demand:,.0f}")

    fix_node_ids(project_dir)

    project = Project()
    project.open(str(project_dir))
    try:
        df, _skims, _sl = execute_assignment(
            project,
            mat,
            algorithm=algorithm,
            max_iter=max_iter,
            rgap_target=rgap,
        )
    finally:
        project.close()
        mat.close()

    tot_col = _detect_volume_col(df)
    total_vol = float(df[tot_col].sum()) if tot_col else 0.0
    logger.info(f"  Result: {len(df)} links, total_vol={total_vol:,.0f}")

    return df
