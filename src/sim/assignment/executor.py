"""Low-level traffic assignment execution and result enrichment."""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TrafficAssignment, TrafficClass

from sim.assignment.config import multiclass_matrix_core_status
from sim.assignment.graph import build_graph, _resolve_time_field, _resolve_vdf_params

logger = logging.getLogger(__name__)

_UE_ALGORITHMS = frozenset({"bfw", "cfw", "msa"})
_AON_ALGORITHMS = frozenset({"aon", "all-or-nothing"})
_RGAP_QUALITY_THRESHOLD = 1e-3


def _validate_algorithm(algorithm: str, *, allow_aon: bool = False) -> str:
    """Validate and normalize the assignment algorithm name.

    Raises ``ValueError`` if the algorithm is AoN and ``allow_aon`` is False.
    """
    alg = algorithm.strip().lower()
    if alg in _AON_ALGORITHMS:
        if not allow_aon:
            raise ValueError(
                f"Algorithm '{algorithm}' is All-or-Nothing and is not permitted "
                f"for production assignment. AoN ignores congestion feedback and "
                f"produces structurally incorrect flows. Use 'bfw' (recommended), "
                f"'cfw', or 'msa' for user-equilibrium assignment. "
                f"Set calibration.allow_aon=true only for diagnostic use."
            )
        logger.warning(
            "AoN algorithm selected with allow_aon=true. "
            "Results are NOT suitable for calibration or forecasting."
        )
    elif alg not in _UE_ALGORITHMS:
        logger.warning(
            "Algorithm '%s' is not in the standard UE set (%s). "
            "Proceeding, but verify AequilibraE supports it.",
            algorithm, ", ".join(sorted(_UE_ALGORITHMS)),
        )
    return alg


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
    allow_aon: bool = False,
    strict_convergence: bool = False,
    assignment_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, Optional[AequilibraeMatrix], Dict[str, np.ndarray]]:
    """Run assignment on an already-open project with a loaded matrix.

    Returns ``(link_volume_df, skims, select_link_matrices)``.

    Parameters
    ----------
    allow_aon : bool
        If False (default), AoN algorithms raise ``ValueError``.
    strict_convergence : bool
        If True, raises ``RuntimeError`` when assignment does not converge
        within the target relative gap.
    assignment_cfg : dict, optional
        Full ``assignment`` section from config; used for graph-level settings
        like ``blocked_centroid_flows`` when no pre-built graph is supplied.
    """
    algorithm = _validate_algorithm(algorithm, allow_aon=allow_aon)
    if graph is None:
        graph = build_graph(
            project, mat,
            bpr_parameters=bpr_parameters,
            assignment_cfg=assignment_cfg,
        )

    time_field = _resolve_time_field(graph)
    gdf = graph.graph
    vdf_params = _resolve_vdf_params(bpr_parameters, list(gdf.columns))

    use_gc = fixed_cost_field and fixed_cost_multiplier > 0 and fixed_cost_field in gdf.columns
    mat_cores = [str(x) for x in (list(mat.names) if hasattr(mat, "names") else [])]
    mc_enabled, mc_required, mc_missing = multiclass_matrix_core_status(mat_cores, multi_class)
    use_multi = bool(multi_class) and mc_enabled
    if multi_class:
        if use_multi:
            logger.info(
                "Multi-class equilibrium: %d class(es), cores %s (matrix exposes %d OD name(s)).",
                len(multi_class),
                mc_required,
                len(mat_cores),
            )
        else:
            logger.warning(
                "Multi-class equilibrium DISABLED — missing OD cores %s "
                "(matrix has %d name(s); first names: %s). "
                "Using a single TrafficClass.",
                mc_missing,
                len(mat_cores),
                mat_cores[:12],
            )

    traffic_classes: list = []
    primary_tc = None
    _aux_matrices: list = []

    if use_multi:
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
    convergence_meta: Dict[str, Any] = {
        "algorithm": algorithm,
        "rgap_target": rgap_target,
        "max_iter": max_iter,
        "final_rgap": None,
        "n_iterations": None,
        "converged": False,
    }
    try:
        report_df = assig.report()
        if report_df is not None and not report_df.empty:
            last_row = report_df.iloc[-1]
            final_rgap = float(last_row.get("rgap", last_row.get("Relative Gap", float("nan"))))
            n_iters = len(report_df)
            convergence_meta["final_rgap"] = final_rgap
            convergence_meta["n_iterations"] = n_iters
            if final_rgap > rgap_target:
                convergence_meta["converged"] = False
                msg = (
                    f"Assignment did NOT converge: rgap={final_rgap:.6f} > "
                    f"target={rgap_target:.6f} after {n_iters}/{max_iter} iterations"
                )
                if strict_convergence:
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                convergence_meta["converged"] = True
                logger.info(f"  Assignment converged: rgap={final_rgap:.6f} in {n_iters} iterations")
                if final_rgap > _RGAP_QUALITY_THRESHOLD:
                    logger.warning(
                        "  Quality note: rgap=%.6f meets target but exceeds "
                        "recommended quality threshold (%.0e). Consider tightening "
                        "rgap_target for production use.",
                        final_rgap, _RGAP_QUALITY_THRESHOLD,
                    )
    except RuntimeError:
        raise
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
                if hasattr(primary_tc, "_aon_results") and primary_tc._aon_results is not None:
                    skims = primary_tc._aon_results.skims
                else:
                    logger.warning(
                        "_aon_results unavailable (AequilibraE API change?), "
                        "falling back to results.skims"
                    )
                    skims = primary_tc.results.skims
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

    _enrich_with_peak_hour_and_los(df, graph)

    return df, skims, sl_matrices, convergence_meta


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
