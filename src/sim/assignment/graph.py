"""Graph construction: build and configure the AequilibraE car-mode graph."""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix


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
    # when the OD matrix represents full-day demand.
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
