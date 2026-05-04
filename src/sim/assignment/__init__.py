"""``sim.assignment`` -- traffic assignment: graph building, execution, CLI.

Public API re-exported here for convenience.  Downstream code can import
from ``sim.assignment`` directly or from the individual sub-modules.
"""
# -- config ------------------------------------------------------------------
from sim.assignment.config import (
    _apply_bpr_defaults,
    _resolve_multi_class,
    resolve_daily_cap_factor_default,
)

# -- preflight ---------------------------------------------------------------
from sim.assignment.preflight import (
    fix_node_ids,
)

# -- graph -------------------------------------------------------------------
from sim.assignment.graph import (
    _resolve_vdf_params,
    build_graph,
)

# -- executor ----------------------------------------------------------------
from sim.assignment.executor import (
    _detect_volume_col,
    _validate_algorithm,
    _voc_to_los,
    execute_assignment,
)

# -- pipeline (CLI entry-points) ---------------------------------------------
from sim.assignment.pipeline import (
    run_assignment,
    run_warm_skim_assignment,
)

__all__ = [
    # config
    "_apply_bpr_defaults",
    "_resolve_multi_class",
    "resolve_daily_cap_factor_default",
    # preflight
    "fix_node_ids",
    # graph
    "build_graph",
    "_resolve_vdf_params",
    # executor
    "execute_assignment",
    "_validate_algorithm",
    "_voc_to_los",
    "_detect_volume_col",
    # pipeline
    "run_assignment",
    "run_warm_skim_assignment",
]
