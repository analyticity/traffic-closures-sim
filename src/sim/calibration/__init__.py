"""Traffic model calibration package.

Public API — all previously importable names from ``sim.calibration``
are re-exported here so that ``from sim.calibration import X`` keeps
working after the monolith→package refactor.
"""
from __future__ import annotations

# --- Core calibration entry points ---
from sim.calibration.odme import (  # noqa: F401
    run_odme_calibration,
    run_entropy_odme,
    _entropy_update_step,
    _spiess_update_step,
)
from sim.calibration.legacy import (  # noqa: F401
    run_calibration,
    run_multistage_calibration,
)
from sim.calibration.supply_tuning import (  # noqa: F401
    SupplyParams,
    apply_supply_params,
    compute_objective,
    run_supply_tuning,
)

# --- Validation ---
from sim.calibration.validation import (  # noqa: F401
    run_validation_only,
    run_match_diagnostics,
    validate_journey_times,
    compute_validation_benchmarks,
    compute_class_speed_comparison,
    match_csd_to_links,
    _check_final_convergence,
)

# --- Observed data loaders ---
from sim.calibration.observed import (  # noqa: F401
    load_pentlogram,
    load_csd,
    load_csd_unfiltered,
    load_csd_as_link_counts,
    split_csd_for_calibration,
    validate_geometries_or_fail,
    aggregate_csd_by_class,
    aggregate_model_by_class,
    _classify_csd_road,
    _CSD_COMPATIBLE_LINK_TYPES,
    _load_network_links,
)

# --- Matching ---
from sim.calibration.matching import (  # noqa: F401
    match_counts_to_links,
    match_quality_report,
    _bearing_from_geom,
    _bearing_diff,
)

# --- Metrics / statistics ---
from sim.calibration.metrics import (  # noqa: F401
    compute_geh,
    compute_stats,
    compute_extended_link_metrics,
    compute_class_volume_breakdown,
    _coarse_road_class,
)

# --- Gateway ---
from sim.calibration.gateway import (  # noqa: F401
    _build_screenline_gateway_map,
)

# --- Context / helpers ---
from sim.calibration.context import (  # noqa: F401
    _CalibrationContext,
    _sr_val,
    _compute_count_weights,
    _odme_objective,
)

# --- Screenlines ---
from sim.calibration.screenlines import (  # noqa: F401
    ScreenlineDef,
    ScreenlineResult,
    load_screenlines,
    load_screenlines_with_auto,
    resolve_screenline_links,
    evaluate_all_screenlines,
    auto_generate_screenlines,
)

# FSM state model (now integrated into _CalibrationContext)
from sim.calibration.state import (  # noqa: F401
    CalibrationState,
    CalibrationRun,
    TRANSITIONS,
)
