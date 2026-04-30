"""``sim.demand`` -- build OD matrices from commuting, supernetwork, and
synthetic segments; register them in AequilibraE.

Public API re-exported here for convenience.  Downstream code can import
from ``sim.demand`` directly or from the individual sub-modules.
"""
# -- pipeline (entry points) ------------------------------------------------
from sim.demand.pipeline import (
    assert_build_demand_prerequisites,
    load_or_build_od_matrix,
)

# -- configuration -----------------------------------------------------------
from sim.demand.config import (
    DemandBuildCfg,
    ExternalProcessingCfg,
    PeriodShares,
    PurposeConv,
    _normalize_named_weights,
    _normalize_period_shares,
    _parse_gateway_pair_weights,
    _validate_shares,
)

# -- naming / zone lookup ----------------------------------------------------
from sim.demand.naming import (
    _build_zone_name_index,
    _match_zone_id,
    _strip_geo_suffix,
)

# -- seeds -------------------------------------------------------------------
from sim.demand.seeds import (
    _estimate_total_daily_trips_from_csd,
)

# -- temporal profiles -------------------------------------------------------
from sim.demand.temporal import (  # noqa: F401
    run_learn_profile,
    load_profile,
    classify_day,
    get_day_factor,
    get_period_share,
    get_combined_factor,
    get_demand_period_shares,
    day_info,
)

__all__ = [
    # pipeline
    "load_or_build_od_matrix",
    "assert_build_demand_prerequisites",
    # dataclasses
    "DemandBuildCfg",
    "ExternalProcessingCfg",
    "PeriodShares",
    "PurposeConv",
    # config helpers (tested)
    "_validate_shares",
    "_normalize_period_shares",
    "_normalize_named_weights",
    "_parse_gateway_pair_weights",
    # naming helpers (tested)
    "_strip_geo_suffix",
    "_build_zone_name_index",
    "_match_zone_id",
    # seed helpers (tested)
    "_estimate_total_daily_trips_from_csd",
]
