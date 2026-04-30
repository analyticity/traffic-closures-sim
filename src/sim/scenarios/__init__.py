"""``sim.scenarios`` -- what-if scenario engine with closure support.

Re-exports public API so downstream code can ``from sim.scenarios import X``.
"""
from __future__ import annotations

from sim.scenarios.engine import (  # noqa: F401
    apply_scenario_to_graph,
    submit_scenario,
    get_job,
    ScenarioJob,
)

from sim.scenarios.state import (  # noqa: F401
    JobStatus,
    JOB_TRANSITIONS,
    InvalidJobTransitionError,
    transition_job,
)

from sim.scenarios.closures import (  # noqa: F401
    closures_for_date,
    closures_geojson_for_date,
)
