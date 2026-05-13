"""BPR alpha sensitivity (numeric): same V/C reacts more on higher alpha (parallel-corridor what-if)."""
from __future__ import annotations

import pytest

from sim.assignment.config import _apply_bpr_defaults
from sim.diagnostics.parallel_corridor import bpr_alpha_beta_for_link_type, bpr_delay_multiplier


def test_delay_multiplier_rises_with_alpha_at_fixed_voc():
    voc = 0.85
    low = bpr_delay_multiplier(voc, alpha=0.15, beta=4.0)
    high = bpr_delay_multiplier(voc, alpha=0.35, beta=4.0)
    assert high > low


def test_default_secondary_steeper_than_motorway_at_same_voc():
    bpr = _apply_bpr_defaults({})
    am, _ = bpr_alpha_beta_for_link_type("motorway", bpr)
    as_, _ = bpr_alpha_beta_for_link_type("secondary", bpr)
    voc = 0.9
    assert bpr_delay_multiplier(voc, as_) > bpr_delay_multiplier(voc, am)
