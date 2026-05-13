"""Corridor aggregation and post-corridor exclusion for count–link matching."""

from __future__ import annotations

import numpy as np

from sim.calibration.matching import _compute_match_confidence


def test_compute_match_confidence_flags_insane_corridor_volume() -> None:
    """Most I/13×I/27 style: summed parallel arcs ≫ observed cars.

    With the geometric-trust redesign, high match_quality (>=0.65) bypasses
    volume checks.  Use a lower match_quality to exercise the volume-ratio
    exclusion path that catches corridor over-aggregation.
    """
    conf = _compute_match_confidence(
        20985.0,
        5733.0,
        "trunk",
        np.inf,
        0.55,
        threshold=0.25,
    )
    assert conf < 0.25


def test_compute_match_confidence_keeps_plausible_corridor() -> None:
    conf = _compute_match_confidence(
        10400.0,
        8475.0,
        "primary",
        np.inf,
        0.84,
        threshold=0.25,
    )
    assert conf >= 0.25
