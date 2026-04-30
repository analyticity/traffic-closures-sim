"""Calibration FSM state model.

Explicit enum-based states and a transition table that replace the
previously implicit ad-hoc state management spread across local
variables, side-effects, and boolean flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class CalibrationState(Enum):
    """Explicit states for the calibration workflow."""

    INIT = auto()
    ASSIGN = auto()
    MATCH = auto()
    EVALUATE = auto()
    UPDATE_DEMAND = auto()
    PERSIST_CHECKPOINT = auto()
    FINALIZE_BEST = auto()
    FINALIZE_SUCCESS = auto()
    DEGRADED_INPUT = auto()
    STALLED = auto()
    FAILED = auto()


TRANSITIONS: Dict[CalibrationState, Set[CalibrationState]] = {
    CalibrationState.INIT: {CalibrationState.ASSIGN, CalibrationState.FAILED},
    CalibrationState.ASSIGN: {CalibrationState.MATCH, CalibrationState.FAILED},
    CalibrationState.MATCH: {CalibrationState.EVALUATE, CalibrationState.DEGRADED_INPUT},
    CalibrationState.EVALUATE: {
        CalibrationState.UPDATE_DEMAND,
        CalibrationState.FINALIZE_BEST,
        CalibrationState.STALLED,
    },
    CalibrationState.UPDATE_DEMAND: {CalibrationState.PERSIST_CHECKPOINT, CalibrationState.FAILED},
    CalibrationState.PERSIST_CHECKPOINT: {CalibrationState.ASSIGN},
    CalibrationState.FINALIZE_BEST: {CalibrationState.FINALIZE_SUCCESS, CalibrationState.FAILED},
    CalibrationState.STALLED: {CalibrationState.FINALIZE_BEST},
    CalibrationState.DEGRADED_INPUT: {CalibrationState.EVALUATE},
    CalibrationState.FINALIZE_SUCCESS: set(),
    CalibrationState.FAILED: set(),
}


@dataclass
class CalibrationRun:
    """Mutable runtime state of a calibration run."""

    state: CalibrationState = CalibrationState.INIT
    iteration: int = 0
    best_Z: float = float("inf")
    best_iteration: int = 0
    stop_reason: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
