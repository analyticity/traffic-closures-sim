"""Scenario job lifecycle state model.

Provides an explicit ``JobStatus`` enum that replaces the ad-hoc string
status previously used by ``ScenarioJob``.  The enum correctly
distinguishes ``QUEUED`` from ``RUNNING`` (critical when the
``ThreadPoolExecutor`` has ``max_workers=1``) and adds ``CANCELLED``
and ``TIMED_OUT`` for graceful lifecycle management.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Set


class JobStatus(Enum):
    """Lifecycle states of a scenario job."""

    QUEUED = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMED_OUT = auto()


JOB_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.TIMED_OUT},
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.TIMED_OUT: set(),
}


class InvalidJobTransitionError(Exception):
    """Raised when a job status transition is not permitted."""


def transition_job(current: JobStatus, target: JobStatus) -> JobStatus:
    """Return *target* if the transition is valid, else raise."""
    allowed = JOB_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidJobTransitionError(
            f"Job transition {current.name} -> {target.name} not allowed. "
            f"Permitted targets: {sorted(s.name for s in allowed)}"
        )
    return target
