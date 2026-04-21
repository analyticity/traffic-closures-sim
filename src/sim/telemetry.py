"""Logging configuration and run-level correlation IDs.

Replaces ad-hoc ``print()`` calls with structured logging that supports
hierarchical loggers, severity levels, and correlation identifiers
(``run_id``, ``iteration``) for calibration audit trails.
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

run_id_var: ContextVar[str] = ContextVar("run_id", default="")
iteration_var: ContextVar[int] = ContextVar("iteration", default=0)


class RunContextFilter(logging.Filter):
    """Inject ``run_id`` and ``iteration`` into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get("")  # type: ignore[attr-defined]
        record.iteration = iteration_var.get(0)  # type: ignore[attr-defined]
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root-level logging with a human-readable format."""
    fmt = (
        "%(asctime)s %(levelname)-7s [%(name)s] "
        "run=%(run_id)s it=%(iteration)s  %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    handler.addFilter(RunContextFilter())

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)


def new_run_id() -> str:
    """Generate and activate a new run correlation ID."""
    rid = uuid.uuid4().hex[:12]
    run_id_var.set(rid)
    return rid


def set_iteration(it: int) -> None:
    iteration_var.set(it)
