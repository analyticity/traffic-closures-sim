"""Logging configuration with structured context injection.

Configures root-level logging and injects ``run_id`` / ``iteration``
context variables into every log record for calibration audit trails.
"""
from __future__ import annotations

import logging
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
