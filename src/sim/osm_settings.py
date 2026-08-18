"""Central place for osmnx / Overpass runtime settings.

The public Overpass instance (``overpass-api.de``) rate-limits and occasionally
refuses connections outright — a full pipeline run queries it several times
(network import, tag enrichment, admin boundaries for zones), which is enough
to get throttled. Point the pipeline at a mirror without touching code:

    export OVERPASS_URL=https://overpass.kumi.systems/api
    docker run -e OVERPASS_URL=https://overpass.kumi.systems/api ...

Known public mirrors (same API, same data, different capacity):
    https://overpass-api.de/api            default, strictest limits
    https://overpass.kumi.systems/api      usually the fastest
    https://overpass.osm.ch/api
    https://overpass.private.coffee/api

``OVERPASS_TIMEOUT`` (seconds, default 180) is also honoured.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_applied = False


def apply_osmnx_settings(overpass_url: Optional[str] = None, *, force: bool = False) -> None:
    """Apply Overpass endpoint / timeout settings to osmnx.

    Safe to call repeatedly; the work is done once per process unless *force*.
    Silently does nothing when osmnx is not importable, so callers that only
    need it as a side effect do not have to guard the import.
    """
    global _applied
    if _applied and not force:
        return

    try:
        import osmnx as ox
    except ImportError:
        return

    url = overpass_url or os.environ.get("OVERPASS_URL") or ""
    url = url.strip().rstrip("/")

    if url:
        # osmnx >= 2.0 uses `overpass_url`, 1.x uses `overpass_endpoint`.
        for attr in ("overpass_url", "overpass_endpoint"):
            if hasattr(ox.settings, attr):
                setattr(ox.settings, attr, url)
                logger.info("Overpass endpoint set to %s (via ox.settings.%s)", url, attr)
                break
        else:
            logger.warning(
                "OVERPASS_URL=%s given but this osmnx build exposes neither "
                "settings.overpass_url nor settings.overpass_endpoint — ignoring.",
                url,
            )

    _apply_timeout(ox)
    _applied = True


def _apply_timeout(ox) -> None:
    timeout_raw = os.environ.get("OVERPASS_TIMEOUT", "180")
    try:
        timeout = int(float(timeout_raw))
    except ValueError:
        logger.warning("OVERPASS_TIMEOUT=%r is not a number — keeping the default.", timeout_raw)
        timeout = 0
    if timeout > 0 and hasattr(ox.settings, "timeout"):
        ox.settings.timeout = timeout


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def cool_off(what: str = "next Overpass query") -> None:
    """Pause before a query that follows another heavy one on the same server.

    ``OVERPASS_COOLOFF_S`` (default 20) — set to 0 to disable.
    """
    seconds = _env_float("OVERPASS_COOLOFF_S", 20.0)
    if seconds <= 0:
        return
    logger.info("Waiting %.0f s %s (Overpass slot cool-off).", seconds, what)
    time.sleep(seconds)


def overpass_retry(fn: Callable[[], T], *, what: str = "Overpass query") -> T:
    """Call *fn*, retrying on connection errors with exponential backoff.

    ``overpass-api.de`` allows only a couple of concurrent slots per IP. When a
    big query has just finished, the next one can be refused at the TCP level
    (``ConnectionRefusedError``/``Errno 111``) rather than answered with an HTTP
    status, which osmnx's own rate-limit handling does not cover. A short wait
    is normally enough for a slot to free up.

    Tunable via ``OVERPASS_RETRIES`` (default 4) and ``OVERPASS_BACKOFF_S``
    (default 30 — first wait; each further attempt doubles it).
    """
    attempts = max(int(_env_float("OVERPASS_RETRIES", 4)), 1)
    backoff = max(_env_float("OVERPASS_BACKOFF_S", 30.0), 1.0)

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            if not _is_connection_error(exc):
                raise
            last = exc
            if attempt == attempts:
                break
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "%s refused by the Overpass server (attempt %d/%d): %s. "
                "Waiting %.0f s for a slot to free up.",
                what, attempt, attempts, type(exc).__name__, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"{what} failed after {attempts} attempts — the Overpass server kept refusing "
        f"the connection. Either wait a few minutes, or point the pipeline at a mirror:\n"
        f"    OVERPASS_URL=https://overpass.osm.ch/api\n"
        f"    (docker build --build-arg OVERPASS_URL=...)"
    ) from last


def _is_connection_error(exc: BaseException) -> bool:
    """True for the connection-level failures that are worth retrying."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
            return True
        if type(exc).__name__ in {
            "ConnectionError",          # requests.exceptions.ConnectionError
            "ConnectTimeout",
            "ReadTimeout",
            "MaxRetryError",
            "NewConnectionError",
        }:
            return True
        exc = exc.__cause__ or exc.__context__
    return False
