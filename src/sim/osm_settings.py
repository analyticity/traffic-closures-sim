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

from typing import Optional

logger = logging.getLogger(__name__)


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
