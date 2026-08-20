"""One place to decide which Overpass endpoint the pipeline talks to.

Three different libraries reach for Overpass during a run and each has its own
default, which makes an outage hard to route around:

* AequilibraE's OSM importer — ``parameters.yml`` ships
  ``http://overpass-api.de/api``, over plain HTTP.
* osmnx (zone polygons, link enrichment) — ``ox.settings.overpass_url``.
* our own ``scripts/fetch_zone_boundaries.py``.

``overpass-api.de`` publishes two A records and has repeatedly had one of them
refuse connections while the other served fine, and has been down on both at
once.  Pinning a host in ``/etc/hosts`` (``docker build --add-host``) only helps
in the first case and goes stale when the addresses rotate, so the endpoint is
configurable instead:

    OVERPASS_ENDPOINT=https://overpass.osm.ch/api   # environment wins

or in ``config/<city>/sim.yaml``::

    osm:
      overpass_endpoint: "https://overpass.osm.ch/api"

The value is the API base without ``/interpreter`` — the same convention
AequilibraE and osmnx both use.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ENV_VAR = "OVERPASS_ENDPOINT"
DEFAULT_ENDPOINT = "https://overpass-api.de/api"


def endpoint(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Endpoint to use: environment, then config, then upstream default."""
    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return from_env.rstrip("/")
    from_cfg = str(((cfg or {}).get("osm") or {}).get("overpass_endpoint", "") or "").strip()
    if from_cfg:
        return from_cfg.rstrip("/")
    return DEFAULT_ENDPOINT


def export_to_env(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Publish the configured endpoint so call sites without ``cfg`` see it too."""
    value = endpoint(cfg)
    os.environ[ENV_VAR] = value
    return value


def configure_osmnx(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Point osmnx at the configured endpoint.  Safe to call repeatedly."""
    value = endpoint(cfg)
    try:
        import osmnx as ox

        if str(getattr(ox.settings, "overpass_url", "")) != value:
            ox.settings.overpass_url = value
            logger.info("osmnx Overpass endpoint: %s", value)
    except ImportError:
        pass
    return value


def configure_aequilibrae(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Write the endpoint into AequilibraE's parameters before an OSM import.

    AequilibraE reads ``osm.overpass_endpoint`` from the parameters file of the
    open project, so this has to run after the project exists and before
    ``create_from_osm``.
    """
    value = endpoint(cfg)
    try:
        from aequilibrae import Parameters

        p = Parameters()
        osm = p.parameters.setdefault("osm", {})
        if str(osm.get("overpass_endpoint", "")) != value:
            osm["overpass_endpoint"] = value
            p.write_back()
            logger.info("AequilibraE Overpass endpoint: %s", value)
    except Exception as e:  # noqa: BLE001 — never block the import over this
        logger.warning("Could not set AequilibraE Overpass endpoint (%s): %s", value, e)
    return value
