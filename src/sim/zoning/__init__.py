"""``sim.zoning`` -- transport-analysis zoning pipeline.

Builds the model AOI from the road network, loads zones from modular
sources (see ``sim.zoning.sources``), discovers boundary gateways,
creates centroid connectors, and exports diagnostics / maps.

Public API
----------
build_zones_and_connectors
    End-to-end pipeline entry point (called from ``run.py``).
"""
from sim.zoning.pipeline import _population_needs_remap, build_zones_and_connectors

__all__ = [
    "build_zones_and_connectors",
    "_population_needs_remap",
]
