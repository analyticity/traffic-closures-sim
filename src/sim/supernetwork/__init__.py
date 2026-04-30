"""``sim.supernetwork`` -- coarse national network for external-zone gateway mapping.

Builds a simplified NetworkX digraph from OSM PBF major roads, snaps external
municipality centroids and model-boundary gateways, computes shortest-path
costs, and classifies commuting relations as internal, inbound, outbound,
or through-traffic.

Public API
----------
run_build_supernetwork
    End-to-end pipeline entry point (called from ``run.py``).
"""
from sim.supernetwork.pipeline import run_build_supernetwork

__all__ = [
    "run_build_supernetwork",
]
