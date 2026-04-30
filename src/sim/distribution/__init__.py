"""``sim.distribution`` -- trip distribution: gravity calibration + IPF.

Public API re-exported here for convenience.  Downstream code can import
from ``sim.distribution`` directly or from the individual sub-modules.
"""
# -- P/A vectors -------------------------------------------------------------
from sim.distribution.pa_vectors import build_pa_vectors

# -- impedance ---------------------------------------------------------------
from sim.distribution.impedance import _euclidean_impedance

# -- gravity + IPF -----------------------------------------------------------
from sim.distribution.gravity import calibrate_gravity_simple, run_ipf

# -- pipeline (CLI entry-point) ----------------------------------------------
from sim.distribution.pipeline import run_distribution

__all__ = [
    # pa_vectors
    "build_pa_vectors",
    # impedance
    "_euclidean_impedance",
    # gravity
    "calibrate_gravity_simple",
    "run_ipf",
    # pipeline
    "run_distribution",
]
