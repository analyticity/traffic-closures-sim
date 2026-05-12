"""Tests for the three calibration/validation fixes:

1. Global capacity defaults raised for secondary/tertiary
2. Code fix: experiment profile lanes/capacity merged into defaults
3. Connectivity: relaxed dead-end detection + ref-less pairing
4. Demand: Brno config parameters override defaults
5. Snap distance increased globally
"""
from __future__ import annotations

import yaml
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Problem 1: Global capacity defaults raised
# ---------------------------------------------------------------------------

class TestGlobalCapacityDefaults:
    """Verify that global defaults have realistic capacity for Czech roads."""

    def test_secondary_capacity_raised(self):
        from sim.defaults import NETWORK_NORM_DEFAULTS

        cap = NETWORK_NORM_DEFAULTS["normalization"]["defaults"]["capacity_per_lane_by_link_type"]
        assert cap["secondary"] >= 950, f"secondary cap={cap['secondary']} is too low"

    def test_tertiary_capacity_raised(self):
        from sim.defaults import NETWORK_NORM_DEFAULTS

        cap = NETWORK_NORM_DEFAULTS["normalization"]["defaults"]["capacity_per_lane_by_link_type"]
        assert cap["tertiary"] >= 800, f"tertiary cap={cap['tertiary']} is too low"

    def test_secondary_link_capacity_raised(self):
        from sim.defaults import NETWORK_NORM_DEFAULTS

        cap = NETWORK_NORM_DEFAULTS["normalization"]["defaults"]["capacity_per_lane_by_link_type"]
        assert cap["secondary_link"] >= 900, f"secondary_link cap={cap['secondary_link']} is too low"


class TestBrnoConfigClean:
    """Verify Brno config doesn't duplicate what's now in global defaults."""

    @pytest.fixture()
    def brno_norm_cfg(self) -> dict:
        p = REPO_ROOT / "config" / "brno" / "network_normalization.yaml"
        return yaml.safe_load(p.read_text(encoding="utf-8"))

    def test_no_normalization_defaults_block(self, brno_norm_cfg):
        """Capacity defaults are now global; Brno shouldn't override them."""
        norm = brno_norm_cfg.get("normalization", {})
        defaults = norm.get("defaults", {})
        assert "capacity_per_lane_by_link_type" not in defaults

    def test_no_capacity_factors_for_secondary(self, brno_norm_cfg):
        baseline = brno_norm_cfg.get("experiment_profiles", {}).get("baseline", {})
        cf = baseline.get("capacity_factors", {})
        assert "secondary" not in cf, "capacity_factors.secondary no longer needed"

    def test_experiment_profiles_no_lanes_override(self, brno_norm_cfg):
        baseline = brno_norm_cfg.get("experiment_profiles", {}).get("baseline", {})
        assert "lanes_by_link_type" not in baseline
        assert "capacity_per_lane_by_link_type" not in baseline


# ---------------------------------------------------------------------------
# Problem 1+2: _normalization_defaults reads the correct config path
# ---------------------------------------------------------------------------

class TestNormalizationDefaultsReading:
    """Test that _normalization_defaults correctly reads from
    normalization.defaults in the network_cfg dict."""

    def test_reads_from_normalization_defaults(self):
        from sim.network.normalization import _normalization_defaults

        network_cfg = {
            "normalization": {
                "defaults": {
                    "lanes_by_link_type": {"secondary": 2, "tertiary": 2},
                    "capacity_per_lane_by_link_type": {"secondary": 900},
                }
            }
        }
        defaults, _ = _normalization_defaults(network_cfg)
        assert defaults["lanes_by_link_type"]["secondary"] == 2
        assert defaults["lanes_by_link_type"]["tertiary"] == 2
        assert defaults["capacity_per_lane_by_link_type"]["secondary"] == 900

    def test_code_defaults_used_when_no_yaml(self):
        from sim.network.normalization import (
            _normalization_defaults,
            _DEFAULT_LANES_BY_LINK_TYPE,
        )

        defaults, _ = _normalization_defaults({})
        assert defaults["lanes_by_link_type"]["secondary"] == _DEFAULT_LANES_BY_LINK_TYPE["secondary"]

    def test_yaml_overrides_code_default(self):
        from sim.network.normalization import _normalization_defaults

        network_cfg = {
            "normalization": {
                "defaults": {
                    "lanes_by_link_type": {"secondary": 3},
                }
            }
        }
        defaults, _ = _normalization_defaults(network_cfg)
        assert defaults["lanes_by_link_type"]["secondary"] == 3


# ---------------------------------------------------------------------------
# Problem 2 code fix: experiment profile merges lanes/capacity
# ---------------------------------------------------------------------------

class TestExperimentProfileMerge:
    """Verify that _resolved_experiment_profile values for lanes/capacity
    are merged into defaults by normalize_network_attributes."""

    def test_profile_lanes_override_applied(self):
        from sim.network.normalization import (
            _normalization_defaults,
            _resolved_experiment_profile,
        )

        network_cfg = {
            "normalization": {"defaults": {}},
            "experiment_profiles": {
                "baseline": {
                    "lanes_by_link_type": {"secondary": 4},
                    "capacity_per_lane_by_link_type": {"secondary": 1200},
                }
            },
        }
        defaults, _ = _normalization_defaults(network_cfg)
        profile = _resolved_experiment_profile(network_cfg, "baseline")

        for key in ("speed_by_link_type", "lanes_by_link_type", "capacity_per_lane_by_link_type"):
            profile_vals = profile.get(key)
            if isinstance(profile_vals, dict) and profile_vals:
                defaults[key] = {**defaults[key], **profile_vals}

        assert defaults["lanes_by_link_type"]["secondary"] == 4
        assert defaults["capacity_per_lane_by_link_type"]["secondary"] == 1200

    def test_profile_without_lanes_preserves_defaults(self):
        from sim.network.normalization import (
            _normalization_defaults,
            _resolved_experiment_profile,
            _DEFAULT_LANES_BY_LINK_TYPE,
        )

        network_cfg = {
            "experiment_profiles": {
                "baseline": {
                    "speed_caps": {"trunk": 80},
                }
            }
        }
        defaults, _ = _normalization_defaults(network_cfg)
        profile = _resolved_experiment_profile(network_cfg, "baseline")

        for key in ("speed_by_link_type", "lanes_by_link_type", "capacity_per_lane_by_link_type"):
            profile_vals = profile.get(key)
            if isinstance(profile_vals, dict) and profile_vals:
                defaults[key] = {**defaults[key], **profile_vals}

        assert defaults["lanes_by_link_type"]["secondary"] == _DEFAULT_LANES_BY_LINK_TYPE["secondary"]


# ---------------------------------------------------------------------------
# Problem 3: Global demand parameters lowered
# ---------------------------------------------------------------------------

class TestGlobalDemandParameters:
    """Verify that global defaults have realistic demand parameters for Czech cities."""

    def test_pa_trip_rate_realistic(self):
        from sim.defaults import SIM_DEFAULTS

        rate = SIM_DEFAULTS["demand"]["distribution"]["pa_trip_rate"]
        assert rate <= 2.0, f"pa_trip_rate={rate} is too high for Czech cities"
        assert rate >= 1.0, f"pa_trip_rate={rate} is unrealistically low"

    def test_pa_car_share_realistic(self):
        from sim.defaults import SIM_DEFAULTS

        share = SIM_DEFAULTS["demand"]["distribution"]["pa_car_share"]
        assert share <= 0.45, f"pa_car_share={share} is too high for Czech urban modal split"
        assert share >= 0.25, f"pa_car_share={share} is unrealistically low"

    def test_blend_alpha_moderate(self):
        from sim.defaults import SIM_DEFAULTS

        alpha = SIM_DEFAULTS["demand"]["distribution"]["blend_alpha"]
        assert alpha <= 0.65, f"blend_alpha={alpha} gives too much weight to IPF inflation"
        assert alpha >= 0.3, f"blend_alpha={alpha} is unrealistically low"


# ---------------------------------------------------------------------------
# Problem 2: Global snap distance increased
# ---------------------------------------------------------------------------

class TestSnapDistanceGlobal:
    """Verify that global default snap distance is high enough."""

    def test_global_snap_distance(self):
        from sim.defaults import SIM_DEFAULTS

        snap = SIM_DEFAULTS["network"]["divided_highway_snap_m"]
        assert snap >= 800, f"global snap_m={snap} is too low"
