"""Unit tests for pure helpers in sim.assignment."""
from __future__ import annotations

import pandas as pd
import pytest

from sim.assignment import (
    _detect_volume_col,
    _resolve_vdf_params,
    _voc_to_los,
)


# ---------------------------------------------------------------------------
# _voc_to_los
# ---------------------------------------------------------------------------
class TestVocToLos:
    def test_a_boundary(self):
        assert _voc_to_los(0.35) == "A"

    def test_b_just_above(self):
        assert _voc_to_los(0.351) == "B"

    def test_b_boundary(self):
        assert _voc_to_los(0.55) == "B"

    def test_c(self):
        assert _voc_to_los(0.70) == "C"

    def test_d(self):
        assert _voc_to_los(0.85) == "D"

    def test_e(self):
        assert _voc_to_los(1.00) == "E"

    def test_f(self):
        assert _voc_to_los(1.01) == "F"

    def test_zero(self):
        assert _voc_to_los(0) == "A"


# ---------------------------------------------------------------------------
# _resolve_vdf_params
# ---------------------------------------------------------------------------
class TestResolveVdfParams:
    def test_per_link_true(self):
        bpr = {"per_link": True, "alpha": 0.15, "beta": 4.0}
        result = _resolve_vdf_params(bpr, ["link_id", "link_type"])
        assert result == {"alpha": "alpha", "beta": "beta"}

    def test_per_link_true_no_link_type_col(self):
        """per_link requested but link_type missing => falls to global default."""
        bpr = {"per_link": True, "alpha": 0.2, "beta": 5.0}
        result = _resolve_vdf_params(bpr, ["link_id"])
        assert result == {"alpha": 0.85, "beta": 4.0}

    def test_per_link_false_custom(self):
        bpr = {"per_link": False, "alpha": 0.2, "beta": 5.0}
        result = _resolve_vdf_params(bpr, [])
        assert result == {"alpha": 0.2, "beta": 5.0}

    def test_fallback_uses_standard_bpr_alpha(self):
        result = _resolve_vdf_params(None, [])
        assert result["alpha"] == pytest.approx(0.85)
        assert result["beta"] == pytest.approx(4.0)

    def test_bpr_with_defaults(self):
        bpr = {"per_link": False, "alpha_default": 0.25}
        result = _resolve_vdf_params(bpr, [])
        assert result["alpha"] == 0.25
        assert result["beta"] == 4.0


# ---------------------------------------------------------------------------
# _detect_volume_col
# ---------------------------------------------------------------------------
class TestDetectVolumeCol:
    def test_finds_nonzero_tot(self):
        df = pd.DataFrame({"local_tot": [0, 0], "through_tot": [10, 20]})
        assert _detect_volume_col(df) == "through_tot"

    def test_prefers_first_nonzero(self):
        df = pd.DataFrame({"a_tot": [1, 0], "b_tot": [0, 1]})
        assert _detect_volume_col(df) == "a_tot"

    def test_all_zero_falls_back(self):
        df = pd.DataFrame({"a_tot": [0, 0], "b_tot": [0, 0]})
        result = _detect_volume_col(df)
        assert result in ("a_tot", "b_tot")

    def test_no_tot_column(self):
        df = pd.DataFrame({"volume": [1, 2], "flow": [3, 4]})
        assert _detect_volume_col(df) is None

    def test_fallback_to_partial_match(self):
        df = pd.DataFrame({"total_flow": [1, 2], "other": [3, 4]})
        assert _detect_volume_col(df) == "total_flow"
