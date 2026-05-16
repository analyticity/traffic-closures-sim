"""CSD observed lookup for composite gateway matched_ref (e.g. D1;50)."""

from sim.calibration.screenlines import _csd_observed_from_matched_ref


def test_composite_ref_d1_semicolon_50_uses_d1_csd():
    csd = {
        "D1": {"sv": 47875.0, "o": 40000.0},
        "50": {"sv": 13678.0, "o": 11000.0},
    }
    sv, o = _csd_observed_from_matched_ref("D1;50", csd)
    assert sv == 47875.0
    assert o == 40000.0


def test_single_ref_unchanged():
    csd = {"D1": {"sv": 1000.0, "o": 800.0}}
    sv, o = _csd_observed_from_matched_ref("D1", csd)
    assert sv == 1000.0
    assert o == 800.0


def test_unknown_composite_returns_zero():
    sv, o = _csd_observed_from_matched_ref("D1;50", {})
    assert sv == 0.0
    assert o == 0.0
