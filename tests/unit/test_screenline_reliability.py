"""Screenline reliability exclusion for benchmarks (ratio band 0.2–5.0)."""
from sim.calibration.screenlines import (
    auto_screenline_unreliable,
    screenline_benchmark_unreliable,
    screenline_excluded_from_benchmark,
)


def test_unreliable_below_threshold():
    assert auto_screenline_unreliable(
        "auto_gw_D55_SE", 740, 20451, 0.036, ratio_below=0.20, ratio_above=5.0,
    )


def test_unreliable_zero_modeled():
    assert auto_screenline_unreliable(
        "auto_gw_570_W", 0, 6196, 0.0, ratio_below=0.20, ratio_above=5.0,
    )


def test_unreliable_above_threshold():
    assert auto_screenline_unreliable(
        "auto_gw_437_E", 18490, 2600, 7.11, ratio_below=0.20, ratio_above=5.0,
    )


def test_reliable_auto_counts():
    assert not auto_screenline_unreliable(
        "auto_gw_D35_NW", 35674, 32417, 1.10, ratio_below=0.20, ratio_above=5.0,
    )


def test_manual_never_auto_unreliable():
    assert not auto_screenline_unreliable(
        "manual_olomouc_pavlovicka", 19663, 6822, 2.88, ratio_below=0.20, ratio_above=5.0,
    )


def test_benchmark_unreliable_applies_to_manual_outside_band():
    assert screenline_benchmark_unreliable(0, 5000, 0.0, ratio_below=0.20, ratio_above=5.0)
    assert screenline_excluded_from_benchmark(
        "manual_brno_test",
        {"modeled_total": 0, "observed_total": 5000, "ratio": 0.0},
    )


def test_benchmark_reliable_manual_inside_band():
    assert not screenline_excluded_from_benchmark(
        "manual_brno_test",
        {"modeled_total": 4500, "observed_total": 5000, "ratio": 0.9},
    )
