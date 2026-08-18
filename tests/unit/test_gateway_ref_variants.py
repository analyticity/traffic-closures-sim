"""Road-ref token expansion for gateway whitelist matching.

Czech class II/III roads are mostly tagged in OSM with a bare number
(``ref=602``), while the config is written in the readable form (``II/602``).
The expansion has to bridge that, without changing the pre-existing I/D
behaviour that the D1 / I/43 / I/52 gateways depend on.
"""

from sim.zoning.gateways import norm_text, road_ref_token_variants


def _variants(token: str) -> set[str]:
    return road_ref_token_variants(token, norm_text(token))


# --- pre-existing behaviour must not change --------------------------------

def test_motorway_ref_still_expands_to_numeric_and_i_form():
    assert _variants("D1") == {"D1", "1", "I1"}


def test_first_class_ref_still_expands_to_numeric_and_d_form():
    assert _variants("I/43") == {"I43", "43", "D43"}
    assert _variants("I/52") == {"I52", "52", "D52"}


def test_bare_number_stays_alone():
    # An unclassed token must not suddenly match I/602, D602, ...
    assert _variants("602") == {"602"}


# --- class II / III / R ----------------------------------------------------

def test_second_class_ref_expands_to_bare_number():
    assert _variants("II/602") == {"II602", "602"}
    assert _variants("II/380") == {"II380", "380"}
    assert _variants("II/430") == {"II430", "430"}


def test_third_class_ref_expands_to_bare_number():
    # "III" has to be matched before "II", otherwise the prefix strip is wrong.
    assert _variants("III/41614") == {"III41614", "41614"}


def test_expressway_ref_expands_to_bare_number():
    assert _variants("R52") == {"R52", "52"}


def test_second_class_does_not_leak_into_first_class_form():
    # II/602 must not produce I602 / D602 — those are different roads.
    assert "I602" not in _variants("II/602")
    assert "D602" not in _variants("II/602")


# --- robustness ------------------------------------------------------------

def test_separators_and_case_are_normalised():
    assert _variants("ii / 602") == _variants("II/602")
    assert _variants("II-602") == _variants("II/602")


def test_empty_token_yields_nothing():
    assert _variants("") == set()


def test_non_numeric_suffix_is_left_alone():
    # No digits after the class prefix -> nothing to strip, no crash.
    assert _variants("IIabc") == {"IIABC"}
