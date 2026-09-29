"""Unit tests for the counts ODME screenlines may use under ``csd_split``.

A road in the withheld share of the traffic census must not reach any
screenline, or the calibration fits the very counts the validation reports.
"""
from __future__ import annotations

import pandas as pd

from sim.calibration.context import _drop_withheld_roads


class TestDropWithheldRoads:
    def test_removes_every_section_of_a_withheld_road(self):
        csd = pd.DataFrame({"sil": ["41", "41", "D1", "384"], "sv": [1, 2, 3, 4]})
        out = _drop_withheld_roads(csd, {"41", "384"})
        assert list(out["sil"]) == ["D1"]
        assert list(out["sv"]) == [3]

    def test_matches_numeric_road_numbers(self):
        csd = pd.DataFrame({"sil": [41, 52], "sv": [1, 2]})
        out = _drop_withheld_roads(csd, {"41"})
        assert list(out["sil"]) == [52]

    def test_nothing_withheld_keeps_all(self):
        csd = pd.DataFrame({"sil": ["41", "D1"]})
        assert len(_drop_withheld_roads(csd, set())) == 2

    def test_missing_frame_passes_through(self):
        assert _drop_withheld_roads(None, {"41"}) is None

    def test_does_not_modify_input(self):
        csd = pd.DataFrame({"sil": ["41", "D1"]})
        _drop_withheld_roads(csd, {"41"})
        assert list(csd["sil"]) == ["41", "D1"]
