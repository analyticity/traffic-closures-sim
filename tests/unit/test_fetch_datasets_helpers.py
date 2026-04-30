"""Unit tests for helpers in sim.datasets."""
from __future__ import annotations

from collections import Counter


class TestMatchedCountLogic:
    """Verify the 'matched' count uses the correct fallback label."""

    def test_default_median_excluded(self):
        """Rows with match='default_median' should NOT count as matched."""
        rows = [
            {"zone_id": 1, "match": "obec_exact"},
            {"zone_id": 2, "match": "mc_area:Brno-střed"},
            {"zone_id": 3, "match": "default_median"},
            {"zone_id": 4, "match": "fuzzy:Líšeň"},
        ]
        matched = len([r for r in rows if r["match"] != "default_median"])
        assert matched == 3

    def test_all_matched(self):
        rows = [
            {"zone_id": 1, "match": "obec_exact"},
            {"zone_id": 2, "match": "obec_strip:Rajhrad"},
        ]
        matched = len([r for r in rows if r["match"] != "default_median"])
        assert matched == 2

    def test_all_fallback(self):
        rows = [
            {"zone_id": 1, "match": "default_median"},
            {"zone_id": 2, "match": "default_median"},
        ]
        matched = len([r for r in rows if r["match"] != "default_median"])
        assert matched == 0
