"""Multi-class matrix core gating (matches execute_assignment behaviour)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sim.assignment.config import (
    _resolve_multi_class,
    multiclass_matrix_core_status,
)


def test_multiclass_enabled_when_cores_present():
    mat_cores = ["wd_daily", "wd_daily_local", "wd_daily_external_through", "other"]
    mc = _resolve_multi_class({"enabled": True})
    ok, req, miss = multiclass_matrix_core_status(mat_cores, mc)
    assert ok
    assert req == ["wd_daily_local", "wd_daily_external_through"]
    assert miss == []


def test_multiclass_disabled_when_core_missing():
    mat_cores = ["wd_daily", "wd_daily_local"]
    mc = _resolve_multi_class({"enabled": True})
    ok, req, miss = multiclass_matrix_core_status(mat_cores, mc)
    assert not ok
    assert "wd_daily_external_through" in miss


def test_multiclass_disabled_when_config_off():
    mat_cores = ["wd_daily_local"]
    mc = _resolve_multi_class({"enabled": False})
    ok, req, miss = multiclass_matrix_core_status(mat_cores, mc)
    assert not ok
    assert req == []


@pytest.mark.skipif(
    not Path("data/brno/demand/od_matrix.aem").exists(),
    reason="Brno OD matrix fixture not present",
)
def test_brno_matrix_file_has_multiclass_cores():
    from aequilibrae.matrix import AequilibraeMatrix

    mat = AequilibraeMatrix()
    try:
        mat.load(str(Path("data/brno/demand/od_matrix.aem")))
        names = [str(x) for x in mat.names]
    finally:
        mat.close()
    mc = _resolve_multi_class({"enabled": True})
    ok, req, miss = multiclass_matrix_core_status(names, mc)
    assert ok, f"missing cores {miss}; required {req}"
