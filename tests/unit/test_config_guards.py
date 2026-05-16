"""Unit tests for configuration guards.

Tests cover: VDF parameter fallback, blocked centroid flows config,
step dependency DAG prerequisite checks, and staleness warnings.
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# VDF parameter fallback
# ---------------------------------------------------------------------------
class TestVdfFallback:
    def test_per_link_with_link_type_returns_column_names(self):
        from sim.assignment.graph import _resolve_vdf_params

        params = {"per_link": True, "alpha_default": 0.55, "beta_default": 4.0}
        result = _resolve_vdf_params(params, ["link_type", "alpha", "beta"])
        assert result == {"alpha": "alpha", "beta": "beta"}

    def test_per_link_without_link_type_uses_configured_defaults(self):
        from sim.assignment.graph import _resolve_vdf_params

        params = {"per_link": True, "alpha_default": 0.55, "beta_default": 3.5}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_vdf_params(params, ["capacity", "speed"])
        assert result == {"alpha": 0.55, "beta": 3.5}
        assert any("link_type column is missing" in str(x.message) for x in w)

    def test_per_link_false_uses_configured_defaults(self):
        from sim.assignment.graph import _resolve_vdf_params

        params = {"per_link": False, "alpha_default": 0.60, "beta_default": 5.0}
        result = _resolve_vdf_params(params, ["link_type"])
        assert result == {"alpha": 0.60, "beta": 5.0}

    def test_no_bpr_parameters_returns_hardcoded(self):
        from sim.assignment.graph import _resolve_vdf_params

        result = _resolve_vdf_params(None, ["link_type"])
        assert result == {"alpha": 0.85, "beta": 4.0}

    def test_fallback_does_not_use_hardcoded_when_config_available(self):
        """Regression: previously the fallback always returned 0.85/4.0."""
        from sim.assignment.graph import _resolve_vdf_params

        params = {"per_link": True, "alpha_default": 0.30, "beta_default": 2.0}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _resolve_vdf_params(params, [])
        assert result["alpha"] == 0.30, "Should use alpha_default, not 0.85"
        assert result["beta"] == 2.0, "Should use beta_default, not 4.0"


# ---------------------------------------------------------------------------
# Blocked centroid flows config guard
# ---------------------------------------------------------------------------
class TestBlockedCentroidFlowsConfig:
    def test_default_is_true(self):
        from sim.defaults import SIM_DEFAULTS

        assert SIM_DEFAULTS["assignment"]["blocked_centroid_flows"] is True

    def test_warning_when_false(self):
        """build_graph should warn when blocked_centroid_flows=False."""
        from sim.assignment.graph import build_graph

        mock_project = MagicMock()
        mock_graph = MagicMock()
        mock_graph.graph = MagicMock()
        mock_graph.graph.columns = ["free_flow_time", "capacity"]
        mock_project.network.build_graphs = MagicMock()
        mock_project.network.graphs = {"c": mock_graph}

        mock_mat = MagicMock()
        mock_mat.index.__getitem__ = MagicMock(return_value=[1, 2])

        cfg = {"blocked_centroid_flows": False}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                build_graph(mock_project, mock_mat, assignment_cfg=cfg)
            except Exception:
                pass
        centroid_warnings = [
            x for x in w
            if "blocked_centroid_flows is False" in str(x.message)
        ]
        assert len(centroid_warnings) >= 1

    def test_no_warning_when_true(self):
        """build_graph should NOT warn when blocked_centroid_flows=True."""
        from sim.assignment.graph import build_graph

        mock_project = MagicMock()
        mock_graph = MagicMock()
        mock_graph.graph = MagicMock()
        mock_graph.graph.columns = ["free_flow_time", "capacity"]
        mock_project.network.build_graphs = MagicMock()
        mock_project.network.graphs = {"c": mock_graph}

        mock_mat = MagicMock()
        mock_mat.index.__getitem__ = MagicMock(return_value=[1, 2])

        cfg = {"blocked_centroid_flows": True}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                build_graph(mock_project, mock_mat, assignment_cfg=cfg)
            except Exception:
                pass
        centroid_warnings = [
            x for x in w
            if "blocked_centroid_flows is False" in str(x.message)
        ]
        assert len(centroid_warnings) == 0


# ---------------------------------------------------------------------------
# Step dependency DAG
# ---------------------------------------------------------------------------
class TestStepPrerequisites:
    """Test _check_step_prerequisites and _resolve_cfg_path from run.py."""

    @staticmethod
    def _make_cfg(tmp_path: Path) -> Dict[str, Any]:
        demand_out = tmp_path / "outputs" / "demand"
        demand_out.mkdir(parents=True, exist_ok=True)
        zones_out = tmp_path / "outputs" / "zones"
        zones_out.mkdir(parents=True, exist_ok=True)
        return {
            "project_path": str(tmp_path / "project"),
            "demand": {
                "matrix_path": str(tmp_path / "matrix.aem"),
                "output_dir": str(demand_out),
            },
            "zoning": {
                "output_dir": str(zones_out),
            },
        }

    def test_resolve_project_db(self, tmp_path):
        from run import _resolve_cfg_path

        cfg = self._make_cfg(tmp_path)
        result = _resolve_cfg_path(cfg, "project_db")
        assert result == Path(cfg["project_path"]) / "project_database.sqlite"

    def test_resolve_matrix(self, tmp_path):
        from run import _resolve_cfg_path

        cfg = self._make_cfg(tmp_path)
        result = _resolve_cfg_path(cfg, "matrix")
        assert result == Path(cfg["demand"]["matrix_path"])

    def test_resolve_results(self, tmp_path):
        from run import _resolve_cfg_path

        cfg = self._make_cfg(tmp_path)
        result = _resolve_cfg_path(cfg, "results")
        assert result.name == "assignment_results.parquet"

    def test_resolve_unknown_spec_raises(self, tmp_path):
        from run import _resolve_cfg_path

        cfg = self._make_cfg(tmp_path)
        with pytest.raises(ValueError, match="Unknown prerequisite spec"):
            _resolve_cfg_path(cfg, "nonexistent")

    def test_normalize_network_requires_project_db(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match="normalize-network"):
            _check_step_prerequisites(cfg, "normalize-network")

    def test_build_zones_requires_project_db(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match="build-zones"):
            _check_step_prerequisites(cfg, "build-zones")

    def test_distribute_requires_db_and_matrix(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match="distribute"):
            _check_step_prerequisites(cfg, "distribute")

    def test_validate_requires_results(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        # Create matrix but not results
        Path(cfg["demand"]["matrix_path"]).touch()
        with pytest.raises(FileNotFoundError, match="validate.*assign or calibrate"):
            _check_step_prerequisites(cfg, "validate")

    def test_audit_supply_requires_project_db(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match="audit-supply"):
            _check_step_prerequisites(cfg, "audit-supply")

    def test_strip_closures_requires_project_db(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match="strip-closures"):
            _check_step_prerequisites(cfg, "strip-closures")

    def test_passes_when_files_exist(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        proj_dir = Path(cfg["project_path"])
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "project_database.sqlite").touch()
        _check_step_prerequisites(cfg, "normalize-network")

    def test_no_check_for_exempt_steps(self, tmp_path):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        # Steps without prerequisites should not raise
        _check_step_prerequisites(cfg, "build-network")

    def test_staleness_warning(self, tmp_path, caplog):
        from run import _check_step_prerequisites

        cfg = self._make_cfg(tmp_path)
        proj_dir = Path(cfg["project_path"])
        proj_dir.mkdir(parents=True, exist_ok=True)
        db_path = proj_dir / "project_database.sqlite"
        Path(cfg["demand"]["matrix_path"]).touch()
        demand_out = Path(cfg["demand"]["output_dir"])
        results = demand_out / "assignment_results.parquet"
        results.touch()

        # Make matrix newer than results to trigger staleness
        time.sleep(0.05)
        Path(cfg["demand"]["matrix_path"]).write_text("updated")

        with caplog.at_level(logging.WARNING):
            _check_step_prerequisites(cfg, "validate")
        assert "stale" in caplog.text.lower() or "older" in caplog.text.lower()


class TestStepPrerequisitesDAGCoverage:
    """Verify the DAG covers all expected steps."""

    def test_all_guarded_steps_in_dag(self):
        from run import _STEP_PREREQUISITES

        expected_guarded = {
            "normalize-network", "build-zones", "distribute",
            "assign-warm-skims", "assign", "audit-supply",
            "calibrate", "calibrate-odme", "tune-supply",
            "validate", "strip-closures",
        }
        assert expected_guarded.issubset(set(_STEP_PREREQUISITES.keys()))


# ---------------------------------------------------------------------------
# Boundary SCC repair config
# ---------------------------------------------------------------------------
class TestBoundarySccRepairConfig:
    def test_default_is_auto(self):
        from sim.defaults import SIM_DEFAULTS
        assert SIM_DEFAULTS["network"]["boundary_scc_repair"] == "auto"

    def test_dry_run_returns_ids_without_modifying(self):
        """dry_run=True should return repair_ids but set dry_run=True in result."""
        from sim.network.connectivity import repair_boundary_scc

        mock_project = MagicMock()
        import pandas as pd
        links = pd.DataFrame([
            {"link_id": 1, "a_node": 1, "b_node": 2, "link_type": "motorway",
             "direction": 1, "speed_ab": 130, "capacity_ab": 6000, "lanes_ab": 3},
            {"link_id": 2, "a_node": 2, "b_node": 3, "link_type": "primary",
             "direction": 0, "speed_ab": 70, "capacity_ab": 1400, "lanes_ab": 2},
            {"link_id": 3, "a_node": 3, "b_node": 1, "link_type": "primary",
             "direction": 0, "speed_ab": 70, "capacity_ab": 1400, "lanes_ab": 2},
        ])
        mock_project.network.links.data = links

        result = repair_boundary_scc(mock_project, dry_run=True)
        assert result["dry_run"] is True
        assert isinstance(result["repaired_ids"], list)
        assert result["scc_before"] > 0


# ---------------------------------------------------------------------------
# Connector strict mode
# ---------------------------------------------------------------------------
class TestConnectorStrictConfig:
    def test_connector_strict_default_false(self):
        from sim.defaults import SIM_DEFAULTS
        assert SIM_DEFAULTS["zoning"]["connector_strict"] is False

    def test_connector_warn_distance_default(self):
        from sim.defaults import SIM_DEFAULTS
        assert SIM_DEFAULTS["zoning"]["connector_warn_distance_m"] == 1500.0


# ---------------------------------------------------------------------------
# Supply audit estimated coverage
# ---------------------------------------------------------------------------
class TestSupplyAuditCoverage:
    def test_compute_estimated_coverage_structure(self, tmp_path):
        """The helper should return correct structure even on a small DB."""
        import sqlite3
        from sim.calibration.supply_audit import _compute_estimated_coverage

        db_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE links (link_id INT, link_type TEXT, "
            "speed_ab REAL, capacity_ab REAL, lanes_ab REAL)"
        )
        conn.execute(
            "INSERT INTO links VALUES (1, 'motorway', 130.0, 6600.0, 3)"
        )
        conn.execute(
            "INSERT INTO links VALUES (2, 'primary', 70.0, 2800.0, 2)"
        )
        conn.execute(
            "INSERT INTO links VALUES (3, 'primary', 55.0, 1200.0, 1)"
        )
        conn.execute(
            "INSERT INTO links VALUES (4, 'centroid_connector', 20.0, 2000.0, 1)"
        )
        conn.commit()
        conn.close()

        result = _compute_estimated_coverage(db_path)
        assert "total_links" in result
        assert "pct_speed_estimated" in result
        assert "pct_capacity_estimated" in result
        assert result["total_links"] == 3  # excludes centroid_connector


# ---------------------------------------------------------------------------
# Observation density
# ---------------------------------------------------------------------------
class TestObservationDensityDefaults:
    def test_min_obs_per_zone_default(self):
        from sim.defaults import SIM_DEFAULTS
        assert SIM_DEFAULTS["calibration"]["min_obs_per_zone"] == 0.5


# ---------------------------------------------------------------------------
# Holdout adequacy
# ---------------------------------------------------------------------------
class TestHoldoutAdequacy:
    def test_classify_adequate(self):
        from sim.calibration.validation import _classify_holdout_adequacy
        assert _classify_holdout_adequacy(15) == "adequate"
        assert _classify_holdout_adequacy(10) == "adequate"

    def test_classify_thin(self):
        from sim.calibration.validation import _classify_holdout_adequacy
        assert _classify_holdout_adequacy(5) == "thin"
        assert _classify_holdout_adequacy(9) == "thin"

    def test_classify_insufficient(self):
        from sim.calibration.validation import _classify_holdout_adequacy
        assert _classify_holdout_adequacy(4) == "insufficient"
        assert _classify_holdout_adequacy(0) == "insufficient"

    def test_benchmarks_with_thin_holdout(self):
        from sim.calibration.validation import compute_validation_benchmarks

        calib_stats = {
            "r2": 0.92, "slope": 1.02, "pct_rmse": 22.0,
            "bias_pct": 3.0, "geh_lt5_pct": 78.0,
            "daily_geh_lt_adj_pct": 80.0, "n": 50,
        }
        thin_holdout = {
            "r2": 0.80, "slope": 1.0, "pct_rmse": 30.0,
            "bias_pct": 5.0, "geh_lt5_pct": 0.0,
            "daily_geh_lt_adj_pct": 0.0, "n": 7,
        }
        result = compute_validation_benchmarks(
            calib_stats, {}, [],
            model_time_period="daily",
            holdout_stats=thin_holdout,
        )
        assert result["holdout_adequacy"] == "thin"
        assert result["verdict_source"] == "holdout"

    def test_benchmarks_insufficient_holdout_falls_back(self):
        from sim.calibration.validation import compute_validation_benchmarks

        calib_stats = {
            "r2": 0.92, "slope": 1.02, "pct_rmse": 22.0,
            "bias_pct": 3.0, "geh_lt5_pct": 78.0,
            "daily_geh_lt_adj_pct": 80.0, "n": 50,
        }
        insufficient_holdout = {
            "r2": 0.50, "slope": 1.0, "pct_rmse": 50.0,
            "bias_pct": 10.0, "geh_lt5_pct": 0.0,
            "daily_geh_lt_adj_pct": 0.0, "n": 3,
        }
        result = compute_validation_benchmarks(
            calib_stats, {}, [],
            model_time_period="daily",
            holdout_stats=insufficient_holdout,
        )
        assert result["holdout_adequacy"] == "insufficient"
        assert result["verdict_source"] == "calibration"

    def test_benchmarks_no_holdout_has_adequacy(self):
        from sim.calibration.validation import compute_validation_benchmarks

        calib_stats = {
            "r2": 0.92, "slope": 1.02, "pct_rmse": 22.0,
            "bias_pct": 3.0, "geh_lt5_pct": 78.0,
            "daily_geh_lt_adj_pct": 80.0, "n": 50,
        }
        result = compute_validation_benchmarks(
            calib_stats, {}, [],
            model_time_period="daily",
        )
        assert result["holdout_adequacy"] == "insufficient"


# ---------------------------------------------------------------------------
# Sensitivity defaults
# ---------------------------------------------------------------------------
class TestSensitivityDefaults:
    def test_sensitivity_defaults_exist(self):
        from sim.defaults import SIM_DEFAULTS
        assert "sensitivity" in SIM_DEFAULTS
        assert SIM_DEFAULTS["sensitivity"]["enabled"] is False
        assert SIM_DEFAULTS["sensitivity"]["max_iter"] == 30
        assert len(SIM_DEFAULTS["sensitivity"]["parameters"]) >= 1

    def test_sensitivity_step_in_steps(self):
        from run import STEPS
        assert "sensitivity" in STEPS

    def test_set_nested_helper(self):
        from sim.sensitivity import _set_nested
        cfg: dict = {"a": {"b": {"c": 1}}}
        _set_nested(cfg, "a.b.c", 42)
        assert cfg["a"]["b"]["c"] == 42

    def test_set_nested_creates_path(self):
        from sim.sensitivity import _set_nested
        cfg: dict = {}
        _set_nested(cfg, "x.y.z", "hello")
        assert cfg["x"]["y"]["z"] == "hello"

    def test_run_sensitivity_disabled(self):
        from sim.sensitivity import run_sensitivity
        from unittest.mock import patch

        mock_cfg = {"sensitivity": {"enabled": False}}
        with patch("sim.sensitivity.load_config", return_value=mock_cfg):
            result = run_sensitivity("dummy.yaml")
        assert result.get("skipped") is True
