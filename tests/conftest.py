"""Shared fixtures for the simulation test suite."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def minimal_sim_yaml(tmp_path: Path) -> Path:
    """Write a minimal sim.yaml that points all outputs into *tmp_path*.

    Uses ``model_bbox`` (small WGS84 rectangle near Brno centre) so the
    build-network step downloads only a tiny OSM extract and avoids the
    geocode / place_name branch.
    """
    project_dir = tmp_path / "project" / "test_aeq"
    maps_dir = tmp_path / "outputs" / "maps"
    network_dir = tmp_path / "outputs" / "network"

    cfg = {
        "project_path": str(project_dir),
        "crs_epsg": 5514,
        "model_bbox": [16.600, 49.190, 16.620, 49.200],
        "network": {
            "maps_dir": str(maps_dir),
            "output_dir": str(network_dir),
            "drivable_network": {"enabled": True, "require_mode_car": True},
            "isolated_components": {"enabled": True},
        },
    }

    yaml_path = tmp_path / "sim.yaml"
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    return yaml_path
