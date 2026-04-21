"""Integration / e2e test for the ``build-network`` pipeline step.

Run locally with::

    RUN_BUILD_NETWORK_E2E=1 python -m pytest tests/integration/ -v -m integration

The test is skipped by default because it needs network access (Overpass API)
and heavy dependencies (aequilibrae, osmnx).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SKIP_REASON = "Set RUN_BUILD_NETWORK_E2E=1 to run build-network e2e tests"


def _should_skip() -> bool:
    return os.environ.get("RUN_BUILD_NETWORK_E2E", "").strip() not in ("1", "true", "yes")


@pytest.fixture()
def build_network_output(minimal_sim_yaml: Path, tmp_path: Path):
    """Run ``python run.py --config <yaml> build-network`` and return paths."""
    if _should_skip():
        pytest.skip(_SKIP_REASON)
    pytest.importorskip("aequilibrae")
    pytest.importorskip("osmnx")

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "--config", str(minimal_sim_yaml), "build-network"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"build-network exited with code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )

    import yaml
    cfg = yaml.safe_load(minimal_sim_yaml.read_text(encoding="utf-8"))
    return {
        "project_dir": Path(cfg["project_path"]),
        "maps_dir": Path(cfg["network"]["maps_dir"]),
        "stdout": result.stdout,
    }


class TestBuildNetworkCLI:
    def test_project_database_created(self, build_network_output):
        db = build_network_output["project_dir"] / "project_database.sqlite"
        assert db.exists(), f"Expected project DB at {db}"
        assert db.stat().st_size > 0

    def test_network_counts_json(self, build_network_output):
        counts_file = build_network_output["maps_dir"] / "network_counts.json"
        assert counts_file.exists(), f"Expected {counts_file}"
        data = json.loads(counts_file.read_text(encoding="utf-8"))
        assert data["links"] > 0, "Network should contain at least 1 link"
        assert data["nodes"] > 0, "Network should contain at least 1 node"

    def test_geojson_exports(self, build_network_output):
        maps = build_network_output["maps_dir"]
        for name in ("links_native.geojson", "model_bbox_native.geojson"):
            p = maps / name
            assert p.exists(), f"Missing export: {p}"
            assert p.stat().st_size > 100, f"Export suspiciously small: {p}"

    def test_png_map_created(self, build_network_output):
        png = build_network_output["maps_dir"] / "links_native.png"
        assert png.exists(), f"Expected PNG at {png}"
        assert png.stat().st_size > 1000

    def test_stdout_mentions_done(self, build_network_output):
        assert "NETWORK BUILD DONE" in build_network_output["stdout"]
