"""Quick visual checks after each step."""
from __future__ import annotations

from pathlib import Path

from sim.io_project import load_config


def viz_step1_project_bootstrap(config_path: str | Path = "config/sim.yaml") -> None:
    cfg = load_config(config_path)

    print("=== SIM BOOTSTRAP CHECK ===")
    print("project_path:", cfg.get("project_path"))
    print("crs_epsg:", cfg.get("crs_epsg"))

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    print("\nOutputs tree:")
    for p in sorted(out.rglob("*")):
        print(" -", p)

if __name__ == "__main__":
    viz_step1_project_bootstrap()
