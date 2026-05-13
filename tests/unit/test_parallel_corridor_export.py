"""parallel_corridor diagnostics: merge assignment parquet with links DB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from sim.assignment.config import _apply_bpr_defaults
from sim.diagnostics.parallel_corridor import build_parallel_corridor_table


def _tiny_sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE links ("
        "link_id INTEGER, link_type TEXT, direction INTEGER, distance REAL, modes TEXT, "
        "speed_ab REAL, travel_time_ab REAL, capacity_ab REAL, lanes_ab INTEGER, "
        "osm_id INTEGER, osm_ref_norm TEXT)"
    )
    conn.execute(
        "INSERT INTO links VALUES "
        "(1,'motorway',1,1000.0,'c',100,60,4000,2,111,'D0_x'),"
        "(2,'motorway',1,1000.0,'c',100,60,4000,2,111,'D0_x')"
    )
    conn.commit()
    conn.close()


def test_build_parallel_corridor_table_merges_volumes(tmp_path):
    db = tmp_path / "project_database.sqlite"
    _tiny_sqlite(db)
    pq = tmp_path / "assignment_results.parquet"
    pd.DataFrame(
        {
            "link_id": [1, 2],
            "PCE_tot": [500.0, 0.0],
            "VOC_max": [0.5, 0.0],
            "Delay_factor_Max": [1.01, 1.0],
            "wd_daily_local_tot": [400.0, 0.0],
            "wd_daily_external_through_tot": [100.0, 0.0],
        }
    ).to_parquet(pq, index=False)

    presets = {
        "corridors": [
            {
                "id": "test_mw_pair",
                "validation_screenline": "sl_x",
                "description": "unit test",
                "link_ids": [1, 2],
            }
        ]
    }
    bpr = _apply_bpr_defaults({})
    df = build_parallel_corridor_table(
        presets=presets,
        assignment_parquet=pq,
        project_db=db,
        bpr_merged=bpr,
    )
    assert len(df) == 2
    assert set(df["link_id"]) == {1, 2}
    assert float(df.loc[df["link_id"] == 1, "PCE_tot"].iloc[0]) == pytest.approx(500.0)
    assert "bpr_implied_from_VOC_max" in df.columns


def test_corridor_filter(tmp_path):
    db = tmp_path / "p.sqlite"
    _tiny_sqlite(db)
    pq = tmp_path / "a.parquet"
    pd.DataFrame(
        {
            "link_id": [1],
            "PCE_tot": [1.0],
            "VOC_max": [0.1],
            "Delay_factor_Max": [1.0],
        }
    ).to_parquet(pq, index=False)
    presets = {
        "corridors": [
            {"id": "keep", "link_ids": [1], "validation_screenline": None, "description": ""},
            {"id": "drop", "link_ids": [2], "validation_screenline": None, "description": ""},
        ]
    }
    bpr = _apply_bpr_defaults({})
    df = build_parallel_corridor_table(
        presets=presets,
        assignment_parquet=pq,
        project_db=db,
        bpr_merged=bpr,
        corridor_filter="keep",
    )
    assert len(df) == 1
    assert df.iloc[0]["corridor_id"] == "keep"
