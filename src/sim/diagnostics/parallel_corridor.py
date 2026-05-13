"""Join assignment link volumes with DB supply fields for parallel-corridor studies."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from sim.assignment.config import _apply_bpr_defaults
from sim.io_project import load_config


def bpr_delay_multiplier(v_over_c: float, alpha: float, beta: float = 4.0) -> float:
    """BPR travel-time multiplier ``1 + alpha * (V/C)^beta`` with non-negative V/C."""
    x = max(float(v_over_c), 0.0)
    return 1.0 + float(alpha) * (x ** float(beta))


def daily_capacity_factor_for_link_type(link_type: str, bpr_merged: Dict[str, Any]) -> float:
    """Scalar DCF applied in ``build_graph`` for *link_type* (matches graph.py)."""
    dcf_raw = bpr_merged.get("daily_capacity_factor", 10.0)
    if isinstance(dcf_raw, dict):
        default = float(dcf_raw.get("default", 10.0))
        by_lt: Dict[str, Any] = dcf_raw.get("by_link_type") or {}
        return float(by_lt.get(str(link_type), default))
    return float(dcf_raw)


def bpr_alpha_beta_for_link_type(link_type: str, bpr_merged: Dict[str, Any]) -> tuple[float, float]:
    by_lt = bpr_merged.get("by_link_type") or {}
    a_def = float(bpr_merged.get("alpha_default", 0.85))
    b_def = float(bpr_merged.get("beta_default", 4.0))
    row = by_lt.get(str(link_type)) or {}
    return float(row.get("alpha", a_def)), float(row.get("beta", b_def))


def load_presets_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "corridors" not in raw:
        raise ValueError(f"Invalid presets YAML {path}: top-level 'corridors' required")
    return raw


def load_links_supply_frame(db_path: Path, link_ids: Iterable[int]) -> pd.DataFrame:
    ids = sorted({int(x) for x in link_ids})
    if not ids:
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    try:
        qmarks = ",".join("?" * len(ids))
        sql = (
            "SELECT link_id, link_type, direction, distance, modes, "
            "speed_ab, travel_time_ab, capacity_ab, lanes_ab, osm_id, osm_ref_norm "
            f"FROM links WHERE link_id IN ({qmarks})"
        )
        return pd.read_sql_query(sql, conn, params=ids)
    finally:
        conn.close()


def build_parallel_corridor_table(
    *,
    presets: Dict[str, Any],
    assignment_parquet: Path,
    project_db: Path,
    bpr_merged: Dict[str, Any],
    corridor_filter: Optional[str] = None,
) -> pd.DataFrame:
    """One row per link per corridor preset, merged with assignment volumes."""
    rows: List[Dict[str, Any]] = []
    assign_df = pd.read_parquet(assignment_parquet)
    if "link_id" not in assign_df.columns:
        raise ValueError("assignment_results.parquet must contain 'link_id'")

    assign_cols = [
        c
        for c in (
            "link_id",
            "PCE_tot",
            "VOC_max",
            "Delay_factor_Max",
            "wd_daily_local_tot",
            "wd_daily_external_through_tot",
        )
        if c in assign_df.columns
    ]
    assign_sub = assign_df[assign_cols].copy()

    for cor in presets["corridors"]:
        cid = str(cor["id"])
        if corridor_filter and cid != corridor_filter:
            continue
        link_ids = [int(x) for x in cor.get("link_ids", [])]
        supply = load_links_supply_frame(project_db, link_ids)
        if supply.empty:
            continue
        merged = supply.merge(
            assign_sub[assign_sub["link_id"].isin(link_ids)],
            on="link_id",
            how="left",
        )
        for _, r in merged.iterrows():
            lt = str(r.get("link_type") or "")
            dcf = daily_capacity_factor_for_link_type(lt, bpr_merged)
            cap_h = float(r.get("capacity_ab") or 0.0)
            cap_day = cap_h * dcf
            alpha, beta = bpr_alpha_beta_for_link_type(lt, bpr_merged)
            voc = float(r.get("VOC_max", 0.0) or 0.0)
            implied = bpr_delay_multiplier(voc, alpha, beta)
            rows.append(
                {
                    "corridor_id": cid,
                    "validation_screenline": cor.get("validation_screenline"),
                    "corridor_note": cor.get("description"),
                    "link_id": int(r["link_id"]),
                    "link_type": lt,
                    "osm_id": r.get("osm_id"),
                    "osm_ref_norm": r.get("osm_ref_norm"),
                    "direction": int(r.get("direction", 0) or 0),
                    "distance_m": float(r.get("distance") or 0.0),
                    "lanes_ab": r.get("lanes_ab"),
                    "speed_ab_kmh": float(r.get("speed_ab") or 0.0),
                    "travel_time_ab_s": float(r.get("travel_time_ab") or 0.0),
                    "capacity_ab_hourly": cap_h,
                    "daily_capacity_factor": dcf,
                    "capacity_post_dcf_vph_equiv": cap_day,
                    "bpr_alpha": alpha,
                    "bpr_beta": beta,
                    "PCE_tot": float(r.get("PCE_tot", 0.0) or 0.0),
                    "wd_daily_local_tot": float(r.get("wd_daily_local_tot", 0.0) or 0.0),
                    "wd_daily_external_through_tot": float(r.get("wd_daily_external_through_tot", 0.0) or 0.0),
                    "VOC_max": voc,
                    "Delay_factor_Max_model": float(r.get("Delay_factor_Max", 0.0) or 0.0),
                    "bpr_implied_from_VOC_max": implied,
                }
            )
    return pd.DataFrame(rows)


def export_parallel_corridor_csv(
    config_path: Path,
    *,
    presets_path: Path,
    assignment_parquet: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    corridor_filter: Optional[str] = None,
) -> Path:
    """Load sim config, presets, and parquet; write merged CSV. Returns output path."""
    cfg = load_config(str(config_path))
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    out_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = assignment_parquet or (out_dir / "assignment_results.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Assignment parquet not found: {parquet_path}")
    db_path = project_dir / "project_database.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Project database not found: {db_path}")

    assign_cfg = cfg.get("assignment") or {}
    bpr_merged = _apply_bpr_defaults(assign_cfg.get("bpr") or {})
    presets = load_presets_yaml(presets_path)
    df = build_parallel_corridor_table(
        presets=presets,
        assignment_parquet=parquet_path,
        project_db=db_path,
        bpr_merged=bpr_merged,
        corridor_filter=corridor_filter,
    )
    out_path = output_csv or (out_dir / "parallel_corridor_diagnostics.csv")
    df.to_csv(out_path, index=False)
    meta = {
        "config": str(config_path),
        "presets": str(presets_path),
        "assignment_parquet": str(parquet_path),
        "n_rows": int(len(df)),
    }
    (out_dir / "parallel_corridor_diagnostics_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
