"""Traffic assignment: load OD matrix, build graph, run AequilibraE assignment.

Public API
----------
``run_assignment(config_path)``
    CLI entry-point – loads config, runs a single assignment, saves results.

``execute_assignment(project, mat, *, algorithm, …) -> pd.DataFrame``
    Low-level reusable function.  Takes an *open* AequilibraE Project and a
    *loaded* AequilibraeMatrix and returns a link-volume DataFrame.  Used by
    the iterative calibration loop so it can swap matrices between iterations
    without re-opening the project every time.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import TrafficAssignment, TrafficClass

from sim.io_project import load_config


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_connectors(project_dir: Path) -> Dict[str, Any]:
    """Verify that centroid connectors actually reach the road network."""
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    try:
        total_conn = conn.execute(
            "SELECT COUNT(*) FROM links WHERE link_type='centroid_connector'"
        ).fetchone()[0]
        total_cents = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE is_centroid=1"
        ).fetchone()[0]
        good = conn.execute("""
            SELECT COUNT(DISTINCT l.a_node) FROM links l
            WHERE l.link_type='centroid_connector'
            AND EXISTS (
                SELECT 1 FROM links l2
                WHERE (l2.a_node=l.b_node OR l2.b_node=l.b_node)
                AND l2.link_type != 'centroid_connector'
            )
        """).fetchone()[0]
        return {
            "connectors": total_conn,
            "centroids": total_cents,
            "centroids_connected": good,
            "ok": good == total_cents and total_cents > 0,
        }
    finally:
        conn.close()


def fix_node_ids(project_dir: Path) -> int:
    """Renumber nodes > uint32 to small IDs (AequilibraE graph requires uint32)."""
    uint32_max = int(np.iinfo(np.uint32).max)
    db = str(project_dir / "project_database.sqlite")
    conn = sqlite3.connect(db)
    try:
        big = conn.execute(
            "SELECT node_id FROM nodes WHERE node_id > ?", (uint32_max,)
        ).fetchall()
        if not big:
            return 0
        existing = set(r[0] for r in conn.execute(
            "SELECT node_id FROM nodes WHERE node_id <= ?", (uint32_max,)
        ))
        for (old_id,) in big:
            new_id = 1
            while new_id in existing:
                new_id += 1
            conn.execute("UPDATE nodes SET node_id=? WHERE node_id=?", (-new_id, old_id))
            conn.execute("UPDATE links SET a_node=? WHERE a_node=?", (-new_id, old_id))
            conn.execute("UPDATE links SET b_node=? WHERE b_node=?", (-new_id, old_id))
            existing.add(new_id)
        neg = conn.execute("SELECT node_id FROM nodes WHERE node_id < 0").fetchall()
        for (nid,) in neg:
            conn.execute("UPDATE nodes SET node_id=? WHERE node_id=?", (-nid, nid))
            conn.execute("UPDATE links SET a_node=? WHERE a_node=?", (-nid, nid))
            conn.execute("UPDATE links SET b_node=? WHERE b_node=?", (-nid, nid))
        conn.commit()
        return len(big)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Low-level assignment (reusable by calibration loop)
# ---------------------------------------------------------------------------

def execute_assignment(
    project: Project,
    mat: AequilibraeMatrix,
    *,
    algorithm: str = "bfw",
    max_iter: int = 100,
    rgap_target: float = 0.001,
    save_skims: bool = False,
    select_links: Optional[Dict[str, list]] = None,
) -> Tuple[pd.DataFrame, Optional[AequilibraeMatrix], Dict[str, np.ndarray]]:
    """Run assignment on an already-open project with a loaded matrix.

    Returns ``(link_volume_df, skims, select_link_matrices)`` where
    *skims* is an ``AequilibraeMatrix`` when ``save_skims=True``,
    and *select_link_matrices* is ``{screenline_name: OD_array}``
    when ``select_links`` is provided.
    """
    project.network.build_graphs(modes=["c"])
    graph = project.network.graphs["c"]
    graph.prepare_graph(mat.index[:], remove_dead_ends=False)

    gcols = list(graph.graph.columns)
    time_field = next(
        (c for c in ("free_flow_time", "travel_time") if c in gcols), None
    )
    if time_field is None:
        raise RuntimeError(f"No time field in graph. Columns: {gcols}")

    # Apply intersection/signal delay factors to create road class hierarchy.
    # OSM gives all urban roads ~50 km/h (Czech built-up area limit), so
    # without this, secondary/tertiary/residential are equally attractive
    # as primary/trunk for route choice. These factors account for traffic
    # signals, intersections, and urban friction that lower effective speed.
    # Applied to the graph IN MEMORY only -- does NOT modify the DB.
    _DELAY_FACTORS = {
        "motorway": 1.0, "motorway_link": 1.0,
        "trunk": 1.0, "trunk_link": 1.05,
        "primary": 1.15, "primary_link": 1.20,
        "secondary": 1.40, "secondary_link": 1.50,
        "tertiary": 1.60, "tertiary_link": 1.70,
        "residential": 1.80, "unclassified": 1.60,
        "living_street": 2.00, "service": 2.00,
    }
    if "link_type" in graph.network.columns:
        for lt, factor in _DELAY_FACTORS.items():
            if factor <= 1.0:
                continue
            mask = graph.network["link_type"].astype(str) == lt
            if mask.any():
                graph.network.loc[mask, time_field] = (
                    graph.network.loc[mask, time_field] * factor
                )

    graph.set_graph(time_field)
    graph.set_skimming([time_field])
    graph.set_blocked_centroid_flows(True)

    tc = TrafficClass(name="car", graph=graph, matrix=mat)

    if select_links:
        try:
            tc.set_select_links(select_links)
        except Exception:
            pass

    assig = TrafficAssignment()
    assig.set_classes([tc])
    assig.set_vdf("BPR")
    assig.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assig.set_capacity_field("capacity")
    assig.set_time_field(time_field)
    assig.set_algorithm(algorithm)
    assig.max_iter = max_iter
    assig.rgap_target = rgap_target
    assig.set_cores(1)

    assig.execute()

    result = assig.results()
    df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    if "link_id" not in df.columns:
        if df.index.name == "link_id":
            df = df.reset_index()
        else:
            df["link_id"] = graph.graph["link_id"].values[: len(df)]
    elif df.index.name == "link_id":
        df = df.reset_index(drop=True)

    skims = None
    if save_skims:
        try:
            skims = tc.results.skims
        except Exception:
            pass

    sl_matrices: Dict[str, np.ndarray] = {}
    if select_links:
        for sl_name in select_links:
            try:
                sl_matrices[sl_name] = tc.results.select_link_od.matrix[sl_name][:, :].copy()
            except Exception:
                pass

    return df, skims, sl_matrices


def _detect_volume_col(df: pd.DataFrame) -> str | None:
    """Find the first ``*_tot`` column with non-zero sum."""
    return next(
        (c for c in df.columns if c.endswith("_tot") and df[c].sum() > 0),
        next((c for c in df.columns if "tot" in c.lower()), None),
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def run_assignment(config_path: str | Path = "config/sim.yaml") -> None:
    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithm = str(calib_cfg.get("algorithm", "bfw"))
    max_iter = int(calib_cfg.get("max_iter", 100))
    rgap = float(calib_cfg.get("rgap_target", 0.001))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))

    print("=== TRAFFIC ASSIGNMENT ===")

    # Pre-flight
    print("\n1) Pre-flight checks ...")
    renamed = fix_node_ids(project_dir)
    if renamed:
        print(f"   Fixed {renamed} overflow node IDs")
    conn_info = _check_connectors(project_dir)
    print(f"   Connectors: {conn_info['connectors']}, "
          f"centroids: {conn_info['centroids']}, "
          f"connected: {conn_info['centroids_connected']}")
    if not conn_info["ok"]:
        print("   WARNING: not all centroids reach the road network!")

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}")

    # Load matrix
    print(f"\n2) Loading matrix: {matrix_path}")
    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])
    total_demand = float(mat.matrix_view.sum())
    print(f"   Core '{core_name}': {mat.zones} zones, demand={total_demand:,.0f}")

    save_skims = bool(calib_cfg.get("save_skims", False))

    # Run
    print(f"\n3) Running {algorithm.upper()} ...")
    project = Project()
    project.open(str(project_dir))
    try:
        df, skims, _sl = execute_assignment(
            project, mat,
            algorithm=algorithm, max_iter=max_iter, rgap_target=rgap,
            save_skims=save_skims,
        )
    finally:
        project.close()
        mat.close()

    tot_col = _detect_volume_col(df)
    total_vol = float(df[tot_col].sum()) if tot_col else 0.0
    print(f"\n4) Results: {len(df)} links, vol_col={tot_col}, total={total_vol:,.0f}")

    if total_vol <= 0:
        import sqlite3 as _sq
        print("\n   *** ZERO VOLUME — diagnostics ***")
        print(f"   Matrix demand: {total_demand:,.0f}")
        print(f"   Connectors OK: {conn_info['ok']}  "
              f"(connected={conn_info['centroids_connected']}/{conn_info['centroids']})")
        db = str(project_dir / "project_database.sqlite")
        cn = _sq.connect(db)
        modes = cn.execute(
            "SELECT modes, COUNT(*) FROM links WHERE link_type='centroid_connector' GROUP BY modes"
        ).fetchall()
        print(f"   Connector modes: {modes}")
        bad_tt = cn.execute(
            "SELECT SUM(CASE WHEN travel_time_ab<=0 OR travel_time_ab IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM links"
        ).fetchone()
        print(f"   Links with bad travel_time: {bad_tt[0]}/{bad_tt[1]}")
        cn.close()
        print("   Likely causes: connectors not reaching road network, or centroid IDs mismatch.")

    out_path = output_dir / "assignment_results.parquet"
    df.to_parquet(str(out_path), index=False)
    print(f"   Saved: {out_path}")

    if skims is not None:
        skim_path = output_dir / "skims.aem"
        try:
            skims.export(str(skim_path))
            print(f"   Skims saved: {skim_path}")
        except Exception as e:
            print(f"   WARNING: could not save skims: {e}")


# ---------------------------------------------------------------------------
# Temporal assignment (date + period)
# ---------------------------------------------------------------------------

def run_temporal_assignment(
    config_path: str | Path,
    date_str: str,
    period: str = "daily",
) -> pd.DataFrame:
    """Run assignment for a specific date, scaling the base OD by the learned
    day factor and optional period share.

    Returns the link-volume DataFrame (same format as baseline assignment).
    """
    from sim.temporal import load_profile, get_combined_factor, classify_day

    cfg = load_config(config_path)
    project_dir = Path(cfg["project_path"])
    demand_cfg = cfg.get("demand") or {}
    calib_cfg = cfg.get("calibration") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))

    algorithm = str(calib_cfg.get("algorithm", "bfw"))
    max_iter = int(calib_cfg.get("max_iter", 100))
    rgap = float(calib_cfg.get("rgap_target", 0.001))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))

    profile = load_profile(cfg)
    factor = get_combined_factor(date_str, period, profile)
    day_type = classify_day(date_str)

    print(f"=== TEMPORAL ASSIGNMENT: {date_str} ({day_type}), period={period}, factor={factor:.3f} ===")

    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])

    # Scale matrix data in memory (do NOT save — keep original on disk)
    mat.matrix_view[:, :] = mat.matrix_view[:, :] * factor
    total_demand = float(mat.matrix_view.sum())
    print(f"  Scaled demand: {total_demand:,.0f}")

    fix_node_ids(project_dir)

    project = Project()
    project.open(str(project_dir))
    try:
        df, _skims, _sl = execute_assignment(
            project, mat,
            algorithm=algorithm, max_iter=max_iter, rgap_target=rgap,
        )
    finally:
        project.close()
        mat.close()

    tot_col = _detect_volume_col(df)
    total_vol = float(df[tot_col].sum()) if tot_col else 0.0
    print(f"  Result: {len(df)} links, total_vol={total_vol:,.0f}")

    return df
