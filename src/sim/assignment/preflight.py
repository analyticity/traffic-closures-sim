"""Pre-flight checks: connector verification and node-ID overflow fix."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np


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
