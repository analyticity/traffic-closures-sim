"""SQLite database helpers for AequilibraE network manipulation."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence, Union

from aequilibrae import Project

from sim.network.db_paths import resolve_project_database_path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: float = 120.0
DEFAULT_CHUNK_SIZE: int = 450


def project_db_path(project_or_dir: Union[Project, Path]) -> Path:
    """Resolve the SQLite database file for an AequilibraE project or directory."""
    return resolve_project_database_path(project_or_dir)


@contextmanager
def project_db(
    project_or_dir: Union[Project, Path],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[sqlite3.Connection]:
    """Context manager that opens the project DB, commits on success, and always closes."""
    conn = sqlite3.connect(str(project_db_path(project_or_dir)), timeout=timeout)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def refresh_network(project: Project) -> None:
    """Safely refresh network, nodes, and links caches on the AequilibraE project."""
    for fn in (
        project.network.nodes.refresh,
        project.network.links.refresh,
    ):
        try:
            fn()
        except Exception:
            pass


def bulk_delete_by_ids(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    ids: Sequence[int],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """DELETE rows from *table* whose *id_column* is in *ids*, in chunks.

    Returns the total number of IDs submitted (not ``rowcount``, which
    is unreliable with chunked execution).
    """
    if not ids:
        return 0
    for i in range(0, len(ids), chunk_size):
        part = ids[i : i + chunk_size]
        placeholders = ",".join("?" * len(part))
        conn.execute(
            f"DELETE FROM {table} WHERE {id_column} IN ({placeholders})",
            list(part),
        )
    return len(ids)
