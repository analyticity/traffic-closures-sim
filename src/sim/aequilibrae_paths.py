"""Shared paths for AequilibraE on-disk projects."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from aequilibrae import Project


def resolve_project_database_path(project_or_dir: Union[Project, Path, str]) -> Path:
    """Return the SQLite DB used by AequilibraE.

    Prefer ``project_database.sqlite`` under ``project_base_path`` (AequilibraE default).
    If missing, fall back to the first ``*.sqlite`` / ``*.db`` in that directory.
    """
    if isinstance(project_or_dir, Project):
        project_dir = Path(str(project_or_dir.project_base_path))
    else:
        project_dir = Path(project_or_dir)

    canonical = project_dir / "project_database.sqlite"
    if canonical.is_file():
        return canonical

    candidates = (
        list(project_dir.glob("*.sqlite"))
        + list(project_dir.glob("*.db"))
        + list(project_dir.glob("*.sqlite3"))
    )
    if candidates:
        return Path(candidates[0])

    return canonical
