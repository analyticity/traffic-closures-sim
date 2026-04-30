"""AequilibraE matrix I/O: write .aem files and register in project."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from aequilibrae.matrix import AequilibraeMatrix

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_aem(
    matrix_path: Path,
    index_ids: np.ndarray,
    cores: Dict[str, np.ndarray],
    *,
    matrix_name: str,
) -> None:
    _ensure_dir(matrix_path.parent)
    core_names = list(cores.keys())

    mat = AequilibraeMatrix()
    mat.create_empty(
        file_name=str(matrix_path),
        zones=int(len(index_ids)),
        matrix_names=core_names,
        data_type=np.float64,
        memory_only=False,
    )

    try:
        mat.setName(matrix_name)
    except Exception:
        pass

    try:
        mat.setDescription("OD matrix built from commuting, supernetwork, and residual synthetic seeds")
    except Exception:
        pass

    mat.index[:] = index_ids.astype(np.int64)
    for core_name in core_names:
        mat.matrix[core_name][:, :] = cores[core_name]

    mat.save()
    mat.close()


def _register_in_project(project_dir: Path, matrix_path: Path) -> None:
    from aequilibrae import Project

    project = Project()
    project.open(str(project_dir))
    try:
        matrices_dir = Path(project.project_base_path) / "matrices"
        _ensure_dir(matrices_dir)

        target = matrices_dir / matrix_path.name
        if matrix_path.resolve() != target.resolve():
            shutil.copy2(matrix_path, target)

        try:
            project.matrices.update_database()
            project.matrices.reload()
        except Exception:
            pass
    finally:
        project.close()
