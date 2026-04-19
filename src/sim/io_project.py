"""Configuration loading and project helper functions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; values in ``override`` win."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _as_abs_path(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        p = value
    elif isinstance(value, str):
        v = os.path.expandvars(os.path.expanduser(value))
        p = Path(v)
    else:
        return value

    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def load_config(config_path: str | Path = "config/sim.yaml") -> Dict[str, Any]:
    p = Path(config_path).expanduser().resolve()
    base_dir = p.parent

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping (dict), got: {type(cfg).__name__}")

    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(p)
    cfg["_meta"]["base_dir"] = str(base_dir)
    # Project root (parent of config/) — data files are stored here, not under config/
    project_root = base_dir.parent
    cfg["_meta"]["project_root"] = str(project_root)

    if "project_path" in cfg:
        cfg["project_path"] = _as_abs_path(cfg["project_path"], project_root)

    zoning = cfg.get("zoning") or {}
    if isinstance(zoning, dict):
        for k in ("cache_file", "output_dir"):
            if k in zoning:
                zoning[k] = _as_abs_path(zoning[k], project_root)
        sources = zoning.get("sources")
        if isinstance(sources, list):
            for src in sources:
                if isinstance(src, dict):
                    for path_key in ("path", "file"):
                        if path_key in src and src[path_key]:
                            src[path_key] = _as_abs_path(src[path_key], project_root)
        cfg["zoning"] = zoning

    demand = cfg.get("demand") or {}
    if isinstance(demand, dict):
        for k in ("matrix_path", "output_dir"):
            if k in demand:
                demand[k] = _as_abs_path(demand[k], project_root)
        cfg["demand"] = demand

    network = cfg.get("network") or {}
    if isinstance(network, dict):
        for k in ("output_dir", "maps_dir"):
            if k in network:
                network[k] = _as_abs_path(network[k], project_root)
        nc = network.get("normalization_config")
        if nc:
            nc_abs = Path(_as_abs_path(nc, project_root))
            if not nc_abs.is_file():
                raise FileNotFoundError(
                    f"network.normalization_config not found: {nc_abs}"
                )
            blob = yaml.safe_load(nc_abs.read_text(encoding="utf-8")) or {}
            file_norm = blob.get("normalization") if isinstance(blob.get("normalization"), dict) else {}
            file_exp = (
                blob.get("experiment_profiles")
                if isinstance(blob.get("experiment_profiles"), dict)
                else {}
            )
            inline_norm = (
                network.get("normalization") if isinstance(network.get("normalization"), dict) else {}
            )
            inline_exp = (
                network.get("experiment_profiles")
                if isinstance(network.get("experiment_profiles"), dict)
                else {}
            )
            network["normalization"] = _deep_merge(file_norm, inline_norm)
            network["experiment_profiles"] = _deep_merge(file_exp, inline_exp)
        cfg["network"] = network

    datasets = cfg.get("datasets") or {}
    if isinstance(datasets, dict):
        for k in ("root_dir", "cache_dir"):
            if k in datasets:
                datasets[k] = _as_abs_path(datasets[k], project_root)
        sources = datasets.get("sources") or {}
        if isinstance(sources, dict):
            for _, scfg in sources.items():
                if isinstance(scfg, dict) and "out_path" in scfg:
                    scfg["out_path"] = _as_abs_path(scfg["out_path"], project_root)
        cfg["datasets"] = datasets

    return cfg


def resolve_project_database_path(cfg: Dict[str, Any]) -> Path:
    """Return the AequilibraE project database path from config."""
    project_path = cfg.get("project_path", "project")
    return Path(project_path) / "project_database.sqlite"


def load_locale(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load locale-specific configuration (holidays, mappings, etc.).

    The locale file path is taken from ``cfg["locale_path"]``, defaulting
    to ``config/locale.yaml`` relative to the project root.
    """
    project_root = Path(cfg.get("_meta", {}).get("project_root", "."))
    locale_path = Path(cfg.get("locale_path", project_root / "config" / "locale.yaml"))
    if not locale_path.is_absolute():
        locale_path = (project_root / locale_path).resolve()
    if locale_path.exists():
        return yaml.safe_load(locale_path.read_text(encoding="utf-8")) or {}
    return {}


def get_metric_epsg(cfg: Dict[str, Any]) -> int:
    """Return the metric CRS EPSG code from config (``crs_epsg`` key)."""
    return int(cfg.get("crs_epsg", 5514))
