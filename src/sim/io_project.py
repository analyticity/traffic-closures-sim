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


def _find_project_root(config_file: Path) -> Path:
    """Walk up from *config_file* to find the repo/project root.

    The root is the nearest ancestor that directly contains a ``src``
    or ``config`` directory.  This allows config files to live at any
    nesting depth (``config/sim.yaml``, ``config/brno/sim.yaml``, etc.)
    without breaking relative-path resolution.
    """
    candidate = config_file.parent
    for _ in range(10):
        if (candidate / "src").is_dir() or (candidate / "config").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return config_file.parent.parent


def _derive_city_slug(base_dir: Path, cfg: Dict[str, Any]) -> str:
    """Derive a short city identifier from the config directory or project_path."""
    name = base_dir.name
    if name not in ("config", ".", ""):
        return name
    pp = cfg.get("project_path", "")
    if pp:
        stem = Path(pp).name.removesuffix("_aeq")
        if stem:
            return stem
    return "default"


def _set_nested_default(cfg: Dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a nested config value only if it is not already present."""
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
        if not isinstance(d, dict):
            return
    d.setdefault(keys[-1], value)


def load_config(config_path: str | Path = "config/brno/sim.yaml") -> Dict[str, Any]:
    from sim.defaults import SIM_DEFAULTS

    p = Path(config_path).expanduser().resolve()
    base_dir = p.parent

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping (dict), got: {type(raw).__name__}")

    cfg = _deep_merge(SIM_DEFAULTS, raw)

    city_slug = _derive_city_slug(base_dir, cfg)

    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(p)
    cfg["_meta"]["base_dir"] = str(base_dir)
    cfg["_meta"]["city_slug"] = city_slug
    project_root = _find_project_root(p)
    cfg["_meta"]["project_root"] = str(project_root)

    cfg.setdefault("crs_epsg", 5514)

    _set_nested_default(cfg, ["datasets", "cache_dir"], f"data/{city_slug}/cache")
    _set_nested_default(cfg, ["demand", "matrix_path"], f"data/{city_slug}/demand/od_matrix.aem")
    _set_nested_default(cfg, ["demand", "output_dir"], f"outputs/{city_slug}/baseline/demand")
    _set_nested_default(cfg, ["zoning", "output_dir"], f"outputs/{city_slug}/baseline/zones")
    _set_nested_default(cfg, ["network", "output_dir"], f"outputs/{city_slug}/baseline/network")
    _set_nested_default(cfg, ["network", "maps_dir"], f"outputs/{city_slug}/baseline/maps")
    _set_nested_default(cfg, ["supernetwork", "output_dir"], f"outputs/{city_slug}/baseline/supernetwork")
    _set_nested_default(cfg, ["supernetwork", "cache_dir"], f"data/{city_slug}/cache/supernetwork")

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

    supernetwork = cfg.get("supernetwork") or {}
    if isinstance(supernetwork, dict):
        for k in ("output_dir", "cache_dir"):
            if k in supernetwork:
                supernetwork[k] = _as_abs_path(supernetwork[k], project_root)
        cfg["supernetwork"] = supernetwork

    datasets = cfg.get("datasets") or {}
    if isinstance(datasets, dict):
        for k in ("root_dir", "cache_dir"):
            if k in datasets:
                datasets[k] = _as_abs_path(datasets[k], project_root)
        # Merge YAML sources on top of built-in CZ dataset defaults.
        from sim.fetch_datasets import _merge_dataset_sources  # lazy to avoid circular import
        datasets["sources"] = _merge_dataset_sources(datasets.get("sources"))
        sources = datasets["sources"]
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
    to ``locale.yaml`` in the same directory as the sim config file.
    National defaults (Czech Republic) from ``LOCALE_DEFAULTS`` are used
    as the base; the YAML file overrides them.
    """
    from sim.defaults import LOCALE_DEFAULTS

    meta = cfg.get("_meta", {})
    project_root = Path(meta.get("project_root", "."))
    config_dir = Path(meta.get("base_dir", project_root / "config"))
    locale_path = Path(cfg.get("locale_path", config_dir / "locale.yaml"))
    if not locale_path.is_absolute():
        locale_path = (project_root / locale_path).resolve()
    raw: Dict[str, Any] = {}
    if locale_path.exists():
        raw = yaml.safe_load(locale_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(LOCALE_DEFAULTS, raw)


def get_metric_epsg(cfg: Dict[str, Any]) -> int:
    """Return the metric CRS EPSG code from config (``crs_epsg`` key)."""
    return int(cfg.get("crs_epsg", 5514))
