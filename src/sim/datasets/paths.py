"""Resolved dataset path functions.

These are imported from many modules across the codebase and form the
stable public API for locating national-level data files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sim.datasets.utils import slug


def resolved_commuting_full_cr_parquet_path(cfg: Dict[str, Any]) -> Path:
    """Path to the national SLDB full-CR parquet (same for every city).

    Stored next to the downloaded CSV under ``data/sources/`` so it survives
    per-city ``clean`` and is not duplicated per city.
    """
    src = ((cfg.get("datasets") or {}).get("sources") or {}).get("commuting_sldb2021") or {}
    if src.get("full_cr_out_parquet"):
        return Path(str(src["full_cr_out_parquet"]))
    op = src.get("out_path")
    csv_p = Path(str(op)) if op else Path("data/sources/csu/sldb2021/dojizdka_obce.csv")
    return csv_p.parent / f"{csv_p.stem}_full_cr.parquet"


def resolved_cz_place_centroids_parquet_path(cfg: Dict[str, Any]) -> Path:
    """National RUIAN centroids parquet next to the downloaded ZIP."""
    src = ((cfg.get("datasets") or {}).get("sources") or {}).get("cz_place_centroids") or {}
    if src.get("out_parquet"):
        return Path(str(src["out_parquet"]))
    op = src.get("out_path")
    zip_p = Path(str(op)) if op else Path("data/sources/cz/places/ruian_csv_adr_st.zip")
    return zip_p.parent / "cz_place_centroids.parquet"


def resolved_csd2025_validation_parquet_path(cfg: Dict[str, Any]) -> Path:
    """CSD validation parquet next to the downloaded XLSX."""
    ds = cfg.get("datasets") or {}
    src = (ds.get("sources") or {}).get("validation_csd2025_v2") or {}
    if src.get("out_parquet"):
        return Path(str(src["out_parquet"]))
    op = src.get("out_path")
    if not op:
        return Path(str(ds.get("cache_dir", "data/cache"))) / "v2_csd2025.parquet"
    xlsx_p = Path(str(op))
    return xlsx_p.parent / f"{slug(xlsx_p.stem)}.parquet"
