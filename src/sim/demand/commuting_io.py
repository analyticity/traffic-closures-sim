"""Commuting data reading and filtering."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from sim.demand.config import DemandBuildCfg

logger = logging.getLogger(__name__)


def _read_commuting(bcfg: DemandBuildCfg) -> pd.DataFrame:
    if bcfg.external.enabled and bcfg.external.use_full_cr_dataset:
        if bcfg.commuting_full_cr_parquet.exists():
            logger.info("Reading full-CR commuting parquet: %s", bcfg.commuting_full_cr_parquet)
            return pd.read_parquet(bcfg.commuting_full_cr_parquet)

    if bcfg.commuting_filtered_parquet.exists():
        logger.info("Reading filtered commuting parquet: %s", bcfg.commuting_filtered_parquet)
        return pd.read_parquet(bcfg.commuting_filtered_parquet)

    if bcfg.external.enabled and bcfg.external.use_full_cr_dataset and bcfg.commuting_csv.exists():
        logger.info("Reading raw commuting CSV: %s", bcfg.commuting_csv)
        return pd.read_csv(
            bcfg.commuting_csv,
            sep=bcfg.csv_delimiter,
            encoding=bcfg.csv_encoding,
            low_memory=False,
        )

    raise FileNotFoundError(
        f"Commuting data not found.\n"
        f"  tried: {bcfg.commuting_full_cr_parquet}\n"
        f"  tried: {bcfg.commuting_filtered_parquet}\n"
        f"  tried: {bcfg.commuting_csv}"
    )


def _apply_origin_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df

    mask = pd.Series(False, index=df.index)
    any_applied = False

    for clause in filters:
        field = clause.get("field")
        vals = clause.get("values") or []
        if not field or not vals or field not in df.columns:
            continue

        any_applied = True
        vals_norm = [str(v).strip().lower() for v in vals]
        mask = mask | df[field].astype(str).str.strip().str.lower().isin(vals_norm)

    return df[mask].copy() if any_applied else df


def _filter_commuting(df: pd.DataFrame, bcfg: DemandBuildCfg) -> pd.DataFrame:
    if "lokalizace" in df.columns and bcfg.include_lokalizace:
        df = df[df["lokalizace"].astype(str).isin(bcfg.include_lokalizace)].copy()

    if not (bcfg.external.enabled and bcfg.external.use_full_cr_dataset):
        df = _apply_origin_filters(df, bcfg.origin_filters)

    for col in ("dojizdka_prace", "dojizdka_skola"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
