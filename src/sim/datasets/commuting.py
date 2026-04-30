"""SLDB 2021 commuting data preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from sim._text import norm_col as _norm_col
from sim.datasets.utils import (
    coerce_object_columns_for_parquet,
    ensure_dir,
    find_col,
    read_csv_flexible,
)


def preprocess_commuting_sldb2021(
    src_csv: Path,
    out_parquet: Path,
    *,
    delimiter: str,
    encoding: str,
    filter_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df = read_csv_flexible(src_csv, delimiter=delimiter, encoding=encoding)
    original_cols = list(df.columns)

    if (filter_cfg or {}).get("enabled", False):
        origin_cfg = filter_cfg.get("origin") or {}
        dest_cfg = filter_cfg.get("destination") or {}

        def apply_side_filter(side_df: pd.DataFrame, side: str, cfg: Dict[str, Any]) -> pd.DataFrame:
            if cfg.get("keep_all", False):
                return side_df
            clauses = cfg.get("keep_if_any_matches") or []
            if not clauses:
                return side_df

            mask = pd.Series(False, index=side_df.index)
            for clause in clauses:
                field = clause["field"]
                col = find_col(side_df, field)
                if col is None:
                    raise KeyError(
                        f"Filter requires column '{field}' ({side}), but it was not found in SLDB commuting CSV. "
                        f"Available columns: {original_cols[:60]}{' ...' if len(original_cols) > 60 else ''}"
                    )
                values = [str(v).strip().lower() for v in (clause.get("values") or [])]
                mask = mask | side_df[col].astype(str).str.strip().str.lower().isin(values)
            return side_df[mask].copy()

        df = apply_side_filter(df, "origin", origin_cfg)
        if not dest_cfg.get("keep_all", True) and dest_cfg.get("keep_if_any_matches"):
            df = apply_side_filter(df, "destination", dest_cfg)

    df.columns = [_norm_col(c) for c in df.columns]
    df = coerce_object_columns_for_parquet(df)
    ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)
    return {"rows": int(len(df)), "cols": int(len(df.columns)), "out_parquet": str(out_parquet)}
