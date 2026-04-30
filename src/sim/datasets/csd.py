"""CSD 2025 XLSX/Parquet preprocessing and column normalization."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sim._text import norm_col as _norm_col
from sim.datasets.utils import (
    coerce_object_columns_for_parquet,
    ensure_dir,
    slug,
)
from sim.datasets.paths import resolved_csd2025_validation_parquet_path

logger = logging.getLogger(__name__)

_CSD_VOLUME_HINT_COLS = frozenset({
    "sv", "s", "o", "l", "tv", "t", "pn", "tn", "a", "al", "m", "tvp", "rpdi",
})


def _parquet_column_names(path: Path) -> set[str]:
    """Column names without loading the full table into memory."""
    try:
        import pyarrow.parquet as pq  # type: ignore

        return set(pq.read_schema(str(path)).names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def _csd_validation_parquet_schema_ok(path: Path) -> bool:
    names = _parquet_column_names(path)
    if "sil" not in names:
        return False
    return bool(names & _CSD_VOLUME_HINT_COLS)


# ---------------------------------------------------------------------------
# normalize_csd_count_columns
# ---------------------------------------------------------------------------

def normalize_csd_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``sv``, ``o``, and ``tv`` exist for calibration / validation.

    Official CSD 2025 exports often expose ``sv`` (all motor vehicles), ``o``
    (passenger cars), ``tv`` (heavy).  Some spreadsheets only publish class
    breakdowns (``S`` total, ``L`` light, ``T``/``PN``/``TN``/...); after
    :func:`sim._text.norm_col` those become ``s``, ``l``, ``t``, ``pn``, ...
    This function fills the canonical three totals so ``load_csd`` and
    ``load_csd_as_link_counts`` keep working unchanged.
    """
    out = df.copy()

    def _num(col: str) -> Optional[pd.Series]:
        if col not in out.columns:
            return None
        return pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    def _nonzero_series(s: Optional[pd.Series]) -> bool:
        if s is None:
            return False
        return bool(float(s.fillna(0).abs().sum()) > 0)

    # --- sv: all motor vehicles ---
    sv = _num("sv")
    if not _nonzero_series(sv):
        s_col = _num("s")
        if _nonzero_series(s_col):
            out["sv"] = s_col
        else:
            parts: list[pd.Series] = []
            for c in ("l", "t", "pn", "tn", "a", "al", "m"):
                v = _num(c)
                if _nonzero_series(v):
                    parts.append(v)
            if parts:
                acc = parts[0]
                for v in parts[1:]:
                    acc = acc + v
                out["sv"] = acc
            else:
                out["sv"] = 0.0
    else:
        out["sv"] = sv

    # --- o: passenger cars (proxy ``L`` / light vehicles when ``o`` absent) ---
    o_s = _num("o")
    if not _nonzero_series(o_s):
        l_col = _num("l")
        if _nonzero_series(l_col):
            out["o"] = l_col
        else:
            out["o"] = 0.0
    else:
        out["o"] = o_s

    # --- tv: heavy vehicles ---
    tv_s = _num("tv")
    if not _nonzero_series(tv_s):
        parts = []
        for c in ("t", "pn", "tn", "a", "al"):
            v = _num(c)
            if _nonzero_series(v):
                parts.append(v)
        if parts:
            acc = parts[0]
            for v in parts[1:]:
                acc = acc + v
            out["tv"] = acc
        else:
            out["tv"] = (out["sv"] - out["o"]).clip(lower=0.0)
    else:
        out["tv"] = tv_s

    return out


# ---------------------------------------------------------------------------
# XLSX preprocessing
# ---------------------------------------------------------------------------

def preprocess_xlsx_table(
    xlsx_path: Path,
    out_parquet: Path,
    *,
    key_cols: Optional[set[str]] = None,
    max_header_row_scan: int = 4,
    summary_tag: str = "xlsx_table",
) -> Dict[str, Any]:
    xls = pd.ExcelFile(xlsx_path)
    sheets = xls.sheet_names
    key_cols = key_cols or set()

    best_score = -1
    best_sheet = None
    best_header_row = 0
    best_df = None
    best_cols: List[str] = []

    for sheet in sheets:
        for hrow in range(max_header_row_scan + 1):
            try:
                preview = xls.parse(sheet, header=hrow, nrows=50)
            except Exception:
                continue
            cols_norm = [_norm_col(c) for c in preview.columns]
            if key_cols:
                score = sum(1 for c in cols_norm if c in key_cols)
                if score == 0:
                    continue
            else:
                score = len([c for c in cols_norm if c])
            if score > best_score:
                best_score = score
                best_sheet = sheet
                best_header_row = hrow
                best_df = xls.parse(sheet, header=hrow)
                best_cols = cols_norm

    if best_df is None or best_sheet is None:
        raise RuntimeError(f"Could not parse any sheet from {xlsx_path}")

    df = best_df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    df.insert(0, "sheet", str(best_sheet))
    df = coerce_object_columns_for_parquet(df)

    ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)

    summary = {
        "kind": summary_tag,
        "source_xlsx": str(xlsx_path),
        "sheets": sheets,
        "chosen_sheet": best_sheet,
        "chosen_header_row": int(best_header_row),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(df.columns),
        "best_score": int(best_score),
        "best_preview_columns_norm": best_cols,
    }
    summary_path = out_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"parquet": str(out_parquet), "summary": str(summary_path), **summary}


def preprocess_csd_xlsx(xlsx_path: Path, out_parquet: Path) -> Dict[str, Any]:
    summary = preprocess_xlsx_table(
        xlsx_path,
        out_parquet,
        key_cols={"sil", "rpdi", "sv", "s", "o", "l", "tv", "t"},
        max_header_row_scan=4,
        summary_tag="csd_xlsx",
    )
    df = pd.read_parquet(out_parquet)
    df = normalize_csd_count_columns(df)
    ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)
    return summary


# ---------------------------------------------------------------------------
# ensure_csd2025_validation_parquet
# ---------------------------------------------------------------------------

def ensure_csd2025_validation_parquet(cfg: Dict[str, Any]) -> Path:
    """Return the CSD validation parquet path, repairing stale files if needed.

    Older pipeline runs wrote a parquet whose first Excel row was taken as the
    header (``Unnamed:*`` columns only).  That breaks anything expecting
    ``sil`` and volume columns.  If the file on disk is stale but the source
    XLSX exists, re-run :func:`preprocess_csd_xlsx` in place.
    """
    parquet_path = resolved_csd2025_validation_parquet_path(cfg)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"CSD parquet not found: {parquet_path}. Run fetch-data first."
        )
    if _csd_validation_parquet_schema_ok(parquet_path):
        return parquet_path

    ds = cfg.get("datasets") or {}
    src = (ds.get("sources") or {}).get("validation_csd2025_v2") or {}
    xlsx_raw = src.get("out_path")
    if not xlsx_raw:
        raise FileNotFoundError(
            f"CSD parquet at {parquet_path} is missing usable CSD columns "
            "(need 'sil' plus a volume column). "
            "No validation_csd2025_v2.out_path is configured. Run fetch-data or delete the bad parquet."
        )
    xlsx_path = Path(str(xlsx_raw))
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"CSD parquet at {parquet_path} is missing usable CSD columns "
            f"and source XLSX not found at {xlsx_path}. Delete the parquet and run fetch-data."
        )

    logger.warning(
        "CSD parquet schema is stale or incomplete; re-preprocessing from %s",
        xlsx_path,
    )
    preprocess_csd_xlsx(xlsx_path, parquet_path)
    if not _csd_validation_parquet_schema_ok(parquet_path):
        raise RuntimeError(
            f"Re-preprocessing {xlsx_path} did not produce usable CSD columns "
            f"(need 'sil' plus at least one of {_CSD_VOLUME_HINT_COLS!r})."
        )
    return parquet_path


def classify_csd_road(sil: str) -> str:
    """Classify a CSD road number (``sil``) into a coarse OSM-style class.

    Returns one of: ``"motorway"``, ``"trunk"``, ``"secondary"``,
    ``"tertiary"``, ``"other"``.

    Czech convention: D-prefix → motorway, 1–99 → trunk (class I),
    100–999 → secondary (class II), 1000+ → tertiary (class III).
    """
    s = str(sil).strip().upper()
    if s.startswith("D"):
        return "motorway"
    try:
        num = int(s.replace("M", ""))
        if num < 100:
            return "trunk"
        if num < 1000:
            return "secondary"
        return "tertiary"
    except ValueError:
        return "trunk" if "M" in s else "other"
