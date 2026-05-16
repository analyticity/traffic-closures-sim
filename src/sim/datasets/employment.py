"""Employment data derivation from SLDB 2021 commuting flows.

Produces ``zone_employment.parquet`` by aggregating incoming commuting
flows per destination municipality (a proxy for workplace attractiveness),
then mapping to zone IDs via the same fuzzy matching used for population.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from sim._text import strip_diacritics as _strip_diacritics
from sim.datasets.population import _load_place_name_suffixes
from sim.datasets.utils import (
    coerce_object_columns_for_parquet,
    ensure_dir,
    read_csv_flexible,
    resolve_input_col,
)

logger = logging.getLogger(__name__)


def _norm(text: Any) -> str:
    return _strip_diacritics(text).lower().strip()


def derive_zone_employment(
    commuting_csv_or_parquet: Path,
    out_parquet: Path,
    zones_geojson: Optional[Path] = None,
    *,
    delimiter: str = "auto",
    encoding: str = "auto",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive employment per zone from commuting destination totals.

    Employment is approximated as the total number of incoming commuters
    per destination municipality, which correlates strongly with actual
    workplace counts.
    """
    import difflib

    if str(commuting_csv_or_parquet).endswith(".parquet"):
        df = pd.read_parquet(commuting_csv_or_parquet)
    else:
        df = read_csv_flexible(
            commuting_csv_or_parquet, delimiter=delimiter, encoding=encoding,
        )

    dest_col = resolve_input_col(
        df, {}, "cp_nazev",
        ["cp_nazev", "nazev_cilove_obce", "dest_name", "cp_obec", "cp_okres",
         "doj_obec"],
    )
    count_col = resolve_input_col(
        df, {}, "hodnota",
        ["hodnota", "value", "pocet", "count", "celkem",
         "dojizdka_prace", "dojizdka_celkova", "dojizdka_celkova_denni"],
    )
    purpose_col = resolve_input_col(
        df, {}, "ucel",
        ["ucel", "purpose", "ucel_txt", "ucel_kod", "lokalizace"],
    )

    if dest_col is None or count_col is None:
        raise RuntimeError(
            f"Commuting CSV/parquet missing required columns for employment derivation. "
            f"Need destination name and count. Available: {list(df.columns)[:30]}"
        )

    df["_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)

    # When the count column is already work-specific (dojizdka_prace),
    # skip purpose filtering to avoid discarding valid rows.
    _count_is_work_specific = count_col in ("dojizdka_prace",)
    if purpose_col is not None and not _count_is_work_specific:
        work_mask = df[purpose_col].astype(str).str.lower().str.contains(
            r"prac|work|zamest|1", na=False,
        )
        df_work = df[work_mask].copy()
        if df_work.empty:
            df_work = df.copy()
    else:
        df_work = df.copy()
    # Filter out in-place rows (lokalizace=0_na_adrese_OP) when present
    if "lokalizace" in df_work.columns:
        inter_municipal = ~df_work["lokalizace"].astype(str).str.startswith("0")
        if inter_municipal.any():
            df_work = df_work[inter_municipal].copy()

    employment_by_dest = (
        df_work.groupby(dest_col, as_index=False)["_count"]
        .sum()
        .rename(columns={dest_col: "dest_name", "_count": "employment"})
    )
    employment_by_dest["dest_name"] = employment_by_dest["dest_name"].astype(str).str.strip()

    emp_lookup = {
        str(r["dest_name"]).strip(): int(r["employment"])
        for _, r in employment_by_dest.iterrows()
    }
    emp_norm = {_norm(k): k for k in emp_lookup}

    if not (zones_geojson and zones_geojson.exists()):
        result = pd.DataFrame([
            {"zone_id": 0, "zone_name": name, "employment": emp}
            for name, emp in emp_lookup.items()
        ])
        ensure_dir(out_parquet.parent)
        result.to_parquet(out_parquet, index=False)
        return {
            "parquet": str(out_parquet),
            "zones_total": len(result),
            "matched": 0,
            "total_employment": int(result["employment"].sum()),
            "source": "commuting_destinations",
        }

    zones = gpd.read_file(zones_geojson)

    suffixes = _load_place_name_suffixes(cfg)

    result_rows: List[Dict[str, Any]] = []
    for _, zrow in zones.iterrows():
        zid = int(zrow["zone_id"])
        zname = str(zrow["name"])
        znorm = _norm(zname)

        if znorm in emp_norm:
            original = emp_norm[znorm]
            result_rows.append({
                "zone_id": zid, "zone_name": zname,
                "employment": emp_lookup[original], "match": "exact",
            })
            continue

        matched = False
        for suffix in suffixes:
            stripped = znorm.replace(suffix, "")
            if stripped != znorm and stripped in emp_norm:
                original = emp_norm[stripped]
                result_rows.append({
                    "zone_id": zid, "zone_name": zname,
                    "employment": emp_lookup[original],
                    "match": f"strip:{original}",
                })
                matched = True
                break
        if matched:
            continue

        close = difflib.get_close_matches(znorm, list(emp_norm.keys()), n=1, cutoff=0.82)
        if close:
            original = emp_norm[close[0]]
            candidate_emp = emp_lookup[original]
            ratio = difflib.SequenceMatcher(None, znorm, close[0]).ratio()
            if ratio < 0.95 and candidate_emp > 5000:
                logger.warning(
                    "Employment fuzzy match rejected for zone '%s' -> '%s' "
                    "(employment=%d, ratio=%.3f): destination too large for imprecise name match",
                    zname, original, candidate_emp, ratio,
                )
                result_rows.append({
                    "zone_id": zid, "zone_name": zname,
                    "employment": 0, "match": "fuzzy_rejected",
                })
            else:
                result_rows.append({
                    "zone_id": zid, "zone_name": zname,
                    "employment": candidate_emp,
                    "match": f"fuzzy:{original}",
                })
        else:
            result_rows.append({
                "zone_id": zid, "zone_name": zname,
                "employment": 0, "match": "unmatched",
            })

    result = coerce_object_columns_for_parquet(pd.DataFrame(result_rows))
    # Bump when employment matching rules change so zoning can invalidate stale parquet.
    result["match_engine_version"] = 2
    ensure_dir(out_parquet.parent)
    result.to_parquet(out_parquet, index=False)

    matched_count = sum(1 for r in result_rows if int(r["employment"]) > 0)
    total_emp = int(result["employment"].sum())
    logger.info(
        "Employment derivation: %d zones, %d matched, total=%d",
        len(result_rows), matched_count, total_emp,
    )
    return {
        "parquet": str(out_parquet),
        "zones_total": len(result_rows),
        "matched": matched_count,
        "total_employment": total_emp,
        "source": "commuting_destinations",
    }
