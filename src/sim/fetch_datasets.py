#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent.parent
    if _src_dir.exists() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

from sim.defaults import SIM_DEFAULTS
from sim.io_project import get_metric_epsg, load_config
from sim._text import strip_diacritics as _strip_diacritics, norm_col as _norm_col

_logger = logging.getLogger(__name__)

# --- Default Czech dataset registry (YAML merges on top) ---

_CZ_DEFAULT_SOURCES: Dict[str, Dict[str, Any]] = {
    "commuting_sldb2021": {
        "enabled": True,
        "provider": "csu_open_data_csv",
        "url": "https://csu.gov.cz/docs/107508/4dbdab3b-905c-deff-4cfa-e4828a6fa2de/dojizdka_obce.csv?version=1.0",
        "out_path": "data/sources/csu/sldb2021/dojizdka_obce.csv",
        "purposes": ["work", "school"],
        "preprocess": {"write_filtered": True, "write_full_cr": True},
    },
    "validation_csd2025_v2": {
        "enabled": True,
        "year": 2025,
        "provider": "http_file",
        "url": "https://www.rsd.cz/documents/38144/3734982/V2_CSD_2025.xlsx/50663492-395b-0fd4-f365-d18440997ca5?t=1773922634346",
        "out_path": "data/sources/rsd/csd2025/V2_CSD2025.xlsx",
        "format": {"type": "xlsx"},
        "usage": {"validation_target": "aadt_screenlines"},
    },
    "population_sldb2021": {
        "enabled": True,
        "provider": "http_file",
        "url": "https://csu.gov.cz/docs/107508/79c509a0-261c-b4dd-d58d-c05955a24a2c/sldb2021_pohlavi.csv",
        "out_path": "data/sources/csu/sldb2021/populace_pohlavi.csv",
        "format": {"type": "csv"},
        "usage": {"socioeconomic": "population_per_zone"},
    },
    "cz_place_centroids": {
        "enabled": True,
        "provider": "atom_file",
        "url": "https://atom.cuzk.gov.cz/get.ashx?theme=RUIAN-CSV-ADR-ST",
        "feed_cache_path": "data/sources/cz/places/ruian_csv_adr_st.atom.xml",
        "out_path": "data/sources/cz/places/ruian_csv_adr_st.zip",
        "format": {
            "type": "zip_csv",
            "asset_pattern": r"(?i)(^|/)[0-9]{8}_OB_ADR_csv\.zip$",
            "member_pattern": r"(?i)\.csv$",
            "delimiter": ";",
            "encoding": "cp1250",
            "source_crs_epsg": 2065,
            "output_crs_epsg": 4326,
        },
        "columns": {
            "place_code": "Kód obce",
            "place_name": "Název obce",
            "x": "Souřadnice X",
            "y": "Souřadnice Y",
        },
        "usage": {"supernetwork_places": "grouped_point_centroids"},
    },
    "closures_pg": {
        "enabled": True,
        "provider": "postgres_closures",
        "table": "restrictions",
        "status_whitelist": [],
        "min_observed_days": 2,
        "usage": {"calibration_target": "baseline_closures"},
    },
    "cz_roads_major_pbf": {
        "enabled": True,
        "provider": "http_file",
        "url": "https://download.geofabrik.de/europe/czech-republic-latest.osm.pbf",
        "out_path": "data/sources/osm/czech-republic-latest.osm.pbf",
    },
}


def _merge_dataset_sources(yaml_sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge YAML dataset source overrides on top of built-in CZ defaults."""
    import copy
    merged = copy.deepcopy(_CZ_DEFAULT_SOURCES)
    if not yaml_sources:
        return merged
    for key, overrides in yaml_sources.items():
        if key in merged:
            if overrides is None:
                continue
            merged[key].update(overrides)
        else:
            merged[key] = overrides or {}
    return merged


# --- Small helpers ---

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


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
    return xlsx_p.parent / f"{_slug(xlsx_p.stem)}.parquet"


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


def ensure_csd2025_validation_parquet(cfg: Dict[str, Any]) -> Path:
    """Return the CSD validation parquet path, repairing stale files if needed.

    Older pipeline runs wrote a parquet whose first Excel row was taken as the
    header (``Unnamed:*`` columns only).  That breaks anything expecting
    ``sil`` and volume columns.  If the file on disk is stale but the source XLSX
    exists, re-run :func:`preprocess_csd_xlsx` in place.
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

    _logger.warning(
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


def normalize_csd_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``sv``, ``o``, and ``tv`` exist for calibration / validation.

    Official CSD 2025 exports often expose ``sv`` (all motor vehicles), ``o``
    (passenger cars), ``tv`` (heavy).  Some spreadsheets only publish class
    breakdowns (``S`` total, ``L`` light, ``T``/``PN``/``TN``/…); after
    :func:`sim._text.norm_col` those become ``s``, ``l``, ``t``, ``pn``, …
    This function fills the canonical three totals so :func:`load_csd` and
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


def _find_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    target = _norm_col(wanted)
    for col in df.columns:
        if _norm_col(col) == target:
            return col
    return None


def _resolve_input_col(
    df: pd.DataFrame,
    columns_cfg: Dict[str, Any],
    cfg_key: str,
    candidates: Iterable[str],
) -> Optional[str]:
    configured = columns_cfg.get(cfg_key)
    if configured:
        if configured in df.columns:
            return configured
        found = _find_col(df, str(configured))
        if found:
            return found

    for cand in candidates:
        found = _find_col(df, cand)
        if found:
            return found
    return None


def _coerce_object_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].where(out[col].notna(), None).astype("string")
    return out


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _looks_mojibaked(df: pd.DataFrame) -> bool:
    suspicious = ["�", "Ã", "Ø", "ð", "\x00"]
    for col in list(df.columns[:10]):
        if any(ch in str(col) for ch in suspicious):
            return True

    obj_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    for col in obj_cols[:10]:
        for value in df[col].dropna().astype(str).head(30):
            if any(ch in value for ch in suspicious):
                return True
    return False


def _read_csv_attempt(path: Path, *, delimiter: Optional[str], encoding: str) -> pd.DataFrame:
    if delimiter is None:
        return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    return pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)


def _read_csv_flexible(path: Path, *, delimiter: str = "auto", encoding: str = "auto") -> pd.DataFrame:
    fallback_encodings = ["cp1250", "windows-1250", "utf-8", "utf-8-sig", "iso-8859-2"]
    encodings = [encoding] + [e for e in fallback_encodings if e != encoding] if encoding not in ("", None, "auto") else fallback_encodings
    delimiters: List[Optional[str]] = [",", ";", "\t", "|", None] if delimiter in ("", None, "auto") else [delimiter]

    raw_head = path.read_bytes()[:4096]
    last_err: Optional[Exception] = None

    for enc in encodings:
        for delim in delimiters:
            try:
                df = _read_csv_attempt(path, delimiter=delim, encoding=enc)
                if _looks_mojibaked(df):
                    raise UnicodeError(f"CSV decoded as mojibake: encoding={enc!r}, delimiter={delim!r}")

                if len(df.columns) == 1:
                    try:
                        head_text = raw_head.decode(enc, errors="strict")
                    except Exception:
                        head_text = ""
                    if any(sep in head_text for sep in [",", ";", "\t", "|"]):
                        raise ValueError(f"CSV parsed into one column with encoding={enc!r}, delimiter={delim!r}")

                return df
            except Exception as exc:
                last_err = exc

    raise RuntimeError(f"Failed to read CSV {path}: {last_err}")


# --- Download helpers ---

def download_file(
    url: str,
    out_path: Path,
    *,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    _ensure_dir(out_path.parent)

    if out_path.exists() and not force:
        return {"status": "cached", "path": str(out_path), "bytes": out_path.stat().st_size}

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout_s, headers=headers)
            response.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.write_bytes(response.content)
            tmp.replace(out_path)
            return {"status": "downloaded", "path": str(out_path), "bytes": out_path.stat().st_size}
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(float(attempt))

    raise RuntimeError(f"Failed to download {url} -> {out_path}: {last_err}")


def _score_atom_asset(url: str) -> Tuple[int, int, str]:
    name = url.rsplit("/", 1)[-1]
    score = 0
    if re.match(r"(?i)^\d{8}_OB_ADR_csv\.zip$", name):
        score += 1000
    if re.match(r"(?i)^\d{8}_OB_\d+_ADR\.csv\.zip$", name):
        score -= 1000
    if re.match(r"(?i)^\d{8}_OB_.*\.zip$", name):
        score += 50
    return (score, len(name), name.lower())


def download_atom_latest_file(
    feed_url: str,
    out_path: Path,
    *,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
    asset_pattern: Optional[str] = None,
    feed_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    _ensure_dir(out_path.parent)
    if feed_cache_path is not None:
        _ensure_dir(feed_cache_path.parent)

    if out_path.exists() and not force:
        return {
            "status": "cached",
            "feed_url": feed_url,
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
        }

    xml_text: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(feed_url, timeout=timeout_s, headers=headers)
            response.raise_for_status()
            xml_text = response.text
            break
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(float(attempt))

    if xml_text is None:
        raise RuntimeError(f"Failed to download ATOM feed {feed_url}: {last_err}")

    if feed_cache_path is not None:
        feed_cache_path.write_text(xml_text, encoding="utf-8")

    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ATOM feed {feed_url}: {exc}") from exc

    hrefs: List[str] = []
    for elem in root.iter():
        if elem.tag.lower().endswith("link"):
            href = elem.attrib.get("href")
            if href:
                hrefs.append(urljoin(feed_url, href))

    hrefs_unique = list(dict.fromkeys(hrefs))
    if asset_pattern:
        rgx = re.compile(asset_pattern)
        hrefs_unique = [h for h in hrefs_unique if rgx.search(h)]

    if not hrefs_unique:
        raise RuntimeError(f"No downloadable asset found in ATOM feed {feed_url} (asset_pattern={asset_pattern!r})")

    hrefs_unique = sorted(hrefs_unique, key=_score_atom_asset, reverse=True)
    asset_url = hrefs_unique[0]
    info = download_file(
        asset_url,
        out_path,
        timeout_s=timeout_s,
        retries=retries,
        headers=headers,
        force=force,
    )
    info.update({
        "feed_url": feed_url,
        "asset_url": asset_url,
        "asset_candidates": hrefs_unique[:10],
    })
    if feed_cache_path is not None:
        info["feed_cache_path"] = str(feed_cache_path)
    return info


# --- Grouped point centroids ---

def _build_grouped_point_centroids_from_df(
    df: pd.DataFrame,
    out_parquet: Path,
    *,
    source_crs_epsg: int,
    output_crs_epsg: int,
    columns_cfg: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if gpd is None:
        raise RuntimeError("geopandas is required for grouped point centroid preprocessing")

    columns_cfg = columns_cfg or {}

    place_code_col = _resolve_input_col(df, columns_cfg, "place_code", ["kod_obce", "obec_kod", "municipality_code", "place_code"])
    place_name_col = _resolve_input_col(df, columns_cfg, "place_name", ["nazev_obce", "obec_nazev", "obec", "municipality_name", "place_name", "name"])
    district_name_col = _resolve_input_col(df, columns_cfg, "district_name", ["nazev_momc", "nazev_obvodu_prahy", "nazev_casti_obce", "district_name", "district"])
    admin_level_col = _resolve_input_col(df, columns_cfg, "admin_level", ["admin_level", "level", "typ", "type"])
    x_col = _resolve_input_col(df, columns_cfg, "x", ["souradnice_x", "x", "lon", "longitude"])
    y_col = _resolve_input_col(df, columns_cfg, "y", ["souradnice_y", "y", "lat", "latitude"])

    if place_name_col is None:
        raise RuntimeError(f"Grouped point dataset is missing place_name column. Columns: {list(df.columns)}")
    if x_col is None or y_col is None:
        raise RuntimeError(f"Grouped point dataset is missing x/y columns. Columns: {list(df.columns)}")

    tmp = pd.DataFrame({
        "place_code": df[place_code_col].astype(str).str.strip() if place_code_col else "",
        "place_name": df[place_name_col].astype(str).str.strip(),
        "district_name": df[district_name_col].fillna("").astype(str).str.strip() if district_name_col else "",
        "admin_level": df[admin_level_col].fillna("").astype(str).str.strip() if admin_level_col else "municipality",
        "_x": _coerce_numeric_series(df[x_col]),
        "_y": _coerce_numeric_series(df[y_col]),
    })

    tmp = tmp.dropna(subset=["_x", "_y"]).copy()
    tmp = tmp[tmp["place_name"] != ""].copy()

    group_cols = ["place_name"]
    if tmp["place_code"].astype(str).str.strip().ne("").any():
        group_cols = ["place_code", "place_name"]
    group_cols += ["district_name", "admin_level"]

    rows: List[Dict[str, Any]] = []
    for _, grp in tmp.groupby(group_cols, dropna=False, sort=True):
        mx = float(grp["_x"].mean())
        my = float(grp["_y"].mean())
        idx = ((grp["_x"] - mx) ** 2 + (grp["_y"] - my) ** 2).idxmin()
        rep = grp.loc[idx]
        rows.append({
            "place_code": str(rep.get("place_code", "")).strip(),
            "place_name": str(rep.get("place_name", "")).strip(),
            "district_name": str(rep.get("district_name", "")).strip(),
            "admin_level": str(rep.get("admin_level", "municipality")).strip() or "municipality",
            "point_count": int(len(grp)),
            "x": float(rep["_x"]),
            "y": float(rep["_y"]),
        })

    out = _coerce_object_columns_for_parquet(pd.DataFrame(rows))
    gdf = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs=f"EPSG:{int(source_crs_epsg)}",
    )
    if int(output_crs_epsg) != int(source_crs_epsg):
        gdf = gdf.to_crs(epsg=int(output_crs_epsg))

    _ensure_dir(out_parquet.parent)
    gdf.to_parquet(out_parquet, index=False)

    result: Dict[str, Any] = {
        "rows": int(len(gdf)),
        "cols": int(len(gdf.columns)),
        "out_parquet": str(out_parquet),
    }
    if extra_info:
        result.update(extra_info)
    return result


def preprocess_grouped_points_to_centroids(
    src_csv: Path,
    out_parquet: Path,
    *,
    delimiter: str = "auto",
    encoding: str = "auto",
    source_crs_epsg: int = 5514,
    output_crs_epsg: int = 4326,
    columns_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df = _read_csv_flexible(src_csv, delimiter=delimiter, encoding=encoding)
    return _build_grouped_point_centroids_from_df(
        df,
        out_parquet,
        source_crs_epsg=source_crs_epsg,
        output_crs_epsg=output_crs_epsg,
        columns_cfg=columns_cfg,
    )


def preprocess_grouped_points_zip_to_centroids(
    zip_path: Path,
    out_parquet: Path,
    *,
    member_pattern: Optional[str] = None,
    delimiter: str = "auto",
    encoding: str = "auto",
    source_crs_epsg: int = 5514,
    output_crs_epsg: int = 4326,
    columns_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    frames: List[pd.DataFrame] = []
    matched_members: List[str] = []
    pattern = re.compile(member_pattern) if member_pattern else None

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if pattern is not None:
            names = [n for n in names if pattern.search(n)]
        if not names:
            raise RuntimeError(f"No ZIP members matched member_pattern={member_pattern!r} in {zip_path}")

        for name in names:
            raw = zf.read(name)
            tmp_path = zip_path.parent / f".__tmp_{Path(name).name}"
            try:
                tmp_path.write_bytes(raw)
                frames.append(_read_csv_flexible(tmp_path, delimiter=delimiter, encoding=encoding))
                matched_members.append(name)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    frames = [f for f in frames if not f.empty and not f.isna().all(axis=None)]
    df = pd.concat(frames, ignore_index=True)
    return _build_grouped_point_centroids_from_df(
        df,
        out_parquet,
        source_crs_epsg=source_crs_epsg,
        output_crs_epsg=output_crs_epsg,
        columns_cfg=columns_cfg,
        extra_info={"zip_members_used": len(matched_members)},
    )


# --- Preprocessors (dataset-specific) ---

def preprocess_commuting_sldb2021(
    src_csv: Path,
    out_parquet: Path,
    *,
    delimiter: str,
    encoding: str,
    filter_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df = _read_csv_flexible(src_csv, delimiter=delimiter, encoding=encoding)
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
                col = _find_col(side_df, field)
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
    df = _coerce_object_columns_for_parquet(df)
    _ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)
    return {"rows": int(len(df)), "cols": int(len(df.columns)), "out_parquet": str(out_parquet)}


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
    df = _coerce_object_columns_for_parquet(df)

    _ensure_dir(out_parquet.parent)
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
    _ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)
    return summary


def _load_mc_to_ku(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Load municipal-part → cadastral-unit mapping from locale config."""
    if cfg:
        from sim.io_project import load_locale
        locale = load_locale(cfg)
        mapping = locale.get("municipal_parts_to_cadastral")
        if mapping and isinstance(mapping, dict):
            return mapping
    return {}


_DEFAULT_CZ_SUFFIXES = [
    " u brna", " u prahy", " u mostu", " u olomouce", " u ostravy",
    " nad labem", " nad svitavou", " nad sazavou", " nad vltavou",
    " nad orlici", " nad moravou", " nad jihlavou", " nad luznici",
    " v cechach", " na morave", " pod rizem", " pod radhostem",
]


def _load_place_name_suffixes(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Load place-name suffixes to strip during zone↔obec matching.

    Returns the list from ``locale.yaml`` → ``place_name_suffixes_to_strip``
    if present; otherwise falls back to common Czech municipality suffixes.
    """
    if cfg:
        from sim.io_project import load_locale
        locale = load_locale(cfg)
        custom = locale.get("place_name_suffixes_to_strip")
        if custom and isinstance(custom, list):
            return [str(s).lower() for s in custom]
    return list(_DEFAULT_CZ_SUFFIXES)


def preprocess_population_sldb2021(
    csv_path: Path,
    out_parquet: Path,
    zones_geojson: Optional[Path] = None,
    *,
    delimiter: str = "auto",
    encoding: str = "auto",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import difflib

    def norm(text: Any) -> str:
        return _strip_diacritics(text).lower().strip()

    df = _read_csv_flexible(csv_path, delimiter=delimiter, encoding=encoding)

    pohlavi_col = _resolve_input_col(df, {}, "pohlavi_kod", ["pohlavi_kod", "kod_pohlavi", "pohlavi"])
    hodnota_col = _resolve_input_col(df, {}, "hodnota", ["hodnota", "value", "pocet"])
    nazev_col = _resolve_input_col(df, {}, "uzemi_txt", ["uzemi_txt", "nazev", "nazev_uzemi", "uzemi"])
    uzemi_cis_col = _resolve_input_col(df, {}, "uzemi_cis", ["uzemi_cis", "typ_uzemi", "uzemni_uroven", "uzemi_typ"])
    uzemi_kod_col = _resolve_input_col(df, {}, "uzemi_kod", ["uzemi_kod", "kod_uzemi", "kod"])

    missing = [
        key for key, value in {
            "pohlavi_kod": pohlavi_col,
            "hodnota": hodnota_col,
            "uzemi_txt": nazev_col,
            "uzemi_cis": uzemi_cis_col,
            "uzemi_kod": uzemi_kod_col,
        }.items() if value is None
    ]
    if missing:
        raise RuntimeError(f"Population CSV is missing required columns {missing}. Available columns: {list(df.columns)}")

    tmp = pd.DataFrame({
        "pohlavi_kod": df[pohlavi_col],
        "hodnota": df[hodnota_col],
        "uzemi_txt": df[nazev_col],
        "uzemi_cis": df[uzemi_cis_col],
        "uzemi_kod": df[uzemi_kod_col],
    })

    total = tmp[tmp["pohlavi_kod"].isna()].copy()
    total["hodnota"] = pd.to_numeric(total["hodnota"], errors="coerce").fillna(0).astype(int)
    total["nazev"] = total["uzemi_txt"].astype(str).str.strip()

    place_name = (cfg or {}).get("osm", {}).get("place_name", "")
    city_prefix = place_name.split(",")[0].strip() if place_name else ""
    mc_filter = (
        (pd.to_numeric(total["uzemi_cis"], errors="coerce") == 44)
        & (total["nazev"].str.startswith(city_prefix, na=False) if city_prefix else False)
    )
    city_mc = total[mc_filter]
    mc_pop = {str(r["nazev"]).strip(): int(r["hodnota"]) for _, r in city_mc.iterrows()}

    obce = total[pd.to_numeric(total["uzemi_cis"], errors="coerce") == 43][["uzemi_kod", "nazev", "hodnota"]].copy()
    obce = obce.sort_values("hodnota", ascending=False).drop_duplicates(subset="nazev", keep="first")
    obec_pop = {str(r["nazev"]).strip(): int(r["hodnota"]) for _, r in obce.iterrows()}
    obec_norm = {norm(k): k for k in obec_pop}

    if not (zones_geojson and zones_geojson.exists()):
        result = pd.DataFrame([
            {"zone_id": 0, "zone_name": name, "population": pop, "match": "no_zones"}
            for name, pop in {**mc_pop, **obec_pop}.items()
        ])
        _ensure_dir(out_parquet.parent)
        result.to_parquet(out_parquet, index=False)
        return {"parquet": str(out_parquet), "zones_total": len(result), "matched": 0, "total_population": int(result["population"].sum())}

    zones = gpd.read_file(zones_geojson)
    mc_to_ku = _load_mc_to_ku(cfg)
    ku_to_mc = {norm(ku): mc for mc, ku_list in mc_to_ku.items() for ku in ku_list}

    zones_m = zones.to_crs(epsg=get_metric_epsg(cfg or {}))
    zone_area = {int(r["zone_id"]): r.geometry.area for _, r in zones_m.iterrows()}

    # MC→KU mapping is optional; skip for cities without municipal parts.
    mc_zone_ids: Dict[str, List[int]] = {}
    zone_mc_map: Dict[int, str] = {}
    if ku_to_mc:
        for _, zrow in zones.iterrows():
            zid = int(zrow["zone_id"])
            znorm = norm(zrow["name"])
            if znorm in ku_to_mc:
                mc = ku_to_mc[znorm]
                mc_zone_ids.setdefault(mc, []).append(zid)
                zone_mc_map[zid] = mc

    # Name suffixes to strip when matching zone names to SLDB obce.
    # Loaded from locale config; falls back to common Czech suffixes.
    suffixes_to_strip = _load_place_name_suffixes(cfg)

    max_zone_mc_share = 0.70
    result_rows: List[Dict[str, Any]] = []

    for _, zrow in zones.iterrows():
        zid = int(zrow["zone_id"])
        zname = str(zrow["name"])
        znorm = norm(zname)

        if zid in zone_mc_map:
            mc = zone_mc_map[zid]
            zone_ids = mc_zone_ids.get(mc, [])
            total_area = sum(zone_area.get(z, 0.0) for z in zone_ids) or 1.0
            share = min(zone_area.get(zid, 0.0) / total_area, max_zone_mc_share)
            pop = int(round(mc_pop.get(mc, 0) * share))
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": pop, "match": f"mc_area:{mc}"})
            continue

        if znorm in obec_norm:
            original = obec_norm[znorm]
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": obec_pop[original], "match": "obec_exact"})
            continue

        matched = False
        for suffix in suffixes_to_strip:
            stripped = znorm.replace(suffix, "")
            if stripped != znorm and stripped in obec_norm:
                original = obec_norm[stripped]
                result_rows.append({"zone_id": zid, "zone_name": zname, "population": obec_pop[original], "match": f"obec_strip:{original}"})
                matched = True
                break
        if matched:
            continue

        close = difflib.get_close_matches(znorm, list(obec_norm.keys()), n=1, cutoff=0.7)
        if close:
            original = obec_norm[close[0]]
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": obec_pop[original], "match": f"fuzzy:{original}"})
        else:
            all_pops = [v for v in {**mc_pop, **obec_pop}.values() if v > 0]
            avg = int(sorted(all_pops)[len(all_pops) // 2]) if all_pops else 1000
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": avg, "match": "default_median"})

    result = _coerce_object_columns_for_parquet(pd.DataFrame(result_rows))
    _ensure_dir(out_parquet.parent)
    result.to_parquet(out_parquet, index=False)

    matched = len([r for r in result_rows if r["match"] != "default_median"])
    total_pop = int(result["population"].sum())
    return {
        "parquet": str(out_parquet),
        "zones_total": len(result_rows),
        "matched": matched,
        "total_population": total_pop,
    }


# --- ArcGIS ---

def _load_aoi_bbox_wgs84(cfg: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    mb = cfg.get("model_bbox")
    if mb and len(mb) == 4 and None not in mb:
        return tuple(float(x) for x in mb)

    if gpd is None:
        return None

    zones_dir = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))
    for candidate in [zones_dir / "model_area.geojson", Path(cfg.get("network", {}).get("maps_dir", "outputs/baseline/maps")) / "model_bbox_wgs84.geojson"]:
        if candidate.exists():
            try:
                gdf = gpd.read_file(candidate)
                if gdf.crs is None:
                    gdf = gdf.set_crs(epsg=4326)
                elif gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                return tuple(float(x) for x in gdf.total_bounds)
            except Exception:
                pass
    return None


def arcgis_query_geojson(
    service_url: str,
    layer_id: int,
    *,
    where: str,
    out_fields: List[str],
    bbox_wgs84: Optional[Tuple[float, float, float, float]],
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    base = service_url.rstrip("/") + f"/{layer_id}/query"
    all_features: List[Dict[str, Any]] = []
    offset = 0
    page_size = 2000
    params_common = {
        "f": "geojson",
        "where": where,
        "outFields": ",".join(out_fields) if out_fields else "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": str(page_size),
    }

    if bbox_wgs84 is not None:
        w, s, e, n = bbox_wgs84
        params_common.update({
            "geometry": f"{w},{s},{e},{n}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        })

    while True:
        params = dict(params_common)
        params["resultOffset"] = str(offset)

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(base, params=params, timeout=timeout_s, headers=headers)
                response.raise_for_status()
                data = response.json()
                feats = data.get("features") or []
                all_features.extend(feats)
                exceeded = bool(data.get("exceededTransferLimit", False))
                break
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(float(attempt))
                else:
                    raise RuntimeError(f"ArcGIS query failed: {base} ({last_err})") from last_err

        if not feats:
            break
        offset += len(feats)
        # GeoJSON format omits exceededTransferLimit; fall back to
        # checking whether we got a full page of results.
        if not exceeded and len(feats) < page_size:
            break

    return {"type": "FeatureCollection", "features": all_features}


def _postprocess_pentlogram(
    gdf: "gpd.GeoDataFrame",
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
) -> Optional["gpd.GeoDataFrame"]:
    """Compute observed volumes and run spatial outlier filtering.

    Returns a cleaned GeoDataFrame ready for diagnostics, or None on error.
    """
    try:
        import numpy as np

        units_cfg = (source_cfg.get("units") or {})
        car_mult = float(units_cfg.get("car_24_multiplier", 1000))

        work = gdf.copy()
        for col in ("car_24", "truc_24"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

        total_vehicles = work.get("car_24", 0) * car_mult
        truck_pct = work.get("truc_24", 0).clip(0, 100)
        work["observed_car"] = total_vehicles
        work["observed_truck"] = total_vehicles * truck_pct / 100.0
        work["observed_total"] = total_vehicles

        work = work[work["observed_total"] > 0].copy()

        out_epsg = int((source_cfg.get("output") or {}).get(
            "out_epsg", get_metric_epsg(cfg)))

        from sim.calibration import _flag_neighbor_outliers
        work = _flag_neighbor_outliers(work, metric_epsg=out_epsg)
        return work
    except Exception as exc:
        _logger.warning("Pentlogram post-processing failed: %s", exc)
        return None


# --- PostgreSQL closures ---

_PG_FULL_CLOSURE_TYPES = frozenset({"road_closed", "roadClosed"})
_PG_LANE_REDUCTION_TYPES = frozenset({
    "laneClosures", "narrowLanes", "singleAlternateLineTraffic", "contraflow",
})


def _map_pg_severity(restriction_type: str, pg_severity: str) -> str:
    """Map PG ``restriction_type`` + ``severity`` to internal severity value."""
    if restriction_type in _PG_FULL_CLOSURE_TYPES:
        return "full"
    if restriction_type in _PG_LANE_REDUCTION_TYPES:
        return "lane_reduction"
    severity_map = {"standstill": "full", "serious": "lane_reduction", "moderate": "speed_limit"}
    return severity_map.get(pg_severity, "lane_reduction")


def fetch_postgres_closures(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    force: bool = False,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Fetch closure/restriction data from the PostgreSQL ``restrictions`` table.

    Produces a parquet file with the same schema consumed by
    :func:`sim.network_normalization.load_closures` (``lon``, ``lat``,
    ``severity``, ``start``, ``end``, ``road_ref``, ``description_cs``).
    """
    import psycopg2

    if gpd is None:
        raise RuntimeError("geopandas is required for postgres_closures provider")

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    _ensure_dir(cache_dir)
    cache_path = cache_dir / "closures.parquet"

    if cache_path.exists() and not force:
        _logger.info("Using cached %s", cache_path)
        gdf = gpd.read_parquet(cache_path)
        return {
            "features": int(len(gdf)),
            "parquet": str(cache_path),
            "closures_parquet": str(cache_path),
            "closures_count": int(len(gdf)),
        }

    _db_defaults = SIM_DEFAULTS["datasets"]["closures_db"]
    db_cfg = cfg.get("closures_db") or {}
    conn = psycopg2.connect(
        host=db_cfg.get("host", _db_defaults["host"]),
        port=int(db_cfg.get("port", _db_defaults["port"])),
        dbname=db_cfg.get("dbname", _db_defaults["dbname"]),
        user=db_cfg.get("user", _db_defaults["user"]),
        password=db_cfg.get("password", _db_defaults["password"]),
    )

    table = source_cfg.get("table", "restrictions")

    query = f"""
        SELECT
            id,
            restriction_type,
            restriction_subtype,
            severity       AS pg_severity,
            status,
            road_number,
            street_name,
            city,
            description_cs,
            max_speed_kmh,
            valid_from,
            valid_to,
            first_seen,
            last_seen,
            ST_Y(location_point_geog::geometry) AS lat,
            ST_X(location_point_geog::geometry) AS lon
        FROM {table}
        WHERE location_point_geog IS NOT NULL
        ORDER BY id
    """

    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if df.empty:
        _logger.info("PostgreSQL closures: no rows returned")
        return {"features": 0, "parquet": str(cache_path), "closures_count": 0}

    _logger.info("PostgreSQL closures: fetched %d rows", len(df))

    bbox = _load_aoi_bbox_wgs84(cfg)
    if bbox is not None:
        w, s, e, n = bbox
        margin = float(SIM_DEFAULTS["datasets"]["closures_db"]["bbox_margin_deg"])
        before = len(df)
        mask = (
            (df["lon"] >= w - margin) & (df["lon"] <= e + margin)
            & (df["lat"] >= s - margin) & (df["lat"] <= n + margin)
        )
        df = df[mask].copy()
        _logger.info(
            "PostgreSQL closures: %d total -> %d in model area",
            before,
            len(df),
        )

    if df.empty:
        _logger.info("PostgreSQL closures: no features in model area")
        return {"features": 0, "parquet": str(cache_path), "closures_count": 0}

    # --- Status filtering ---
    if "status" in df.columns:
        unique_statuses = sorted(df["status"].dropna().unique().tolist())
        _logger.info("PostgreSQL closures: status values found: %s", unique_statuses)

    status_whitelist = source_cfg.get("status_whitelist")
    if status_whitelist and "status" in df.columns:
        allowed = {str(s).strip().lower() for s in status_whitelist}
        before_status = len(df)
        df = df[df["status"].fillna("").astype(str).str.strip().str.lower().isin(allowed)].copy()
        _logger.info(
            "PostgreSQL closures: status filter (%s): %d -> %d",
            ", ".join(sorted(allowed)),
            before_status,
            len(df),
        )

    # --- Minimum observation duration filter ---
    min_observed_days = source_cfg.get("min_observed_days")
    if min_observed_days is not None and "first_seen" in df.columns and "last_seen" in df.columns:
        min_days = float(min_observed_days)
        fs = pd.to_datetime(df["first_seen"], errors="coerce", utc=True)
        ls = pd.to_datetime(df["last_seen"], errors="coerce", utc=True)
        duration = (ls - fs).dt.total_seconds() / 86400.0
        before_dur = len(df)
        keep = duration.isna() | (duration >= min_days)
        df = df[keep].copy()
        _logger.info(
            "PostgreSQL closures: min_observed_days=%s: %d -> %d",
            min_days,
            before_dur,
            len(df),
        )

    if df.empty:
        _logger.info("PostgreSQL closures: no features after filtering")
        return {"features": 0, "parquet": str(cache_path), "closures_count": 0}

    df["severity"] = df.apply(
        lambda r: _map_pg_severity(
            str(r.get("restriction_type", "")),
            str(r.get("pg_severity", "")),
        ),
        axis=1,
    )

    df["road_ref"] = df["road_number"].fillna("").astype(str).str.strip()

    for ts_col, fallback_col, out_col in [
        ("valid_from", "first_seen", "start"),
        ("valid_to", "last_seen", "end"),
    ]:
        primary = (
            pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            if ts_col in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        fallback = (
            pd.to_datetime(df[fallback_col], errors="coerce", utc=True)
            if fallback_col in df.columns
            else pd.Series(pd.NaT, index=df.index)
        )
        merged = primary.fillna(fallback)
        df[out_col] = merged.dt.strftime("%Y-%m-%d").fillna("")

    if "description_cs" not in df.columns:
        df["description_cs"] = ""
    else:
        df["description_cs"] = df["description_cs"].fillna("")

    from shapely.geometry import Point

    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    full_count = int((gdf["severity"] == "full").sum())
    partial_count = int((gdf["severity"] == "lane_reduction").sum())
    speed_count = int((gdf["severity"] == "speed_limit").sum())
    _logger.info(
        "PostgreSQL closures: %d events in model area "
        "(%d full, %d lane_reduction, %d speed_limit)",
        len(gdf),
        full_count,
        partial_count,
        speed_count,
    )

    gdf = _coerce_object_columns_for_parquet(gdf)
    gdf.to_parquet(cache_path, index=False)

    return {
        "features": int(len(gdf)),
        "parquet": str(cache_path),
        "closures_parquet": str(cache_path),
        "closures_count": int(len(gdf)),
    }


_NDIC_FULL_CLOSURE_KEYWORDS = (
    "uzavřen", "uzavírk", "neprůjezd", "uzavrena", "uzavirka",
    "úplná uzavírka", "úplná uzavěrka",
)

_NDIC_SEVERITY_KEYWORDS: Dict[str, str] = {
    "úplná uzavírka": "full",
    "uzavření": "full",
    "uzavřen": "full",
    "neprůjezdnost": "full",
    "částečná uzavírka": "lane_reduction",
    "omezení provozu": "lane_reduction",
    "omezení": "lane_reduction",
    "snížení rychlosti": "speed_limit",
}


def _infer_ndic_severity(row: Dict[str, Any]) -> str:
    """Infer closure severity from NDIC event/class description fields."""
    texts = []
    for key in ("event_popis1", "event_popis2", "event_popis3",
                "trida_popis1", "trida_popis2", "trida_popis3",
                "txt", "otxt", "txpl_text"):
        v = row.get(key)
        if v and str(v).strip():
            texts.append(str(v).strip().lower())
    combined = " ".join(texts)

    for keyword, severity in _NDIC_SEVERITY_KEYWORDS.items():
        if keyword.lower() in combined:
            return severity

    for kw in _NDIC_FULL_CLOSURE_KEYWORDS:
        if kw.lower() in combined:
            return "full"

    return "lane_reduction"


def _parse_ndic_date(raw: str) -> str:
    """Convert NDIC date ``DD.MM.YYYY HH:MM`` → ``YYYY-MM-DD``."""
    raw = str(raw).strip()
    if not raw:
        return ""
    try:
        parts = raw.split(" ")[0].split(".")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    except Exception:
        pass
    return raw[:10]


def _postprocess_ndic_closures(
    gdf: "gpd.GeoDataFrame",
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
) -> Optional["gpd.GeoDataFrame"]:
    """Transform NDIC closure point data into a standardized format.

    Adds ``severity``, ``road_ref``, ``start``, ``end`` columns.
    Clips to the model AOI and converts coordinates to WGS84.
    Returns a GeoDataFrame with closure attributes, or None on error.
    """
    try:
        if gdf.empty:
            _logger.info("NDIC closures: no features returned")
            return None

        work = gdf.copy()

        bbox = _load_aoi_bbox_wgs84(cfg)
        if bbox is not None:
            before = len(work)
            work_wgs = work.to_crs(epsg=4326) if (work.crs and work.crs.to_epsg() != 4326) else work
            w, s, e, n = bbox
            margin = float(SIM_DEFAULTS["datasets"]["closures_db"]["bbox_margin_deg"])
            mask = (
                (work_wgs.geometry.x >= w - margin) & (work_wgs.geometry.x <= e + margin) &
                (work_wgs.geometry.y >= s - margin) & (work_wgs.geometry.y <= n + margin)
            )
            work = work[mask.values].copy()
            _logger.info(
                "NDIC closures: %d nationwide -> %d in model area",
                before,
                len(work),
            )

        if work.empty:
            _logger.info("NDIC closures: no features in model area")
            return None

        work["severity"] = work.apply(
            lambda r: _infer_ndic_severity(r.to_dict()), axis=1,
        )

        if "cislo_silnice" in work.columns:
            work["road_ref"] = work["cislo_silnice"].fillna("").astype(str).str.strip()
        else:
            work["road_ref"] = ""

        for src_col, dst_col in [("zacatek", "start"), ("konec", "end")]:
            if src_col in work.columns:
                work[dst_col] = work[src_col].fillna("").astype(str).apply(_parse_ndic_date)
            else:
                work[dst_col] = ""

        work_wgs = work.to_crs(epsg=4326) if (work.crs and work.crs.to_epsg() != 4326) else work
        work["lon"] = work_wgs.geometry.x.values
        work["lat"] = work_wgs.geometry.y.values

        full_count = int((work["severity"] == "full").sum())
        partial_count = int((work["severity"] == "lane_reduction").sum())
        speed_count = int((work["severity"] == "speed_limit").sum())
        _logger.info(
            "NDIC closures: %d events in model area "
            "(%d full, %d lane_reduction, %d speed_limit)",
            len(work),
            full_count,
            partial_count,
            speed_count,
        )

        return work
    except Exception as exc:
        _logger.warning("NDIC closure post-processing failed: %s", exc)
        import traceback
        traceback.print_exc()
        return None


def fetch_arcgis_feature_service(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    if gpd is None:
        raise RuntimeError("geopandas is required for arcgis_feature_service provider")

    service_url = source_cfg["service_url"]
    layer_id = int(source_cfg["layer_id"])
    out_path = Path(source_cfg["out_path"])
    out_epsg = int((source_cfg.get("output") or {}).get("out_epsg", get_metric_epsg(cfg)))

    query_cfg = source_cfg.get("query") or {}
    bbox = _load_aoi_bbox_wgs84(cfg) if (query_cfg.get("geometry_clip") or {}).get("enabled", False) and (query_cfg.get("geometry_clip") or {}).get("use_aoi", False) else None

    _ensure_dir(out_path.parent)

    if out_path.exists() and not force:
        _logger.info("Using cached %s", out_path)
    else:
        src_timeout = int(source_cfg.get("timeout_s", timeout_s))
        geojson = arcgis_query_geojson(
            service_url,
            layer_id,
            where=str(query_cfg.get("where", "1=1")),
            out_fields=query_cfg.get("out_fields") or ["*"],
            bbox_wgs84=bbox,
            timeout_s=max(timeout_s, src_timeout),
            retries=retries,
            headers=headers,
        )
        out_path.write_text(json.dumps(geojson), encoding="utf-8")

    gdf = gpd.read_file(out_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=out_epsg, allow_override=True)
    elif gdf.crs.to_epsg() != out_epsg:
        gdf = gdf.to_crs(epsg=out_epsg)

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    _ensure_dir(cache_dir)
    cache_parquet = cache_dir / f"{_slug(out_path.stem)}.parquet"
    gdf.to_parquet(cache_parquet, index=False)
    info = {"features": int(len(gdf)), "geojson": str(out_path), "parquet": str(cache_parquet), "epsg": out_epsg}

    usage = (source_cfg.get("usage") or {})
    if usage.get("calibration_target") == "link_counts":
        cleaned = _postprocess_pentlogram(gdf, cfg, source_cfg)
        if cleaned is not None:
            cleaned_path = cache_dir / f"{_slug(out_path.stem)}_cleaned.parquet"
            cleaned.to_parquet(cleaned_path, index=False)
            info["cleaned_parquet"] = str(cleaned_path)
            info["cleaned_features"] = int(len(cleaned))
            _logger.info(
                "Pentlogram cleaned: %d -> %d segments",
                len(gdf),
                len(cleaned),
            )

    if usage.get("calibration_target") == "baseline_closures":
        closures_gdf = _postprocess_ndic_closures(gdf, cfg, source_cfg)
        if closures_gdf is not None:
            closures_path = cache_dir / "closures_ndic.parquet"
            closures_gdf = _coerce_object_columns_for_parquet(closures_gdf)
            closures_gdf.to_parquet(closures_path, index=False)
            info["closures_parquet"] = str(closures_path)
            info["closures_count"] = int(len(closures_gdf))

    return info


# --- Police CR traffic-info (XML to closures JSON) ---

# Event codes that indicate a closure or restriction (not an accident report).
_CLOSURE_EVENT_CODES = {
    401, 402, 403, 404, 405,   # road works / construction
    500, 501, 502, 503, 504,   # lane(s) closed
    505, 506, 507,             # carriageway closed / blocked
    508, 509, 510,             # road closed / impassable
    700, 701, 702, 703, 704,   # maintenance / resurfacing
    801, 802, 803, 804,        # roadwork signs, temporary signals
    981, 982, 983,             # obstruction on road
}

# Codes that imply only a speed or lane restriction, not full closure.
_LANE_REDUCTION_CODES = {500, 501, 502, 503, 504, 802, 803, 804}


def _parse_police_xml_closures(
    xml_text: str,
    *,
    bbox_wgs84: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """Parse the Police CR traffic-info XML and extract closure/restriction events."""
    closures: List[Dict[str, Any]] = []
    msg_pattern = re.compile(r"<MSG\b.*?</MSG>", re.DOTALL)
    tsta_pattern = re.compile(r'<TSTA\s+date="([^"]+)"\s+time="([^"]*)"')
    tsto_pattern = re.compile(r'<TSTO\s+date="([^"]+)"\s+time="([^"]*)"')
    mtxt_pattern = re.compile(r"<MTXT[^>]*>(.*?)</MTXT>", re.DOTALL)
    msgid_pattern = re.compile(r'<MSG\s+id="([^"]*)"')
    sbeg_pattern = re.compile(r'<SBEG\s+x="([^"]+)"\s+y="([^"]+)"')
    send_pattern = re.compile(r'<SEND\s+x="([^"]+)"\s+y="([^"]+)"')
    evi_pattern = re.compile(r'<EVI\s+eventcode="(\d+)"')

    for msg_match in msg_pattern.finditer(xml_text):
        msg = msg_match.group(0)

        codes = {int(c) for c in evi_pattern.findall(msg)}
        relevant = codes & _CLOSURE_EVENT_CODES
        if not relevant:
            continue

        sbeg = sbeg_pattern.search(msg)
        if not sbeg:
            continue
        lon, lat = float(sbeg.group(1)), float(sbeg.group(2))

        if bbox_wgs84:
            w, s, e, n = bbox_wgs84
            margin = float(SIM_DEFAULTS["datasets"]["closures_db"]["bbox_margin_deg"])
            if not (w - margin <= lon <= e + margin and s - margin <= lat <= n + margin):
                continue

        send = send_pattern.search(msg)
        lon_end = float(send.group(1)) if send else lon
        lat_end = float(send.group(2)) if send else lat

        tsta = tsta_pattern.search(msg)
        tsto = tsto_pattern.search(msg)
        start_date = tsta.group(1) if tsta else None
        end_date = tsto.group(1) if tsto else None

        mtxt = mtxt_pattern.search(msg)
        text = mtxt.group(1).strip() if mtxt else ""

        msgid = msgid_pattern.search(msg)
        msg_id = msgid.group(1) if msgid else ""

        road_ref_match = re.search(r"(?:na silnici|na dálnici|na komunikaci)\s+(\S+)", text)
        road_ref_d = re.search(r"\b(D\d+)\b", text)
        road_ref = road_ref_match.group(1) if road_ref_match else (road_ref_d.group(1) if road_ref_d else None)

        muni_match = re.search(r"v obci\s+([^;,]+)", text)
        municipality = muni_match.group(1).strip() if muni_match else None

        if relevant & _LANE_REDUCTION_CODES and not (relevant - _LANE_REDUCTION_CODES):
            severity = "lane_reduction"
        elif any(w in text.lower() for w in ("uzavřen", "uzavírk", "neprůjezd")):
            severity = "full"
        else:
            severity = "lane_reduction"

        closures.append({
            "id": msg_id,
            "start": start_date,
            "end": end_date,
            "text": text,
            "lon": lon,
            "lat": lat,
            "lon_end": lon_end,
            "lat_end": lat_end,
            "severity": severity,
            "road_ref": road_ref,
            "municipality": municipality,
            "event_codes": sorted(relevant),
        })

    return closures


def fetch_police_traffic_xml(
    cfg: Dict[str, Any],
    source_cfg: Dict[str, Any],
    *,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    """Fetch Police CR traffic-info XML, extract closures, write JSON cache."""
    url = source_cfg.get("url", "http://aplikace.policie.cz/dopravni-informace/GetFile.aspx")
    out_path = Path(source_cfg["out_path"])

    if out_path.exists() and not force:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return {
            "status": "cached",
            "path": str(out_path),
            "closures": len(data.get("closures", [])),
        }

    bbox = _load_aoi_bbox_wgs84(cfg)

    last_err: Optional[Exception] = None
    xml_text: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout_s, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            xml_text = resp.text
            break
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(float(attempt))

    if xml_text is None:
        raise RuntimeError(f"Failed to fetch Police traffic XML from {url}: {last_err}")

    closures = _parse_police_xml_closures(xml_text, bbox_wgs84=bbox)

    result = {
        "fetched_at": _now_iso(),
        "source_url": url,
        "bbox_wgs84": list(bbox) if bbox else None,
        "total_xml_size": len(xml_text),
        "closures": closures,
    }

    _ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    n_msg = len(re.findall(r"<MSG ", xml_text))
    _logger.info(
        "Police closures: %d events in model area (from %d total messages)",
        len(closures),
        n_msg,
    )

    return {
        "status": "fetched",
        "path": str(out_path),
        "closures": len(closures),
    }


# --- Provider handlers ---

def _default_cache_parquet(cache_dir: Path, out_path: Path) -> Path:
    return cache_dir / f"{_slug(out_path.stem)}.parquet"


def _handle_http_file(
    cfg: Dict[str, Any],
    scfg: Dict[str, Any],
    *,
    cache_dir: Path,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    out_path = Path(scfg["out_path"])
    info: Dict[str, Any] = {
        "download": download_file(scfg["url"], out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
    }

    fmt_cfg = scfg.get("format") or {}
    usage = scfg.get("usage") or {}
    fmt_type = fmt_cfg.get("type")
    if fmt_type is None:
        ext = out_path.suffix.lstrip(".").lower()
        if ext in ("xlsx", "xls"):
            fmt_type = "xlsx"
        elif ext == "csv":
            fmt_type = "csv"

    if fmt_type == "xlsx":
        out_parquet = Path(scfg.get("out_parquet") or str(out_path.parent / f"{_slug(out_path.stem)}.parquet"))
        if usage.get("validation_target") == "aadt_screenlines" or usage.get("calibration_target") == "aadt_screenlines":
            info["preprocess"] = preprocess_csd_xlsx(out_path, out_parquet)
        else:
            info["preprocess"] = preprocess_xlsx_table(out_path, out_parquet)
        return info

    if fmt_type == "csv" and usage.get("socioeconomic") == "population_per_zone":
        zones_path = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones")) / "zones.geojson"
        out_parquet = cache_dir / "zone_population.parquet"
        info["preprocess"] = preprocess_population_sldb2021(
            out_path,
            out_parquet,
            zones_geojson=zones_path if zones_path.exists() else None,
            delimiter=str(fmt_cfg.get("delimiter", "auto")),
            encoding=str(fmt_cfg.get("encoding", "auto")),
            cfg=cfg,
        )
    return info


def _handle_csu_open_data_csv(
    scfg: Dict[str, Any],
    *,
    cache_dir: Path,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    out_path = Path(scfg["out_path"])
    fmt = scfg.get("format") or {}
    preprocess_cfg = scfg.get("preprocess") or {}
    info: Dict[str, Any] = {
        "download": download_file(scfg["url"], out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force),
        "preprocess": {},
    }

    delimiter = str(fmt.get("delimiter", "auto"))
    encoding = str(fmt.get("encoding", "auto"))

    if bool(preprocess_cfg.get("write_filtered", True)):
        filtered_out = Path(scfg.get("filtered_out_parquet") or (cache_dir / f"{_slug(out_path.stem)}.parquet"))
        info["preprocess"]["filtered"] = preprocess_commuting_sldb2021(
            out_path,
            filtered_out,
            delimiter=delimiter,
            encoding=encoding,
            filter_cfg=scfg.get("filter") or {},
        )

    if bool(preprocess_cfg.get("write_full_cr", False)):
        full_out = Path(scfg.get("full_cr_out_parquet") or (out_path.parent / f"{out_path.stem}_full_cr.parquet"))
        info["preprocess"]["full_cr"] = preprocess_commuting_sldb2021(
            out_path,
            full_out,
            delimiter=delimiter,
            encoding=encoding,
            filter_cfg={"enabled": False},
        )

    return info


def _handle_atom_file(
    scfg: Dict[str, Any],
    *,
    cache_dir: Path,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    out_path = Path(scfg["out_path"])
    fmt_cfg = scfg.get("format") or {}
    usage = scfg.get("usage") or {}

    info: Dict[str, Any] = {
        "download": download_atom_latest_file(
            scfg["url"],
            out_path,
            timeout_s=timeout_s,
            retries=retries,
            headers=headers,
            force=force,
            asset_pattern=fmt_cfg.get("asset_pattern"),
            feed_cache_path=Path(scfg["feed_cache_path"]) if scfg.get("feed_cache_path") else None,
        )
    }

    if usage.get("supernetwork_places") == "grouped_point_centroids":
        out_parquet = Path(scfg.get("out_parquet") or str(out_path.parent / "cz_place_centroids.parquet"))
        if fmt_cfg.get("type") == "zip_csv":
            info["preprocess"] = preprocess_grouped_points_zip_to_centroids(
                out_path,
                out_parquet,
                member_pattern=fmt_cfg.get("member_pattern"),
                delimiter=str(fmt_cfg.get("delimiter", "auto")),
                encoding=str(fmt_cfg.get("encoding", "auto")),
                source_crs_epsg=int(fmt_cfg.get("source_crs_epsg", 5514)),
                output_crs_epsg=int(fmt_cfg.get("output_crs_epsg", 4326)),
                columns_cfg=scfg.get("columns") or {},
            )
        else:
            extracted_path = Path(scfg.get("extracted_path") or out_path.with_suffix(".csv"))
            _ensure_dir(extracted_path.parent)
            with zipfile.ZipFile(out_path, "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    raise RuntimeError(f"No files found in ZIP archive: {out_path}")
                selected = next((n for n in names if re.search(str(fmt_cfg.get("member_pattern")), n)), names[0]) if fmt_cfg.get("member_pattern") else names[0]
                extracted_path.write_bytes(zf.read(selected))
            info["extract"] = {"zip_path": str(out_path), "member_name": selected, "path": str(extracted_path), "bytes": extracted_path.stat().st_size}
            info["preprocess"] = preprocess_grouped_points_to_centroids(
                extracted_path,
                out_parquet,
                delimiter=str(fmt_cfg.get("delimiter", "auto")),
                encoding=str(fmt_cfg.get("encoding", "auto")),
                source_crs_epsg=int(fmt_cfg.get("source_crs_epsg", 5514)),
                output_crs_epsg=int(fmt_cfg.get("output_crs_epsg", 4326)),
                columns_cfg=scfg.get("columns") or {},
            )

    return info


# --- Main runner ---

def run_fetch_datasets(
    config_path: str | Path = "config/brno/sim.yaml",
    force: bool = False,
    only: Optional[List[str]] = None,
) -> None:
    cfg = load_config(config_path)
    ds = cfg.get("datasets") or {}
    if ds.get("enabled") is False:
        _logger.info("datasets.enabled=false, nothing to do")
        return

    timeout_s = int((ds.get("http") or {}).get("timeout_s", 60))
    retries = int((ds.get("http") or {}).get("retries", 3))
    user_agent = str((ds.get("http") or {}).get("user_agent", "simulation-pipeline/1.0"))
    headers = {"User-Agent": user_agent}

    cache_dir = Path(ds.get("cache_dir", "data/cache"))
    _ensure_dir(cache_dir)

    manifest: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "config": str(config_path),
        "sources": {},
    }

    errors: list[tuple[str, str]] = []
    selected = set(only) if only else None
    all_sources = _merge_dataset_sources(ds.get("sources"))
    for key, scfg in all_sources.items():
        if selected is not None and key not in selected:
            continue
        if not (scfg or {}).get("enabled", False):
            continue

        provider = str(scfg.get("provider", "")).strip()
        _logger.info("%s: provider=%s", key, provider)

        try:
            if provider == "http_file":
                info = _handle_http_file(
                    cfg,
                    scfg,
                    cache_dir=cache_dir,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                    force=force,
                )
            elif provider == "csu_open_data_csv":
                info = _handle_csu_open_data_csv(
                    scfg,
                    cache_dir=cache_dir,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                    force=force,
                )
            elif provider == "atom_file":
                info = _handle_atom_file(
                    scfg,
                    cache_dir=cache_dir,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                    force=force,
                )
            elif provider == "arcgis_feature_service":
                info = fetch_arcgis_feature_service(
                    cfg,
                    scfg,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                    force=force,
                )
            elif provider == "police_traffic_xml":
                info = fetch_police_traffic_xml(
                    cfg,
                    scfg,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                    force=force,
                )
            elif provider == "postgres_closures":
                info = fetch_postgres_closures(
                    cfg,
                    scfg,
                    force=force,
                )
            else:
                raise ValueError(f"Unknown provider '{provider}' for source '{key}'")
        except Exception as exc:
            _logger.error("%s: %s", key, exc)
            errors.append((key, str(exc)))
            manifest["sources"][key] = {"error": str(exc)}
            continue

        manifest["sources"][key] = info

    manifest_path = cache_dir / "datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _logger.info("Wrote manifest: %s", manifest_path)

    if errors:
        names = ", ".join(k for k, _ in errors)
        _logger.warning("%d source(s) failed: %s", len(errors), names)
        for k, msg in errors:
            _logger.warning("  %s: %s", k, msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/brno/sim.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    run_fetch_datasets(config_path=args.config, force=args.force, only=args.only)


if __name__ == "__main__":
    main()
