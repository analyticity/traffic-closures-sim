"""Shared utilities for dataset fetching and preprocessing.

CSV reading, file downloads, ATOM feed parsing, column resolution,
parquet coercion, AOI bbox loading.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from sim._text import norm_col as _norm_col

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    target = _norm_col(wanted)
    for col in df.columns:
        if _norm_col(col) == target:
            return col
    return None


def resolve_input_col(
    df: pd.DataFrame,
    columns_cfg: Dict[str, Any],
    cfg_key: str,
    candidates: Iterable[str],
) -> Optional[str]:
    configured = columns_cfg.get(cfg_key)
    if configured:
        if configured in df.columns:
            return configured
        found = find_col(df, str(configured))
        if found:
            return found
    for cand in candidates:
        found = find_col(df, cand)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# Parquet / DataFrame coercion
# ---------------------------------------------------------------------------

def coerce_object_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].where(out[col].notna(), None).astype("string")
    return out


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

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


def read_csv_flexible(path: Path, *, delimiter: str = "auto", encoding: str = "auto") -> pd.DataFrame:
    fallback_encodings = ["cp1250", "windows-1250", "utf-8", "utf-8-sig", "iso-8859-2"]
    encodings = (
        [encoding] + [e for e in fallback_encodings if e != encoding]
        if encoding not in ("", None, "auto")
        else fallback_encodings
    )
    delimiters: List[Optional[str]] = (
        [",", ";", "\t", "|", None]
        if delimiter in ("", None, "auto")
        else [delimiter]
    )

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


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    out_path: Path,
    *,
    timeout_s: int,
    retries: int,
    headers: Dict[str, str],
    force: bool,
) -> Dict[str, Any]:
    ensure_dir(out_path.parent)
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
    ensure_dir(out_path.parent)
    if feed_cache_path is not None:
        ensure_dir(feed_cache_path.parent)

    if out_path.exists() and not force:
        return {
            "status": "cached", "feed_url": feed_url,
            "path": str(out_path), "bytes": out_path.stat().st_size,
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
    info = download_file(asset_url, out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
    info.update({"feed_url": feed_url, "asset_url": asset_url, "asset_candidates": hrefs_unique[:10]})
    if feed_cache_path is not None:
        info["feed_cache_path"] = str(feed_cache_path)
    return info


# ---------------------------------------------------------------------------
# AOI bounding box
# ---------------------------------------------------------------------------

def load_aoi_bbox_wgs84(cfg: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    mb = cfg.get("model_bbox")
    if mb and len(mb) == 4 and None not in mb:
        return tuple(float(x) for x in mb)

    if gpd is None:
        return None

    zones_dir = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))
    maps_dir = Path(cfg.get("network", {}).get("maps_dir", "outputs/baseline/maps"))
    for candidate in [zones_dir / "model_area.geojson", maps_dir / "model_bbox_wgs84.geojson"]:
        if candidate.exists():
            try:
                gdf_c = gpd.read_file(candidate)
                if gdf_c.crs is None:
                    gdf_c = gdf_c.set_crs(epsg=4326)
                elif gdf_c.crs.to_epsg() != 4326:
                    gdf_c = gdf_c.to_crs(epsg=4326)
                return tuple(float(x) for x in gdf_c.total_bounds)
            except Exception:
                pass
    return None
