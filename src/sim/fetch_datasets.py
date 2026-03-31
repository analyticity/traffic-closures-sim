#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
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

# Allow running as a standalone script
if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent.parent
    if _src_dir.exists() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

from sim.io_project import load_config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def _strip_diacritics(text: Any) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def _norm_col(name: Any) -> str:
    text = _strip_diacritics(name).strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


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


# ---------------------------------------------------------------------------
# Generic grouped-points preprocessing
# ---------------------------------------------------------------------------

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

    df = pd.concat(frames, ignore_index=True)
    return _build_grouped_point_centroids_from_df(
        df,
        out_parquet,
        source_crs_epsg=source_crs_epsg,
        output_crs_epsg=output_crs_epsg,
        columns_cfg=columns_cfg,
        extra_info={"zip_members_used": len(matched_members)},
    )


# ---------------------------------------------------------------------------
# Specific preprocessors
# ---------------------------------------------------------------------------

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
            score = sum(1 for c in cols_norm if c in key_cols) if key_cols else len([c for c in cols_norm if c])
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
    return preprocess_xlsx_table(
        xlsx_path,
        out_parquet,
        key_cols={"sv", "o", "tv", "sil", "rpdi"},
        max_header_row_scan=4,
        summary_tag="csd_xlsx",
    )


_MC_TO_KU: Dict[str, List[str]] = {
    "Brno-střed": ["Město Brno", "Staré Brno", "Veveří", "Pisárky", "Stránice", "Zábrdovice", "Trnitá", "Štýřice"],
    "Brno-Žabovřesky": ["Žabovřesky"],
    "Brno-Královo Pole": ["Královo Pole", "Ponava", "Sadová"],
    "Brno-sever": ["Husovice", "Černá Pole", "Lesná", "Soběšice"],
    "Brno-Židenice": ["Židenice"],
    "Brno-Černovice": ["Černovice"],
    "Brno-jih": ["Komárov", "Horní Heršpice", "Dolní Heršpice", "Přízřenice"],
    "Brno-Bohunice": ["Bohunice"],
    "Brno-Starý Lískovec": ["Starý Lískovec"],
    "Brno-Nový Lískovec": ["Nový Lískovec"],
    "Brno-Kohoutovice": ["Kohoutovice"],
    "Brno-Jundrov": ["Jundrov"],
    "Brno-Bystrc": ["Bystrc"],
    "Brno-Kníničky": ["Kníničky"],
    "Brno-Komín": ["Komín"],
    "Brno-Medlánky": ["Medlánky"],
    "Brno-Řečkovice a Mokrá Hora": ["Řečkovice", "Mokrá Hora"],
    "Brno-Maloměřice a Obřany": ["Maloměřice", "Obřany"],
    "Brno-Líšeň": ["Líšeň"],
    "Brno-Slatina": ["Slatina"],
    "Brno-Tuřany": ["Tuřany", "Brněnské Ivanovice", "Holásky", "Dvorska"],
    "Brno-Chrlice": ["Chrlice"],
    "Brno-Bosonohy": ["Bosonohy"],
    "Brno-Ivanovice": ["Ivanovice"],
    "Brno-Žebětín": ["Žebětín"],
    "Brno-Jehnice": ["Jehnice"],
    "Brno-Útěchov": ["Útěchov u Brna"],
    "Brno-Ořešín": ["Ořešín"],
}


def preprocess_population_sldb2021(
    csv_path: Path,
    out_parquet: Path,
    zones_geojson: Optional[Path] = None,
    *,
    delimiter: str = "auto",
    encoding: str = "auto",
) -> Dict[str, Any]:
    import difflib

    def norm(text: Any) -> str:
        t = unicodedata.normalize("NFKD", str(text).strip())
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        return t.lower().strip()

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

    brno_mc = total[(pd.to_numeric(total["uzemi_cis"], errors="coerce") == 44) & total["nazev"].str.startswith("Brno", na=False)]
    mc_pop = {str(r["nazev"]).strip(): int(r["hodnota"]) for _, r in brno_mc.iterrows()}

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
    ku_to_mc = {norm(ku): mc for mc, ku_list in _MC_TO_KU.items() for ku in ku_list}

    zones_m = zones.to_crs(epsg=5514)
    zone_area = {int(r["zone_id"]): r.geometry.area for _, r in zones_m.iterrows()}

    mc_zone_ids: Dict[str, List[int]] = {}
    zone_mc_map: Dict[int, str] = {}
    for _, zrow in zones.iterrows():
        zid = int(zrow["zone_id"])
        znorm = norm(zrow["name"])
        if znorm in ku_to_mc:
            mc = ku_to_mc[znorm]
            mc_zone_ids.setdefault(mc, []).append(zid)
            zone_mc_map[zid] = mc

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
        for suffix in [" u brna", " nad svitavou"]:
            stripped = znorm.replace(suffix, "")
            if stripped in obec_norm:
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
            avg = int(mc_pop.get("Brno-střed", 70000) / 8)
            result_rows.append({"zone_id": zid, "zone_name": zname, "population": avg, "match": "default_avg"})

    result = _coerce_object_columns_for_parquet(pd.DataFrame(result_rows))
    _ensure_dir(out_parquet.parent)
    result.to_parquet(out_parquet, index=False)

    matched = len([r for r in result_rows if r["match"] != "default_avg"])
    total_pop = int(result["population"].sum())
    return {
        "parquet": str(out_parquet),
        "zones_total": len(result_rows),
        "matched": matched,
        "total_population": total_pop,
    }


# ---------------------------------------------------------------------------
# ArcGIS helpers
# ---------------------------------------------------------------------------

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
        if not exceeded:
            break

    return {"type": "FeatureCollection", "features": all_features}


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
    out_epsg = int((source_cfg.get("output") or {}).get("out_epsg", cfg.get("crs_epsg", 5514)))

    query_cfg = source_cfg.get("query") or {}
    bbox = _load_aoi_bbox_wgs84(cfg) if (query_cfg.get("geometry_clip") or {}).get("enabled", False) and (query_cfg.get("geometry_clip") or {}).get("use_aoi", False) else None

    geojson = arcgis_query_geojson(
        service_url,
        layer_id,
        where=str(query_cfg.get("where", "1=1")),
        out_fields=query_cfg.get("out_fields") or ["*"],
        bbox_wgs84=bbox,
        timeout_s=timeout_s,
        retries=retries,
        headers=headers,
    )

    _ensure_dir(out_path.parent)
    if (not out_path.exists()) or force:
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
    return {"features": int(len(gdf)), "geojson": str(out_path), "parquet": str(cache_parquet), "epsg": out_epsg}


# ---------------------------------------------------------------------------
# Provider handlers
# ---------------------------------------------------------------------------

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

    if fmt_type == "xlsx":
        out_parquet = _default_cache_parquet(cache_dir, out_path)
        if usage.get("validation_target") == "aadt_screenlines" or usage.get("calibration_target") == "aadt_screenlines":
            info["preprocess"] = preprocess_csd_xlsx(out_path, out_parquet)
        else:
            info["preprocess"] = preprocess_xlsx_table(out_path, out_parquet)
        return info

    if fmt_type == "csv" and usage.get("socioeconomic") == "population_per_zone":
        zones_path = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones")) / "zones.geojson"
        out_parquet = _default_cache_parquet(cache_dir, out_path)
        info["preprocess"] = preprocess_population_sldb2021(
            out_path,
            out_parquet,
            zones_geojson=zones_path if zones_path.exists() else None,
            delimiter=str(fmt_cfg.get("delimiter", "auto")),
            encoding=str(fmt_cfg.get("encoding", "auto")),
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
        full_out = Path(scfg.get("full_cr_out_parquet") or (cache_dir / f"{_slug(out_path.stem)}_full_cr.parquet"))
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
        out_parquet = Path(scfg["out_parquet"])
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_fetch_datasets(
    config_path: str | Path = "config/sim.yaml",
    force: bool = False,
    only: Optional[List[str]] = None,
) -> None:
    cfg = load_config(config_path)
    ds = cfg.get("datasets") or {}
    if not ds.get("enabled", False):
        print("datasets.enabled=false -> nothing to do")
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

    selected = set(only) if only else None
    for key, scfg in (ds.get("sources") or {}).items():
        if selected is not None and key not in selected:
            continue
        if not (scfg or {}).get("enabled", False):
            continue

        provider = str(scfg.get("provider", "")).strip()
        print(f"[{key}] provider={provider}")

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
        else:
            raise ValueError(f"Unknown provider '{provider}' for source '{key}'")

        manifest["sources"][key] = info

    manifest_path = cache_dir / "datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Wrote manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    run_fetch_datasets(config_path=args.config, force=args.force, only=args.only)


if __name__ == "__main__":
    main()
