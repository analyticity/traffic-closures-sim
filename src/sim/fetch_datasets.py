#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

# Při spuštění jako skript přidat src do path pro import sim
if __name__ == "__main__":
    _src_dir = Path(__file__).resolve().parent.parent
    if _src_dir.exists() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

from sim.io_project import load_config


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _norm_col(c: str) -> str:
    c = c.strip().lower()
    c = c.replace(" ", "_")
    c = re.sub(r"[^a-z0-9_]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    target = _norm_col(name)
    for c in df.columns:
        if _norm_col(str(c)) == target:
            return c
    return None


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
            r = requests.get(url, timeout=timeout_s, headers=headers)
            r.raise_for_status()
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with tmp.open("wb") as f:
                f.write(r.content)
            tmp.replace(out_path)
            return {"status": "downloaded", "path": str(out_path), "bytes": out_path.stat().st_size}
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.0 * attempt)
    raise RuntimeError(f"Failed to download {url} -> {out_path}: {last_err}")


def preprocess_commuting_sldb2021(
    src_csv: Path,
    out_parquet: Path,
    *,
    delimiter: str,
    encoding: str,
    filter_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    df = pd.read_csv(src_csv, sep=delimiter, encoding=encoding, low_memory=False)
    original_cols = list(df.columns)

    if (filter_cfg or {}).get("enabled", False):
        origin_cfg = (filter_cfg.get("origin") or {})
        dest_cfg = (filter_cfg.get("destination") or {})

        def _apply_one_side(side_df: pd.DataFrame, side: str, cfg: Dict[str, Any]) -> pd.DataFrame:
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
                values = clause.get("values") or []
                vals_norm = [str(v).strip().lower() for v in values]
                col_vals = side_df[col].astype(str).str.strip().str.lower()
                mask = mask | col_vals.isin(vals_norm)
            return side_df[mask].copy()

        df = _apply_one_side(df, "origin", origin_cfg)

        if not dest_cfg.get("keep_all", True) and dest_cfg.get("keep_if_any_matches"):
            df = _apply_one_side(df, "destination", dest_cfg)

    df.columns = [_norm_col(str(c)) for c in df.columns]
    _ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)

    return {"rows": int(len(df)), "cols": int(len(df.columns)), "out_parquet": str(out_parquet)}


def arcgis_query_geojson(
    service_url: str,
    layer_id: int,
    *,
    where: str,
    out_fields: List[str],
    bbox_wgs84: Optional[Tuple[float, float, float, float]],
    in_epsg: int,
    out_epsg: int,
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
        "outSR": str(out_epsg),
        "resultRecordCount": str(page_size),
    }
    if bbox_wgs84 is not None:
        w, s, e, n = bbox_wgs84
        params_common.update(
            {
                "geometry": f"{w},{s},{e},{n}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": str(in_epsg),
                "spatialRel": "esriSpatialRelIntersects",
            }
        )

    last_err: Optional[Exception] = None
    while True:
        params = dict(params_common)
        params["resultOffset"] = str(offset)

        for attempt in range(1, retries + 1):
            try:
                r = requests.get(base, params=params, timeout=timeout_s, headers=headers)
                r.raise_for_status()
                js = r.json()
                feats = js.get("features") or []
                all_features.extend(feats)
                exceeded = bool(js.get("exceededTransferLimit", False))
                break
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(1.0 * attempt)
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

    output_cfg = source_cfg.get("output") or {}
    out_epsg = int(output_cfg.get("out_epsg", cfg.get("crs_epsg", 5514)))

    query_cfg = source_cfg.get("query") or {}
    where = str(query_cfg.get("where", "1=1"))
    out_fields = query_cfg.get("out_fields") or ["*"]

    bbox = None
    geometry_clip = (query_cfg.get("geometry_clip") or {})
    if geometry_clip.get("enabled", False) and geometry_clip.get("use_aoi", False):
        w, s, e, n = cfg.get("model_bbox", [None, None, None, None])
        if None not in (w, s, e, n):
            bbox = (float(w), float(s), float(e), float(n))

    geojson = arcgis_query_geojson(
        service_url,
        layer_id,
        where=where,
        out_fields=out_fields,
        bbox_wgs84=bbox,
        in_epsg=4326,
        out_epsg=out_epsg,
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
    elif out_epsg and gdf.crs.to_epsg() != out_epsg:
        gdf = gdf.to_crs(epsg=out_epsg)

    cache_dir = Path(cfg["datasets"]["cache_dir"])
    _ensure_dir(cache_dir)
    cache_parquet = cache_dir / f"{_slug(out_path.stem)}.parquet"
    gdf.to_parquet(cache_parquet, index=False)

    return {"features": int(len(gdf)), "geojson": str(out_path), "parquet": str(cache_parquet), "epsg": out_epsg}


def preprocess_csd2020_xlsx(xlsx_path: Path, out_parquet: Path) -> Dict[str, Any]:
    xls = pd.ExcelFile(xlsx_path)
    sheets = xls.sheet_names

    best_sheet = None
    best_score = -1
    best_df = None
    best_header_row = 0

    key_cols = {"sv", "o", "tv", "sil", "rpdi"}
    for sh in sheets:
        for hrow in (0, 1, 2):
            try:
                df = xls.parse(sh, header=hrow, nrows=50)
            except Exception:
                continue
            cols = [_norm_col(str(c)) for c in df.columns]
            score = sum(1 for c in cols if c in key_cols)
            if score > best_score:
                best_score = score
                best_sheet = sh
                best_header_row = hrow
                best_df = xls.parse(sh, header=hrow)

    if best_df is None:
        raise RuntimeError(f"Could not parse any sheet from {xlsx_path}")

    df = best_df.copy()
    df.columns = [_norm_col(str(c)) for c in df.columns]
    df.insert(0, "sheet", str(best_sheet))

    # Sloupce object (smíšené typy z Excelu) převést na string kvůli Parquet
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)

    _ensure_dir(out_parquet.parent)
    df.to_parquet(out_parquet, index=False)

    summary = {"sheets": sheets, "chosen_sheet": best_sheet, "rows": int(len(df)), "cols": int(len(df.columns))}
    summary_path = out_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"parquet": str(out_parquet), "summary": str(summary_path), **summary}


def run_fetch_datasets(
    config_path: str | Path = "config/sim.yaml",
    force: bool = False,
    only: Optional[List[str]] = None,
) -> None:
    """Stáhne a připraví datové sady dle config.datasets (volatelné z run.py i jako skript)."""
    cfg = load_config(config_path)
    ds = (cfg.get("datasets") or {})
    if not ds.get("enabled", False):
        print("datasets.enabled=false -> nothing to do")
        return

    timeout_s = int((ds.get("http") or {}).get("timeout_s", 60))
    retries = int((ds.get("http") or {}).get("retries", 3))
    user_agent = str((ds.get("http") or {}).get("user_agent", "simulation-pipeline/1.0"))
    headers = {"User-Agent": user_agent}

    cache_dir = Path(ds.get("cache_dir", "data/cache"))
    _ensure_dir(cache_dir)

    manifest: Dict[str, Any] = {"generated_at": _now_iso(), "config": str(config_path), "sources": {}}

    sources = ds.get("sources") or {}
    selected = set(only) if only else None

    for key, scfg in sources.items():
        if selected is not None and key not in selected:
            continue
        if not (scfg or {}).get("enabled", False):
            continue

        provider = str(scfg.get("provider", "")).strip()
        print(f"[{key}] provider={provider}")

        if provider == "http_file":
            url = scfg["url"]
            out_path = Path(scfg["out_path"])
            dl = download_file(url, out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
            info: Dict[str, Any] = {"download": dl}

            fmt = (scfg.get("format") or {}).get("type")
            if fmt == "xlsx":
                out_parquet = cache_dir / f"{_slug(out_path.stem)}.parquet"
                info["preprocess"] = preprocess_csd2020_xlsx(out_path, out_parquet)

            manifest["sources"][key] = info
            continue

        if provider == "csu_open_data_csv":
            url = scfg["url"]
            out_path = Path(scfg["out_path"])
            dl = download_file(url, out_path, timeout_s=timeout_s, retries=retries, headers=headers, force=force)

            fmt = scfg.get("format") or {}
            delimiter = str(fmt.get("delimiter", ";"))
            encoding = str(fmt.get("encoding", "utf-8"))

            out_parquet = cache_dir / f"{_slug(out_path.stem)}.parquet"
            pre = preprocess_commuting_sldb2021(
                out_path,
                out_parquet,
                delimiter=delimiter,
                encoding=encoding,
                filter_cfg=scfg.get("filter") or {},
            )
            manifest["sources"][key] = {"download": dl, "preprocess": pre}
            continue

        if provider == "arcgis_feature_service":
            info = fetch_arcgis_feature_service(cfg, scfg, timeout_s=timeout_s, retries=retries, headers=headers, force=force)
            manifest["sources"][key] = info
            continue

        raise ValueError(f"Unknown provider '{provider}' for source '{key}'")

    manifest_path = cache_dir / "datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Wrote manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sim.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    run_fetch_datasets(config_path=args.config, force=args.force, only=args.only)


if __name__ == "__main__":
    main()
    