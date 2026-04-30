"""RUIAN grouped point centroids preprocessing."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from sim.datasets.utils import (
    coerce_numeric_series,
    coerce_object_columns_for_parquet,
    ensure_dir,
    read_csv_flexible,
    resolve_input_col,
)


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

    place_code_col = resolve_input_col(df, columns_cfg, "place_code", ["kod_obce", "obec_kod", "municipality_code", "place_code"])
    place_name_col = resolve_input_col(df, columns_cfg, "place_name", ["nazev_obce", "obec_nazev", "obec", "municipality_name", "place_name", "name"])
    district_name_col = resolve_input_col(df, columns_cfg, "district_name", ["nazev_momc", "nazev_obvodu_prahy", "nazev_casti_obce", "district_name", "district"])
    admin_level_col = resolve_input_col(df, columns_cfg, "admin_level", ["admin_level", "level", "typ", "type"])
    x_col = resolve_input_col(df, columns_cfg, "x", ["souradnice_x", "x", "lon", "longitude"])
    y_col = resolve_input_col(df, columns_cfg, "y", ["souradnice_y", "y", "lat", "latitude"])

    if place_name_col is None:
        raise RuntimeError(f"Grouped point dataset is missing place_name column. Columns: {list(df.columns)}")
    if x_col is None or y_col is None:
        raise RuntimeError(f"Grouped point dataset is missing x/y columns. Columns: {list(df.columns)}")

    tmp = pd.DataFrame({
        "place_code": df[place_code_col].astype(str).str.strip() if place_code_col else "",
        "place_name": df[place_name_col].astype(str).str.strip(),
        "district_name": df[district_name_col].fillna("").astype(str).str.strip() if district_name_col else "",
        "admin_level": df[admin_level_col].fillna("").astype(str).str.strip() if admin_level_col else "municipality",
        "_x": coerce_numeric_series(df[x_col]),
        "_y": coerce_numeric_series(df[y_col]),
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

    out = coerce_object_columns_for_parquet(pd.DataFrame(rows))
    gdf = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs=f"EPSG:{int(source_crs_epsg)}",
    )
    if int(output_crs_epsg) != int(source_crs_epsg):
        gdf = gdf.to_crs(epsg=int(output_crs_epsg))

    ensure_dir(out_parquet.parent)
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
    df = read_csv_flexible(src_csv, delimiter=delimiter, encoding=encoding)
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
                frames.append(read_csv_flexible(tmp_path, delimiter=delimiter, encoding=encoding))
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
