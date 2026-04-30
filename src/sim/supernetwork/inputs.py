"""Data loading: model area, gateways, commuting, place centroids, external unit resolution."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from sim._text import norm_name as _norm_name
from sim.datasets.utils import ensure_dir
from sim.supernetwork.config import SuperCfg

logger = logging.getLogger(__name__)


def _make_place_key(name_norm: str, district_norm: str = "") -> str:
    return f"{name_norm}|{district_norm}" if district_norm else name_norm


def _norm_obec_code(value: Any) -> str:
    """Stable municipality id from CSU (op_obec_kod / doj_obec_kod) as string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (np.floating, np.integer)):
        if isinstance(value, np.floating) and np.isnan(float(value)):
            return ""
        iv = int(value)
        return str(iv) if iv > 0 else ""
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def _candidate_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        found = mapping.get(str(name).strip().lower())
        if found is not None:
            return found
    return None


def load_model_area(path: Path, metric_epsg: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=metric_epsg, allow_override=True)
    elif gdf.crs.to_epsg() != metric_epsg:
        gdf = gdf.to_crs(epsg=metric_epsg)
    return gdf


def load_internal_zone_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    gdf = gpd.read_file(path)
    if "name" not in gdf.columns:
        return set()
    internal = gdf[gdf.get("is_external", 0).fillna(0).astype(int) == 0]
    return {_norm_name(v, keep_slash=True) for v in internal["name"].dropna().astype(str)}


def load_gateways(cfg: SuperCfg) -> gpd.GeoDataFrame:
    if cfg.gateway_seed_lookup_path.exists():
        gdf = gpd.read_parquet(cfg.gateway_seed_lookup_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=cfg.metric_epsg, allow_override=True)
        elif gdf.crs.to_epsg() != cfg.metric_epsg:
            gdf = gdf.to_crs(epsg=cfg.metric_epsg)
        return gdf.copy()

    if cfg.gateway_diagnostics_path.exists():
        df = pd.read_csv(cfg.gateway_diagnostics_path)
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["boundary_x"], df["boundary_y"]),
            crs=f"EPSG:{cfg.metric_epsg}",
        )

    raise FileNotFoundError("Gateway diagnostics not found")


def read_commuting(cfg: SuperCfg) -> pd.DataFrame:
    if cfg.full_cr_commuting_parquet.exists():
        return pd.read_parquet(cfg.full_cr_commuting_parquet)
    if cfg.full_cr_commuting_csv.exists():
        return pd.read_csv(cfg.full_cr_commuting_csv, sep=",", encoding="utf-8", low_memory=False)
    raise FileNotFoundError(
        "Full-CR commuting dataset not found. Expected parquet at "
        f"{cfg.full_cr_commuting_parquet} or CSV at {cfg.full_cr_commuting_csv}. "
        "Run fetch-data (dataset commuting_sldb2021) so the CSV exists and the full-CR parquet is built."
    )


def aggregate_commuting_pairs(df: pd.DataFrame) -> pd.DataFrame:
    origin_col = _candidate_col(df, ["op_obec"])
    dest_col = _candidate_col(df, ["doj_obec"])
    work_col = _candidate_col(df, ["dojizdka_prace"])
    school_col = _candidate_col(df, ["dojizdka_skola"])
    origin_dist_col = _candidate_col(df, ["op_okres", "origin_okres", "origin_district"])
    dest_dist_col = _candidate_col(df, ["doj_okres", "dest_okres", "dest_district"])
    origin_code_col = _candidate_col(df, ["op_obec_kod", "op_kod_obce"])
    dest_code_col = _candidate_col(df, ["doj_obec_kod", "doj_kod_obce"])

    if origin_col is None or dest_col is None:
        raise RuntimeError("Commuting dataset is missing op_obec / doj_obec")

    tmp = pd.DataFrame({
        "origin_place": df[origin_col].astype(str).str.strip(),
        "dest_place": df[dest_col].astype(str).str.strip(),
        "origin_district": df[origin_dist_col].fillna("").astype(str).str.strip() if origin_dist_col else "",
        "dest_district": df[dest_dist_col].fillna("").astype(str).str.strip() if dest_dist_col else "",
        "origin_place_code": (
            df[origin_code_col].map(_norm_obec_code) if origin_code_col else pd.Series("", index=df.index, dtype=object)
        ),
        "dest_place_code": (
            df[dest_code_col].map(_norm_obec_code) if dest_code_col else pd.Series("", index=df.index, dtype=object)
        ),
        "persons_work": pd.to_numeric(df[work_col], errors="coerce").fillna(0.0) if work_col else 0.0,
        "persons_school": pd.to_numeric(df[school_col], errors="coerce").fillna(0.0) if school_col else 0.0,
    })
    tmp = tmp[(tmp["origin_place"] != "") & (tmp["dest_place"] != "")].copy()

    group_cols = [
        "origin_place_code",
        "dest_place_code",
        "origin_place",
        "origin_district",
        "dest_place",
        "dest_district",
    ]
    agg = (
        tmp.groupby(group_cols, as_index=False)[["persons_work", "persons_school"]]
        .sum()
        .reset_index(drop=True)
    )
    for side in ["origin", "dest"]:
        agg[f"{side}_place_norm"] = agg[f"{side}_place"].map(_norm_name)
        agg[f"{side}_district_norm"] = agg[f"{side}_district"].map(_norm_name)
        agg[f"{side}_place_key"] = agg.apply(
            lambda r, s=side: _make_place_key(r[f"{s}_place_norm"], r[f"{s}_district_norm"]),
            axis=1,
        )
    agg["origin_unit_id"] = np.where(agg["origin_place_code"].astype(str).str.len() > 0, agg["origin_place_code"], agg["origin_place_key"])
    agg["dest_unit_id"] = np.where(agg["dest_place_code"].astype(str).str.len() > 0, agg["dest_place_code"], agg["dest_place_key"])
    return agg


def load_place_centroids(cfg: SuperCfg) -> gpd.GeoDataFrame:
    path = cfg.place_centroids_path
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=cfg.place_centroids_crs_epsg, allow_override=True)
    if gdf.crs.to_epsg() != cfg.metric_epsg:
        gdf = gdf.to_crs(epsg=cfg.metric_epsg)

    name_col = _candidate_col(gdf, ["place_name", "name", "obec", "nazev", "municipality"])
    code_col = _candidate_col(gdf, ["place_code", "kod_obce", "obec_kod"])
    district_col = _candidate_col(gdf, ["district_name", "district", "okres", "okres_name"])
    admin_col = _candidate_col(gdf, ["admin_level", "level", "kind", "type"])

    if name_col is None:
        raise RuntimeError("Place centroid dataset has no place name column")

    out = gdf.copy()
    out["place_name"] = out[name_col].astype(str).str.strip()
    out["place_name_norm"] = out["place_name"].map(_norm_name)
    out["place_code"] = out[code_col].fillna("").astype(str).str.strip() if code_col else ""
    out["district_name"] = out[district_col].fillna("").astype(str).str.strip() if district_col else ""
    out["district_norm"] = out["district_name"].map(_norm_name)
    out["admin_level"] = out[admin_col].fillna("").astype(str).str.strip() if admin_col else ""
    out["place_key"] = out.apply(lambda r: _make_place_key(r["place_name_norm"], r["district_norm"]), axis=1)
    out = out[out["place_name_norm"] != ""].copy()

    out["_score"] = out["place_code"].ne("").astype(int) + out["district_norm"].ne("").astype(int)
    out = out.sort_values(["place_key", "_score"], ascending=[True, False]).drop_duplicates("place_key", keep="first")
    return out[["place_code", "place_name", "place_name_norm", "district_name", "district_norm", "admin_level", "place_key", "geometry"]].copy()


def resolve_external_units(
    place_centroids: gpd.GeoDataFrame,
    commuting_pairs: pd.DataFrame,
    internal_zone_names: set[str],
    unresolved_path: Path,
) -> gpd.GeoDataFrame:
    """
    Map each external commuting unit to RUIAN centroid geometry.
    Prefer CSU obec code (op_obec_kod / doj_obec_kod) -> place_code, then place_key, then unique name.
    Output rows are keyed by unit_id (code or name|district key) for consistent gateway lookup.
    """
    used = pd.concat([
        commuting_pairs[
            [
                "origin_place",
                "origin_place_norm",
                "origin_district",
                "origin_district_norm",
                "origin_place_key",
                "origin_place_code",
                "origin_unit_id",
            ]
        ].rename(columns={
            "origin_place": "place_name",
            "origin_place_norm": "place_name_norm",
            "origin_district": "district_name",
            "origin_district_norm": "district_norm",
            "origin_place_key": "place_key",
            "origin_place_code": "place_code",
            "origin_unit_id": "unit_id",
        }),
        commuting_pairs[
            [
                "dest_place",
                "dest_place_norm",
                "dest_district",
                "dest_district_norm",
                "dest_place_key",
                "dest_place_code",
                "dest_unit_id",
            ]
        ].rename(columns={
            "dest_place": "place_name",
            "dest_place_norm": "place_name_norm",
            "dest_district": "district_name",
            "dest_district_norm": "district_norm",
            "dest_place_key": "place_key",
            "dest_place_code": "place_code",
            "dest_unit_id": "unit_id",
        }),
    ], ignore_index=True).drop_duplicates(subset=["unit_id"], keep="first")

    used = used[~used["place_name_norm"].isin(internal_zone_names)].copy()
    if used.empty:
        raise RuntimeError("No external places found after filtering internal names")

    used["place_code"] = used["place_code"].fillna("").astype(str).str.strip()
    used["unit_id"] = used["unit_id"].astype(str).str.strip()
    used["_row_id"] = np.arange(len(used), dtype=np.int64)

    centroids = place_centroids.copy()
    centroids["place_code"] = centroids["place_code"].fillna("").astype(str).str.strip()
    name_counts = centroids.groupby("place_name_norm").size().rename("name_candidate_count")
    centroids = centroids.merge(name_counts, on="place_name_norm", how="left")

    cen_keep = [
        "place_code",
        "place_key",
        "place_name",
        "place_name_norm",
        "district_name",
        "district_norm",
        "admin_level",
        "geometry",
        "name_candidate_count",
    ]
    cen_keep = [c for c in cen_keep if c in centroids.columns]
    cen_base = centroids[cen_keep].copy()

    cen_geom_cols = [
        "place_code",
        "place_key",
        "place_name",
        "district_name",
        "place_name_norm",
        "district_norm",
        "admin_level",
        "geometry",
    ]
    cen_geom_cols = [c for c in cen_geom_cols if c in cen_base.columns]

    matched_parts: List[pd.DataFrame] = []
    matched_ids: set[int] = set()

    by_code = (
        cen_base[cen_base["place_code"] != ""]
        .drop_duplicates(subset=["place_code"], keep="first")[cen_geom_cols]
        .copy()
    )
    code_rows = used[used["place_code"] != ""][["unit_id", "place_code", "_row_id"]].copy()
    if not code_rows.empty and not by_code.empty:
        m_code = code_rows.merge(by_code, on="place_code", how="inner")
        matched_parts.append(m_code)
        matched_ids.update(int(x) for x in m_code["_row_id"].tolist())

    rem = used[~used["_row_id"].isin(matched_ids)].copy()
    if not rem.empty:
        m_key = rem[["unit_id", "place_key", "_row_id"]].merge(
            cen_base[cen_geom_cols].drop_duplicates(subset=["place_key"], keep="first"),
            on="place_key",
            how="inner",
        )
        if not m_key.empty:
            matched_parts.append(m_key)
            matched_ids.update(int(x) for x in m_key["_row_id"].tolist())

    rem2 = used[~used["_row_id"].isin(matched_ids)].copy()
    if not rem2.empty:
        uniq_name = cen_base[cen_base["name_candidate_count"] == 1][cen_geom_cols].copy()
        m_name = rem2[["unit_id", "place_name_norm", "_row_id"]].merge(
            uniq_name,
            on="place_name_norm",
            how="inner",
        )
        if not m_name.empty:
            matched_parts.append(m_name)
            matched_ids.update(int(x) for x in m_name["_row_id"].tolist())

    still_unresolved = used[~used["_row_id"].isin(matched_ids)].copy()

    ensure_dir(unresolved_path.parent)
    if not still_unresolved.empty:
        still_unresolved.drop(columns=["_row_id"], errors="ignore").to_parquet(unresolved_path, index=False)

    if not matched_parts:
        raise RuntimeError("No centroid matches for external places")

    merged = pd.concat(matched_parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["unit_id"], keep="first")
    merged = merged.drop(columns=["_row_id"], errors="ignore")

    out = gpd.GeoDataFrame(merged, geometry=merged["geometry"], crs=place_centroids.crs)

    logger.info("Resolved external places: %d; unresolved: %d", len(out), len(still_unresolved))
    return out.reset_index(drop=True)
