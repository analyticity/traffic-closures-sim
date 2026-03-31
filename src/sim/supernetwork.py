from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from sim.io_project import load_config

try:
    import pyogrio
except Exception:  # pragma: no cover
    pyogrio = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _strip_diacritics(text: Any) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def _norm_name(value: Any) -> str:
    text = _strip_diacritics(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w\s\-/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _make_place_key(name_norm: str, district_norm: str = "") -> str:
    return f"{name_norm}|{district_norm}" if district_norm else name_norm


def _norm_obec_code(value: Any) -> str:
    """Stable municipality id from ČSÚ (op_obec_kod / doj_obec_kod) as string."""
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


def _parse_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not m:
        return None
    out = float(m.group(0).replace(",", "."))
    if "mph" in text:
        out *= 1.60934
    return out


def _highway_default_speed_kmh(highway: Any) -> float:
    highway = str(highway or "")
    defaults = {
        "motorway": 130.0,
        "motorway_link": 80.0,
        "trunk": 90.0,
        "trunk_link": 70.0,
        "primary": 70.0,
        "primary_link": 50.0,
        "secondary": 55.0,
        "secondary_link": 45.0,
        "tertiary": 50.0,
        "tertiary_link": 40.0,
    }
    return defaults.get(highway, 60.0)


def _edge_speed_kmh(maxspeed: Any, highway: Any) -> float:
    parsed = _parse_numeric(maxspeed)
    return parsed if parsed and parsed > 0 else _highway_default_speed_kmh(highway)


def _meters_to_seconds(distance_m: float, speed_kmh: float) -> float:
    return float(distance_m) / max(float(speed_kmh) / 3.6, 0.1)


def _persons_to_vehicles(work: float, school: float, cfg_root: Dict[str, Any]) -> float:
    def conv(persons: float, branch: str) -> float:
        conv_cfg = _get(cfg_root, "demand", "conversion", branch, default={}) or {}
        car_share = float(conv_cfg.get("car_share", 0.0))
        occupancy = max(float(conv_cfg.get("occupancy", 1.0)), 0.01)
        trips_per_person = float(conv_cfg.get("trips_per_person", 2.0))
        return max(float(persons), 0.0) * trips_per_person * car_share / occupancy

    return conv(work, "work") + conv(school, "school")


def _iter_lines(geom: Any) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperCfg:
    metric_epsg: int
    output_dir: Path
    cache_dir: Path
    pbf_path: Path
    place_centroids_path: Path
    place_centroids_crs_epsg: int
    highway_types: List[str]
    contract_graph: bool
    contract_degree: int
    max_candidate_gateways: int
    detour_ratio_max: float
    max_extra_minutes: float
    allow_same_gateway_pair: bool
    model_area_path: Path
    zones_path: Path
    gateway_seed_lookup_path: Path
    gateway_diagnostics_path: Path
    full_cr_commuting_parquet: Path
    full_cr_commuting_csv: Path
    national_nodes_path: Path
    national_edges_path: Path
    external_unit_lookup_path: Path
    external_gateway_lookup_path: Path
    through_gateway_pairs_path: Path
    unresolved_places_path: Path
    classified_relations_path: Path
    raw_major_roads_cache: Path


def build_cfg(cfg_root: Dict[str, Any]) -> SuperCfg:
    sn = cfg_root.get("supernetwork") or {}
    outputs = sn.get("outputs") or {}
    cache_dir = _as_path(sn.get("cache_dir", "data/cache/supernetwork"))
    output_dir = _as_path(sn.get("output_dir", "outputs/baseline/supernetwork"))
    zoning_output_dir = _as_path(_get(cfg_root, "zoning", "output_dir", default="outputs/baseline/zones"))
    commute_src = _get(cfg_root, "datasets", "sources", "commuting_sldb2021", default={}) or {}
    highway_types = list(
        _get(sn, "national_network", "highway_types", default=["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"])
    )
    hw_sig = hashlib.sha1(
        "|".join(sorted({str(x).strip() for x in highway_types if str(x).strip()})).encode("utf-8")
    ).hexdigest()[:12]

    return SuperCfg(
        metric_epsg=int(cfg_root.get("crs_epsg", 5514)),
        output_dir=output_dir,
        cache_dir=cache_dir,
        pbf_path=_as_path(sn.get("pbf_path", "data/sources/osm/czech-republic-latest.osm.pbf")),
        place_centroids_path=_as_path(sn.get("place_centroids_path", "data/cache/cz_place_centroids.parquet")),
        place_centroids_crs_epsg=int(sn.get("place_centroids_crs_epsg", 4326)),
        highway_types=highway_types,
        contract_graph=bool(sn.get("contract_graph", True)),
        contract_degree=int(sn.get("contract_exclude_degree_leq", 2)),
        max_candidate_gateways=int(_get(sn, "gateway_mapping", "max_candidate_gateways", default=3)),
        detour_ratio_max=float(_get(sn, "relation_filter", "detour_ratio_max", default=1.25)),
        max_extra_minutes=float(_get(sn, "relation_filter", "max_extra_minutes", default=20.0)),
        allow_same_gateway_pair=bool(_get(sn, "relation_filter", "allow_same_gateway_pair", default=False)),
        model_area_path=zoning_output_dir / "model_area.geojson",
        zones_path=zoning_output_dir / "zones.geojson",
        gateway_seed_lookup_path=_as_path(_get(cfg_root, "zoning", "external_gateways", "export_lookup_path", default="data/cache/gateway_lookup_seed.parquet")),
        gateway_diagnostics_path=zoning_output_dir / "gateway_diagnostics.csv",
        full_cr_commuting_parquet=_as_path(commute_src.get("full_cr_out_parquet", "data/cache/dojizdka_obce_full_cr.parquet")),
        full_cr_commuting_csv=_as_path(commute_src.get("out_path", "data/sources/csu/sldb2021/dojizdka_obce.csv")),
        national_nodes_path=_as_path(
            outputs.get("national_nodes", cache_dir / f"national_nodes_{hw_sig}.parquet")
        ),
        national_edges_path=_as_path(
            outputs.get("national_edges", cache_dir / f"national_edges_{hw_sig}.parquet")
        ),
        external_unit_lookup_path=_as_path(outputs.get("external_unit_lookup", "data/cache/external_unit_lookup.parquet")),
        external_gateway_lookup_path=_as_path(outputs.get("external_gateway_lookup", "data/cache/external_gateway_lookup.parquet")),
        through_gateway_pairs_path=_as_path(outputs.get("through_gateway_pairs", "data/cache/through_gateway_pairs.parquet")),
        unresolved_places_path=cache_dir / "unresolved_external_places.parquet",
        classified_relations_path=cache_dir / "classified_external_relations.parquet",
        raw_major_roads_cache=cache_dir / f"major_roads_raw_{hw_sig}.parquet",
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

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
    return {_norm_name(v) for v in internal["name"].dropna().astype(str)}


def load_gateways(cfg: SuperCfg) -> gpd.GeoDataFrame:
    if cfg.gateway_seed_lookup_path.exists():
        gdf = gpd.read_parquet(cfg.gateway_seed_lookup_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=cfg.metric_epsg, allow_override=True)
        elif gdf.crs.to_epsg() != cfg.metric_epsg:
            gdf = gdf.to_crs(epsg=cfg.metric_epsg)
        return gdf[[c for c in gdf.columns if c in {"gateway_name", "geometry"} or c not in set()]].copy()

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
    raise FileNotFoundError("Full-CR commuting dataset not found")


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
            lambda r: _make_place_key(r[f"{side}_place_norm"], r[f"{side}_district_norm"]),
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

    # keep one row per exact place key; if duplicates remain, prefer one with code and district
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
    Map each external commuting unit to RÚIAN centroid geometry.
    Prefer ČSÚ obec code (op_obec_kod / doj_obec_kod) -> place_code, then place_key, then unique name.
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

    _ensure_dir(unresolved_path.parent)
    if not still_unresolved.empty:
        still_unresolved.drop(columns=["_row_id"], errors="ignore").to_parquet(unresolved_path, index=False)

    if not matched_parts:
        raise RuntimeError("No centroid matches for external places")

    merged = pd.concat(matched_parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["unit_id"], keep="first")
    merged = merged.drop(columns=["_row_id"], errors="ignore")

    out = gpd.GeoDataFrame(merged, geometry=merged["geometry"], crs=place_centroids.crs)

    print(f"Resolved external places: {len(out)}; unresolved: {len(still_unresolved)}")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def read_osm_lines_from_pbf(pbf_path: Path, highway_types: List[str]) -> gpd.GeoDataFrame:
    wanted = sorted(set(str(x).strip() for x in highway_types if str(x).strip()))
    where = " OR ".join([f"highway = '{h}'" for h in wanted])
    cols = ["osm_id", "name", "ref", "highway", "oneway", "maxspeed", "geometry"]

    if pyogrio is not None:
        try:
            gdf = pyogrio.read_dataframe(pbf_path, layer="lines", columns=cols, where=where)
            if gdf is not None and not gdf.empty:
                return gdf
        except Exception:
            pass

    gdf = gpd.read_file(pbf_path, layer="lines")
    return gdf[gdf["highway"].astype(str).isin(wanted)][[c for c in cols if c in gdf.columns]].copy()


def load_or_extract_major_roads(cfg: SuperCfg) -> gpd.GeoDataFrame:
    _ensure_dir(cfg.raw_major_roads_cache.parent)
    if cfg.raw_major_roads_cache.exists():
        roads = gpd.read_parquet(cfg.raw_major_roads_cache)
        if roads.crs is None:
            roads = roads.set_crs(epsg=cfg.metric_epsg, allow_override=True)
        return roads

    roads = read_osm_lines_from_pbf(cfg.pbf_path, cfg.highway_types)
    if roads.crs is None:
        roads = roads.set_crs(epsg=4326, allow_override=True)
    roads = roads.to_crs(epsg=cfg.metric_epsg)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads.to_parquet(cfg.raw_major_roads_cache, index=False)
    return roads


def _coord_key(x: float, y: float, ndigits: int = 3) -> Tuple[float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits))


def add_or_relax_edge(G: nx.DiGraph, u: int, v: int, length_m: float, travel_time_s: float, highway: str, name: str, ref: str) -> None:
    if u == v:
        return
    if G.has_edge(u, v):
        if travel_time_s < float(G[u][v].get("travel_time_s", math.inf)):
            G[u][v].update(length_m=float(length_m), travel_time_s=float(travel_time_s), highway=highway, name=name, ref=ref)
    else:
        G.add_edge(u, v, length_m=float(length_m), travel_time_s=float(travel_time_s), highway=highway, name=name, ref=ref)


def build_graph_from_roads(roads: gpd.GeoDataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    coord_to_node: Dict[Tuple[float, float], int] = {}
    next_id = 1

    def node_id(x: float, y: float) -> int:
        nonlocal next_id
        key = _coord_key(x, y)
        if key not in coord_to_node:
            coord_to_node[key] = next_id
            G.add_node(next_id, x=float(key[0]), y=float(key[1]))
            next_id += 1
        return coord_to_node[key]

    for _, row in roads.iterrows():
        speed_kmh = _edge_speed_kmh(row.get("maxspeed"), row.get("highway"))
        oneway = str(row.get("oneway", "") or "").strip().lower()
        reverse_only = oneway == "-1"
        forward_only = oneway in {"yes", "1", "true", "t"} or reverse_only

        for line in _iter_lines(row.geometry):
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                u = node_id(a[0], a[1])
                v = node_id(b[0], b[1])
                seg_len = float(LineString([a, b]).length)
                tt = _meters_to_seconds(seg_len, speed_kmh)
                highway = str(row.get("highway", "") or "")
                name = str(row.get("name", "") or "")
                ref = str(row.get("ref", "") or "")
                if reverse_only:
                    add_or_relax_edge(G, v, u, seg_len, tt, highway, name, ref)
                elif forward_only:
                    add_or_relax_edge(G, u, v, seg_len, tt, highway, name, ref)
                else:
                    add_or_relax_edge(G, u, v, seg_len, tt, highway, name, ref)
                    add_or_relax_edge(G, v, u, seg_len, tt, highway, name, ref)

    return G


def snap_points_to_graph(G: nx.DiGraph, points: gpd.GeoDataFrame, label_col: str) -> pd.DataFrame:
    node_ids = np.array(list(G.nodes()), dtype=np.int64)
    xs = np.array([G.nodes[n]["x"] for n in node_ids], dtype=float)
    ys = np.array([G.nodes[n]["y"] for n in node_ids], dtype=float)
    pts = np.column_stack([points.geometry.x.to_numpy(dtype=float), points.geometry.y.to_numpy(dtype=float)])

    if cKDTree is not None:
        tree = cKDTree(np.column_stack([xs, ys]))
        dists, idxs = tree.query(pts, k=1)
    else:
        idxs, dists = [], []
        for px, py in pts:
            dist2 = (xs - px) ** 2 + (ys - py) ** 2
            idx = int(np.argmin(dist2))
            idxs.append(idx)
            dists.append(float(np.sqrt(dist2[idx])))
        idxs = np.array(idxs, dtype=int)
        dists = np.array(dists, dtype=float)

    rows = []
    for i, (_, row) in enumerate(points.iterrows()):
        nid = int(node_ids[int(idxs[i])])
        rows.append({
            label_col: row[label_col],
            "graph_node": nid,
            "graph_x": float(G.nodes[nid]["x"]),
            "graph_y": float(G.nodes[nid]["y"]),
            "snap_distance_m": float(dists[i]),
        })
    return pd.DataFrame(rows)


def is_contractible(G: nx.DiGraph, node: int, protected: set[int], degree_threshold: int) -> bool:
    if node in protected or not G.has_node(node):
        return False
    # Avoid repeated expensive to_undirected() graph-view construction.
    # For a DiGraph without parallel edges, undirected degree equals unique neighbor count.
    neighbors = (set(G.predecessors(node)) | set(G.successors(node))) - {node}
    undirected_degree = len(neighbors)
    if undirected_degree != 2:
        return False
    return undirected_degree <= degree_threshold


def contract_graph(G_in: nx.DiGraph, protected: set[int], degree_threshold: int) -> nx.DiGraph:
    G = G_in.copy()
    queue = [n for n in list(G.nodes()) if is_contractible(G, int(n), protected, degree_threshold)]

    while queue:
        n = int(queue.pop())
        if not is_contractible(G, n, protected, degree_threshold):
            continue
        nbrs = list((set(G.predecessors(n)) | set(G.successors(n))) - {n})
        if len(nbrs) != 2:
            continue
        a, b = int(nbrs[0]), int(nbrs[1])

        if G.has_edge(a, n) and G.has_edge(n, b):
            e1, e2 = G[a][n], G[n][b]
            add_or_relax_edge(G, a, b, float(e1["length_m"]) + float(e2["length_m"]), float(e1["travel_time_s"]) + float(e2["travel_time_s"]), str(e1.get("highway", e2.get("highway", ""))), str(e1.get("name", e2.get("name", ""))), str(e1.get("ref", e2.get("ref", ""))))
        if G.has_edge(b, n) and G.has_edge(n, a):
            e1, e2 = G[b][n], G[n][a]
            add_or_relax_edge(G, b, a, float(e1["length_m"]) + float(e2["length_m"]), float(e1["travel_time_s"]) + float(e2["travel_time_s"]), str(e1.get("highway", e2.get("highway", ""))), str(e1.get("name", e2.get("name", ""))), str(e1.get("ref", e2.get("ref", ""))))
        G.remove_node(n)
        for m in (a, b):
            if is_contractible(G, m, protected, degree_threshold):
                queue.append(m)

    return G


def graph_to_gdfs(G: nx.DiGraph, metric_epsg: int) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = gpd.GeoDataFrame(
        [{"node_id": int(n), "x": float(d["x"]), "y": float(d["y"]), "geometry": Point(float(d["x"]), float(d["y"]))} for n, d in G.nodes(data=True)],
        geometry="geometry",
        crs=f"EPSG:{metric_epsg}",
    )
    edges = gpd.GeoDataFrame(
        [{
            "source": int(u),
            "target": int(v),
            "length_m": float(d["length_m"]),
            "travel_time_s": float(d["travel_time_s"]),
            "highway": str(d.get("highway", "")),
            "name": str(d.get("name", "")),
            "ref": str(d.get("ref", "")),
            "geometry": LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]),
        } for u, v, d in G.edges(data=True)],
        geometry="geometry",
        crs=f"EPSG:{metric_epsg}",
    )
    return nodes, edges


def graph_from_parquets(nodes_path: Path, edges_path: Path) -> nx.DiGraph:
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(int(r["node_id"]), x=float(r["x"]), y=float(r["y"]))
    for _, r in edges.iterrows():
        add_or_relax_edge(G, int(r["source"]), int(r["target"]), float(r["length_m"]), float(r["travel_time_s"]), str(r.get("highway", "")), str(r.get("name", "")), str(r.get("ref", "")))
    return G


# ---------------------------------------------------------------------------
# Lookup + classification
# ---------------------------------------------------------------------------

def single_source_costs(G: nx.DiGraph, source_node: int) -> Dict[int, float]:
    return nx.single_source_dijkstra_path_length(G, source=source_node, weight="travel_time_s")


def build_gateway_costs(
    G: nx.DiGraph,
    gateways: gpd.GeoDataFrame,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], pd.DataFrame]:
    gateway_nodes = {str(r["gateway_name"]): int(r["graph_node"]) for _, r in gateways.iterrows()}
    costs_from_gateway = {name: single_source_costs(G, node) for name, node in gateway_nodes.items()}
    reverse = G.reverse(copy=False)
    costs_to_gateway = {name: single_source_costs(reverse, node) for name, node in gateway_nodes.items()}
    rows = []
    for a, a_node in gateway_nodes.items():
        for b, b_node in gateway_nodes.items():
            if a == b:
                continue
            cost = costs_from_gateway[a].get(b_node)
            rows.append({"gateway_in": a, "gateway_out": b, "internal_cost_s": float(cost) if cost is not None else np.nan})
    return costs_from_gateway, costs_to_gateway, pd.DataFrame(rows)


def build_gateway_lookup(
    units: gpd.GeoDataFrame,
    costs_from_gateway: Dict[str, Dict[int, float]],
    costs_to_gateway: Dict[str, Dict[int, float]],
    cfg: SuperCfg,
) -> pd.DataFrame:
    columns = [
        "unit_id",
        "place_key",
        "place_name",
        "district_name",
        "admin_level",
        "unit_graph_node",
        "gateway_name",
        "rank_in",
        "rank_out",
        "route_cost_to_gateway_s",
        "route_cost_from_gateway_s",
        "route_cost_s",
    ]
    rows = []
    for _, unit in units.iterrows():
        scored_in = []
        scored_out = []
        unit_node = int(unit["graph_node"])
        gateway_names = sorted(set(costs_from_gateway.keys()) | set(costs_to_gateway.keys()))
        for gw_name in gateway_names:
            to_cost = costs_to_gateway.get(gw_name, {}).get(unit_node)
            from_cost = costs_from_gateway.get(gw_name, {}).get(unit_node)
            if to_cost is not None and np.isfinite(to_cost):
                scored_in.append((gw_name, float(to_cost)))
            if from_cost is not None and np.isfinite(from_cost):
                scored_out.append((gw_name, float(from_cost)))

        if not scored_in and not scored_out:
            continue

        in_rank = {name: rank for rank, (name, _) in enumerate(sorted(scored_in, key=lambda x: x[1])[:cfg.max_candidate_gateways], start=1)}
        out_rank = {name: rank for rank, (name, _) in enumerate(sorted(scored_out, key=lambda x: x[1])[:cfg.max_candidate_gateways], start=1)}

        selected = sorted(set(in_rank.keys()) | set(out_rank.keys()))
        for gw_name in selected:
            to_cost = costs_to_gateway.get(gw_name, {}).get(unit_node)
            from_cost = costs_from_gateway.get(gw_name, {}).get(unit_node)
            preferred = from_cost if from_cost is not None and np.isfinite(from_cost) else to_cost
            rows.append({
                "unit_id": str(unit["unit_id"]).strip(),
                "place_key": unit["place_key"],
                "place_name": unit["place_name"],
                "district_name": unit.get("district_name", ""),
                "admin_level": unit.get("admin_level", ""),
                "unit_graph_node": unit_node,
                "gateway_name": gw_name,
                "rank_in": in_rank.get(gw_name),
                "rank_out": out_rank.get(gw_name),
                "route_cost_to_gateway_s": float(to_cost) if to_cost is not None and np.isfinite(to_cost) else np.nan,
                "route_cost_from_gateway_s": float(from_cost) if from_cost is not None and np.isfinite(from_cost) else np.nan,
                "route_cost_s": float(preferred) if preferred is not None and np.isfinite(preferred) else np.nan,
            })
    return pd.DataFrame(rows, columns=columns)


def shortest_path_cost(G: nx.DiGraph, source: int, target: int, cache: Dict[Tuple[int, int], float]) -> float:
    key = (int(source), int(target))
    if key in cache:
        return cache[key]
    try:
        value = float(nx.shortest_path_length(G, source=source, target=target, weight="travel_time_s"))
    except Exception:
        value = float("nan")
    cache[key] = value
    return value


def classify_relations(
    G: nx.DiGraph,
    commuting_pairs: pd.DataFrame,
    gateway_lookup: pd.DataFrame,
    gateway_pair_costs: pd.DataFrame,
    internal_zone_names: set[str],
    cfg_root: Dict[str, Any],
    cfg: SuperCfg,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Backward/empty-schema guard: keep compatibility with older lookup schema.
    if "rank_in" not in gateway_lookup.columns:
        if "rank" in gateway_lookup.columns:
            gateway_lookup = gateway_lookup.copy()
            gateway_lookup["rank_in"] = gateway_lookup["rank"]
        else:
            gateway_lookup["rank_in"] = np.nan
    if "rank_out" not in gateway_lookup.columns:
        if "rank" in gateway_lookup.columns:
            gateway_lookup["rank_out"] = gateway_lookup["rank"]
        else:
            gateway_lookup["rank_out"] = np.nan
    if "route_cost_to_gateway_s" not in gateway_lookup.columns:
        gateway_lookup["route_cost_to_gateway_s"] = gateway_lookup.get("route_cost_s", np.nan)
    if "route_cost_from_gateway_s" not in gateway_lookup.columns:
        gateway_lookup["route_cost_from_gateway_s"] = gateway_lookup.get("route_cost_s", np.nan)

    if (
        "unit_id" in gateway_lookup.columns
        and not gateway_lookup.empty
        and gateway_lookup["unit_id"].astype(str).str.strip().ne("").any()
    ):
        id_col = "unit_id"
    else:
        id_col = "place_key"
    best_in = (
        gateway_lookup.dropna(subset=["rank_in"])
        .sort_values([id_col, "rank_in"])
        .drop_duplicates(id_col, keep="first")
    )
    best_out = (
        gateway_lookup.dropna(subset=["rank_out"])
        .sort_values([id_col, "rank_out"])
        .drop_duplicates(id_col, keep="first")
    )
    unit_map_in = {str(r[id_col]).strip(): r for _, r in best_in.iterrows()}
    unit_map_out = {str(r[id_col]).strip(): r for _, r in best_out.iterrows()}
    pair_map = {(str(r["gateway_in"]), str(r["gateway_out"])): float(r["internal_cost_s"]) for _, r in gateway_pair_costs.iterrows()}
    direct_cost_cache: Dict[Tuple[int, int], float] = {}

    rows = []
    through_rows = []
    for _, r in commuting_pairs.iterrows():
        o_norm = str(r["origin_place_norm"])
        d_norm = str(r["dest_place_norm"])
        o_key = str(r["origin_place_key"])
        d_key = str(r["dest_place_key"])
        o_uid = str(r.get("origin_unit_id", o_key)).strip()
        d_uid = str(r.get("dest_unit_id", d_key)).strip()
        o_internal = o_norm in internal_zone_names
        d_internal = d_norm in internal_zone_names
        vehicles_daily = _persons_to_vehicles(float(r["persons_work"]), float(r["persons_school"]), cfg_root)

        rec = {
            "origin_place": r["origin_place"],
            "dest_place": r["dest_place"],
            "origin_place_key": o_key,
            "dest_place_key": d_key,
            "origin_unit_id": o_uid,
            "dest_unit_id": d_uid,
            "vehicles_daily": vehicles_daily,
            "classification": None,
            "gateway_in": None,
            "gateway_out": None,
            "direct_cost_s": None,
            "via_brno_cost_s": None,
            "detour_ratio": None,
            "extra_minutes": None,
            "rejection_reason": None,
            "accepted": False,
        }

        if o_internal and d_internal:
            rec["classification"] = "internal_internal"
            rows.append(rec)
            continue
        if (not o_internal) and d_internal:
            rec["classification"] = "external_internal"
            if o_uid in unit_map_in:
                rec["gateway_in"] = str(unit_map_in[o_uid]["gateway_name"])
                rec["accepted"] = True
            else:
                rec["rejection_reason"] = "missing_origin_gateway_in"
            rows.append(rec)
            continue
        if o_internal and (not d_internal):
            rec["classification"] = "internal_external"
            if d_uid in unit_map_out:
                rec["gateway_out"] = str(unit_map_out[d_uid]["gateway_name"])
                rec["accepted"] = True
            else:
                rec["rejection_reason"] = "missing_dest_gateway_out"
            rows.append(rec)
            continue

        rec["classification"] = "external_external"
        o_info = unit_map_in.get(o_uid)
        d_info = unit_map_out.get(d_uid)
        if o_info is None or d_info is None:
            rec["rejection_reason"] = "missing_external_gateway"
            rows.append(rec)
            continue

        gw_in = str(o_info["gateway_name"])
        gw_out = str(d_info["gateway_name"])
        rec["gateway_in"] = gw_in
        rec["gateway_out"] = gw_out
        if gw_in == gw_out and not cfg.allow_same_gateway_pair:
            rec["rejection_reason"] = "same_gateway_not_allowed"
            rows.append(rec)
            continue

        pair_cost = pair_map.get((gw_in, gw_out), float("nan"))
        direct_cost = shortest_path_cost(G, int(o_info["unit_graph_node"]), int(d_info["unit_graph_node"]), direct_cost_cache)
        o_to_gateway = float(o_info["route_cost_to_gateway_s"]) if np.isfinite(o_info.get("route_cost_to_gateway_s", np.nan)) else float("nan")
        gateway_to_d = float(d_info["route_cost_from_gateway_s"]) if np.isfinite(d_info.get("route_cost_from_gateway_s", np.nan)) else float("nan")
        via_cost = (
            o_to_gateway + float(pair_cost) + gateway_to_d
            if np.isfinite(pair_cost) and np.isfinite(o_to_gateway) and np.isfinite(gateway_to_d)
            else float("nan")
        )

        rec["direct_cost_s"] = direct_cost if np.isfinite(direct_cost) else None
        rec["via_brno_cost_s"] = via_cost if np.isfinite(via_cost) else None
        if np.isfinite(direct_cost) and np.isfinite(via_cost) and direct_cost > 0:
            rec["detour_ratio"] = via_cost / direct_cost
            extra_minutes = (via_cost - direct_cost) / 60.0
            rec["extra_minutes"] = extra_minutes
            rec["accepted"] = (
                (via_cost / direct_cost) <= cfg.detour_ratio_max
                and extra_minutes <= cfg.max_extra_minutes
            )
            if not rec["accepted"]:
                if (via_cost / direct_cost) > cfg.detour_ratio_max:
                    rec["rejection_reason"] = "detour_ratio_exceeded"
                else:
                    rec["rejection_reason"] = "extra_minutes_exceeded"
        else:
            rec["rejection_reason"] = "missing_costs"

        if rec["accepted"]:
            through_rows.append({
                "gateway_in": gw_in,
                "gateway_out": gw_out,
                "vehicles_daily": vehicles_daily,
                "relations": 1,
            })
        rows.append(rec)

    classified = pd.DataFrame(rows)
    if through_rows:
        through_pairs = pd.DataFrame(through_rows)
        through_pairs = through_pairs.groupby(["gateway_in", "gateway_out"], as_index=False)[
            ["vehicles_daily", "relations"]
        ].sum()
    else:
        # Empty list -> DataFrame had no columns; demand preflight needs stable schema.
        through_pairs = pd.DataFrame(columns=["gateway_in", "gateway_out", "vehicles_daily"])
    return classified, through_pairs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_overview(edges_metric: gpd.GeoDataFrame, model_area: gpd.GeoDataFrame, gateways: gpd.GeoDataFrame, units: gpd.GeoDataFrame, out_png: Path) -> None:
    _ensure_dir(out_png.parent)
    fig, ax = plt.subplots(figsize=(10, 10))
    model_area.boundary.plot(ax=ax, linewidth=1)
    edges_metric.plot(ax=ax, linewidth=0.3)
    gateways.plot(ax=ax, markersize=30)
    units.plot(ax=ax, markersize=5)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str = "config/sim.yaml") -> Dict[str, Any]:
    cfg_root = load_config(config_path)
    cfg = build_cfg(cfg_root)
    _ensure_dir(cfg.output_dir)
    _ensure_dir(cfg.cache_dir)

    model_area = load_model_area(cfg.model_area_path, cfg.metric_epsg)
    gateways = load_gateways(cfg)
    internal_zone_names = load_internal_zone_names(cfg.zones_path)

    commuting_raw = read_commuting(cfg)
    commuting_pairs = aggregate_commuting_pairs(commuting_raw)
    place_centroids = load_place_centroids(cfg)
    external_units = resolve_external_units(place_centroids, commuting_pairs, internal_zone_names, cfg.unresolved_places_path)

    if cfg.national_nodes_path.exists() and cfg.national_edges_path.exists():
        G = graph_from_parquets(cfg.national_nodes_path, cfg.national_edges_path)
        nodes_metric = gpd.read_parquet(cfg.national_nodes_path)
        edges_metric = gpd.read_parquet(cfg.national_edges_path)
    else:
        roads = load_or_extract_major_roads(cfg)
        G_raw = build_graph_from_roads(roads)

        gw_metric = gateways.to_crs(epsg=cfg.metric_epsg) if gateways.crs and gateways.crs.to_epsg() != cfg.metric_epsg else gateways.copy()
        units_metric = external_units.to_crs(epsg=cfg.metric_epsg) if external_units.crs.to_epsg() != cfg.metric_epsg else external_units.copy()

        gw_raw = gw_metric.merge(snap_points_to_graph(G_raw, gw_metric[["gateway_name", "geometry"]], "gateway_name"), on="gateway_name", how="left")
        unit_raw = units_metric.merge(
            snap_points_to_graph(G_raw, units_metric[["unit_id", "geometry"]], "unit_id"),
            on="unit_id",
            how="left",
        )
        protected = set(gw_raw["graph_node"].dropna().astype(int)) | set(unit_raw["graph_node"].dropna().astype(int))
        G = contract_graph(G_raw, protected, cfg.contract_degree) if cfg.contract_graph else G_raw
        nodes_metric, edges_metric = graph_to_gdfs(G, cfg.metric_epsg)
        nodes_metric.to_parquet(cfg.national_nodes_path, index=False)
        edges_metric.to_parquet(cfg.national_edges_path, index=False)

    gateways_join = gateways.merge(snap_points_to_graph(G, gateways[["gateway_name", "geometry"]].to_crs(epsg=cfg.metric_epsg), "gateway_name"), on="gateway_name", how="left")
    units_join = external_units.merge(
        snap_points_to_graph(
            G,
            external_units[["unit_id", "geometry"]].to_crs(epsg=cfg.metric_epsg),
            "unit_id",
        ),
        on="unit_id",
        how="left",
    )
    units_join = units_join.dropna(subset=["graph_node"]).copy()

    # Guard against stale/misaligned cached supernetwork graph.
    # If nearly all external units snap to one node, the graph/cache is invalid for CZ-wide lookup.
    unique_unit_nodes = int(units_join["graph_node"].nunique()) if not units_join.empty else 0
    if len(units_join) > 100 and unique_unit_nodes <= 1:
        raise RuntimeError(
            "Invalid supernetwork cache detected: all external units snapped to a single graph node. "
            "Remove stale cache files in data/cache/supernetwork "
            "(national_nodes.parquet, national_edges.parquet, major_roads_raw.parquet) "
            "and rerun build-supernetwork."
        )

    unit_lookup = units_join[
        ["unit_id", "place_code", "place_name", "district_name", "admin_level", "place_key", "graph_node", "graph_x", "graph_y", "geometry"]
    ].rename(columns={"graph_node": "unit_graph_node"})
    unit_lookup.to_parquet(cfg.external_unit_lookup_path, index=False)

    costs_from_gateway, costs_to_gateway, gateway_pair_costs = build_gateway_costs(G, gateways_join)
    gateway_lookup = build_gateway_lookup(units_join, costs_from_gateway, costs_to_gateway, cfg)
    gateway_lookup.to_parquet(cfg.external_gateway_lookup_path, index=False)

    classified, through_pairs = classify_relations(G, commuting_pairs, gateway_lookup, gateway_pair_costs, internal_zone_names, cfg_root, cfg)
    classified.to_parquet(cfg.classified_relations_path, index=False)
    through_pairs.to_parquet(cfg.through_gateway_pairs_path, index=False)

    rejected_ext_ext = classified[(classified["classification"] == "external_external") & (classified["accepted"] == False)]
    reason_counts = (
        rejected_ext_ext["rejection_reason"].fillna("unspecified").value_counts().to_dict()
        if not rejected_ext_ext.empty else {}
    )
    ext_ext_total = int((classified["classification"] == "external_external").sum())
    ext_ext_accepted = int(((classified["classification"] == "external_external") & (classified["accepted"] == True)).sum())
    resolved_coverage = (len(units_join) / max(len(external_units), 1)) * 100.0
    print(f"External unit coverage: {len(units_join)}/{len(external_units)} ({resolved_coverage:.1f}%)")
    print(f"External-external accepted: {ext_ext_accepted}/{ext_ext_total}")
    if reason_counts:
        print(f"External-external rejection reasons: {reason_counts}")

    ei_ok = classified[(classified["classification"] == "external_internal") & (classified["accepted"] == True)]
    inbound_by_gw: Dict[str, float] = {}
    if not ei_ok.empty and "gateway_in" in ei_ok.columns:
        inbound_by_gw = ei_ok.groupby("gateway_in", dropna=False)["vehicles_daily"].sum().sort_values(ascending=False).to_dict()
        inbound_by_gw = {str(k): float(v) for k, v in inbound_by_gw.items() if str(k).strip()}

    ie_ok = classified[(classified["classification"] == "internal_external") & (classified["accepted"] == True)]
    outbound_by_gw: Dict[str, float] = {}
    if not ie_ok.empty and "gateway_out" in ie_ok.columns:
        outbound_by_gw = ie_ok.groupby("gateway_out", dropna=False)["vehicles_daily"].sum().sort_values(ascending=False).to_dict()
        outbound_by_gw = {str(k): float(v) for k, v in outbound_by_gw.items() if str(k).strip()}

    through_by_pair: List[Dict[str, Any]] = []
    if not through_pairs.empty:
        for _, tr in through_pairs.iterrows():
            through_by_pair.append({
                "gateway_in": str(tr["gateway_in"]),
                "gateway_out": str(tr["gateway_out"]),
                "vehicles_daily": float(tr["vehicles_daily"]),
            })

    if inbound_by_gw:
        print("Gateway inbound (external -> model), vehicles/day:")
        for k, v in sorted(inbound_by_gw.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:,.0f}")
    if outbound_by_gw:
        print("Gateway outbound (model -> external), vehicles/day:")
        for k, v in sorted(outbound_by_gw.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:,.0f}")
    if through_by_pair:
        print("Through traffic (gateway -> gateway), vehicles/day:")
        for item in sorted(through_by_pair, key=lambda x: -x["vehicles_daily"])[:20]:
            print(f"  {item['gateway_in']} -> {item['gateway_out']}: {item['vehicles_daily']:,.0f}")
        if len(through_by_pair) > 20:
            print(f"  ... +{len(through_by_pair) - 20} more pairs")

    gateways_geojson = cfg.output_dir / "gateway_points.geojson"
    units_geojson = cfg.output_dir / "used_external_units.geojson"
    plot_png = cfg.output_dir / "supernetwork_overview.png"

    gateways_join.to_crs(epsg=4326).to_file(gateways_geojson, driver="GeoJSON")
    units_join.to_crs(epsg=4326).to_file(units_geojson, driver="GeoJSON")
    plot_overview(edges_metric, model_area, gateways_join.to_crs(epsg=cfg.metric_epsg), units_join.to_crs(epsg=cfg.metric_epsg), plot_png)

    summary = {
        "gateway_count": int(len(gateways_join)),
        "resolved_external_units": int(len(units_join)),
        "unresolved_external_places_path": str(cfg.unresolved_places_path),
        "classified_relations": int(len(classified)),
        "accepted_external_external": ext_ext_accepted,
        "external_external_total": ext_ext_total,
        "external_external_rejections": reason_counts,
        "external_unit_coverage_pct": round(resolved_coverage, 2),
        "through_pairs_count": int(len(through_pairs)),
        "through_pairs_total_vehicles_daily": float(through_pairs["vehicles_daily"].sum()) if not through_pairs.empty else 0.0,
        "gateway_inbound_vehicles_daily": inbound_by_gw,
        "gateway_outbound_vehicles_daily": outbound_by_gw,
        "through_gateway_pairs_detail": through_by_pair,
        "graph_nodes": int(len(nodes_metric)),
        "graph_edges": int(len(edges_metric)),
        "outputs": {
            "national_nodes": str(cfg.national_nodes_path),
            "national_edges": str(cfg.national_edges_path),
            "external_unit_lookup": str(cfg.external_unit_lookup_path),
            "external_gateway_lookup": str(cfg.external_gateway_lookup_path),
            "through_gateway_pairs": str(cfg.through_gateway_pairs_path),
            "classified_relations": str(cfg.classified_relations_path),
            "gateways_geojson": str(gateways_geojson),
            "used_units_geojson": str(units_geojson),
            "plot": str(plot_png),
        },
    }
    summary_path = cfg.output_dir / "supernetwork_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

def run_build_supernetwork(config_path: str | Path = "config/sim.yaml") -> dict[str, Any]:
    return run(config_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified supernetwork builder")
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args()

    summary = run(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
