"""Build OD matrices from commuting data and synthetic gateway segments and register them in an AequilibraE project."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import load_config


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_GEO_SUFFIXES = re.compile(r"\s+(u|nad|pod|na|ve|pri|při)\s+\S+$", re.IGNORECASE)


def _strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _norm_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = _strip_diacritics(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_geo_suffix(name_norm: str) -> str:
    return _GEO_SUFFIXES.sub("", name_norm).strip()


# ---------------------------------------------------------------------------
# Generic config helpers
# ---------------------------------------------------------------------------

def _get(cfg: Any, path: List[str], default: Any = None) -> Any:
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _zero_matrix(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


# ---------------------------------------------------------------------------
# Zone name lookup
# ---------------------------------------------------------------------------

def _build_zone_name_index(
    zones_gdf: gpd.GeoDataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    primary: Dict[str, int] = {}
    stripped: Dict[str, int] = {}

    for _, row in zones_gdf.iterrows():
        zone_id = int(row["zone_id"])
        name_norm = _norm_name(row.get("name", ""))
        if not name_norm:
            continue

        primary.setdefault(name_norm, zone_id)

        stripped_name = _strip_geo_suffix(name_norm)
        if stripped_name != name_norm:
            stripped.setdefault(stripped_name, zone_id)

    return primary, stripped


def _match_zone_id(
    name: str,
    primary: Dict[str, int],
    stripped: Dict[str, int],
) -> Optional[int]:
    import difflib

    name_norm = _norm_name(name)
    if not name_norm:
        return None

    if name_norm in primary:
        return primary[name_norm]

    stripped_name = _strip_geo_suffix(name_norm)
    if stripped_name in primary:
        return primary[stripped_name]

    if name_norm in stripped:
        return stripped[name_norm]

    if stripped_name in stripped:
        return stripped[stripped_name]

    for prefix in ("mesto ", "obec ", "mestys "):
        if name_norm.startswith(prefix):
            rest = name_norm[len(prefix):]
            if rest in primary:
                return primary[rest]

    all_names = list(primary.keys()) + list(stripped.keys())
    for candidate in (name_norm, stripped_name):
        close = difflib.get_close_matches(candidate, all_names, n=1, cutoff=0.85)
        if close:
            match_name = close[0]
            return primary.get(match_name) or stripped.get(match_name)

    return None


# ---------------------------------------------------------------------------
# Gateway zones
# ---------------------------------------------------------------------------

def _load_gateways(zones_gdf: gpd.GeoDataFrame) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load gateway corridors from synthetic external zones created by zoning.py.

    Expected columns:
      - is_external
      - gateway_name (preferred)
      - or EXT_* names as fallback
    """
    if "is_external" not in zones_gdf.columns:
        return {}

    external = zones_gdf[zones_gdf["is_external"].fillna(0).astype(int) == 1].copy()
    if external.empty:
        return {}

    grouped: Dict[str, List[int]] = {}
    for _, row in external.iterrows():
        gateway_name = str(row.get("gateway_name", "")).strip()
        if not gateway_name:
            name = str(row.get("name", "")).strip()
            gateway_name = name.replace("EXT_", "", 1) if name.startswith("EXT_") else name

        if not gateway_name:
            continue

        grouped.setdefault(gateway_name, []).append(int(row["zone_id"]))

    gateways: Dict[str, List[Tuple[int, float]]] = {}
    for gateway_name, zone_ids in grouped.items():
        if not zone_ids:
            continue
        weight = 1.0 / len(zone_ids)
        gateways[gateway_name] = [(zid, weight) for zid in zone_ids]

    if gateways:
        print(f"  Using synthetic external gateway zones: {len(gateways)} corridors")
        for name, members in gateways.items():
            print(f"    {name}: {[z for z, _ in members]}")

    return gateways


def _assign_external_to_gateway(
    place_name: str,
    gateways: Dict[str, List[Tuple[int, float]]],
) -> List[Tuple[int, float]]:
    """
    Deterministic fallback assignment of an unmatched external municipality
    to one configured gateway corridor.

    This is intentionally simple: without external municipality coordinates,
    any corridor assignment is heuristic.
    """
    if not gateways:
        return []

    gateway_names = sorted(gateways.keys())
    digest = hashlib.sha256(place_name.encode()).hexdigest()
    idx = int(digest[:15], 16) % len(gateway_names)
    return gateways[gateway_names[idx]]


# ---------------------------------------------------------------------------
# Hub group
# ---------------------------------------------------------------------------

def _load_zone_population(cache_dir: Path = Path("data/cache")) -> Dict[int, int]:
    pop_path = cache_dir / "zone_population.parquet"
    if not pop_path.exists():
        return {}
    pop_df = pd.read_parquet(pop_path)
    return dict(zip(pop_df["zone_id"].astype(int), pop_df["population"].astype(int)))


def _build_hub_group(
    zones_gdf: gpd.GeoDataFrame,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    csv_place_names: set[str],
    *,
    hub_name: str,
    hub_source_rank: Optional[int],
    zone_population: Optional[Dict[int, int]] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Build a synthetic 'hub' group for the main multi-zone core city.

    Generic strategy:
    - the hub name must be present in the commuting CSV
    - hub zones are candidate zones from one source_rank (configurable or inferred)
    - zones already claimed by other directly matched municipalities are excluded
    """
    hub_norm = _norm_name(hub_name)
    if not hub_norm:
        return {}

    csv_norm_names = {_norm_name(x) for x in csv_place_names}
    if hub_norm not in csv_norm_names:
        return {}

    claimed: set[int] = set()
    for place in csv_place_names:
        if _norm_name(place) == hub_norm:
            continue
        zid = _match_zone_id(place, primary, stripped)
        if zid is not None:
            claimed.add(zid)

    internal = zones_gdf[zones_gdf.get("is_external", 0).fillna(0).astype(int) == 0].copy()
    if internal.empty:
        return {}

    if hub_source_rank is None:
        hub_source_rank = int(internal["source_rank"].min()) if "source_rank" in internal.columns else 0

    use_population = bool(zone_population)
    members: list[Tuple[int, float]] = []

    for _, row in internal.iterrows():
        zone_id = int(row["zone_id"])
        if zone_id in claimed:
            continue
        if int(row.get("source_rank", -1)) != int(hub_source_rank):
            continue

        if use_population:
            weight = max(float(zone_population.get(zone_id, 1)), 1.0)
        else:
            weight = max(float(row.geometry.area), 1.0)

        members.append((zone_id, weight))

    if len(members) < 2:
        return {}

    ids, weights = zip(*members)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    method = "population" if use_population else "area"
    print(f"  Hub group '{hub_name}': {len(members)} zones, weighted by {method}")

    return {hub_norm: [(zid, float(wi)) for zid, wi in zip(ids, w)]}


# ---------------------------------------------------------------------------
# Demand configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PurposeConv:
    car_share: float
    occupancy: float
    trips_per_person: float


@dataclass(frozen=True)
class PeriodShares:
    outbound: Dict[str, float]
    return_: Dict[str, float]


@dataclass(frozen=True)
class DemandBuildCfg:
    commuting_parquet: Path
    commuting_csv: Path
    csv_delimiter: str
    csv_encoding: str
    include_lokalizace: List[str]
    only_internal_pairs: bool
    origin_filters: List[Dict[str, Any]]
    conv_work: PurposeConv
    conv_school: PurposeConv
    periods: List[str]
    shares_work: PeriodShares
    shares_school: PeriodShares
    output_dir: Path
    matrix_path: Path
    matrix_name: str


def _resolve_commuting_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    csv_path = _get(cfg, ["datasets", "sources", "commuting_sldb2021", "out_path"])
    if csv_path:
        csv_path = _as_path(csv_path)
    else:
        csv_path = _as_path("data/sources/csu/sldb2021/dojizdka_obce.csv")

    cache_dir = _as_path(_get(cfg, ["datasets", "cache_dir"], "data/cache"))
    parquet_path = cache_dir / f"{csv_path.stem}.parquet"
    return csv_path, parquet_path


def _build_cfg(cfg: Dict[str, Any]) -> DemandBuildCfg:
    demand = cfg.get("demand") or {}
    csv_path, parquet_path = _resolve_commuting_paths(cfg)
    fmt = _get(cfg, ["datasets", "sources", "commuting_sldb2021", "format"], {}) or {}

    include_lok = _get(demand, ["sldb", "include_lokalizace"], ["0_na_adrese_OP", "1_meziobecni"])
    if not isinstance(include_lok, list):
        include_lok = ["0_na_adrese_OP", "1_meziobecni"]

    origin_filters = _get(
        cfg,
        ["datasets", "sources", "commuting_sldb2021", "filter", "origin", "keep_if_any_matches"],
        [],
    )
    if not isinstance(origin_filters, list):
        origin_filters = []

    conv_cfg = _get(demand, ["conversion"], {}) or {}
    conv_work = PurposeConv(
        car_share=float(_get(conv_cfg, ["work", "car_share"], 0.60)),
        occupancy=float(_get(conv_cfg, ["work", "occupancy"], 1.25)),
        trips_per_person=float(_get(conv_cfg, ["work", "trips_per_person"], 2.0)),
    )
    conv_school = PurposeConv(
        car_share=float(_get(conv_cfg, ["school", "car_share"], 0.25)),
        occupancy=float(_get(conv_cfg, ["school", "occupancy"], 1.30)),
        trips_per_person=float(_get(conv_cfg, ["school", "trips_per_person"], 2.0)),
    )

    periods = _get(demand, ["time_slices", "periods"], ["am", "ip", "pm", "ev"])
    if not isinstance(periods, list) or not periods:
        periods = ["am", "ip", "pm", "ev"]
    periods = [str(p).lower().strip() for p in periods]

    weekday_cfg = _get(demand, ["time_slices", "weekday"], {}) or {}
    shares_work = PeriodShares(
        outbound=dict(_get(weekday_cfg, ["work", "outbound"], {"am": 0.80, "ip": 0.20})),
        return_=dict(_get(weekday_cfg, ["work", "return"], {"pm": 0.80, "ev": 0.20})),
    )
    shares_school = PeriodShares(
        outbound=dict(_get(weekday_cfg, ["school", "outbound"], {"am": 0.90, "ip": 0.10})),
        return_=dict(_get(weekday_cfg, ["school", "return"], {"pm": 0.70, "ev": 0.30})),
    )

    output_dir = _as_path(demand.get("output_dir", "outputs/baseline/demand"))
    matrix_path = _as_path(demand.get("matrix_path", "data/demand/od_matrix.aem"))
    matrix_name = str(demand.get("matrix_name", "demand")).strip()

    return DemandBuildCfg(
        commuting_parquet=parquet_path,
        commuting_csv=csv_path,
        csv_delimiter=str(fmt.get("delimiter", ",")),
        csv_encoding=str(fmt.get("encoding", "utf-8")),
        include_lokalizace=[str(x) for x in include_lok],
        only_internal_pairs=bool(_get(demand, ["sldb", "only_internal_pairs"], True)),
        origin_filters=origin_filters,
        conv_work=conv_work,
        conv_school=conv_school,
        periods=periods,
        shares_work=shares_work,
        shares_school=shares_school,
        output_dir=output_dir,
        matrix_path=matrix_path,
        matrix_name=matrix_name,
    )


def _validate_shares(periods: List[str], shares: PeriodShares, label: str) -> None:
    for side_name, d in [("outbound", shares.outbound), ("return", shares.return_)]:
        total = sum(float(v) for v in d.values())
        for key in d:
            if str(key).lower().strip() not in periods:
                raise ValueError(f"{label}.{side_name}: unknown period '{key}', allowed: {periods}")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{label}.{side_name}: shares must sum to 1.0 (got {total:.4f})")


def _normalize_period_shares(raw: Optional[Dict[str, Any]], periods: List[str]) -> Dict[str, float]:
    """
    Normalize arbitrary period shares for non-commuting segments.
    Falls back to even split if not configured or invalid.
    """
    if not raw:
        return {p: 1.0 / len(periods) for p in periods}

    shares = {p: float(raw.get(p, 0.0)) for p in periods}
    total = sum(shares.values())

    if total <= 0:
        return {p: 1.0 / len(periods) for p in periods}

    return {p: v / total for p, v in shares.items()}


def _normalize_named_weights(
    names: List[str],
    raw: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if not names:
        return {}

    raw = raw or {}
    vals = {name: max(float(raw.get(name, 0.0)), 0.0) for name in names}
    total = sum(vals.values())

    if total <= 0:
        return {name: 1.0 / len(names) for name in names}

    return {name: value / total for name, value in vals.items()}


def _parse_gateway_pair_weights(
    raw_pairs: Optional[List[Dict[str, Any]]],
    gateway_names: List[str],
    *,
    default_pair_weight: float = 0.0,
) -> Dict[Tuple[str, str], float]:
    gateway_names = sorted(gateway_names)
    gateway_set = set(gateway_names)
    weights: Dict[Tuple[str, str], float] = {}

    base = max(float(default_pair_weight), 0.0)

    if raw_pairs:
        if base > 0:
            for a in gateway_names:
                for b in gateway_names:
                    if a != b:
                        weights[(a, b)] = base
    else:
        base = base if base > 0 else 1.0
        for a in gateway_names:
            for b in gateway_names:
                if a != b:
                    weights[(a, b)] = base
        return weights

    for item in raw_pairs or []:
        a = str(item.get("from", "")).strip()
        b = str(item.get("to", "")).strip()

        if a not in gateway_set or b not in gateway_set or a == b:
            continue

        w = max(float(item.get("weight", 0.0)), 0.0)
        weights[(a, b)] = w

        if bool(item.get("bidirectional", False)):
            weights[(b, a)] = w

    return {k: v for k, v in weights.items() if v > 0}


# ---------------------------------------------------------------------------
# Read and filter commuting data
# ---------------------------------------------------------------------------

def _read_commuting(bcfg: DemandBuildCfg) -> pd.DataFrame:
    if bcfg.commuting_parquet.exists():
        print(f"  Reading preprocessed parquet: {bcfg.commuting_parquet}")
        return pd.read_parquet(bcfg.commuting_parquet)

    if bcfg.commuting_csv.exists():
        print(f"  Reading raw CSV: {bcfg.commuting_csv}")
        return pd.read_csv(
            bcfg.commuting_csv,
            sep=bcfg.csv_delimiter,
            encoding=bcfg.csv_encoding,
            low_memory=False,
        )

    raise FileNotFoundError(
        f"Commuting data not found.\n"
        f"  tried: {bcfg.commuting_parquet}\n"
        f"  tried: {bcfg.commuting_csv}"
    )


def _apply_origin_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df

    mask = pd.Series(False, index=df.index)
    any_applied = False

    for clause in filters:
        field = clause.get("field")
        vals = clause.get("values") or []
        if not field or not vals or field not in df.columns:
            continue

        any_applied = True
        vals_norm = [str(v).strip().lower() for v in vals]
        mask = mask | df[field].astype(str).str.strip().str.lower().isin(vals_norm)

    return df[mask].copy() if any_applied else df


def _filter_commuting(df: pd.DataFrame, bcfg: DemandBuildCfg) -> pd.DataFrame:
    if "lokalizace" in df.columns and bcfg.include_lokalizace:
        df = df[df["lokalizace"].astype(str).isin(bcfg.include_lokalizace)].copy()

    df = _apply_origin_filters(df, bcfg.origin_filters)

    for col in ("dojizdka_prace", "dojizdka_skola"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# OD seed builders
# ---------------------------------------------------------------------------

def _build_gravity_seed(
    zone_ids: np.ndarray,
    population: Dict[int, int],
    *,
    trip_rate: float,
    car_share: float,
    occupancy: float,
    beta: float,
    impedance: Optional[np.ndarray] = None,
    centroids_gdf: Optional[gpd.GeoDataFrame] = None,
    excluded_zone_ids: Optional[set[int]] = None,
    metric_epsg: int,
) -> np.ndarray:
    excluded_zone_ids = excluded_zone_ids or set()
    n = len(zone_ids)

    pop = np.array(
        [
            0.0 if int(z) in excluded_zone_ids else max(population.get(int(z), 0), 0)
            for z in zone_ids
        ],
        dtype=np.float64,
    )

    productions = pop * trip_rate * car_share / max(occupancy, 0.01)
    attractions = pop.copy()

    if impedance is not None and impedance.shape == (n, n):
        imp = impedance.copy()
    elif centroids_gdf is not None and len(centroids_gdf) >= n:
        cg = centroids_gdf
        if cg.crs is not None and cg.crs.to_epsg() != metric_epsg:
            cg = cg.to_crs(epsg=metric_epsg)

        points = cg.geometry.representative_point()
        z2i = {int(z): i for i, z in enumerate(zone_ids)}
        coords = np.zeros((n, 2), dtype=np.float64)

        for idx, (_, row) in enumerate(cg.iterrows()):
            zid = int(row.get("zone_id", 0))
            if zid in z2i:
                coords[z2i[zid]] = [points.iloc[idx].x, points.iloc[idx].y]

        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        imp = np.sqrt(dx ** 2 + dy ** 2)
    else:
        imp = np.ones((n, n), dtype=np.float64)

    imp = np.maximum(imp, 100.0)
    deterrence = np.exp(-float(beta) * imp)
    np.fill_diagonal(deterrence, 0.0)

    od = productions[:, None] * attractions[None, :] * deterrence
    total_prod = productions.sum()
    total_od = od.sum()

    if total_od > 0 and total_prod > 0:
        od *= total_prod / total_od

    return od


def _build_external_local_seed(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    population: Dict[int, int],
    *,
    total_daily_trips: float,
    corridor_weights: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = _zero_matrix(n)

    if not gateways or total_daily_trips <= 0:
        return od

    gateway_zone_ids = {
        int(zid)
        for gateway_zones in gateways.values()
        for zid, _ in gateway_zones
        if int(zid) in z2i
    }

    internal_zone_ids = [int(z) for z in zone_ids if int(z) not in gateway_zone_ids]
    if not internal_zone_ids:
        return od

    internal_pop = np.array(
        [max(population.get(z, 0), 0) for z in internal_zone_ids],
        dtype=np.float64,
    )
    if internal_pop.sum() <= 0:
        return od
    internal_pop = internal_pop / internal_pop.sum()

    corridor_names = sorted(gateways.keys())
    corridor_weights_norm = _normalize_named_weights(corridor_names, corridor_weights)

    for gateway_name in corridor_names:
        gateway_zones = [(zid, w) for zid, w in gateways[gateway_name] if int(zid) in z2i]
        if not gateway_zones:
            continue

        corridor_total = total_daily_trips * corridor_weights_norm.get(gateway_name, 0.0)
        if corridor_total <= 0:
            continue

        inbound_total = corridor_total * 0.5
        outbound_total = corridor_total * 0.5

        for gateway_zone_id, gateway_weight in gateway_zones:
            gi = z2i[int(gateway_zone_id)]

            for internal_zone_id, share in zip(internal_zone_ids, internal_pop):
                ii = z2i[int(internal_zone_id)]
                if gi == ii:
                    continue

                od[gi, ii] += outbound_total * float(gateway_weight) * float(share)
                od[ii, gi] += inbound_total * float(gateway_weight) * float(share)

    return od


def _build_external_through_seed(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    *,
    total_daily_trips: float,
    pair_weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> np.ndarray:
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = _zero_matrix(n)

    if not gateways or total_daily_trips <= 0:
        return od

    gateway_names = sorted(gateways.keys())
    if not gateway_names:
        return od

    if not pair_weights:
        pair_weights = {
            (a, b): 1.0
            for a in gateway_names
            for b in gateway_names
            if a != b
        }

    valid_pairs = {
        (a, b): max(float(w), 0.0)
        for (a, b), w in pair_weights.items()
        if a in gateways and b in gateways and a != b and float(w) > 0
    }

    total_weight = sum(valid_pairs.values())
    if total_weight <= 0:
        return od

    for (a_name, b_name), pair_weight in valid_pairs.items():
        pair_total = total_daily_trips * float(pair_weight) / total_weight

        a_zones = [(zid, w) for zid, w in gateways[a_name] if int(zid) in z2i]
        b_zones = [(zid, w) for zid, w in gateways[b_name] if int(zid) in z2i]
        if not a_zones or not b_zones:
            continue

        for a_zid, a_w in a_zones:
            ai = z2i[int(a_zid)]
            for b_zid, b_w in b_zones:
                bi = z2i[int(b_zid)]
                if ai == bi:
                    continue
                od[ai, bi] += pair_total * float(a_w) * float(b_w)

    return od


def _persons_to_vehicles(persons: float, conv: PurposeConv) -> float:
    if persons <= 0:
        return 0.0
    return persons * conv.trips_per_person * conv.car_share / max(conv.occupancy, 0.01)


def _zone_index(zone_ids: np.ndarray) -> Dict[int, int]:
    return {int(z): i for i, z in enumerate(zone_ids)}


def _build_od_cores(
    df: pd.DataFrame,
    zone_ids: np.ndarray,
    bcfg: DemandBuildCfg,
    *,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    groups: Dict[str, List[Tuple[int, float]]],
    gateways: Optional[Dict[str, List[Tuple[int, float]]]],
    use_gateway_fallback: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    _validate_shares(bcfg.periods, bcfg.shares_work, "weekday.work")
    _validate_shares(bcfg.periods, bcfg.shares_school, "weekday.school")

    z2i = _zone_index(zone_ids)
    zset = set(z2i.keys())

    period_cores = [f"wd_{p}" for p in bcfg.periods]
    core_names = period_cores + ["wd_daily"]
    mats: Dict[str, np.ndarray] = {
        name: _zero_matrix(len(zone_ids))
        for name in core_names
    }

    stats: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "mapped_direct": 0,
        "mapped_group": 0,
        "mapped_gateway": 0,
        "missing_origin": 0,
        "missing_destination": 0,
        "pairs_used": 0,
    }

    origin_col = "op_obec" if "op_obec" in df.columns else None
    dest_col = "doj_obec" if "doj_obec" in df.columns else None
    if origin_col is None or dest_col is None:
        raise RuntimeError(f"Expected columns op_obec/doj_obec, got: {list(df.columns)}")

    external_cache: Dict[str, List[Tuple[int, float]]] = {}
    gateway_mode = bool(gateways) and bool(use_gateway_fallback) and not bcfg.only_internal_pairs

    def resolve_place(name: str) -> Tuple[List[Tuple[int, float]], str]:
        zone_id = _match_zone_id(name, primary, stripped)
        if zone_id is not None:
            return [(zone_id, 1.0)], "direct"

        key = _norm_name(name)
        if key in groups:
            return groups[key], "group"

        if gateway_mode:
            if key not in external_cache:
                external_cache[key] = _assign_external_to_gateway(key, gateways or {})
            cands = external_cache[key]
            if cands:
                return cands, "gateway"

        return [], "missing"

    for _, row in df.iterrows():
        origin_name = str(row[origin_col]).strip() if pd.notna(row[origin_col]) else ""
        dest_name = str(row[dest_col]).strip() if pd.notna(row[dest_col]) else ""

        if not origin_name:
            stats["missing_origin"] += 1
            continue
        if not dest_name:
            stats["missing_destination"] += 1
            continue

        origin_candidates, origin_mode = resolve_place(origin_name)
        dest_candidates, dest_mode = resolve_place(dest_name)

        if not origin_candidates:
            stats["missing_origin"] += 1
            continue
        if not dest_candidates:
            stats["missing_destination"] += 1
            continue

        for mode in (origin_mode, dest_mode):
            if mode == "direct":
                stats["mapped_direct"] += 1
            elif mode == "group":
                stats["mapped_group"] += 1
            elif mode == "gateway":
                stats["mapped_gateway"] += 1

        work_v = _persons_to_vehicles(float(row.get("dojizdka_prace", 0)), bcfg.conv_work)
        school_v = _persons_to_vehicles(float(row.get("dojizdka_skola", 0)), bcfg.conv_school)

        if work_v <= 0 and school_v <= 0:
            continue

        for oz, ow in origin_candidates:
            if oz not in zset:
                continue
            oi = z2i[oz]

            for dz, dw in dest_candidates:
                if dz not in zset:
                    continue
                di = z2i[dz]
                factor = ow * dw

                for period, share in bcfg.shares_work.outbound.items():
                    mats[f"wd_{period}"][oi, di] += work_v * share * factor
                for period, share in bcfg.shares_school.outbound.items():
                    mats[f"wd_{period}"][oi, di] += school_v * share * factor
                for period, share in bcfg.shares_work.return_.items():
                    mats[f"wd_{period}"][di, oi] += work_v * share * factor
                for period, share in bcfg.shares_school.return_.items():
                    mats[f"wd_{period}"][di, oi] += school_v * share * factor

                stats["pairs_used"] += 1

    mats["wd_daily"] = sum(mats[name] for name in period_cores)

    summary = {
        **stats,
        "zones": int(len(zone_ids)),
        "cores_sum": {name: round(float(mats[name].sum()), 1) for name in core_names},
        "nonzero_cells": {name: int(np.count_nonzero(mats[name])) for name in core_names},
    }

    return mats, summary


# ---------------------------------------------------------------------------
# AequilibraE output
# ---------------------------------------------------------------------------

def _write_aem(
    matrix_path: Path,
    index_ids: np.ndarray,
    cores: Dict[str, np.ndarray],
    *,
    matrix_name: str,
) -> None:
    _ensure_dir(matrix_path.parent)
    core_names = list(cores.keys())

    mat = AequilibraeMatrix()
    mat.create_empty(
        file_name=str(matrix_path),
        zones=int(len(index_ids)),
        matrix_names=core_names,
        data_type=np.float64,
        memory_only=False,
    )

    try:
        mat.setName(matrix_name)
    except Exception:
        pass

    try:
        mat.setDescription("OD matrix built from commuting and synthetic segment seeds")
    except Exception:
        pass

    mat.index[:] = index_ids.astype(np.int32)
    for core_name in core_names:
        mat.matrix[core_name][:, :] = cores[core_name]

    mat.save()
    mat.close()


def _register_in_project(project_dir: Path, matrix_path: Path) -> None:
    from aequilibrae import Project

    project = Project()
    project.open(str(project_dir))
    try:
        matrices_dir = Path(project.project_base_path) / "matrices"
        _ensure_dir(matrices_dir)

        target = matrices_dir / matrix_path.name
        if matrix_path.resolve() != target.resolve():
            shutil.copy2(matrix_path, target)

        try:
            project.matrices.update_database()
            project.matrices.reload()
        except Exception:
            pass
    finally:
        project.close()


# ---------------------------------------------------------------------------
# Load zones
# ---------------------------------------------------------------------------

def _load_zones(cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    zoning_dir = _get(cfg, ["zoning", "output_dir"], "outputs/baseline/zones")
    zones_path = Path(zoning_dir) / "zones.geojson"

    if not zones_path.exists():
        raise FileNotFoundError(f"zones.geojson not found: {zones_path}. Run build-zones first.")

    gdf = gpd.read_file(zones_path)
    if "zone_id" not in gdf.columns:
        raise RuntimeError(f"zones.geojson missing 'zone_id': {zones_path}")

    if "name" not in gdf.columns:
        gdf["name"] = ""

    gdf["zone_id"] = pd.to_numeric(gdf["zone_id"], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=["zone_id"]).copy()
    gdf["zone_id"] = gdf["zone_id"].astype(int)
    gdf["name"] = gdf["name"].astype(str)

    if "is_external" not in gdf.columns:
        gdf["is_external"] = 0
    gdf["is_external"] = pd.to_numeric(gdf["is_external"], errors="coerce").fillna(0).astype(int)

    if "gateway_name" not in gdf.columns:
        gdf["gateway_name"] = ""
    gdf["gateway_name"] = gdf["gateway_name"].fillna("").astype(str)

    return gdf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_or_build_od_matrix(config_path: str | Path = "config/sim.yaml") -> None:
    cfg = load_config(config_path)
    bcfg = _build_cfg(cfg)

    print("=== BUILD OD MATRIX ===")

    print("Loading zones ...")
    zones_gdf = _load_zones(cfg)

    external_zone_ids: set[int] = set(
        zones_gdf.loc[zones_gdf["is_external"].fillna(0).astype(int) == 1, "zone_id"].astype(int)
    )
    print(f"  External zones: {len(external_zone_ids)}")

    zone_ids = np.array(sorted(zones_gdf["zone_id"].astype(int).unique()), dtype=np.int64)

    zoning_dir = _get(cfg, ["zoning", "output_dir"], "outputs/baseline/zones")
    mapping_path = Path(zoning_dir) / "zone_centroid_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"zone_centroid_mapping.json not found at {mapping_path}. Run build-zones first."
        )

    raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    zone_to_centroid: Dict[int, int] = {int(k): int(v) for k, v in raw_mapping.items()}
    print(f"  Loaded zone->centroid mapping ({len(zone_to_centroid)} entries)")

    centroid_ids = np.array([zone_to_centroid.get(int(z), int(z)) for z in zone_ids], dtype=np.int64)
    print(f"  {len(zone_ids)} zones loaded (centroid IDs: {centroid_ids.min()}-{centroid_ids.max()})")

    primary, stripped = _build_zone_name_index(zones_gdf)
    print(f"  Name index: {len(primary)} primary, {len(stripped)} stripped entries")

    print("Reading commuting data ...")
    df_raw = _read_commuting(bcfg)

    zone_population = _load_zone_population(
        Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))
    )

    origin_place_col = "op_obec" if "op_obec" in df_raw.columns else None
    csv_places: set[str] = set()
    if origin_place_col:
        csv_places = set(df_raw[origin_place_col].dropna().astype(str).unique())

    hub_cfg = _get(cfg, ["demand", "hub_group"], {}) or {}
    hub_name = str(hub_cfg.get("name", "")).strip()
    if not hub_name:
        place = str(_get(cfg, ["osm", "place_name"], "") or "")
        hub_name = place.split(",")[0].strip() if place else ""

    hub_source_rank = hub_cfg.get("source_rank")
    if hub_source_rank is not None:
        hub_source_rank = int(hub_source_rank)

    groups = _build_hub_group(
        zones_gdf,
        primary,
        stripped,
        csv_places,
        hub_name=hub_name,
        hub_source_rank=hub_source_rank,
        zone_population=zone_population,
    )
    if groups:
        for group_name, members in groups.items():
            print(f"  Group '{group_name}': {len(members)} zones")

    gateways = _load_gateways(zones_gdf)
    if gateways:
        print(f"  Gateways: {len(gateways)} corridors")
        for gateway_name, gateway_zones in gateways.items():
            print(f"    {gateway_name}: {len(gateway_zones)} zones")
    else:
        print("  No external gateway zones found")

    df = _filter_commuting(df_raw, bcfg)
    print(f"  {len(df)} rows after filtering")

    use_gateway_fallback_in_commuting = bool(
        _get(cfg, ["demand", "use_gateway_fallback_in_commuting"], False)
    )

    print("Building commuting OD cores ...")
    cores, summary = _build_od_cores(
        df,
        zone_ids,
        bcfg,
        primary=primary,
        stripped=stripped,
        groups=groups,
        gateways=gateways,
        use_gateway_fallback=use_gateway_fallback_in_commuting,
    )

    segments_cfg = _get(cfg, ["demand", "segments"], {}) or {}
    other_cfg = segments_cfg.get("other", {}) or {}
    external_local_cfg = segments_cfg.get("external_local", {}) or {}
    external_through_cfg = segments_cfg.get("external_through", {}) or {}

    n_zones = len(zone_ids)

    if other_cfg and other_cfg.get("source", "gravity") == "gravity":
        print("Building 'other' trips (gravity seed) ...")
        other_daily = _build_gravity_seed(
            zone_ids,
            zone_population,
            trip_rate=float(other_cfg.get("trip_rate", 1.5)),
            car_share=float(other_cfg.get("car_share", 0.45)),
            occupancy=float(other_cfg.get("occupancy", 1.40)),
            beta=float(other_cfg.get("beta", 0.0001)),
            centroids_gdf=zones_gdf,
            excluded_zone_ids=external_zone_ids,
            metric_epsg=int(cfg.get("crs_epsg", 5514)),
        )
        print(f"  'other' daily total: {float(other_daily.sum()):,.0f}")
    else:
        other_daily = _zero_matrix(n_zones)

    if external_local_cfg and external_local_cfg.get("source") == "gateway_local" and gateways:
        print("Building external-local trips (gateway ↔ internal) ...")
        external_local_daily = _build_external_local_seed(
            zone_ids,
            gateways,
            zone_population,
            total_daily_trips=float(external_local_cfg.get("total_daily_trips", 50000)),
            corridor_weights=external_local_cfg.get("corridor_weights", {}) or {},
        )
        print(f"  'external_local' daily total: {float(external_local_daily.sum()):,.0f}")
    else:
        external_local_daily = _zero_matrix(n_zones)

    if external_through_cfg and external_through_cfg.get("source") == "gateway_pairs" and gateways:
        print("Building external-through trips (gateway ↔ gateway) ...")
        pair_weights = _parse_gateway_pair_weights(
            external_through_cfg.get("pairs", []),
            sorted(gateways.keys()),
            default_pair_weight=float(external_through_cfg.get("default_pair_weight", 0.0)),
        )
        external_through_daily = _build_external_through_seed(
            zone_ids,
            gateways,
            total_daily_trips=float(external_through_cfg.get("total_daily_trips", 100000)),
            pair_weights=pair_weights,
        )
        print(f"  'external_through' daily total: {float(external_through_daily.sum()):,.0f}")
    else:
        external_through_daily = _zero_matrix(n_zones)

    other_period_shares = _normalize_period_shares(
        _get(cfg, ["demand", "time_slices", "segments", "other"], None),
        bcfg.periods,
    )
    external_local_period_shares = _normalize_period_shares(
        _get(cfg, ["demand", "time_slices", "segments", "external_local"], None),
        bcfg.periods,
    )
    external_through_period_shares = _normalize_period_shares(
        _get(cfg, ["demand", "time_slices", "segments", "external_through"], None),
        bcfg.periods,
    )

    all_cores: Dict[str, np.ndarray] = {}

    # Preserve commuting-only cores explicitly
    for period in bcfg.periods:
        all_cores[f"wd_{period}_commuting"] = cores.get(f"wd_{period}", _zero_matrix(n_zones)).copy()
    all_cores["wd_daily_commuting"] = cores.get("wd_daily", _zero_matrix(n_zones)).copy()

    # Non-commuting and external segment cores
    for period in bcfg.periods:
        all_cores[f"wd_{period}_other"] = other_daily * other_period_shares[period]
        all_cores[f"wd_{period}_external_local"] = external_local_daily * external_local_period_shares[period]
        all_cores[f"wd_{period}_external_through"] = external_through_daily * external_through_period_shares[period]
        all_cores[f"wd_{period}_external"] = (
            all_cores[f"wd_{period}_external_local"] +
            all_cores[f"wd_{period}_external_through"]
        )

    all_cores["wd_daily_other"] = other_daily.copy()
    all_cores["wd_daily_external_local"] = external_local_daily.copy()
    all_cores["wd_daily_external_through"] = external_through_daily.copy()
    all_cores["wd_daily_external"] = external_local_daily + external_through_daily

    # Final combined cores used by the rest of the pipeline
    for period in bcfg.periods:
        all_cores[f"wd_{period}"] = (
            all_cores[f"wd_{period}_commuting"]
            + all_cores[f"wd_{period}_other"]
            + all_cores[f"wd_{period}_external_local"]
            + all_cores[f"wd_{period}_external_through"]
        )

    all_cores["wd_daily"] = sum(all_cores[f"wd_{period}"] for period in bcfg.periods)

    segment_totals = {
        "commuting": round(float(all_cores["wd_daily_commuting"].sum()), 0),
        "other": round(float(other_daily.sum()), 0),
        "external_local": round(float(external_local_daily.sum()), 0),
        "external_through": round(float(external_through_daily.sum()), 0),
        "external_total": round(float(all_cores["wd_daily_external"].sum()), 0),
        "combined_daily": round(float(all_cores["wd_daily"].sum()), 0),
    }
    print(f"  Segment totals: {segment_totals}")

    print(f"Writing AEM matrix: {bcfg.matrix_path}")
    _write_aem(bcfg.matrix_path, centroid_ids, all_cores, matrix_name=bcfg.matrix_name)

    _ensure_dir(bcfg.output_dir)

    summary_data = {
        "commuting_source": str(
            bcfg.commuting_parquet if bcfg.commuting_parquet.exists() else bcfg.commuting_csv
        ),
        "matrix_path": str(bcfg.matrix_path),
        "conversion": {
            "work": {
                "car_share": bcfg.conv_work.car_share,
                "occupancy": bcfg.conv_work.occupancy,
                "trips_per_person": bcfg.conv_work.trips_per_person,
            },
            "school": {
                "car_share": bcfg.conv_school.car_share,
                "occupancy": bcfg.conv_school.occupancy,
                "trips_per_person": bcfg.conv_school.trips_per_person,
            },
        },
        "periods": bcfg.periods,
        "segments": segment_totals,
        "groups": {k: {"zones": len(v)} for k, v in groups.items()},
        "gateways": {k: len(v) for k, v in gateways.items()},
        **summary,
        "final_cores_sum": {name: round(float(all_cores[name].sum()), 1) for name in sorted(all_cores.keys())},
        "final_nonzero_cells": {name: int(np.count_nonzero(all_cores[name])) for name in sorted(all_cores.keys())},
    }

    summary_path = bcfg.output_dir / "od_summary.json"
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    project_dir = cfg.get("project_path")
    if project_dir and Path(project_dir).exists():
        print(f"Registering matrix in AequilibraE project: {project_dir}")
        _register_in_project(Path(project_dir), bcfg.matrix_path)

    print("\n--- OD build summary ---")
    print(f"  Zones:           {summary['zones']}")
    print(f"  Rows in:         {summary['rows_in']}")
    print(f"  Pairs used:      {summary['pairs_used']}")
    print(f"  Direct matches:  {summary['mapped_direct']}")
    print(f"  Group matches:   {summary['mapped_group']}")
    print(f"  Gateway matches: {summary['mapped_gateway']}")
    print(f"  Missing origin:  {summary['missing_origin']}")
    print(f"  Missing dest:    {summary['missing_destination']}")

    for core_name in sorted(all_cores.keys()):
        total = round(float(all_cores[core_name].sum()), 1)
        nonzero = int(np.count_nonzero(all_cores[core_name]))
        print(f"  {core_name:24s}  total={total:>12.1f}  nonzero={nonzero}")

    print(f"\n  Matrix:  {bcfg.matrix_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    load_or_build_od_matrix()
