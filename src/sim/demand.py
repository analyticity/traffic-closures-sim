"""Build OD matrices from SLDB 2021 commuting data and register in AequilibraE project."""
from __future__ import annotations

import hashlib
import json
import math
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

_GEO_SUFFIXES = re.compile(
    r"\s+(u|nad|pod|na|ve|pri|při)\s+\S+$", re.IGNORECASE
)


def _strip_diacritics(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def _norm_name(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = _strip_diacritics(s).lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_geo_suffix(name_norm: str) -> str:
    return _GEO_SUFFIXES.sub("", name_norm).strip()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get(cfg: Any, path: List[str], default: Any = None) -> Any:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_path(p: Any) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Zone name index
# ---------------------------------------------------------------------------

def _build_zone_name_index(
    zones_gdf: gpd.GeoDataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (primary_index, stripped_index) mapping norm name -> zone_id."""
    primary: Dict[str, int] = {}
    stripped: Dict[str, int] = {}

    for _, r in zones_gdf.iterrows():
        zid = int(r["zone_id"])
        nm = _norm_name(r.get("name", ""))
        if not nm:
            continue
        primary.setdefault(nm, zid)
        sn = _strip_geo_suffix(nm)
        if sn != nm:
            stripped.setdefault(sn, zid)
    return primary, stripped


def _match_zone_id(
    name: str,
    primary: Dict[str, int],
    stripped: Dict[str, int],
) -> Optional[int]:
    nm = _norm_name(name)
    if not nm:
        return None
    if nm in primary:
        return primary[nm]
    sn = _strip_geo_suffix(nm)
    if sn in primary:
        return primary[sn]
    if nm in stripped:
        return stripped[nm]
    if sn in stripped:
        return stripped[sn]
    for prefix in ("mesto ", "obec ", "mestys "):
        if nm.startswith(prefix):
            rest = nm[len(prefix):]
            if rest in primary:
                return primary[rest]
    import difflib
    all_names = list(primary.keys()) + list(stripped.keys())
    for candidate in (nm, sn):
        close = difflib.get_close_matches(candidate, all_names, n=1, cutoff=0.85)
        if close:
            match_name = close[0]
            return primary.get(match_name) or stripped.get(match_name)
    return None


# ---------------------------------------------------------------------------
# Gateway zones for external trips
# ---------------------------------------------------------------------------

def _angle_in_range(angle: float, lo: float, hi: float) -> bool:
    """Check if *angle* (0-360) is within [lo, hi], handling wrap-around."""
    angle = angle % 360
    lo = lo % 360
    hi = hi % 360
    if lo <= hi:
        return lo <= angle <= hi
    return angle >= lo or angle <= hi


def _angular_distance_deg(a: float, b: float) -> float:
    """Smallest absolute angular distance in degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _load_gateways(
    cfg: Dict[str, Any],
    zones_gdf: gpd.GeoDataFrame,
    zone_population: Optional[Dict[int, int]] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Prefer synthetic external gateway zones from zoning output.
    Fallback to old boundary-zone matching only if EXT_* zones are absent.
    """
    if "is_external" in zones_gdf.columns:
        ext = zones_gdf[zones_gdf["is_external"].fillna(0).astype(int) == 1].copy()
    else:
        ext = zones_gdf.iloc[0:0].copy()

    if not ext.empty:
        out: Dict[str, List[Tuple[int, float]]] = {}
        for _, r in ext.iterrows():
            gw_name = str(r.get("gateway_name", "")).strip()
            if not gw_name:
                nm = str(r.get("name", "")).strip()
                gw_name = nm.replace("EXT_", "", 1) if nm.startswith("EXT_") else nm
            zid = int(r["zone_id"])
            out.setdefault(gw_name, []).append((zid, 1.0))

        out = {
            k: [(zid, 1.0 / len(v)) for zid, _ in v]
            for k, v in out.items()
        }

        print(f"  Using synthetic external gateway zones: {len(out)} corridors")
        for gn, lst in out.items():
            print(f"    {gn}: {[z for z, _ in lst]}")
        return out

    # -----------------------------
    # fallback: old boundary-zone mode
    # -----------------------------
    gw_list = _get(cfg, ["demand", "gateways"], [])
    if not gw_list or zones_gdf.empty:
        return {}

    max_zones_per_gateway = int(_get(cfg, ["demand", "max_zones_per_gateway"], 2))

    if zones_gdf.crs is not None and zones_gdf.crs.to_epsg() != 5514:
        zones_m = zones_gdf.to_crs(epsg=5514)
    else:
        zones_m = zones_gdf

    pts = zones_m.geometry.representative_point()
    cx = float(pts.x.mean())
    cy = float(pts.y.mean())

    dists = np.sqrt((pts.x - cx) ** 2 + (pts.y - cy) ** 2)
    boundary_threshold = float(np.percentile(dists, 75))

    raw_candidates: Dict[str, List[Tuple[int, float, float]]] = {}
    zone_best_gateway: Dict[int, Tuple[str, float, float]] = {}

    for gw in gw_list:
        name = gw.get("name", "unknown")
        ar = gw.get("angle_range", [0, 360])
        lo, hi = float(ar[0]), float(ar[1])
        center_angle = float(gw.get("direction_deg", 0.0)) % 360.0

        cand = []
        for i, (_, row) in enumerate(zones_m.iterrows()):
            if dists.iloc[i] < boundary_threshold:
                continue

            zid = int(row["zone_id"])
            pt = pts.iloc[i]
            angle = math.degrees(math.atan2(pt.x - cx, pt.y - cy)) % 360.0

            if not _angle_in_range(angle, lo, hi):
                continue

            ang_mismatch = _angular_distance_deg(angle, center_angle)
            dist = float(dists.iloc[i])
            cand.append((zid, ang_mismatch, dist))

        cand.sort(key=lambda x: (x[1], -x[2]))
        raw_candidates[name] = cand

    for gw_name, cand in raw_candidates.items():
        for zid, ang_mismatch, dist in cand:
            prev = zone_best_gateway.get(zid)
            new_key = (ang_mismatch, -dist)
            if prev is None:
                zone_best_gateway[zid] = (gw_name, ang_mismatch, -dist)
            else:
                prev_key = (prev[1], prev[2])
                if new_key < prev_key:
                    zone_best_gateway[zid] = (gw_name, ang_mismatch, -dist)

    by_gateway: Dict[str, List[Tuple[int, float]]] = {
        gw.get("name", "unknown"): [] for gw in gw_list
    }

    assigned_rows = []
    for zid, (gw_name, ang_mismatch, neg_dist) in zone_best_gateway.items():
        assigned_rows.append((gw_name, zid, ang_mismatch, -neg_dist))

    for gw_name in by_gateway.keys():
        rows = [(zid, ang_mismatch, dist) for g, zid, ang_mismatch, dist in assigned_rows if g == gw_name]
        rows.sort(key=lambda x: (x[1], -x[2]))
        rows = rows[:max_zones_per_gateway]

        if not rows:
            continue

        dist_arr = np.array([max(r[2], 1.0) for r in rows], dtype=np.float64)
        w = dist_arr / dist_arr.sum()
        by_gateway[gw_name] = [(int(zid), float(wi)) for (zid, _, _), wi in zip(rows, w)]

    return {k: v for k, v in by_gateway.items() if v}


def _assign_external_to_gateway(
    obec_name: str,
    gateways: Dict[str, List[Tuple[int, float]]],
    model_center: Tuple[float, float],
) -> List[Tuple[int, float]]:
    """Assign an external municipality to a gateway based on name hashing."""
    if not gateways:
        return []

    gw_names = sorted(gateways.keys())
    h = int(hashlib.sha256(obec_name.encode()).hexdigest()[:15], 16)
    gw = gw_names[h % len(gw_names)]
    return gateways[gw]


# ---------------------------------------------------------------------------
# Brno group (dynamic detection from zones + CSV municipalities)
# ---------------------------------------------------------------------------

def _load_zone_population(cache_dir: Path = Path("data/cache")) -> Dict[int, int]:
    """Load zone population from zone_population.parquet if available."""
    pop_path = cache_dir / "zone_population.parquet"
    if not pop_path.exists():
        return {}
    pop_df = pd.read_parquet(pop_path)
    return dict(zip(pop_df["zone_id"].astype(int), pop_df["population"].astype(int)))


def _build_brno_group(
    zones_gdf: gpd.GeoDataFrame,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    csv_obec_names: set[str],
    zone_population: Optional[Dict[int, int]] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Identify zones that belong to Brno municipality (not standalone towns).

    Weight by population if available, otherwise fall back to polygon area.
    """
    brno_norm = _norm_name("BRNO")
    claimed: set[int] = set()
    for obec in csv_obec_names:
        if _norm_name(obec) == brno_norm:
            continue
        zid = _match_zone_id(obec, primary, stripped)
        if zid is not None:
            claimed.add(zid)

    use_pop = bool(zone_population)
    brno_zones: list[Tuple[int, float]] = []
    for _, r in zones_gdf.iterrows():
        v = r.get("is_external", 0)
        is_external = 0 if pd.isna(v) else int(v)
        if is_external == 1:
            continue
        zid = int(r["zone_id"])
        if zid in claimed:
            continue
        if int(r.get("source_rank", -1)) == 0:
            if use_pop:
                weight = max(float(zone_population.get(zid, 1)), 1.0)
            else:
                weight = max(float(r.geometry.area), 1.0)
            brno_zones.append((zid, weight))

    if len(brno_zones) < 2:
        return {}

    ids, weights = zip(*brno_zones)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    method = "population" if use_pop else "area"
    print(f"  Brno group: {len(brno_zones)} zones, weighted by {method}")
    return {brno_norm: [(zid, float(wi)) for zid, wi in zip(ids, w)]}


# ---------------------------------------------------------------------------
# Demand build configuration
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

    include_lok = _get(demand, ["sldb", "include_lokalizace"],
                       ["0_na_adrese_OP", "1_meziobecni"])
    if not isinstance(include_lok, list):
        include_lok = ["0_na_adrese_OP", "1_meziobecni"]

    origin_filters = _get(cfg, [
        "datasets", "sources", "commuting_sldb2021",
        "filter", "origin", "keep_if_any_matches",
    ], [])
    if not isinstance(origin_filters, list):
        origin_filters = []

    conv = _get(demand, ["conversion"], {}) or {}
    conv_work = PurposeConv(
        car_share=float(_get(conv, ["work", "car_share"], 0.60)),
        occupancy=float(_get(conv, ["work", "occupancy"], 1.25)),
        trips_per_person=float(_get(conv, ["work", "trips_per_person"], 2.0)),
    )
    conv_school = PurposeConv(
        car_share=float(_get(conv, ["school", "car_share"], 0.25)),
        occupancy=float(_get(conv, ["school", "occupancy"], 1.30)),
        trips_per_person=float(_get(conv, ["school", "trips_per_person"], 2.0)),
    )

    periods = _get(demand, ["time_slices", "periods"], ["am", "ip", "pm", "ev"])
    if not isinstance(periods, list) or not periods:
        periods = ["am", "ip", "pm", "ev"]
    periods = [str(p).lower().strip() for p in periods]

    ts = _get(demand, ["time_slices", "weekday"], {}) or {}
    shares_work = PeriodShares(
        outbound=dict(_get(ts, ["work", "outbound"], {"am": 0.80, "ip": 0.20})),
        return_=dict(_get(ts, ["work", "return"], {"pm": 0.80, "ev": 0.20})),
    )
    shares_school = PeriodShares(
        outbound=dict(_get(ts, ["school", "outbound"], {"am": 0.90, "ip": 0.10})),
        return_=dict(_get(ts, ["school", "return"], {"pm": 0.70, "ev": 0.30})),
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
        s = sum(float(v) for v in d.values())
        for k in d:
            if str(k).lower().strip() not in periods:
                raise ValueError(f"{label}.{side_name}: unknown period '{k}', allowed: {periods}")
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"{label}.{side_name}: shares must sum to 1.0 (got {s:.4f})")


# ---------------------------------------------------------------------------
# Read & filter commuting data
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
        f"Commuting data not found. Run 'python run.py build-demand' first.\n"
        f"  tried: {bcfg.commuting_parquet}\n  tried: {bcfg.commuting_csv}"
    )


def _apply_origin_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(False, index=df.index)
    any_applied = False
    for clause in filters:
        f = clause.get("field")
        vals = clause.get("values") or []
        if not f or not vals or f not in df.columns:
            continue
        any_applied = True
        vals_norm = [str(v).strip().lower() for v in vals]
        mask = mask | df[f].astype(str).str.strip().str.lower().isin(vals_norm)
    return df[mask].copy() if any_applied else df


def _filter_commuting(df: pd.DataFrame, bcfg: DemandBuildCfg) -> pd.DataFrame:
    if "lokalizace" in df.columns and bcfg.include_lokalizace:
        df = df[df["lokalizace"].astype(str).isin(bcfg.include_lokalizace)].copy()
    df = _apply_origin_filters(df, bcfg.origin_filters)
    for c in ("dojizdka_prace", "dojizdka_skola"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Build OD cores
# ---------------------------------------------------------------------------

def _build_gravity_seed(
    zone_ids: np.ndarray,
    population: Dict[int, int],
    trip_rate: float,
    car_share: float,
    occupancy: float,
    impedance: Optional[np.ndarray] = None,
    centroids_gdf: Optional[gpd.GeoDataFrame] = None,
    excluded_zone_ids: Optional[set[int]] = None,
) -> np.ndarray:
    """Synthetic 'other' trip seed via simple gravity formulation."""
    excluded_zone_ids = excluded_zone_ids or set()
    n = len(zone_ids)
    pop = np.array(
        [
            0.0 if int(z) in excluded_zone_ids else max(population.get(int(z), 0), 0)
            for z in zone_ids
        ],
        dtype=np.float64,
    )
    prod = pop * trip_rate * car_share / max(occupancy, 0.01)
    attr = pop.copy()

    if impedance is not None and impedance.shape == (n, n):
        imp = impedance.copy()
    elif centroids_gdf is not None and len(centroids_gdf) >= n:
        if centroids_gdf.crs is not None and centroids_gdf.crs.to_epsg() != 5514:
            cg = centroids_gdf.to_crs(epsg=5514)
        else:
            cg = centroids_gdf
        pts = cg.geometry.representative_point()
        z2i = {int(z): i for i, z in enumerate(zone_ids)}
        coords = np.zeros((n, 2), dtype=np.float64)
        for idx, (_, row) in enumerate(cg.iterrows()):
            zid = int(row.get("zone_id", 0))
            if zid in z2i:
                coords[z2i[zid]] = [pts.iloc[idx].x, pts.iloc[idx].y]
        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        imp = np.sqrt(dx ** 2 + dy ** 2)
    else:
        imp = np.ones((n, n), dtype=np.float64)

    imp = np.maximum(imp, 100.0)
    beta = 0.0001
    f = np.exp(-beta * imp)
    np.fill_diagonal(f, 0)

    od = prod[:, None] * attr[None, :] * f
    total_prod = prod.sum()
    total_od = od.sum()
    if total_od > 0 and total_prod > 0:
        od *= total_prod / total_od

    return od


def _build_external_seed(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    population: Dict[int, int],
    total_daily_trips: float,
    through_share: float = 0.30,
) -> np.ndarray:
    """Build external OD using corridor-based gateways.

    Key changes vs old version:
    - through traffic is distributed corridor-to-corridor, not zone-to-zone globally
    - gateway-to-internal traffic does not spill into other gateway zones
    - internal destinations are weighted only over non-gateway zones
    """
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = np.zeros((n, n), dtype=np.float64)

    if not gateways or total_daily_trips <= 0:
        return od

    gateway_zone_ids = set()
    for zones in gateways.values():
        for zid, _ in zones:
            if zid in z2i:
                gateway_zone_ids.add(int(zid))

    internal_zone_ids = [int(z) for z in zone_ids if int(z) not in gateway_zone_ids]
    if not internal_zone_ids:
        return od

    internal_pop = np.array([max(population.get(z, 0), 0) for z in internal_zone_ids], dtype=np.float64)
    if internal_pop.sum() <= 0:
        return od
    internal_pop = internal_pop / internal_pop.sum()

    corridor_names = sorted(gateways.keys())
    n_corr = len(corridor_names)
    if n_corr == 0:
        return od

    # ------------------------------------------------------------------
    # Part 1: external <-> internal
    # ------------------------------------------------------------------
    ext_internal = total_daily_trips * (1.0 - through_share)
    ext_internal_half = ext_internal / 2.0

    per_corridor = ext_internal_half / n_corr

    for gw_name in corridor_names:
        gw_zones = [(zid, w) for zid, w in gateways[gw_name] if zid in z2i]
        if not gw_zones:
            continue

        for gw_zid, gw_w in gw_zones:
            gi = z2i[int(gw_zid)]
            gw_zone_trips = per_corridor * float(gw_w)

            for dz, share in zip(internal_zone_ids, internal_pop):
                di = z2i[int(dz)]
                if gi == di:
                    continue
                od[gi, di] += gw_zone_trips * float(share)
                od[di, gi] += gw_zone_trips * float(share)

    # ------------------------------------------------------------------
    # Part 2: through traffic (corridor -> corridor)
    # ------------------------------------------------------------------
    through_daily = total_daily_trips * through_share

    if n_corr > 1 and through_daily > 0:
        ordered_pairs = [(a, b) for a in corridor_names for b in corridor_names if a != b]
        trips_per_pair = through_daily / len(ordered_pairs)

        for a_name, b_name in ordered_pairs:
            a_zones = [(zid, w) for zid, w in gateways[a_name] if zid in z2i]
            b_zones = [(zid, w) for zid, w in gateways[b_name] if zid in z2i]
            if not a_zones or not b_zones:
                continue

            for a_zid, a_w in a_zones:
                ai = z2i[int(a_zid)]
                for b_zid, b_w in b_zones:
                    bi = z2i[int(b_zid)]
                    if ai == bi:
                        continue
                    od[ai, bi] += trips_per_pair * float(a_w) * float(b_w)

        through_total = sum(
            od[z2i[a_zid], z2i[b_zid]]
            for a_name in corridor_names
            for b_name in corridor_names
            if a_name != b_name
            for a_zid, _ in gateways[a_name]
            for b_zid, _ in gateways[b_name]
            if a_zid in z2i and b_zid in z2i and z2i[a_zid] != z2i[b_zid]
        )
        print(f"  Through-traffic (corridor->corridor): {through_total:,.0f} trips/day "
              f"({n_corr} corridors)")

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
    zones_gdf: Optional[gpd.GeoDataFrame] = None,
    zone_population: Optional[Dict[int, int]] = None,
    gateways: Optional[Dict[str, List[Tuple[int, float]]]] = None,
    use_gateway_fallback: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    _validate_shares(bcfg.periods, bcfg.shares_work, "weekday.work")
    _validate_shares(bcfg.periods, bcfg.shares_school, "weekday.school")

    z2i = _zone_index(zone_ids)
    z_set = set(z2i.keys())
    period_cores = [f"wd_{p}" for p in bcfg.periods]
    core_names = period_cores + ["wd_daily"]
    n = len(zone_ids)
    mats: Dict[str, np.ndarray] = {c: np.zeros((n, n), dtype=np.float64) for c in core_names}

    stats: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "mapped_direct": 0,
        "mapped_group": 0,
        "mapped_gateway": 0,
        "missing_origin": 0,
        "missing_destination": 0,
        "skipped_external": 0,
        "pairs_used": 0,
    }

    op_col = "op_obec" if "op_obec" in df.columns else None
    doj_col = "doj_obec" if "doj_obec" in df.columns else None
    if op_col is None or doj_col is None:
        raise RuntimeError(f"Expected columns op_obec/doj_obec, got: {list(df.columns)}")

    _ext_cache: Dict[str, List[Tuple[int, float]]] = {}
    use_gateways = bool(gateways) and bool(use_gateway_fallback) and not bcfg.only_internal_pairs
    model_center: Tuple[float, float] = (0.0, 0.0)
    if zones_gdf is not None and not zones_gdf.empty:
        if zones_gdf.crs is not None and zones_gdf.crs.to_epsg() != 5514:
            _zm = zones_gdf.to_crs(epsg=5514)
        else:
            _zm = zones_gdf
        _rp = _zm.geometry.representative_point()
        model_center = (float(_rp.x.mean()), float(_rp.y.mean()))

    def _resolve(name: str) -> Tuple[List[Tuple[int, float]], str]:
        zid = _match_zone_id(name, primary, stripped)
        if zid is not None:
            return [(zid, 1.0)], "direct"
        key = _norm_name(name)
        if key in groups:
            return groups[key], "group"
        if use_gateways:
            if key not in _ext_cache:
                _ext_cache[key] = _assign_external_to_gateway(
                    key, gateways, model_center,
                )
            cands = _ext_cache[key]
            if cands:
                return cands, "gateway"
        return [], "miss"

    for _, r in df.iterrows():
        o_name = str(r[op_col]).strip() if pd.notna(r[op_col]) else ""
        d_name = str(r[doj_col]).strip() if pd.notna(r[doj_col]) else ""
        if not o_name or not d_name:
            stats["missing_origin" if not o_name else "missing_destination"] += 1
            continue

        o_cand, o_mode = _resolve(o_name)
        d_cand, d_mode = _resolve(d_name)

        if not o_cand:
            stats["missing_origin"] += 1
            continue
        if not d_cand:
            stats["missing_destination"] += 1
            continue

        for mode in (o_mode, d_mode):
            if mode == "direct":
                stats["mapped_direct"] += 1
            elif mode == "group":
                stats["mapped_group"] += 1
            elif mode == "gateway":
                stats["mapped_gateway"] += 1

        work_v = _persons_to_vehicles(float(r.get("dojizdka_prace", 0)), bcfg.conv_work)
        school_v = _persons_to_vehicles(float(r.get("dojizdka_skola", 0)), bcfg.conv_school)
        if work_v <= 0 and school_v <= 0:
            continue

        for oz, ow in o_cand:
            if oz not in z_set:
                continue
            oi = z2i[oz]
            for dz, dw in d_cand:
                if dz not in z_set:
                    continue
                if bcfg.only_internal_pairs and (oz not in z_set or dz not in z_set):
                    stats["skipped_external"] += 1
                    continue
                di = z2i[dz]
                f = ow * dw

                for p, sh in bcfg.shares_work.outbound.items():
                    mats[f"wd_{p}"][oi, di] += work_v * sh * f
                for p, sh in bcfg.shares_school.outbound.items():
                    mats[f"wd_{p}"][oi, di] += school_v * sh * f
                for p, sh in bcfg.shares_work.return_.items():
                    mats[f"wd_{p}"][di, oi] += work_v * sh * f
                for p, sh in bcfg.shares_school.return_.items():
                    mats[f"wd_{p}"][di, oi] += school_v * sh * f

                stats["pairs_used"] += 1

    mats["wd_daily"] = sum(mats[c] for c in period_cores)

    summary = {
        **stats,
        "zones": int(len(zone_ids)),
        "cores_sum": {c: round(float(mats[c].sum()), 1) for c in core_names},
        "nonzero_cells": {c: int(np.count_nonzero(mats[c])) for c in core_names},
    }
    return mats, summary


# ---------------------------------------------------------------------------
# AEM output
# ---------------------------------------------------------------------------

def _write_aem(
    matrix_path: Path,
    index_ids: np.ndarray,
    cores: Dict[str, np.ndarray],
    *,
    matrix_name: str,
) -> None:
    """Write OD cores to an AequilibraE .aem file."""
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
        mat.setDescription(
            "OD from SLDB 2021 commuting (work+school) -> vehicle trips; weekday time-sliced."
        )
    except Exception:
        pass

    mat.index[:] = index_ids.astype(np.int32)
    for c in core_names:
        mat.matrix[c][:, :] = cores[c]
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
    zdir = _get(cfg, ["zoning", "output_dir"], "outputs/baseline/zones")
    zones_path = Path(zdir) / "zones.geojson"
    if not zones_path.exists():
        raise FileNotFoundError(
            f"zones.geojson not found: {zones_path}. Run: python run.py build-zones"
        )
    g = gpd.read_file(zones_path)
    if "zone_id" not in g.columns:
        raise RuntimeError(f"zones.geojson missing 'zone_id': {zones_path}")
    if "name" not in g.columns:
        g["name"] = ""
    g["zone_id"] = pd.to_numeric(g["zone_id"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["zone_id"]).copy()
    g["zone_id"] = g["zone_id"].astype(int)
    g["name"] = g["name"].astype(str)

    if "is_external" not in g.columns:
        g["is_external"] = 0
    g["is_external"] = pd.to_numeric(g["is_external"], errors="coerce").fillna(0).astype(int)

    if "gateway_name" not in g.columns:
        g["gateway_name"] = ""
    g["gateway_name"] = g["gateway_name"].fillna("").astype(str)

    return g


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_or_build_od_matrix(config_path: str | Path = "config/sim.yaml") -> None:
    cfg = load_config(config_path)
    bcfg = _build_cfg(cfg)

    print("=== BUILD OD MATRIX ===")

    print("Loading zones ...")
    zones_gdf = _load_zones(cfg)
    external_zone_ids: set[int] = set()
    if "is_external" in zones_gdf.columns:
        external_zone_ids = set(
            zones_gdf.loc[
                zones_gdf["is_external"].fillna(0).astype(int) == 1,
                "zone_id"
            ].astype(int)
        )
    print(f"  External zones: {len(external_zone_ids)}")
    zone_ids = np.array(
        sorted(zones_gdf["zone_id"].astype(int).unique()), dtype=np.int64
    )

    zdir = _get(cfg, ["zoning", "output_dir"], "outputs/baseline/zones")
    mapping_path = Path(zdir) / "zone_centroid_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"zone_centroid_mapping.json not found at {mapping_path}. "
            "Run 'python run.py build-zones' first."
        )
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    zone_to_centroid: Dict[int, int] = {int(k): int(v) for k, v in raw.items()}
    print(f"  Loaded zone->centroid mapping ({len(zone_to_centroid)} entries)")

    centroid_ids = np.array(
        [zone_to_centroid.get(int(z), int(z)) for z in zone_ids], dtype=np.int64
    )
    print(f"  {len(zone_ids)} zones loaded (centroid IDs: {centroid_ids.min()}-{centroid_ids.max()})")

    primary, stripped = _build_zone_name_index(zones_gdf)
    print(f"  Name index: {len(primary)} primary, {len(stripped)} stripped entries")

    print("Reading commuting data ...")
    df_raw = _read_commuting(bcfg)

    zone_population = _load_zone_population(
        Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))
    )
    obec_col = "op_obec" if "op_obec" in df_raw.columns else None
    csv_obecs: set[str] = set()
    if obec_col:
        csv_obecs = set(df_raw[obec_col].dropna().astype(str).unique())
    groups = _build_brno_group(zones_gdf, primary, stripped, csv_obecs,
                               zone_population=zone_population)
    if groups:
        for gname, members in groups.items():
            print(f"  Group '{gname}': {len(members)} zones")

    gateways = _load_gateways(cfg, zones_gdf, zone_population=zone_population)
    if gateways:
        print(f"  Gateways: {len(gateways)} corridors")
        for gn, gz in gateways.items():
            print(f"    {gn}: {len(gz)} boundary zones")
    else:
        print("  No gateways configured - external trips use fallback")

    df = _filter_commuting(df_raw, bcfg)
    print(f"  {len(df)} rows after filtering (lokalizace + origin region)")

    print("Building commuting OD cores ...")
    use_gateway_fallback_in_commuting = bool(
        _get(cfg, ["demand", "use_gateway_fallback_in_commuting"], False)
    )
    cores, summary = _build_od_cores(
        df,
        zone_ids,
        bcfg,
        primary=primary,
        stripped=stripped,
        groups=groups,
        zones_gdf=zones_gdf,
        zone_population=zone_population,
        gateways=gateways,
        use_gateway_fallback=use_gateway_fallback_in_commuting,
    )

    seg_cfg = _get(cfg, ["demand", "segments"], {}) or {}
    other_cfg = seg_cfg.get("other", {}) or {}
    n_zones = len(zone_ids)

    if other_cfg and other_cfg.get("source", "gravity") == "gravity":
        print("Building 'other' trips (gravity seed) ...")
        other_daily = _build_gravity_seed(
            zone_ids,
            zone_population,
            trip_rate=float(other_cfg.get("trip_rate", 1.5)),
            car_share=float(other_cfg.get("car_share", 0.45)),
            occupancy=float(other_cfg.get("occupancy", 1.40)),
            centroids_gdf=zones_gdf,
            excluded_zone_ids=external_zone_ids,
        )
        other_total = float(other_daily.sum())
        print(f"  'other' daily total: {other_total:,.0f}")
    else:
        other_daily = np.zeros((n_zones, n_zones), dtype=np.float64)

    ext_cfg = seg_cfg.get("external", {}) or {}
    if ext_cfg and ext_cfg.get("source") == "gateway_gravity" and gateways:
        print("Building external trips (gateway seed) ...")
        ext_daily = _build_external_seed(
            zone_ids, gateways, zone_population,
            total_daily_trips=float(ext_cfg.get("total_daily_trips", 80000)),
            through_share=float(ext_cfg.get("through_share", 0.30)),
        )
        ext_total = float(ext_daily.sum())
        print(f"  'external' daily total: {ext_total:,.0f}")
    else:
        ext_daily = np.zeros((n_zones, n_zones), dtype=np.float64)

    period_cores = [f"wd_{p}" for p in bcfg.periods]
    period_shares_even = {p: 1.0 / len(bcfg.periods) for p in bcfg.periods}

    all_cores: Dict[str, np.ndarray] = {}
    for cn in list(cores.keys()):
        all_cores[cn] = cores[cn]

    for p in bcfg.periods:
        share = period_shares_even[p]
        all_cores[f"wd_{p}_other"] = other_daily * share
        all_cores[f"wd_{p}_external"] = ext_daily * share
    all_cores["wd_daily_other"] = other_daily.copy()
    all_cores["wd_daily_external"] = ext_daily.copy()

    for p in bcfg.periods:
        all_cores[f"wd_{p}"] = (
            cores.get(f"wd_{p}", np.zeros((n_zones, n_zones)))
            + all_cores[f"wd_{p}_other"]
            + all_cores[f"wd_{p}_external"]
        )
    all_cores["wd_daily"] = sum(all_cores[f"wd_{p}"] for p in bcfg.periods)

    segment_totals = {
        "commuting": round(float(cores.get("wd_daily", np.zeros(1)).sum()), 0),
        "other": round(float(other_daily.sum()), 0),
        "external": round(float(ext_daily.sum()), 0),
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
            "work": {"car_share": bcfg.conv_work.car_share,
                     "occupancy": bcfg.conv_work.occupancy,
                     "trips_per_person": bcfg.conv_work.trips_per_person},
            "school": {"car_share": bcfg.conv_school.car_share,
                       "occupancy": bcfg.conv_school.occupancy,
                       "trips_per_person": bcfg.conv_school.trips_per_person},
        },
        "periods": bcfg.periods,
        "segments": segment_totals,
        "groups": {k: {"zones": len(v)} for k, v in groups.items()},
        "gateways": {k: len(v) for k, v in gateways.items()},
        **summary,
    }
    summary_path = bcfg.output_dir / "od_summary.json"
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8"
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
    for c in sorted(all_cores.keys()):
        v = round(float(all_cores[c].sum()), 1)
        nz = int(np.count_nonzero(all_cores[c]))
        print(f"  {c:24s}  total={v:>12.1f}  nonzero={nz}")
    print(f"\n  Matrix:  {bcfg.matrix_path}")
    print(f"  Summary: {summary_path}")
