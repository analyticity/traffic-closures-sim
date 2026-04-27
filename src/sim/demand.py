"""Build OD matrices from commuting data, supernetwork-derived external flows,
and optional residual synthetic gateway segments; register them in AequilibraE.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix

from sim.fetch_datasets import resolved_commuting_full_cr_parquet_path
from sim.io_project import get_metric_epsg, load_config
from sim._text import norm_name as _norm_name
from sim._metrics import persons_to_vehicles


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_GEO_SUFFIXES = re.compile(r"\s+(u|nad|pod|na|ve|pri|při)\s+\S+$", re.IGNORECASE)


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


def _assign_external_to_gateway_fallback(
    place_name: str,
    gateways: Dict[str, List[Tuple[int, float]]],
) -> List[Tuple[int, float]]:
    """
    Legacy deterministic fallback when no supernetwork lookup exists.
    Kept only as a last resort for compatibility/debug.
    """
    if not gateways:
        return []

    gateway_names = sorted(gateways.keys())
    stable = hashlib.blake2b(place_name.encode("utf-8"), digest_size=8).hexdigest()
    idx = int(stable, 16) % len(gateway_names)
    return gateways[gateway_names[idx]]


def _preflight_external_inputs(
    bcfg: "DemandBuildCfg",
    gateways: Dict[str, List[Tuple[int, float]]],
) -> None:
    if not bcfg.external.enabled:
        return

    if not gateways:
        raise RuntimeError(
            "External processing is enabled, but no gateway zones were found. "
            "Run build-zones and verify external gateways configuration."
        )

    lookup_path = bcfg.external.gateway_lookup_path
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Missing external gateway lookup: {lookup_path}. "
            "Run build-supernetwork before build-demand."
        )

    lookup_df = pd.read_parquet(lookup_path)
    if lookup_df.empty:
        raise RuntimeError(
            f"External gateway lookup is empty: {lookup_path}. "
            "Rebuild supernetwork and check centroid/gateway mapping coverage."
        )

    required_lookup = {"gateway_name"}
    missing_lookup = required_lookup - set(lookup_df.columns)
    has_place = ("place_name" in lookup_df.columns) or ("place_name_norm" in lookup_df.columns)
    if missing_lookup or not has_place:
        raise RuntimeError(
            f"External gateway lookup schema mismatch at {lookup_path}. "
            f"Missing required columns: {sorted(missing_lookup)}; "
            "expected place_name or place_name_norm."
        )

    usable_lookup = lookup_df[lookup_df["gateway_name"].astype(str).isin(set(gateways.keys()))]
    if usable_lookup.empty:
        raise RuntimeError(
            f"External gateway lookup at {lookup_path} has no rows matching current gateway names. "
            "Re-run build-zones and build-supernetwork in the same pipeline run."
        )

    if bcfg.external.use_through_traffic:
        through_path = bcfg.external.through_pairs_path
        if not through_path.exists():
            raise FileNotFoundError(
                f"Missing through gateway pairs: {through_path}. "
                "Run build-supernetwork before build-demand."
            )
        through_df = pd.read_parquet(through_path)
        required_through = {"gateway_in", "gateway_out", "vehicles_daily"}
        missing_through = required_through - set(through_df.columns)
        if missing_through:
            raise RuntimeError(
                f"through_gateway_pairs schema mismatch at {through_path}. "
                f"Missing required columns: {sorted(missing_through)}."
            )


def _load_external_gateway_lookup_directional(
    path: Path,
    gateways: Dict[str, List[Tuple[int, float]]],
) -> Tuple[
    Dict[str, List[Tuple[int, float]]],
    Dict[str, List[Tuple[int, float]]],
    Dict[str, str],
    Dict[str, str],
]:
    """Build **directional** gateway lookup for external places.

    The supernetwork stores separate ``rank_in`` / ``rank_out`` and
    ``route_cost_to_gateway_s`` / ``route_cost_from_gateway_s`` per
    (place, gateway) pair.  A town south of Brno may enter the model
    via D2_S (inbound) yet exit via I52_S (outbound).  Collapsing to
    a single gateway per place loses that directional information and
    starves some corridors of demand.

    Returns
    -------
    inbound_members : place_norm -> [(zone_id, weight), ...]
        Gateway zones for traffic **entering** the model (place is origin).
    outbound_members : place_norm -> [(zone_id, weight), ...]
        Gateway zones for traffic **leaving** the model (place is destination).
    inbound_gateway : place_norm -> gateway_name
    outbound_gateway : place_norm -> gateway_name
    """
    _empty: Tuple[dict, dict, dict, dict] = ({}, {}, {}, {})

    if not path.exists():
        print(f"  External gateway lookup not found: {path}")
        return _empty

    df = pd.read_parquet(path)
    if df.empty:
        print(f"  External gateway lookup is empty: {path}")
        return _empty

    if "place_name_norm" not in df.columns:
        if "place_name" in df.columns:
            df["place_name_norm"] = df["place_name"].astype(str).map(_norm_name)
        else:
            raise RuntimeError(
                f"External gateway lookup missing place_name/place_name_norm: {path}"
            )

    if "gateway_name" not in df.columns:
        raise RuntimeError(f"External gateway lookup missing gateway_name: {path}")

    df["place_name_norm"] = df["place_name_norm"].astype(str).map(_norm_name)
    df["gateway_name"] = df["gateway_name"].astype(str).str.strip()
    df = df[df["place_name_norm"] != ""].copy()
    df = df[df["gateway_name"].isin(set(gateways.keys()))].copy()
    if df.empty:
        print(f"  External gateway lookup has no usable rows after filtering: {path}")
        return _empty

    has_directional = (
        "rank_in" in df.columns and "rank_out" in df.columns
    )

    if has_directional:
        in_cost = "route_cost_to_gateway_s" if "route_cost_to_gateway_s" in df.columns else "route_cost_s"
        out_cost = "route_cost_from_gateway_s" if "route_cost_from_gateway_s" in df.columns else "route_cost_s"
    else:
        in_cost = "route_cost_s" if "route_cost_s" in df.columns else None
        out_cost = in_cost

    def _build_weighted_members(
        df_src: pd.DataFrame,
        cost_col: Optional[str],
    ) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[str, str]]:
        """Build place -> [(zone_id, weight)] using ALL candidates with
        cost-inverse weighting instead of picking only the best one."""
        members: Dict[str, List[Tuple[int, float]]] = {}
        primary_gw: Dict[str, str] = {}

        if cost_col is None or cost_col not in df_src.columns:
            return members, primary_gw

        for place, grp in df_src.groupby("place_name_norm"):
            pn = str(place).strip()
            if not pn:
                continue

            costs = pd.to_numeric(grp[cost_col], errors="coerce").values
            gw_names = grp["gateway_name"].values

            valid_mask = np.isfinite(costs) & (costs > 0)
            if not valid_mask.any():
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                inv_costs = np.where(valid_mask, 1.0 / costs, 0.0)
            total_inv = inv_costs.sum()
            if total_inv <= 0:
                continue
            weights = inv_costs / total_inv

            zone_list: List[Tuple[int, float]] = []
            best_gw: Optional[str] = None
            best_w = -1.0

            for gw_name_raw, w in zip(gw_names, weights):
                if w <= 0:
                    continue
                gw_name = str(gw_name_raw).strip()
                if gw_name not in gateways:
                    continue
                if best_gw is None or w > best_w:
                    best_gw = gw_name
                    best_w = w
                for zone_id, zone_w in gateways[gw_name]:
                    zone_list.append((zone_id, zone_w * w))

            if zone_list:
                members[pn] = zone_list
                primary_gw[pn] = best_gw or ""

        return members, primary_gw

    inbound_members, inbound_gateway = _build_weighted_members(df, in_cost)
    outbound_members, outbound_gateway = _build_weighted_members(df, out_cost)

    n_differ = sum(
        1 for pn in inbound_gateway
        if pn in outbound_gateway and inbound_gateway[pn] != outbound_gateway[pn]
    )

    gw_counts_in: Dict[str, int] = {}
    for pn, zlist in inbound_members.items():
        for gw_name in set(str(g) for g in [inbound_gateway.get(pn, "")]):
            gw_counts_in[gw_name] = gw_counts_in.get(gw_name, 0) + 1
    top_in = sorted(gw_counts_in.items(), key=lambda x: -x[1])
    gw_summary = ", ".join(f"{g}={c}" for g, c in top_in[:5])

    print(
        f"  External gateway lookup loaded (multi-candidate weighted): "
        f"{len(inbound_members)} inbound, {len(outbound_members)} outbound places "
        f"({n_differ} differ in/out); primary gw: {gw_summary}"
    )
    return inbound_members, outbound_members, inbound_gateway, outbound_gateway


def _load_through_gateway_pairs(path: Path) -> pd.DataFrame:
    """
    Load aggregated external->external flows accepted by coarse supernetwork.

    Expected columns:
      - gateway_in
      - gateway_out
      - vehicles_daily
    """
    if not path.exists():
        print(f"  Through gateway pairs not found: {path}")
        return pd.DataFrame(columns=["gateway_in", "gateway_out", "vehicles_daily"])

    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["gateway_in", "gateway_out", "vehicles_daily"])

    required = {"gateway_in", "gateway_out", "vehicles_daily"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"through_gateway_pairs missing required columns {sorted(missing)}: {path}"
        )

    df = df.copy()
    df["gateway_in"] = df["gateway_in"].astype(str).str.strip()
    df["gateway_out"] = df["gateway_out"].astype(str).str.strip()
    df["vehicles_daily"] = pd.to_numeric(df["vehicles_daily"], errors="coerce").fillna(0.0)
    df = df[(df["gateway_in"] != "") & (df["gateway_out"] != "") & (df["vehicles_daily"] > 0)].copy()
    return df


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
class ExternalProcessingCfg:
    enabled: bool
    use_full_cr_dataset: bool
    use_external_internal: bool
    use_internal_external: bool
    use_through_traffic: bool
    gateway_lookup_path: Path
    through_pairs_path: Path
    allow_legacy_fallback: bool
    external_commuting_scale: float


@dataclass(frozen=True)
class DemandBuildCfg:
    commuting_filtered_parquet: Path
    commuting_full_cr_parquet: Path
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
    external: ExternalProcessingCfg


def _resolve_commuting_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    csv_path = _get(cfg, ["datasets", "sources", "commuting_sldb2021", "out_path"])
    if csv_path:
        csv_path = _as_path(csv_path)
    else:
        csv_path = _as_path("data/sources/csu/sldb2021/dojizdka_obce.csv")

    cache_dir = _as_path(_get(cfg, ["datasets", "cache_dir"], "data/cache"))
    filtered_parquet = _as_path(
        _get(cfg, ["datasets", "sources", "commuting_sldb2021", "filtered_out_parquet"],
             cache_dir / f"{csv_path.stem}.parquet")
    )
    full_cr_explicit = _get(cfg, ["datasets", "sources", "commuting_sldb2021", "full_cr_out_parquet"], None)
    full_cr_parquet = (
        _as_path(full_cr_explicit)
        if full_cr_explicit
        else resolved_commuting_full_cr_parquet_path(cfg)
    )
    return csv_path, filtered_parquet, full_cr_parquet


def _build_cfg(cfg: Dict[str, Any]) -> DemandBuildCfg:
    demand = cfg.get("demand") or {}
    csv_path, filtered_parquet, full_cr_parquet = _resolve_commuting_paths(cfg)
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

    ext_cfg = _get(demand, ["sldb", "external_processing"], {}) or {}
    seg_ext_through = _get(demand, ["segments", "external_through"], {}) or {}
    _cache = _as_path(_get(cfg, ["datasets", "cache_dir"], "data/cache"))

    gateway_lookup_path = _as_path(
        ext_cfg.get("external_gateway_lookup_path", str(_cache / "external_gateway_lookup.parquet"))
    )
    through_pairs_path = _as_path(
        ext_cfg.get(
            "through_gateway_pairs_path",
            seg_ext_through.get("data_driven_pairs_path", str(_cache / "through_gateway_pairs.parquet")),
        )
    )

    external = ExternalProcessingCfg(
        enabled=bool(ext_cfg.get("enabled", True)),
        use_full_cr_dataset=bool(ext_cfg.get("use_full_cr_dataset", True)),
        use_external_internal=bool(ext_cfg.get("use_external_internal", True)),
        use_internal_external=bool(ext_cfg.get("use_internal_external", True)),
        use_through_traffic=bool(ext_cfg.get("use_through_traffic", True)),
        gateway_lookup_path=gateway_lookup_path,
        through_pairs_path=through_pairs_path,
        allow_legacy_fallback=bool(_get(demand, ["use_gateway_fallback_in_commuting"], False)),
        external_commuting_scale=float(ext_cfg.get("external_commuting_scale", 0.75)),
    )

    return DemandBuildCfg(
        commuting_filtered_parquet=filtered_parquet,
        commuting_full_cr_parquet=full_cr_parquet,
        commuting_csv=csv_path,
        csv_delimiter=str(fmt.get("delimiter", ",")),
        csv_encoding=str(fmt.get("encoding", "utf-8")),
        include_lokalizace=[str(x) for x in include_lok],
        only_internal_pairs=bool(_get(demand, ["sldb", "only_internal_pairs"], False)),
        origin_filters=origin_filters,
        conv_work=conv_work,
        conv_school=conv_school,
        periods=periods,
        shares_work=shares_work,
        shares_school=shares_school,
        output_dir=output_dir,
        matrix_path=matrix_path,
        matrix_name=matrix_name,
        external=external,
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
    if not raw:
        return {p: 1.0 / len(periods) for p in periods}

    shares = {p: float(raw.get(p, 0.0)) for p in periods}
    total = sum(shares.values())
    if total <= 0:
        return {p: 1.0 / len(periods) for p in periods}
    return {p: v / total for p, v in shares.items()}


_LINK_TYPE_WEIGHT_SCALE: Dict[str, float] = {
    "motorway": 1.0,
    "motorway_link": 1.0,
    "trunk": 0.9,
    "trunk_link": 0.9,
    "primary": 0.7,
    "primary_link": 0.7,
    "secondary": 0.4,
    "secondary_link": 0.4,
}


def _normalize_named_weights(
    names: List[str],
    raw: Optional[Dict[str, Any]],
    *,
    gateway_link_types: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    if not names:
        return {}

    raw = raw or {}

    if not raw and not gateway_link_types:
        return {name: 1.0 / len(names) for name in names}

    if not raw and gateway_link_types:
        vals = {
            name: _LINK_TYPE_WEIGHT_SCALE.get(
                str(gateway_link_types.get(name, "")), 0.2
            )
            for name in names
        }
        total = sum(vals.values())
        if total <= 0:
            return {name: 1.0 / len(names) for name in names}
        return {name: v / total for name, v in vals.items()}

    specified = [max(float(v), 0.0) for v in raw.values() if v is not None]
    default_val = float(np.median(specified)) if specified else 1.0

    def _default_for(name: str) -> float:
        if gateway_link_types and name in gateway_link_types:
            lt = str(gateway_link_types[name])
            scale = _LINK_TYPE_WEIGHT_SCALE.get(lt, 0.2)
            return default_val * scale
        return default_val

    vals = {name: max(float(raw.get(name, _default_for(name))), 0.0) for name in names}
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
    if bcfg.external.enabled and bcfg.external.use_full_cr_dataset:
        if bcfg.commuting_full_cr_parquet.exists():
            print(f"  Reading full-CR commuting parquet: {bcfg.commuting_full_cr_parquet}")
            return pd.read_parquet(bcfg.commuting_full_cr_parquet)

    if bcfg.commuting_filtered_parquet.exists():
        print(f"  Reading filtered commuting parquet: {bcfg.commuting_filtered_parquet}")
        return pd.read_parquet(bcfg.commuting_filtered_parquet)

    if bcfg.external.enabled and bcfg.external.use_full_cr_dataset and bcfg.commuting_csv.exists():
        print(f"  Reading raw commuting CSV: {bcfg.commuting_csv}")
        return pd.read_csv(
            bcfg.commuting_csv,
            sep=bcfg.csv_delimiter,
            encoding=bcfg.csv_encoding,
            low_memory=False,
        )

    raise FileNotFoundError(
        f"Commuting data not found.\n"
        f"  tried: {bcfg.commuting_full_cr_parquet}\n"
        f"  tried: {bcfg.commuting_filtered_parquet}\n"
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

    # Skip origin filter for full-CR dataset to preserve external-internal links.
    if not (bcfg.external.enabled and bcfg.external.use_full_cr_dataset):
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
    gateway_link_types: Optional[Dict[str, str]] = None,
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
    corridor_weights_norm = _normalize_named_weights(
        corridor_names, corridor_weights, gateway_link_types=gateway_link_types,
    )

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


def _build_external_through_from_pairs(
    zone_ids: np.ndarray,
    gateways: Dict[str, List[Tuple[int, float]]],
    pairs_df: pd.DataFrame,
) -> np.ndarray:
    n = len(zone_ids)
    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    od = _zero_matrix(n)

    if pairs_df.empty or not gateways:
        return od

    for _, row in pairs_df.iterrows():
        gateway_in = str(row["gateway_in"]).strip()
        gateway_out = str(row["gateway_out"]).strip()
        total = float(row.get("vehicles_daily", 0.0))

        if total <= 0 or gateway_in == gateway_out:
            continue
        if gateway_in not in gateways or gateway_out not in gateways:
            continue

        in_members = [(zid, w) for zid, w in gateways[gateway_in] if int(zid) in z2i]
        out_members = [(zid, w) for zid, w in gateways[gateway_out] if int(zid) in z2i]
        if not in_members or not out_members:
            continue

        for in_zid, in_w in in_members:
            ii = z2i[int(in_zid)]
            for out_zid, out_w in out_members:
                oi = z2i[int(out_zid)]
                if ii == oi:
                    continue
                od[ii, oi] += total * float(in_w) * float(out_w)

    return od



def _zone_index(zone_ids: np.ndarray) -> Dict[int, int]:
    return {int(z): i for i, z in enumerate(zone_ids)}


def _resolve_place_candidates_directional(
    name: str,
    *,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    groups: Dict[str, List[Tuple[int, float]]],
    inbound_lookup: Dict[str, List[Tuple[int, float]]],
    outbound_lookup: Dict[str, List[Tuple[int, float]]],
    gateways: Dict[str, List[Tuple[int, float]]],
    allow_legacy_fallback: bool,
) -> Tuple[
    List[Tuple[int, float]], List[Tuple[int, float]], str, str,
]:
    """Resolve place name to zone candidates with separate in/out gateways.

    Returns (inbound_candidates, outbound_candidates, mode, kind).
    For internal places both lists are identical.
    """
    zone_id = _match_zone_id(name, primary, stripped)
    if zone_id is not None:
        c = [(zone_id, 1.0)]
        return c, c, "direct", "internal"

    key = _norm_name(name)
    if key in groups:
        c = groups[key]
        return c, c, "group", "internal"

    in_cands = inbound_lookup.get(key)
    out_cands = outbound_lookup.get(key)
    if in_cands or out_cands:
        # Keep directions separate: empty list for missing direction means
        # the OD loop will skip trips in that direction rather than route
        # through the wrong gateway.
        return in_cands or [], out_cands or [], "lookup", "external"

    if allow_legacy_fallback and gateways:
        cands = _assign_external_to_gateway_fallback(key, gateways)
        if cands:
            return cands, cands, "legacy_fallback", "external"

    return [], [], "missing", "missing"


def _build_od_cores(
    df: pd.DataFrame,
    zone_ids: np.ndarray,
    bcfg: DemandBuildCfg,
    *,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    groups: Dict[str, List[Tuple[int, float]]],
    gateways: Dict[str, List[Tuple[int, float]]],
    external_lookup: Dict[str, List[Tuple[int, float]]],
    external_lookup_out: Optional[Dict[str, List[Tuple[int, float]]]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    _validate_shares(bcfg.periods, bcfg.shares_work, "weekday.work")
    _validate_shares(bcfg.periods, bcfg.shares_school, "weekday.school")

    if external_lookup_out is None:
        external_lookup_out = external_lookup

    z2i = _zone_index(zone_ids)
    zset = set(z2i.keys())
    n = len(zone_ids)

    period_cores = [f"wd_{p}" for p in bcfg.periods]
    core_names = period_cores + ["wd_daily"]
    mats: Dict[str, np.ndarray] = {name: _zero_matrix(n) for name in core_names}

    stats: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "mapped_direct": 0,
        "mapped_group": 0,
        "mapped_external_lookup": 0,
        "mapped_external_legacy_fallback": 0,
        "missing_origin": 0,
        "missing_destination": 0,
        "pairs_used": 0,
        "skipped_external_external_rows": 0,
    }

    origin_col = "op_obec" if "op_obec" in df.columns else None
    dest_col = "doj_obec" if "doj_obec" in df.columns else None
    if origin_col is None or dest_col is None:
        raise RuntimeError(f"Expected columns op_obec/doj_obec, got: {list(df.columns)}")

    # -- Phase 1: Pre-cache directional name resolutions --------------------
    # External places need different gateway zones depending on whether they
    # are the origin (traffic enters model -> inbound gateway) or the
    # destination (traffic exits model -> outbound gateway).
    origin_names = df[origin_col].fillna("").astype(str).str.strip().values
    dest_names = df[dest_col].fillna("").astype(str).str.strip().values

    all_unique_names: set[str] = set(origin_names) | set(dest_names)
    all_unique_names.discard("")

    _EMPTY_DIR: Tuple[list, list, str, str] = ([], [], "missing", "missing")
    name_cache_dir: Dict[str, Tuple[
        List[Tuple[int, float]], List[Tuple[int, float]], str, str,
    ]] = {}
    for name in all_unique_names:
        name_cache_dir[name] = _resolve_place_candidates_directional(
            name,
            primary=primary,
            stripped=stripped,
            groups=groups,
            inbound_lookup=external_lookup,
            outbound_lookup=external_lookup_out,
            gateways=gateways,
            allow_legacy_fallback=bcfg.external.allow_legacy_fallback,
        )
    print(f"  Pre-cached {len(name_cache_dir)} unique place-name resolutions (directional)")

    # -- Phase 2: Vectorize vehicle conversion ------------------------------
    work_col_name = "dojizdka_prace"
    school_col_name = "dojizdka_skola"
    work_raw = (
        pd.to_numeric(df[work_col_name], errors="coerce").fillna(0.0).values
        if work_col_name in df.columns else np.zeros(len(df))
    )
    school_raw = (
        pd.to_numeric(df[school_col_name], errors="coerce").fillna(0.0).values
        if school_col_name in df.columns else np.zeros(len(df))
    )

    w_conv = persons_to_vehicles(1.0, car_share=bcfg.conv_work.car_share,
                                occupancy=bcfg.conv_work.occupancy,
                                trips_per_person=bcfg.conv_work.trips_per_person)
    s_conv = persons_to_vehicles(1.0, car_share=bcfg.conv_school.car_share,
                                occupancy=bcfg.conv_school.occupancy,
                                trips_per_person=bcfg.conv_school.trips_per_person)
    work_v_all = np.maximum(work_raw, 0.0) * w_conv
    school_v_all = np.maximum(school_raw, 0.0) * s_conv

    # -- Phase 3: Pre-compute per-period share arrays -----------------------
    period_list = bcfg.periods
    n_periods = len(period_list)
    work_out = np.array([bcfg.shares_work.outbound.get(p, 0.0) for p in period_list])
    school_out = np.array([bcfg.shares_school.outbound.get(p, 0.0) for p in period_list])
    work_ret = np.array([bcfg.shares_work.return_.get(p, 0.0) for p in period_list])
    school_ret = np.array([bcfg.shares_school.return_.get(p, 0.0) for p in period_list])

    mat_arrays = [mats[f"wd_{p}"] for p in period_list]

    # -- Phase 4: Iterate rows using directional cached arrays --------------
    _MODE_STAT = {"direct": "mapped_direct", "group": "mapped_group",
                  "lookup": "mapped_external_lookup",
                  "legacy_fallback": "mapped_external_legacy_fallback"}
    n_rows = len(df)
    use_ext_int = bcfg.external.use_external_internal
    use_int_ext = bcfg.external.use_internal_external
    only_internal = bcfg.only_internal_pairs
    ext_comm_scale = bcfg.external.external_commuting_scale

    for idx in range(n_rows):
        on = origin_names[idx]
        dn = dest_names[idx]

        if not on:
            stats["missing_origin"] += 1
            continue
        if not dn:
            stats["missing_destination"] += 1
            continue

        o_entry = name_cache_dir.get(on, _EMPTY_DIR)
        d_entry = name_cache_dir.get(dn, _EMPTY_DIR)
        o_in_cands, o_out_cands, o_mode, o_kind = o_entry
        d_in_cands, d_out_cands, d_mode, d_kind = d_entry

        if not o_in_cands and not o_out_cands:
            stats["missing_origin"] += 1
            continue
        if not d_in_cands and not d_out_cands:
            stats["missing_destination"] += 1
            continue

        if o_kind == "external" and d_kind == "external":
            stats["skipped_external_external_rows"] += 1
            continue

        if only_internal and (o_kind != "internal" or d_kind != "internal"):
            continue
        if o_kind == "external" and d_kind == "internal" and not use_ext_int:
            continue
        if o_kind == "internal" and d_kind == "external" and not use_int_ext:
            continue

        o_stat = _MODE_STAT.get(o_mode)
        if o_stat:
            stats[o_stat] += 1
        d_stat = _MODE_STAT.get(d_mode)
        if d_stat:
            stats[d_stat] += 1

        wv = work_v_all[idx]
        sv = school_v_all[idx]
        if wv <= 0 and sv <= 0:
            continue

        fwd = wv * work_out + sv * school_out
        ret = wv * work_ret + sv * school_ret

        is_external_pair = (o_kind == "external") or (d_kind == "external")
        if is_external_pair and ext_comm_scale != 1.0:
            fwd = fwd * ext_comm_scale
            ret = ret * ext_comm_scale

        # Forward trip: origin -> destination
        #   origin external = traffic enters model -> use inbound gateway
        #   dest   external = traffic exits model  -> use outbound gateway
        fwd_o_cands = o_in_cands
        fwd_d_cands = d_out_cands

        # Return trip: destination -> origin (reverse direction)
        #   dest becomes origin (enters model) -> use inbound gateway
        #   origin becomes dest (exits model)  -> use outbound gateway
        ret_o_cands = d_in_cands
        ret_d_cands = o_out_cands

        for oz, ow in fwd_o_cands:
            if oz not in zset:
                continue
            oi = z2i[oz]
            for dz, dw in fwd_d_cands:
                if dz not in zset:
                    continue
                di = z2i[dz]
                f = float(ow) * float(dw)
                for pi in range(n_periods):
                    mat_arrays[pi][oi, di] += fwd[pi] * f

        for oz, ow in ret_o_cands:
            if oz not in zset:
                continue
            oi = z2i[oz]
            for dz, dw in ret_d_cands:
                if dz not in zset:
                    continue
                di = z2i[dz]
                f = float(ow) * float(dw)
                for pi in range(n_periods):
                    mat_arrays[pi][oi, di] += ret[pi] * f

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
        mat.setDescription("OD matrix built from commuting, supernetwork, and residual synthetic seeds")
    except Exception:
        pass

    mat.index[:] = index_ids.astype(np.int64)
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

def load_or_build_od_matrix(config_path: str | Path = "config/brno/sim.yaml") -> None:
    t0 = time.perf_counter()
    marks: Dict[str, float] = {}
    t_prev = t0

    def _mark(name: str) -> None:
        nonlocal t_prev
        now = time.perf_counter()
        marks[name] = round(now - t_prev, 3)
        t_prev = now

    cfg = load_config(config_path)
    bcfg = _build_cfg(cfg)

    print("=== BUILD OD MATRIX ===")

    print("Loading zones ...")
    zones_gdf = _load_zones(cfg)
    _mark("load_zones")

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
    _mark("read_commuting")

    zone_population = _load_zone_population(
        Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))
    )
    _mark("load_zone_population")

    origin_place_col = "op_obec" if "op_obec" in df_raw.columns else None
    dest_place_col = "doj_obec" if "doj_obec" in df_raw.columns else None
    csv_places: set[str] = set()
    if origin_place_col:
        csv_places |= set(df_raw[origin_place_col].dropna().astype(str).unique())
    if dest_place_col:
        csv_places |= set(df_raw[dest_place_col].dropna().astype(str).unique())

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
    _mark("build_hub_groups")

    gateways = _load_gateways(zones_gdf)
    if gateways:
        print(f"  Gateways: {len(gateways)} corridors")
        for gateway_name, gateway_zones in gateways.items():
            print(f"    {gateway_name}: {len(gateway_zones)} zones")
    else:
        print("  No external gateway zones found")

    _preflight_external_inputs(bcfg, gateways)
    _mark("preflight_external_inputs")

    external_lookup: Dict[str, List[Tuple[int, float]]] = {}
    external_lookup_out: Dict[str, List[Tuple[int, float]]] = {}
    if bcfg.external.enabled:
        (
            external_lookup,
            external_lookup_out,
            _gw_in_names,
            _gw_out_names,
        ) = _load_external_gateway_lookup_directional(
            bcfg.external.gateway_lookup_path,
            gateways,
        )

    df = _filter_commuting(df_raw, bcfg)
    print(f"  {len(df)} rows after filtering")
    _mark("filter_commuting")

    print("Building commuting OD cores ...")
    cores, summary = _build_od_cores(
        df,
        zone_ids,
        bcfg,
        primary=primary,
        stripped=stripped,
        groups=groups,
        gateways=gateways,
        external_lookup=external_lookup,
        external_lookup_out=external_lookup_out,
    )
    _mark("build_commuting_cores")

    segments_cfg = _get(cfg, ["demand", "segments"], {}) or {}
    other_cfg = segments_cfg.get("other", {}) or {}
    external_local_cfg = segments_cfg.get("external_local", {}) or {}
    external_through_cfg = segments_cfg.get("external_through", {}) or {}

    n_zones = len(zone_ids)

    # --- Internal "other" trips ---
    if other_cfg and other_cfg.get("source", "gravity") == "gravity":
        print("Building 'other' trips (gravity seed) ...")
        other_defaults = (other_cfg.get("defaults") or {})
        other_daily = _build_gravity_seed(
            zone_ids,
            zone_population,
            trip_rate=float(other_cfg.get("trip_rate", other_defaults.get("trip_rate", 1.0))),
            car_share=float(other_cfg.get("car_share", other_defaults.get("car_share", 0.38))),
            occupancy=float(other_cfg.get("occupancy", other_defaults.get("occupancy", 1.45))),
            beta=float(other_cfg.get("beta", other_defaults.get("beta", 0.00030))),
            centroids_gdf=zones_gdf,
            excluded_zone_ids=external_zone_ids,
            metric_epsg=get_metric_epsg(cfg),
        )
        print(f"  'other' daily total: {float(other_daily.sum()):,.0f}")
    else:
        other_daily = _zero_matrix(n_zones)
    _mark("build_other_seed")

    # --- Residual synthetic external_local (optional) ---
    _el_source = external_local_cfg.get("source", "gateway_local") if external_local_cfg else None
    if (
        external_local_cfg
        and bool(external_local_cfg.get("enabled", True))
        and _el_source == "gateway_local"
        and gateways
        and float(external_local_cfg.get("total_daily_trips", 0.0)) > 0
    ):
        print("Building residual synthetic external-local trips (gateway ↔ internal) ...")
        gw_link_types: Optional[Dict[str, str]] = None
        zoning_out = Path(cfg.get("zoning", {}).get("output_dir", "outputs/baseline/zones"))
        gw_diag_path = zoning_out / "gateway_diagnostics.csv"
        if gw_diag_path.exists():
            _gd = pd.read_csv(gw_diag_path)
            if "gateway_name" in _gd.columns:
                lt_col = "predominant_link_type" if "predominant_link_type" in _gd.columns else "link_type"
                if lt_col in _gd.columns:
                    gw_link_types = dict(zip(_gd["gateway_name"].astype(str), _gd[lt_col].astype(str)))
        external_local_daily = _build_external_local_seed(
            zone_ids,
            gateways,
            zone_population,
            total_daily_trips=float(external_local_cfg.get("total_daily_trips", 0.0)),
            corridor_weights=external_local_cfg.get("corridor_weights", {}) or {},
            gateway_link_types=gw_link_types,
        )
        print(f"  'external_local' daily total: {float(external_local_daily.sum()):,.0f}")
    else:
        external_local_daily = _zero_matrix(n_zones)
    _mark("build_external_local")

    # --- Data-driven external_through from coarse supernetwork ---
    data_driven_through_daily = _zero_matrix(n_zones)
    if bcfg.external.enabled and bcfg.external.use_through_traffic:
        through_pairs_df = _load_through_gateway_pairs(bcfg.external.through_pairs_path)
        if not through_pairs_df.empty:
            print("Building data-driven external-through trips from supernetwork gateway pairs ...")
            data_driven_through_daily = _build_external_through_from_pairs(
                zone_ids,
                gateways,
                through_pairs_df,
            )
            through_scale = float(
                _get(cfg, ["demand", "sldb", "external_processing", "through_traffic_scale"], 0.65) or 0.65
            )
            if through_scale != 1.0:
                data_driven_through_daily *= through_scale
                print(f"  Applied through_traffic_scale={through_scale:.2f}")
            print(f"  'external_through_data' daily total: {float(data_driven_through_daily.sum()):,.0f}")
    _mark("build_external_through_data")

    # --- Optional residual synthetic external_through ---
    if (
        external_through_cfg
        and bool(external_through_cfg.get("enabled", True))
        and external_through_cfg.get("source") == "gateway_pairs"
        and gateways
        and float(external_through_cfg.get("total_daily_trips", 0.0)) > 0
    ):
        print("Building residual synthetic external-through trips (gateway ↔ gateway) ...")
        pair_weights = _parse_gateway_pair_weights(
            external_through_cfg.get("pairs", []),
            sorted(gateways.keys()),
            default_pair_weight=float(external_through_cfg.get("default_pair_weight", 0.0)),
        )
        residual_external_through_daily = _build_external_through_seed(
            zone_ids,
            gateways,
            total_daily_trips=float(external_through_cfg.get("total_daily_trips", 0.0)),
            pair_weights=pair_weights,
        )
        print(f"  'external_through_residual' daily total: {float(residual_external_through_daily.sum()):,.0f}")
    else:
        residual_external_through_daily = _zero_matrix(n_zones)
    _mark("build_external_through_residual")

    external_through_daily = data_driven_through_daily + residual_external_through_daily

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
            all_cores[f"wd_{period}_external_local"]
            + all_cores[f"wd_{period}_external_through"]
        )

    all_cores["wd_daily_other"] = other_daily.copy()
    all_cores["wd_daily_external_local"] = external_local_daily.copy()
    all_cores["wd_daily_external_through_data"] = data_driven_through_daily.copy()
    all_cores["wd_daily_external_through_residual"] = residual_external_through_daily.copy()
    all_cores["wd_daily_external_through"] = external_through_daily.copy()
    all_cores["wd_daily_external"] = external_local_daily + external_through_daily
    all_cores["wd_daily_local"] = (
        all_cores["wd_daily_commuting"] + other_daily + external_local_daily
    )

    # Final combined cores used by the rest of the pipeline
    for period in bcfg.periods:
        all_cores[f"wd_{period}"] = (
            all_cores[f"wd_{period}_commuting"]
            + all_cores[f"wd_{period}_other"]
            + all_cores[f"wd_{period}_external_local"]
            + all_cores[f"wd_{period}_external_through"]
        )

    all_cores["wd_daily"] = sum(all_cores[f"wd_{period}"] for period in bcfg.periods)
    _mark("assemble_cores")

    segment_totals = {
        "commuting": round(float(all_cores["wd_daily_commuting"].sum()), 0),
        "other": round(float(other_daily.sum()), 0),
        "external_local": round(float(external_local_daily.sum()), 0),
        "external_through_data": round(float(data_driven_through_daily.sum()), 0),
        "external_through_residual": round(float(residual_external_through_daily.sum()), 0),
        "external_through_total": round(float(external_through_daily.sum()), 0),
        "external_total": round(float(all_cores["wd_daily_external"].sum()), 0),
        "combined_daily": round(float(all_cores["wd_daily"].sum()), 0),
    }
    print(f"  Segment totals: {segment_totals}")

    print(f"Writing AEM matrix: {bcfg.matrix_path}")
    _write_aem(bcfg.matrix_path, centroid_ids, all_cores, matrix_name=bcfg.matrix_name)
    _mark("write_aem_matrix")

    _ensure_dir(bcfg.output_dir)

    summary_data = {
        "commuting_source": str(
            bcfg.commuting_full_cr_parquet
            if (bcfg.external.enabled and bcfg.external.use_full_cr_dataset and bcfg.commuting_full_cr_parquet.exists())
            else (bcfg.commuting_filtered_parquet if bcfg.commuting_filtered_parquet.exists() else bcfg.commuting_csv)
        ),
        "external_gateway_lookup": str(bcfg.external.gateway_lookup_path),
        "through_gateway_pairs": str(bcfg.external.through_pairs_path),
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
        "external_lookup_places_inbound": len(external_lookup),
        "external_lookup_places_outbound": len(external_lookup_out),
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
    _mark("register_matrix")

    summary_data["timing_breakdown_s"] = marks
    summary_data["elapsed_total_s"] = round(time.perf_counter() - t0, 3)
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- OD build summary ---")
    print(f"  Zones:                         {summary['zones']}")
    print(f"  Rows in:                       {summary['rows_in']}")
    print(f"  Pairs used:                    {summary['pairs_used']}")
    print(f"  Direct matches:                {summary['mapped_direct']}")
    print(f"  Group matches:                 {summary['mapped_group']}")
    print(f"  External lookup matches:       {summary['mapped_external_lookup']}")
    print(f"  External legacy fallback:      {summary['mapped_external_legacy_fallback']}")
    print(f"  Skipped external-external OD:  {summary['skipped_external_external_rows']}")
    print(f"  Missing origin:                {summary['missing_origin']}")
    print(f"  Missing dest:                  {summary['missing_destination']}")

    for core_name in sorted(all_cores.keys()):
        total = round(float(all_cores[core_name].sum()), 1)
        nonzero = int(np.count_nonzero(all_cores[core_name]))
        print(f"  {core_name:30s} total={total:>12.1f}  nonzero={nonzero}")

    print(f"\n  Matrix:  {bcfg.matrix_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Timing:  {marks}")


def assert_build_demand_prerequisites(cfg: dict) -> None:
    """Fail fast before ``load_or_build_od_matrix`` (used by ``run.py`` preflight)."""
    root = Path(cfg["_meta"]["project_root"])
    zoning = cfg.get("zoning") or {}
    zout = Path(zoning.get("output_dir", "outputs/baseline/zones"))
    zones_path = zout / "zones.geojson"
    mapping_path = zout / "zone_centroid_mapping.json"
    if not zones_path.exists():
        raise FileNotFoundError(
            f"build-demand: missing zones export {zones_path}. Run build-zones first."
        )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"build-demand: missing centroid mapping {mapping_path}. Run build-zones first."
        )

    ext = ((cfg.get("demand") or {}).get("sldb") or {}).get("external_processing") or {}
    if not ext.get("enabled", False):
        return

    demand = cfg.get("demand") or {}
    seg_ext_through = (demand.get("segments") or {}).get("external_through") or {}
    _cache_dir = str(Path(cfg.get("datasets", {}).get("cache_dir", "data/cache")))

    def _resolve(p: Any) -> Path:
        path = Path(p) if not isinstance(p, Path) else p
        return path if path.is_absolute() else (root / path).resolve()

    gw = _resolve(
        ext.get("external_gateway_lookup_path", f"{_cache_dir}/external_gateway_lookup.parquet")
    )
    tp_default = ext.get("through_gateway_pairs_path")
    if not tp_default:
        tp_default = seg_ext_through.get(
            "data_driven_pairs_path", f"{_cache_dir}/through_gateway_pairs.parquet"
        )
    tp = _resolve(tp_default)

    if not gw.exists():
        raise FileNotFoundError(
            f"build-demand: missing external gateway lookup {gw}. Run build-supernetwork first."
        )
    if ext.get("use_through_traffic", True) and not tp.exists():
        raise FileNotFoundError(
            f"build-demand: missing through gateway pairs {tp}. Run build-supernetwork first."
        )

    sn = cfg.get("supernetwork") or {}
    sn_out = _resolve(sn.get("output_dir", "outputs/baseline/supernetwork"))
    summary = sn_out / "supernetwork_summary.json"
    if not summary.exists():
        raise FileNotFoundError(
            f"build-demand: missing supernetwork summary {summary}. Run build-supernetwork first."
        )


if __name__ == "__main__":
    load_or_build_od_matrix()
