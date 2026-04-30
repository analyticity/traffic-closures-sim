"""Demand build configuration: frozen dataclasses and YAML parsing."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sim.datasets.paths import resolved_commuting_full_cr_parquet_path
from sim.io_project import as_path, get_nested

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
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


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _resolve_commuting_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    csv_path = get_nested(cfg, ["datasets", "sources", "commuting_sldb2021", "out_path"])
    if csv_path:
        csv_path = as_path(csv_path)
    else:
        csv_path = as_path("data/sources/csu/sldb2021/dojizdka_obce.csv")

    cache_dir = as_path(get_nested(cfg, ["datasets", "cache_dir"], "data/cache"))
    filtered_parquet = as_path(
        get_nested(cfg, ["datasets", "sources", "commuting_sldb2021", "filtered_out_parquet"],
             cache_dir / f"{csv_path.stem}.parquet")
    )
    full_cr_explicit = get_nested(cfg, ["datasets", "sources", "commuting_sldb2021", "full_cr_out_parquet"], None)
    full_cr_parquet = (
        as_path(full_cr_explicit)
        if full_cr_explicit
        else resolved_commuting_full_cr_parquet_path(cfg)
    )
    return csv_path, filtered_parquet, full_cr_parquet


def _build_cfg(cfg: Dict[str, Any]) -> DemandBuildCfg:
    demand = cfg.get("demand") or {}
    csv_path, filtered_parquet, full_cr_parquet = _resolve_commuting_paths(cfg)
    fmt = get_nested(cfg, ["datasets", "sources", "commuting_sldb2021", "format"], {}) or {}

    include_lok = get_nested(demand, ["sldb", "include_lokalizace"], ["0_na_adrese_OP", "1_meziobecni"])
    if not isinstance(include_lok, list):
        include_lok = ["0_na_adrese_OP", "1_meziobecni"]

    origin_filters = get_nested(
        cfg,
        ["datasets", "sources", "commuting_sldb2021", "filter", "origin", "keep_if_any_matches"],
        [],
    )
    if not isinstance(origin_filters, list):
        origin_filters = []

    conv_cfg = get_nested(demand, ["conversion"], {}) or {}
    conv_work = PurposeConv(
        car_share=float(get_nested(conv_cfg, ["work", "car_share"], 0.60)),
        occupancy=float(get_nested(conv_cfg, ["work", "occupancy"], 1.25)),
        trips_per_person=float(get_nested(conv_cfg, ["work", "trips_per_person"], 2.0)),
    )
    conv_school = PurposeConv(
        car_share=float(get_nested(conv_cfg, ["school", "car_share"], 0.25)),
        occupancy=float(get_nested(conv_cfg, ["school", "occupancy"], 1.30)),
        trips_per_person=float(get_nested(conv_cfg, ["school", "trips_per_person"], 2.0)),
    )

    periods = get_nested(demand, ["time_slices", "periods"], ["am", "ip", "pm", "ev"])
    if not isinstance(periods, list) or not periods:
        periods = ["am", "ip", "pm", "ev"]
    periods = [str(p).lower().strip() for p in periods]

    weekday_cfg = get_nested(demand, ["time_slices", "weekday"], {}) or {}
    shares_work = PeriodShares(
        outbound=dict(get_nested(weekday_cfg, ["work", "outbound"], {"am": 0.80, "ip": 0.20})),
        return_=dict(get_nested(weekday_cfg, ["work", "return"], {"pm": 0.80, "ev": 0.20})),
    )
    shares_school = PeriodShares(
        outbound=dict(get_nested(weekday_cfg, ["school", "outbound"], {"am": 0.90, "ip": 0.10})),
        return_=dict(get_nested(weekday_cfg, ["school", "return"], {"pm": 0.70, "ev": 0.30})),
    )

    output_dir = as_path(demand.get("output_dir", "outputs/baseline/demand"))
    matrix_path = as_path(demand.get("matrix_path", "data/demand/od_matrix.aem"))
    matrix_name = str(demand.get("matrix_name", "demand")).strip()

    ext_cfg = get_nested(demand, ["sldb", "external_processing"], {}) or {}
    seg_ext_through = get_nested(demand, ["segments", "external_through"], {}) or {}
    _cache = as_path(get_nested(cfg, ["datasets", "cache_dir"], "data/cache"))

    gateway_lookup_path = as_path(
        ext_cfg.get("external_gateway_lookup_path", str(_cache / "external_gateway_lookup.parquet"))
    )
    through_pairs_path = as_path(
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
        allow_legacy_fallback=bool(get_nested(demand, ["use_gateway_fallback_in_commuting"], False)),
        external_commuting_scale=float(ext_cfg.get("external_commuting_scale", 0.75)),
    )

    return DemandBuildCfg(
        commuting_filtered_parquet=filtered_parquet,
        commuting_full_cr_parquet=full_cr_parquet,
        commuting_csv=csv_path,
        csv_delimiter=str(fmt.get("delimiter", ",")),
        csv_encoding=str(fmt.get("encoding", "utf-8")),
        include_lokalizace=[str(x) for x in include_lok],
        only_internal_pairs=bool(get_nested(demand, ["sldb", "only_internal_pairs"], False)),
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


# ---------------------------------------------------------------------------
# Weight / share normalization
# ---------------------------------------------------------------------------

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
) -> Dict[tuple, float]:
    gateway_names = sorted(gateway_names)
    gateway_set = set(gateway_names)
    weights: Dict[tuple, float] = {}

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
