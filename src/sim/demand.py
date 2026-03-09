"""Build OD matrices from SLDB 2021 commuting data and register in AequilibraE project."""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
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
    """Return (primary_index, stripped_index) mapping norm name → zone_id."""
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
    # "mesto brno" → try just "brno"
    for prefix in ("mesto ", "obec ", "mestys "):
        if nm.startswith(prefix):
            rest = nm[len(prefix):]
            if rest in primary:
                return primary[rest]
    return None


# ---------------------------------------------------------------------------
# Brno group  (dynamic detection from zones + CSV municipalities)
# ---------------------------------------------------------------------------

def _build_brno_group(
    zones_gdf: gpd.GeoDataFrame,
    primary: Dict[str, int],
    stripped: Dict[str, int],
    csv_obec_names: set[str],
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Identify zones that belong to Brno municipality (not standalone towns).

    Logic: take all source_rank=0 zones whose name does NOT match any
    non-"BRNO" municipality from the commuting CSV.  Weight by polygon area.
    """
    brno_norm = _norm_name("BRNO")
    claimed: set[int] = set()
    for obec in csv_obec_names:
        if _norm_name(obec) == brno_norm:
            continue
        zid = _match_zone_id(obec, primary, stripped)
        if zid is not None:
            claimed.add(zid)

    brno_zones: list[Tuple[int, float]] = []
    for _, r in zones_gdf.iterrows():
        zid = int(r["zone_id"])
        if zid in claimed:
            continue
        if int(r.get("source_rank", -1)) == 0:
            area = max(float(r.geometry.area), 1.0)
            brno_zones.append((zid, area))

    if len(brno_zones) < 2:
        return {}

    ids, areas = zip(*brno_zones)
    a = np.array(areas, dtype=np.float64)
    w = a / a.sum()
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
        "missing_origin": 0,
        "missing_destination": 0,
        "skipped_external": 0,
        "pairs_used": 0,
    }

    op_col = "op_obec" if "op_obec" in df.columns else None
    doj_col = "doj_obec" if "doj_obec" in df.columns else None
    if op_col is None or doj_col is None:
        raise RuntimeError(f"Expected columns op_obec/doj_obec, got: {list(df.columns)}")

    def _resolve(name: str) -> Tuple[List[Tuple[int, float]], str]:
        zid = _match_zone_id(name, primary, stripped)
        if zid is not None:
            return [(zid, 1.0)], "direct"
        key = _norm_name(name)
        if key in groups:
            return groups[key], "group"
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

        if o_mode == "direct":
            stats["mapped_direct"] += 1
        else:
            stats["mapped_group"] += 1
        if d_mode == "direct":
            stats["mapped_direct"] += 1
        else:
            stats["mapped_group"] += 1

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
    """Write OD cores to an AequilibraE .aem file.

    ``index_ids`` are the centroid node IDs used as the matrix index
    (must match the graph centroids for assignment to work).
    """
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
    return g


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_or_build_od_matrix(config_path: str | Path = "config/sim.yaml") -> None:
    cfg = load_config(config_path)
    bcfg = _build_cfg(cfg)

    print("=== BUILD OD MATRIX ===")

    # 1) Zones + zone→centroid mapping
    print("Loading zones ...")
    zones_gdf = _load_zones(cfg)
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

    # 2) Name indexes
    primary, stripped = _build_zone_name_index(zones_gdf)
    print(f"  Name index: {len(primary)} primary, {len(stripped)} stripped entries")

    # 3) Read commuting
    print("Reading commuting data ...")
    df_raw = _read_commuting(bcfg)

    # 4) Brno group (needs unique op_obec from data)
    obec_col = "op_obec" if "op_obec" in df_raw.columns else None
    csv_obecs: set[str] = set()
    if obec_col:
        csv_obecs = set(df_raw[obec_col].dropna().astype(str).unique())
    groups = _build_brno_group(zones_gdf, primary, stripped, csv_obecs)
    if groups:
        for gname, members in groups.items():
            print(f"  Group '{gname}': {len(members)} zones")

    # 5) Filter
    df = _filter_commuting(df_raw, bcfg)
    print(f"  {len(df)} rows after filtering (lokalizace + origin region)")

    # 6) Build OD cores
    print("Building OD cores ...")
    cores, summary = _build_od_cores(
        df, zone_ids, bcfg,
        primary=primary, stripped=stripped, groups=groups,
    )

    # 7) Write AEM matrix (indexed by centroid_node_id to match graph)
    print(f"Writing AEM matrix: {bcfg.matrix_path}")
    _write_aem(bcfg.matrix_path, centroid_ids, cores, matrix_name=bcfg.matrix_name)

    # 8) Summary JSON
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
        "groups": {k: {"zones": len(v)} for k, v in groups.items()},
        **summary,
    }
    summary_path = bcfg.output_dir / "od_summary.json"
    summary_path.write_text(
        json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 9) Register in AequilibraE project
    project_dir = cfg.get("project_path")
    if project_dir and Path(project_dir).exists():
        print(f"Registering matrix in AequilibraE project: {project_dir}")
        _register_in_project(Path(project_dir), bcfg.matrix_path)

    # 10) Report
    print("\n--- OD build summary ---")
    print(f"  Zones:           {summary['zones']}")
    print(f"  Rows in:         {summary['rows_in']}")
    print(f"  Pairs used:      {summary['pairs_used']}")
    print(f"  Direct matches:  {summary['mapped_direct']}")
    print(f"  Group matches:   {summary['mapped_group']}")
    print(f"  Missing origin:  {summary['missing_origin']}")
    print(f"  Missing dest:    {summary['missing_destination']}")
    for c, v in summary["cores_sum"].items():
        nz = summary["nonzero_cells"][c]
        print(f"  {c:12s}  total={v:>12.1f}  nonzero_cells={nz}")
    print(f"\n  Matrix:  {bcfg.matrix_path}")
    print(f"  Summary: {summary_path}")
