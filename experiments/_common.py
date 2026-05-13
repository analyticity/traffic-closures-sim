"""Shared utilities for thesis experiment scripts.

All experiment scripts live under ``experiments/`` and use these helpers
to load pipeline artifacts, run scenario assignments, and persist results.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

# Ensure src/ is importable when running from repo root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from sim.io_project import load_config
from sim._metrics import aggregate_daily_volumes

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config/brno/sim.yaml"
EXPERIMENTS_OUTPUT = Path("outputs/experiments")


# ---------------------------------------------------------------------------
# Config & path helpers
# ---------------------------------------------------------------------------

def city_from_config(config_path: str) -> str:
    """Derive a short city slug from the config file path (e.g. 'most')."""
    parts = Path(config_path).parts
    # Expected layout: config/<city>/sim.yaml
    try:
        idx = parts.index("config")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return Path(config_path).stem


def city_display_name(cfg: Dict[str, Any]) -> str:
    """Human-readable city name from the loaded config (e.g. 'Most')."""
    place = cfg.get("osm", {}).get("place_name", "")
    if place:
        return place.split(",")[0].strip()
    return ""


def _parse_experiment_args(name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Experiment: {name}")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to city sim.yaml config (default: %(default)s)",
    )
    return parser.parse_args()


def init_experiment(
    name: str,
    config_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    """Load config and create output directory for *name*. Returns (cfg, out_dir).

    When *config_path* is not supplied, ``--config`` is read from ``sys.argv``,
    falling back to ``DEFAULT_CONFIG``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    if config_path is None:
        args = _parse_experiment_args(name)
        config_path = args.config

    cfg = load_config(config_path)
    city = city_from_config(config_path)
    out_dir = EXPERIMENTS_OUTPUT / city / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, out_dir


def _network_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("network", {}).get("output_dir", "outputs/baseline/network"))


def _demand_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("demand", {}).get("output_dir", "outputs/baseline/demand"))


def _cache_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))


# ---------------------------------------------------------------------------
# Load pipeline artifacts
# ---------------------------------------------------------------------------

def load_network_links(cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the network links GeoDataFrame (GPKG preferred, falls back to parquet)."""
    nd = _network_dir(cfg)
    for ext in ("gpkg", "parquet", "geojson"):
        p = nd / f"network_links.{ext}"
        if p.exists():
            if ext == "parquet":
                return gpd.GeoDataFrame(pd.read_parquet(p))
            return gpd.read_file(p)
    raise FileNotFoundError(f"No network_links found in {nd}")


def load_assignment_results(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load assignment_results.parquet and ensure daily volume columns exist."""
    p = _demand_dir(cfg) / "assignment_results.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Assignment results not found: {p}")
    df = pd.read_parquet(p)
    aggregate_daily_volumes(df)
    return df


def load_baseline_links(cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    """Merge network geometry with baseline assignment volumes."""
    links = load_network_links(cfg)
    assign = load_assignment_results(cfg)

    vol_cols = [c for c in assign.columns if c != "link_id" and c not in links.columns]
    gdf = links.merge(assign[["link_id"] + vol_cols], on="link_id", how="left")
    aggregate_daily_volumes(gdf)
    return gdf


def load_segment_map(cfg: Dict[str, Any], links: Optional[pd.DataFrame] = None) -> Dict[int, int]:
    """segment_id -> link_id mapping via road_segments.parquet."""
    from sim.datasets.segments_fetch import build_segment_to_link_map

    seg_path = _cache_dir(cfg) / "road_segments.parquet"
    if not seg_path.exists():
        logger.warning("road_segments.parquet not found at %s", seg_path)
        return {}
    if links is None:
        links = load_network_links(cfg)
    return build_segment_to_link_map(seg_path, links)


def load_jams_stats(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load jams_segment_stats.parquet."""
    p = _cache_dir(cfg) / "jams_segment_stats.parquet"
    if not p.exists():
        raise FileNotFoundError(f"jams_segment_stats.parquet not found: {p}")
    return pd.read_parquet(p)


def load_closures(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load closures.parquet."""
    p = _cache_dir(cfg) / "closures.parquet"
    if not p.exists():
        raise FileNotFoundError(f"closures.parquet not found: {p}")
    return pd.read_parquet(p)


# Default metric CRS for Czech Republic (S-JTSK / Křovák East-North)
_METRIC_CRS = "EPSG:5514"


def ensure_metric_links(links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return *links* in a metric CRS suitable for distance calculations.

    If the GeoDataFrame is already in a projected (metric) CRS, returns it
    unchanged.  Otherwise reprojects to EPSG:5514 (S-JTSK).
    """
    if links.crs is None or links.crs.is_geographic:
        return links.to_crs(_METRIC_CRS)
    return links


def match_links_near_point(
    lat: float,
    lon: float,
    links: gpd.GeoDataFrame,
    max_dist_m: float = 200,
    *,
    max_links: int = 50,
) -> gpd.GeoDataFrame:
    """Find network links within *max_dist_m* meters of a WGS84 point.

    Returns a subset GeoDataFrame (in the original links CRS).
    Falls back to the single nearest link if none found within the buffer.
    Caps results at *max_links* to prevent accidental full-network selection.
    """
    from shapely.geometry import Point

    metric_links = ensure_metric_links(links)
    pt = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)], crs="EPSG:4326"
    ).to_crs(metric_links.crs).geometry.iloc[0]

    dists = metric_links.geometry.distance(pt)
    nearby_mask = dists < max_dist_m

    if not nearby_mask.any():
        nearest_idx = dists.idxmin()
        return links.loc[[nearest_idx]]

    nearby_idx = dists[nearby_mask].nsmallest(max_links).index
    return links.loc[nearby_idx]


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_vht(df: pd.DataFrame, vol_col: str = "wd_daily_tot",
                time_col: str = "Congested_Time_Max") -> float:
    """Vehicle Hours Traveled = sum(volume * congested_time_hours)."""
    vol = df[vol_col].fillna(0) if vol_col in df.columns else 0
    tt = df[time_col].fillna(0) if time_col in df.columns else 0
    return float(np.sum(vol * tt / 3600.0))


def compute_vkt(df: pd.DataFrame, vol_col: str = "wd_daily_tot") -> float:
    """Vehicle Kilometers Traveled = sum(volume * distance_km)."""
    vol = df[vol_col].fillna(0) if vol_col in df.columns else 0
    dist = df["distance"].fillna(0) / 1000.0 if "distance" in df.columns else 0
    return float(np.sum(vol * dist))


def compute_scenario_kpis(
    baseline: pd.DataFrame,
    scenario: pd.DataFrame,
) -> Dict[str, float]:
    """Compute delta KPIs between baseline and scenario assignment results."""
    b_vht = compute_vht(baseline)
    s_vht = compute_vht(scenario)
    b_vkt = compute_vkt(baseline)
    if "distance" not in scenario.columns and "distance" in baseline.columns:
        scenario_with_dist = scenario.merge(
            baseline[["link_id", "distance"]].drop_duplicates("link_id"),
            on="link_id", how="left",
        )
        s_vkt = compute_vkt(scenario_with_dist)
    else:
        s_vkt = compute_vkt(scenario)

    b_overloaded = int((baseline.get("VOC_max", pd.Series(dtype=float)).fillna(0) > 1.0).sum())
    s_overloaded = int((scenario.get("VOC_max", pd.Series(dtype=float)).fillna(0) > 1.0).sum())

    return {
        "baseline_vht": round(b_vht, 1),
        "scenario_vht": round(s_vht, 1),
        "delta_vht": round(s_vht - b_vht, 1),
        "delta_vht_pct": round((s_vht - b_vht) / max(b_vht, 1) * 100, 2),
        "baseline_vkt": round(b_vkt, 1),
        "scenario_vkt": round(s_vkt, 1),
        "delta_vkt": round(s_vkt - b_vkt, 1),
        "baseline_overloaded": b_overloaded,
        "scenario_overloaded": s_overloaded,
        "delta_overloaded": s_overloaded - b_overloaded,
    }


# ---------------------------------------------------------------------------
# Scenario assignment runner
# ---------------------------------------------------------------------------

def run_scenario_assignment(
    cfg: Dict[str, Any],
    scenario_links: List[Dict[str, Any]],
    *,
    algorithm: Optional[str] = None,
    max_iter: Optional[int] = None,
    rgap: Optional[float] = None,
) -> pd.DataFrame:
    """Run a single scenario assignment and return the results DataFrame.

    Opens project/matrix, builds graph, applies scenario modifications
    in-memory, runs BFW, returns enriched results. Does NOT write to disk
    or modify the on-disk project.
    """
    from aequilibrae import Project
    from aequilibrae.matrix import AequilibraeMatrix
    from sim.assignment import build_graph, execute_assignment, fix_node_ids
    from sim.scenarios.engine import apply_scenario_to_graph

    calib_cfg = cfg.get("calibration") or {}
    assign_cfg = cfg.get("assignment") or {}
    demand_cfg = cfg.get("demand") or {}

    _algorithm = algorithm or str(calib_cfg.get("algorithm", "bfw"))
    _max_iter = max_iter or int(assign_cfg.get("scenario_max_iter", calib_cfg.get("max_iter", 100)))
    _rgap = rgap or float(assign_cfg.get("scenario_rgap", calib_cfg.get("rgap_target", 0.001)))
    core_name = str(calib_cfg.get("core_name", "wd_daily"))
    bpr_params = dict(assign_cfg.get("bpr") or {}) or None

    mc_cfg = assign_cfg.get("multi_class") or {}
    multi_classes = (
        list(mc_cfg["classes"])
        if mc_cfg.get("enabled") and "classes" in mc_cfg
        else None
    )

    gc_cfg = assign_cfg.get("generalized_cost") or {}
    gc_enabled = bool(gc_cfg.get("enabled", False))
    gc_field = str(gc_cfg["fixed_cost_field"]) if gc_enabled and "fixed_cost_field" in gc_cfg else None
    gc_mult = float(gc_cfg.get("fixed_cost_multiplier", 0.0)) if gc_enabled else 0.0
    gc_vot = float(gc_cfg.get("vot", 1.0))

    project_path = Path(cfg["project_path"])
    fix_node_ids(project_path)

    mat = AequilibraeMatrix()
    mat.load(str(demand_cfg.get("matrix_path")))
    mat.computational_view([core_name])

    project = Project()
    project.open(str(project_path))
    try:
        graph = build_graph(project, mat, bpr_parameters=bpr_params, assignment_cfg=assign_cfg)
        if scenario_links:
            apply_scenario_to_graph(graph, scenario_links)

        df, _skims, _sl, _conv = execute_assignment(
            project, mat,
            algorithm=_algorithm,
            max_iter=_max_iter,
            rgap_target=_rgap,
            bpr_parameters=bpr_params,
            multi_class=multi_classes,
            fixed_cost_field=gc_field,
            fixed_cost_multiplier=gc_mult,
            vot=gc_vot,
            graph=graph,
        )
    finally:
        project.close()
        mat.close()

    aggregate_daily_volumes(df)
    return df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Saved %s", path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s", path)


def save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = gdf.copy()
    if out.crs and out.crs.to_epsg() != 4326:
        out = out.to_crs(epsg=4326)
    out.to_file(path, driver="GeoJSON")
    logger.info("Saved %s", path)


def save_figure(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Experiment metric helpers (plausibility / scenarios / congestion)
# ---------------------------------------------------------------------------

def compute_factor_of_2_pct(
    modeled: np.ndarray,
    observed: np.ndarray,
) -> Dict[str, Any]:
    """Share of count stations where modeled/observed lies in [0.5, 2].

    Suitable for **daily** OD models with sparse CSD coverage (no GEH gates).
    """
    m = np.asarray(modeled, dtype=float)
    o = np.asarray(observed, dtype=float)
    valid = np.isfinite(m) & np.isfinite(o) & (o > 0)
    if not valid.any():
        return {"n": 0, "pct_in_factor_of_2": None, "median_ratio": None}
    m_v, o_v = m[valid], o[valid]
    ratio = m_v / o_v
    in_band = (ratio >= 0.5) & (ratio <= 2.0)
    return {
        "n": int(valid.sum()),
        "pct_in_factor_of_2": round(float(np.mean(in_band) * 100), 2),
        "median_ratio": round(float(np.median(ratio)), 4),
        "mean_ratio": round(float(np.mean(ratio)), 4),
    }


def top_n_links_by_delta(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    *,
    vol_col: str = "wd_daily_tot",
    n: int = 10,
    exclude_link_ids: Optional[Set[int]] = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Largest absolute volume changes between baseline and scenario (by *vol_col*)."""
    b = baseline_df[["link_id", vol_col]].copy() if "link_id" in baseline_df.columns else pd.DataFrame()
    s = scenario_df[["link_id", vol_col]].copy()
    if b.empty or vol_col not in b.columns or vol_col not in s.columns:
        return pd.DataFrame()
    merged = b.merge(s, on="link_id", how="inner", suffixes=("_base", "_scen"))
    merged["delta_vol"] = merged[f"{vol_col}_scen"].fillna(0) - merged[f"{vol_col}_base"].fillna(0)
    merged["abs_delta_vol"] = merged["delta_vol"].abs()
    if exclude_link_ids:
        merged = merged[~merged["link_id"].astype(int).isin(exclude_link_ids)]
    merged = merged.sort_values("abs_delta_vol", ascending=ascending)
    return merged.head(n).reset_index(drop=True)


def compute_quartile_comparison(
    df: pd.DataFrame,
    vc_col: str,
    jam_col: str,
    *,
    n_quartiles: int = 4,
) -> Dict[str, Any]:
    """Summarise *jam_col* by V/C quartiles + Q4/Q1 median ratio and Mann–Whitney U.

    Returns JSON-serialisable dict for experiment summaries.
    """
    from scipy import stats as sp_stats

    d = df[[vc_col, jam_col]].dropna().copy()
    d = d[(d[vc_col] >= 0) & (d[jam_col] >= 0)]
    if len(d) < n_quartiles * 5:
        return {"n_links": len(d), "error": "too_few_rows_for_quartiles"}

    try:
        d["vc_quartile"] = pd.qcut(d[vc_col], n_quartiles, labels=False, duplicates="drop")
    except ValueError:
        return {"n_links": len(d), "error": "qcut_failed"}

    medians = d.groupby("vc_quartile", observed=True)[jam_col].median().to_dict()
    medians_str = {f"Q{int(k) + 1}": float(v) for k, v in sorted(medians.items())}

    q1 = d[d["vc_quartile"] == 0][jam_col]
    q4 = d[d["vc_quartile"] == n_quartiles - 1][jam_col]
    ratio = None
    mw_p = None
    if len(q1) > 2 and len(q4) > 2:
        m1, m4 = float(q1.median()), float(q4.median())
        ratio = round(m4 / max(m1, 1e-9), 4) if m1 > 0 else None
        try:
            _, mw_p = sp_stats.mannwhitneyu(q4, q1, alternative="greater")
            mw_p = float(mw_p)
        except ValueError:
            mw_p = None

    return {
        "n_links": len(d),
        "quartile_medians": medians_str,
        "median_ratio_Q4_over_Q1": ratio,
        "mannwhitney_greater_p": mw_p,
    }
