"""Trip distribution: gravity calibration + IPF adjustment.

Pipeline step ``distribute`` sits between ``build-demand`` and ``assign``:
1. Load the seed OD matrix (from build-demand)
2. Load impedance skims (from previous assignment, or Euclidean fallback)
3. Calibrate gravity model from commuting seed + impedance
4. Build P/A vectors from population
5. Apply gravity model for synthetic "other" trips
6. Run IPF on combined seed to match P/A row/column totals
7. Save adjusted matrix
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae.matrix import AequilibraeMatrix

from sim.io_project import load_config


# ---------------------------------------------------------------------------
# P/A vector building
# ---------------------------------------------------------------------------

def build_pa_vectors(
    zone_ids: np.ndarray,
    population: Dict[int, int],
    trip_rate: float = 2.5,
    car_share: float = 0.50,
    occupancy: float = 1.3,
) -> pd.DataFrame:
    """Build production/attraction vectors from population.

    With no employment data, P_i = A_i (symmetric proxy).
    """
    pa = pd.DataFrame({"zone_id": zone_ids.astype(int)})
    pa["population"] = pa["zone_id"].map(lambda z: population.get(int(z), 0))
    daily = pa["population"] * trip_rate * car_share / max(occupancy, 0.01)
    pa["production"] = daily
    pa["attraction"] = daily
    return pa


# ---------------------------------------------------------------------------
# Impedance loading
# ---------------------------------------------------------------------------

def _load_impedance(
    output_dir: Path,
    zone_ids: np.ndarray,
) -> Optional[np.ndarray]:
    """Load skim matrix from previous assignment as impedance."""
    skim_path = output_dir / "skims.aem"
    if not skim_path.exists():
        return None
    try:
        mat = AequilibraeMatrix()
        mat.load(str(skim_path))
        core_names = list(mat.names)
        if not core_names:
            mat.close()
            return None
        data = mat.matrix[core_names[0]][:, :]
        mat.close()
        if data.shape[0] == len(zone_ids):
            return data.astype(np.float64)
        return None
    except Exception:
        return None


def _euclidean_impedance(zones_gdf: gpd.GeoDataFrame, zone_ids: np.ndarray) -> np.ndarray:
    """Fallback impedance: Euclidean distance between zone centroids."""
    if zones_gdf.crs is not None and zones_gdf.crs.to_epsg() != 5514:
        zg = zones_gdf.to_crs(epsg=5514)
    else:
        zg = zones_gdf

    z2i = {int(z): i for i, z in enumerate(zone_ids)}
    n = len(zone_ids)
    coords = np.zeros((n, 2), dtype=np.float64)
    for _, row in zg.iterrows():
        zid = int(row["zone_id"])
        if zid in z2i:
            pt = row.geometry.representative_point()
            coords[z2i[zid]] = [pt.x, pt.y]

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    return np.sqrt(dx ** 2 + dy ** 2)


# ---------------------------------------------------------------------------
# Gravity calibration & application
# ---------------------------------------------------------------------------

def calibrate_gravity_simple(
    seed: np.ndarray,
    impedance: np.ndarray,
    function: str = "EXPO",
) -> Dict[str, float]:
    """Calibrate a simple deterrence function from seed OD + impedance.

    Uses AequilibraE's GravityCalibration when available, otherwise
    falls back to a least-squares fit of the deterrence parameter.
    """
    try:
        from aequilibrae.distribution import GravityCalibration
        from aequilibrae.matrix import AequilibraeMatrix as AEM

        n = seed.shape[0]
        seed_mat = AEM()
        seed_mat.create_empty(zones=n, matrix_names=["seed"], memory_only=True)
        seed_mat.index[:] = np.arange(1, n + 1, dtype=np.int32)
        seed_mat.matrix["seed"][:, :] = seed
        seed_mat.computational_view(["seed"])

        imp_mat = AEM()
        imp_mat.create_empty(zones=n, matrix_names=["cost"], memory_only=True)
        imp_mat.index[:] = np.arange(1, n + 1, dtype=np.int32)
        imp_mat.matrix["cost"][:, :] = impedance
        imp_mat.computational_view(["cost"])

        gc = GravityCalibration(matrix=seed_mat, impedance=imp_mat, function=function)
        gc.execute()
        params = {"function": function}
        if hasattr(gc, "model") and gc.model is not None:
            for attr in ("alpha", "beta", "gamma"):
                if hasattr(gc.model, attr):
                    params[attr] = float(getattr(gc.model, attr))
        seed_mat.close()
        imp_mat.close()
        return params
    except Exception:
        pass

    # Fallback: fit beta for T_ij ~ exp(-beta * c_ij)
    flat_t = seed.ravel()
    flat_c = impedance.ravel()
    mask = (flat_t > 0) & (flat_c > 0) & np.isfinite(flat_t) & np.isfinite(flat_c)
    if mask.sum() < 10:
        return {"function": "EXPO", "beta": 0.0001}

    log_t = np.log(flat_t[mask])
    c = flat_c[mask]
    beta = -float(np.polyfit(c, log_t, 1)[0])
    beta = max(beta, 1e-6)
    return {"function": "EXPO", "beta": round(beta, 6)}


def apply_gravity(
    productions: np.ndarray,
    attractions: np.ndarray,
    impedance: np.ndarray,
    beta: float = 0.0001,
) -> np.ndarray:
    """Generate synthetic OD from gravity model with exponential deterrence."""
    imp = np.maximum(impedance, 100.0)
    f = np.exp(-beta * imp)
    np.fill_diagonal(f, 0)
    od = productions[:, None] * attractions[None, :] * f
    total_p = productions.sum()
    total_od = od.sum()
    if total_od > 0 and total_p > 0:
        od *= total_p / total_od
    return od


# ---------------------------------------------------------------------------
# IPF (Iterative Proportional Fitting)
# ---------------------------------------------------------------------------

def run_ipf(
    seed: np.ndarray,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    max_iter: int = 200,
    tolerance: float = 0.001,
) -> np.ndarray:
    """IPF to adjust OD row/column totals to production/attraction targets.

    Uses AequilibraE's Ipf when available, otherwise a pure-numpy fallback.
    """
    try:
        from aequilibrae.distribution import Ipf
        from aequilibrae.matrix import AequilibraeMatrix as AEM

        n = seed.shape[0]
        seed_mat = AEM()
        seed_mat.create_empty(zones=n, matrix_names=["seed"], memory_only=True)
        seed_mat.index[:] = np.arange(1, n + 1, dtype=np.int32)
        seed_mat.matrix["seed"][:, :] = seed
        seed_mat.computational_view(["seed"])

        ipf = Ipf(matrix=seed_mat, rows=target_rows, columns=target_cols)
        ipf.max_iterations = max_iter
        ipf.tolerance = tolerance
        ipf.execute()

        result = ipf.output.matrix_view[:, :].copy()
        seed_mat.close()
        ipf.output.close()
        return result
    except Exception:
        pass

    # Numpy fallback (Furness method)
    mat = seed.copy().astype(np.float64)
    mat = np.maximum(mat, 1e-12)
    for _ in range(max_iter):
        row_sums = mat.sum(axis=1)
        row_factors = np.where(row_sums > 0, target_rows / row_sums, 1.0)
        mat *= row_factors[:, None]

        col_sums = mat.sum(axis=0)
        col_factors = np.where(col_sums > 0, target_cols / col_sums, 1.0)
        mat *= col_factors[None, :]

        row_err = np.max(np.abs(mat.sum(axis=1) - target_rows))
        col_err = np.max(np.abs(mat.sum(axis=0) - target_cols))
        if max(row_err, col_err) < tolerance:
            break

    return mat


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_distribution(config_path: str | Path = "config/sim.yaml") -> None:
    """Run the distribution step: gravity calibration + IPF on the seed OD."""
    cfg = load_config(config_path)
    demand_cfg = cfg.get("demand") or {}
    dist_cfg = _get_nested(cfg, ["demand", "distribution"], {})

    if not dist_cfg.get("enabled", True):
        print("Distribution step disabled in config — skipping.")
        return

    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    output_dir.mkdir(parents=True, exist_ok=True)
    core_name = str(cfg.get("calibration", {}).get("core_name", "wd_daily"))

    print("=== TRIP DISTRIBUTION (Gravity + IPF) ===")

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}. Run build-demand first.")

    # Load seed matrix
    mat = AequilibraeMatrix()
    mat.load(str(matrix_path))
    mat.computational_view([core_name])
    seed = mat.matrix[core_name][:, :].copy().astype(np.float64)
    zone_index = mat.index[:].copy()
    n = len(zone_index)
    print(f"  Seed: {n} zones, total={seed.sum():,.0f}")

    # Load zones and population
    zones_gdf = _load_zones_for_distribution(cfg)
    zone_ids = np.array(sorted(zones_gdf["zone_id"].astype(int).unique()), dtype=np.int64)
    population = _load_population(cfg)

    # Load or compute impedance
    impedance = _load_impedance(output_dir, zone_index)
    if impedance is not None:
        print(f"  Impedance: loaded from skims.aem")
    else:
        print(f"  Impedance: Euclidean distance (no skims available)")
        impedance = _euclidean_impedance(zones_gdf, zone_ids)
        if impedance.shape[0] != n:
            print(f"  WARNING: impedance shape mismatch ({impedance.shape[0]} vs {n}), using uniform")
            impedance = np.ones((n, n), dtype=np.float64) * 5000.0

    # Gravity calibration
    deterrence = str(dist_cfg.get("deterrence_function", "EXPO"))
    print(f"  Calibrating gravity model ({deterrence}) ...")
    params = calibrate_gravity_simple(seed, impedance, function=deterrence)
    beta = params.get("beta", 0.0001)
    print(f"  Gravity params: {params}")

    # Build P/A vectors
    pa_trip_rate = float(dist_cfg.get("pa_trip_rate", 2.5))
    pa_car_share = float(dist_cfg.get("pa_car_share", 0.50))
    pa_occupancy = float(dist_cfg.get("pa_occupancy", 1.3))
    pa = build_pa_vectors(zone_ids, population, pa_trip_rate, pa_car_share, pa_occupancy)
    print(f"  P/A: total_production={pa['production'].sum():,.0f}")

    target_rows = pa["production"].values
    target_cols = pa["attraction"].values

    # IPF to match row/column totals
    ipf_max_iter = int(dist_cfg.get("ipf_max_iter", 200))
    ipf_tol = float(dist_cfg.get("ipf_tolerance", 0.001))
    print(f"  Running IPF (max_iter={ipf_max_iter}) ...")
    adjusted = run_ipf(seed, target_rows, target_cols,
                       max_iter=ipf_max_iter, tolerance=ipf_tol)

    # Regularize: blend adjusted matrix with original seed to prevent divergence
    alpha = float(dist_cfg.get("blend_alpha", 0.7))
    blended = alpha * adjusted + (1.0 - alpha) * seed
    print(f"  Blended (alpha={alpha}): total={blended.sum():,.0f} "
          f"(seed={seed.sum():,.0f}, ipf={adjusted.sum():,.0f})")

    # Write back
    mat.matrix[core_name][:, :] = blended
    mat.save()
    mat.close()
    print(f"  Updated matrix: {matrix_path}")

    # Save distribution report
    report = {
        "gravity_params": params,
        "pa_config": {
            "trip_rate": pa_trip_rate,
            "car_share": pa_car_share,
            "occupancy": pa_occupancy,
        },
        "totals": {
            "seed": round(float(seed.sum()), 0),
            "ipf_adjusted": round(float(adjusted.sum()), 0),
            "blended": round(float(blended.sum()), 0),
            "target_productions": round(float(target_rows.sum()), 0),
        },
        "ipf": {"max_iter": ipf_max_iter, "tolerance": ipf_tol},
        "blend_alpha": alpha,
    }
    report_path = output_dir / "distribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Report: {report_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_nested(cfg: Any, path: list, default: Any = None) -> Any:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_zones_for_distribution(cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    zdir = _get_nested(cfg, ["zoning", "output_dir"], "outputs/baseline/zones")
    zones_path = Path(zdir) / "zones.geojson"
    if not zones_path.exists():
        raise FileNotFoundError(f"zones.geojson not found: {zones_path}")
    g = gpd.read_file(zones_path)
    g["zone_id"] = pd.to_numeric(g["zone_id"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["zone_id"]).copy()
    g["zone_id"] = g["zone_id"].astype(int)
    return g


def _load_population(cfg: Dict[str, Any]) -> Dict[int, int]:
    cache_dir = Path(_get_nested(cfg, ["datasets", "cache_dir"], "data/cache"))
    pop_path = cache_dir / "zone_population.parquet"
    if not pop_path.exists():
        return {}
    pop_df = pd.read_parquet(pop_path)
    return dict(zip(pop_df["zone_id"].astype(int), pop_df["population"].astype(int)))
