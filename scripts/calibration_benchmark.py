#!/usr/bin/env python3
"""Calibration benchmark: save seed OD, run calibration variants, compare results.

Usage:
    # Save the current seed matrix
    python scripts/calibration_benchmark.py --config config/brno/sim.yaml save

    # Restore a saved seed matrix
    python scripts/calibration_benchmark.py --config config/brno/sim.yaml restore --snapshot <path>

    # Run full benchmark (saves seed, runs all variants, writes comparison report)
    python scripts/calibration_benchmark.py --config config/brno/sim.yaml benchmark
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sim.io_project import load_config


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _benchmark_dir(cfg: Dict[str, Any]) -> Path:
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    d = output_dir.parent / "calibration_benchmark"
    _ensure_dir(d)
    return d


def save_seed(config_path: str, *, tag: str = "") -> Path:
    """Copy the current OD matrix to a timestamped safe location.

    Returns the path to the saved snapshot.
    """
    cfg = load_config(config_path)
    demand_cfg = cfg.get("demand") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    bdir = _benchmark_dir(cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    snapshot_name = f"seed_od_matrix_{ts}{suffix}.aem"
    snapshot_path = bdir / snapshot_name

    if not matrix_path.exists():
        raise FileNotFoundError(f"OD matrix not found: {matrix_path}")

    # Prefer .aem.orig (the untouched seed) if it exists
    orig = matrix_path.with_suffix(".aem.orig")
    source = orig if orig.exists() else matrix_path
    shutil.copy2(source, snapshot_path)

    # Also snapshot calibration report if present
    report_src = Path(demand_cfg.get("output_dir", "outputs/baseline/demand")) / "calibration_report.json"
    if report_src.exists():
        report_snap = bdir / f"calibration_report_{ts}{suffix}.json"
        shutil.copy2(report_src, report_snap)

    print(f"Seed saved: {snapshot_path}  (from {source})")
    return snapshot_path


def restore_seed(config_path: str, snapshot_path: str) -> None:
    """Restore an OD matrix from a saved snapshot."""
    cfg = load_config(config_path)
    demand_cfg = cfg.get("demand") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))

    snap = Path(snapshot_path)
    if not snap.exists():
        raise FileNotFoundError(f"Snapshot not found: {snap}")

    shutil.copy2(snap, matrix_path)
    # Also update .aem.orig so calibration sees this as the seed
    orig = matrix_path.with_suffix(".aem.orig")
    shutil.copy2(snap, orig)
    print(f"Restored: {snap} -> {matrix_path} (and .aem.orig)")


def _restore_seed_from(cfg: Dict[str, Any], snapshot_path: Path) -> None:
    """Internal helper: restore seed matrix before each variant run."""
    demand_cfg = cfg.get("demand") or {}
    matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
    orig = matrix_path.with_suffix(".aem.orig")
    shutil.copy2(snapshot_path, matrix_path)
    shutil.copy2(snapshot_path, orig)


# ---------------------------------------------------------------------------
# Benchmark variant definitions
# ---------------------------------------------------------------------------

def _define_variants() -> List[Dict[str, Any]]:
    """Return a list of calibration variant configurations."""
    return [
        {
            "name": "odme_default",
            "method": "odme",
            "overrides": {},
        },
        {
            "name": "odme_aggressive",
            "method": "odme",
            "overrides": {
                "odme": {
                    "max_outer_iterations": 40,
                    "gradient_descent_iterations": 8,
                    "convergence_tol": 0.0005,
                    "max_deviation": 6.0,
                },
            },
        },
        {
            "name": "odme_conservative",
            "method": "odme",
            "overrides": {
                "odme": {
                    "max_outer_iterations": 15,
                    "gradient_descent_iterations": 3,
                    "max_deviation": 2.5,
                    "global_residual_damping": 0.15,
                },
            },
        },
        {
            "name": "odme_uniform_weights",
            "method": "odme",
            "overrides": {
                "odme": {"weight_function": "uniform"},
            },
        },
        {
            "name": "odme_inverse_weights",
            "method": "odme",
            "overrides": {
                "odme": {"weight_function": "inverse"},
            },
        },
        {
            "name": "odme_no_gateway",
            "method": "odme",
            "overrides": {
                "gateway_calibration": {"enabled": False},
            },
        },
        {
            "name": "fsm_default",
            "method": "fsm",
            "overrides": {},
        },
        {
            "name": "fsm_high_damping",
            "method": "fsm",
            "overrides": {
                "scaling": {"damping": 0.6},
            },
        },
        {
            "name": "fsm_low_damping",
            "method": "fsm",
            "overrides": {
                "scaling": {"damping": 0.15},
            },
        },
        {
            "name": "entropy_odme",
            "method": "entropy_odme",
            "overrides": {},
        },
        {
            "name": "multistage",
            "method": "multistage",
            "overrides": {},
        },
        {
            "name": "odme_class_correction",
            "method": "odme",
            "overrides": {
                "odme": {
                    "class_residual_enabled": True,
                    "class_residual_damping": 0.15,
                },
            },
        },
        {
            "name": "odme_class_aggressive",
            "method": "odme",
            "overrides": {
                "odme": {
                    "class_residual_enabled": True,
                    "class_residual_damping": 0.25,
                    "max_outer_iterations": 40,
                    "gradient_descent_iterations": 8,
                    "max_deviation": 6.0,
                    "convergence_tol": 0.0005,
                },
            },
        },
    ]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _read_seed_total(snapshot_path: Path, *, core_name: str = "wd_daily") -> float:
    """Read the total demand from a saved seed matrix."""
    try:
        from aequilibrae.matrix import AequilibraeMatrix
        mat = AequilibraeMatrix()
        mat.load(str(snapshot_path))
        names = list(mat.names)
        if not names:
            mat.close()
            return 0.0
        target = core_name if core_name in names else names[0]
        total = float(mat.matrix[target][:, :].sum())
        mat.close()
        return total
    except Exception:
        return 0.0


def _read_result_total(cfg: Dict[str, Any]) -> float:
    """Read the total demand from the current OD matrix after calibration."""
    try:
        from aequilibrae.matrix import AequilibraeMatrix
        demand_cfg = cfg.get("demand") or {}
        matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
        if not matrix_path.is_absolute():
            project_root = cfg.get("_meta", {}).get("project_root", ".")
            matrix_path = Path(project_root) / matrix_path
        core_name = cfg.get("calibration", {}).get("core_name", "wd_daily")
        mat = AequilibraeMatrix()
        mat.load(str(matrix_path))
        names = list(mat.names)
        if not names:
            mat.close()
            return 0.0
        target = core_name if core_name in names else names[0]
        total = float(mat.matrix[target][:, :].sum())
        mat.close()
        return total
    except Exception:
        return 0.0


def _collect_metrics(
    cfg: Dict[str, Any],
    seed_total: float,
    *,
    result_total: float | None = None,
) -> Dict[str, Any]:
    """Read calibration_report.json and extract comparison metrics.

    Parameters
    ----------
    cfg : dict
        Loaded config (used for output_dir / report path).
    seed_total : float
        Total demand of the seed OD matrix.
    result_total : float, optional
        Pre-read total demand after calibration.  When supplied the
        function skips re-reading the matrix (avoids a race with seed
        restore).
    """
    demand_cfg = cfg.get("demand") or {}
    output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
    report_path = output_dir / "calibration_report.json"

    if not report_path.exists():
        return {"error": "no calibration_report.json"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    final = report.get("final", {})

    if result_total is None:
        result_total = _read_result_total(cfg)
    od_change_pct = (
        (result_total - seed_total) / max(seed_total, 1) * 100
        if seed_total > 0 else 0.0
    )

    # Screenline summary (only non-excluded screenlines)
    sl = report.get("screenlines", {})
    sl_ratios = [
        sr.get("ratio", 1.0)
        for sr in sl.values()
        if sr.get("observed_total", 0) > 0
        and sr.get("ratio") is not None
        and not sr.get("excluded_from_benchmark", False)
    ]
    sl_max_dev = max((abs(r - 1.0) * 100 for r in sl_ratios), default=0.0)
    sl_mean_ratio = sum(sl_ratios) / max(len(sl_ratios), 1) if sl_ratios else 0.0

    return {
        "iterations": report.get("iterations", 0),
        "converged": report.get("converged", False),
        "r2": final.get("r2"),
        "slope": final.get("slope"),
        "pct_rmse": final.get("pct_rmse"),
        "bias_pct": final.get("bias_pct"),
        "geh_lt5_pct": final.get("geh_lt5_pct"),
        "geh_lt10_pct": final.get("geh_lt10_pct"),
        "n_counts": final.get("n", final.get("n_count_posts")),
        "sl_max_deviation_pct": round(sl_max_dev, 1),
        "sl_mean_ratio": round(sl_mean_ratio, 3),
        "od_total_after": round(result_total, 0),
        "od_change_from_seed_pct": round(od_change_pct, 2),
    }


def run_benchmark(
    config_path: str,
    *,
    variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run calibration benchmark: save seed, run variants, compare results.

    Parameters
    ----------
    config_path : str
        Path to the city sim.yaml config.
    variants : list of str, optional
        If given, only run these variant names. Otherwise run all.
    """
    from sim.calibration import (
        run_calibration,
        run_odme_calibration,
        run_entropy_odme,
        run_multistage_calibration,
    )

    cfg = load_config(config_path)
    bdir = _benchmark_dir(cfg)
    snapshot = save_seed(config_path, tag="benchmark")
    calib_core = cfg.get("calibration", {}).get("core_name", "wd_daily")
    seed_total = _read_seed_total(snapshot, core_name=calib_core)

    all_variants = _define_variants()
    if variants:
        all_variants = [v for v in all_variants if v["name"] in variants]

    results: Dict[str, Any] = {
        "seed_snapshot": str(snapshot),
        "seed_total": round(seed_total, 0),
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "variants": {},
    }

    method_dispatch = {
        "odme": run_odme_calibration,
        "fsm": run_calibration,
        "entropy_odme": run_entropy_odme,
        "multistage": run_multistage_calibration,
    }

    for variant in all_variants:
        name = variant["name"]
        method = variant["method"]
        overrides = variant["overrides"]

        print(f"\n{'='*60}")
        print(f"  VARIANT: {name}  (method={method})")
        print(f"{'='*60}")

        # Restore seed
        _restore_seed_from(cfg, snapshot)

        # Apply overrides to calibration config
        real_cfg = load_config(config_path)
        calib_cfg = real_cfg.get("calibration") or {}
        merged_calib = _deep_merge(calib_cfg, overrides)
        real_cfg["calibration"] = merged_calib

        # Write temporary config next to the original so city_slug
        # derivation (based on parent directory name) stays correct.
        import yaml
        raw_yaml_path = Path(config_path).expanduser().resolve()
        tmp_cfg_path = raw_yaml_path.parent / f"_tmp_benchmark_{name}.yaml"
        raw = yaml.safe_load(raw_yaml_path.read_text(encoding="utf-8")) or {}
        raw_calib = raw.get("calibration") or {}
        raw["calibration"] = _deep_merge(raw_calib, overrides)
        raw["calibration"]["method"] = method
        tmp_cfg_path.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

        runner = method_dispatch.get(method)
        if runner is None:
            results["variants"][name] = {"error": f"unknown method: {method}"}
            continue

        t0 = time.monotonic()
        try:
            runner(str(tmp_cfg_path))
            elapsed = time.monotonic() - t0
            # Read matrix total BEFORE the next iteration restores the seed
            post_total = _read_result_total(real_cfg)
            metrics = _collect_metrics(real_cfg, seed_total, result_total=post_total)
            metrics["runtime_s"] = round(elapsed, 1)
            metrics["status"] = "ok"
        except Exception as exc:
            import traceback
            elapsed = time.monotonic() - t0
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            metrics = {
                "status": "error",
                "error": str(exc),
                "runtime_s": round(elapsed, 1),
            }

        results["variants"][name] = metrics
        print(f"  -> {name}: {metrics.get('status', '?')} in {elapsed:.0f}s")

        # Save per-variant calibration report for post-hoc analysis
        demand_cfg_r = real_cfg.get("demand") or {}
        variant_report_src = Path(
            demand_cfg_r.get("output_dir", "outputs/baseline/demand")
        ) / "calibration_report.json"
        if variant_report_src.exists():
            variant_report_dst = bdir / f"calibration_report_{name}.json"
            shutil.copy2(variant_report_src, variant_report_dst)

        # Cleanup temp config
        tmp_cfg_path.unlink(missing_ok=True)

    # Restore seed back to original state
    _restore_seed_from(cfg, snapshot)

    # Write comparison report
    report_path = bdir / "comparison_report.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nComparison report: {report_path}")

    # Print summary table
    _print_summary(results)
    return results


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a text summary table of benchmark results."""
    variants = results.get("variants", {})
    if not variants:
        return

    header = f"{'Variant':<25s} {'R²':>6s} {'Slope':>6s} {'%RMSE':>7s} {'Bias%':>7s} {'GEH<5%':>7s} {'SL_dev%':>7s} {'OD_chg%':>8s} {'Time':>6s} {'Status':<8s}"
    print(f"\n{'='*90}")
    print(header)
    print(f"{'-'*90}")
    for name, m in variants.items():
        if m.get("status") == "error":
            print(f"{name:<25s} {'—':>6s} {'—':>6s} {'—':>7s} {'—':>7s} {'—':>7s} {'—':>7s} {'—':>8s} {m.get('runtime_s', 0):>5.0f}s {'ERROR':<8s}")
            continue
        print(
            f"{name:<25s} "
            f"{m.get('r2', 0) or 0:>6.3f} "
            f"{m.get('slope', 0) or 0:>6.3f} "
            f"{m.get('pct_rmse', 0) or 0:>7.1f} "
            f"{m.get('bias_pct', 0) or 0:>7.1f} "
            f"{m.get('geh_lt5_pct', 0) or 0:>7.1f} "
            f"{m.get('sl_max_deviation_pct', 0):>7.1f} "
            f"{m.get('od_change_from_seed_pct', 0):>8.1f} "
            f"{m.get('runtime_s', 0):>5.0f}s "
            f"{'OK' if m.get('converged') else 'noconv':<8s}"
        )
    print(f"{'='*90}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration benchmark tool")
    parser.add_argument("--config", default="config/brno/sim.yaml", help="City config YAML")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("save", help="Save current seed OD matrix")
    p_restore = sub.add_parser("restore", help="Restore a saved seed")
    p_restore.add_argument("--snapshot", required=True, help="Path to saved .aem snapshot")

    p_bench = sub.add_parser("benchmark", help="Run full calibration benchmark")
    p_bench.add_argument("--variants", nargs="*", help="Only run these variant names")

    args = parser.parse_args()

    if args.command == "save":
        save_seed(args.config)
    elif args.command == "restore":
        restore_seed(args.config, args.snapshot)
    elif args.command == "benchmark":
        run_benchmark(args.config, variants=args.variants)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
