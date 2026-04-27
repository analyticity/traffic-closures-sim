#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import shutil
import subprocess
from pathlib import Path

import yaml


def set_deep(dct, path, value):
    cur = dct
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def apply_overrides(cfg, overrides):
    out = copy.deepcopy(cfg)
    for path, value in overrides.items():
        set_deep(out, path.split("."), value)
    return out


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_row(run_id, block, calib, valid, rc):
    final = calib.get("final", {})
    pent = valid.get("pentlogram", {})
    return {
        "run_id": run_id,
        "block": block,
        "return_code": rc,
        "calib_iter": final.get("iteration"),
        "calib_r2": final.get("r2"),
        "calib_pct_rmse": final.get("pct_rmse"),
        "calib_geh_lt5_pct": final.get("geh_lt5_pct"),
        "calib_sum_modeled": final.get("sum_modeled"),
        "calib_sum_observed": final.get("sum_observed"),
        "valid_r2": pent.get("r2"),
        "valid_pct_rmse": pent.get("pct_rmse"),
        "valid_geh_lt5_pct": pent.get("geh_lt5_pct"),
        "valid_sum_modeled": pent.get("sum_modeled"),
        "valid_sum_observed": pent.get("sum_observed"),
    }


def main():
    ap = argparse.ArgumentParser(description="Run calibration tuning variants batch")
    ap.add_argument("--base-config", default="config/brno/sim.yaml")
    ap.add_argument("--work-root", default="outputs/tuning")
    ap.add_argument("--outer-iterations", type=int, default=6)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit number of variants for smoke validation")
    args = ap.parse_args()

    root = Path(args.work_root)
    run_root = root / "batch_runs"
    cfg_root = Path(".tmp_tuning_configs")
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    base_cfg_path = Path(args.base_config)
    with base_cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    pol = (((base_cfg.get("experiments") or {}).get("policy") or {}))
    tr = (pol.get("tuning_ranges") or {})
    assignment = tr.get("assignment") or {}
    matching = tr.get("matching") or {}
    scaling = tr.get("scaling") or {}
    gateway = tr.get("gateway") or {}
    ext = tr.get("external_daily_trips") or [10000]
    max_iters = assignment.get("max_iter") or [150, 200, 100]
    rgaps = assignment.get("rgap_target") or [0.002]
    match_buffers = matching.get("match_buffer_m") or [15.0, 35.0]
    dampings = scaling.get("damping") or [0.08, 0.05]
    min_factors = scaling.get("min_factor") or [0.75, 0.9]
    max_factors = scaling.get("max_factor") or [1.1, 1.25]
    gw_dampings = gateway.get("damping") or [0.08]

    variants = [
        ("A1", "assignment", {"calibration.max_iter": int(max_iters[min(1, len(max_iters)-1)])}),
        ("A2", "assignment", {"calibration.max_iter": int(max_iters[-1])}),
        ("A3", "assignment", {"calibration.max_iter": int(max_iters[0]), "calibration.rgap_target": float(rgaps[-1])}),
        ("S1", "scaling", {"calibration.scaling.damping": float(dampings[0])}),
        ("S2", "scaling", {"calibration.scaling.damping": float(dampings[-1])}),
        ("S3", "scaling", {"calibration.scaling.damping": float(dampings[0]), "calibration.scaling.min_factor": float(min_factors[0]), "calibration.scaling.max_factor": float(max_factors[-1])}),
        ("S4", "scaling", {"calibration.scaling.damping": float(dampings[0]), "calibration.scaling.min_factor": float(min_factors[-1]), "calibration.scaling.max_factor": float(max_factors[0])}),
        ("G1", "gateway", {"calibration.gateway_calibration.enabled": False}),
        ("G2", "gateway", {"calibration.gateway_calibration.enabled": True, "calibration.gateway_calibration.damping": float(gw_dampings[0])}),
        ("G3", "gateway", {"calibration.gateway_calibration.enabled": True, "calibration.gateway_calibration.min_factor": float(min_factors[-1]), "calibration.gateway_calibration.max_factor": float(max_factors[0])}),
        ("M1", "matching", {"calibration.match_buffer_m": float(match_buffers[0])}),
        ("M2", "matching", {"calibration.match_buffer_m": float(match_buffers[-1])}),
        ("M3", "matching", {"calibration.match_direction_aware": False}),
        ("D1", "demand", {"demand.segments.external_local.enabled": True, "demand.segments.external_local.total_daily_trips": int(ext[0])}),
        ("D2", "demand", {"demand.segments.external_through.enabled": True, "demand.segments.external_through.total_daily_trips": int(ext[0])}),
        ("D3", "demand", {"demand.segments.external_local.enabled": True, "demand.segments.external_local.total_daily_trips": int(ext[0]), "demand.segments.external_through.enabled": True, "demand.segments.external_through.total_daily_trips": int(ext[0])}),
    ]
    if args.limit and args.limit > 0:
        variants = variants[: int(args.limit)]

    rows = []
    shared_overrides = {"calibration.max_iterations": args.outer_iterations}

    for run_id, block, overrides in variants:
        merged = dict(shared_overrides)
        merged.update(overrides)
        cfg = apply_overrides(base_cfg, merged)
        cfg_path = cfg_root / f"{run_id}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

        print(f"[{run_id}] calibrate...")
        run_kwargs = {}
        if args.quiet:
            run_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        cal = subprocess.run(
            ["python", "run.py", "--config", str(cfg_path), "calibrate"],
            check=False,
            **run_kwargs,
        )
        print(f"[{run_id}] validate...")
        val = subprocess.run(
            ["python", "run.py", "--config", str(cfg_path), "validate"],
            check=False,
            **run_kwargs,
        )
        rc = 0 if (cal.returncode == 0 and val.returncode == 0) else 1

        out_dir = run_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        src_calib = Path("outputs/baseline/demand/calibration_report.json")
        src_valid = Path("outputs/baseline/demand/validation_report.json")
        dst_calib = out_dir / "calibration_report.json"
        dst_valid = out_dir / "validation_report.json"
        if src_calib.exists():
            shutil.copy2(src_calib, dst_calib)
        if src_valid.exists():
            shutil.copy2(src_valid, dst_valid)

        row = metric_row(run_id, block, load_json(dst_calib), load_json(dst_valid), rc)
        rows.append(row)

    csv_path = run_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {csv_path}")
    shutil.rmtree(cfg_root, ignore_errors=True)


if __name__ == "__main__":
    main()
