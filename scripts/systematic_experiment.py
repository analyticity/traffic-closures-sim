#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class RunSpec:
    run_id: str
    phase: str
    block: str
    short_iters: int
    overrides: Dict[str, object]
    tags: str


def set_deep(dct: dict, path: List[str], value):
    cur = dct
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def apply_overrides(cfg: dict, overrides: Dict[str, object]) -> dict:
    out = copy.deepcopy(cfg)
    for path, value in overrides.items():
        set_deep(out, path.split("."), value)
    return out


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_command(cmd: List[str], quiet: bool) -> subprocess.CompletedProcess:
    kwargs = {"check": False}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.run(cmd, **kwargs)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def tail_from_offset(path: Path, offset: int) -> str:
    txt = read_text(path)
    if offset < 0:
        offset = 0
    if offset > len(txt):
        return ""
    return txt[offset:]


def classify_failure(
    calibrate_exit_code: int,
    validate_exit_code: int,
    has_traceback: bool,
    has_artifacts: bool,
    rgap_not_reached: bool,
) -> str:
    if has_traceback or not has_artifacts:
        return "technical_fail"
    if calibrate_exit_code != 0 or validate_exit_code != 0:
        return "model_fail_only"
    if rgap_not_reached:
        return "model_fail_only"
    return "ok"


def collect_metrics(calib: dict, valid: dict) -> Dict[str, object]:
    final = calib.get("final", {})
    pent = valid.get("pentlogram", {})
    return {
        "calib_iteration": final.get("iteration"),
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


def metrics_score(row: Dict[str, object]) -> float:
    r2 = row.get("valid_r2")
    pr = row.get("valid_pct_rmse")
    geh = row.get("valid_geh_lt5_pct")
    if r2 is None or pr is None or geh is None:
        return -10**9
    # Higher is better: r2 and GEH<5 up, pct_rmse down
    return (r2 * 100.0) + (geh * 2.0) - pr


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)


def _policy(base_cfg: dict) -> dict:
    return (((base_cfg.get("experiments") or {}).get("policy") or {}))


def make_screening_specs(base_cfg: dict) -> List[RunSpec]:
    pol = _policy(base_cfg)
    tr = pol.get("tuning_ranges") or {}
    assign = tr.get("assignment") or {}
    matching = tr.get("matching") or {}
    scaling = tr.get("scaling") or {}
    gateway = tr.get("gateway") or {}
    ext_trips = tr.get("external_daily_trips") or [10000, 20000, 30000]
    max_iters = assign.get("max_iter") or [100, 150, 200]
    rgaps = assign.get("rgap_target") or [0.001, 0.002]
    match_buffers = matching.get("match_buffer_m") or [10.0, 25.0, 45.0]
    dampings = scaling.get("damping") or [0.05, 0.08, 0.15]
    min_factors = scaling.get("min_factor") or [0.75, 0.9]
    max_factors = scaling.get("max_factor") or [1.1, 1.25]
    gw_dampings = gateway.get("damping") or [0.08, 0.15]

    specs = []
    # Baseline reproducibility (Block 0)
    specs += [
        RunSpec("B0_R1", "baseline", "0", 4, {}, "baseline,repro"),
        RunSpec("B0_R2", "baseline", "0", 4, {}, "baseline,repro"),
    ]
    # Block 1: assignment
    specs += [
        RunSpec("A1", "screening", "1", 4, {"calibration.max_iter": int(max_iters[0]), "calibration.rgap_target": float(rgaps[0])}, "assignment"),
        RunSpec("A2", "screening", "1", 4, {"calibration.max_iter": int(max_iters[min(1, len(max_iters)-1)]), "calibration.rgap_target": float(rgaps[0])}, "assignment"),
        RunSpec("A3", "screening", "1", 4, {"calibration.max_iter": int(max_iters[-1]), "calibration.rgap_target": float(rgaps[0])}, "assignment"),
        RunSpec("A4", "screening", "1", 4, {"calibration.max_iter": int(max_iters[0]), "calibration.rgap_target": float(rgaps[-1])}, "assignment"),
    ]
    # Block 2: matching
    specs += [
        RunSpec("M1", "screening", "2", 4, {"calibration.match_buffer_m": float(match_buffers[0]), "calibration.match_direction_aware": True, "calibration.match_conflict_resolution": "nearest"}, "matching"),
        RunSpec("M2", "screening", "2", 4, {"calibration.match_buffer_m": float(match_buffers[min(1, len(match_buffers)-1)]), "calibration.match_direction_aware": True, "calibration.match_conflict_resolution": "nearest"}, "matching"),
        RunSpec("M3", "screening", "2", 4, {"calibration.match_buffer_m": float(match_buffers[-1]), "calibration.match_direction_aware": True, "calibration.match_conflict_resolution": "nearest"}, "matching"),
        RunSpec("M4", "screening", "2", 4, {"calibration.match_buffer_m": float(match_buffers[min(1, len(match_buffers)-1)]), "calibration.match_direction_aware": False, "calibration.match_conflict_resolution": "nearest"}, "matching"),
    ]
    # Block 3: scaling vs gateway (2x2)
    specs += [
        RunSpec("GSG1", "screening", "3", 4, {"calibration.gateway_calibration.enabled": False, "calibration.scaling.damping": float(dampings[0]), "calibration.scaling.min_factor": float(min_factors[-1]), "calibration.scaling.max_factor": float(max_factors[0])}, "scale_gateway"),
        RunSpec("GSG2", "screening", "3", 4, {"calibration.gateway_calibration.enabled": False, "calibration.scaling.damping": float(dampings[-1]), "calibration.scaling.min_factor": float(min_factors[0]), "calibration.scaling.max_factor": float(max_factors[-1])}, "scale_gateway"),
        RunSpec("GSG3", "screening", "3", 4, {"calibration.gateway_calibration.enabled": True, "calibration.gateway_calibration.damping": float(gw_dampings[0]), "calibration.scaling.damping": float(dampings[0]), "calibration.scaling.min_factor": float(min_factors[-1]), "calibration.scaling.max_factor": float(max_factors[0])}, "scale_gateway"),
        RunSpec("GSG4", "screening", "3", 4, {"calibration.gateway_calibration.enabled": True, "calibration.gateway_calibration.damping": float(gw_dampings[-1]), "calibration.scaling.damping": float(dampings[-1]), "calibration.scaling.min_factor": float(min_factors[0]), "calibration.scaling.max_factor": float(max_factors[-1])}, "scale_gateway"),
        RunSpec("GSG5", "screening", "3", 4, {"calibration.gateway_calibration.enabled": True, "calibration.gateway_calibration.damping": float(gw_dampings[0]), "calibration.scaling.damping": float(dampings[min(1, len(dampings)-1)]), "calibration.scaling.min_factor": float(min_factors[-1]), "calibration.scaling.max_factor": float(max_factors[0])}, "scale_gateway"),
        RunSpec("GSG6", "screening", "3", 4, {"calibration.gateway_calibration.enabled": False, "calibration.scaling.damping": float(dampings[min(1, len(dampings)-1)]), "calibration.scaling.min_factor": float(min_factors[-1]), "calibration.scaling.max_factor": float(max_factors[0])}, "scale_gateway"),
    ]
    # Block 4: external demand
    specs += [
        RunSpec("D1", "screening", "4", 4, {"demand.segments.external_local.enabled": False, "demand.segments.external_through.enabled": False}, "external"),
        RunSpec("D2", "screening", "4", 4, {"demand.segments.external_local.enabled": True, "demand.segments.external_local.total_daily_trips": int(ext_trips[0]), "demand.segments.external_through.enabled": False}, "external"),
        RunSpec("D3", "screening", "4", 4, {"demand.segments.external_local.enabled": False, "demand.segments.external_through.enabled": True, "demand.segments.external_through.total_daily_trips": int(ext_trips[0])}, "external"),
        RunSpec("D4", "screening", "4", 4, {"demand.segments.external_local.enabled": True, "demand.segments.external_local.total_daily_trips": int(ext_trips[min(1, len(ext_trips)-1)]), "demand.segments.external_through.enabled": True, "demand.segments.external_through.total_daily_trips": int(ext_trips[min(1, len(ext_trips)-1)])}, "external"),
        RunSpec("D5", "screening", "4", 4, {"demand.segments.external_local.enabled": True, "demand.segments.external_local.total_daily_trips": int(ext_trips[-1]), "demand.segments.external_through.enabled": False}, "external"),
        RunSpec("D6", "screening", "4", 4, {"demand.segments.external_local.enabled": False, "demand.segments.external_through.enabled": True, "demand.segments.external_through.total_daily_trips": int(ext_trips[-1])}, "external"),
    ]
    return specs


def choose_top(rows: List[Dict[str, object]], n: int) -> List[str]:
    screened = [r for r in rows if r["phase"] == "screening" and r.get("failure_class") != "technical_fail"]
    screened.sort(key=metrics_score, reverse=True)
    top = []
    for row in screened:
        if row["run_id"] not in top:
            top.append(row["run_id"])
        if len(top) >= n:
            break
    return top


def main():
    ap = argparse.ArgumentParser(description="Systematic mismatch experiment runner")
    ap.add_argument("--base-config", default="config/brno/sim.yaml")
    ap.add_argument("--out-root", default="outputs/tuning/systematic_experiment")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--confirmatory-runs", type=int, default=3)
    ap.add_argument("--full-iterations", type=int, default=15)
    ap.add_argument("--aeq-log", default="project/brno_aeq/aequilibrae.log")
    ap.add_argument("--smoke-limit", type=int, default=0, help="limit number of screening specs for smoke validation")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    cfg_root = out_root / "configs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    specs = make_screening_specs(base_cfg)
    if args.smoke_limit and args.smoke_limit > 0:
        specs = specs[: int(args.smoke_limit)]

    matrix_rows = []
    result_rows = []
    baseline_rows = []

    aeq_log = Path(args.aeq_log)
    for spec in specs:
        overrides = dict(spec.overrides)
        overrides["calibration.max_iterations"] = spec.short_iters
        cfg = apply_overrides(base_cfg, overrides)
        cfg_path = cfg_root / f"{spec.run_id}.yaml"
        write_yaml(cfg_path, cfg)

        matrix_rows.append(
            {
                "run_id": spec.run_id,
                "phase": spec.phase,
                "block": spec.block,
                "tags": spec.tags,
                "short_iters": spec.short_iters,
                "overrides_json": json.dumps(overrides, ensure_ascii=True, sort_keys=True),
                "config_path": str(cfg_path),
            }
        )

        run_dir = runs_root / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        before_len = len(read_text(aeq_log))
        cal = run_command(["python", "run.py", "--config", str(cfg_path), "calibrate"], args.quiet)
        val = run_command(["python", "run.py", "--config", str(cfg_path), "validate"], args.quiet)
        log_delta = tail_from_offset(aeq_log, before_len)
        rc = 0 if cal.returncode == 0 and val.returncode == 0 else 1

        src_cal = Path("outputs/baseline/demand/calibration_report.json")
        src_val = Path("outputs/baseline/demand/validation_report.json")
        dst_cal = run_dir / "calibration_report.json"
        dst_val = run_dir / "validation_report.json"
        if src_cal.exists():
            shutil.copy2(src_cal, dst_cal)
        if src_val.exists():
            shutil.copy2(src_val, dst_val)
        metrics = collect_metrics(load_json(dst_cal), load_json(dst_val))
        has_traceback = "Traceback" in log_delta
        rgap_not_reached = "Desired RGap" in log_delta and "NOT reached" in log_delta
        has_artifacts = dst_cal.exists() and dst_val.exists()
        failure_class = classify_failure(
            cal.returncode,
            val.returncode,
            has_traceback,
            has_artifacts,
            rgap_not_reached,
        )
        row = {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "block": spec.block,
            "tags": spec.tags,
            "return_code": rc,
            "calibrate_exit_code": cal.returncode,
            "validate_exit_code": val.returncode,
            "has_traceback": has_traceback,
            "rgap_not_reached": rgap_not_reached,
            "has_artifacts": has_artifacts,
            "failure_class": failure_class,
            **metrics,
        }
        result_rows.append(row)
        if spec.phase == "baseline":
            baseline_rows.append(row)

    # Block 5 cross-check from supernetwork summary
    super_summary = load_json(Path("outputs/baseline/supernetwork/supernetwork_summary.json"))
    super_out = {
        "gateway_count": super_summary.get("gateway_count"),
        "through_pairs_count": super_summary.get("through_pairs_count"),
        "through_pairs_total_vehicles_daily": super_summary.get("through_pairs_total_vehicles_daily"),
        "gateway_inbound_vehicles_daily": super_summary.get("gateway_inbound_vehicles_daily"),
        "gateway_outbound_vehicles_daily": super_summary.get("gateway_outbound_vehicles_daily"),
    }
    (out_root / "supernetwork_crosscheck.json").write_text(
        json.dumps(super_out, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # Confirmatory full runs for top screening candidates
    top = choose_top(result_rows, args.confirmatory_runs)
    confirm_rows = []
    for idx, run_id in enumerate(top, start=1):
        src_cfg = cfg_root / f"{run_id}.yaml"
        cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
        cfg["calibration"]["max_iterations"] = args.full_iterations
        cid = f"C{idx}_{run_id}"
        cfg_path = cfg_root / f"{cid}.yaml"
        write_yaml(cfg_path, cfg)

        run_dir = runs_root / cid
        run_dir.mkdir(parents=True, exist_ok=True)
        before_len = len(read_text(aeq_log))
        cal = run_command(["python", "run.py", "--config", str(cfg_path), "calibrate"], args.quiet)
        val = run_command(["python", "run.py", "--config", str(cfg_path), "validate"], args.quiet)
        log_delta = tail_from_offset(aeq_log, before_len)
        rc = 0 if cal.returncode == 0 and val.returncode == 0 else 1

        src_cal = Path("outputs/baseline/demand/calibration_report.json")
        src_val = Path("outputs/baseline/demand/validation_report.json")
        dst_cal = run_dir / "calibration_report.json"
        dst_val = run_dir / "validation_report.json"
        if src_cal.exists():
            shutil.copy2(src_cal, dst_cal)
        if src_val.exists():
            shutil.copy2(src_val, dst_val)
        metrics = collect_metrics(load_json(dst_cal), load_json(dst_val))
        has_traceback = "Traceback" in log_delta
        rgap_not_reached = "Desired RGap" in log_delta and "NOT reached" in log_delta
        has_artifacts = dst_cal.exists() and dst_val.exists()
        failure_class = classify_failure(
            cal.returncode,
            val.returncode,
            has_traceback,
            has_artifacts,
            rgap_not_reached,
        )
        row = {
            "run_id": cid,
            "phase": "confirmatory",
            "block": "C",
            "tags": f"from:{run_id}",
            "return_code": rc,
            "calibrate_exit_code": cal.returncode,
            "validate_exit_code": val.returncode,
            "has_traceback": has_traceback,
            "rgap_not_reached": rgap_not_reached,
            "has_artifacts": has_artifacts,
            "failure_class": failure_class,
            **metrics,
        }
        result_rows.append(row)
        confirm_rows.append(row)

    # pick winner from confirmatory, fallback to best screening
    winner = None
    healthy_confirm = [r for r in confirm_rows if r.get("failure_class") != "technical_fail"]
    if healthy_confirm:
        healthy_confirm.sort(key=metrics_score, reverse=True)
        winner = healthy_confirm[0]["run_id"]
    else:
        if top:
            winner = top[0]

    # final replication x2
    repl_rows = []
    if winner:
        winner_cfg = cfg_root / f"{winner}.yaml"
        if not winner_cfg.exists():
            # winner can be screening id
            winner_cfg = cfg_root / f"{winner.replace('C1_', '').replace('C2_', '').replace('C3_', '')}.yaml"
        cfg = yaml.safe_load(winner_cfg.read_text(encoding="utf-8"))
        cfg["calibration"]["max_iterations"] = args.full_iterations
        for i in range(1, 3):
            rid = f"R{i}_{winner}"
            cfg_path = cfg_root / f"{rid}.yaml"
            write_yaml(cfg_path, cfg)
            run_dir = runs_root / rid
            run_dir.mkdir(parents=True, exist_ok=True)
            before_len = len(read_text(aeq_log))
            cal = run_command(["python", "run.py", "--config", str(cfg_path), "calibrate"], args.quiet)
            val = run_command(["python", "run.py", "--config", str(cfg_path), "validate"], args.quiet)
            log_delta = tail_from_offset(aeq_log, before_len)
            rc = 0 if cal.returncode == 0 and val.returncode == 0 else 1
            src_cal = Path("outputs/baseline/demand/calibration_report.json")
            src_val = Path("outputs/baseline/demand/validation_report.json")
            dst_cal = run_dir / "calibration_report.json"
            dst_val = run_dir / "validation_report.json"
            if src_cal.exists():
                shutil.copy2(src_cal, dst_cal)
            if src_val.exists():
                shutil.copy2(src_val, dst_val)
            metrics = collect_metrics(load_json(dst_cal), load_json(dst_val))
            has_traceback = "Traceback" in log_delta
            rgap_not_reached = "Desired RGap" in log_delta and "NOT reached" in log_delta
            has_artifacts = dst_cal.exists() and dst_val.exists()
            failure_class = classify_failure(
                cal.returncode,
                val.returncode,
                has_traceback,
                has_artifacts,
                rgap_not_reached,
            )
            row = {
                "run_id": rid,
                "phase": "replication",
                "block": "R",
                "tags": f"winner:{winner}",
                "return_code": rc,
                "calibrate_exit_code": cal.returncode,
                "validate_exit_code": val.returncode,
                "has_traceback": has_traceback,
                "rgap_not_reached": rgap_not_reached,
                "has_artifacts": has_artifacts,
                "failure_class": failure_class,
                **metrics,
            }
            result_rows.append(row)
            repl_rows.append(row)

    write_csv(out_root / "experiment_matrix.csv", matrix_rows)
    # ranking
    ranked = []
    for r in result_rows:
        rr = dict(r)
        rr["score"] = metrics_score(r)
        ranked.append(rr)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    write_csv(out_root / "results_summary.csv", ranked)

    # baseline reproducibility summary
    baseline_out = {
        "n_runs": len(baseline_rows),
        "runs": baseline_rows,
    }
    (out_root / "baseline_repro_summary.json").write_text(
        json.dumps(baseline_out, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # final candidate config
    final_cfg_target = out_root / "final_candidate.yaml"
    if winner:
        src_cfg = cfg_root / f"{winner}.yaml"
        if src_cfg.exists():
            shutil.copy2(src_cfg, final_cfg_target)

    # RCA quick report
    report = {
        "winner": winner,
        "top_screening": top,
        "confirmatory_count": len(confirm_rows),
        "replication_count": len(repl_rows),
        "technical_fail_count": len([r for r in result_rows if r.get("failure_class") == "technical_fail"]),
        "model_fail_only_count": len([r for r in result_rows if r.get("failure_class") == "model_fail_only"]),
        "observations": [
            "Primary ranking score combines higher r2, higher GEH<5, and lower pct_rmse.",
            "Failure classification distinguishes technical_fail from model_fail_only.",
            "Cross-check supernetwork stats exported separately for gateway/through diagnostics.",
            "Use results_summary.csv for final manual sanity check before production config adoption.",
        ],
    }
    (out_root / "rca_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
