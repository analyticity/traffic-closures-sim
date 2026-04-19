#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


@dataclass
class StepSpec:
    name: str
    expected_artifacts: List[str]


STEPS: List[StepSpec] = [
    StepSpec("clean", []),
    StepSpec("check", []),  # known semantic hole in run.py branch handling
    StepSpec("build-network", ["project/brno_aeq/project_database.sqlite"]),
    StepSpec("normalize-network", ["outputs/baseline/network"]),
    StepSpec("build-zones", ["outputs/baseline/zones/zones.geojson", "outputs/baseline/zones/model_area.geojson"]),
    StepSpec("fetch-data", ["data/cache", "data/sources"]),
    StepSpec("build-supernetwork", ["outputs/baseline/supernetwork/supernetwork_summary.json", "data/cache/through_gateway_pairs.parquet"]),
    StepSpec("build-demand", ["data/demand/od_matrix.aem"]),
    StepSpec(
        "assign-warm-skims",
        ["outputs/baseline/demand/skims.aem", "outputs/baseline/demand/assignment_results.parquet"],
    ),
    StepSpec("distribute", ["data/demand/od_matrix.aem"]),
    StepSpec("assign", ["outputs/baseline/demand/assignment_results.parquet"]),
    StepSpec("calibrate", ["outputs/baseline/demand/calibration_report.json"]),
    StepSpec("validate", ["outputs/baseline/demand/validation_report.json"]),
    StepSpec("learn-profile", ["outputs/baseline/demand/temporal_profile.json"]),
]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_step(config: Path, step: str, timeout_s: int) -> Tuple[int, float, str]:
    t0 = time.time()
    try:
        cp = subprocess.run(
            ["python", "run.py", "--config", str(config), step],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        dt = time.time() - t0
        out = (cp.stdout or "") + "\n" + (cp.stderr or "")
        return cp.returncode, dt, out
    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        out_s = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err_s = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")
        out = out_s + "\n" + err_s + "\nTIMEOUT"
        return 124, dt, out


def step_timeout(step: str, default_s: int) -> int:
    # Heavy steps need longer windows than lightweight validation/reporting steps.
    overrides = {
        "build-network": max(default_s, 1200),
        "fetch-data": max(default_s, 1200),
        "build-supernetwork": max(default_s, 900),
    }
    return overrides.get(step, default_s)


def validate_artifacts(root: Path, expected: List[str]) -> Tuple[bool, List[str], List[str]]:
    missing: List[str] = []
    present: List[str] = []
    for rel in expected:
        p = root / rel
        if p.exists():
            present.append(rel)
        else:
            missing.append(rel)
    return len(missing) == 0, present, missing


def validate_artifact_content(root: Path, step: str) -> List[str]:
    issues: List[str] = []
    if step == "build-zones":
        zp = root / "outputs/baseline/zones/zones.geojson"
        if zp.exists():
            obj = json.loads(zp.read_text(encoding="utf-8"))
            feats = obj.get("features") or []
            props = (feats[0].get("properties") or {}) if feats else {}
            if "zone_id" not in props:
                issues.append("zones_missing_zone_id")
    elif step == "build-supernetwork":
        sp = root / "outputs/baseline/supernetwork/supernetwork_summary.json"
        if sp.exists():
            obj = json.loads(sp.read_text(encoding="utf-8"))
            if int(obj.get("gateway_count", 0)) <= 0:
                issues.append("supernetwork_gateway_count_zero")
    elif step == "build-demand":
        cp = root / "outputs/baseline/demand/od_matrix_summary.json"
        if cp.exists():
            obj = json.loads(cp.read_text(encoding="utf-8"))
            if float(obj.get("total_daily_trips", 0.0)) <= 0:
                issues.append("od_matrix_total_daily_nonpositive")
    elif step == "calibrate":
        cp = root / "outputs/baseline/demand/calibration_report.json"
        if cp.exists():
            obj = json.loads(cp.read_text(encoding="utf-8"))
            final = obj.get("final") or {}
            if float(final.get("sum_modeled", 0.0)) <= 0:
                issues.append("calibration_sum_modeled_nonpositive")
    elif step == "learn-profile":
        tp = root / "outputs/baseline/demand/temporal_profile.json"
        if tp.exists():
            obj = json.loads(tp.read_text(encoding="utf-8"))
            if "day_factors" not in obj or "day_period_shares" not in obj:
                issues.append("temporal_profile_incomplete")
    return issues


def infer_warnings(step: str, log_text: str) -> List[str]:
    warns: List[str] = []
    if "Traceback" in log_text:
        warns.append("traceback_detected")
    if "WARNING" in log_text:
        warns.append("warning_in_log")
    if "Desired RGap" in log_text and "NOT reached" in log_text:
        warns.append("assignment_rgap_not_reached")
    if step == "check" and not log_text.strip():
        warns.append("check_step_no_output")
    if "missing required inputs:" in log_text:
        warns.append("preflight_missing_inputs")
    return warns


def build_magic_inventory(repo_root: Path, out_dir: Path) -> None:
    cfg_path = repo_root / "config" / "sim.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    rows: List[Dict[str, object]] = []

    def walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(f"{prefix}[{i}]", v)
        else:
            if isinstance(obj, (int, float)):
                rows.append(
                    {
                        "source": "config",
                        "key": prefix,
                        "value": obj,
                        "category": "must_be_data_driven" if abs(float(obj)) > 0 and prefix.endswith(("_m", "_pct", "_factor", "_ratio")) else "keep_with_justification",
                        "proposal": "derive from empirical quantiles / CV grid",
                    }
                )

    walk("", cfg)

    code_files = [
        repo_root / "src" / "sim" / "calibration.py",
        repo_root / "src" / "sim" / "fetch_datasets.py",
        repo_root / "scripts" / "metric_principle_experiment.py",
    ]
    lit = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])")
    for fp in code_files:
        if not fp.exists():
            continue
        for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            hits = lit.findall(line)
            for h in hits:
                if h in {"0", "1", "2"}:
                    continue
                rows.append(
                    {
                        "source": "code",
                        "key": f"{fp.relative_to(repo_root)}:{i}",
                        "value": h,
                        "category": "must_be_data_driven",
                        "proposal": "replace constant with data-estimated parameter or config-driven value",
                    }
                )

    rows = sorted(rows, key=lambda r: (str(r["source"]), str(r["key"])))
    write_csv(out_dir / "magic_constants_inventory.csv", rows)


def write_metrics_consistency(repo_root: Path, out_dir: Path) -> None:
    candidates = [
        repo_root / "outputs" / "tuning" / "metric_principle_experiment_v2" / "results_summary_enriched.csv",
        repo_root / "outputs" / "tuning" / "metric_principle_experiment" / "results_summary_enriched.csv",
    ]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        (out_dir / "metrics_consistency_report.md").write_text(
            "# Metrics Consistency Report\n\nNo experiment file found.\n",
            encoding="utf-8",
        )
        return
    df = pd.read_csv(p)
    lines = [
        "# Metrics Consistency Report",
        "",
        f"- Rows: {len(df)}",
        f"- Phases: {sorted(df['phase'].dropna().unique().tolist())}",
    ]
    if "phase" in df.columns and "score_ObjB" in df.columns:
        for phase in ["pilot", "principle_ab", "confirmatory"]:
            sub = df[df["phase"] == phase]
            if sub.empty:
                continue
            lines += [
                "",
                f"## {phase}",
                f"- score_ObjB range: {sub['score_ObjB'].min():.4f} .. {sub['score_ObjB'].max():.4f}",
                f"- valid_r2 range: {sub['valid_r2'].min():.4f} .. {sub['valid_r2'].max():.4f}",
                f"- class_bias_max_abs range: {sub['ext_class_bias_max_abs_pct'].min():.2f} .. {sub['ext_class_bias_max_abs_pct'].max():.2f}",
            ]
    (out_dir / "metrics_consistency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fix_packages(out_dir: Path) -> None:
    txt = """# Data-driven Refactor Proposal

## Package A: Hard consistency fixes
- Enforce step contracts (explicit check-step action in `run.py`).
- Add strict artifact schema checks (CRS, column sets, units).

## Package B: Objective and acceptance policy
- Keep ObjB primary, ObjC secondary, ObjA legacy.
- Keep hard reject on class bias and warning thresholds for wMAPE/GEH.

## Package C: De-hack configuration
- Replace static buffers/factors with quantile-based defaults estimated from data.
- Keep only justified constants with explicit comments and calibration source.

## Package D: Reproducibility
- Immutable config snapshots for every run.
- Deterministic seeds where supported and run metadata manifest.
"""
    (out_dir / "data_driven_refactor_proposal.md").write_text(txt, encoding="utf-8")


def write_unused_policy_report(repo_root: Path, out_dir: Path) -> None:
    cfg_path = repo_root / "config" / "sim.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    pol = (((cfg.get("experiments") or {}).get("policy") or {}))
    used_keys = {
        "objective.primary",
        "objective.secondary",
        "objective.legacy",
        "objective.objB_weights.spearman",
        "objective.objB_weights.geh_lt5",
        "objective.objB_weights.wmape_pct",
        "objective.objB_weights.class_bias_max_abs_pct",
        "objective.objB_weights.pct_rmse",
        "objective.objC_weights.spearman",
        "objective.objC_weights.mean_mae_class",
        "objective.objC_weights.abs_log_sum_ratio",
        "objective.objC_weights.pct_rmse",
        "objective.sum_ratio_clip.min",
        "objective.sum_ratio_clip.max",
        "guardrails.hard_class_bias_max_abs_pct",
        "guardrails.warn_wmape_pct",
        "guardrails.warn_geh_lt5_pct",
        "tuning_ranges.assignment.max_iter",
        "tuning_ranges.assignment.rgap_target",
        "tuning_ranges.matching.match_buffer_m",
        "tuning_ranges.scaling.damping",
        "tuning_ranges.scaling.min_factor",
        "tuning_ranges.scaling.max_factor",
        "tuning_ranges.gateway.damping",
        "tuning_ranges.external_daily_trips",
    }
    seen: List[str] = []

    def walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            seen.append(prefix)

    walk("", pol)
    unused = sorted(k for k in seen if k not in used_keys)
    lines = ["# Unused Policy Keys", "", f"- total_defined: {len(seen)}", f"- unused: {len(unused)}", ""]
    lines += [f"- {k}" for k in unused] if unused else ["- none"]
    (out_dir / "unused_policy_keys.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_with_previous(repo_root: Path, out_dir: Path, current_rows: List[Dict[str, object]]) -> None:
    prev = repo_root / "outputs" / "audit" / "full_pipeline_audit_prev" / "pipeline_audit.csv"
    cur = pd.DataFrame(current_rows)
    summary = {"has_previous": prev.exists(), "current_ok_steps": int((cur["status"] == "ok").sum())}
    if prev.exists():
        p = pd.read_csv(prev)
        summary["previous_ok_steps"] = int((p["status"] == "ok").sum())
        summary["delta_ok_steps"] = summary["current_ok_steps"] - summary["previous_ok_steps"]
    (out_dir / "before_after_compare.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full pipeline validity audit")
    ap.add_argument("--config", default="config/sim.yaml")
    ap.add_argument("--out-root", default="outputs/audit/full_pipeline_audit")
    ap.add_argument("--timeout-s", type=int, default=1200)
    ap.add_argument("--stop-on-fail", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "step_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_src = (repo_root / args.config).resolve()
    shutil.copy2(config_src, out_root / "config_snapshot.yaml")

    rows: List[Dict[str, object]] = []
    for spec in STEPS:
        rc, dt, out = run_step(config_src, spec.name, step_timeout(spec.name, args.timeout_s))
        # clean step can remove outputs/ entirely, including this audit directory
        out_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        ok_artifacts, present, missing = validate_artifacts(repo_root, spec.expected_artifacts)
        warns = infer_warnings(spec.name, out)
        content_issues = validate_artifact_content(repo_root, spec.name)
        warns.extend(content_issues)
        status = "ok"
        if rc != 0 or (spec.expected_artifacts and not ok_artifacts):
            status = "fail"
            if "preflight_missing_inputs" in warns:
                status = "blocked"
        elif warns:
            status = "warning"
        row = {
            "step": spec.name,
            "exit_code": rc,
            "elapsed_s": round(dt, 2),
            "status": status,
            "warnings": json.dumps(warns, ensure_ascii=True),
            "expected_artifacts": json.dumps(spec.expected_artifacts, ensure_ascii=True),
            "present_artifacts": json.dumps(present, ensure_ascii=True),
            "missing_artifacts": json.dumps(missing, ensure_ascii=True),
        }
        rows.append(row)
        (logs_dir / f"{spec.name}.log").write_text(out, encoding="utf-8", errors="ignore")
        if args.stop_on_fail and status in {"fail", "blocked"}:
            break

    write_csv(out_root / "pipeline_audit.csv", rows)
    (out_root / "pipeline_audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")

    build_magic_inventory(repo_root, out_root)
    write_metrics_consistency(repo_root, out_root)
    write_fix_packages(out_root)
    write_unused_policy_report(repo_root, out_root)
    compare_with_previous(repo_root, out_root, rows)
    print(f"Audit complete. Outputs: {out_root}")


if __name__ == "__main__":
    main()

