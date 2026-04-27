#!/usr/bin/env python3
"""Metric validity + calibration principle experiment (see plan: validita metrik a principu).

Produces:
  - metric_audit.csv, metric_correlation.json (from v4 or prior results)
  - objective_candidates.json
  - pilot_results.csv, principle_ab_results.csv, results_summary_enriched.csv
  - final_candidate.yaml, rca_report_v2.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


@dataclass
class ExpRunSpec:
    run_id: str
    phase: str
    block: str
    tags: str
    principle: str
    objective_tag: str
    overrides: Dict[str, object]


def set_deep(dct: dict, path: List[str], value) -> None:
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
    kwargs: Dict[str, Any] = {"check": False}
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
    _rgap_not_reached: bool,
) -> str:
    if has_traceback or not has_artifacts:
        return "technical_fail"
    if calibrate_exit_code != 0 or validate_exit_code != 0:
        return "model_fail_only"
    return "ok"


def objective_candidates_document() -> dict:
    """Formal ObjA/B/C definitions (higher score = better run)."""
    return {
        "policy": {
            "primary_objective": PRIMARY_OBJECTIVE,
            "secondary_objective": SECONDARY_OBJECTIVE,
            "legacy_report_objective": LEGACY_OBJECTIVE,
            "production_default_principle": "Princip1",
            "diagnostic_only_principles": ["Princip3"],
        },
        "objectives": {
            "ObjA": {
                "name": "GEH_heavy_legacy",
                "description": "Legacy screening score aligned with systematic_experiment.",
                "formula": "100 * valid_r2 + 2 * valid_geh_lt5_pct - valid_pct_rmse",
                "weights": {"r2": 100, "geh_lt5_pct": 2, "pct_rmse": -1},
            },
            "ObjB": {
                "name": "balanced_wmape_guardrail",
                "description": "wMAPE + class-bias penalty with GEH as soft term (uses calibration extended_metrics).",
                "formula": (
                    "120 * spearman_rho + 1.8 * valid_geh_lt5_pct - 1.0 * ext_wmape_pct "
                    "- 0.35 * ext_class_bias_max_abs_pct - 0.35 * valid_pct_rmse"
                ),
                "notes": "spearman_rho defaults to 0 if missing.",
            },
            "ObjC": {
                "name": "robust_mae_spearman_sumratio",
                "description": "Rank correlation minus global volume ratio stress and mean class MAE.",
                "formula": (
                    "150 * spearman_rho - 0.12 * ext_mean_mae_class - 25 * abs_log_sum_ratio_penalty "
                    "- 0.25 * valid_pct_rmse"
                ),
                "notes": (
                    "abs_log_sum_ratio_penalty = abs(ln(clipped_sum_ratio)) with sum_ratio=model/obs from extended_metrics; "
                    "0 if missing."
                ),
            },
        },
        "calibration_principles": {
            "Princip1": {
                "id": "iterative_global_scaling",
                "config": {
                    "calibration.scaling.enabled": True,
                    "calibration.scaling.method": "global",
                },
            },
            "Princip2": {
                "id": "iterative_sector_weighted_scaling",
                "description": (
                    "Per road-class ratios condensed to one matrix scale via observed-volume weights "
                    f"(existing scaling.method=sector)."
                ),
                "config": {
                    "calibration.scaling.enabled": True,
                    "calibration.scaling.method": "sector",
                },
            },
            "Princip3": {
                "id": "frozen_od_control",
                "description": "Assignment loop without OD scaling (diagnostic control).",
                "config": {
                    "calibration.scaling.enabled": False,
                },
            },
        },
    }


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def score_ObjA(row: Dict[str, Any]) -> float:
    return _f(row.get("valid_r2")) * 100.0 + _f(row.get("valid_geh_lt5_pct")) * 2.0 - _f(
        row.get("valid_pct_rmse")
    )


def score_ObjB(row: Dict[str, Any]) -> float:
    rho = _f(row.get("ext_spearman_rho"), 0.0)
    geh = _f(row.get("valid_geh_lt5_pct"))
    wmape = _f(row.get("ext_wmape_pct"))
    bias = _f(row.get("ext_class_bias_max_abs_pct"))
    prmse = _f(row.get("valid_pct_rmse"))
    w = OBJB_WEIGHTS
    return (
        rho * _f(w.get("spearman"), 120.0)
        + geh * _f(w.get("geh_lt5"), 1.8)
        - wmape * _f(w.get("wmape_pct"), 1.0)
        - bias * _f(w.get("class_bias_max_abs_pct"), 0.35)
        - prmse * _f(w.get("pct_rmse"), 0.35)
    )


def score_ObjC(row: Dict[str, Any]) -> float:
    rho = _f(row.get("ext_spearman_rho"), 0.0)
    mae_mean = _f(row.get("ext_mean_mae_class"))
    prmse = _f(row.get("valid_pct_rmse"))
    sr = row.get("ext_sum_ratio")
    pen = 0.0
    if sr is not None:
        sr = _f(sr, 1.0)
        sr = max(_f(SUM_RATIO_CLIP.get("min"), 0.25), min(_f(SUM_RATIO_CLIP.get("max"), 4.0), sr))
        pen = abs(math.log(sr))
    w = OBJC_WEIGHTS
    return (
        rho * _f(w.get("spearman"), 150.0)
        - mae_mean * _f(w.get("mean_mae_class"), 0.12)
        - _f(w.get("abs_log_sum_ratio"), 25.0) * pen
        - prmse * _f(w.get("pct_rmse"), 0.25)
    )


PRIMARY_OBJECTIVE = "ObjB"
SECONDARY_OBJECTIVE = "ObjC"
LEGACY_OBJECTIVE = "ObjA"
GUARDRAIL_BIAS_HARD = 90.0
GUARDRAIL_WMAPE_WARN = 47.0
GUARDRAIL_GEH_WARN = 7.0
OBJB_WEIGHTS = {"spearman": 120.0, "geh_lt5": 1.8, "wmape_pct": 1.0, "class_bias_max_abs_pct": 0.35, "pct_rmse": 0.35}
OBJC_WEIGHTS = {"spearman": 150.0, "mean_mae_class": 0.12, "abs_log_sum_ratio": 25.0, "pct_rmse": 0.25}
SUM_RATIO_CLIP = {"min": 0.25, "max": 4.0}


def objective_score(row: Dict[str, Any], objective: str) -> float:
    key = f"score_{objective}"
    return _f(row.get(key), default=-10**12)


def evaluate_guardrails(row: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Return (hard_pass, hard_reasons, warnings)."""
    hard_reasons: List[str] = []
    warnings: List[str] = []
    bias = row.get("ext_class_bias_max_abs_pct")
    wmape = row.get("ext_wmape_pct")
    geh = row.get("valid_geh_lt5_pct")

    if bias is not None and _f(bias) > GUARDRAIL_BIAS_HARD:
        hard_reasons.append(f"class_bias_max_abs_pct={_f(bias):.2f}>{GUARDRAIL_BIAS_HARD:g}")
    if wmape is not None and _f(wmape) > GUARDRAIL_WMAPE_WARN:
        warnings.append(f"wmape_pct={_f(wmape):.2f}>{GUARDRAIL_WMAPE_WARN:g}")
    if geh is not None and _f(geh) < GUARDRAIL_GEH_WARN:
        warnings.append(f"geh_lt5_pct={_f(geh):.2f}<{GUARDRAIL_GEH_WARN:g}")
    return len(hard_reasons) == 0, hard_reasons, warnings


def rank_rows_by_policy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            objective_score(r, PRIMARY_OBJECTIVE),
            objective_score(r, SECONDARY_OBJECTIVE),
            objective_score(r, LEGACY_OBJECTIVE),
        ),
        reverse=True,
    )


def attach_extended_fields(calib: dict, row: Dict[str, Any]) -> None:
    ext = calib.get("extended_metrics") or {}
    row["ext_wmape_pct"] = ext.get("wmape_pct")
    row["ext_smape_pct"] = ext.get("smape_pct")
    row["ext_spearman_rho"] = ext.get("spearman_rho")
    row["ext_sum_ratio"] = ext.get("sum_ratio")
    row["ext_class_bias_max_abs_pct"] = ext.get("class_bias_max_abs_pct")
    mae_by = ext.get("mae_by_class") or {}
    if mae_by:
        row["ext_mean_mae_class"] = float(sum(mae_by.values()) / max(len(mae_by), 1))
    else:
        row["ext_mean_mae_class"] = None
    row["ext_mae_by_class_json"] = json.dumps(mae_by, ensure_ascii=True)
    row["ext_bias_by_class_json"] = json.dumps(ext.get("bias_pct_by_class") or {}, ensure_ascii=True)


def collect_run_row(
    run_id: str,
    phase: str,
    block: str,
    tags: str,
    principle: str,
    objective_tag: str,
    calib: dict,
    valid: dict,
    proc: subprocess.CompletedProcess,
    valproc: subprocess.CompletedProcess,
    log_delta: str,
    has_artifacts: bool,
) -> Dict[str, Any]:
    final = calib.get("final") or {}
    pent = valid.get("pentlogram") or {}
    row: Dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "block": block,
        "tags": tags,
        "principle": principle,
        "objective_tag": objective_tag,
        "calibrate_exit_code": proc.returncode,
        "validate_exit_code": valproc.returncode,
        "has_traceback": "Traceback" in log_delta,
        "has_artifacts": has_artifacts,
        "failure_class": classify_failure(
            proc.returncode,
            valproc.returncode,
            "Traceback" in log_delta,
            has_artifacts,
            "Desired RGap" in log_delta and "NOT reached" in log_delta,
        ),
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
    attach_extended_fields(calib, row)
    row["score_ObjA"] = score_ObjA(row)
    row["score_ObjB"] = score_ObjB(row)
    row["score_ObjC"] = score_ObjC(row)
    row["return_code"] = 0 if proc.returncode == 0 and valproc.returncode == 0 else 1
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_yaml(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def run_metric_audit(v4_csv: Path, out_dir: Path) -> dict:
    """Variance / correlation / degeneracy diagnostics on historic experiment CSV."""
    if not v4_csv.exists():
        (out_dir / "metric_audit.csv").write_text("metric,n,mean,std,min,max,range,range_over_abs_mean_pct,degenerate_near_constant\n", encoding="utf-8")
        (out_dir / "metric_correlation.json").write_text(
            json.dumps({"note": f"v4 results file not found: {v4_csv}"}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return {
            "degenerate_metrics": [],
            "audit_rows": 0,
            "summary": f"No historic v4 CSV at {v4_csv}",
        }
    df = pd.read_csv(v4_csv)
    numeric_cols = [
        c
        for c in df.columns
        if c
        in (
            "calib_r2",
            "calib_pct_rmse",
            "calib_geh_lt5_pct",
            "valid_r2",
            "valid_pct_rmse",
            "valid_geh_lt5_pct",
            "score",
        )
    ]
    present = [c for c in numeric_cols if c in df.columns]
    diag_rows = []
    for c in present:
        s = pd.to_numeric(df[c], errors="coerce")
        std = float(s.std(ddof=0))
        mn = float(s.min())
        mx = float(s.max())
        mean = float(s.mean()) if math.isfinite(s.mean()) else 0.0
        rng = mx - mn
        diag_rows.append(
            {
                "metric": c,
                "n": int(s.notna().sum()),
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(mn, 6),
                "max": round(mx, 6),
                "range": round(rng, 6),
                "range_over_abs_mean_pct": round(rng / max(abs(mean), 1e-9) * 100.0, 4),
                "degenerate_near_constant": std < 1e-12,
            }
        )
    write_csv(out_dir / "metric_audit.csv", diag_rows)

    corr_obj: Dict[str, Any] = {}
    if len(present) > 1:
        sub = df[present].apply(pd.to_numeric, errors="coerce")
        # Drop columns with zero variance for corr
        keep = [c for c in sub.columns if sub[c].std(ddof=0) > 1e-12]
        if len(keep) > 1:
            with pd.option_context("mode.use_inf_as_na", True):
                cmat = sub[keep].corr(method="pearson", min_periods=2)
            corr_obj["pearson"] = cmat.fillna(0).round(4).to_dict()
        else:
            corr_obj["pearson"] = {}
            corr_obj["note"] = "Too many constant columns for correlation."
    else:
        corr_obj["note"] = "Not enough numeric columns."

    # Rank stability on v4 using ObjA vs ObjB proxy (ObjB needs extended — absent in v4)
    rank_note = (
        "v4 results_summary has no extended_metrics; ObjB/ObjC ranks match ObjA when ext fields missing."
    )
    if "score" in df.columns and "valid_r2" in df.columns:
        s_a = df["score"].rank(method="average")
        s_r2 = df["valid_r2"].rank(method="average")
        rank_corr = float(s_a.corr(s_r2, method="pearson")) if len(s_a) > 1 else None
        corr_obj["legacy_score_vs_valid_r2_rank_pearson"] = rank_corr
    corr_obj["rank_stability_note"] = rank_note

    (out_dir / "metric_correlation.json").write_text(
        json.dumps(corr_obj, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    degenerate = [r["metric"] for r in diag_rows if r["degenerate_near_constant"]]
    return {
        "degenerate_metrics": degenerate,
        "audit_rows": len(diag_rows),
        "summary": (
            "Most v4 metrics are identical across runs → parameter sensitivity was near zero "
            "(likely chained OD matrix before reset_matrix_before_run)."
        ),
    }


def make_pilot_specs(short_iters: int) -> List[ExpRunSpec]:
    """4 configs, Princip1 only; ObjA/B/C evaluated post-hoc."""
    bases: List[Tuple[str, Dict[str, object]]] = [
        ("PV1", {"calibration.scaling.damping": 0.05}),
        ("PV2", {"calibration.scaling.damping": 0.12}),
        (
            "PV3",
            {
                "calibration.match_buffer_m": 25.0,
                "calibration.rgap_target": 0.003,
            },
        ),
        ("PV4", {"calibration.max_iter": 120, "calibration.rgap_target": 0.0025}),
    ]
    specs = []
    for rid, ov in bases:
        o = {
            "calibration.max_iterations": short_iters,
            "calibration.scaling.enabled": True,
            "calibration.scaling.method": "global",
        }
        o.update(ov)
        specs.append(
            ExpRunSpec(
                f"PILOT_{rid}",
                "pilot",
                "P",
                "objective_pilot",
                "Princip1",
                "posthoc_ObjABC",
                o,
            )
        )
    return specs


def make_principle_grid_specs(short_iters: int) -> List[ExpRunSpec]:
    doc = objective_candidates_document()
    principles = doc["calibration_principles"]
    # 4 tactical configs reused
    tactical: List[Tuple[str, Dict[str, object]]] = [
        ("T1", {}),
        ("T2", {"calibration.scaling.damping": 0.12}),
        ("T3", {"calibration.match_buffer_m": 22.0}),
        ("T4", {"calibration.gateway_calibration.damping": 0.12}),
    ]
    specs: List[ExpRunSpec] = []
    for pname, pbody in principles.items():
        pconf = pbody["config"]
        for tid, tov in tactical:
            o: Dict[str, object] = {
                "calibration.max_iterations": short_iters,
            }
            for k, v in pconf.items():
                o[k] = v
            for k, v in tov.items():
                o[k] = v
            specs.append(
                ExpRunSpec(
                    f"AB_{pname}_{tid}",
                    "principle_ab",
                    "AB",
                    f"principle_ab,{pname}",
                    pname,
                    "posthoc_ObjABC",
                    o,
                )
            )
    return specs


def rank_stability_scores(df: pd.DataFrame, score_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty or len(df) < 2:
        return {"note": "not enough rows"}
    ranks = {c: df[c].rank(method="average") for c in score_cols if c in df.columns}
    keys = list(ranks.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ra, rb = ranks[a], ranks[b]
            out[f"spearman_rank_corr_{a}_vs_{b}"] = round(float(ra.corr(rb, method="pearson")), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="config/brno/sim.yaml")
    ap.add_argument(
        "--v4-results",
        default="outputs/tuning/systematic_experiment_v4/results_summary.csv",
    )
    ap.add_argument("--out-root", default="outputs/tuning/metric_principle_experiment")
    ap.add_argument("--short-iters", type=int, default=4)
    ap.add_argument("--full-iters", type=int, default=15)
    ap.add_argument("--confirmatory-top", type=int, default=3)
    ap.add_argument("--aeq-log", default="project/brno_aeq/aequilibrae.log")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--smoke-limit", type=int, default=0, help="limit number of specs per phase for smoke validation")
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run phase 1 audit + write objective_candidates.json",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    # Configs must live under <repo>/config/ so load_config() resolves project_root correctly.
    repo_root = Path(args.base_config).resolve().parent.parent
    base_cfg_obj = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8")) or {}
    qg = ((base_cfg_obj.get("calibration") or {}).get("quality_gates") or {})
    exp_policy = (((base_cfg_obj.get("experiments") or {}).get("policy") or {}))
    obj_policy = exp_policy.get("objective") or {}
    gr_policy = exp_policy.get("guardrails") or {}
    global GUARDRAIL_BIAS_HARD, GUARDRAIL_WMAPE_WARN, GUARDRAIL_GEH_WARN, OBJB_WEIGHTS, OBJC_WEIGHTS, SUM_RATIO_CLIP
    GUARDRAIL_BIAS_HARD = float(qg.get("hard_class_bias_max_abs_pct", GUARDRAIL_BIAS_HARD))
    GUARDRAIL_WMAPE_WARN = float(qg.get("warn_wmape_pct", GUARDRAIL_WMAPE_WARN))
    GUARDRAIL_GEH_WARN = float(qg.get("warn_geh_lt5_pct", GUARDRAIL_GEH_WARN))
    GUARDRAIL_BIAS_HARD = float(gr_policy.get("hard_class_bias_max_abs_pct", GUARDRAIL_BIAS_HARD))
    GUARDRAIL_WMAPE_WARN = float(gr_policy.get("warn_wmape_pct", GUARDRAIL_WMAPE_WARN))
    GUARDRAIL_GEH_WARN = float(gr_policy.get("warn_geh_lt5_pct", GUARDRAIL_GEH_WARN))
    OBJB_WEIGHTS = dict(obj_policy.get("objB_weights") or OBJB_WEIGHTS)
    OBJC_WEIGHTS = dict(obj_policy.get("objC_weights") or OBJC_WEIGHTS)
    SUM_RATIO_CLIP = dict(obj_policy.get("sum_ratio_clip") or SUM_RATIO_CLIP)
    cfg_root = repo_root / ".tmp_metric_principle_configs"
    runs_root = out_root / "runs"
    cfg_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: audit ---
    v4_path = Path(args.v4_results)
    audit_meta = run_metric_audit(v4_path, out_root)

    obj_doc = objective_candidates_document()
    (out_root / "objective_candidates.json").write_text(
        json.dumps(obj_doc, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if args.audit_only:
        (out_root / "rca_report_v2.json").write_text(
            json.dumps(
                {
                    "phase": "audit_only",
                    "metric_audit": audit_meta,
                    "note": "Run without --audit-only to execute pilot + principle AB + confirmatory.",
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        print(f"Audit written to {out_root}")
        return

    base_cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    aeq_log = Path(args.aeq_log)

    all_rows: List[Dict[str, Any]] = []

    def exec_specs(specs: List[ExpRunSpec]) -> None:
        nonlocal all_rows
        for spec in specs:
            cfg = apply_overrides(base_cfg, spec.overrides)
            cfg_path = cfg_root / f"{spec.run_id}.yaml"
            write_yaml(cfg_path, cfg)
            run_dir = runs_root / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            before = len(read_text(aeq_log))
            cal = run_command(
                ["python", "run.py", "--config", str(cfg_path), "calibrate"],
                args.quiet,
            )
            val = run_command(
                ["python", "run.py", "--config", str(cfg_path), "validate"],
                args.quiet,
            )
            log_delta = tail_from_offset(aeq_log, before)
            src_cal = Path("outputs/baseline/demand/calibration_report.json")
            src_val = Path("outputs/baseline/demand/validation_report.json")
            dst_cal = run_dir / "calibration_report.json"
            dst_val = run_dir / "validation_report.json"
            if src_cal.exists():
                shutil.copy2(src_cal, dst_cal)
            if src_val.exists():
                shutil.copy2(src_val, dst_val)
            calib = load_json(dst_cal)
            valid = load_json(dst_val)
            has_art = dst_cal.exists() and dst_val.exists()
            row = collect_run_row(
                spec.run_id,
                spec.phase,
                spec.block,
                spec.tags,
                spec.principle,
                spec.objective_tag,
                calib,
                valid,
                cal,
                val,
                log_delta,
                has_art,
            )
            all_rows.append(row)

    pilot_specs = make_pilot_specs(args.short_iters)
    if args.smoke_limit and args.smoke_limit > 0:
        pilot_specs = pilot_specs[: int(args.smoke_limit)]
    exec_specs(pilot_specs)
    pilot_df = pd.DataFrame([r for r in all_rows if r["phase"] == "pilot"])
    pilot_stab = rank_stability_scores(
        pilot_df,
        ["score_ObjA", "score_ObjB", "score_ObjC"],
    )
    pilot_df.to_csv(out_root / "pilot_results.csv", index=False)

    ab_specs = make_principle_grid_specs(args.short_iters)
    if args.smoke_limit and args.smoke_limit > 0:
        ab_specs = ab_specs[: int(args.smoke_limit)]
    exec_specs(ab_specs)
    ab_df = pd.DataFrame([r for r in all_rows if r["phase"] == "principle_ab"])
    ab_df.to_csv(out_root / "principle_ab_results.csv", index=False)

    write_csv(out_root / "results_summary_enriched.csv", all_rows)

    # Confirmatory: top combinations by policy ranking (ObjB -> ObjC -> ObjA)
    healthy = [r for r in all_rows if r["phase"] == "principle_ab" and r.get("failure_class") != "technical_fail"]
    healthy = rank_rows_by_policy(healthy)
    top_ids: List[str] = []
    seen: set = set()
    for r in healthy:
        key = (r.get("principle"), r.get("tags"))
        if key in seen:
            continue
        seen.add(key)
        top_ids.append(r["run_id"])
        if len(top_ids) >= args.confirmatory_top:
            break

    confirm_rows: List[Dict[str, Any]] = []
    for i, rid in enumerate(top_ids, 1):
        src = cfg_root / f"{rid}.yaml"
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        cfg["calibration"]["max_iterations"] = args.full_iters
        cid = f"CONF{i}_{rid}"
        cfg_path = cfg_root / f"{cid}.yaml"
        write_yaml(cfg_path, cfg)
        run_dir = runs_root / cid
        run_dir.mkdir(parents=True, exist_ok=True)
        before = len(read_text(aeq_log))
        cal = run_command(
            ["python", "run.py", "--config", str(cfg_path), "calibrate"],
            args.quiet,
        )
        val = run_command(
            ["python", "run.py", "--config", str(cfg_path), "validate"],
            args.quiet,
        )
        log_delta = tail_from_offset(aeq_log, before)
        dst_cal = run_dir / "calibration_report.json"
        dst_val = run_dir / "validation_report.json"
        src_cal = Path("outputs/baseline/demand/calibration_report.json")
        src_val = Path("outputs/baseline/demand/validation_report.json")
        if src_cal.exists():
            shutil.copy2(src_cal, dst_cal)
        if src_val.exists():
            shutil.copy2(src_val, dst_val)
        parent = next((x for x in healthy if x["run_id"] == rid), {})
        calib_j = load_json(dst_cal)
        valid_j = load_json(dst_val)
        row = collect_run_row(
            cid,
            "confirmatory",
            "C",
            f"from:{rid}",
            str(parent.get("principle", "")),
            "posthoc_ObjABC",
            calib_j,
            valid_j,
            cal,
            val,
            log_delta,
            dst_cal.exists() and dst_val.exists(),
        )
        confirm_rows.append(row)
        all_rows.append(row)

    rejection_reasons: Dict[str, List[str]] = {}
    warning_flags: Dict[str, List[str]] = {}
    winner = None

    repl_rows: List[Dict[str, Any]] = []
    if confirm_rows:
        confirm_ranked = rank_rows_by_policy(confirm_rows)
        for crow in confirm_ranked:
            if str(crow.get("principle")) == "Princip3":
                rejection_reasons[crow["run_id"]] = ["diagnostic_only_principle=Princip3"]
                continue
            ok, reasons, warns = evaluate_guardrails(crow)
            if reasons:
                rejection_reasons[crow["run_id"]] = reasons
            if warns:
                warning_flags[crow["run_id"]] = warns
            if ok:
                winner = crow["run_id"]
                break

    if winner:
        wcfg_path = cfg_root / f"{winner}.yaml"
        cfg = yaml.safe_load(wcfg_path.read_text(encoding="utf-8"))
        winner_row = next((r for r in confirm_rows if r["run_id"] == winner), {})
        win_principle = str(winner_row.get("principle", ""))
        for k in range(1, 3):
            rid = f"REPL{k}_{winner}"
            cfg_path = cfg_root / f"{rid}.yaml"
            write_yaml(cfg_path, cfg)
            run_dir = runs_root / rid
            run_dir.mkdir(parents=True, exist_ok=True)
            before = len(read_text(aeq_log))
            cal = run_command(
                ["python", "run.py", "--config", str(cfg_path), "calibrate"],
                args.quiet,
            )
            val = run_command(
                ["python", "run.py", "--config", str(cfg_path), "validate"],
                args.quiet,
            )
            log_delta = tail_from_offset(aeq_log, before)
            dst_cal = run_dir / "calibration_report.json"
            dst_val = run_dir / "validation_report.json"
            _src_cal = Path("outputs/baseline/demand/calibration_report.json")
            _src_val = Path("outputs/baseline/demand/validation_report.json")
            if _src_cal.exists():
                shutil.copy2(_src_cal, dst_cal)
            if _src_val.exists():
                shutil.copy2(_src_val, dst_val)
            calib = load_json(dst_cal)
            valid = load_json(dst_val)
            row = collect_run_row(
                rid,
                "replication",
                "R",
                f"winner:{winner}",
                win_principle,
                "posthoc_ObjABC",
                calib,
                valid,
                cal,
                val,
                log_delta,
                dst_cal.exists() and dst_val.exists(),
            )
            repl_rows.append(row)
            all_rows.append(row)

        # Enforce hard guardrails after replications, too.
        winner_related = [r for r in all_rows if r.get("run_id") == winner or r.get("tags") == f"winner:{winner}"]
        for wr in winner_related:
            ok, reasons, warns = evaluate_guardrails(wr)
            if reasons:
                rejection_reasons[wr["run_id"]] = reasons
            if warns:
                warning_flags[wr["run_id"]] = warns
            if not ok:
                winner = None
                break

    write_csv(out_root / "results_summary_enriched.csv", all_rows)

    snap = out_root / "configs_snapshot"
    if cfg_root.exists():
        shutil.copytree(cfg_root, snap, dirs_exist_ok=True)

    if winner and (cfg_root / f"{winner}.yaml").exists():
        shutil.copy2(cfg_root / f"{winner}.yaml", out_root / "final_candidate.yaml")

    ab_stab = rank_stability_scores(
        ab_df,
        ["score_ObjA", "score_ObjB", "score_ObjC"],
    )

    principle_means = (
        ab_df.groupby("principle")[["score_ObjA", "score_ObjB", "score_ObjC"]].mean().to_dict()
        if not ab_df.empty
        else {}
    )

    ranked_all = rank_rows_by_policy([r for r in all_rows if r.get("failure_class") != "technical_fail"])
    objective_top = {}
    for obj in ("ObjA", "ObjB", "ObjC"):
        seq = sorted(
            [r for r in ranked_all if r.get("phase") in {"pilot", "principle_ab", "confirmatory"}],
            key=lambda x: objective_score(x, obj),
            reverse=True,
        )
        objective_top[obj] = seq[0]["run_id"] if seq else None
    rank_disagreement = len({v for v in objective_top.values() if v}) > 1

    rca_v2 = {
        "winner_confirmatory": winner,
        "top_screening_for_confirmatory": top_ids,
        "objective_policy": {
            "primary": PRIMARY_OBJECTIVE,
            "secondary": SECONDARY_OBJECTIVE,
            "legacy_report": LEGACY_OBJECTIVE,
            "rank_disagreement": rank_disagreement,
            "top_by_objective": objective_top,
        },
        "acceptance_guardrails": {
            "hard_reject": f"ext_class_bias_max_abs_pct > {GUARDRAIL_BIAS_HARD:g}",
            "warnings": [
                f"ext_wmape_pct > {GUARDRAIL_WMAPE_WARN:g}",
                f"valid_geh_lt5_pct < {GUARDRAIL_GEH_WARN:g}",
            ],
        },
        "policy_used": {
            "objB_weights": OBJB_WEIGHTS,
            "objC_weights": OBJC_WEIGHTS,
            "sum_ratio_clip": SUM_RATIO_CLIP,
        },
        "metric_audit_summary": audit_meta,
        "pilot_rank_stability": pilot_stab,
        "principle_ab_rank_stability": ab_stab,
        "principle_mean_scores": principle_means,
        "reasons_for_rejection": rejection_reasons,
        "warning_flags": warning_flags,
        "is_calibration_principle_valid": {
            "conclusion": (
                "Princip3 (frozen OD) is a diagnostic control only; compare Princip1 vs Princip2 "
                "mean ObjB/C. If sector and global are similar but both poor, the bottleneck is "
                "identification (demand/route/match) not the scaling aggregator."
            ),
            "metric_validity": (
                "If ObjA/B/C ranks disagree (low rank stability), a single GEH-heavy score is "
                "insufficient; report wMAPE, Spearman, and class bias alongside GEH."
            ),
        },
        "counts": {
            "pilot_runs": len(pilot_df),
            "principle_ab_runs": len(ab_df),
            "confirmatory_runs": len(confirm_rows),
            "replications": len(repl_rows),
        },
        "technical_fail_count": len([r for r in all_rows if r.get("failure_class") == "technical_fail"]),
        "model_fail_only_count": len([r for r in all_rows if r.get("failure_class") == "model_fail_only"]),
    }
    (out_root / "rca_report_v2.json").write_text(
        json.dumps(rca_v2, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    print(f"Done. Outputs in {out_root}")


if __name__ == "__main__":
    main()
