"""Temporal traffic profiles learned from CSD2020 data.

Provides day-type classification, day factors, and period shares
derived from observed traffic counts — not hardcoded.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sim.io_project import load_config

# ---------------------------------------------------------------------------
# Day classification
# ---------------------------------------------------------------------------

# Czech public holidays (recurring, month-day)
_HOLIDAYS_MD = [
    (1, 1), (5, 1), (5, 8), (7, 5), (7, 6),
    (9, 28), (10, 28), (11, 17), (12, 24), (12, 25), (12, 26),
]


def classify_day(d: date | str) -> str:
    """Return day type: 'workday', 'saturday', 'sunday', or 'holiday'."""
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    if (d.month, d.day) in _HOLIDAYS_MD:
        return "holiday"
    wd = d.weekday()
    if wd < 5:
        return "workday"
    if wd == 5:
        return "saturday"
    return "sunday"


# ---------------------------------------------------------------------------
# Learn profile from CSD2020
# ---------------------------------------------------------------------------

def _classify_csd_road(sil: str) -> str:
    s = str(sil).strip().upper()
    if s.startswith("D"):
        return "motorway"
    try:
        num = int(s.replace("M", ""))
        if num < 100:
            return "primary"
        if num < 400:
            return "secondary"
        return "tertiary"
    except ValueError:
        return "primary" if "M" in s else "other"


def learn_day_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute day-type factors and period shares from CSD2020.

    Returns a profile dict and saves it to ``temporal_profile.json``.
    """
    cache_dir = Path(cfg.get("datasets", {}).get("cache_dir", "data/cache"))
    parquet = cache_dir / "v2_csd2025.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"CSD2020 parquet not found: {parquet}. Run fetch-data first.")

    df = pd.read_parquet(str(parquet))
    if "kk" in df.columns:
        df = df[df["kk"].astype(str).str.contains("064", na=False)].copy()

    num_cols = ["o", "ipd_o", "ivd_o", "sv", "ipd_sv", "ivd_sv",
                "is_den", "is_ve_er", "is_noc", "sil"]
    for c in num_cols:
        if c in df.columns and c != "sil":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # --- Global day factors (cars) ---
    valid = df[df["o"] > 0].copy()
    workday_factor = float((valid["ipd_o"] / valid["o"]).mean()) if len(valid) else 1.07
    weekend_factor = float((valid["ivd_o"] / valid["o"]).mean()) if len(valid) else 0.83

    # Saturday vs Sunday: CSD doesn't separate them; approximate from typical splits
    saturday_factor = round(weekend_factor * 1.10, 3)
    sunday_factor = round(weekend_factor * 0.90, 3)
    holiday_factor = round(sunday_factor * 0.85, 3)

    day_factors = {
        "workday": round(workday_factor, 3),
        "saturday": saturday_factor,
        "sunday": sunday_factor,
        "holiday": holiday_factor,
    }

    # --- Per road-class factors ---
    if "sil" in df.columns:
        df["road_class"] = df["sil"].apply(_classify_csd_road)
    else:
        df["road_class"] = "other"

    by_class: Dict[str, Dict[str, float]] = {}
    for rc in df["road_class"].unique():
        sub = df[(df["road_class"] == rc) & (df["o"] > 0)]
        if len(sub) < 5:
            continue
        wf = float((sub["ipd_o"] / sub["o"]).mean())
        wef = float((sub["ivd_o"] / sub["o"]).mean())
        by_class[rc] = {"workday": round(wf, 3), "weekend": round(wef, 3)}

    # --- Day period shares (den / vecer / noc) ---
    period_df = df[["is_den", "is_ve_er", "is_noc"]].copy()
    total = period_df.sum(axis=1)
    mask = total > 0
    day_share = float((period_df.loc[mask, "is_den"] / total[mask]).mean()) if mask.any() else 0.785
    evening_share = float((period_df.loc[mask, "is_ve_er"] / total[mask]).mean()) if mask.any() else 0.137
    night_share = float((period_df.loc[mask, "is_noc"] / total[mask]).mean()) if mask.any() else 0.078

    profile = {
        "source": "CSD2020_JMK",
        "sections_used": int(len(valid)),
        "day_factors": day_factors,
        "day_factors_by_road_class": by_class,
        "day_period_shares": {
            "day": round(day_share, 3),
            "evening": round(evening_share, 3),
            "night": round(night_share, 3),
        },
    }

    out_dir = Path(cfg.get("demand", {}).get("output_dir", "outputs/baseline/demand"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "temporal_profile.json"
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Temporal profile saved: {out_path}")
    print(f"  Day factors: {day_factors}")
    print(f"  Period shares: day={day_share:.1%} evening={evening_share:.1%} night={night_share:.1%}")
    print(f"  Road classes: {list(by_class.keys())}")

    return profile


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def load_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load the learned temporal profile from disk."""
    out_dir = Path(cfg.get("demand", {}).get("output_dir", "outputs/baseline/demand"))
    path = out_dir / "temporal_profile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"temporal_profile.json not found at {path}. Run: python run.py learn-profile"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_day_factor(d: date | str, profile: Dict[str, Any]) -> float:
    """Return the traffic volume multiplier for a given date."""
    day_type = classify_day(d)
    return float(profile["day_factors"].get(day_type, 1.0))


def get_period_share(period: str, profile: Dict[str, Any]) -> float:
    """Return the share of daily traffic for a period (day/evening/night/daily)."""
    if period == "daily":
        return 1.0
    return float(profile["day_period_shares"].get(period, 1.0))


def get_combined_factor(d: date | str, period: str, profile: Dict[str, Any]) -> float:
    """Return day_factor * period_share for a date + period combination."""
    return get_day_factor(d, profile) * get_period_share(period, profile)


def get_demand_period_shares(profile: Dict[str, Any]) -> Dict[str, float]:
    """Map demand model periods (am, ip, pm, ev) to shares of daily traffic.

    Derived from CSD2020 day/evening/night breakdown plus typical
    intra-day distribution for Czech urban networks.
    """
    shares = profile.get("day_period_shares", {})
    day_s = float(shares.get("day", 0.785))
    eve_s = float(shares.get("evening", 0.137))
    night_s = float(shares.get("night", 0.078))

    return {
        "am": round(day_s * 0.35, 4),
        "ip": round(day_s * 0.40, 4),
        "pm": round(day_s * 0.25, 4),
        "ev": round(eve_s + night_s, 4),
        "daily": 1.0,
    }


def day_info(d: date | str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return full info for a date (type, factor, period shares)."""
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    day_type = classify_day(d)
    factor = get_day_factor(d, profile)
    return {
        "date": d.isoformat(),
        "weekday": d.strftime("%A"),
        "day_type": day_type,
        "day_factor": factor,
        "period_shares": profile["day_period_shares"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_learn_profile(config_path: str | Path = "config/sim.yaml") -> None:
    cfg = load_config(config_path)
    print("=== LEARN TEMPORAL PROFILE ===")
    learn_day_profile(cfg)
