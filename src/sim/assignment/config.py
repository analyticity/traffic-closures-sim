"""BPR and multi-class methodology defaults, YAML merge helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sim.defaults import SIM_DEFAULTS

_BPR_DEFAULTS = SIM_DEFAULTS["assignment"]["bpr"]
_DEFAULT_BPR_BY_LINK_TYPE: Dict[str, Dict[str, float]] = _BPR_DEFAULTS["by_link_type"]
_DEFAULT_DAILY_CAP_FACTOR: Dict[str, Any] = _BPR_DEFAULTS["daily_capacity_factor"]
_DEFAULT_MULTI_CLASS = SIM_DEFAULTS["assignment"]["multi_class"]["classes"]


def _apply_bpr_defaults(bpr_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge YAML BPR overrides on top of built-in defaults."""
    base: Dict[str, Any] = {
        "vdf_function": _BPR_DEFAULTS["vdf_function"],
        "per_link": _BPR_DEFAULTS["per_link"],
        "alpha_default": _BPR_DEFAULTS["alpha_default"],
        "beta_default": _BPR_DEFAULTS["beta_default"],
        "daily_capacity_factor": dict(_DEFAULT_DAILY_CAP_FACTOR),
        "by_link_type": {k: dict(v) for k, v in _DEFAULT_BPR_BY_LINK_TYPE.items()},
    }
    if not bpr_params:
        return base
    merged = dict(base)
    for key, val in bpr_params.items():
        if key == "by_link_type" and isinstance(val, dict):
            merged_lt = {k: dict(v) for k, v in _DEFAULT_BPR_BY_LINK_TYPE.items()}
            for lt, lt_val in val.items():
                if lt in merged_lt and isinstance(lt_val, dict):
                    merged_lt[lt].update(lt_val)
                else:
                    merged_lt[lt] = lt_val
            merged["by_link_type"] = merged_lt
        elif key == "daily_capacity_factor" and isinstance(val, dict):
            merged_dcf = dict(_DEFAULT_DAILY_CAP_FACTOR)
            for dk, dv in val.items():
                if dk == "by_link_type" and isinstance(dv, dict):
                    merged_dcf_lt = dict(_DEFAULT_DAILY_CAP_FACTOR.get("by_link_type", {}))
                    merged_dcf_lt.update(dv)
                    merged_dcf["by_link_type"] = merged_dcf_lt
                else:
                    merged_dcf[dk] = dv
            merged["daily_capacity_factor"] = merged_dcf
        else:
            merged[key] = val
    return merged


def _resolve_multi_class(mc_cfg: Optional[Dict[str, Any]]) -> Optional[list]:
    """Return multi-class list, using built-in defaults when YAML omits classes."""
    if mc_cfg is None:
        mc_cfg = {}
    if not mc_cfg.get("enabled", True):
        return None
    return list(mc_cfg.get("classes", _DEFAULT_MULTI_CLASS))


def multiclass_matrix_core_status(
    mat_cores: Sequence[str],
    multi_class: Optional[list],
) -> Tuple[bool, List[str], List[str]]:
    """Return whether multi-class UE can run with the given matrix cores.

    Mirrors the gate in ``execute_assignment``: every configured class
    ``core`` must appear in ``mat_cores`` (typically ``mat.names`` on an
    open ``AequilibraeMatrix``).

    Returns
    -------
    enabled
        True when *multi_class* is non-empty and all class cores exist.
    required
        Ordered list of class core names from *multi_class*.
    missing
        Subset of *required* not found in *mat_cores*.
    """
    if not multi_class:
        return False, [], []
    have = {str(x) for x in mat_cores}
    required: List[str] = []
    for c in multi_class:
        core = c.get("core")
        if core is not None:
            required.append(str(core))
    missing = [c for c in required if c not in have]
    return (not missing, required, missing)


def resolve_daily_cap_factor_default(bpr_cfg: dict) -> float:
    """Extract the scalar default from ``daily_capacity_factor`` (which may be a dict or float)."""
    dcf_raw = bpr_cfg.get("daily_capacity_factor", 10.0)
    if isinstance(dcf_raw, dict):
        return float(dcf_raw.get("default", 10.0))
    return float(dcf_raw)
