"""Calibration loop context: shared setup, teardown, and iteration helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix

from sim.assignment import (
    build_graph,
    execute_assignment,
    fix_node_ids,
    _apply_bpr_defaults,
    resolve_daily_cap_factor_default,
    _resolve_multi_class,
)
from sim.calibration.gateway import (
    _apply_gateway_calibration,
    _build_screenline_gateway_map,
    _compute_gateway_modeled_volumes,
    _load_gateway_observed,
    _load_gateway_zone_map,
)
from sim.calibration.matching import (
    match_counts_to_links,
    match_quality_report,
    _export_matching_diagnostics,
)
from sim.calibration.metrics import (
    compute_stats,
    compute_geh,
    _compute_class_residuals,
    _supplement_class_ratios_from_screenlines,
    _apply_class_residual_correction,
    compute_extended_link_metrics,
    compute_class_volume_breakdown,
)
from sim.calibration.observed import (
    load_csd,
    load_csd_unfiltered,
    load_csd_as_link_counts,
    load_pentlogram,
    split_csd_for_calibration,
    validate_geometries_or_fail,
    _load_network_links,
)
from sim.calibration.state import CalibrationRun, CalibrationState, TRANSITIONS
from sim.io_project import get_metric_epsg, get_nested, load_config


logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _check_supply_audit(output_dir: Path, *, require: bool = True) -> Optional[Dict[str, Any]]:
    """Check that supply parameters have been audited or tuned before ODME.

    Looks for ``supply_audit.json`` or ``supply_tuning_report.json``.
    Returns the audit data if found, ``None`` otherwise.
    Raises ``RuntimeError`` if *require* is True and no audit exists.
    """
    for name in ("supply_audit.json", "supply_tuning_report.json"):
        path = output_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    if require:
        raise RuntimeError(
            "No supply audit found. Before running ODME calibration, "
            "either run 'tune-supply' or 'audit-supply' to verify that "
            "network speeds, capacities, and VDF parameters are reasonable. "
            "ODME must not compensate for uncalibrated supply-side errors. "
            "Set calibration.require_supply_audit=false to skip this check."
        )
    return None


def _check_assignment_convergence(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Verify that a converged base-year UE assignment exists.

    Returns the convergence metadata dict if found, ``None`` otherwise.
    Logs a warning if the assignment did not converge.
    """
    candidates = [
        "pre_odme_convergence.json",
        "assignment_convergence.json",
    ]
    conv_path = None
    for name in candidates:
        p = output_dir / name
        if p.exists():
            conv_path = p
            break
    if conv_path is None:
        logger.warning(
            "No assignment convergence JSON found in %s. "
            "Cannot verify that base-year assignment converged before ODME.",
            output_dir,
        )
        return None
    try:
        meta = json.loads(conv_path.read_text(encoding="utf-8"))
        if not meta.get("converged", False):
            logger.warning(
                "Base-year assignment did NOT converge (rgap=%s). "
                "ODME results may be unreliable.",
                meta.get("final_rgap"),
            )
        return meta
    except Exception:
        logger.debug("Failed to read %s", conv_path.name, exc_info=True)
        return None


def _detect_cross_screenline_collisions(
    sl_query: Dict[str, list],
    *,
    strict: bool = False,
) -> None:
    """Warn (or raise) when the same link_id appears in multiple screenlines.

    Cross-screenline overlap means the ODME objective over-weights that
    corridor because its volume is counted once per screenline.
    """
    from collections import defaultdict

    link_to_screenlines: Dict[int, List[str]] = defaultdict(list)
    for sl_name, link_tuples in sl_query.items():
        for lid, _d in link_tuples:
            link_to_screenlines[lid].append(sl_name)

    collisions = {
        lid: sls for lid, sls in link_to_screenlines.items() if len(sls) > 1
    }
    if not collisions:
        return

    lines = [
        f"  link {lid}: screenlines {sls}" for lid, sls in sorted(collisions.items())
    ]
    msg = (
        f"Cross-screenline collision: {len(collisions)} link(s) appear in "
        f"multiple screenlines. This over-weights those corridors in ODME.\n"
        + "\n".join(lines)
    )
    if strict:
        raise RuntimeError(
            msg + "\nSet calibration.screenline_dedup_strict=false to "
            "downgrade to a warning."
        )
    logger.warning(msg)


def _sr_val(sr: Any, key: str) -> float:
    """Extract a numeric value from a screenline result (dict or dataclass)."""
    if isinstance(sr, dict):
        return float(sr.get(key, 0) or 0)
    return float(getattr(sr, key, 0) or 0)


def _compute_count_weights(
    observed: np.ndarray,
    method: str = "inverse_sqrt",
) -> np.ndarray:
    """Weight vector for count-post observations.

    ``inverse_sqrt`` (default): w_a = 1 / sqrt(max(obs_a, 100)).
    Balances high- and low-volume count posts so the objective function
    is not dominated by motorway links (Aimsun OD-adj. TN, 2019).
    """
    obs = np.maximum(np.asarray(observed, dtype=np.float64), 100.0)
    if method == "inverse_sqrt":
        return 1.0 / np.sqrt(obs)
    if method == "inverse":
        return 1.0 / obs
    # uniform — plain least squares
    return np.ones_like(obs)


def _odme_objective(
    modeled: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted sum-of-squares objective  Z = sum_a w_a*(v_a - v_a^obs)^2."""
    residuals = np.asarray(modeled, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    return float(np.sum(weights * residuals ** 2))


class _CalibrationContext:
    """Shared setup / teardown for calibration loops.

    Encapsulates config loading, matrix backup/restore, observed data,
    screenline resolution, project/graph setup, and seed elasticity bounds
    so that ``run_calibration`` and ``run_odme_calibration`` can reuse
    the same initialisation logic.
    """

    def __init__(self, config_path: str | Path = "config/brno/sim.yaml") -> None:
        from sim.calibration.screenlines import load_screenlines, load_screenlines_with_auto, resolve_screenline_links, _dedup_cross_screenline_links

        self.config_path = config_path
        cfg = load_config(config_path)
        self.cfg = cfg
        self.project_dir = Path(cfg["project_path"])

        demand_cfg = cfg.get("demand") or {}
        self.calib_cfg = calib_cfg = cfg.get("calibration") or {}
        self.matrix_path = Path(demand_cfg.get("matrix_path", "data/demand/od_matrix.aem"))
        self.output_dir = Path(demand_cfg.get("output_dir", "outputs/baseline/demand"))
        _ensure_dir(self.output_dir)

        # Assignment parameters
        self.algorithm = str(calib_cfg.get("algorithm", "bfw"))
        self.max_iter_assign = int(calib_cfg.get("max_iter", 150))
        self.rgap = float(calib_cfg.get("rgap_target", 0.002))
        self.core_name = str(calib_cfg.get("core_name", "wd_daily"))

        assign_cfg = cfg.get("assignment") or {}
        gc_cfg = assign_cfg.get("generalized_cost") or {}
        gc_enabled = bool(gc_cfg.get("enabled", True))
        self.gc_field: Optional[str] = (
            str(gc_cfg.get("fixed_cost_field", "distance"))
            if gc_enabled else None
        )
        self.gc_mult: float = float(gc_cfg.get("fixed_cost_multiplier", 0.006)) if gc_enabled else 0.0
        self.gc_vot: float = float(gc_cfg.get("vot", 1.0))
        bpr_cfg = _apply_bpr_defaults(assign_cfg.get("bpr") or {})
        self.cfg_bpr: Optional[Dict[str, object]] = bpr_cfg
        mc_cfg = assign_cfg.get("multi_class") or {}
        self.cfg_multi: Optional[list] = _resolve_multi_class(mc_cfg)
        self.daily_cap_factor = resolve_daily_cap_factor_default(bpr_cfg)
        self.cores = int(assign_cfg.get("cores", 0))

        # Matching parameters
        self.buffer_m = float(calib_cfg.get("match_buffer_m", 50.0))
        self.direction_aware = bool(calib_cfg.get("match_direction_aware", True))
        self.conflict_res = str(calib_cfg.get("match_conflict_resolution", "nearest"))
        self.agg_corridor = bool(calib_cfg.get("aggregate_corridor", True))
        self.match_quality_min = float(calib_cfg.get("match_quality_min", 0.50))
        self.count_target = str(calib_cfg.get("count_target", "motor_total"))
        _COUNT_TARGET_COL = {
            "car_only": "observed_car",
            "motor_total": "observed_motor_total",
            "total": "observed_total",
        }
        self.obs_col = _COUNT_TARGET_COL.get(self.count_target, "observed_total")
        if self.count_target not in _COUNT_TARGET_COL:
            logger.warning(
                f"  WARNING: unknown count_target '{self.count_target}', "
                f"falling back to observed_total"
            )

        # Convergence
        conv_cfg = calib_cfg.get("convergence") or {}
        self.daily_conv = conv_cfg.get("daily") or {}

        # ODME / elasticity
        self.odme_cfg = calib_cfg.get("odme") or {}
        self.max_deviation = float(self.odme_cfg.get("max_deviation", 4.0))

        # Screenline factor bounds (shared by ODME and FSM Spiess updates)
        sl_factors = calib_cfg.get("screenline_factors") or {}
        self.sl_ratio_max = float(sl_factors.get("ratio_max", 5.0))
        self.sl_ratio_min = float(sl_factors.get("ratio_min", 0.2))
        self.sl_clip_min = float(sl_factors.get("clip_min", 0.5))
        self.sl_clip_max = float(sl_factors.get("clip_max", 2.0))
        self.sl_global_min = float(sl_factors.get("global_min", 0.8))
        self.sl_global_max = float(sl_factors.get("global_max", 1.25))

        # Pre-flight
        fix_node_ids(self.project_dir)
        if not self.matrix_path.exists():
            raise FileNotFoundError(f"OD matrix not found: {self.matrix_path}")

        # P0-3: Verify supply-side readiness before ODME
        require_supply_audit = bool(calib_cfg.get("require_supply_audit", True))
        self._supply_audit = _check_supply_audit(
            self.output_dir, require=require_supply_audit,
        )
        if self._supply_audit:
            logger.info("  Supply audit: verified (%s)", next(
                n for n in ("supply_audit.json", "supply_tuning_report.json")
                if (self.output_dir / n).exists()
            ))

        # P0-3: Verify base-year assignment converged
        self._assignment_convergence = _check_assignment_convergence(self.output_dir)

        # Backup / restore seed matrix — versioned by content hash
        def _file_hash(path: Path) -> str:
            h = hashlib.sha256()
            h.update(path.read_bytes())
            return h.hexdigest()[:16]

        backup = self.matrix_path.with_suffix(".aem.orig")
        current_hash = _file_hash(self.matrix_path)

        if backup.exists():
            backup_hash = _file_hash(backup)
            if backup_hash != current_hash:
                logger.warning(
                    "Seed matrix changed since last backup "
                    "(backup hash=%s, current hash=%s). "
                    "Updating .aem.orig to new seed.",
                    backup_hash, current_hash,
                )
                shutil.copy2(self.matrix_path, backup)
        else:
            shutil.copy2(self.matrix_path, backup)
            logger.info("Created seed matrix backup: %s", backup)

        reset_matrix = bool(calib_cfg.get("reset_matrix_before_run", True))
        if reset_matrix and backup.exists():
            shutil.copy2(backup, self.matrix_path)
            logger.info("  Restored OD matrix from .aem.orig")

        # Observed counts + network links
        self.links_gdf = _load_network_links(self.project_dir)
        self.count_source = str(calib_cfg.get("count_source", "csd_split"))

        if self.count_source == "csd_split":
            split_cfg = calib_cfg.get("csd_split") or {}
            csd_full = load_csd(cfg)
            calib_csd, _valid_csd = split_csd_for_calibration(
                csd_full,
                strategy=str(split_cfg.get("strategy", "alternating")),
                calib_share=float(split_cfg.get("calib_share", 0.65)),
                random_seed=int(split_cfg.get("random_seed", 42)),
            )
            self.pent = load_csd_as_link_counts(calib_csd, self.links_gdf)
            if self.pent.empty:
                raise RuntimeError(
                    "CSD split produced no link-level observations for calibration. "
                    "Check that osm_ref values match CSD road numbers."
                )
            logger.info(f"  CSD-split calibration: {len(self.pent)} link observations")
        else:
            self.pent = load_pentlogram(cfg)
            validate_geometries_or_fail(
                self.pent, name="pentlogram", expected_epsg=get_metric_epsg(self.cfg),
            )
            logger.info(f"  Pentlogram: {len(self.pent)} observed segments")

        # Screenlines (manual YAML + auto-generated from gateways/CSD)
        csd_for_auto = None
        csd_full_for_gw = None
        if self.count_source == "csd_split":
            try:
                csd_for_auto = load_csd(cfg)
            except Exception:
                pass
        try:
            csd_full_for_gw = load_csd_unfiltered(cfg)
        except Exception:
            pass
        self.screenlines = load_screenlines_with_auto(
            cfg, csd_df=csd_for_auto, csd_df_full=csd_full_for_gw,
        )
        self.sl_query: Optional[Dict[str, list]] = None
        if self.screenlines:
            self.sl_query = {}
            for sl in self.screenlines:
                resolved = resolve_screenline_links(sl, self.links_gdf)
                if resolved:
                    sl.links = resolved
                    self.sl_query[sl.name] = [(lid, d) for lid, d in resolved]
            logger.info(
                f"  Screenlines: {len(self.screenlines)} defined, "
                f"{len(self.sl_query)} with links"
            )
            _detect_cross_screenline_collisions(
                self.sl_query,
                strict=bool(calib_cfg.get("screenline_dedup_strict", False)),
            )

            if bool(calib_cfg.get("screenline_cross_dedup", True)):
                self.sl_query, dedup_log = _dedup_cross_screenline_links(
                    self.sl_query,
                )
                if dedup_log:
                    logger.info(
                        "Cross-screenline dedup removed %d link "
                        "assignment(s):", len(dedup_log),
                    )
                    for line in dedup_log:
                        logger.info(line)
                    sl_by_name = {sl.name: sl for sl in self.screenlines}
                    for sl_name, link_tuples in self.sl_query.items():
                        if sl_name in sl_by_name:
                            sl_by_name[sl_name].links = link_tuples
                    empty_sls = [
                        name for name, links in self.sl_query.items()
                        if not links
                    ]
                    for name in empty_sls:
                        del self.sl_query[name]
                        logger.warning(
                            "Screenline '%s' has no links after "
                            "cross-dedup, removing", name,
                        )
                    self.screenlines = [
                        sl for sl in self.screenlines
                        if sl.name not in empty_sls
                    ]

        # Open matrix (stays open across iterations)
        self.mat = AequilibraeMatrix()
        self.mat.load(str(self.matrix_path))
        self.mat.computational_view([self.core_name])

        # Seed matrix for elasticity bounds
        seed = self.mat.matrix[self.core_name][:, :].copy().astype(np.float64)
        self.seed_lower = seed / self.max_deviation
        self.seed_upper = seed * self.max_deviation
        self.seed_lower[seed <= 0] = 0.0
        self.seed_upper[seed <= 0] = 0.0

        # Observation density check
        n_zones = len(self.mat.index[:])
        n_obs = len(self.pent) if self.pent is not None and not self.pent.empty else 0
        obs_ratio = n_obs / max(n_zones, 1)
        min_ratio = float(calib_cfg.get("min_obs_per_zone", 0.5))
        logger.info(
            "  Observation density: %d observations for %d zones "
            "(%.2f obs/zone, threshold %.2f)",
            n_obs, n_zones, obs_ratio, min_ratio,
        )
        if obs_ratio < min_ratio:
            logger.warning(
                "Observation density %.2f obs/zone is below threshold %.2f. "
                "Calibration may be weakly identified. Consider adding "
                "more count sources or corridor observations.",
                obs_ratio, min_ratio,
            )

        # Swap closures for the calibration period before opening the project
        bc_cfg = cfg.get("baseline_closures") or {}
        calib_period = bc_cfg.get("calibration_period")
        if bc_cfg.get("enabled", False) and calib_period:
            from sim.network.closures import swap_db_closures

            swap_db_closures(config_path, measurement_period=calib_period)
            logger.info("  Closures swapped to calibration period: %s", calib_period)

        # AequilibraE project + graph (reused across iterations)
        self.project = Project()
        self.project.open(str(self.project_dir))
        self.cfg_assignment = assign_cfg
        self.cached_graph = build_graph(
            self.project, self.mat,
            bpr_parameters=self.cfg_bpr,
            assignment_cfg=assign_cfg,
        )

        # Gateway calibration — fallbacks match defaults.py
        gw_cal_cfg = calib_cfg.get("gateway_calibration") or {}
        self.gw_cal_enabled = bool(gw_cal_cfg.get("enabled", True))
        self.gw_cal_damping = float(gw_cal_cfg.get("damping", 0.18))
        self.gw_cal_min_factor = float(gw_cal_cfg.get("min_factor", 0.70))
        self.gw_cal_max_factor = float(gw_cal_cfg.get("max_factor", 1.40))
        self.gw_rebase_seed_bounds = bool(gw_cal_cfg.get("rebase_seed_bounds", True))
        self.gw_zone_map: Dict[str, np.ndarray] = {}
        self.gw_observed: Dict[str, float] = {}

        self.sl_gw_map: Dict[str, str] = {}
        if self.gw_cal_enabled:
            try:
                self.gw_zone_map = _load_gateway_zone_map(cfg, self.mat.index[:])
                gw_names = set(self.gw_zone_map.keys())
                self.sl_gw_map = _build_screenline_gateway_map(
                    self.screenlines, gw_names,
                )
                self.gw_observed = _load_gateway_observed(
                    cfg,
                    screenlines_list=self.screenlines,
                    gateway_zone_names=gw_names,
                )
                if self.gw_zone_map and self.gw_observed:
                    active = set(self.gw_zone_map) & set(self.gw_observed)
                    logger.info(
                        f"  Gateway calibration: {len(active)} gateways with observed data "
                        f"({', '.join(sorted(active))})"
                    )
                    if self.sl_gw_map:
                        logger.info(
                            f"  Screenline→gateway map: {len(self.sl_gw_map)} entries "
                            f"({', '.join(f'{k}→{v}' for k, v in sorted(self.sl_gw_map.items()))})"
                        )
                else:
                    logger.info(
                        "  Gateway calibration: no matching zone/observed data — disabled"
                    )
                    self.gw_cal_enabled = False
            except Exception:
                logger.exception("Gateway calibration init failed")
                self.gw_cal_enabled = False

        # Tracking state (encapsulated in CalibrationRun for FSM audit)
        self.run = CalibrationRun()
        self.best_Z: float = float("inf")
        self.best_demand: Optional[np.ndarray] = None
        self.best_iteration: int = 0
        self.history: List[Dict[str, Any]] = []

    # -- FSM state management --

    def _transition(self, new_state: CalibrationState) -> None:
        """Transition FSM to *new_state* with validation and audit logging."""
        old = self.run.state
        allowed = TRANSITIONS.get(old, set())
        if new_state not in allowed:
            logger.warning(
                "FSM: invalid transition %s → %s (allowed: %s)",
                old.name, new_state.name,
                ", ".join(s.name for s in sorted(allowed, key=lambda s: s.value)) or "none",
            )
        self.run.state = new_state
        logger.debug("FSM: %s → %s", old.name, new_state.name)

    # -- Shared helpers for the iteration body --

    def run_assignment(self, *, save_skims: bool = False) -> tuple:
        """Execute one equilibrium assignment pass.

        Returns ``(vol_df, skims, sl_matrices)`` — convergence metadata
        from the 4th return value of ``execute_assignment`` is logged
        but not propagated to callers that expect a 3-tuple.
        """
        vol_df, skims, sl_matrices, _conv_meta = execute_assignment(
            self.project, self.mat,
            algorithm=self.algorithm,
            max_iter=self.max_iter_assign,
            rgap_target=self.rgap,
            save_skims=save_skims,
            select_links=self.sl_query,
            bpr_parameters=self.cfg_bpr,
            fixed_cost_field=self.gc_field,
            fixed_cost_multiplier=self.gc_mult,
            vot=self.gc_vot,
            multi_class=self.cfg_multi,
            graph=self.cached_graph,
            cores=self.cores,
        )
        return vol_df, skims, sl_matrices

    def run_iteration_assignment(self, *, save_skims: bool = False) -> tuple:
        """Assignment + multi-class volume sum + vol_col detection.

        Returns (vol_df, skims, sl_matrices, vol_col, total_vol).
        """
        self._transition(CalibrationState.ASSIGN)
        self.run.iteration += 1
        vol_df, skims, sl_matrices = self.run_assignment(save_skims=save_skims)
        from sim._metrics import resolve_volume_column
        vol_df, vol_col = resolve_volume_column(vol_df)
        total_vol = float(vol_df[vol_col].sum()) if vol_col else 0.0
        return vol_df, skims, sl_matrices, vol_col, total_vol

    def match_and_filter(self, vol_df: pd.DataFrame, vol_col: Optional[str], *, iteration: int = 1):
        """Match counts to links, exclusion filter, export diagnostics on first iter.

        Returns (matched, valid, compare_col).
        """
        self._transition(CalibrationState.MATCH)
        links_with_vol = self.links_gdf.copy()
        if vol_col and "link_id" in vol_df.columns:
            links_with_vol = links_with_vol.merge(
                vol_df[["link_id", vol_col]], on="link_id", how="left",
            )
        matched = match_counts_to_links(
            self.pent, links_with_vol, buffer_m=self.buffer_m,
            direction_aware=self.direction_aware,
            conflict_resolution=self.conflict_res,
            aggregate_corridor=self.agg_corridor,
            vol_col=vol_col,
            match_quality_min=self.match_quality_min,
            skip_exclusion=(iteration > 1),
        )
        if iteration == 1:
            if "_excluded" in matched.columns and "objectid" in matched.columns:
                excl_ids = set(matched.loc[matched["_excluded"], "objectid"].dropna().astype(int))
                if excl_ids:
                    n_before = len(self.pent)
                    self.pent = self.pent[~self.pent["objectid"].isin(excl_ids)].copy()
                    logger.info(f"  Pre-filter: removed {n_before - len(self.pent)} excluded stations")
            _export_matching_diagnostics(
                matched,
                "_corridor_volume" if "_corridor_volume" in matched.columns else vol_col,
                self.obs_col, self.output_dir,
            )
        compare_col = "_corridor_volume" if "_corridor_volume" in matched.columns else vol_col
        valid = matched.dropna(subset=[compare_col, self.obs_col])
        valid = valid[valid[self.obs_col] > 0].copy()
        if "_excluded" in valid.columns:
            valid = valid[~valid["_excluded"]].copy()
        return matched, valid, compare_col

    def compute_objective_and_stats(self, valid: pd.DataFrame, compare_col: str, weight_method="inverse_sqrt"):
        """Compute Z objective and link-fit statistics.

        Returns (Z, stats).
        """
        self._transition(CalibrationState.EVALUATE)
        mod_all = valid[compare_col].values.astype(np.float64)
        obs_all = valid[self.obs_col].values.astype(np.float64)
        w_all = _compute_count_weights(obs_all, method=weight_method)
        Z = _odme_objective(mod_all, obs_all, w_all)
        stats = compute_stats(mod_all, obs_all, daily_capacity_factor=self.daily_cap_factor)
        return Z, stats

    def update_best_state(self, Z: float, iteration: int) -> None:
        """Track best Z and save demand snapshot if improved."""
        if Z < self.best_Z:
            self.best_Z = Z
            self.best_demand = self.mat.matrix[self.core_name][:, :].copy()
            self.best_iteration = iteration

    def evaluate_and_log_screenlines(self, vol_df: pd.DataFrame, matched: gpd.GeoDataFrame, vol_col: Optional[str]):
        """Evaluate screenlines and return (sl_results_dict, max_pct_dev)."""
        from sim.calibration.screenlines import evaluate_all_screenlines

        sl_results: Dict[str, Any] = {}
        max_sl_pct_dev = 0.0
        if self.screenlines and vol_col:
            sl_res = evaluate_all_screenlines(
                self.screenlines, vol_df, matched, vol_col, self.obs_col, self.links_gdf,
            )
            for sn, sr in sl_res.items():
                sl_results[sn] = sr.to_dict()
                if sr.observed_total > 0 and np.isfinite(sr.ratio):
                    dev = abs(sr.ratio - 1.0) * 100.0
                    max_sl_pct_dev = max(max_sl_pct_dev, dev)
                    logger.info(
                        f"  SL '{sn}': mod={sr.modeled_total:,.0f} "
                        f"obs={sr.observed_total:,.0f} ratio={sr.ratio:.2f} GEH={sr.geh:.1f}"
                    )
        return sl_results, max_sl_pct_dev

    def mark_stalled(self, reason: str) -> None:
        """Record stall/convergence reason for FSM audit trail."""
        self.run.stop_reason = reason
        self._transition(CalibrationState.STALLED)

    def check_convergence(self, stats: Dict[str, Any], max_sl_pct_dev: float, sl_results: Dict[str, Any]) -> bool:
        """Unified daily convergence check.  Returns True if converged.

        GEH is intentionally excluded for daily models — it was designed for
        hourly flows and its Poisson assumption breaks down at daily volumes.
        Daily convergence relies on R², slope, %RMSE, bias, and screenline
        deviation instead.
        """
        daily_conv = self.daily_conv
        cur_r2 = float(stats.get("r2") or 0.0)
        cur_slope = float(stats.get("slope") or 0.0)
        cur_prmse = float(stats.get("pct_rmse") or 999.0)
        cur_bias = abs(float(stats.get("bias_pct") or 999.0))
        daily_r2_target = float(daily_conv.get("r2_target", 0.80))
        daily_slope_range = daily_conv.get("slope_range", [0.85, 1.15])
        daily_pct_rmse_max = float(daily_conv.get("pct_rmse_max", 35.0))
        daily_bias_max = float(daily_conv.get("bias_abs_max_pct", 15.0))
        daily_sl_max_dev = float(daily_conv.get("screenline_max_pct_deviation", 15.0))
        return (
            cur_r2 >= daily_r2_target
            and float(daily_slope_range[0]) <= cur_slope <= float(daily_slope_range[1])
            and cur_prmse <= daily_pct_rmse_max
            and cur_bias <= daily_bias_max
            and (max_sl_pct_dev <= daily_sl_max_dev if sl_results else True)
        )

    def apply_global_residual(
        self,
        demand: np.ndarray,
        valid: pd.DataFrame,
        compare_col: str,
        *,
        damping: float,
    ) -> None:
        """Global obs/mod residual correction in-place."""
        obs_all = valid[self.obs_col].values.astype(np.float64)
        mod_all = valid[compare_col].values.astype(np.float64)
        sum_obs = float(obs_all.sum())
        sum_mod = float(mod_all.sum())
        if sum_mod > 0 and sum_obs > 0:
            global_ratio = sum_obs / sum_mod
            global_factor = 1.0 + damping * (global_ratio - 1.0)
            global_factor = float(np.clip(global_factor, self.sl_global_min, self.sl_global_max))
            if abs(global_factor - 1.0) > 0.003:
                demand *= global_factor
                np.clip(demand, self.seed_lower, self.seed_upper, out=demand)
                np.maximum(demand, 0.0, out=demand)
                logger.info(
                    f"  Global residual: obs/mod={global_ratio:.3f} "
                    f"→ factor={global_factor:.4f}  demand={demand.sum():,.0f}"
                )

    def apply_class_residual(
        self,
        demand: np.ndarray,
        vol_df: pd.DataFrame,
        valid: pd.DataFrame,
        compare_col: str,
        sl_matrices: Any,
        sl_results: Dict[str, Any],
        *,
        damping: float,
    ) -> None:
        """Per-road-class residual correction in-place."""
        class_res_min_counts = int(self.odme_cfg.get("class_residual_min_counts", 3))
        class_ratios = _compute_class_residuals(
            valid, self.obs_col, compare_col, min_counts=class_res_min_counts,
        )
        class_ratios = _supplement_class_ratios_from_screenlines(
            sl_results, self.links_gdf, class_ratios,
        )
        if class_ratios:
            cr_logs = _apply_class_residual_correction(
                demand, vol_df, self.links_gdf, class_ratios,
                sl_matrices, sl_results,
                damping=damping,
                clip_min=self.sl_clip_min,
                clip_max=self.sl_clip_max,
                reference_total=getattr(self, "_post_gateway_total", None),
            )
            np.clip(demand, self.seed_lower, self.seed_upper, out=demand)
            np.maximum(demand, 0.0, out=demand)
            if cr_logs:
                logger.info("  Class residual corrections:")
                for crl in cr_logs:
                    logger.info(f"    {crl}")

    def apply_gateway_and_rebase(self, demand: np.ndarray, vol_df: pd.DataFrame, vol_col: Optional[str], outer_it: int) -> None:
        """Gateway calibration + seed rebase on first iter. In-place on demand."""
        if not self.gw_cal_enabled or not vol_col:
            return
        gw_modeled = _compute_gateway_modeled_volumes(
            vol_df, self.screenlines, vol_col, sl_gw_map=self.sl_gw_map,
        )
        gw_corrections = _apply_gateway_calibration(
            demand, self.gw_zone_map, self.gw_observed, gw_modeled,
            damping=self.gw_cal_damping,
            min_factor=self.gw_cal_min_factor,
            max_factor=self.gw_cal_max_factor,
            seed_lower=self.seed_lower,
            seed_upper=self.seed_upper,
        )
        if gw_corrections:
            logger.info(f"  Gateway calibration ({len(gw_corrections)}):")
            for gc_line in gw_corrections:
                logger.info(f"    {gc_line}")
        self._post_gateway_total = float(demand.sum())
        if outer_it == 1 and self.gw_rebase_seed_bounds:
            rebased = demand.copy()
            self.seed_lower = rebased / self.max_deviation
            self.seed_upper = rebased * self.max_deviation
            self.seed_lower[rebased <= 0] = 0.0
            self.seed_upper[rebased <= 0] = 0.0
            logger.info("  Seed bounds rebased after first gateway calibration")

    def apply_demand_cap_and_save(self, demand: np.ndarray, total_demand: float, *, max_change_pct: float) -> None:
        """Cap per-iteration demand change and write back to matrix."""
        self._transition(CalibrationState.PERSIST_CHECKPOINT)
        iter_total = float(demand.sum())
        if total_demand > 0 and max_change_pct > 0:
            change_pct = (iter_total - total_demand) / total_demand * 100
            if abs(change_pct) > max_change_pct:
                cap_factor = total_demand * (
                    1.0 + np.sign(change_pct) * max_change_pct / 100.0
                ) / max(iter_total, 1.0)
                demand *= cap_factor
                logger.info(
                    f"  Iter demand cap: {change_pct:+.1f}% exceeds "
                    f"±{max_change_pct:.0f}%, clamped to {demand.sum():,.0f}"
                )
        self.mat.matrix[self.core_name][:, :] = demand
        self.mat.save()
        logger.info(f"  Matrix saved. Total demand: {demand.sum():,.0f}")

    def restore_best_and_close(self) -> None:
        """Restore best-Z demand matrix, save, and close resources."""
        self._transition(CalibrationState.FINALIZE_BEST)
        try:
            if self.best_demand is not None:
                self.mat.matrix[self.core_name][:, :] = self.best_demand
                self.mat.save()
                logger.info(
                    f"  Restored best matrix from iteration {self.best_iteration} "
                    f"(Z={self.best_Z:,.1f})"
                )
            self.mat.close()
            self.project.close()
        except Exception:
            logger.exception("restore_best_and_close failed")

    def finalize_best_state(self) -> Optional[pd.DataFrame]:
        """Run a final assignment on the restored best-demand matrix.

        Must be called *after* ``restore_best_and_close()`` so that the
        on-disk matrix already contains the best demand.  Returns the
        vol_df from this final assignment so the caller can persist
        artifacts that are consistent with the best OD matrix.
        """
        if self.best_demand is None:
            self.run.stop_reason = self.run.stop_reason or "no_improvement"
            return None

        mat = AequilibraeMatrix()
        mat.load(str(self.matrix_path))
        mat.computational_view([self.core_name])

        project = Project()
        project.open(str(self.project_dir))
        try:
            graph = build_graph(
                project, mat,
                bpr_parameters=self.cfg_bpr,
                assignment_cfg=self.cfg_assignment,
            )
            vol_df, _skims, _sl, conv_meta = execute_assignment(
                project, mat,
                algorithm=self.algorithm,
                max_iter=self.max_iter_assign,
                rgap_target=self.rgap,
                bpr_parameters=self.cfg_bpr,
                fixed_cost_field=self.gc_field,
                fixed_cost_multiplier=self.gc_mult,
                vot=self.gc_vot,
                multi_class=self.cfg_multi,
                graph=graph,
                cores=self.cores,
            )
            logger.info(
                f"  Final assignment on best-state demand completed "
                f"(iteration {self.best_iteration})"
            )
            post_conv_path = self.output_dir / "post_odme_convergence.json"
            post_conv_path.write_text(
                json.dumps(conv_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"  Post-ODME convergence: {post_conv_path}")
            self._transition(CalibrationState.FINALIZE_SUCCESS)
            return vol_df
        finally:
            mat.close()
            project.close()

    def save_results(self, vol_df: Optional[pd.DataFrame] = None) -> None:
        """Persist assignment results parquet and report path."""
        if self.history and vol_df is not None:
            results_path = self.output_dir / "assignment_results.parquet"
            vol_df.to_parquet(str(results_path), index=False)
            logger.info(f"  Saved: {results_path}")


__all__ = [
    "_CalibrationContext",
    "_compute_count_weights",
    "_ensure_dir",
    "_odme_objective",
    "_sr_val",
    "match_quality_report",
    "compute_geh",
    "compute_extended_link_metrics",
    "compute_class_volume_breakdown",
    "get_nested",
]
