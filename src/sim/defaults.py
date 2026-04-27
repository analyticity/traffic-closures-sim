"""Centralized pipeline defaults.

Every configurable parameter has a default value here, organized into three
dicts that mirror the YAML config structure:

* ``SIM_DEFAULTS``          -- mirrors ``sim.yaml``
* ``LOCALE_DEFAULTS``       -- mirrors ``locale.yaml``
* ``NETWORK_NORM_DEFAULTS`` -- mirrors ``network_normalization.yaml``

When ``load_config`` / ``load_locale`` load a YAML file, the user values are
deep-merged **on top of** these defaults.  This means:

* Config YAML files can stay minimal (only city-specific overrides).
* Any value can still be overridden from YAML when a city needs it.
* All tuneable knobs are visible and documented in one place.

Values were originally in the old unified ``config/sim.yaml`` (Brno reference
config) and in per-module ``_DEFAULT_*`` constants.  Czech-Republic national
defaults (speeds, holidays, road classification) apply unless the locale
file overrides them.
"""
from __future__ import annotations

from typing import Any, Dict

# ═══════════════════════════════════════════════════════════════════════════
#  SIM_DEFAULTS  –  mirrors sim.yaml
# ═══════════════════════════════════════════════════════════════════════════

SIM_DEFAULTS: Dict[str, Any] = {

    "crs_epsg": 5514,

    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "fallback_map_center": [49.75, 15.47],
    },

    # ------------------------------------------------------------------
    #  Network
    # ------------------------------------------------------------------
    "network": {
        "experiment_profile": "baseline",
        "drivable_network": {
            "enabled": True,
            "require_mode_car": True,
        },
        "isolated_components": {
            "enabled": True,
        },
        "map_export": {
            "dpi": 400,
            "figsize": [15.0, 15.0],
            "palette": {
                "figure": "#FFFFFF",
                "title": "#000000",
                "links_before": "#70747D",
                "links_after": "#545454",
                "bbox": "#3A4442",
            },
        },
    },

    # ------------------------------------------------------------------
    #  Zoning
    # ------------------------------------------------------------------
    "zoning": {
        "external_gateways": {
            "boundary_buffer_m": 1000.0,
            "min_gateway_separation_m": 800,
            "max_anchor_distance_m": 2000,
            "allowed_link_types": [
                "motorway", "motorway_link",
                "trunk", "trunk_link",
                "primary", "primary_link",
                "secondary", "secondary_link",
            ],
            "auto_discover": {
                "enabled": True,
                "link_types": ["secondary", "secondary_link"],
                "min_lanes": 1,
                "max_gateways": 30,
            },
            "export_lookup": True,
        },
    },

    # ------------------------------------------------------------------
    #  Supernetwork
    # ------------------------------------------------------------------
    "supernetwork": {
        "place_centroids_crs_epsg": 4326,
        "contract_graph": True,
        "contract_exclude_degree_leq": 2,
        "national_network": {
            "highway_types": [
                "motorway", "motorway_link",
                "trunk", "trunk_link",
                "primary", "primary_link",
                "secondary", "secondary_link",
            ],
        },
        "eligible_gateway_types": [
            "motorway", "motorway_link",
            "trunk", "trunk_link",
            "primary", "primary_link",
        ],
        "relation_filter": {
            "detour_ratio_max": 1.25,
            "max_extra_minutes": 18.0,
            "allow_same_gateway_pair": False,
        },
    },

    # ------------------------------------------------------------------
    #  Assignment
    # ------------------------------------------------------------------
    "assignment": {
        "cores": 0,

        "warm_skim_pass": {
            "algorithm": "bfw",
            "max_iter": 30,
            "rgap_target": 0.01,
        },

        "generalized_cost": {
            "enabled": True,
            "fixed_cost_field": "distance",
            "fixed_cost_multiplier": 0.005,
            "vot": 1.0,
        },

        "bpr": {
            "vdf_function": "BPR",
            "per_link": True,
            "alpha_default": 0.85,
            "beta_default": 4.0,
            "daily_capacity_factor": {
                "default": 10.0,
                "by_link_type": {
                    "motorway": 14.0,
                    "motorway_link": 11.1,
                    "trunk": 11.1,
                    "trunk_link": 10.0,
                    "primary": 10.0,
                    "primary_link": 10.0,
                    "secondary": 10.0,
                    "secondary_link": 10.0,
                    "tertiary": 9.1,
                    "tertiary_link": 9.1,
                    "residential": 8.3,
                    "living_street": 8.3,
                    "service": 8.3,
                    "unclassified": 9.1,
                },
            },
            "by_link_type": {
                "motorway":       {"alpha": 0.45, "beta": 4.0},
                "motorway_link":  {"alpha": 0.55, "beta": 4.0},
                "trunk":          {"alpha": 0.70, "beta": 4.0},
                "trunk_link":     {"alpha": 0.75, "beta": 4.0},
                "primary":        {"alpha": 0.90, "beta": 4.0},
                "primary_link":   {"alpha": 0.95, "beta": 4.0},
                "secondary":      {"alpha": 0.95, "beta": 4.0},
                "secondary_link": {"alpha": 0.95, "beta": 4.0},
                "tertiary":       {"alpha": 0.95, "beta": 4.0},
                "tertiary_link":  {"alpha": 0.95, "beta": 4.0},
                "residential":    {"alpha": 1.20, "beta": 4.0},
                "unclassified":   {"alpha": 0.95, "beta": 4.0},
                "living_street":  {"alpha": 1.50, "beta": 4.0},
                "service":        {"alpha": 1.50, "beta": 4.0},
            },
        },

        "multi_class": {
            "enabled": True,
            "classes": [
                {"name": "local",   "core": "wd_daily_local",             "vot": 1.0, "pce": 1.0},
                {"name": "through", "core": "wd_daily_external_through",  "vot": 2.0, "pce": 1.5},
            ],
        },
    },

    # ------------------------------------------------------------------
    #  Demand
    # ------------------------------------------------------------------
    "demand": {
        "matrix_name": "demand",

        "sldb": {
            "include_lokalizace": ["0_na_adrese_OP", "1_meziobecni"],
            "only_internal_pairs": False,
            "external_processing": {
                "enabled": True,
                "use_full_cr_dataset": True,
                "use_external_internal": True,
                "use_internal_external": True,
                "use_through_traffic": True,
                "drop_external_external_outside_model": True,
                "through_traffic_scale": 0.50,
                "external_commuting_scale": 0.65,
            },
        },

        "conversion": {
            "work":   {"car_share": 0.60, "occupancy": 1.25, "trips_per_person": 2.0},
            "school": {"car_share": 0.25, "occupancy": 1.30, "trips_per_person": 2.0},
        },

        "segments": {
            "other": {
                "source": "gravity",
                "trip_rate": 1.0,
                "car_share": 0.35,
                "occupancy": 1.50,
                "beta": 0.00030,
            },
            "external_through": {
                "enabled": False,
            },
        },

        "time_slices": {
            "periods": ["am", "ip", "pm", "ev"],
            "weekday": {
                "work": {
                    "outbound": {"am": 0.80, "ip": 0.20},
                    "return":   {"pm": 0.80, "ev": 0.20},
                },
                "school": {
                    "outbound": {"am": 0.90, "ip": 0.10},
                    "return":   {"pm": 0.70, "ev": 0.30},
                },
            },
            "segments": {
                "other":            {"am": 0.25, "ip": 0.30, "pm": 0.30, "ev": 0.15},
                "external_local":   {"am": 0.28, "ip": 0.22, "pm": 0.30, "ev": 0.20},
                "external_through": {"am": 0.24, "ip": 0.28, "pm": 0.30, "ev": 0.18},
            },
        },

        "temporal_policy": {
            "weekend_split": {
                "saturday_multiplier": 1.10,
                "sunday_multiplier": 0.90,
                "holiday_multiplier_to_sunday": 0.85,
            },
            "fallback_day_period_shares": {
                "day": 0.785,
                "evening": 0.137,
                "night": 0.078,
            },
            "demand_period_split_from_day": {
                "am": 0.35,
                "ip": 0.40,
                "pm": 0.25,
            },
        },

        "distribution": {
            "enabled": False,
            "impedance": "auto",
            "deterrence_function": "EXPO",
            "pa_trip_rate": 2.5,
            "pa_car_share": 0.50,
            "pa_occupancy": 1.3,
            "ipf_max_iter": 200,
            "ipf_tolerance": 0.001,
            "blend_alpha": 0.7,
            "uniform_impedance_fallback": 5000.0,
        },
    },

    # ------------------------------------------------------------------
    #  Calibration
    # ------------------------------------------------------------------
    "calibration": {
        "algorithm": "bfw",
        "max_iter": 150,
        "rgap_target": 0.002,
        "core_name": "wd_daily",
        "model_time_period": "daily",

        "count_source": "csd_split",

        "csd_split": {
            "strategy": "alternating",
            "calib_share": 0.65,
            "random_seed": 42,
        },

        "gateway_calibration": {
            "enabled": True,
            "damping": 0.18,
            "min_factor": 0.70,
            "max_factor": 1.40,
        },

        "match_buffer_m": 50.0,
        "match_direction_aware": True,
        "match_conflict_resolution": "nearest",
        "match_quality_min": 0.50,
        "count_target": "motor_total",
        "aggregate_corridor": True,
        "save_skims": False,
        "supply_tuning": {"enabled": False},

        "max_iterations": 50,
        "reset_matrix_before_run": True,

        "convergence": {
            "geh_lt5_target_pct": 85.0,
            "min_improvement_pct": -5.0,
            "daily": {
                "r2_target": 0.80,
                "slope_range": [0.85, 1.15],
                "pct_rmse_max": 35.0,
                "screenline_max_pct_deviation": 15.0,
                "bias_abs_max_pct": 15.0,
            },
        },

        "scaling": {
            "enabled": True,
            "method": "select_link",
            "damping": 0.40,
            "min_factor": 0.50,
            "max_factor": 2.00,
        },

        "quality_gates": {
            "hard_class_bias_max_abs_pct": 90.0,
            "objective_patience": 15,
        },

        "odme": {
            "max_outer_iterations": 25,
            "gradient_descent_iterations": 5,
            "max_deviation": 4.0,
            "weight_function": "inverse_sqrt",
            "convergence_tol": 0.001,
            "global_residual_damping": 0.25,
        },

        "matching": {
            "corridor_score_weights": {"distance": 0.30, "bearing": 0.30, "name": 0.40},
            "road_class_weights": {
                "motorway": 1.00, "trunk": 0.95, "primary": 0.90,
                "secondary": 0.75, "tertiary": 0.55, "residential": 0.30,
                "other": 0.20,
            },
            "link_type_penalty": 0.6,
            "min_vol_for_csd_lw": 100,
            "synthetic_objectid_base": 8_000_000,
        },

        "benchmarks": {
            "geh_lt5_pass_pct": 85.0,
            "jt_pass_pct": 85.0,
        },

        "screenline_factors": {
            "ratio_max": 5.0,
            "ratio_min": 0.2,
            "clip_min": 0.5,
            "clip_max": 2.0,
            "global_min": 0.8,
            "global_max": 1.25,
        },
    },

    # ------------------------------------------------------------------
    #  Datasets
    # ------------------------------------------------------------------
    "datasets": {
        "enabled": True,
        "root_dir": "data/sources",
        "http": {
            "timeout_s": 60,
            "retries": 3,
            "user_agent": "simulation-pipeline/1.0",
        },
        "closures_db": {
            "host": "REDACTED_HOST",
            "port": 5432,
            "dbname": "traffic",
            "user": "",
            "password": "",
            "bbox_margin_deg": 0.02,
        },
    },

    # ------------------------------------------------------------------
    #  Baseline closures
    # ------------------------------------------------------------------
    "baseline_closures": {
        "matching": {
            "max_distance_m": 150,
            "require_road_ref_match": False,
        },
        "severity_map": {
            "full":           {"capacity_factor": 0.05, "speed_factor": 0.10},
            "lane_reduction": {"capacity_factor": 0.50, "speed_factor": 0.70},
            "speed_limit":    {"capacity_factor": 1.00, "speed_factor": 0.50},
        },
        "measurement_period": {"start": "2025-01-01", "end": "2025-12-31"},
        "calibration_period": {"start": "2025-01-01", "end": "2025-12-31"},
        "validation_period":  {"start": "2025-01-01", "end": "2025-12-31"},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  LOCALE_DEFAULTS  –  mirrors locale.yaml  (Czech Republic national)
# ═══════════════════════════════════════════════════════════════════════════

LOCALE_DEFAULTS: Dict[str, Any] = {
    "holidays_md": [
        [1, 1], [5, 1], [5, 8], [7, 5], [7, 6],
        [9, 28], [10, 28], [11, 17], [12, 24], [12, 25], [12, 26],
    ],

    "road_classification": {
        "motorway_prefix": "D",
        "trunk_max": 99,
        "secondary_max": 999,
    },

    "reference_speeds": {
        "motorway": 130,
        "motorway_link": 80,
        "trunk": 90,
        "trunk_link": 70,
        "primary": 70,
        "primary_link": 50,
        "secondary": 60,
        "secondary_link": 40,
        "tertiary": 50,
        "tertiary_link": 30,
        "residential": 30,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  NETWORK_NORM_DEFAULTS  –  mirrors network_normalization.yaml
# ═══════════════════════════════════════════════════════════════════════════

NETWORK_NORM_DEFAULTS: Dict[str, Any] = {
    "normalization": {
        "thresholds": {
            "min_speed_kmh": 5.0,
            "min_capacity_vph": 50.0,
            "fallback_speed_kmh": 50.0,
            "generic_capacity_per_lane": 900,
            "min_travel_time_s": 0.01,
        },
        "defaults": {
            "speed_by_link_type": {
                "motorway": 130.0,
                "motorway_link": 80.0,
                "trunk": 90.0,
                "trunk_link": 70.0,
                "primary": 70.0,
                "primary_link": 50.0,
                "secondary": 50.0,
                "secondary_link": 40.0,
                "tertiary": 40.0,
                "tertiary_link": 35.0,
                "unclassified": 35.0,
                "road": 35.0,
                "residential": 50.0,
                "service": 20.0,
                "living_street": 20.0,
            },
            "lanes_by_link_type": {
                "motorway": 3,
                "motorway_link": 2,
                "trunk": 2,
                "trunk_link": 2,
                "primary": 2,
                "primary_link": 1,
                "secondary": 1,
                "secondary_link": 1,
                "tertiary": 1,
                "tertiary_link": 1,
                "unclassified": 1,
                "road": 1,
                "residential": 1,
                "service": 1,
                "living_street": 1,
            },
            "capacity_per_lane_by_link_type": {
                "motorway_link": 1800,
                "motorway": 2200,
                "trunk_link": 1500,
                "trunk": 1800,
                "primary_link": 1100,
                "primary": 1400,
                "secondary_link": 850,
                "secondary": 900,
                "tertiary_link": 650,
                "tertiary": 700,
                "unclassified": 600,
                "road": 600,
                "residential": 500,
                "service": 300,
                "living_street": 150,
            },
        },
    },
}
