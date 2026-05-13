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
        "boundary_scc_repair": "auto",
        "divided_highway_snap_m": 1000,
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
        "connector_warn_distance_m": 1500.0,
        "connector_strict": False,
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
        "blocked_centroid_flows": True,

        "warm_skim_pass": {
            "algorithm": "bfw",
            "max_iter": 80,
            "rgap_target": 0.02,
        },

        # Generalized cost adds a distance-based monetary penalty to travel
        # time during assignment.  AequilibraE computes:
        #   gc = travel_time + (fixed_cost_field * fixed_cost_multiplier) / vot
        # With distance in meters, multiplier 0.005 means 5 cost-units/km.
        # VOT=1.0 treats 1 cost-unit = 1 second of travel time, so this adds
        # an effective 5 sec/km distance penalty (discourages long detours).
        # Calibrate against local willingness-to-pay or VOT surveys if available.
        "generalized_cost": {
            "enabled": True,
            "fixed_cost_field": "distance",
            "fixed_cost_multiplier": 0.005,
            "vot": 1.0,
        },

        "bpr": {
            "vdf_function": "BPR",
            "per_link": True,
            "alpha_default": 0.55,
            "beta_default": 4.0,
            "daily_capacity_factor": {
                "default": 9.0,
                "by_link_type": {
                    "motorway": 13.0,
                    "motorway_link": 11.0,
                    "trunk": 12.0,
                    "trunk_link": 10.0,
                    "primary": 10.0,
                    "primary_link": 10.0,
                    "secondary": 9.0,
                    "secondary_link": 9.0,
                    "tertiary": 8.0,
                    "tertiary_link": 8.0,
                    "residential": 6.5,
                    "living_street": 5.5,
                    "service": 5.5,
                    "unclassified": 8.0,
                },
            },
            "by_link_type": {
                "motorway":       {"alpha": 0.15, "beta": 4.0},
                "motorway_link":  {"alpha": 0.25, "beta": 4.0},
                "trunk":          {"alpha": 0.45, "beta": 4.0},
                "trunk_link":     {"alpha": 0.50, "beta": 4.0},
                "primary":        {"alpha": 0.60, "beta": 4.0},
                "primary_link":   {"alpha": 0.65, "beta": 4.0},
                "secondary":      {"alpha": 0.70, "beta": 4.0},
                "secondary_link": {"alpha": 0.70, "beta": 4.0},
                "tertiary":       {"alpha": 0.75, "beta": 4.0},
                "tertiary_link":  {"alpha": 0.75, "beta": 4.0},
                "residential":    {"alpha": 0.85, "beta": 4.0},
                "unclassified":   {"alpha": 0.75, "beta": 4.0},
                "living_street":  {"alpha": 1.00, "beta": 4.0},
                "service":        {"alpha": 1.00, "beta": 4.0},
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
        "od_global_scale": 1.0,

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
            "work":   {"car_share": 0.48, "occupancy": 1.20, "trips_per_person": 2.0},
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
            "enabled": True,
            "impedance": "skim",
            "allow_euclidean_fallback": False,
            "allow_unconverged_skims": False,
            "segments": ["other"],
            "employment_source": "auto",
            "require_employment": True,
            "deterrence_function": "EXPO",
            "min_gravity_beta": 0.0001,
            "pa_trip_rate": 1.8,
            "pa_car_share": 0.40,
            "pa_occupancy": 1.3,
            "ipf_max_iter": 200,
            "ipf_tolerance": 0.001,
            "blend_alpha": 0.55,
            "uniform_impedance_fallback": 5000.0,
            "max_total_multiplier": 3.5,
        },
    },

    # ------------------------------------------------------------------
    #  Calibration
    # ------------------------------------------------------------------
    "calibration": {
        "algorithm": "bfw",
        "allow_aon": False,
        "strict_convergence": False,
        "max_iter": 200,
        "rgap_target": 0.002,
        "core_name": "wd_daily",
        "model_time_period": "daily",
        "min_obs_per_zone": 0.5,

        "count_source": "csd_split",
        "require_holdout": True,
        "method": "entropy_odme",

        "match_buffer_m": 120.0,
        "match_direction_aware": True,
        "match_conflict_resolution": "nearest",
        "match_quality_min": 0.25,
        "count_target": "motor_total",
        "aggregate_corridor": True,
        "save_skims": False,
        "require_supply_audit": True,

        "csd_split": {
            "strategy": "corridor",
            "calib_share": 0.65,
            "random_seed": 42,
        },

        "gateway_calibration": {
            "enabled": True,
            "damping": 0.20,
            "min_factor": 0.60,
            "max_factor": 1.45,
            "rebase_seed_bounds": True,
        },

        "supply_tuning": {
            "enabled": True,
            "road_classes": ["motorway", "trunk", "primary", "secondary", "tertiary"],
            "speed_factor_range": [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],
            "capacity_factor_range": [0.70, 0.85, 1.0, 1.15, 1.30, 1.50],
            "inner_max_iterations": 3,
        },

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
            "objective_patience": 25,
        },

        "odme": {
            "max_outer_iterations": 35,
            "gradient_descent_iterations": 8,
            "stall_patience": 14,
            "entropy_step_size": 0.20,
            "convergence_tol": 0.001,
            "max_deviation": 4.0,
            "weight_function": "inverse_sqrt",
            "global_residual_damping": 0.20,
            "max_iter_change_pct": 12.0,
            "assign_max_iter": 100,
            "assign_rgap_target": 0.008,
            "class_residual_enabled": True,
            "class_residual_damping": 0.06,
            "class_residual_min_counts": 3,
            "stages": [
                {
                    "name": "stabilize",
                    "max_iterations": 16,
                    "max_deviation": 5.0,
                    "global_residual_damping": 0.15,
                    "max_iter_change_pct": 15.0,
                    "gateway_calibration": {
                        "enabled": True,
                        "damping": 0.15,
                    },
                },
                {
                    "name": "refine",
                    "max_iterations": 25,
                    "max_deviation": 3.5,
                    "global_residual_damping": 0.20,
                    "max_iter_change_pct": 10.0,
                },
            ],
        },

        "validation": {
            "match_buffer_m": 120.0,
            "match_quality_min": 0.25,
        },

        "auto_screenlines": {
            "enabled": True,
            "gateway_screenlines": True,
            "csd_screenlines": True,
            "csd_min_aadt": 5000,
            "csd_agg_method": "length_weighted",
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
            "global_min": 0.70,
            "global_max": 1.50,
        },

        "screenline_dedup_strict": False,
        "screenline_cross_dedup": True,

        "multistage": {
            "max_total_change_pct": 50.0,
            "multistage_max_deviation": 3.0,
        },
    },

    # ------------------------------------------------------------------
    #  Sensitivity
    # ------------------------------------------------------------------
    "sensitivity": {
        "enabled": False,
        "max_iter": 30,
        "rgap_target": 0.05,
        "parameters": [
            {
                "path": "demand.sldb.external_processing.through_traffic_scale",
                "values": [0.25, 0.50, 0.75, 1.00],
            },
        ],
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

    "csd_region_map": {
        "region_hints": {
            "ústecký": "042", "ustecky": "042",
            "jihomoravský": "064", "jihomoravsky": "064",
            "středočeský": "020", "stredocesky": "020",
            "jihočeský": "031", "jihocesky": "031",
            "plzeňský": "032", "plzensky": "032",
            "karlovarský": "041", "karlovarsky": "041",
            "liberecký": "051", "liberecky": "051",
            "královéhradecký": "052", "kralovehradecky": "052",
            "pardubický": "053", "pardubicky": "053",
            "vysočina": "063",
            "olomoucký": "071", "olomoucky": "071",
            "zlínský": "072", "zlinsky": "072",
            "moravskoslezský": "080", "moravskoslezsky": "080",
        },
        "city_region": {
            "most": "042", "teplice": "042", "ústí": "042", "usti": "042",
            "chomutov": "042", "děčín": "042", "decin": "042", "litvínov": "042",
            "litvinov": "042", "louny": "042", "žatec": "042",
            "brno": "064",
            "praha": "020", "prague": "020",
            "ostrava": "080", "opava": "080", "karviná": "080",
            "plzeň": "032", "pilsen": "032",
            "liberec": "051",
            "olomouc": "071",
            "zlín": "072", "zlin": "072",
            "pardubice": "053",
            "hradec": "052",
            "české budějovice": "031", "ceske budejovice": "031",
            "karlovy vary": "041",
            "jihlava": "063",
        },
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
        "practical_speed": {
            "enabled": True,
            "base_factor": 0.85,
            "intersection_penalty_per_km": 0.02,
            "min_intersection_degree": 3,
            "min_speed_kmh": 5.0,
        },
        "defaults": {
            "speed_by_link_type": {
                "motorway": 130.0,
                "motorway_link": 80.0,
                "trunk": 90.0,
                "trunk_link": 60.0,
                "primary": 90.0,
                "primary_link": 50.0,
                "secondary": 70.0,
                "secondary_link": 50.0,
                "tertiary": 50.0,
                "tertiary_link": 40.0,
                "unclassified": 40.0,
                "road": 40.0,
                "residential": 50.0,
                "service": 30.0,
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
                "secondary_link": 900,
                "secondary": 1000,
                "tertiary_link": 700,
                "tertiary": 800,
                "unclassified": 600,
                "road": 600,
                "residential": 500,
                "service": 300,
                "living_street": 150,
            },
        },
    },
}
