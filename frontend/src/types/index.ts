export interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}

export interface GeoJSONFeature {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: Record<string, unknown>;
}

export interface LinkProperties {
  link_id: number;
  link_type: string;
  name: string | null;
  speed: number;
  capacity: number;
  lanes: number;
  distance: number;
  wd_daily_tot: number;
  wd_daily_ab: number;
  wd_daily_ba: number;
  VOC_max: number;
  VOC_AB: number;
  VOC_BA: number;
  Congested_Time_Max: number;
  Congested_Time_AB: number;
  Congested_Time_BA: number;
  Delay_factor_Max: number;
  Delay_factor_AB: number;
  Delay_factor_BA: number;
}

export interface ZoneProperties {
  zone_id: number;
  name: string;
  source_rank: number;
  population?: number;
}

export interface CentroidProperties {
  zone_id: number;
  name: string;
  centroid_node_id: number;
}

export interface CalibrationIteration {
  iteration: number;
  demand_total: number;
  assigned_total: number;
  n: number;
  r2: number | null;
  rmse: number;
  pct_rmse: number | null;
  geh_mean: number;
  geh_median: number;
  geh_lt5_pct: number;
  geh_lt10_pct: number;
  sum_modeled: number;
  sum_observed: number;
}

export interface CalibrationReport {
  iterations: number;
  converged: boolean;
  history: CalibrationIteration[];
  final: CalibrationIteration;
  config: {
    max_iterations: number;
    geh_target: number;
    scale_method: string;
    damping: number;
  };
}

export interface ValidationReport {
  pentlogram?: {
    matched: number;
    n: number;
    r2: number | null;
    rmse: number;
    geh_lt5_pct: number;
    geh_lt10_pct: number;
    sum_modeled: number;
    sum_observed: number;
  };
  csd2020_observed?: Array<{
    road_class: string;
    sections: number;
    mean_sv: number;
    mean_o: number;
  }>;
  csd2020_modeled?: Array<{
    link_type: string;
    links: number;
    mean_vol: number;
    total_vol: number;
  }>;
}

export interface OdSummary {
  zones: number;
  rows_in: number;
  mapped_direct: number;
  mapped_group: number;
  missing_origin: number;
  missing_destination: number;
  pairs_used: number;
  cores_sum: Record<string, number>;
  nonzero_cells: Record<string, number>;
}
