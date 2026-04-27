export interface GeoJSONFeatureCollection {
  type: "FeatureCollection";
  features: GeoJSONFeature[];
}

export interface GeoJSONFeature {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: Record<string, unknown>;
}

/** From ``GET /api/meta`` — reflects the city loaded in the API server config. */
export interface ModelMeta {
  place_name: string;
  city_slug: string;
  title_short: string;
  map_center: { lat: number; lng: number };
  features?: {
    has_closures?: boolean;
  };
}

export interface LinkProperties {
  link_id: number;
  /** 0 = obousměrný úsek v síti, 1 = jen A→B, -1 = jen B→A (viz export sítě) */
  direction?: number;
  a_node?: number;
  b_node?: number;
  link_type: string;
  name: string | null;
  osm_ref?: string | null;
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
  peak_hour_vol_AB?: number;
  peak_hour_vol_BA?: number;
  K_factor?: number;
  LOS_max?: string;
  Congested_Time_Max: number;
  Congested_Time_AB: number;
  Congested_Time_BA: number;
  Delay_factor_Max: number;
  Delay_factor_AB: number;
  Delay_factor_BA: number;
  baseline_vol?: number;
  delta_vol?: number;
  delta_pct?: number;
  abs_delta_vol?: number;
  baseline_voc?: number;
  delta_voc?: number;
  baseline_ct?: number;
  delta_ct?: number;
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

/** Link-level fit vs observed counts (CSD calibration subset or pentlogram). */
export interface ValidationCountFit {
  matched: number;
  n: number;
  r2: number | null;
  rmse: number;
  geh_lt5_pct: number;
  geh_lt10_pct: number;
  sum_modeled: number;
  sum_observed: number;
}

export interface ValidationReport {
  /** When ``calibration.count_source`` is ``pentlogram``. */
  pentlogram?: ValidationCountFit;
  /** When ``calibration.count_source`` is ``csd_split`` (reference step 1). */
  calibration_reference?: ValidationCountFit;
  csd_observed?: Array<{
    road_class: string;
    sections: number;
    mean_sv: number;
    mean_o: number;
  }>;
  csd_modeled?: Array<{
    link_type: string;
    links: number;
    mean_vol: number;
    total_vol: number;
  }>;
}

export interface ScenarioLink {
  link_id: number;
  name: string | null;
  link_type: string;
  lanes: number;
  direction: "both" | "ab" | "ba";
  closure_type: "full" | "lanes";
  lanes_remaining: number;
  /** Sloupec `direction` ze sítě při přidání z mapy — jen pro rozhraní (zakázané položky ve výběru směru). */
  network_direction?: number;
}

export interface ScenarioRunRequest {
  links: ScenarioLink[];
}

export interface ScenarioJobStatus {
  id: string;
  status: "running" | "done" | "error";
  error?: string;
  started_at: number;
  elapsed_seconds: number;
}

// ---------------------------------------------------------------------------
// Closures (date-based)
// ---------------------------------------------------------------------------

export interface ClosureFeatureProperties {
  link_id: number;
  direction: "both" | "ab" | "ba";
  closure_type: "full" | "lanes";
  lanes: number;
  lanes_remaining: number;
  name: string;
  link_type: string;
  severity: "full" | "lane_reduction" | "speed_limit";
  closure_text: string;
  start: string;
  end: string;
}

// ---------------------------------------------------------------------------
// Delta / Scenario summary
// ---------------------------------------------------------------------------

export interface DeltaSummaryLink {
  link_id: number;
  name: string;
  link_type: string;
  delta_vol: number;
  delta_pct: number;
  baseline_vol: number;
  scenario_vol: number;
  delta_voc: number;
}

export interface DeltaSummary {
  total_links_affected: number;
  mean_abs_delta_vol: number;
  mean_abs_delta_pct: number;
  max_increase: DeltaSummaryLink | null;
  max_decrease: DeltaSummaryLink | null;
  top_affected: DeltaSummaryLink[];
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

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

export interface BiasStationProperties {
  link_id: number;
  name: string;
  link_type: string;
  observed: number;
  modeled: number;
  ratio: number;
  error: number;
  geh: number;
  corridor_n_links: number;
}

export interface BiasClusterProperties {
  cluster_id: number;
  n_stations: number;
  names: string[];
  link_ids: number[];
  observed: number;
  modeled: number;
  ratio: number;
  error: number;
  geh: number;
}

export interface BiasClusteredResponse extends GeoJSONFeatureCollection {
  clustered: boolean;
  eps_m: number;
  n_clusters: number;
}

export interface CorridorScore {
  name: string;
  n_stations: number;
  sum_observed: number;
  sum_modeled: number;
  ratio: number;
  geh: number | null;
  bias_pct: number;
  link_types: string[];
  link_ids: number[];
  geometry: { type: string; coordinates: unknown };
}

export interface CorridorsResponse {
  corridors: CorridorScore[];
}

export interface CorridorPath {
  geometry: { type: string; coordinates: [number, number][] } | null;
  travel_time: number;
  free_flow_time?: number;
  link_time?: number;
  intersection_delay?: number;
  distance: number;
  n_links: number;
  link_ids: number[];
  label: string;
}

export interface ZoneInfo {
  zone_id: number;
  name: string;
  is_external: boolean;
  coords: [number, number];
}

export interface ZonesListResponse {
  zones: ZoneInfo[];
}

export interface ZoneRouteEndpoint {
  zone_id: number;
  name: string;
  node: number;
  coords: [number, number] | null;
}

export interface ZoneRouteBiasStation {
  link_id: number;
  name: string;
  observed: number;
  modeled: number;
  ratio: number;
  coords: [number, number] | null;
}

export interface ZoneRouteResponse {
  origin: ZoneRouteEndpoint;
  destination: ZoneRouteEndpoint;
  path: {
    geometry: { type: string; coordinates: [number, number][] } | null;
    travel_time: number;
    free_flow_time?: number;
    link_time?: number;
    intersection_delay?: number;
    distance: number;
    n_links: number;
  };
  streets: { name: string; distance: number }[];
  bias_stations: ZoneRouteBiasStation[];
}

export interface CorridorDiagnosis {
  corridor_name: string;
  corridor_stats: {
    sum_observed: number;
    sum_modeled: number;
    ratio: number;
    n_stations: number;
  } | null;
  corridor_links: GeoJSONFeatureCollection;
  free_flow_path: CorridorPath | null;
  via_corridor_path: CorridorPath | null;
  diagnosis: string;
  time_diff_pct?: number;
  dist_diff_pct?: number;
}

export interface ThroughTrafficLinkProperties {
  link_id: number;
  link_type: string;
  through_volume: number;
  total_volume: number;
  through_share: number;
}

export interface ThroughTrafficResponse {
  links: GeoJSONFeatureCollection;
  gateways: GeoJSONFeatureCollection;
  screenlines: GeoJSONFeatureCollection;
}

export interface RouteEndpoint {
  node: number;
  coords: [number, number] | null;
}

export interface RouteComparison {
  name: string;
  origin: RouteEndpoint;
  destination: RouteEndpoint;
  free_flow_path: GeoJSONFeature;
  high_volume_links: GeoJSONFeatureCollection;
}

export interface RoutesResponse {
  routes: RouteComparison[];
}
