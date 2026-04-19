import { useQuery, useMutation } from "@tanstack/react-query";
import { apiFetch, apiPost } from "./client";
import type {
  BiasClusteredResponse,
  CalibrationReport,
  CorridorDiagnosis,
  CorridorsResponse,
  GeoJSONFeatureCollection,
  OdSummary,
  RoutesResponse,
  ScenarioJobStatus,
  ScenarioRunRequest,
  ThroughTrafficResponse,
  ValidationReport,
  ZoneRouteResponse,
  ZonesListResponse,
} from "../types";
import type { DayInfo } from "../stores/mapStore";

export const useLinks = (linkTypes: string[], date?: string, period?: string) =>
  useQuery({
    queryKey: ["links", linkTypes, date, period],
    queryFn: () => {
      const params = new URLSearchParams();
      if (linkTypes.length) params.set("link_types", linkTypes.join(","));
      if (date) params.set("date", date);
      if (period && period !== "daily") params.set("period", period);
      const qs = params.toString();
      return apiFetch<GeoJSONFeatureCollection>(`/api/links${qs ? `?${qs}` : ""}`);
    },
  });

export const useDayInfo = (date: string) =>
  useQuery({
    queryKey: ["day-info", date],
    queryFn: () => apiFetch<DayInfo>(`/api/temporal/day-info?date=${date}`),
    retry: false,
  });

export const useZones = () =>
  useQuery({
    queryKey: ["zones"],
    queryFn: () => apiFetch<GeoJSONFeatureCollection>("/api/zones"),
  });

export const useCentroids = () =>
  useQuery({
    queryKey: ["centroids"],
    queryFn: () => apiFetch<GeoJSONFeatureCollection>("/api/centroids"),
  });

export const useModelArea = () =>
  useQuery({
    queryKey: ["model-area"],
    queryFn: () => apiFetch<GeoJSONFeatureCollection>("/api/model-area"),
  });

export const useCalibration = () =>
  useQuery({
    queryKey: ["reports", "calibration"],
    queryFn: () => apiFetch<CalibrationReport>("/api/reports/calibration"),
  });

export const useValidation = () =>
  useQuery({
    queryKey: ["reports", "validation"],
    queryFn: () => apiFetch<ValidationReport>("/api/reports/validation"),
  });

export const useOdSummary = () =>
  useQuery({
    queryKey: ["reports", "od-summary"],
    queryFn: () => apiFetch<OdSummary>("/api/reports/od-summary"),
  });

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

export const useBiasData = (clustered = false, eps = 100) =>
  useQuery({
    queryKey: ["diagnostics", "bias", clustered, eps],
    queryFn: () => {
      const params = new URLSearchParams();
      if (clustered) {
        params.set("cluster", "true");
        params.set("eps", String(eps));
      }
      const qs = params.toString();
      return apiFetch<GeoJSONFeatureCollection | BiasClusteredResponse>(
        `/api/diagnostics/bias${qs ? `?${qs}` : ""}`,
      );
    },
  });

export const useThroughTrafficData = () =>
  useQuery({
    queryKey: ["diagnostics", "through-traffic"],
    queryFn: () => apiFetch<ThroughTrafficResponse>("/api/diagnostics/through-traffic"),
  });

export const useRouteData = () =>
  useQuery({
    queryKey: ["diagnostics", "routes"],
    queryFn: () => apiFetch<RoutesResponse>("/api/diagnostics/routes"),
  });

export const useCorridors = () =>
  useQuery({
    queryKey: ["diagnostics", "corridors"],
    queryFn: () => apiFetch<CorridorsResponse>("/api/diagnostics/corridors"),
  });

export const useCorridorDiagnosis = (name: string | null) =>
  useQuery({
    queryKey: ["diagnostics", "corridor-diagnosis", name],
    queryFn: () =>
      apiFetch<CorridorDiagnosis>(
        `/api/diagnostics/corridor-diagnosis?name=${encodeURIComponent(name!)}`,
      ),
    enabled: !!name,
  });

export const useZonesList = () =>
  useQuery({
    queryKey: ["diagnostics", "zones-list"],
    queryFn: () => apiFetch<ZonesListResponse>("/api/diagnostics/zones-list"),
  });

export const useZoneRoute = (origin: number | null, destination: number | null) =>
  useQuery({
    queryKey: ["diagnostics", "zone-route", origin, destination],
    queryFn: () =>
      apiFetch<ZoneRouteResponse>(
        `/api/diagnostics/zone-route?origin=${origin}&destination=${destination}`,
      ),
    enabled: origin != null && destination != null && origin !== destination,
  });

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

export const useRunScenario = () =>
  useMutation({
    mutationFn: (req: ScenarioRunRequest) =>
      apiPost<ScenarioJobStatus>("/api/scenarios/run", req),
  });

export const useScenarioStatus = (jobId: string | null) =>
  useQuery({
    queryKey: ["scenario-status", jobId],
    queryFn: () => apiFetch<ScenarioJobStatus>(`/api/scenarios/${jobId}/status`),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "done" || status === "error") return false;
      return 2000;
    },
  });

export const useScenarioResults = (jobId: string | null, enabled: boolean) =>
  useQuery({
    queryKey: ["scenario-results", jobId],
    queryFn: () => apiFetch<GeoJSONFeatureCollection>(`/api/scenarios/${jobId}/results`),
    enabled: !!jobId && enabled,
  });
