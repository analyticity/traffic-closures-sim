import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";
import type {
  CalibrationReport,
  GeoJSONFeatureCollection,
  OdSummary,
  ValidationReport,
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
