import { create } from "zustand";

const ALL_LINK_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"];
const STORAGE_KEY = "sim-map-state";

interface DayInfo {
  date: string;
  weekday: string;
  day_type: string;
  day_factor: number;
  period_shares: Record<string, number>;
}

interface MapState {
  showLinks: boolean;
  showZones: boolean;
  showCentroids: boolean;
  showModelArea: boolean;
  showClosures: boolean;
  linkTypes: string[];
  selectedDate: string;
  selectedPeriod: string;
  dayInfo: DayInfo | null;
  toggleLayer: (layer: "links" | "zones" | "centroids" | "modelArea" | "closures") => void;
  toggleLinkType: (t: string) => void;
  setDate: (d: string) => void;
  setPeriod: (p: string) => void;
  setDayInfo: (info: DayInfo | null) => void;
}

function loadSaved(): Partial<MapState> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function persist(state: Partial<MapState>) {
  const { showLinks, showZones, showCentroids, showModelArea, showClosures, linkTypes, selectedDate, selectedPeriod } = state as MapState;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    showLinks, showZones, showCentroids, showModelArea, showClosures, linkTypes, selectedDate, selectedPeriod,
  }));
}

const today = new Date().toISOString().slice(0, 10);
const saved = loadSaved();

export const useMapStore = create<MapState>((set) => ({
  showLinks: saved.showLinks ?? true,
  showZones: saved.showZones ?? true,
  showCentroids: saved.showCentroids ?? false,
  showModelArea: saved.showModelArea ?? true,
  showClosures: saved.showClosures ?? true,
  linkTypes: saved.linkTypes ?? [...ALL_LINK_TYPES],
  selectedDate: saved.selectedDate ?? today,
  selectedPeriod: saved.selectedPeriod ?? "daily",
  dayInfo: null,

  toggleLayer: (layer) =>
    set((s) => {
      const next = layer === "links" ? { ...s, showLinks: !s.showLinks }
        : layer === "zones" ? { ...s, showZones: !s.showZones }
        : layer === "centroids" ? { ...s, showCentroids: !s.showCentroids }
        : layer === "closures" ? { ...s, showClosures: !s.showClosures }
        : { ...s, showModelArea: !s.showModelArea };
      persist(next);
      return next;
    }),

  toggleLinkType: (t) =>
    set((s) => {
      const next = { ...s, linkTypes: s.linkTypes.includes(t) ? s.linkTypes.filter((x) => x !== t) : [...s.linkTypes, t] };
      persist(next);
      return next;
    }),

  setDate: (selectedDate) => set((s) => { persist({ ...s, selectedDate }); return { selectedDate }; }),
  setPeriod: (selectedPeriod) => set((s) => { persist({ ...s, selectedPeriod }); return { selectedPeriod }; }),
  setDayInfo: (dayInfo) => set({ dayInfo }),
}));

export { ALL_LINK_TYPES };
export type { DayInfo };
