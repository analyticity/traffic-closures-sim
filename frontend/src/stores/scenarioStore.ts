import { create } from "zustand";
import { apiPost, apiFetch } from "../api/client";
import type {
  GeoJSONFeatureCollection,
  ScenarioLink,
  ScenarioJobStatus,
  ScenarioRunRequest,
} from "../types";

type ViewMode = "baseline" | "scenario" | "delta";

interface ScenarioState {
  links: ScenarioLink[];
  viewMode: ViewMode;
  scenarioResults: GeoJSONFeatureCollection | null;
  isRunning: boolean;
  jobId: string | null;
  jobStatus: ScenarioJobStatus | null;
  error: string | null;

  addLink: (link: Omit<ScenarioLink, "direction" | "closure_type" | "lanes_remaining">) => void;
  removeLink: (link_id: number) => void;
  updateLink: (link_id: number, patch: Partial<ScenarioLink>) => void;
  clearAll: () => void;
  setViewMode: (mode: ViewMode) => void;
  runScenario: () => void;
}

const POLL_INTERVAL_MS = 2000;

export const useScenarioStore = create<ScenarioState>((set, get) => ({
  links: [],
  viewMode: "baseline",
  scenarioResults: null,
  isRunning: false,
  jobId: null,
  jobStatus: null,
  error: null,

  addLink: (link) =>
    set((s) => {
      if (s.links.some((l) => l.link_id === link.link_id)) return s;
      return {
        links: [
          ...s.links,
          { ...link, direction: "both", closure_type: "full", lanes_remaining: Math.max(link.lanes - 1, 1) },
        ],
        scenarioResults: null,
        error: null,
      };
    }),

  removeLink: (link_id) =>
    set((s) => ({
      links: s.links.filter((l) => l.link_id !== link_id),
      scenarioResults: null,
      error: null,
    })),

  updateLink: (link_id, patch) =>
    set((s) => ({
      links: s.links.map((l) => (l.link_id === link_id ? { ...l, ...patch } : l)),
      scenarioResults: null,
      error: null,
    })),

  clearAll: () =>
    set({ links: [], scenarioResults: null, viewMode: "baseline", jobId: null, jobStatus: null, error: null }),

  setViewMode: (mode) => set({ viewMode: mode }),

  runScenario: async () => {
    const { links } = get();
    if (!links.length) return;

    set({ isRunning: true, error: null, scenarioResults: null, jobId: null, jobStatus: null });

    try {
      const req: ScenarioRunRequest = { links };
      const status = await apiPost<ScenarioJobStatus>("/api/scenarios/run", req);
      set({ jobId: status.id, jobStatus: status });

      const poll = async () => {
        const { jobId } = get();
        if (!jobId) return;

        try {
          const s = await apiFetch<ScenarioJobStatus>(`/api/scenarios/${jobId}/status`);
          set({ jobStatus: s });

          if (s.status === "done") {
            const results = await apiFetch<GeoJSONFeatureCollection>(
              `/api/scenarios/${jobId}/results`,
            );
            set({ scenarioResults: results, isRunning: false, viewMode: "scenario" });
          } else if (s.status === "error") {
            set({ isRunning: false, error: s.error ?? "Simulace selhala." });
          } else {
            setTimeout(poll, POLL_INTERVAL_MS);
          }
        } catch (e) {
          set({ isRunning: false, error: String(e) });
        }
      };

      setTimeout(poll, POLL_INTERVAL_MS);
    } catch (e) {
      set({ isRunning: false, error: String(e) });
    }
  },
}));
