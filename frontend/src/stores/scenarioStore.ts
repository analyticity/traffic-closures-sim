import { create } from "zustand";
import { apiPost, apiFetch } from "../api/client";
import type {
  GeoJSONFeatureCollection,
  ScenarioLink,
  ScenarioJobStatus,
  ScenarioRunRequest,
} from "../types";

type ViewMode = "baseline" | "scenario" | "delta";

/** Přidání z mapy: směr lze odvodit z `network_direction` (sloupec sítě), pokud není zadán. */
export type ScenarioAddPayload = Omit<ScenarioLink, "direction" | "closure_type" | "lanes_remaining"> &
  Partial<Pick<ScenarioLink, "direction" | "closure_type" | "lanes_remaining">> & {
    network_direction?: number;
  };

function inferScenarioDirection(networkDirection: number | undefined): ScenarioLink["direction"] {
  if (networkDirection === -1) return "ba";
  return "ab";
}

interface ScenarioState {
  links: ScenarioLink[];
  viewMode: ViewMode;
  scenarioResults: GeoJSONFeatureCollection | null;
  isRunning: boolean;
  jobId: string | null;
  jobStatus: ScenarioJobStatus | null;
  error: string | null;

  addLink: (link: ScenarioAddPayload) => void;
  addLinks: (links: ScenarioAddPayload[]) => void;
  /** Bez `direction` smaže všechny položky s daným `link_id`. */
  removeLink: (link_id: number, direction?: ScenarioLink["direction"]) => void;
  updateLink: (link_id: number, direction: ScenarioLink["direction"], patch: Partial<ScenarioLink>) => void;
  clearAll: () => void;
  setViewMode: (mode: ViewMode) => void;
  runScenario: () => void;
  runClosureScenario: (date: string) => void;
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

  addLink: (raw) =>
    set((s) => {
      const { network_direction, direction: d0, closure_type, lanes_remaining, ...base } = raw;
      const direction: ScenarioLink["direction"] = d0 ?? inferScenarioDirection(network_direction);
      const closure: ScenarioLink["closure_type"] = closure_type ?? "full";
      const lr =
        lanes_remaining !== undefined
          ? lanes_remaining
          : Math.max((base as { lanes: number }).lanes - 1, 1);
      if (s.links.some((l) => l.link_id === base.link_id && l.direction === direction)) return s;
      const row: ScenarioLink = {
        ...(base as Omit<ScenarioLink, "direction" | "closure_type" | "lanes_remaining">),
        direction,
        closure_type: closure,
        lanes_remaining: lr,
        ...(network_direction !== undefined ? { network_direction } : {}),
      };
      return {
        links: [...s.links, row],
        scenarioResults: null,
        error: null,
      };
    }),

  addLinks: (raws) =>
    set((s) => {
      const existing = new Set(s.links.map((l) => `${l.link_id}:${l.direction}`));
      const newLinks: ScenarioLink[] = [];
      for (const raw of raws) {
        const { network_direction, direction: d0, closure_type, lanes_remaining, ...base } = raw;
        const direction: ScenarioLink["direction"] = d0 ?? inferScenarioDirection(network_direction);
        const key = `${base.link_id}:${direction}`;
        if (existing.has(key)) continue;
        existing.add(key);
        const closure: ScenarioLink["closure_type"] = closure_type ?? "full";
        const lr =
          lanes_remaining !== undefined
            ? lanes_remaining
            : Math.max((base as { lanes: number }).lanes - 1, 1);
        newLinks.push({
          ...(base as Omit<ScenarioLink, "direction" | "closure_type" | "lanes_remaining">),
          direction,
          closure_type: closure,
          lanes_remaining: lr,
          ...(network_direction !== undefined ? { network_direction } : {}),
        });
      }
      if (!newLinks.length) return s;
      return {
        links: [...s.links, ...newLinks],
        scenarioResults: null,
        error: null,
      };
    }),

  removeLink: (link_id, direction) =>
    set((s) => ({
      links: s.links.filter((l) =>
        direction === undefined
          ? l.link_id !== link_id
          : !(l.link_id === link_id && l.direction === direction),
      ),
      scenarioResults: null,
      error: null,
    })),

  updateLink: (link_id, direction, patch) =>
    set((s) => ({
      links: s.links.map((l) =>
        l.link_id === link_id && l.direction === direction ? { ...l, ...patch } : l,
      ),
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

  runClosureScenario: async (date: string) => {
    set({ isRunning: true, error: null, scenarioResults: null, jobId: null, jobStatus: null });

    try {
      const status = await apiPost<ScenarioJobStatus>("/api/closures/run-scenario", { date });
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
