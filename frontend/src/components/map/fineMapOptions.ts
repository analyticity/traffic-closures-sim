import type { MapOptions } from "leaflet";

/** Jemnější zoom (kolečko i tlačítka) — vhodné pro snímky obrazovky / plakát. */
export const FINE_INTERACTION_MAP_OPTIONS = {
  zoomSnap: 0.25,
  zoomDelta: 0.25,
  /** Vyšší hodnota = více „otáčení“ kolečka na jeden krok zoomu (default Leafletu je 60). */
  wheelPxPerZoomLevel: 140,
} satisfies Partial<MapOptions>;
