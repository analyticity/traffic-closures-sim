import { useMemo } from "react";
import { useModelMeta } from "../api/hooks";

/** Neutral fallback while ``/api/meta`` loads (approx. centroid of CZ). */
export const MAP_CENTER_FALLBACK_CZ: [number, number] = [49.75, 15.47];

export type LeafletMapLayout = {
  center: [number, number];
  /** Remount ``MapContainer`` when meta arrives so Leaflet picks up the real centroid. */
  containerKey: string;
};

/** Leaflet view from the API server's loaded city (``osm.place_name`` / zones). */
export function useLeafletMapLayout(): LeafletMapLayout {
  const meta = useModelMeta();
  const center = useMemo(() => {
    const c = meta.data?.map_center;
    if (c) return [c.lat, c.lng] as [number, number];
    return MAP_CENTER_FALLBACK_CZ;
  }, [meta.data?.map_center.lat, meta.data?.map_center.lng]);

  const containerKey = meta.isSuccess
    ? meta.data.city_slug || meta.data.title_short || "model"
    : "loading";

  return { center, containerKey };
}

/** @deprecated Use ``useLeafletMapLayout`` when you need a stable ``MapContainer`` key. */
export function useLeafletMapCenter(): [number, number] {
  return useLeafletMapLayout().center;
}
