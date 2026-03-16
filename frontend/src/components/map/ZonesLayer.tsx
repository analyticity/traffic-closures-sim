import { GeoJSON } from "react-leaflet";
import type { GeoJSONFeatureCollection, ZoneProperties } from "../../types";

const ZONE_STYLE = {
  color: "#3b82f6",
  weight: 2,
  fillColor: "#3b82f6",
  fillOpacity: 0.08,
};

function onEachFeature(feature: GeoJSON.Feature, layer: L.Layer) {
  const p = feature.properties as ZoneProperties;
  const popStr = p.population ? `<br/>Population: ${p.population.toLocaleString()}` : "";
  layer.bindPopup(`<b>Zone ${p.zone_id}</b><br/>${p.name}${popStr}`);
  layer.bindTooltip(p.name || String(p.zone_id), {
    permanent: false,
    direction: "center",
    className: "text-xs",
  });
}

export function ZonesLayer({ data }: { data: GeoJSONFeatureCollection }) {
  return (
    <GeoJSON
      key="zones"
      data={data as never}
      style={ZONE_STYLE}
      onEachFeature={onEachFeature}
    />
  );
}
