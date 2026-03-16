import { CircleMarker, Popup } from "react-leaflet";
import type { GeoJSONFeatureCollection, CentroidProperties } from "../../types";

export function CentroidsLayer({ data }: { data: GeoJSONFeatureCollection }) {
  return (
    <>
      {data.features.map((f, i) => {
        const p = f.properties as unknown as CentroidProperties;
        const [lng, lat] = f.geometry.coordinates as [number, number];
        return (
          <CircleMarker
            key={i}
            center={[lat, lng]}
            radius={5}
            pathOptions={{ color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.9, weight: 1 }}
          >
            <Popup>
              <b>{p.name}</b><br />
              zone_id: {p.zone_id}<br />
              centroid_node: {p.centroid_node_id}
            </Popup>
          </CircleMarker>
        );
      })}
    </>
  );
}
