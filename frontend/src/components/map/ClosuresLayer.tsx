import { CircleMarker, Popup } from "react-leaflet";
import type { GeoJSONFeatureCollection, ClosureFeatureProperties } from "../../types";

const SEVERITY_STYLE: Record<string, { color: string; fillColor: string }> = {
  full:           { color: "#C0392B", fillColor: "#E74C3C" },
  lane_reduction: { color: "#CA6F1E", fillColor: "#E67E22" },
  speed_limit:    { color: "#D4AC0D", fillColor: "#F1C40F" },
};

const SEVERITY_LABEL: Record<string, string> = {
  full: "Úplná uzavírka",
  lane_reduction: "Omezení pruhů",
  speed_limit: "Omezení rychlosti",
};

export function ClosuresLayer({ data }: { data: GeoJSONFeatureCollection }) {
  return (
    <>
      {data.features.map((f, i) => {
        const p = f.properties as unknown as ClosureFeatureProperties;
        const [lng, lat] = f.geometry.coordinates as [number, number];
        const style = SEVERITY_STYLE[p.severity] ?? SEVERITY_STYLE.lane_reduction;

        return (
          <CircleMarker
            key={`closure-${p.link_id}-${i}`}
            center={[lat, lng]}
            radius={7}
            pathOptions={{
              color: style.color,
              fillColor: style.fillColor,
              fillOpacity: 0.85,
              weight: 2,
            }}
          >
            <Popup>
              <div className="text-xs space-y-1">
                <div className="font-semibold text-sm">
                  {p.name || `Link #${p.link_id}`}
                </div>
                <div>
                  <span
                    className="inline-block rounded px-1.5 py-0.5 text-[10px] font-bold text-white"
                    style={{ backgroundColor: style.fillColor }}
                  >
                    {SEVERITY_LABEL[p.severity] ?? p.severity}
                  </span>
                  <span className="ml-1.5 text-gray-500">{p.link_type}</span>
                </div>
                {p.closure_text && (
                  <div className="text-gray-600 max-w-[200px]">{p.closure_text}</div>
                )}
                {(p.start || p.end) && (
                  <div className="text-gray-400">
                    {p.start && <span>{p.start}</span>}
                    {p.start && p.end && <span> — </span>}
                    {p.end && <span>{p.end}</span>}
                  </div>
                )}
                <div className="text-gray-400">
                  link_id: {p.link_id} · {p.closure_type === "full" ? "úplná" : `${p.lanes_remaining}/${p.lanes} pruhů`}
                </div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </>
  );
}

export { SEVERITY_STYLE, SEVERITY_LABEL };
