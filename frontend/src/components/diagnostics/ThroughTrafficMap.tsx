import { useMemo } from "react";
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup, Tooltip } from "react-leaflet";
import { useThroughTrafficData } from "../../api/hooks";
import { useLeafletMapLayout } from "../../map/useLeafletMapCenter";
import type { GeoJSONFeature, ThroughTrafficLinkProperties } from "../../types";

function shareColor(share: number): string {
  if (share > 0.5) return "#C0392B";
  if (share > 0.2) return "#CA6F1E";
  if (share > 0.05) return "#D4AC0D";
  return "#A6A6A6";
}

function shareWeight(throughVol: number, maxVol: number): number {
  if (maxVol <= 0) return 2;
  return 2 + 6 * (throughVol / maxVol);
}

function fmt(n: number): string {
  return Math.round(n).toLocaleString("cs-CZ");
}

export function ThroughTrafficMap() {
  const { data, isLoading, error } = useThroughTrafficData();
  const { center: mapCenter, containerKey } = useLeafletMapLayout();

  const maxThroughVol = useMemo(() => {
    if (!data?.links?.features) return 1;
    return Math.max(
      1,
      ...data.links.features.map(
        (f) => (f.properties as unknown as ThroughTrafficLinkProperties).through_volume,
      ),
    );
  }, [data]);

  const linkStyle = useMemo(() => {
    return (feature: GeoJSON.Feature | undefined) => {
      const p = feature?.properties as ThroughTrafficLinkProperties | undefined;
      if (!p) return { color: "#d1d5db", weight: 2, opacity: 0.5 };
      return {
        color: shareColor(p.through_share),
        weight: shareWeight(p.through_volume, maxThroughVol),
        opacity: 0.8,
      };
    };
  }, [maxThroughVol]);

  const onEachLink = useMemo(() => {
    return (feature: GeoJSON.Feature, layer: L.Layer) => {
      const p = feature.properties as unknown as ThroughTrafficLinkProperties;
      layer.bindPopup(
        `<div style="font-size:13px;line-height:1.6;min-width:200px">
          <div style="font-weight:700;border-bottom:1px solid #e5e7eb;padding-bottom:4px;margin-bottom:4px">
            Link ${p.link_id}
            <span style="float:right;font-weight:400;color:#6b7280;font-size:12px">${p.link_type}</span>
          </div>
          <table style="width:100%">
            <tr><td style="color:#6b7280">Tranzit</td><td style="text-align:right;font-weight:600">${fmt(p.through_volume)} voz/den</td></tr>
            <tr><td style="color:#6b7280">Celkem</td><td style="text-align:right">${fmt(p.total_volume)} voz/den</td></tr>
            <tr><td style="color:#6b7280">Podíl tranzitu</td><td style="text-align:right;font-weight:700;color:${shareColor(p.through_share)}">${(p.through_share * 100).toFixed(1)} %</td></tr>
          </table>
        </div>`,
        { maxWidth: 260 },
      );
    };
  }, []);

  const screenlineStyle = {
    color: "#8E44AD",
    weight: 6,
    opacity: 0.9,
    dashArray: "8 4",
  };

  return (
    <div className="relative h-full w-full">
      <MapContainer key={containerKey} center={mapCenter} zoom={12} className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {data?.links && (
          <GeoJSON
            key={`through-links-${data.links.features.length}`}
            data={data.links as never}
            style={linkStyle}
            onEachFeature={onEachLink}
          />
        )}

        {data?.screenlines && data.screenlines.features.length > 0 && (
          <GeoJSON
            key={`screenlines-${data.screenlines.features.length}`}
            data={data.screenlines as never}
            style={screenlineStyle}
            interactive={false}
          />
        )}

        {data?.gateways?.features.map((f: GeoJSONFeature) => {
          const coords = (f.geometry as { coordinates: [number, number] }).coordinates;
          const name = f.properties.gateway_name as string;
          return (
            <CircleMarker
              key={`gw-${name}`}
              center={[coords[1], coords[0]]}
              radius={10}
              pathOptions={{
                color: "#6C3483",
                fillColor: "#8E44AD",
                fillOpacity: 0.9,
                weight: 2,
              }}
            >
              <Tooltip permanent direction="top" offset={[0, -12]}>
                <span className="font-semibold">{name}</span>
              </Tooltip>
              <Popup>
                <div style={{ fontSize: 13 }}>
                  <strong>{name}</strong><br />
                  Node: {String(f.properties.graph_node)}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      <div className="absolute bottom-6 right-4 z-[1000] rounded-lg bg-white border border-gray-200 p-3 shadow-lg text-xs">
        <div className="font-semibold text-gray-700 mb-2">Podíl tranzitní dopravy</div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block h-2 w-6 rounded" style={{ background: "#C0392B" }} />
          <span>&gt; 50 %</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block h-2 w-6 rounded" style={{ background: "#CA6F1E" }} />
          <span>20–50 %</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block h-2 w-6 rounded" style={{ background: "#D4AC0D" }} />
          <span>5–20 %</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span className="inline-block h-2 w-6 rounded" style={{ background: "#A6A6A6" }} />
          <span>1–5 %</span>
        </div>
        <div className="mt-2 pt-2 border-t border-gray-200 flex items-center gap-2">
          <span className="inline-block h-2 w-6 rounded" style={{ background: "#8E44AD" }} />
          <span>Screenline</span>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <span className="inline-block h-3 w-3 rounded-full" style={{ background: "#8E44AD" }} />
          <span>Gateway</span>
        </div>
      </div>

      {isLoading && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-white px-4 py-2 shadow-lg text-xs text-gray-500">
          Načítání tranzitních dat…
        </div>
      )}
      {error && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-red-50 border border-red-300 px-4 py-2 shadow-lg text-xs text-red-600">
          Chyba: {(error as Error).message}
        </div>
      )}
    </div>
  );
}
