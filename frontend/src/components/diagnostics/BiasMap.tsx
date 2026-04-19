import { useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import { useBiasData } from "../../api/hooks";
import type { BiasClusterProperties, BiasStationProperties } from "../../types";

const BRNO_CENTER: [number, number] = [49.195, 16.608];

function biasColor(ratio: number): string {
  if (ratio > 1.2) return "#dc2626";
  if (ratio < 0.8) return "#2563eb";
  return "#16a34a";
}

function biasLabel(ratio: number): string {
  if (ratio > 1.2) return "Nadhodnocení";
  if (ratio < 0.8) return "Podhodnocení";
  return "Dobrá shoda";
}

function biasRadius(ratio: number, isCluster: boolean): number {
  if (ratio <= 0) return isCluster ? 8 : 4;
  const base = isCluster ? 8 : 4;
  return Math.min(isCluster ? 28 : 20, base + 6 * Math.abs(Math.log2(ratio)));
}

function fmt(n: number): string {
  return Math.round(n).toLocaleString("cs-CZ");
}

function StationPopup({ s }: { s: BiasStationProperties & { lat: number; lng: number } }) {
  return (
    <div style={{ fontSize: 13, lineHeight: 1.6 }}>
      <div
        style={{
          fontWeight: 700,
          fontSize: 14,
          borderBottom: "1px solid #e5e7eb",
          paddingBottom: 4,
          marginBottom: 4,
        }}
      >
        {s.name || "Bez názvu"}
        <span style={{ float: "right", fontWeight: 400, color: "#6b7280", fontSize: 12 }}>
          {s.link_type}
        </span>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={{ color: "#6b7280" }}>Pozorované</td>
            <td style={{ fontWeight: 600, textAlign: "right" }}>{fmt(s.observed)} voz/den</td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Modelované</td>
            <td style={{ fontWeight: 600, textAlign: "right" }}>{fmt(s.modeled)} voz/den</td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Poměr</td>
            <td
              style={{ fontWeight: 700, textAlign: "right", color: biasColor(s.ratio) }}
            >
              {s.ratio.toFixed(2)} ({biasLabel(s.ratio)})
            </td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Chyba</td>
            <td style={{ textAlign: "right" }}>
              {s.error > 0 ? "+" : ""}
              {fmt(s.error)} voz/den
            </td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>GEH</td>
            <td style={{ textAlign: "right" }}>{s.geh.toFixed(1)}</td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Koridorové linky</td>
            <td style={{ textAlign: "right" }}>{s.corridor_n_links}</td>
          </tr>
        </tbody>
      </table>
      <div
        style={{
          marginTop: 4,
          fontSize: 10,
          color: "#9ca3af",
          borderTop: "1px solid #e5e7eb",
          paddingTop: 3,
        }}
      >
        link_id: {s.link_id}
      </div>
    </div>
  );
}

function ClusterPopup({ c }: { c: BiasClusterProperties & { lat: number; lng: number } }) {
  return (
    <div style={{ fontSize: 13, lineHeight: 1.6 }}>
      <div
        style={{
          fontWeight: 700,
          fontSize: 14,
          borderBottom: "1px solid #e5e7eb",
          paddingBottom: 4,
          marginBottom: 4,
        }}
      >
        Cluster #{c.cluster_id}
        <span style={{ float: "right", fontWeight: 400, color: "#6b7280", fontSize: 12 }}>
          {c.n_stations} stanic
        </span>
      </div>
      {c.names.length > 0 && (
        <div style={{ marginBottom: 4, fontSize: 12, color: "#4b5563" }}>
          {c.names.join(", ")}
        </div>
      )}
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={{ color: "#6b7280" }}>Σ Pozorované</td>
            <td style={{ fontWeight: 600, textAlign: "right" }}>{fmt(c.observed)} voz/den</td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Σ Modelované</td>
            <td style={{ fontWeight: 600, textAlign: "right" }}>{fmt(c.modeled)} voz/den</td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Poměr</td>
            <td
              style={{ fontWeight: 700, textAlign: "right", color: biasColor(c.ratio) }}
            >
              {c.ratio.toFixed(2)} ({biasLabel(c.ratio)})
            </td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>Chyba</td>
            <td style={{ textAlign: "right" }}>
              {c.error > 0 ? "+" : ""}
              {fmt(c.error)} voz/den
            </td>
          </tr>
          <tr>
            <td style={{ color: "#6b7280" }}>GEH</td>
            <td style={{ textAlign: "right" }}>{c.geh.toFixed(1)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export function BiasMap() {
  const [clustered, setClustered] = useState(false);
  const { data, isLoading, error } = useBiasData(clustered);

  const stations = useMemo(() => {
    if (!data?.features || clustered) return [];
    return data.features.map((f) => {
      const p = f.properties as unknown as BiasStationProperties;
      const [lng, lat] = (f.geometry as { coordinates: [number, number] }).coordinates;
      return { ...p, lat, lng };
    });
  }, [data, clustered]);

  const clusters = useMemo(() => {
    if (!data?.features || !clustered) return [];
    return data.features.map((f) => {
      const p = f.properties as unknown as BiasClusterProperties;
      const [lng, lat] = (f.geometry as { coordinates: [number, number] }).coordinates;
      return { ...p, lat, lng };
    });
  }, [data, clustered]);

  const summary = useMemo(() => {
    const items = clustered ? clusters : stations;
    if (!items.length) return null;
    const over = items.filter((s) => s.ratio > 1.2).length;
    const under = items.filter((s) => s.ratio < 0.8).length;
    const good = items.length - over - under;
    return { total: items.length, over, under, good };
  }, [stations, clusters, clustered]);

  return (
    <div className="relative h-full w-full">
      <MapContainer center={BRNO_CENTER} zoom={12} className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {!clustered &&
          stations.map((s) => (
            <CircleMarker
              key={s.link_id}
              center={[s.lat, s.lng]}
              radius={biasRadius(s.ratio, false)}
              pathOptions={{
                color: biasColor(s.ratio),
                fillColor: biasColor(s.ratio),
                fillOpacity: 0.7,
                weight: 1,
              }}
            >
              <Popup maxWidth={280}>
                <StationPopup s={s} />
              </Popup>
            </CircleMarker>
          ))}

        {clustered &&
          clusters.map((c) => (
            <CircleMarker
              key={c.cluster_id}
              center={[c.lat, c.lng]}
              radius={biasRadius(c.ratio, true)}
              pathOptions={{
                color: biasColor(c.ratio),
                fillColor: biasColor(c.ratio),
                fillOpacity: 0.6,
                weight: 2,
              }}
            >
              <Popup maxWidth={300}>
                <ClusterPopup c={c} />
              </Popup>
            </CircleMarker>
          ))}
      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-6 right-4 z-[1000] rounded-lg bg-white p-3 shadow-lg text-xs">
        <div className="font-semibold text-gray-700 mb-2">Odchylka model/pozorování</div>

        {/* Cluster toggle */}
        <label className="flex items-center gap-2 mb-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={clustered}
            onChange={(e) => setClustered(e.target.checked)}
            className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
          />
          <span>Clustery (100 m)</span>
        </label>

        <div className="flex items-center gap-2 mb-1">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ background: "#dc2626" }}
          />
          <span>Nadhodnocení (&gt; 120 %)</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ background: "#16a34a" }}
          />
          <span>Dobrá shoda (80–120 %)</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <span
            className="inline-block h-3 w-3 rounded-full"
            style={{ background: "#2563eb" }}
          />
          <span>Podhodnocení (&lt; 80 %)</span>
        </div>
        {summary && (
          <div className="mt-2 pt-2 border-t border-gray-200 text-gray-500">
            Celkem {summary.total} {clustered ? "clusterů" : "stanic"}:
            <span style={{ color: "#dc2626" }}> {summary.over}</span> /
            <span style={{ color: "#16a34a" }}> {summary.good}</span> /
            <span style={{ color: "#2563eb" }}> {summary.under}</span>
          </div>
        )}
      </div>

      {isLoading && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-white px-4 py-2 shadow-lg text-xs text-gray-500">
          Načítání diagnostických dat…
        </div>
      )}
      {error && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-red-50 border border-red-200 px-4 py-2 shadow-lg text-xs text-red-600">
          Chyba: {(error as Error).message}
        </div>
      )}
    </div>
  );
}
