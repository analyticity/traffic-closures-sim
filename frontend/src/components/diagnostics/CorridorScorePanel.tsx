import { useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Popup,
} from "react-leaflet";
import { useCorridors } from "../../api/hooks";
import type { CorridorScore } from "../../types";

const BRNO_CENTER: [number, number] = [49.195, 16.608];

function biasColor(bias: number): string {
  if (bias > 20) return "#C0392B";
  if (bias > 10) return "#CA6F1E";
  if (bias < -20) return "#148F77";
  if (bias < -10) return "#1ABC9C";
  return "#27AE60";
}

function fmt(n: number): string {
  return Math.round(n).toLocaleString("cs-CZ");
}

function BiasBar({ bias }: { bias: number }) {
  const pct = Math.min(100, Math.abs(bias));
  const color = biasColor(bias);
  return (
    <div className="flex items-center gap-1.5 w-24">
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden relative">
        <div
          className="absolute top-0 h-full rounded-full"
          style={{
            background: color,
            width: `${pct}%`,
            left: bias < 0 ? `${100 - pct}%` : 0,
            right: bias >= 0 ? `${100 - pct}%` : 0,
          }}
        />
      </div>
      <span className="text-[11px] font-mono w-12 text-right" style={{ color }}>
        {bias > 0 ? "+" : ""}
        {bias.toFixed(0)}%
      </span>
    </div>
  );
}

export function CorridorScorePanel() {
  const { data, isLoading, error } = useCorridors();
  const [selected, setSelected] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"bias" | "geh" | "observed">("bias");

  const corridors = useMemo(() => {
    if (!data?.corridors) return [];
    const sorted = [...data.corridors];
    if (sortBy === "bias") sorted.sort((a, b) => Math.abs(b.bias_pct) - Math.abs(a.bias_pct));
    else if (sortBy === "geh") sorted.sort((a, b) => (b.geh ?? 0) - (a.geh ?? 0));
    else sorted.sort((a, b) => b.sum_observed - a.sum_observed);
    return sorted;
  }, [data, sortBy]);

  const selectedCorridor = useMemo(
    () => corridors.find((c) => c.name === selected) ?? null,
    [corridors, selected],
  );

  const geojsonForSelected = useMemo(() => {
    if (!selectedCorridor) return null;
    return {
      type: "FeatureCollection" as const,
      features: [
        {
          type: "Feature" as const,
          geometry: selectedCorridor.geometry,
          properties: { name: selectedCorridor.name },
        },
      ],
    };
  }, [selectedCorridor]);

  return (
    <div className="flex h-full">
      {/* Table */}
      <div className="w-[420px] flex flex-col border-r border-gray-200 bg-white overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-200 text-xs">
          <span className="text-gray-500 font-medium">Řadit:</span>
          {(["bias", "geh", "observed"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className={`px-2 py-0.5 rounded transition ${
                sortBy === s ? "bg-gray-100 text-gray-800 font-semibold" : "text-gray-500 hover:bg-gray-50"
              }`}
            >
              {s === "bias" ? "|Bias|" : s === "geh" ? "GEH" : "Pozorované"}
            </button>
          ))}
          <span className="ml-auto text-gray-400">{corridors.length} koridorů</span>
        </div>

        <div className="flex-1 overflow-y-auto">
          {isLoading && <div className="p-4 text-sm text-gray-400">Načítání...</div>}
          {error && <div className="p-4 text-sm text-red-600">Chyba: {(error as Error).message}</div>}

          {corridors.map((c) => (
            <button
              key={c.name}
              onClick={() => setSelected(selected === c.name ? null : c.name)}
              className={`w-full text-left px-3 py-2 border-b border-gray-100 transition text-xs hover:bg-gray-50 ${
                selected === c.name ? "bg-gray-50 border-l-2 border-l-indigo-600" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-gray-800 truncate max-w-[140px]">{c.name}</span>
                <BiasBar bias={c.bias_pct} />
              </div>
              <div className="flex gap-3 mt-0.5 text-[10px] text-gray-400">
                <span>{c.n_stations} stanic</span>
                <span>Poz: {fmt(c.sum_observed)}</span>
                <span>Mod: {fmt(c.sum_modeled)}</span>
                <span>GEH: {c.geh?.toFixed(1) ?? "–"}</span>
              </div>
            </button>
          ))}
        </div>

        {selectedCorridor && (
          <div className="border-t border-gray-200 bg-gray-50 px-3 py-2 text-xs">
            <div className="font-semibold text-gray-800 mb-1">{selectedCorridor.name}</div>
            <table className="w-full text-[11px]">
              <tbody>
                <tr>
                  <td className="text-gray-500 pr-2">Pozorované</td>
                  <td className="text-right font-medium">{fmt(selectedCorridor.sum_observed)} voz/den</td>
                </tr>
                <tr>
                  <td className="text-gray-500 pr-2">Modelované</td>
                  <td className="text-right font-medium">{fmt(selectedCorridor.sum_modeled)} voz/den</td>
                </tr>
                <tr>
                  <td className="text-gray-500 pr-2">Bias</td>
                  <td className="text-right font-semibold" style={{ color: biasColor(selectedCorridor.bias_pct) }}>
                    {selectedCorridor.bias_pct > 0 ? "+" : ""}{selectedCorridor.bias_pct.toFixed(1)}%
                  </td>
                </tr>
                <tr>
                  <td className="text-gray-500 pr-2">GEH</td>
                  <td className="text-right">{selectedCorridor.geh?.toFixed(1) ?? "–"}</td>
                </tr>
                <tr>
                  <td className="text-gray-500 pr-2">Typy linků</td>
                  <td className="text-right">{selectedCorridor.link_types.join(", ")}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        <MapContainer center={BRNO_CENTER} zoom={12} className="h-full w-full">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {geojsonForSelected && (
            <GeoJSON
              key={`corr-${selected}`}
              data={geojsonForSelected as never}
              style={{
                color: biasColor(selectedCorridor!.bias_pct),
                weight: 6,
                opacity: 0.9,
              }}
            >
              <Popup>
                <div style={{ fontSize: 13, fontWeight: 600 }}>
                  {selectedCorridor!.name} ({selectedCorridor!.bias_pct > 0 ? "+" : ""}
                  {selectedCorridor!.bias_pct.toFixed(1)}%)
                </div>
              </Popup>
            </GeoJSON>
          )}

          {!selectedCorridor &&
            corridors.map((c) => (
              <GeoJSON
                key={`all-${c.name}`}
                data={{
                  type: "FeatureCollection",
                  features: [{ type: "Feature", geometry: c.geometry, properties: { name: c.name } }],
                } as never}
                style={{ color: biasColor(c.bias_pct), weight: 3, opacity: 0.6 }}
                eventHandlers={{ click: () => setSelected(c.name) }}
              />
            ))}
        </MapContainer>

        <div className="absolute bottom-6 right-4 z-[1000] rounded-lg bg-white border border-gray-200 p-3 shadow-lg text-xs">
          <div className="font-semibold text-gray-700 mb-2">Odchylka koridoru</div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-2 w-6 rounded" style={{ background: "#C0392B" }} />
            <span>Nadhodnocení &gt; 20%</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-2 w-6 rounded" style={{ background: "#CA6F1E" }} />
            <span>Nadhodnocení 10–20%</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-2 w-6 rounded" style={{ background: "#27AE60" }} />
            <span>Dobrá shoda ±10%</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-2 w-6 rounded" style={{ background: "#1ABC9C" }} />
            <span>Podhodnocení 10–20%</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-6 rounded" style={{ background: "#148F77" }} />
            <span>Podhodnocení &gt; 20%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
