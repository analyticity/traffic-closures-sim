import { useMemo, useState } from "react";
import { MapContainer, TileLayer, GeoJSON, Polyline } from "react-leaflet";
import { useCorridors, useCorridorDiagnosis } from "../../api/hooks";
import type { CorridorPath } from "../../types";

const BRNO_CENTER: [number, number] = [49.195, 16.608];

function fmt(n: number): string {
  return Math.round(n).toLocaleString("cs-CZ");
}

function coordsToLatLngs(coords: [number, number][]): [number, number][] {
  return coords.map(([lng, lat]) => [lat, lng]);
}

function PathInfo({ path, color }: { path: CorridorPath; color: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="mt-1.5 inline-block h-2 w-6 rounded shrink-0" style={{ background: color }} />
      <div>
        <div className="font-medium text-gray-700">{path.label}</div>
        <div className="text-[11px] text-gray-500">
          <span className="font-semibold text-gray-800">
            {(path.travel_time / 60).toFixed(1)} min
          </span>
          {" "}&middot; {(path.distance / 1000).toFixed(2)} km &middot; {path.n_links} linků
        </div>
        {(path.free_flow_time != null || path.intersection_delay != null) && (
          <div className="text-[10px] text-gray-400 mt-0.5">
            {path.free_flow_time != null && (
              <span>ff {(path.free_flow_time / 60).toFixed(1)} min</span>
            )}
            {path.link_time != null && path.free_flow_time != null && path.link_time > path.free_flow_time && (
              <span> + kongesce {((path.link_time - path.free_flow_time) / 60).toFixed(1)} min</span>
            )}
            {path.intersection_delay != null && path.intersection_delay > 0 && (
              <span> + křiž. {(path.intersection_delay / 60).toFixed(1)} min</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function CorridorDiagnosisMap() {
  const { data: corridorsData, isLoading: corridorsLoading } = useCorridors();
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const { data: diagnosis, isLoading: diagLoading, error: diagError } = useCorridorDiagnosis(selectedName);

  const corridorNames = useMemo(() => {
    if (!corridorsData?.corridors) return [];
    return corridorsData.corridors
      .filter((c) => Math.abs(c.bias_pct) > 5)
      .map((c) => ({ name: c.name, bias_pct: c.bias_pct, ratio: c.ratio }));
  }, [corridorsData]);

  const freeFlowCoords = useMemo(() => {
    const g = diagnosis?.free_flow_path?.geometry;
    if (!g || g.type !== "LineString") return null;
    return coordsToLatLngs(g.coordinates);
  }, [diagnosis]);

  const viaCorridorCoords = useMemo(() => {
    const g = diagnosis?.via_corridor_path?.geometry;
    if (!g || g.type !== "LineString") return null;
    return coordsToLatLngs(g.coordinates);
  }, [diagnosis]);

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-[360px] flex flex-col border-r border-gray-200 bg-white overflow-hidden">
        <div className="px-3 py-2 border-b border-gray-200">
          <div className="text-xs font-semibold text-gray-700 mb-1">Vyberte koridor k diagnostice</div>
          <select
            value={selectedName ?? ""}
            onChange={(e) => setSelectedName(e.target.value || null)}
            className="w-full border border-gray-300 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">— Vyberte koridor —</option>
            {corridorNames.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.bias_pct > 0 ? "+" : ""}
                {c.bias_pct.toFixed(0)}%)
              </option>
            ))}
          </select>
        </div>

        {corridorsLoading && (
          <div className="p-4 text-sm text-gray-400">Načítání koridorů...</div>
        )}

        {diagLoading && selectedName && (
          <div className="p-4 text-sm text-gray-400">Analyzuji {selectedName}...</div>
        )}

        {diagError && (
          <div className="p-4 text-sm text-red-600">Chyba: {(diagError as Error).message}</div>
        )}

        {diagnosis && (
          <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs">
            {diagnosis.corridor_stats && (
              <div className="rounded-lg border border-gray-200 p-2">
                <div className="font-semibold text-gray-800 mb-1">{diagnosis.corridor_name}</div>
                <table className="w-full text-[11px]">
                  <tbody>
                    <tr>
                      <td className="text-gray-500">Pozorované</td>
                      <td className="text-right font-medium">
                        {fmt(diagnosis.corridor_stats.sum_observed)} voz/den
                      </td>
                    </tr>
                    <tr>
                      <td className="text-gray-500">Modelované</td>
                      <td className="text-right font-medium">
                        {fmt(diagnosis.corridor_stats.sum_modeled)} voz/den
                      </td>
                    </tr>
                    <tr>
                      <td className="text-gray-500">Poměr</td>
                      <td className="text-right font-semibold">
                        {diagnosis.corridor_stats.ratio.toFixed(2)}
                      </td>
                    </tr>
                    <tr>
                      <td className="text-gray-500">Počet stanic</td>
                      <td className="text-right">{diagnosis.corridor_stats.n_stations}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            <div className="space-y-2">
              {diagnosis.free_flow_path && (
                <PathInfo path={diagnosis.free_flow_path} color="#C0392B" />
              )}
              {diagnosis.via_corridor_path && (
                <PathInfo path={diagnosis.via_corridor_path} color="#148F77" />
              )}
            </div>

            {diagnosis.time_diff_pct != null && (
              <div className="rounded-lg bg-gray-50 border border-gray-200 p-2 text-[11px] text-gray-600">
                <div>
                  Travel time: via koridor o{" "}
                  <strong>
                    {Math.abs(diagnosis.time_diff_pct).toFixed(0)}%{" "}
                    {diagnosis.time_diff_pct > 0 ? "delší" : "kratší"}
                  </strong>
                </div>
                {diagnosis.dist_diff_pct != null && (
                  <div>
                    Vzdálenost: via koridor o{" "}
                    <strong>
                      {Math.abs(diagnosis.dist_diff_pct).toFixed(0)}%{" "}
                      {diagnosis.dist_diff_pct > 0 ? "delší" : "kratší"}
                    </strong>
                  </div>
                )}
              </div>
            )}

            <div className="rounded-lg bg-gray-100 border border-gray-200 p-2 text-[11px] text-gray-700 leading-relaxed">
              {diagnosis.diagnosis}
            </div>
          </div>
        )}
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        <MapContainer center={BRNO_CENTER} zoom={13} className="h-full w-full">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {diagnosis?.corridor_links && diagnosis.corridor_links.features.length > 0 && (
            <GeoJSON
              key={`corr-links-${selectedName}`}
              data={diagnosis.corridor_links as never}
              style={{ color: "#8E44AD", weight: 7, opacity: 0.4 }}
            />
          )}

          {freeFlowCoords && (
            <Polyline
              positions={freeFlowCoords}
              pathOptions={{ color: "#C0392B", weight: 4, opacity: 0.85, dashArray: "8 6" }}
            />
          )}

          {viaCorridorCoords && (
            <Polyline
              positions={viaCorridorCoords}
              pathOptions={{ color: "#148F77", weight: 4, opacity: 0.85 }}
            />
          )}
        </MapContainer>

        <div className="absolute bottom-6 right-4 z-[1000] rounded-lg bg-white border border-gray-200 p-3 shadow-lg text-xs">
          <div className="font-semibold text-gray-700 mb-2">Legenda</div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-2 w-6 rounded" style={{ background: "#8E44AD", opacity: 0.5 }} />
            <span>Koridor (linky)</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-1 w-6 rounded" style={{ background: "#148F77" }} />
            <span>Cesta přes koridor</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-1 w-6 rounded"
              style={{ background: "#C0392B", backgroundImage: "repeating-linear-gradient(90deg, #C0392B 0 4px, transparent 4px 8px)" }}
            />
            <span>Nejkratší cesta (model)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
