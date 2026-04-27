import { useMemo, useState, useCallback } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Tooltip,
  Popup,
} from "react-leaflet";
import { useZonesList, useZoneRoute } from "../../api/hooks";
import { useLeafletMapLayout } from "../../map/useLeafletMapCenter";
import type { ZoneRouteBiasStation } from "../../types";

function fmt(n: number): string {
  return Math.round(n).toLocaleString("cs-CZ");
}

function coordsToLatLngs(coords: [number, number][]): [number, number][] {
  return coords.map(([lng, lat]) => [lat, lng]);
}

function biasColor(ratio: number): string {
  if (ratio > 1.2) return "#C0392B";
  if (ratio < 0.8) return "#148F77";
  return "#27AE60";
}

function BiasMarker({ s }: { s: ZoneRouteBiasStation }) {
  if (!s.coords) return null;
  return (
    <CircleMarker
      center={[s.coords[1], s.coords[0]]}
      radius={6}
      pathOptions={{
        color: biasColor(s.ratio),
        fillColor: biasColor(s.ratio),
        fillOpacity: 0.8,
        weight: 1,
      }}
    >
      <Popup maxWidth={220}>
        <div style={{ fontSize: 12, lineHeight: 1.5 }}>
          <div style={{ fontWeight: 600 }}>{s.name || `Link ${s.link_id}`}</div>
          <div>Pozorované: {fmt(s.observed)} voz/den</div>
          <div>Modelované: {fmt(s.modeled)} voz/den</div>
          <div style={{ color: biasColor(s.ratio), fontWeight: 600 }}>
            Poměr: {s.ratio.toFixed(2)}
          </div>
        </div>
      </Popup>
    </CircleMarker>
  );
}

export function RouteComparisonMap() {
  const { data: zonesData, isLoading: zonesLoading } = useZonesList();
  const { center: mapCenter, containerKey } = useLeafletMapLayout();
  const [origin, setOrigin] = useState<number | null>(null);
  const [destination, setDestination] = useState<number | null>(null);
  const { data: routeData, isLoading: routeLoading, error: routeError } = useZoneRoute(origin, destination);

  const zones = useMemo(() => zonesData?.zones ?? [], [zonesData]);
  const internalZones = useMemo(() => zones.filter((z) => !z.is_external), [zones]);
  const externalZones = useMemo(() => zones.filter((z) => z.is_external), [zones]);

  const pathCoords = useMemo(() => {
    const g = routeData?.path?.geometry;
    if (!g || g.type !== "LineString") return null;
    return coordsToLatLngs(g.coordinates);
  }, [routeData]);

  const pickRandom = useCallback(() => {
    if (zones.length < 2) return;
    const pool = zones.length > 10 ? internalZones : zones;
    if (pool.length < 2) return;
    const i = Math.floor(Math.random() * pool.length);
    let j = Math.floor(Math.random() * (pool.length - 1));
    if (j >= i) j++;
    setOrigin(pool[i].zone_id);
    setDestination(pool[j].zone_id);
  }, [zones, internalZones]);

  const swap = useCallback(() => {
    setOrigin(destination);
    setDestination(origin);
  }, [origin, destination]);

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-[380px] flex flex-col border-r border-gray-200 bg-white overflow-hidden">
        <div className="px-3 py-2 border-b border-gray-200 space-y-2">
          <div className="text-xs font-semibold text-gray-700">Trasa mezi zónami</div>

          <div>
            <label className="text-[10px] text-gray-500 uppercase tracking-wide">Počátek</label>
            <select
              value={origin ?? ""}
              onChange={(e) => setOrigin(e.target.value ? Number(e.target.value) : null)}
              className="w-full border border-gray-300 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">— Vyberte zónu —</option>
              {internalZones.length > 0 && (
                <optgroup label="Městské části">
                  {internalZones.map((z) => (
                    <option key={z.zone_id} value={z.zone_id}>
                      {z.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {externalZones.length > 0 && (
                <optgroup label="Externí (gateway)">
                  {externalZones.map((z) => (
                    <option key={z.zone_id} value={z.zone_id}>
                      {z.name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex-1 border-t border-gray-200" />
            <button
              onClick={swap}
              className="text-xs text-gray-400 hover:text-gray-800 transition px-1"
              title="Prohodit"
            >
              ⇅
            </button>
            <div className="flex-1 border-t border-gray-200" />
          </div>

          <div>
            <label className="text-[10px] text-gray-500 uppercase tracking-wide">Cíl</label>
            <select
              value={destination ?? ""}
              onChange={(e) => setDestination(e.target.value ? Number(e.target.value) : null)}
              className="w-full border border-gray-300 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">— Vyberte zónu —</option>
              {internalZones.length > 0 && (
                <optgroup label="Městské části">
                  {internalZones.map((z) => (
                    <option key={z.zone_id} value={z.zone_id}>
                      {z.name}
                    </option>
                  ))}
                </optgroup>
              )}
              {externalZones.length > 0 && (
                <optgroup label="Externí (gateway)">
                  {externalZones.map((z) => (
                    <option key={z.zone_id} value={z.zone_id}>
                      {z.name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

          <button
            onClick={pickRandom}
            disabled={zones.length < 2}
            className="w-full rounded bg-gray-100 border border-gray-200 text-gray-700 text-xs font-medium py-1.5 hover:bg-gray-200 transition disabled:opacity-50"
          >
            Náhodná trasa
          </button>
        </div>

        {zonesLoading && <div className="p-4 text-sm text-gray-400">Načítání zón...</div>}
        {routeLoading && <div className="p-4 text-sm text-gray-400">Počítám trasu...</div>}
        {routeError && (
          <div className="p-4 text-sm text-red-600">Chyba: {(routeError as Error).message}</div>
        )}

        {routeData && (
          <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs">
            <div className="rounded-lg border border-gray-200 p-2">
              <div className="font-semibold text-gray-800 mb-1">
                {routeData.origin.name} → {routeData.destination.name}
              </div>
              <table className="w-full text-[11px]">
                <tbody>
                  <tr>
                    <td className="text-gray-500">Cestovní čas</td>
                    <td className="text-right font-semibold text-gray-800">
                      {(routeData.path.travel_time / 60).toFixed(1)} min
                    </td>
                  </tr>
                  {routeData.path.link_time != null && (
                    <tr>
                      <td className="text-gray-400 pl-2">z toho kongesce (BPR)</td>
                      <td className="text-right text-gray-500">
                        {(routeData.path.link_time / 60).toFixed(1)} min
                      </td>
                    </tr>
                  )}
                  {routeData.path.intersection_delay != null && routeData.path.intersection_delay > 0 && (
                    <tr>
                      <td className="text-gray-400 pl-2">z toho křižovatky</td>
                      <td className="text-right text-gray-500">
                        {(routeData.path.intersection_delay / 60).toFixed(1)} min
                      </td>
                    </tr>
                  )}
                  {routeData.path.free_flow_time != null && (
                    <tr>
                      <td className="text-gray-400 pl-2">volný tok (ff)</td>
                      <td className="text-right text-gray-400">
                        {(routeData.path.free_flow_time / 60).toFixed(1)} min
                      </td>
                    </tr>
                  )}
                  <tr>
                    <td className="text-gray-500">Vzdálenost</td>
                    <td className="text-right font-medium">
                      {(routeData.path.distance / 1000).toFixed(2)} km
                    </td>
                  </tr>
                  <tr>
                    <td className="text-gray-500">Počet linků</td>
                    <td className="text-right">{routeData.path.n_links}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {routeData.streets.length > 0 && (
              <div>
                <div className="font-semibold text-gray-800 mb-1">Ulice na trase</div>
                <div className="space-y-0.5">
                  {routeData.streets.map((s) => (
                    <div key={s.name} className="flex justify-between text-[11px]">
                      <span className="text-gray-600 truncate">{s.name}</span>
                      <span className="text-gray-400 ml-2 shrink-0">{(s.distance / 1000).toFixed(2)} km</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {routeData.bias_stations.length > 0 && (
              <div>
                <div className="font-semibold text-gray-800 mb-1">
                  Pentlogramové body na trase ({routeData.bias_stations.length})
                </div>
                <div className="space-y-0.5">
                  {routeData.bias_stations.map((s) => (
                    <div key={s.link_id} className="flex items-center justify-between text-[11px]">
                      <span className="text-gray-600 truncate max-w-[140px]">{s.name || `Link ${s.link_id}`}</span>
                      <span className="flex items-center gap-1">
                        <span className="text-gray-400">{fmt(s.observed)}</span>
                        <span className="text-gray-300">/</span>
                        <span className="text-gray-400">{fmt(s.modeled)}</span>
                        <span
                          className="font-mono text-[10px] w-10 text-right font-semibold"
                          style={{ color: biasColor(s.ratio) }}
                        >
                          {s.ratio.toFixed(2)}
                        </span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        <MapContainer key={containerKey} center={mapCenter} zoom={12} className="h-full w-full">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {pathCoords && (
            <Polyline
              positions={pathCoords}
              pathOptions={{ color: "#148F77", weight: 5, opacity: 0.85 }}
            />
          )}

          {routeData?.bias_stations.map((s) => (
            <BiasMarker key={s.link_id} s={s} />
          ))}

          {routeData?.origin.coords && (
            <CircleMarker
              center={[routeData.origin.coords[1], routeData.origin.coords[0]]}
              radius={10}
              pathOptions={{ color: "#fff", fillColor: "#27AE60", fillOpacity: 0.9, weight: 2 }}
            >
              <Tooltip permanent direction="top" offset={[0, -12]}>
                <span style={{ fontWeight: 600 }}>{routeData.origin.name}</span>
              </Tooltip>
            </CircleMarker>
          )}

          {routeData?.destination.coords && (
            <CircleMarker
              center={[routeData.destination.coords[1], routeData.destination.coords[0]]}
              radius={10}
              pathOptions={{ color: "#fff", fillColor: "#C0392B", fillOpacity: 0.9, weight: 2 }}
            >
              <Tooltip permanent direction="bottom" offset={[0, 12]}>
                <span style={{ fontWeight: 600 }}>{routeData.destination.name}</span>
              </Tooltip>
            </CircleMarker>
          )}
        </MapContainer>

        <div className="absolute bottom-6 right-4 z-[1000] rounded-lg bg-white border border-gray-200 p-3 shadow-lg text-xs">
          <div className="font-semibold text-gray-700 mb-2">Legenda</div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-1 w-6 rounded" style={{ background: "#148F77" }} />
            <span>Nejkratší cesta</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-3 w-3 rounded-full" style={{ background: "#27AE60" }} />
            <span>Počátek</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block h-3 w-3 rounded-full" style={{ background: "#C0392B" }} />
            <span>Cíl</span>
          </div>
          <div className="mt-1 pt-1 border-t border-gray-200 font-medium text-gray-600">
            Bias na trase
          </div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "#C0392B" }} />
            <span>Nadhodnocení</span>
          </div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "#27AE60" }} />
            <span>Dobrá shoda</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: "#148F77" }} />
            <span>Podhodnocení</span>
          </div>
        </div>
      </div>
    </div>
  );
}
