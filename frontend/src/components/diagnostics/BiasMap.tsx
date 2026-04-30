import { useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import { useBiasData } from "../../api/hooks";
import { useLeafletMapLayout } from "../../map/useLeafletMapCenter";
import type { BiasClusterProperties, BiasStationProperties, BiasStationStatus } from "../../types";

const STATUS_COLORS: Record<BiasStationStatus, string> = {
  usable: "#27AE60",
  excluded: "#E67E22",
  unmatched: "#8E44AD",
};

const STATUS_LABELS: Record<BiasStationStatus, string> = {
  usable: "Použité (kalibrace)",
  excluded: "Vyloučené",
  unmatched: "Nespárované",
};

function biasColor(ratio: number, status?: BiasStationStatus): string {
  if (status === "unmatched") return STATUS_COLORS.unmatched;
  if (status === "excluded") return STATUS_COLORS.excluded;
  if (ratio > 1.2) return "#C0392B";
  if (ratio < 0.8) return "#148F77";
  return "#27AE60";
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

function StatusBadge({ status }: { status: BiasStationStatus }) {
  const labels: Record<BiasStationStatus, string> = {
    usable: "Použité",
    excluded: "Vyloučené",
    unmatched: "Nespárované",
  };
  return (
    <span
      style={{
        display: "inline-block",
        fontSize: 10,
        fontWeight: 600,
        padding: "1px 6px",
        borderRadius: 4,
        color: "#fff",
        background: STATUS_COLORS[status],
      }}
    >
      {labels[status]}
    </span>
  );
}

function StationPopup({ s }: { s: BiasStationProperties & { lat: number; lng: number } }) {
  const status = s.status ?? "usable";
  return (
    <div style={{ fontSize: 13, lineHeight: 1.6 }}>
      <div
        style={{
          fontWeight: 700,
          fontSize: 14,
          borderBottom: "1px solid #e5e7eb",
          paddingBottom: 4,
          marginBottom: 4,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span>{s.name || "Bez názvu"}</span>
        <StatusBadge status={status} />
      </div>
      {s.link_type && (
        <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>{s.link_type}</div>
      )}
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={{ color: "#6b7280" }}>Pozorované</td>
            <td style={{ fontWeight: 600, textAlign: "right" }}>{fmt(s.observed)} voz/den</td>
          </tr>
          {status !== "unmatched" && (
            <>
              <tr>
                <td style={{ color: "#6b7280" }}>Modelované</td>
                <td style={{ fontWeight: 600, textAlign: "right" }}>{fmt(s.modeled)} voz/den</td>
              </tr>
              <tr>
                <td style={{ color: "#6b7280" }}>Poměr</td>
                <td
                  style={{ fontWeight: 700, textAlign: "right", color: biasColor(s.ratio, status) }}
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
            </>
          )}
        </tbody>
      </table>
      {s.link_id != null && (
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
      )}
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

type VisibleStatuses = Record<BiasStationStatus, boolean>;

export function BiasMap() {
  const [clustered, setClustered] = useState(false);
  const [showAll, setShowAll] = useState(true);
  const [visible, setVisible] = useState<VisibleStatuses>({
    usable: true,
    excluded: true,
    unmatched: true,
  });

  const { data, isLoading, error } = useBiasData(clustered, 100, showAll && !clustered);
  const { center: mapCenter, containerKey } = useLeafletMapLayout();

  const stations = useMemo(() => {
    if (!data?.features || clustered) return [];
    return data.features.map((f) => {
      const p = f.properties as unknown as BiasStationProperties;
      const [lng, lat] = (f.geometry as { coordinates: [number, number] }).coordinates;
      return { ...p, lat, lng };
    });
  }, [data, clustered]);

  const filteredStations = useMemo(() => {
    if (!showAll) return stations;
    return stations.filter((s) => visible[s.status ?? "usable"]);
  }, [stations, showAll, visible]);

  const clusters = useMemo(() => {
    if (!data?.features || !clustered) return [];
    return data.features.map((f) => {
      const p = f.properties as unknown as BiasClusterProperties;
      const [lng, lat] = (f.geometry as { coordinates: [number, number] }).coordinates;
      return { ...p, lat, lng };
    });
  }, [data, clustered]);

  const summary = useMemo(() => {
    if (clustered) {
      if (!clusters.length) return null;
      const over = clusters.filter((s) => s.ratio > 1.2).length;
      const under = clusters.filter((s) => s.ratio < 0.8).length;
      const good = clusters.length - over - under;
      return { total: clusters.length, over, under, good };
    }
    if (!stations.length) return null;
    const counts: Record<BiasStationStatus, number> = { usable: 0, excluded: 0, unmatched: 0 };
    let over = 0;
    let under = 0;
    let good = 0;
    for (const s of stations) {
      const st = s.status ?? "usable";
      counts[st]++;
      if (st === "usable") {
        if (s.ratio > 1.2) over++;
        else if (s.ratio < 0.8) under++;
        else good++;
      }
    }
    return { total: stations.length, over, under, good, ...counts };
  }, [stations, clusters, clustered]);

  const toggleVisible = (status: BiasStationStatus) => {
    setVisible((prev) => ({ ...prev, [status]: !prev[status] }));
  };

  return (
    <div className="relative h-full w-full">
      <MapContainer key={containerKey} center={mapCenter} zoom={12} className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {!clustered &&
          filteredStations.map((s, i) => (
            <CircleMarker
              key={s.link_id ?? `unmatched-${i}`}
              center={[s.lat, s.lng]}
              radius={s.status === "unmatched" ? 5 : biasRadius(s.ratio, false)}
              pathOptions={{
                color: biasColor(s.ratio, s.status),
                fillColor: biasColor(s.ratio, s.status),
                fillOpacity: s.status === "usable" ? 0.7 : 0.5,
                weight: s.status === "usable" ? 1 : 2,
                dashArray: s.status === "unmatched" ? "4 3" : undefined,
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

      <div className="absolute bottom-6 right-4 z-[1000] rounded-lg bg-white border border-gray-200 p-3 shadow-lg text-xs">
        <div className="font-semibold text-gray-700 mb-2">Odchylka model/pozorování</div>

        <label className="flex items-center gap-2 mb-1 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(e) => {
              setShowAll(e.target.checked);
              if (!e.target.checked) setClustered(false);
            }}
            className="rounded"
          />
          <span>Všechna měření</span>
        </label>

        <label className="flex items-center gap-2 mb-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={clustered}
            onChange={(e) => {
              setClustered(e.target.checked);
              if (e.target.checked) setShowAll(false);
            }}
            className="rounded"
          />
          <span>Clustery (100 m)</span>
        </label>

        <div className="border-t border-gray-200 pt-2 mt-1 mb-1">
          <div className="text-[10px] font-medium text-gray-400 uppercase tracking-wider mb-1">Odchylka</div>
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ background: "#C0392B" }}
            />
            <span>Nadhodnocení (&gt; 120 %)</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ background: "#27AE60" }}
            />
            <span>Dobrá shoda (80–120 %)</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ background: "#148F77" }}
            />
            <span>Podhodnocení (&lt; 80 %)</span>
          </div>
        </div>

        {showAll && !clustered && (
          <div className="border-t border-gray-200 pt-2 mt-1 mb-1">
            <div className="text-[10px] font-medium text-gray-400 uppercase tracking-wider mb-1">Stav měření</div>
            {(["usable", "excluded", "unmatched"] as BiasStationStatus[]).map((status) => (
              <label key={status} className="flex items-center gap-2 mb-1 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={visible[status]}
                  onChange={() => toggleVisible(status)}
                  className="rounded"
                />
                <span
                  className="inline-block h-3 w-3 rounded-full border"
                  style={{
                    background: STATUS_COLORS[status],
                    borderColor: STATUS_COLORS[status],
                  }}
                />
                <span>{STATUS_LABELS[status]}</span>
                {summary && status in summary && (
                  <span className="text-gray-400 ml-auto">
                    {(summary as Record<string, number>)[status] ?? 0}
                  </span>
                )}
              </label>
            ))}
          </div>
        )}

        {summary && (
          <div className="mt-2 pt-2 border-t border-gray-200 text-gray-500">
            {showAll && !clustered ? (
              <>
                Celkem {summary.total} měření
              </>
            ) : (
              <>
                Celkem {summary.total} {clustered ? "clusterů" : "stanic"}:
                <span style={{ color: "#C0392B" }}> {summary.over}</span> /
                <span style={{ color: "#27AE60" }}> {summary.good}</span> /
                <span style={{ color: "#148F77" }}> {summary.under}</span>
              </>
            )}
          </div>
        )}
      </div>

      {isLoading && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-white px-4 py-2 shadow-lg text-xs text-gray-500">
          Načítání diagnostických dat…
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
