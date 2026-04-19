import { useEffect, useMemo } from "react";
import { GeoJSON, Marker } from "react-leaflet";
import L from "leaflet";
import type { GeoJSONFeatureCollection, LinkProperties } from "../../types";
import { useScenarioStore } from "../../stores/scenarioStore";

export const VOC_SCALE = [
  { min: 1.0,  color: "#7f1d1d", label: "LOS F  (> 1.00)", los: "F" },
  { min: 0.9,  color: "#dc2626", label: "LOS E  (0.90–1.00)", los: "E" },
  { min: 0.75, color: "#f59e0b", label: "LOS D  (0.75–0.90)", los: "D" },
  { min: 0.55, color: "#fee08b", label: "LOS C  (0.55–0.75)", los: "C" },
  { min: 0.35, color: "#a3e635", label: "LOS B  (0.35–0.55)", los: "B" },
  { min: 0.0,  color: "#16a34a", label: "LOS A  (< 0.35)", los: "A" },
];

function getVocColor(voc: number) {
  for (const { min, color } of VOC_SCALE) if (voc >= min) return color;
  return "#ccc";
}

function getLos(voc: number): string {
  for (const { min, los } of VOC_SCALE) if (voc >= min) return los;
  return "A";
}

function style(feature: GeoJSON.Feature | undefined) {
  const p = feature?.properties as LinkProperties | undefined;
  const voc = p?.VOC_max;
  const vol = p?.wd_daily_tot ?? 0;
  return {
    color: voc != null && !isNaN(voc) ? getVocColor(voc) : "#ccc",
    weight: Math.max(2, Math.min(7, Math.log10(Math.max(vol, 1)) * 1.5)),
    opacity: 0.85,
  };
}

const HIT_STYLE = { weight: 20, opacity: 0, color: "#000" };

const CLOSED_ROAD_STYLE = {
  weight: 7,
  color: "#18181b",
  opacity: 0.95,
};

const LANE_REDUCTION_STYLE = {
  weight: 6,
  color: "#f97316",
  opacity: 0.8,
  dashArray: "8 4",
};

const PLANNING_CLOSED_STYLE = {
  weight: 5,
  color: "#6b7280",
  opacity: 0.6,
  dashArray: "6 4",
};

const PLANNING_REDUCED_STYLE = {
  weight: 5,
  color: "#d97706",
  opacity: 0.5,
  dashArray: "4 6",
};

const NO_ENTRY_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
  <circle cx="14" cy="14" r="13" fill="#dc2626" stroke="#fff" stroke-width="2"/>
  <rect x="5" y="11.5" width="18" height="5" rx="1.5" fill="#fff"/>
</svg>`;

const noEntryIcon = L.divIcon({
  html: NO_ENTRY_SVG,
  className: "",
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

const PLANNING_MARKER_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22">
  <polygon points="12,2 22,20 2,20" fill="#f59e0b" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/>
  <rect x="11" y="8" width="2" height="6" rx="0.5" fill="#fff"/>
  <circle cx="12" cy="16.5" r="1.2" fill="#fff"/>
</svg>`;

const planningIcon = L.divIcon({
  html: PLANNING_MARKER_SVG,
  className: "",
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});

function getLineMidpoint(coords: number[][]): [number, number] | null {
  if (!coords || coords.length === 0) return null;
  if (coords.length === 1) return [coords[0][1], coords[0][0]];

  let totalLen = 0;
  const segLens: number[] = [];
  for (let i = 1; i < coords.length; i++) {
    const dx = coords[i][0] - coords[i - 1][0];
    const dy = coords[i][1] - coords[i - 1][1];
    const seg = Math.sqrt(dx * dx + dy * dy);
    segLens.push(seg);
    totalLen += seg;
  }

  let target = totalLen / 2;
  for (let i = 0; i < segLens.length; i++) {
    if (target <= segLens[i]) {
      const t = target / segLens[i];
      const lng = coords[i][0] + t * (coords[i + 1][0] - coords[i][0]);
      const lat = coords[i][1] + t * (coords[i + 1][1] - coords[i][1]);
      return [lat, lng];
    }
    target -= segLens[i];
  }

  const last = coords[coords.length - 1];
  return [last[1], last[0]];
}

export function linkDisplayName(p: { name?: string | null; osm_ref?: string | null; link_type?: string; link_id?: number }): string {
  if (p.name) return p.name;
  if (p.osm_ref) return `${p.osm_ref} (${p.link_type ?? "road"})`;
  if (p.link_type) return `${p.link_type} #${p.link_id ?? ""}`;
  return `#${p.link_id ?? "?"}`;
}

function fmt(n: number | null | undefined, decimals = 0): string {
  if (n == null || isNaN(n)) return "—";
  return decimals > 0 ? n.toFixed(decimals) : Math.round(n).toLocaleString();
}

const SCENARIO_BTN_ATTR = "data-scenario-link-id";

function onEachFeature(feature: GeoJSON.Feature, layer: L.Layer) {
  const p = feature.properties as unknown as LinkProperties;

  const voc = p.VOC_max ?? 0;
  const vocColor = getVocColor(voc);
  const los = p.LOS_max ?? getLos(voc);

  const inScenario = useScenarioStore.getState().links.some((l) => l.link_id === p.link_id);
  const btnLabel = inScenario ? "Odebrat ze scénáře" : "Přidat do scénáře";
  const btnColor = inScenario ? "#dc2626" : "#2563eb";
  const btnAction = inScenario ? "remove" : "add";

  const displayName = linkDisplayName(p);

  const peakAB = p.peak_hour_vol_AB ?? 0;
  const peakBA = p.peak_hour_vol_BA ?? 0;
  const peakMax = Math.max(peakAB, peakBA);
  const kPct = ((p.K_factor ?? 0.10) * 100).toFixed(0);

  layer.bindPopup(
    `<div style="min-width:260px;font-size:13px;line-height:1.6">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px;border-bottom:1px solid #e5e7eb;padding-bottom:4px">
        ${displayName}
        <span style="float:right;font-weight:400;color:#6b7280">${p.link_type}</span>
      </div>

      <table style="width:100%;border-collapse:collapse">
        <tr><td colspan="3" style="font-weight:600;padding-top:4px;color:#374151">Denni provoz (voz/den)</td></tr>
        <tr style="color:#6b7280;font-size:11px">
          <td>A\u2192B</td><td>B\u2192A</td><td style="font-weight:600;color:#111">Celkem</td>
        </tr>
        <tr>
          <td>${fmt(p.wd_daily_ab)}</td>
          <td>${fmt(p.wd_daily_ba)}</td>
          <td style="font-weight:700;font-size:14px">${fmt(p.wd_daily_tot)}</td>
        </tr>

        <tr><td colspan="3" style="font-weight:600;padding-top:8px;color:#374151">Spickova hodina (PCE/h, K=${kPct}%)</td></tr>
        <tr style="color:#6b7280;font-size:11px">
          <td>A\u2192B</td><td>B\u2192A</td><td style="font-weight:600;color:#111">Max</td>
        </tr>
        <tr>
          <td>${fmt(peakAB)}</td>
          <td>${fmt(peakBA)}</td>
          <td style="font-weight:700">${fmt(peakMax)}</td>
        </tr>

        <tr><td colspan="3" style="font-weight:600;padding-top:8px;color:#374151">Parametry</td></tr>
        <tr>
          <td style="color:#6b7280">Rychlost</td>
          <td colspan="2">${fmt(p.speed)} km/h</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Kapacita</td>
          <td colspan="2">${fmt(p.capacity)} voz/h/smer</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Pruhy</td>
          <td colspan="2">${fmt(p.lanes)}</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Delka</td>
          <td colspan="2">${fmt(p.distance)} m</td>
        </tr>

        <tr><td colspan="3" style="font-weight:600;padding-top:8px;color:#374151">Zatizeni</td></tr>
        <tr>
          <td style="color:#6b7280">V/C</td>
          <td><span style="color:${vocColor};font-weight:600">${fmt(voc, 2)}</span></td>
          <td><span style="display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:700;color:#fff;background:${vocColor}">LOS ${los}</span></td>
        </tr>
        <tr>
          <td style="color:#6b7280">Delay faktor</td>
          <td colspan="2">${fmt(p.Delay_factor_Max, 2)}\u00d7</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Cest. cas (s)</td>
          <td colspan="2">${fmt(p.Congested_Time_Max, 1)}</td>
        </tr>
      </table>

      <div style="margin-top:6px;font-size:10px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:3px">
        link_id: ${p.link_id}
      </div>

      <button
        ${SCENARIO_BTN_ATTR}="${p.link_id}"
        data-action="${btnAction}"
        data-name="${displayName.replace(/"/g, "&quot;")}"
        data-link-type="${p.link_type}"
        data-lanes="${p.lanes}"
        style="
          margin-top:8px;width:100%;padding:6px 0;
          background:${btnColor};color:#fff;border:none;border-radius:6px;
          font-size:12px;font-weight:600;cursor:pointer;
        "
      >${btnLabel}</button>
    </div>`,
    { maxWidth: 320 },
  );
}

function useScenarioPopupHandler() {
  useEffect(() => {
    function handler(e: MouseEvent) {
      const btn = (e.target as HTMLElement).closest<HTMLElement>(`[${SCENARIO_BTN_ATTR}]`);
      if (!btn) return;

      const linkId = Number(btn.getAttribute(SCENARIO_BTN_ATTR));
      const action = btn.getAttribute("data-action");
      const store = useScenarioStore.getState();

      if (action === "remove") {
        store.removeLink(linkId);
      } else {
        store.addLink({
          link_id: linkId,
          name: btn.getAttribute("data-name") || null,
          link_type: btn.getAttribute("data-link-type") || "",
          lanes: Number(btn.getAttribute("data-lanes")) || 2,
        });
      }

      const popup = btn.closest(".leaflet-popup");
      if (popup) {
        const closeBtn = popup.querySelector<HTMLElement>(".leaflet-popup-close-button");
        closeBtn?.click();
      }
    }

    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);
}

export function ScenarioHighlightLayer({ data }: { data: GeoJSONFeatureCollection }) {
  const scenarioLinks = useScenarioStore((s) => s.links);
  const viewMode = useScenarioStore((s) => s.viewMode);
  const hasResults = useScenarioStore((s) => s.scenarioResults !== null);

  const isSimulated = viewMode === "scenario" && hasResults;

  const closedIds = useMemo(
    () => new Set(scenarioLinks.filter((l) => l.closure_type === "full").map((l) => l.link_id)),
    [scenarioLinks],
  );
  const reducedIds = useMemo(
    () => new Set(scenarioLinks.filter((l) => l.closure_type === "lanes").map((l) => l.link_id)),
    [scenarioLinks],
  );

  const closedFeatures = useMemo<GeoJSONFeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: data.features.filter((f) => closedIds.has((f.properties as unknown as LinkProperties).link_id)),
    }),
    [data, closedIds],
  );

  const reducedFeatures = useMemo<GeoJSONFeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: data.features.filter((f) => reducedIds.has((f.properties as unknown as LinkProperties).link_id)),
    }),
    [data, reducedIds],
  );

  const markerPositions = useMemo(() => {
    const midpoints: { id: number; pos: [number, number]; endpoints: string[] }[] = [];

    for (const f of closedFeatures.features) {
      const geom = f.geometry as { type: string; coordinates: unknown };
      let coords: number[][] | null = null;

      if (geom.type === "LineString") {
        coords = geom.coordinates as number[][];
      } else if (geom.type === "MultiLineString") {
        const lines = geom.coordinates as number[][][];
        coords = lines.reduce((a, b) => (a.length >= b.length ? a : b), lines[0]);
      }

      if (!coords || coords.length === 0) continue;

      const mid = getLineMidpoint(coords);
      if (!mid) continue;

      const first = coords[0];
      const last = coords[coords.length - 1];
      const coordKey = (c: number[]) => `${c[0].toFixed(6)},${c[1].toFixed(6)}`;

      midpoints.push({
        id: (f.properties as unknown as LinkProperties).link_id,
        pos: mid,
        endpoints: [coordKey(first), coordKey(last)],
      });
    }

    if (midpoints.length === 0) return [];

    const parent = new Map<number, number>();
    midpoints.forEach((m) => parent.set(m.id, m.id));

    function find(x: number): number {
      while (parent.get(x) !== x) {
        parent.set(x, parent.get(parent.get(x)!)!);
        x = parent.get(x)!;
      }
      return x;
    }
    function union(a: number, b: number) {
      const ra = find(a), rb = find(b);
      if (ra !== rb) parent.set(ra, rb);
    }

    const endpointMap = new Map<string, number>();
    for (const m of midpoints) {
      for (const ep of m.endpoints) {
        const prev = endpointMap.get(ep);
        if (prev !== undefined) {
          union(prev, m.id);
        }
        endpointMap.set(ep, m.id);
      }
    }

    const groups = new Map<number, [number, number][]>();
    for (const m of midpoints) {
      const root = find(m.id);
      if (!groups.has(root)) groups.set(root, []);
      groups.get(root)!.push(m.pos);
    }

    const result: { key: number; pos: [number, number] }[] = [];
    for (const [root, positions] of groups) {
      const lat = positions.reduce((s, p) => s + p[0], 0) / positions.length;
      const lng = positions.reduce((s, p) => s + p[1], 0) / positions.length;
      result.push({ key: root, pos: [lat, lng] });
    }

    return result;
  }, [closedFeatures]);

  const keyBase = useMemo(
    () => `${isSimulated ? "sim" : "plan"}-${scenarioLinks.map((l) => `${l.link_id}:${l.closure_type}`).join(",")}`,
    [isSimulated, scenarioLinks],
  );

  if (closedFeatures.features.length === 0 && reducedFeatures.features.length === 0) return null;

  const closedStyle = isSimulated ? CLOSED_ROAD_STYLE : PLANNING_CLOSED_STYLE;
  const reducedStyle = isSimulated ? LANE_REDUCTION_STYLE : PLANNING_REDUCED_STYLE;

  return (
    <>
      {closedFeatures.features.length > 0 && (
        <GeoJSON
          key={`closed-${keyBase}`}
          data={closedFeatures as never}
          style={closedStyle}
          interactive={false}
        />
      )}
      {reducedFeatures.features.length > 0 && (
        <GeoJSON
          key={`reduced-${keyBase}`}
          data={reducedFeatures as never}
          style={reducedStyle}
          interactive={false}
        />
      )}
      {markerPositions.map(({ key, pos }) => (
        <Marker
          key={`marker-${key}-${isSimulated ? "sim" : "plan"}`}
          position={pos}
          icon={isSimulated ? noEntryIcon : planningIcon}
          interactive={false}
        />
      ))}
    </>
  );
}

export function LinksLayer({ data, dataKey }: { data: GeoJSONFeatureCollection; dataKey?: string }) {
  useScenarioPopupHandler();

  const k = `${dataKey ?? "base"}-${data.features.length}`;
  return (
    <>
      <GeoJSON
        key={`hit-${k}`}
        data={data as never}
        style={HIT_STYLE}
        onEachFeature={onEachFeature}
      />
      <GeoJSON
        key={`vis-${k}`}
        data={data as never}
        style={style}
        interactive={false}
      />
    </>
  );
}
