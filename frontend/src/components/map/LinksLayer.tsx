import { useEffect, useMemo } from "react";
import { GeoJSON, Marker } from "react-leaflet";
import L from "leaflet";
import type { GeoJSONFeature, GeoJSONFeatureCollection, LinkProperties, ScenarioLink } from "../../types";
import { useScenarioStore } from "../../stores/scenarioStore";

type ScenarioDirection = ScenarioLink["direction"];

function pathLengthLngLat(coords: number[][]): number {
  let len = 0;
  for (let i = 1; i < coords.length; i++) {
    const dx = coords[i][0] - coords[i - 1][0];
    const dy = coords[i][1] - coords[i - 1][1];
    len += Math.sqrt(dx * dx + dy * dy);
  }
  return len;
}

/** Vzdálenost dvou bodů WGS84 v metrech (střední šířka Brna je v pořádku). */
function haversineMeters(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const R = 6371000;
  const φ1 = (lat1 * Math.PI) / 180;
  const φ2 = (lat2 * Math.PI) / 180;
  const dφ = ((lat2 - lat1) * Math.PI) / 180;
  const dλ = ((lng2 - lng1) * Math.PI) / 180;
  const s =
    Math.sin(dφ / 2) * Math.sin(dφ / 2) +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(dλ / 2) * Math.sin(dλ / 2);
  return 2 * R * Math.asin(Math.sqrt(Math.min(1, s)));
}

/**
 * Druhá fáze po spojení podle společných uzlů: sloučí markery v geografické blízkosti
 * (křižovatky, paralelní pruhy, mírně posunuté souřadnice uzlu).
 */
const SCENARIO_MARKER_MERGE_DISTANCE_M = 175;

function mergeMarkersByProximity(
  markers: { key: number; pos: [number, number] }[],
  maxDistM: number,
): { key: number; pos: [number, number] }[] {
  const n = markers.length;
  if (n <= 1) return markers;

  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x: number): number {
    let p = parent[x];
    while (p !== parent[p]) {
      parent[p] = parent[parent[p]];
      p = parent[p];
    }
    return p;
  }
  function union(a: number, b: number) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  }

  for (let i = 0; i < n; i++) {
    const [lati, lngi] = markers[i].pos;
    for (let j = i + 1; j < n; j++) {
      const [latj, lngj] = markers[j].pos;
      if (haversineMeters(lati, lngi, latj, lngj) <= maxDistM) union(i, j);
    }
  }

  const buckets = new Map<number, { sumLat: number; sumLng: number; count: number; minKey: number }>();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    const [lat, lng] = markers[i].pos;
    const k = markers[i].key;
    let b = buckets.get(r);
    if (!b) {
      b = { sumLat: 0, sumLng: 0, count: 0, minKey: k };
      buckets.set(r, b);
    }
    b.sumLat += lat;
    b.sumLng += lng;
    b.count += 1;
    if (k < b.minKey) b.minKey = k;
  }

  return Array.from(buckets.values()).map((b) => ({
    key: b.minKey,
    pos: [b.sumLat / b.count, b.sumLng / b.count] as [number, number],
  }));
}

/** Průnik zobrazení scénáře s mapovým segmentem podle směru v síti a ve scénáři. */
export function scenarioTouchesLinkFeature(
  scenarioDir: ScenarioDirection,
  networkDir: number | undefined,
): boolean {
  const nd = networkDir ?? 0;
  if (scenarioDir === "both") return true;
  if (nd === 1) return scenarioDir === "ab";
  if (nd === -1) return scenarioDir === "ba";
  return true;
}

/** Geometrie na mapě — celý úsek. Obousměrný střed úseku v OS má jednu linii pro oba toky; „protisměr“
 *  není druhá polovina souřadnic (to je pořád táž geometrická polyline), ale druhý orientovaný tok v grafu. */
function highlightGeometryForScenario(
  feature: GeoJSONFeature,
  scenarioDir: ScenarioDirection,
): GeoJSONFeature["geometry"] | null {
  const p = feature.properties as unknown as LinkProperties;
  const nd = p.direction ?? 0;
  const g = feature.geometry;

  if (scenarioDir === "ab" && nd === -1) return null;
  if (scenarioDir === "ba" && nd === 1) return null;
  return g;
}

function collectMarkerMidpoints(geo: GeoJSONFeatureCollection): { key: number; pos: [number, number] }[] {
  const midpoints: { id: number; pos: [number, number]; endpoints: string[] }[] = [];

  for (const f of geo.features) {
    const geom = f.geometry as { type: string; coordinates: unknown };
    let coords: number[][] | null = null;

    if (geom.type === "LineString") {
      coords = geom.coordinates as number[][];
    } else if (geom.type === "MultiLineString") {
      const lines = geom.coordinates as number[][][];
      if (lines.length) {
        coords = lines.reduce((a, b) => (pathLengthLngLat(a) >= pathLengthLngLat(b) ? a : b), lines[0]);
      }
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
    const ra = find(a);
    const rb = find(b);
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

  return mergeMarkersByProximity(result, SCENARIO_MARKER_MERGE_DISTANCE_M);
}

export const VOC_SCALE = [
  { min: 1.0,  color: "#C0392B", label: "LOS F  (> 1.00)", los: "F" },
  { min: 0.9,  color: "#E74C3C", label: "LOS E  (0.90–1.00)", los: "E" },
  { min: 0.75, color: "#CA6F1E", label: "LOS D  (0.75–0.90)", los: "D" },
  { min: 0.55, color: "#D4AC0D", label: "LOS C  (0.55–0.75)", los: "C" },
  { min: 0.35, color: "#2ECC71", label: "LOS B  (0.35–0.55)", los: "B" },
  { min: 0.0,  color: "#27AE60", label: "LOS A  (< 0.35)", los: "A" },
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
  color: "#CA6F1E",
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
  color: "#D4AC0D",
  opacity: 0.5,
  dashArray: "4 6",
};

const MARKER_ICON_CLASS = "scenario-closure-marker";

// Lucide "construction" icon paths (ISC license) — závora/barrikáda
const LUCIDE_CONSTRUCTION = `
  <rect x="2" y="6" width="20" height="8" rx="1"/>
  <path d="M17 14v7"/><path d="M7 14v7"/>
  <path d="M17 3v3"/><path d="M7 3v3"/>
  <path d="M10 14 2.3 6.3"/>
  <path d="m14 6 7.7 7.7"/>
  <path d="m8 6 8 8"/>`;

// Lucide "traffic-cone" icon paths (ISC license)
const LUCIDE_TRAFFIC_CONE = `
  <path d="M16.05 10.966a5 2.5 0 0 1-8.1 0"/>
  <path d="m16.923 14.049 4.48 2.04a1 1 0 0 1 .001 1.831l-8.574 3.9a2 2 0 0 1-1.66 0l-8.574-3.91a1 1 0 0 1 0-1.83l4.484-2.04"/>
  <path d="M16.949 14.14a5 2.5 0 1 1-9.9 0L10.063 3.5a2 2 0 0 1 3.874 0z"/>
  <path d="M9.194 6.57a5 2.5 0 0 0 5.61 0"/>`;

// viewBox 72×72 — střed trojúhelníku (vrcholy (36,4),(68,64),(4,64)) je (36, 44); Lucide má viewBox 24×24 → střed ikony (12,12) dáme na (36,44).
function buildTriangleMarkerSvg(
  triFill: string, triStroke: string,
  iconPaths: string, iconStroke: string,
): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
  <polygon points="36,4 68,64 4,64" fill="${triFill}" stroke="${triStroke}" stroke-width="3.5" stroke-linejoin="round"/>
  <g transform="translate(36,44) translate(-12,-12)" fill="none" stroke="${iconStroke}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    ${iconPaths}
  </g>
</svg>`;
}

export function buildFullClosureSvg(planning: boolean): string {
  return buildTriangleMarkerSvg(
    planning ? "#94a3b8" : "#dc2626",
    planning ? "#475569" : "#7f1d1d",
    LUCIDE_CONSTRUCTION,
    "#fff",
  );
}

export function buildLaneConeSvg(planning: boolean): string {
  return buildTriangleMarkerSvg(
    planning ? "#fbbf24" : "#ea580c",
    planning ? "#92400e" : "#7c2d12",
    LUCIDE_TRAFFIC_CONE,
    "#fff",
  );
}

/** Spodní střed trojúhelníku v souřadnicích Leaflet ikony (cca −⅓ oproti původním 56 px). */
const MARKER_SIZE = 37;
const MARKER_ANCHOR_X = MARKER_SIZE * (36 / 72);
const MARKER_ANCHOR_Y = MARKER_SIZE * (64 / 72);

const fullClosureSimIcon = L.divIcon({
  html: buildFullClosureSvg(false),
  className: MARKER_ICON_CLASS,
  iconSize: [MARKER_SIZE, MARKER_SIZE],
  iconAnchor: [MARKER_ANCHOR_X, MARKER_ANCHOR_Y],
});

const fullClosurePlanIcon = L.divIcon({
  html: buildFullClosureSvg(true),
  className: MARKER_ICON_CLASS,
  iconSize: [MARKER_SIZE, MARKER_SIZE],
  iconAnchor: [MARKER_ANCHOR_X, MARKER_ANCHOR_Y],
});

const laneConeSimIcon = L.divIcon({
  html: buildLaneConeSvg(false),
  className: MARKER_ICON_CLASS,
  iconSize: [MARKER_SIZE, MARKER_SIZE],
  iconAnchor: [MARKER_ANCHOR_X, MARKER_ANCHOR_Y],
});

const laneConePlanIcon = L.divIcon({
  html: buildLaneConeSvg(true),
  className: MARKER_ICON_CLASS,
  iconSize: [MARKER_SIZE, MARKER_SIZE],
  iconAnchor: [MARKER_ANCHOR_X, MARKER_ANCHOR_Y],
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
  const netDir = p.direction ?? 0;

  const voc = p.VOC_max ?? 0;
  const vocColor = getVocColor(voc);
  const los = p.LOS_max ?? getLos(voc);

  const links = useScenarioStore.getState().links;
  const inScenario = links.some(
    (l) => l.link_id === p.link_id && scenarioTouchesLinkFeature(l.direction, netDir),
  );
  const btnLabel = inScenario ? "Odebrat ze scénáře" : "Přidat do scénáře";
  const btnColor = inScenario ? "#ef4444" : "#2563eb";
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
          <td>A\u2192B</td><td>B\u2192A</td><td style="font-weight:600;color:#111827">Celkem</td>
        </tr>
        <tr>
          <td>${fmt(p.wd_daily_ab)}</td>
          <td>${fmt(p.wd_daily_ba)}</td>
          <td style="font-weight:700;font-size:14px">${fmt(p.wd_daily_tot)}</td>
        </tr>

        <tr><td colspan="3" style="font-weight:600;padding-top:8px;color:#374151">Spickova hodina (PCE/h, K=${kPct}%)</td></tr>
        <tr style="color:#6b7280;font-size:11px">
          <td>A\u2192B</td><td>B\u2192A</td><td style="font-weight:600;color:#111827">Max</td>
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
        link_id: ${p.link_id}${netDir === 0 ? " · obousměrný úsek" : netDir === 1 ? " · jednosměrka A→B" : " · jednosměrka B→A"}
      </div>

      <button
        ${SCENARIO_BTN_ATTR}="${p.link_id}"
        data-action="${btnAction}"
        data-name="${displayName.replace(/"/g, "&quot;")}"
        data-link-type="${p.link_type}"
        data-lanes="${p.lanes}"
        data-network-direction="${Number.isFinite(netDir) ? netDir : ""}"
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
        const ndRaw = btn.getAttribute("data-network-direction");
        const network_direction =
          ndRaw !== null && ndRaw !== "" && !Number.isNaN(Number(ndRaw)) ? Number(ndRaw) : undefined;
        store.addLink({
          link_id: linkId,
          name: btn.getAttribute("data-name") || null,
          link_type: btn.getAttribute("data-link-type") || "",
          lanes: Number(btn.getAttribute("data-lanes")) || 2,
          network_direction,
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

  const { closedFeatures, reducedFeatures } = useMemo(() => {
    const closed: GeoJSONFeature[] = [];
    const reduced: GeoJSONFeature[] = [];

    for (const sl of scenarioLinks) {
      for (const f of data.features) {
        const lp = f.properties as unknown as LinkProperties;
        if (lp.link_id !== sl.link_id) continue;
        if (!scenarioTouchesLinkFeature(sl.direction, lp.direction)) continue;
        const hg = highlightGeometryForScenario(f as GeoJSONFeature, sl.direction);
        if (!hg) continue;
        const nf: GeoJSONFeature = { ...f, geometry: hg };
        if (sl.closure_type === "full") closed.push(nf);
        else reduced.push(nf);
      }
    }

    return {
      closedFeatures: { type: "FeatureCollection", features: closed } as GeoJSONFeatureCollection,
      reducedFeatures: { type: "FeatureCollection", features: reduced } as GeoJSONFeatureCollection,
    };
  }, [data, scenarioLinks]);

  const closedMarkers = useMemo(() => collectMarkerMidpoints(closedFeatures), [closedFeatures]);
  const reducedMarkers = useMemo(() => collectMarkerMidpoints(reducedFeatures), [reducedFeatures]);

  const keyBase = useMemo(
    () =>
      `${isSimulated ? "sim" : "plan"}-${scenarioLinks
        .map((l) => `${l.link_id}:${l.closure_type}:${l.direction}`)
        .join(",")}`,
    [isSimulated, scenarioLinks],
  );

  if (closedFeatures.features.length === 0 && reducedFeatures.features.length === 0) return null;

  const closedStyle = isSimulated ? CLOSED_ROAD_STYLE : PLANNING_CLOSED_STYLE;
  const reducedStyle = isSimulated ? LANE_REDUCTION_STYLE : PLANNING_REDUCED_STYLE;

  const fullIcon = isSimulated ? fullClosureSimIcon : fullClosurePlanIcon;
  const laneIcon = isSimulated ? laneConeSimIcon : laneConePlanIcon;

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
      {closedMarkers.map(({ key, pos }, i) => (
        <Marker
          key={`m-full-${key}-${i}-${keyBase}`}
          position={pos}
          icon={fullIcon}
          interactive={false}
        />
      ))}
      {reducedMarkers.map(({ key, pos }, i) => (
        <Marker
          key={`m-lane-${key}-${i}-${keyBase}`}
          position={pos}
          icon={laneIcon}
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
