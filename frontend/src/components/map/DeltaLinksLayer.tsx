import { useMemo } from "react";
import { GeoJSON } from "react-leaflet";
import L from "leaflet";
import type { GeoJSONFeatureCollection, LinkProperties } from "../../types";
import { linkDisplayName, VOC_SCALE } from "./LinksLayer";

export const DELTA_SCALE = [
  { threshold: 50, color: "#C0392B", label: "> +50 %" },
  { threshold: 20, color: "#CA6F1E", label: "+20–50 %" },
  { threshold: 5, color: "#D4AC0D", label: "+5–20 %" },
  { threshold: -5, color: "#A6A6A6", label: "-5 – +5 %" },
  { threshold: -20, color: "#1ABC9C", label: "-5 – -20 %" },
  { threshold: -50, color: "#148F77", label: "-20 – -50 %" },
  { threshold: -Infinity, color: "#27AE60", label: "< -50 %" },
] as const;

function getDeltaColor(deltaPct: number): string {
  if (deltaPct > 50) return "#C0392B";
  if (deltaPct > 20) return "#CA6F1E";
  if (deltaPct > 5) return "#D4AC0D";
  if (deltaPct > -5) return "#A6A6A6";
  if (deltaPct > -20) return "#1ABC9C";
  if (deltaPct > -50) return "#148F77";
  return "#27AE60";
}

function deltaStyle(feature: GeoJSON.Feature | undefined) {
  const p = feature?.properties as LinkProperties | undefined;
  let deltaPct = p?.delta_pct ?? 0;
  const absDelta = p?.abs_delta_vol ?? 0;
  const baselineVol = p?.baseline_vol ?? 0;

  if (baselineVol < 500 && absDelta < 200) {
    deltaPct = Math.sign(deltaPct) * Math.min(Math.abs(deltaPct), 10);
  }

  return {
    color: getDeltaColor(deltaPct),
    weight: Math.max(2, Math.min(8, Math.log10(Math.max(absDelta, 1)) * 2)),
    opacity: Math.abs(deltaPct) < 2 ? 0.2 : 0.85,
  };
}

function fmt(n: number | null | undefined, decimals = 0): string {
  if (n == null || isNaN(n)) return "—";
  return decimals > 0 ? n.toFixed(decimals) : Math.round(n).toLocaleString();
}

function signedFmt(n: number | null | undefined, decimals = 0): string {
  if (n == null || isNaN(n)) return "—";
  const prefix = n > 0 ? "+" : "";
  return prefix + (decimals > 0 ? n.toFixed(decimals) : Math.round(n).toLocaleString());
}

function changeColor(val: number): string {
  if (val > 0) return "#C0392B";
  if (val < 0) return "#148F77";
  return "#6b7280";
}

function changeArrow(val: number): string {
  if (val > 0) return "&#9650;";  // ▲
  if (val < 0) return "&#9660;";  // ▼
  return "&#8212;";                // —
}

const HIT_STYLE = { weight: 20, opacity: 0, color: "#000" };

function getVocColorForDelta(voc: number): string {
  for (const { min, color } of VOC_SCALE) if (voc >= min) return color;
  return "#ccc";
}

function getLosForDelta(voc: number): string {
  for (const s of VOC_SCALE) if (voc >= s.min) return s.los;
  return "A";
}

function onEachFeature(feature: GeoJSON.Feature, layer: L.Layer) {
  const p = feature.properties as unknown as LinkProperties;

  const deltaVol = p.delta_vol ?? 0;
  const deltaPct = p.delta_pct ?? 0;
  const deltaVoc = p.delta_voc ?? 0;
  const volColor = changeColor(deltaVol);
  const vocColorChange = changeColor(deltaVoc);

  const baseVoc = p.baseline_voc ?? 0;
  const scenVoc = p.VOC_max ?? 0;
  const baseLosColor = getVocColorForDelta(baseVoc);
  const scenLosColor = getVocColorForDelta(scenVoc);
  const baseLos = p.LOS_max ? getLosForDelta(baseVoc) : getLosForDelta(baseVoc);
  const scenLos = getLosForDelta(scenVoc);

  layer.bindPopup(
    `<div style="min-width:280px;font-size:13px;line-height:1.6">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px;border-bottom:1px solid #e5e7eb;padding-bottom:4px">
        ${linkDisplayName(p)}
        <span style="float:right;font-weight:400;color:#6b7280">${p.link_type}</span>
      </div>

      <table style="width:100%;border-collapse:collapse">
        <tr style="color:#6b7280;font-size:11px">
          <td></td><td style="font-weight:600">Baseline</td><td style="font-weight:600">Scenar</td><td style="font-weight:600">Zmena</td>
        </tr>

        <tr>
          <td style="color:#6b7280">Objem</td>
          <td>${fmt(p.baseline_vol)}</td>
          <td>${fmt(p.wd_daily_tot)}</td>
          <td style="color:${volColor};font-weight:700">
            <span style="font-size:10px">${changeArrow(deltaVol)}</span>
            ${signedFmt(deltaVol)} (${signedFmt(deltaPct, 1)} %)
          </td>
        </tr>

        <tr>
          <td style="color:#6b7280">V/C</td>
          <td>${fmt(baseVoc, 2)}</td>
          <td>${fmt(scenVoc, 2)}</td>
          <td style="color:${vocColorChange};font-weight:700">
            <span style="font-size:10px">${changeArrow(deltaVoc)}</span>
            ${signedFmt(deltaVoc, 3)}
          </td>
        </tr>

        <tr>
          <td style="color:#6b7280">LOS</td>
          <td><span style="padding:1px 5px;border-radius:3px;font-size:11px;font-weight:700;color:#fff;background:${baseLosColor}">${baseLos}</span></td>
          <td><span style="padding:1px 5px;border-radius:3px;font-size:11px;font-weight:700;color:#fff;background:${scenLosColor}">${scenLos}</span></td>
          <td></td>
        </tr>
      </table>

      <div style="margin-top:8px;padding-top:4px;border-top:1px solid #e5e7eb">
        <table style="width:100%;border-collapse:collapse;font-size:12px;color:#6b7280">
          <tr>
            <td>Rychlost</td><td>${fmt(p.speed)} km/h</td>
            <td>Kapacita</td><td>${fmt(p.capacity)} voz/h/smer</td>
          </tr>
          <tr>
            <td>Pruhy</td><td>${fmt(p.lanes)}</td>
            <td>Delka</td><td>${fmt(p.distance)} m</td>
          </tr>
        </table>
      </div>

      <div style="margin-top:4px;font-size:10px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:3px">
        link_id: ${p.link_id}
      </div>
    </div>`,
    { maxWidth: 360 },
  );
}

export function DeltaLinksLayer({
  data,
  dataKey,
}: {
  data: GeoJSONFeatureCollection;
  dataKey?: string;
}) {
  const filtered = useMemo<GeoJSONFeatureCollection>(() => {
    const hasDeltaFields = data.features.some(
      (f) => (f.properties as Record<string, unknown>).delta_vol != null,
    );
    if (!hasDeltaFields) {
      return data;
    }
    return {
      type: "FeatureCollection",
      features: data.features.filter((f) => {
        const p = f.properties as Record<string, unknown>;
        const absDelta = (p.abs_delta_vol as number | undefined) ?? 0;
        const baselineVol = (p.baseline_vol as number | undefined) ?? 0;
        const scenarioVol = (p.wd_daily_tot as number | undefined) ?? 0;
        const maxVol = Math.max(baselineVol, scenarioVol);
        return absDelta >= 100 && maxVol >= 200;
      }),
    };
  }, [data]);

  const k = `delta-${dataKey ?? "d"}-${filtered.features.length}`;
  return (
    <>
      <GeoJSON
        key={`hit-${k}`}
        data={filtered as never}
        style={HIT_STYLE}
        onEachFeature={onEachFeature}
      />
      <GeoJSON
        key={`vis-${k}`}
        data={filtered as never}
        style={deltaStyle}
        interactive={false}
      />
    </>
  );
}
