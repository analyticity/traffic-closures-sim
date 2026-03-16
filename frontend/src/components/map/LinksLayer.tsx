import { GeoJSON } from "react-leaflet";
import type { GeoJSONFeatureCollection, LinkProperties } from "../../types";

const COLORS = [
  { min: 5000, color: "#d73027" },
  { min: 2000, color: "#fc8d59" },
  { min: 500, color: "#fee08b" },
  { min: 100, color: "#91cf60" },
  { min: 0, color: "#1a9850" },
];

function getColor(vol: number) {
  for (const { min, color } of COLORS) if (vol >= min) return color;
  return "#ccc";
}

function style(feature: GeoJSON.Feature | undefined) {
  const vol = (feature?.properties as LinkProperties)?.wd_daily_tot ?? 0;
  return {
    color: getColor(vol),
    weight: Math.max(2, Math.min(7, Math.log10(Math.max(vol, 1)) * 1.5)),
    opacity: 0.85,
  };
}

const HIT_STYLE = { weight: 20, opacity: 0, color: "#000" };

function fmt(n: number | null | undefined, decimals = 0): string {
  if (n == null || isNaN(n)) return "—";
  return decimals > 0 ? n.toFixed(decimals) : Math.round(n).toLocaleString();
}

function onEachFeature(feature: GeoJSON.Feature, layer: L.Layer) {
  const p = feature.properties as unknown as LinkProperties;

  const vocColor =
    (p.VOC_max ?? 0) > 0.9 ? "#dc2626" : (p.VOC_max ?? 0) > 0.7 ? "#f59e0b" : "#16a34a";

  layer.bindPopup(
    `<div style="min-width:240px;font-size:13px;line-height:1.6">
      <div style="font-weight:700;font-size:14px;margin-bottom:4px;border-bottom:1px solid #e5e7eb;padding-bottom:4px">
        ${p.name || "Bez názvu"}
        <span style="float:right;font-weight:400;color:#6b7280">${p.link_type}</span>
      </div>

      <table style="width:100%;border-collapse:collapse">
        <tr><td colspan="3" style="font-weight:600;padding-top:4px;color:#374151">Provoz (voz/den)</td></tr>
        <tr style="color:#6b7280;font-size:11px">
          <td>A→B</td><td>B→A</td><td style="font-weight:600;color:#111">Celkem</td>
        </tr>
        <tr>
          <td>${fmt(p.wd_daily_ab)}</td>
          <td>${fmt(p.wd_daily_ba)}</td>
          <td style="font-weight:700;font-size:14px">${fmt(p.wd_daily_tot)}</td>
        </tr>

        <tr><td colspan="3" style="font-weight:600;padding-top:8px;color:#374151">Parametry</td></tr>
        <tr>
          <td style="color:#6b7280">Rychlost</td>
          <td colspan="2">${fmt(p.speed)} km/h</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Kapacita</td>
          <td colspan="2">${fmt(p.capacity)} voz/h</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Pruhy</td>
          <td colspan="2">${fmt(p.lanes)}</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Délka</td>
          <td colspan="2">${fmt(p.distance)} m</td>
        </tr>

        <tr><td colspan="3" style="font-weight:600;padding-top:8px;color:#374151">Zatížení</td></tr>
        <tr>
          <td style="color:#6b7280">V/C ratio</td>
          <td colspan="2"><span style="color:${vocColor};font-weight:600">${fmt(p.VOC_max, 2)}</span></td>
        </tr>
        <tr>
          <td style="color:#6b7280">Delay faktor</td>
          <td colspan="2">${fmt(p.Delay_factor_Max, 2)}×</td>
        </tr>
        <tr>
          <td style="color:#6b7280">Cest. čas (s)</td>
          <td colspan="2">${fmt(p.Congested_Time_Max, 1)}</td>
        </tr>
      </table>

      <div style="margin-top:6px;font-size:10px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:3px">
        link_id: ${p.link_id}
      </div>
    </div>`,
    { maxWidth: 300 },
  );
}

export function LinksLayer({ data }: { data: GeoJSONFeatureCollection }) {
  const k = data.features.length;
  return (
    <>
      {/* Invisible fat layer for easy clicking */}
      <GeoJSON
        key={`hit-${k}`}
        data={data as never}
        style={HIT_STYLE}
        onEachFeature={onEachFeature}
      />
      {/* Visible thin layer */}
      <GeoJSON
        key={`vis-${k}`}
        data={data as never}
        style={style}
        interactive={false}
      />
    </>
  );
}
