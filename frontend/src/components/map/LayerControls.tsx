import { Layers } from "lucide-react";
import { ALL_LINK_TYPES, useMapStore } from "../../stores/mapStore";

export function LayerControls() {
  const {
    showLinks, showZones, showCentroids, showModelArea, showClosures,
    linkTypes,
    toggleLayer, toggleLinkType,
  } = useMapStore();

  return (
    <div className="absolute left-4 top-4 z-[1000] w-56 space-y-4 rounded-lg bg-white p-4 shadow-lg text-sm">
      <div>
        <div className="mb-2 flex items-center gap-1.5 font-semibold text-gray-700">
          <Layers size={14} /> Vrstvy
        </div>
        {([
          ["links", "Linky (provoz)", showLinks],
          ["zones", "Zóny", showZones],
          ["centroids", "Centroidy", showCentroids],
          ["modelArea", "Oblast modelu", showModelArea],
          ["closures", "Uzavírky", showClosures],
        ] as const).map(([key, label, on]) => (
          <label key={key} className="flex items-center gap-2 py-0.5">
            <input
              type="checkbox"
              checked={on}
              onChange={() => toggleLayer(key)}
              className="rounded"
            />
            <span className="text-gray-600">{label}</span>
          </label>
        ))}
      </div>

      <div>
        <div className="mb-1 font-semibold text-gray-700">Typ silnice</div>
        {ALL_LINK_TYPES.map((t) => (
          <label key={t} className="flex items-center gap-2 py-0.5">
            <input
              type="checkbox"
              checked={linkTypes.includes(t)}
              onChange={() => toggleLinkType(t)}
              className="rounded"
            />
            <span className="text-gray-600">{t}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
