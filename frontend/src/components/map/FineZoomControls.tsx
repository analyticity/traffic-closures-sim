import { createPortal } from "react-dom";
import { Minus, Plus } from "lucide-react";
import { useMap } from "react-leaflet";

const STEP = 0.25;

/**
 * Tlačítka +/− s krokem podle zoomDelta; renderuje se portálem vedle mapy,
 * aby šlo tlačítka umístit mimo leaflet-container (kvůli překryvu panelů).
 */
export function FineZoomControls() {
  const map = useMap();
  const host = map.getContainer().parentElement;
  if (!host) return null;

  return createPortal(
    <div
      className="pointer-events-auto z-[1000] flex flex-col overflow-hidden rounded-lg border border-gray-300 bg-white shadow-md"
      style={{ position: "absolute", left: "15.5rem", bottom: "1.5rem" }}
    >
      <button
        type="button"
        className="flex h-9 w-9 items-center justify-center text-gray-700 hover:bg-gray-100 active:bg-gray-200"
        aria-label="Přiblížit"
        onClick={() => map.zoomIn(STEP)}
      >
        <Plus size={18} strokeWidth={2.25} />
      </button>
      <div className="h-px bg-gray-200" />
      <button
        type="button"
        className="flex h-9 w-9 items-center justify-center text-gray-700 hover:bg-gray-100 active:bg-gray-200"
        aria-label="Oddálit"
        onClick={() => map.zoomOut(STEP)}
      >
        <Minus size={18} strokeWidth={2.25} />
      </button>
    </div>,
    host,
  );
}
