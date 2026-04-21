import { VOC_SCALE } from "./LinksLayer";
import { DELTA_SCALE } from "./DeltaLinksLayer";
import { useScenarioStore } from "../../stores/scenarioStore";
import { useMapStore } from "../../stores/mapStore";

export function Legend() {
  const scenarioLinks = useScenarioStore((s) => s.links);
  const viewMode = useScenarioStore((s) => s.viewMode);
  const hasResults = useScenarioStore((s) => s.scenarioResults !== null);
  const showClosures = useMapStore((s) => s.showClosures);
  const hasClosed = scenarioLinks.some((l) => l.closure_type === "full");
  const hasReduced = scenarioLinks.some((l) => l.closure_type === "lanes");
  const hasScenario = hasClosed || hasReduced;
  const isSimulated = (viewMode === "scenario" || viewMode === "delta") && hasResults;
  const isDelta = viewMode === "delta" && hasResults;

  return (
    <div className="absolute bottom-6 right-6 z-[1000] rounded-lg bg-white p-3 shadow-lg text-xs">
      {isDelta ? (
        <>
          <div className="mb-1 font-semibold text-gray-700">Změna objemu</div>
          {DELTA_SCALE.map(({ color, label }) => (
            <div key={color} className="flex items-center gap-2 py-0.5">
              <span className="inline-block h-3 w-6 rounded" style={{ background: color }} />
              <span className="text-gray-600">{label}</span>
            </div>
          ))}
          <div className="mt-1 text-[10px] text-gray-400">
            Šířka = absolutní změna objemu
          </div>
        </>
      ) : (
        <>
          <div className="mb-1 font-semibold text-gray-700">V/C &mdash; Level of Service</div>
          {VOC_SCALE.map(({ color, label }) => (
            <div key={color} className="flex items-center gap-2 py-0.5">
              <span className="inline-block h-3 w-6 rounded" style={{ background: color }} />
              <span className="text-gray-600">{label}</span>
            </div>
          ))}
        </>
      )}

      {hasScenario && !isDelta && (
        <>
          <div className="mt-2 mb-1 border-t border-gray-200 pt-2 font-semibold text-gray-700">
            {isSimulated ? "Scénář" : "Plánované změny"}
          </div>
          {hasClosed && (
            <div className="flex items-center gap-2 py-0.5">
              <svg viewBox="0 0 72 72" width="13" height="13" className="shrink-0">
                <polygon points="36,4 68,64 4,64" fill={isSimulated ? "#dc2626" : "#94a3b8"} stroke={isSimulated ? "#7f1d1d" : "#475569"} strokeWidth="3.5" strokeLinejoin="round" />
                <g transform="translate(36,44) translate(-12,-12)" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="6" width="20" height="8" rx="1" />
                  <path d="M17 14v7" /><path d="M7 14v7" />
                  <path d="M17 3v3" /><path d="M7 3v3" />
                  <path d="M10 14 2.3 6.3" />
                  <path d="m14 6 7.7 7.7" />
                  <path d="m8 6 8 8" />
                </g>
              </svg>
              <span className="text-gray-600">Úplná uzavírka</span>
            </div>
          )}
          {hasReduced && (
            <div className="flex items-center gap-2 py-0.5">
              <svg viewBox="0 0 72 72" width="13" height="13" className="shrink-0">
                <polygon points="36,4 68,64 4,64" fill={isSimulated ? "#ea580c" : "#fbbf24"} stroke={isSimulated ? "#7c2d12" : "#92400e"} strokeWidth="3.5" strokeLinejoin="round" />
                <g transform="translate(36,44) translate(-12,-12)" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M16.05 10.966a5 2.5 0 0 1-8.1 0" />
                  <path d="m16.923 14.049 4.48 2.04a1 1 0 0 1 .001 1.831l-8.574 3.9a2 2 0 0 1-1.66 0l-8.574-3.91a1 1 0 0 1 0-1.83l4.484-2.04" />
                  <path d="M16.949 14.14a5 2.5 0 1 1-9.9 0L10.063 3.5a2 2 0 0 1 3.874 0z" />
                  <path d="M9.194 6.57a5 2.5 0 0 0 5.61 0" />
                </g>
              </svg>
              <span className="text-gray-600">Omezení pruhů</span>
            </div>
          )}
        </>
      )}

      {showClosures && !isDelta && (
        <>
          <div className="mt-2 mb-1 border-t border-gray-200 pt-2 font-semibold text-gray-700">
            Uzavírky
          </div>
          {([
            ["#E74C3C", "Úplná uzavírka"],
            ["#E67E22", "Omezení pruhů"],
            ["#F1C40F", "Omezení rychlosti"],
          ] as const).map(([color, label]) => (
            <div key={color} className="flex items-center gap-2 py-0.5">
              <span
                className="inline-block h-3 w-3 rounded-full border"
                style={{ backgroundColor: color, borderColor: color }}
              />
              <span className="text-gray-600">{label}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
