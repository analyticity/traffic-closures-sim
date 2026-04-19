import { VOC_SCALE } from "./LinksLayer";
import { DELTA_SCALE } from "./DeltaLinksLayer";
import { useScenarioStore } from "../../stores/scenarioStore";

export function Legend() {
  const scenarioLinks = useScenarioStore((s) => s.links);
  const viewMode = useScenarioStore((s) => s.viewMode);
  const hasResults = useScenarioStore((s) => s.scenarioResults !== null);
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
              {isSimulated ? (
                <span className="inline-flex items-center justify-center h-3 w-6 rounded" style={{ background: "#18181b" }}>
                  <svg viewBox="0 0 28 28" width="10" height="10">
                    <circle cx="14" cy="14" r="13" fill="#dc2626" stroke="#fff" strokeWidth="2" />
                    <rect x="5" y="11.5" width="18" height="5" rx="1.5" fill="#fff" />
                  </svg>
                </span>
              ) : (
                <span className="inline-flex items-center justify-center h-3 w-6 rounded"
                  style={{ background: `repeating-linear-gradient(90deg, #6b7280 0 3px, transparent 3px 7px)` }}
                >
                  <svg viewBox="0 0 24 24" width="10" height="10">
                    <polygon points="12,2 22,20 2,20" fill="#f59e0b" stroke="#fff" strokeWidth="1.5" strokeLinejoin="round" />
                    <rect x="11" y="8" width="2" height="6" rx="0.5" fill="#fff" />
                    <circle cx="12" cy="16.5" r="1.2" fill="#fff" />
                  </svg>
                </span>
              )}
              <span className="text-gray-600">Uzavírka</span>
            </div>
          )}
          {hasReduced && (
            <div className="flex items-center gap-2 py-0.5">
              <span
                className="inline-block h-3 w-6 rounded"
                style={{
                  background: isSimulated
                    ? `repeating-linear-gradient(90deg, #f97316 0 4px, transparent 4px 8px)`
                    : `repeating-linear-gradient(90deg, #d97706 0 2px, transparent 2px 6px)`,
                  opacity: isSimulated ? 1 : 0.6,
                }}
              />
              <span className="text-gray-600">Omezení pruhů</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
