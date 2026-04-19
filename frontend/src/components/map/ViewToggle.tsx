import { useScenarioStore } from "../../stores/scenarioStore";

export function ViewToggle() {
  const { viewMode, setViewMode, scenarioResults } = useScenarioStore();

  if (!scenarioResults) return null;

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1000] flex rounded-lg bg-white shadow-lg border border-gray-200 overflow-hidden">
      <button
        onClick={() => setViewMode("baseline")}
        className={`px-4 py-2 text-xs font-semibold transition-colors ${
          viewMode === "baseline"
            ? "bg-blue-600 text-white"
            : "bg-white text-gray-600 hover:bg-gray-50"
        }`}
      >
        Baseline
      </button>
      <button
        onClick={() => setViewMode("scenario")}
        className={`px-4 py-2 text-xs font-semibold transition-colors ${
          viewMode === "scenario"
            ? "bg-orange-500 text-white"
            : "bg-white text-gray-600 hover:bg-gray-50"
        }`}
      >
        Scénář
      </button>
      <button
        onClick={() => setViewMode("delta")}
        className={`px-4 py-2 text-xs font-semibold transition-colors ${
          viewMode === "delta"
            ? "bg-purple-600 text-white"
            : "bg-white text-gray-600 hover:bg-gray-50"
        }`}
      >
        Rozdíly
      </button>

      {viewMode === "scenario" && (
        <div className="flex items-center px-2 border-l border-gray-200">
          <span className="inline-block h-2 w-2 rounded-full bg-orange-500 animate-pulse" />
        </div>
      )}
      {viewMode === "delta" && (
        <div className="flex items-center px-2 border-l border-gray-200">
          <span className="inline-block h-2 w-2 rounded-full bg-purple-500 animate-pulse" />
        </div>
      )}
    </div>
  );
}
