import { useState } from "react";
import { BiasMap } from "../components/diagnostics/BiasMap";
import { ThroughTrafficMap } from "../components/diagnostics/ThroughTrafficMap";
import { RouteComparisonMap } from "../components/diagnostics/RouteComparisonMap";
import { CorridorScorePanel } from "../components/diagnostics/CorridorScorePanel";
import { CorridorDiagnosisMap } from "../components/diagnostics/CorridorDiagnosisMap";

const TABS = [
  { id: "bias", label: "Odchylky" },
  { id: "corridors", label: "Koridory" },
  { id: "diagnosis", label: "Diagnóza" },
  { id: "through", label: "Tranzit" },
  { id: "routes", label: "Trasy" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export function DiagnosticsPage() {
  const [tab, setTab] = useState<TabId>("bias");

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 border-b bg-white px-4 py-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
              tab === t.id
                ? "bg-indigo-100 text-indigo-700"
                : "text-gray-600 hover:bg-gray-100"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-hidden">
        {tab === "bias" && <BiasMap />}
        {tab === "corridors" && <CorridorScorePanel />}
        {tab === "diagnosis" && <CorridorDiagnosisMap />}
        {tab === "through" && <ThroughTrafficMap />}
        {tab === "routes" && <RouteComparisonMap />}
      </div>
    </div>
  );
}
