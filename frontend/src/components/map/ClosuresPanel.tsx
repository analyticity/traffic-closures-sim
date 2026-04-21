import { useEffect, useState } from "react";
import {
  Construction,
  Play,
  Plus,
  Loader2,
  AlertCircle,
  Calendar,
} from "lucide-react";
import { useClosures } from "../../api/hooks";
import { useScenarioStore } from "../../stores/scenarioStore";
import type { ClosureFeatureProperties, GeoJSONFeatureCollection } from "../../types";
import { SEVERITY_LABEL } from "./ClosuresLayer";

const SEVERITY_COLOR: Record<string, string> = {
  full: "#E74C3C",
  lane_reduction: "#E67E22",
  speed_limit: "#F1C40F",
};

interface Props {
  onClosuresLoaded?: (data: GeoJSONFeatureCollection | null) => void;
}

export function ClosuresPanel({ onClosuresLoaded }: Props) {
  const [date, setDate] = useState<string>("");
  const { addLinks, runClosureScenario, isRunning, jobStatus } = useScenarioStore();

  const { data: closuresData, isFetching, error } = useClosures(date || null);

  const features = closuresData?.features ?? [];
  const count = features.length;

  useEffect(() => {
    onClosuresLoaded?.(date && closuresData ? closuresData : null);
  }, [date, closuresData, onClosuresLoaded]);

  const handleAddToScenario = () => {
    if (!features.length) return;
    const payloads = features.map((f) => {
      const p = f.properties as unknown as ClosureFeatureProperties;
      return {
        link_id: p.link_id,
        name: p.name,
        link_type: p.link_type,
        lanes: p.lanes,
        direction: p.direction,
        closure_type: p.closure_type,
        lanes_remaining: p.lanes_remaining,
      };
    });
    addLinks(payloads);
  };

  const handleRunDirectly = () => {
    if (!date) return;
    runClosureScenario(date);
  };

  function formatElapsed(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m > 0) return `${m} min ${s} s`;
    return `${s} s`;
  }

  return (
    <div className="space-y-3">
      {/* Date picker */}
      <div>
        <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-gray-700">
          <Calendar size={14} /> Datum uzavírek
        </div>
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:border-orange-500 focus:outline-none"
        />
      </div>

      {/* Loading */}
      {isFetching && (
        <div className="py-2 text-center">
          <Loader2 size={14} className="inline animate-spin text-orange-500" />
          <span className="ml-1.5 text-xs text-gray-500">Načítání…</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-1.5 rounded-md bg-red-50 px-2 py-1.5 text-[11px] text-red-700">
          <AlertCircle size={12} className="mt-0.5 shrink-0" />
          <span className="line-clamp-2">{String(error)}</span>
        </div>
      )}

      {/* No results */}
      {!isFetching && date && count === 0 && !error && (
        <div className="py-2 text-center text-xs text-gray-400">
          Žádné uzavírky pro toto datum
        </div>
      )}

      {/* Results */}
      {!isFetching && count > 0 && (
        <>
          <div className="flex items-center gap-1.5 text-xs text-gray-600">
            <Construction size={12} className="text-orange-500" />
            <span className="font-medium">{count} uzavírek</span>
          </div>

          <div className="max-h-[30vh] overflow-y-auto -mx-1 px-1 space-y-0">
            {features.map((f, i) => {
              const p = f.properties as unknown as ClosureFeatureProperties;
              const sevColor = SEVERITY_COLOR[p.severity] ?? SEVERITY_COLOR.lane_reduction;
              return (
                <div
                  key={`${p.link_id}-${i}`}
                  className="border-b border-gray-100 py-1.5 last:border-b-0"
                >
                  <div className="flex items-start gap-1.5">
                    <span
                      className="mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: sevColor }}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[11px] font-medium text-gray-800">
                        {p.name || `Link #${p.link_id}`}
                      </div>
                      <div className="text-[10px] text-gray-400">
                        {SEVERITY_LABEL[p.severity] ?? p.severity}
                        {p.closure_text && ` · ${p.closure_text.slice(0, 40)}`}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Actions */}
          <div className="space-y-1.5 pt-1 border-t border-gray-200">
            <button
              onClick={handleAddToScenario}
              disabled={isRunning}
              className="flex w-full items-center justify-center gap-1.5 rounded-md bg-gray-100 px-2 py-1.5 text-[11px] font-semibold text-gray-700 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Plus size={12} />
              Přidat do scénáře
            </button>
            <button
              onClick={handleRunDirectly}
              disabled={isRunning}
              className="flex w-full items-center justify-center gap-1.5 rounded-md bg-orange-600 px-2 py-1.5 text-[11px] font-semibold text-white hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isRunning ? (
                <>
                  <Loader2 size={12} className="animate-spin" />
                  <span>
                    Simulace…
                    {jobStatus?.elapsed_seconds != null && (
                      <span className="ml-1 font-normal opacity-80">
                        {formatElapsed(jobStatus.elapsed_seconds)}
                      </span>
                    )}
                  </span>
                </>
              ) : (
                <>
                  <Play size={12} />
                  Spustit simulaci
                </>
              )}
            </button>
          </div>
        </>
      )}

      {/* Empty state */}
      {!date && !isFetching && (
        <div className="py-2 text-center text-[11px] text-gray-400">
          Vyberte datum pro zobrazení<br />aktivních uzavírek
        </div>
      )}
    </div>
  );
}
