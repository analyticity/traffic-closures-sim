import { useState } from "react";
import { ChevronDown, ChevronUp, Trash2, Play, X, Loader2, AlertCircle } from "lucide-react";
import { useScenarioStore } from "../../stores/scenarioStore";
import { linkDisplayName } from "./LinksLayer";
import type { ScenarioLink } from "../../types";

function LinkRow({ link }: { link: ScenarioLink }) {
  const { removeLink, updateLink } = useScenarioStore();

  return (
    <div className="border-b border-gray-100 py-2.5 last:border-b-0">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-gray-800">
            {linkDisplayName(link)}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="inline-block rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-medium text-gray-500">
              {link.link_type}
            </span>
            <span className="text-[10px] text-gray-400">#{link.link_id}</span>
          </div>
        </div>
        <button
          onClick={() => removeLink(link.link_id, link.direction)}
          className="mt-0.5 rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors"
          title="Odebrat"
        >
          <Trash2 size={14} />
        </button>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2">
        <div>
          <label className="block text-[10px] font-medium text-gray-500 mb-0.5">Směr</label>
          <select
            value={link.direction}
            onChange={(e) =>
              updateLink(link.link_id, link.direction, {
                direction: e.target.value as ScenarioLink["direction"],
              })
            }
            className="w-full rounded border border-gray-200 bg-white px-1.5 py-1 text-xs text-gray-700"
          >
            <option value="both">Oba směry</option>
            <option value="ab" disabled={link.network_direction === -1}>
              A → B
            </option>
            <option value="ba" disabled={link.network_direction === 1}>
              B → A
            </option>
          </select>
        </div>

        <div>
          <label className="block text-[10px] font-medium text-gray-500 mb-0.5">Typ</label>
          <select
            value={link.closure_type}
            onChange={(e) =>
              updateLink(link.link_id, link.direction, {
                closure_type: e.target.value as ScenarioLink["closure_type"],
              })
            }
            className="w-full rounded border border-gray-200 bg-white px-1.5 py-1 text-xs text-gray-700"
          >
            <option value="full">Úplná uzavírka</option>
            <option value="lanes">Omezení pruhů</option>
          </select>
        </div>
      </div>

      {link.closure_type === "lanes" && (
        <div className="mt-2">
          <label className="block text-[10px] font-medium text-gray-500 mb-0.5">
            Zbývající pruhy (z {link.lanes})
          </label>
          <input
            type="number"
            min={1}
            max={Math.max(link.lanes - 1, 1)}
            value={link.lanes_remaining}
            onChange={(e) =>
              updateLink(link.link_id, link.direction, {
                lanes_remaining: Math.max(1, Math.min(link.lanes - 1, Number(e.target.value))),
              })
            }
            className="w-20 rounded border border-gray-200 bg-white px-1.5 py-1 text-xs text-gray-700"
          />
        </div>
      )}
    </div>
  );
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m > 0) return `${m} min ${s} s`;
  return `${s} s`;
}

export function ScenarioPanel() {
  const {
    links, isRunning, scenarioResults, error, jobStatus,
    clearAll, runScenario,
  } = useScenarioStore();
  const [collapsed, setCollapsed] = useState(false);

  const hasLinks = links.length > 0;
  const hasResults = scenarioResults !== null;

  return (
    <div className="absolute top-4 right-4 z-[1000] w-72 rounded-lg bg-white shadow-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center justify-between px-3 py-2.5 bg-gray-50 border-b border-gray-200 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-700">Scénář</span>
          {hasLinks && (
            <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-blue-500 px-1.5 text-[10px] font-bold text-white">
              {links.length}
            </span>
          )}
          {hasResults && (
            <span className="inline-flex items-center rounded-full bg-green-100 px-1.5 py-0.5 text-[10px] font-medium text-green-700">
              výsledky
            </span>
          )}
        </div>
        {collapsed ? <ChevronDown size={16} className="text-gray-400" /> : <ChevronUp size={16} className="text-gray-400" />}
      </button>

      {!collapsed && (
        <div className="flex flex-col">
          {/* Link list */}
          {hasLinks ? (
            <div className="max-h-[50vh] overflow-y-auto px-3 py-1">
              {links.map((link) => (
                <LinkRow key={`${link.link_id}-${link.direction}`} link={link} />
              ))}
            </div>
          ) : (
            <div className="px-3 py-6 text-center text-xs text-gray-400">
              Klikněte na silnici v mapě<br />a zvolte „Přidat do scénáře"
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="mx-3 mb-2 flex items-start gap-1.5 rounded-md bg-red-50 px-2.5 py-2 text-[11px] text-red-700">
              <AlertCircle size={14} className="mt-0.5 shrink-0" />
              <span className="line-clamp-3">{error}</span>
            </div>
          )}

          {/* Footer actions */}
          {hasLinks && (
            <div className="border-t border-gray-200 px-3 py-2.5 space-y-2">
              <button
                onClick={() => runScenario()}
                disabled={isRunning}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isRunning ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    <span>
                      Simulace běží…
                      {jobStatus?.elapsed_seconds != null && (
                        <span className="ml-1 font-normal opacity-80">
                          {formatElapsed(jobStatus.elapsed_seconds)}
                        </span>
                      )}
                    </span>
                  </>
                ) : (
                  <>
                    <Play size={14} />
                    Spustit simulaci
                  </>
                )}
              </button>
              <button
                onClick={clearAll}
                disabled={isRunning}
                className="flex w-full items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <X size={12} />
                Smazat vše
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
