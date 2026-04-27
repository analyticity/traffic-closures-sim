import { useModelMeta } from "../api/hooks";

export function AboutPage() {
  const meta = useModelMeta();
  const city = meta.data?.title_short ?? "města";

  return (
    <div className="mx-auto max-w-3xl p-8">
      <h1 className="mb-6 text-2xl font-bold text-gray-800">O projektu</h1>

      <div className="space-y-4 text-sm text-gray-700 leading-relaxed">
        <p>
          Dopravní simulace {city} postavená na frameworku{" "}
          <b>AequilibraE</b> s daty z OpenStreetMap a Českého statistického úřadu (SLDB 2021).
        </p>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Pipeline</h2>
        <ol className="list-decimal pl-6 space-y-1">
          <li><b>build-network</b> — import OSM sítě</li>
          <li><b>normalize-network</b> — normalizace rychlostí, kapacit, jízdních dob</li>
          <li><b>build-zones</b> — TAZ zóny z OSM admin hranic + centroidní konektory</li>
          <li><b>fetch-data</b> — stažení SLDB dojížďky, pentlogramu, CSD2020</li>
          <li><b>build-demand</b> — OD matice z dojížďkových dat (práce + škola → vozidla)</li>
          <li><b>assign</b> — traffic assignment (BFW equilibrium)</li>
          <li><b>calibrate</b> — iterativní kalibrace: assign → porovnání → škálování matice</li>
          <li><b>validate</b> — nezávislá validace na referenčních datech</li>
        </ol>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Kalibrační data</h2>
        <ul className="list-disc pl-6 space-y-1">
          <li><b>Pentlogram / lokální sčítání</b> — intenzity dopravy z města (stovky voz/24h)</li>
          <li><b>CSD 2020</b> — celostátní sčítání dopravy (RPDI po úsecích)</li>
        </ul>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Technologie</h2>
        <ul className="list-disc pl-6 space-y-1">
          <li>Backend: Python, AequilibraE, FastAPI</li>
          <li>Frontend: React, TypeScript, Leaflet, Recharts, TailwindCSS</li>
          <li>Data: OSM, ČSÚ SLDB 2021, ŘSD CSD 2020</li>
        </ul>
      </div>
    </div>
  );
}
