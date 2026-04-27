import { useModelMeta } from "../api/hooks";

export function AboutPage() {
  const meta = useModelMeta();
  const city = meta.data?.title_short ?? "města";

  return (
    <div className="mx-auto max-w-3xl overflow-y-auto p-8">
      <h1 className="mb-6 text-2xl font-bold text-gray-800">O projektu</h1>

      <div className="space-y-4 text-sm text-gray-700 leading-relaxed">
        <p>
          Dopravní simulace <b>{city}</b> postavená na frameworku{" "}
          <b>AequilibraE</b> s daty z OpenStreetMap, Českého statistického
          úřadu (SLDB 2021) a Celostátního sčítání dopravy (CSD 2025).
          Pipeline podporuje libovolné české město — stačí vytvořit
          konfiguraci přes <code className="rounded bg-gray-100 px-1 py-0.5 text-xs">python run.py init-city</code>.
        </p>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Pipeline</h2>
        <ol className="list-decimal pl-6 space-y-1">
          <li><b>init-city</b> — generátor konfigurace pro nové město</li>
          <li><b>build-network</b> — import OSM silniční sítě</li>
          <li><b>fetch-data</b> — stažení SLDB, CSD 2025, populace, uzavírek</li>
          <li><b>normalize-network</b> — normalizace rychlostí, kapacit, BPR; aplikace bazálních uzavírek</li>
          <li><b>build-zones</b> — TAZ zóny z admin hranic + brány na okraji modelu + centroidní konektory</li>
          <li><b>build-supernetwork</b> — hrubá národní síť pro tranzitní dopravu</li>
          <li><b>build-demand</b> — OD matice (dojížďka, vnější, tranzitní, syntetické segmenty)</li>
          <li><b>assign-warm-skims</b> — krátký assignment pro síťové impedance</li>
          <li><b>distribute</b> — gravitační model + IPF balancování</li>
          <li><b>assign</b> — traffic assignment (BFW equilibrium)</li>
          <li><b>calibrate</b> / <b>calibrate-odme</b> — iterativní kalibrace poptávky vs. sčítací data</li>
          <li><b>tune-supply</b> — optimalizace parametrů nabídky</li>
          <li><b>validate</b> — nezávislá validace na CSD datech</li>
          <li><b>learn-profile</b> — denní profily z CSD (typový den, víkendy, svátky)</li>
          <li><b>strip-closures</b> — obnovení čistého baseline po validaci</li>
          <li><b>serve</b> — REST API + webová vizualizace</li>
        </ol>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Kalibrační a validační data</h2>
        <ul className="list-disc pl-6 space-y-1">
          <li><b>CSD 2025</b> — celostátní sčítání dopravy (RPDI po úsecích, hlavní kalibrační zdroj)</li>
          <li><b>Pentlogram / lokální sčítání</b> — volitelný zdroj intenzit z města</li>
          <li><b>SLDB 2021</b> — dojížďkové proudy (základ OD matice)</li>
          <li><b>Uzavírky</b> — agregovaná data z externího projektu (více zdrojů)</li>
        </ul>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Funkce aplikace</h2>
        <ul className="list-disc pl-6 space-y-1">
          <li><b>Interaktivní mapa</b> — zobrazení provozu, VOC, zón, uzavírek; temporální profily</li>
          <li><b>Scénáře</b> — co-když analýza uzavírek a omezení (live přepočet)</li>
          <li><b>Diagnostika</b> — odchylky modelu, koridorová analýza, tranzitní doprava, porovnání tras</li>
          <li><b>Reporty</b> — kalibrační a validační metriky (R², GEH, RMSE)</li>
        </ul>

        <h2 className="text-lg font-semibold text-gray-800 pt-4">Technologie</h2>
        <ul className="list-disc pl-6 space-y-1">
          <li>Backend: Python, AequilibraE, FastAPI</li>
          <li>Frontend: React, TypeScript, Leaflet, Recharts, TailwindCSS</li>
          <li>Data: OSM, ČSÚ SLDB 2021, ŘSD CSD 2025, RÚIAN</li>
        </ul>
      </div>
    </div>
  );
}
