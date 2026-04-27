import { useCallback, useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { useLinks, useZones, useCentroids, useModelArea, useDayInfo } from "../api/hooks";
import { useMapStore } from "../stores/mapStore";
import { useScenarioStore } from "../stores/scenarioStore";
import { LinksLayer, ScenarioHighlightLayer } from "../components/map/LinksLayer";
import { DeltaLinksLayer } from "../components/map/DeltaLinksLayer";
import { ZonesLayer } from "../components/map/ZonesLayer";
import { CentroidsLayer } from "../components/map/CentroidsLayer";
import { ClosuresLayer } from "../components/map/ClosuresLayer";
import { LayerControls } from "../components/map/LayerControls";
import { Legend } from "../components/map/Legend";
import { TimeSelector } from "../components/map/TimeSelector";
import { ScenarioPanel } from "../components/map/ScenarioPanel";
import { ClosuresPanel } from "../components/map/ClosuresPanel";
import { ViewToggle } from "../components/map/ViewToggle";
import { FineZoomControls } from "../components/map/FineZoomControls";
import { FINE_INTERACTION_MAP_OPTIONS } from "../components/map/fineMapOptions";
import { useLeafletMapLayout } from "../map/useLeafletMapCenter";
import type { GeoJSONFeatureCollection } from "../types";
import { Calendar, Construction } from "lucide-react";

type BottomTab = "date" | "closures";

function filterByLinkTypes(
  data: GeoJSONFeatureCollection,
  linkTypes: string[],
): GeoJSONFeatureCollection {
  const typeSet = new Set(linkTypes);
  return {
    type: "FeatureCollection",
    features: data.features.filter(
      (f) => typeSet.has(f.properties.link_type as string),
    ),
  };
}

export function MapPage() {
  const { showLinks, showZones, showCentroids, showModelArea, showClosures, linkTypes, selectedDate, selectedPeriod, setDayInfo } = useMapStore();
  const { viewMode, scenarioResults } = useScenarioStore();
  const [closuresData, setClosuresData] = useState<GeoJSONFeatureCollection | null>(null);
  const [bottomTab, setBottomTab] = useState<BottomTab>("date");
  const handleClosuresLoaded = useCallback((data: GeoJSONFeatureCollection | null) => {
    setClosuresData(data);
  }, []);

  const links = useLinks(linkTypes, selectedDate, selectedPeriod);
  const zones = useZones();
  const centroids = useCentroids();
  const modelArea = useModelArea();
  const dayInfoQuery = useDayInfo(selectedDate);

  useEffect(() => {
    setDayInfo(dayInfoQuery.data ?? null);
  }, [dayInfoQuery.data, setDayInfo]);

  const filteredScenario = useMemo(
    () => scenarioResults ? filterByLinkTypes(scenarioResults, linkTypes) : null,
    [scenarioResults, linkTypes],
  );

  const isDelta = viewMode === "delta" && filteredScenario != null;
  const displayData =
    (viewMode === "scenario" || viewMode === "delta") && filteredScenario
      ? filteredScenario
      : links.data;

  const closureCount = closuresData?.features.length ?? 0;
  const { center: mapCenter, containerKey: mapLayoutKey } = useLeafletMapLayout();

  return (
    <div className="relative h-full w-full">
      <MapContainer
        key={mapLayoutKey}
        center={mapCenter}
        zoom={12}
        className="h-full w-full"
        zoomControl={false}
        {...FINE_INTERACTION_MAP_OPTIONS}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {showModelArea && modelArea.data && (
          <GeoJSON
            data={modelArea.data as never}
            style={{ color: "#6366f1", weight: 3, fillOpacity: 0, dashArray: "8 4" }}
          />
        )}

        {showZones && zones.data && <ZonesLayer data={zones.data} />}
        {showLinks && displayData && !isDelta && (
          <LinksLayer data={displayData} dataKey={viewMode === "scenario" && filteredScenario ? "scenario" : "baseline"} />
        )}
        {showLinks && isDelta && filteredScenario && (
          <DeltaLinksLayer data={filteredScenario} dataKey="delta" />
        )}
        {showLinks && links.data && !isDelta && (
          <ScenarioHighlightLayer data={links.data} />
        )}
        {showCentroids && centroids.data && <CentroidsLayer data={centroids.data} />}
        {showClosures && closuresData && closuresData.features.length > 0 && (
          <ClosuresLayer data={closuresData} />
        )}
        <FineZoomControls />
      </MapContainer>

      <LayerControls />
      {showLinks && <Legend />}
      <ViewToggle />
      <ScenarioPanel />

      {/* Bottom-left panel: toggle between Datum and Uzavírky */}
      <div className="absolute bottom-6 left-4 z-[1000] w-56 rounded-lg bg-white shadow-lg overflow-hidden">
        {/* Tab bar */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setBottomTab("date")}
            className={`flex flex-1 items-center justify-center gap-1.5 px-2 py-2 text-xs font-medium transition-colors ${
              bottomTab === "date"
                ? "bg-white text-blue-600 border-b-2 border-blue-600"
                : "bg-gray-50 text-gray-500 hover:text-gray-700"
            }`}
          >
            <Calendar size={12} />
            Datum
          </button>
          <button
            onClick={() => setBottomTab("closures")}
            className={`flex flex-1 items-center justify-center gap-1.5 px-2 py-2 text-xs font-medium transition-colors ${
              bottomTab === "closures"
                ? "bg-white text-orange-600 border-b-2 border-orange-600"
                : "bg-gray-50 text-gray-500 hover:text-gray-700"
            }`}
          >
            <Construction size={12} />
            Uzavírky
            {closureCount > 0 && (
              <span className="inline-flex h-4 min-w-4 items-center justify-center rounded-full bg-orange-500 px-1 text-[9px] font-bold text-white">
                {closureCount}
              </span>
            )}
          </button>
        </div>

        {/* Content */}
        <div className="p-3">
          {bottomTab === "date" && <TimeSelector />}
          {bottomTab === "closures" && (
            <ClosuresPanel onClosuresLoaded={handleClosuresLoaded} />
          )}
        </div>
      </div>

      {(links.isLoading || links.isFetching) && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-white px-4 py-2 shadow-lg text-xs text-gray-500">
          Načítání...
        </div>
      )}
    </div>
  );
}
