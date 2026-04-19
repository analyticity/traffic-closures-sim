import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { useLinks, useZones, useCentroids, useModelArea, useDayInfo } from "../api/hooks";
import { useMapStore } from "../stores/mapStore";
import { useScenarioStore } from "../stores/scenarioStore";
import { LinksLayer, ScenarioHighlightLayer } from "../components/map/LinksLayer";
import { DeltaLinksLayer } from "../components/map/DeltaLinksLayer";
import { ZonesLayer } from "../components/map/ZonesLayer";
import { CentroidsLayer } from "../components/map/CentroidsLayer";
import { LayerControls } from "../components/map/LayerControls";
import { Legend } from "../components/map/Legend";
import { TimeSelector } from "../components/map/TimeSelector";
import { ScenarioPanel } from "../components/map/ScenarioPanel";
import { ViewToggle } from "../components/map/ViewToggle";
import type { GeoJSONFeatureCollection } from "../types";

const BRNO_CENTER: [number, number] = [49.195, 16.608];

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
  const { showLinks, showZones, showCentroids, showModelArea, linkTypes, selectedDate, selectedPeriod, setDayInfo } = useMapStore();
  const { viewMode, scenarioResults } = useScenarioStore();

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

  return (
    <div className="relative h-full w-full">
      <MapContainer center={BRNO_CENTER} zoom={12} className="h-full w-full">
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
      </MapContainer>

      <LayerControls />
      {showLinks && <Legend />}
      <ViewToggle />
      <ScenarioPanel />

      <div className="absolute bottom-6 left-4 z-[1000] w-56 rounded-lg bg-white p-3 shadow-lg">
        <TimeSelector />
      </div>

      {(links.isLoading || links.isFetching) && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1100] rounded-lg bg-white px-4 py-2 shadow-lg text-xs text-gray-500">
          Načítání...
        </div>
      )}
    </div>
  );
}
