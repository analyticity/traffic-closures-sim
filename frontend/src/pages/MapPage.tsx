import { useEffect } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { useLinks, useZones, useCentroids, useModelArea, useDayInfo } from "../api/hooks";
import { useMapStore } from "../stores/mapStore";
import { LinksLayer } from "../components/map/LinksLayer";
import { ZonesLayer } from "../components/map/ZonesLayer";
import { CentroidsLayer } from "../components/map/CentroidsLayer";
import { LayerControls } from "../components/map/LayerControls";
import { Legend } from "../components/map/Legend";
import { TimeSelector } from "../components/map/TimeSelector";

const BRNO_CENTER: [number, number] = [49.195, 16.608];

export function MapPage() {
  const { showLinks, showZones, showCentroids, showModelArea, linkTypes, selectedDate, selectedPeriod, setDayInfo } = useMapStore();

  const links = useLinks(linkTypes, selectedDate, selectedPeriod);
  const zones = useZones();
  const centroids = useCentroids();
  const modelArea = useModelArea();
  const dayInfoQuery = useDayInfo(selectedDate);

  useEffect(() => {
    setDayInfo(dayInfoQuery.data ?? null);
  }, [dayInfoQuery.data, setDayInfo]);

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
        {showLinks && links.data && <LinksLayer data={links.data} />}
        {showCentroids && centroids.data && <CentroidsLayer data={centroids.data} />}
      </MapContainer>

      <LayerControls />
      {showLinks && <Legend />}

      {/* Time selector — bottom left */}
      <div className="absolute bottom-6 left-4 z-[1000] w-56 rounded-lg bg-white p-3 shadow-lg">
        <TimeSelector />
      </div>

      {(links.isLoading || links.isFetching) && (
        <div className="absolute top-16 right-4 z-[1100] rounded-lg bg-white px-4 py-2 shadow-lg text-xs text-gray-500">
          Načítání...
        </div>
      )}
    </div>
  );
}
