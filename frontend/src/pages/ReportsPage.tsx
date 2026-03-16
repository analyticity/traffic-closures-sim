import { useCalibration, useValidation, useOdSummary } from "../api/hooks";
import { CalibrationCard } from "../components/reports/CalibrationCard";
import { ValidationCard } from "../components/reports/ValidationCard";
import { OdSummaryCard } from "../components/reports/OdSummaryCard";

export function ReportsPage() {
  const calibration = useCalibration();
  const validation = useValidation();
  const odSummary = useOdSummary();

  const loading = calibration.isLoading || validation.isLoading || odSummary.isLoading;

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        Načítání reportů...
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <h1 className="mb-6 text-2xl font-bold text-gray-800">Reporty simulace</h1>
      <div className="grid gap-6 lg:grid-cols-2">
        {calibration.data && <CalibrationCard data={calibration.data} />}
        {validation.data && <ValidationCard data={validation.data} />}
        {odSummary.data && <OdSummaryCard data={odSummary.data} />}
      </div>
    </div>
  );
}
