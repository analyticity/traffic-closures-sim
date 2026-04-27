import type { ValidationCountFit, ValidationReport } from "../../types";

function CountFitSection({ label, data }: { label: string; data: ValidationCountFit }) {
  return (
    <div className="mb-6">
      <h3 className="mb-2 text-sm font-semibold text-gray-600">{label}</h3>
      <div className="grid grid-cols-4 gap-3 text-sm">
        <Stat label="Matched" value={data.matched} />
        <Stat label="GEH<5%" value={`${data.geh_lt5_pct?.toFixed(1)}%`} />
        <Stat label="R²" value={data.r2?.toFixed(2) ?? "—"} />
        <Stat label="RMSE" value={data.rmse?.toFixed(0) ?? "—"} />
      </div>
    </div>
  );
}

export function ValidationCard({ data }: { data: ValidationReport }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-lg font-bold text-gray-800">Validace</h2>

      {data.pentlogram && (
        <CountFitSection label="Pentlogram (referenční)" data={data.pentlogram} />
      )}

      {data.calibration_reference && (
        <CountFitSection label="Kalibrační referenční data" data={data.calibration_reference} />
      )}

      {data.csd_observed && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-600">CSD Validation</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200 text-left text-gray-500">
                <th className="py-1">Class</th>
                <th>Sections</th>
                <th>Mean AADT</th>
                <th>Mean Cars</th>
              </tr>
            </thead>
            <tbody>
              {data.csd_observed.map((r) => (
                <tr key={r.road_class} className="border-b border-gray-100">
                  <td className="py-1 font-medium">{r.road_class}</td>
                  <td>{r.sections}</td>
                  <td>{Math.round(r.mean_sv).toLocaleString()}</td>
                  <td>{Math.round(r.mean_o).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md bg-gray-100 p-3 text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-semibold text-gray-800">{value}</div>
    </div>
  );
}
