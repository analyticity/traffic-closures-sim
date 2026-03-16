import type { ValidationReport } from "../../types";

export function ValidationCard({ data }: { data: ValidationReport }) {
  const p = data.pentlogram;

  return (
    <div className="rounded-lg border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-lg font-bold text-gray-800">Validace</h2>

      {p && (
        <div className="mb-6">
          <h3 className="mb-2 text-sm font-semibold text-gray-600">Pentlogram (referenční)</h3>
          <div className="grid grid-cols-4 gap-3 text-sm">
            <Stat label="Matched" value={p.matched} />
            <Stat label="GEH<5%" value={`${p.geh_lt5_pct?.toFixed(1)}%`} />
            <Stat label="R²" value={p.r2?.toFixed(2) ?? "—"} />
            <Stat label="RMSE" value={p.rmse?.toFixed(0) ?? "—"} />
          </div>
        </div>
      )}

      {data.csd2020_observed && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-600">CSD2020 – Jihomoravský kraj</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="py-1">Třída</th>
                <th>Úseků</th>
                <th>Ø RPDI</th>
                <th>Ø Osobní</th>
              </tr>
            </thead>
            <tbody>
              {data.csd2020_observed.map((r) => (
                <tr key={r.road_class} className="border-b">
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
    <div className="rounded-md bg-gray-50 p-3 text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-semibold text-gray-800">{value}</div>
    </div>
  );
}
