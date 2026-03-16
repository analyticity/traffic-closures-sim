import type { OdSummary } from "../../types";

export function OdSummaryCard({ data }: { data: OdSummary }) {
  return (
    <div className="rounded-lg border bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-lg font-bold text-gray-800">OD Matice</h2>

      <div className="mb-4 grid grid-cols-3 gap-3 text-sm">
        <Stat label="Zóny" value={data.zones} />
        <Stat label="OD páry" value={data.pairs_used.toLocaleString()} />
        <Stat label="Missing origin" value={data.missing_origin.toLocaleString()} />
      </div>

      <h3 className="mb-2 text-sm font-semibold text-gray-600">Součty per period (veh/day)</h3>
      <div className="grid grid-cols-5 gap-2 text-xs">
        {Object.entries(data.cores_sum).map(([k, v]) => (
          <div key={k} className="rounded-md bg-gray-50 p-2 text-center">
            <div className="text-gray-500">{k}</div>
            <div className="font-semibold">{Math.round(v).toLocaleString()}</div>
          </div>
        ))}
      </div>
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
