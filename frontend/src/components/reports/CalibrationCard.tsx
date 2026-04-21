import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type { CalibrationReport } from "../../types";

export function CalibrationCard({ data }: { data: CalibrationReport }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-lg font-bold text-gray-800">Kalibrace</h2>

      <div className="mb-4 grid grid-cols-3 gap-4 text-sm">
        <Stat label="Iterace" value={data.iterations} />
        <Stat label="Konvergence" value={data.converged ? "Ano" : "Ne"} />
        <Stat label="Metoda" value={data.config.scale_method} />
      </div>

      <div className="mb-6 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.history}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="iteration" label={{ value: "Iterace", position: "bottom" }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="geh_lt5_pct" name="GEH<5 %" stroke="#148F77" strokeWidth={2} />
            <Line type="monotone" dataKey="geh_lt10_pct" name="GEH<10 %" stroke="#8E44AD" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-200 text-left text-gray-500">
            <th className="py-1">Iter</th>
            <th>Demand</th>
            <th>Assigned</th>
            <th>GEH&lt;5%</th>
            <th>R²</th>
            <th>RMSE</th>
          </tr>
        </thead>
        <tbody>
          {data.history.map((h) => (
            <tr key={h.iteration} className="border-b border-gray-100">
              <td className="py-1">{h.iteration}</td>
              <td>{Math.round(h.demand_total).toLocaleString()}</td>
              <td>{Math.round(h.assigned_total).toLocaleString()}</td>
              <td className="font-medium">{h.geh_lt5_pct.toFixed(1)}%</td>
              <td>{h.r2?.toFixed(2) ?? "—"}</td>
              <td>{h.rmse.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number | boolean }) {
  return (
    <div className="rounded-md bg-gray-100 p-3 text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-semibold text-gray-800">{String(value)}</div>
    </div>
  );
}
