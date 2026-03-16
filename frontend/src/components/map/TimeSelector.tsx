import { Calendar, Clock } from "lucide-react";
import { useMapStore } from "../../stores/mapStore";

const PERIODS = [
  { value: "daily", label: "Celý den" },
  { value: "day", label: "Den (6–18)" },
  { value: "evening", label: "Večer (18–22)" },
  { value: "night", label: "Noc (22–6)" },
];

export function TimeSelector() {
  const { selectedDate, selectedPeriod, dayInfo, setDate, setPeriod } = useMapStore();

  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-gray-700">
          <Calendar size={14} /> Datum
        </div>
        <input
          type="date"
          value={selectedDate}
          onChange={(e) => setDate(e.target.value)}
          className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
        />
        {dayInfo && (
          <div className="mt-1.5 rounded bg-blue-50 px-2 py-1 text-xs text-blue-700">
            <span className="font-medium">{dayInfo.weekday}</span>
            {" — "}
            {dayInfo.day_type === "workday" ? "pracovní den" :
             dayInfo.day_type === "saturday" ? "sobota" :
             dayInfo.day_type === "sunday" ? "neděle" : "svátek"}
            {" — "}
            faktor <span className="font-semibold">{dayInfo.day_factor.toFixed(2)}</span>
          </div>
        )}
      </div>

      <div>
        <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-gray-700">
          <Clock size={14} /> Období
        </div>
        <div className="flex flex-wrap gap-1">
          {PERIODS.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setPeriod(value)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                selectedPeriod === value
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
