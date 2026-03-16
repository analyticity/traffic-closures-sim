const ITEMS = [
  { color: "#d73027", label: "> 5 000" },
  { color: "#fc8d59", label: "2 000 – 5 000" },
  { color: "#fee08b", label: "500 – 2 000" },
  { color: "#91cf60", label: "100 – 500" },
  { color: "#1a9850", label: "< 100" },
];

export function Legend() {
  return (
    <div className="absolute bottom-6 right-6 z-[1000] rounded-lg bg-white p-3 shadow-lg text-xs">
      <div className="mb-1 font-semibold text-gray-700">Volume (veh/day)</div>
      {ITEMS.map(({ color, label }) => (
        <div key={color} className="flex items-center gap-2 py-0.5">
          <span className="inline-block h-3 w-6 rounded" style={{ background: color }} />
          <span className="text-gray-600">{label}</span>
        </div>
      ))}
    </div>
  );
}
