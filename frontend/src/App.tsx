import { Map, BarChart3, Info } from "lucide-react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import { MapPage } from "./pages/MapPage";
import { ReportsPage } from "./pages/ReportsPage";
import { AboutPage } from "./pages/AboutPage";

const NAV = [
  { to: "/", label: "Mapa", icon: Map },
  { to: "/reports", label: "Reporty", icon: BarChart3 },
  { to: "/about", label: "O projektu", icon: Info },
];

export default function App() {
  const { pathname } = useLocation();

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-6 border-b bg-white px-6 py-3 shadow-sm">
        <h1 className="text-lg font-bold text-gray-800">Brno Traffic Simulation</h1>
        <nav className="flex gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                pathname === to
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <Icon size={16} />
              {label}
            </Link>
          ))}
        </nav>
      </header>

      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<MapPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  );
}
