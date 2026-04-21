# A1 Poster — Canva Assembly Guide

**Size**: A1 landscape (841 × 594 mm)  
**Orientation**: Landscape  
**Reading flow**: Left → Center → Right, top to bottom

---

## Color Palette

| Role | Hex | Usage |
|------|-----|-------|
| Header background | `#1e3a5f` | Title strip, table header |
| Network phase | `#1e40af` | Blue — network-related sections |
| Zoning phase | `#0d9488` | Teal — zoning + supernetwork |
| Demand phase | `#16a34a` | Green — demand, contributions |
| Calibration phase | `#ea580c` | Orange — calibration, accent |
| Scenarios phase | `#7c3aed` | Purple — scenarios, delta |
| Body text | `#1e293b` | Dark slate |
| Subtle text | `#64748b` | Captions, minor labels |
| Background | `#ffffff` | White |
| Subtle background | `#f0f4f8` | Table alternating rows |

---

## Header Strip (full width, ~60mm tall)

**Title**: Analysis of the Impact of Traffic Closures on Transport Using Simulations  
**Subtitle**: Bc. Adam Kaňkovský · Supervisor: Ing. Magdaléna Ondrušková · FIT VUT 2026 · Excel@FIT  
**Logos to include**: Excel@FIT logo (left), FIT VUT logo (right)

---

## Column 1 — Problem, Network, Demand (left third)

### Section: Motivation
> Traffic closures redistribute flows network-wide, causing congestion far from the closed area. This open-source pipeline enables What-If analysis of planned closures in Brno using only open data (OSM, Czech census SLDB 2021, CSD 2025 traffic counts).

### Figure 1: Pipeline Overview Diagram
- **File**: `diagrams/pipeline_diagram.png` (or `.pdf` for vector)
- **Description**: 16-step deterministic pipeline, 5 color-coded phases, data sources on left
- **This should be the LARGEST element on the poster — it tells the whole story**

### Figure 2: Model Area — 219 TAZ + 28 Gateways
- **File**: `../../outputs/baseline/zones/zones_map.png`
- **Caption**: 191 internal + 28 gateway zones, Brno + 2 km buffer
- **Key numbers to overlay**: "219 zones, 28 gateways"

### Figure 3: National Supernetwork
- **File**: `../../outputs/baseline/supernetwork/supernetwork_overview.png`
- **Caption**: 6,117 places → 9 eligible gateways, detour ratio ≤ 1.25, 73k veh/day

---

## Column 2 — Scenario Engine and What-If Analysis (center third)

### Figure 4: Scenario Methodology Diagram
- **File**: `diagrams/scenario_diagram.png` (or `.pdf`)
- **Description**: Shows shared inputs → BFW assignment → SPLIT → baseline/scenario → delta analysis
- **Key annotation**: "Non-destructive: project files unchanged"

### Figure 5: Baseline Traffic Map (V/C Ratio)
- **File**: `screenshots/screenshot_01_baseline_vc.png`
- **Caption**: LOS colored links: green (A) → dark red (F), volume-weighted width

### Figure 6: Scenario — Closure Applied
- **File**: `screenshots/screenshot_02_scenario_panel.png`
- **Caption**: Interactive link selection with scenario panel: full closure or lane reduction
- **Shows**: 3 links added to scenario (Šmejkalova, Brěšská, D1), closure type dropdowns

### Figure 7: Delta Analysis — Traffic Redistribution
- **File**: `screenshots/screenshot_03_delta_view.png`
- **Caption**: Red = increase (spill-over), Blue = decrease (diverted), width ∝ |Δvol|
- **This is the most impactful visual — show it prominently**

---

## Column 3 — Calibration, Diagnostics, Results (right third)

### Figure 8: Calibration & Reports
- **File**: `screenshots/screenshot_04_reports.png`
- **Caption**: ODME convergence over 6 iterations, GEH<5% and GEH<10% line chart

### Table 1: Model Metrics
- **File**: `diagrams/table1_metrics.png` (or `.pdf`)
- **Contains**: All key numbers from the actual model outputs
- **Alternative**: Recreate the table natively in Canva for better visual integration

### Figure 9: Corridor Diagnosis
- **File**: `screenshots/screenshot_06_corridor_diagnosis.png`
- **Caption**: Free-flow path (red) vs forced corridor path (blue), time/distance comparison with diagnosis sidebar

### Figure 10: Through-Traffic Analysis
- **File**: `screenshots/screenshot_07_through_traffic.png`
- **Caption**: Links colored by through-traffic share, gateway markers (D1_NW, I23_W, D2_S…), screenlines

### Section: Key Contributions (4-5 bullets)
> • Reproducible 16-step open-source pipeline (OSM + Czech open datasets only)
> • National supernetwork for data-driven through-traffic classification (6,117 places, detour-filtered)
> • Spiess-style ODME calibration with select-link screenlines and gateway damping
> • Non-destructive scenario engine for interactive What-If closure analysis
> • Rich diagnostic API: corridor explanation, bias clustering, through-traffic overlay

---

## Optional: Figure 11 — Bias Map
- **File**: `screenshots/screenshot_05_bias_map.png`
- **Caption**: Model bias markers — red (over-estimation >120%), blue (under-estimation <80%), green (good match), with DBSCAN clusters

---

## All Asset Files Summary

```
excel_materials/
├── screenshots/
│   ├── screenshot_01_baseline_vc.png      → Figure 5
│   ├── screenshot_02_scenario_panel.png   → Figure 6
│   ├── screenshot_03_delta_view.png       → Figure 7
│   ├── screenshot_04_reports.png          → Figure 8
│   ├── screenshot_05_bias_map.png         → Figure 11 (optional)
│   ├── screenshot_06_corridor_diagnosis.png → Figure 9
│   └── screenshot_07_through_traffic.png  → Figure 10
├── diagrams/
│   ├── pipeline_diagram.pdf/.png          → Figure 1
│   ├── scenario_diagram.pdf/.png          → Figure 4
│   └── table1_metrics.pdf/.png            → Table 1
└── paper/
    └── excel-paper.tex                    → Commentary paper (compile on Overleaf)

Existing images (in outputs/baseline/):
├── zones/zones_map.png                    → Figure 2
├── supernetwork/supernetwork_overview.png → Figure 3
└── maps/links_wgs84.png                  → Fallback network view
```
