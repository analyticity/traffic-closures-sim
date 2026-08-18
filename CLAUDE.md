# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A macroscopic transport simulation pipeline for Czech cities (diploma thesis, Brno University of Technology, FIT). Built on **AequilibraE** (static user-equilibrium traffic assignment) with automated data acquisition from OpenStreetMap, Czech Statistical Office (SLDB 2021), Road and Motorway Directorate (CSD 2025), and NDIC/Police ČR closure feeds. Python backend (`src/sim/`) + a React/TypeScript frontend (`frontend/`, a **git submodule** pointing at `analyticity/bp_ux_ui`, branch `feat/simulation`) that renders maps/reports from the API.

## Commands

```bash
# Install (editable + dev deps; add `,geo` extra for OSM network building)
pip install -e ".[dev]"
pip install -e ".[dev,geo]"

# Unit tests
pytest tests/unit -v
pytest tests/unit/test_calibration_helpers.py -v         # single file
pytest tests/unit/test_calibration_helpers.py::test_foo -v  # single test

# Integration tests (heavy, needs network access + geo extra)
RUN_BUILD_NETWORK_E2E=1 pytest tests/integration -v -m integration --timeout=300

# Lint (ruff config in pyproject.toml, line-length=100; not wired into CI, run manually)
ruff check .

# Frontend (submodule)
cd frontend && npm install
npm run dev      # http://localhost:5173
npm run build    # tsc -b && vite build
npm run lint
```

CI (`.github/workflows/tests.yml`) runs unit tests on Python 3.11 + 3.12 on every push/PR; integration tests only run on pushes to `main`.

### Running the pipeline

Everything goes through `run.py`, which dispatches to ordered `STEPS` (see its module docstring for the full list and prerequisite/staleness logic):

```bash
CFG=config/brno/sim.yaml
python run.py --config $CFG clean
python run.py --config $CFG check
python run.py --config $CFG build-network
python run.py --config $CFG fetch-data
python run.py --config $CFG normalize-network
python run.py --config $CFG build-zones
python run.py --config $CFG build-supernetwork
python run.py --config $CFG build-demand
python run.py --config $CFG assign-warm-skims
python run.py --config $CFG distribute
python run.py --config $CFG assign
python run.py --config $CFG audit-supply
python run.py --config $CFG calibrate
python run.py --config $CFG validate
python run.py --config $CFG learn-profile
python run.py --config $CFG strip-closures
python run.py --config $CFG serve
```

Steps can be run independently once their prerequisites exist — `run.py` validates required input files per step (`_STEP_PREREQUISITES`) and warns about stale outputs (`_STALENESS_CHECKS`) before executing.

New city: `python run.py init-city` (interactive) or `python scripts/generate_city_config.py --city "Olomouc" --okres "Olomouc" --kraj "Olomoucký kraj"`.

## Architecture

### Data flow (why step order matters)

OSM import → normalize (BPR/capacity, closures applied) → TAZ zoning/gateways → national supernetwork (for through-traffic classification) → seed OD matrix → warm-skim assignment (produces network skims) → gravity distribution + IPF (uses those skims) → full user-equilibrium assignment (biconjugate Frank-Wolfe) → supply audit → ODME calibration against observed link counts (Spiess gradient method) → independent validation on holdout CSD sections → temporal profile learning → closure stripping (for scenario baselines) → read-only API server.

Distribution requires skims by default (`demand.distribution.impedance=skim`), which is why `assign-warm-skims` must run before `distribute`.

### `src/sim/` module map

- `network/` — OSM import (`db.py`, `osm_enrichment.py`), normalization pipeline (speeds/capacity/BPR), connectivity fixes, closures, export, CRS handling, map export.
- `zoning/` — TAZ zone construction, gateways (external cordon points), centroid connectors, AOI handling.
- `supernetwork/` — coarse national network used to classify/route through-traffic and derive gateway lookups.
- `datasets/` — download & preprocess external sources: CSD counts, SLDB population/commuting, employment, closures, event links, Waze jams (Postgres-backed), segments.
- `demand/` — OD matrix construction (`od_builder.py`, `seeds.py`, `gateways.py`, `commuting_io.py`), temporal day-type profiles (`temporal.py`).
- `distribution/` — gravity model + impedance + IPF balancing of PA vectors.
- `assignment/` — user-equilibrium traffic assignment via AequilibraE (`graph.py`, `executor.py`, `pipeline.py`), pre-flight checks.
- `calibration/` — ODME (`odme.py`, Spiess gradient), supply audit/tuning, screenline evaluation, observed-data matching (CSD road-ref matching), validation, telemetry.
- `diagnostics/` — parallel-corridor and bias diagnostics exports.
- `scenarios/` — scenario engine (closures/state deltas) for what-if analysis on top of a calibrated baseline.
- `sensitivity.py` — multi-parameter sensitivity sweeps.
- `api.py` — read-only FastAPI server exposing pipeline outputs (network, zones, demand, assignment, scenarios) to the frontend; lazy-loads GeoDataFrames as singletons keyed by file mtime.
- `defaults.py`, `io_project.py` — shared defaults and config/project-path resolution (`config/<city>/sim.yaml` → per-city AequilibraE project + `data/<city>/` + `outputs/<city>/`).

### Multi-city config

Each city lives under `config/<city>/`: `sim.yaml` (city-specific: OSM place name, zoning sources, gateway whitelists, calibration tuning) plus `network_normalization.yaml`, `screenlines.yaml`, `locale.yaml`. All methodology defaults live in code (`defaults.py`), not config. Shared national downloads land in `data/sources/` and are reused across cities; per-city data/outputs/AequilibraE projects are isolated (`data/<city>/`, `outputs/<city>/`, `project/<city>_aeq/`).

`config/experiments.yaml` holds settings shared across `experiments/exp01`–`exp03`, `exp05`–`exp11` (baseline validation, congestion patterns, closures, geometry matching, event-link validation, accident correlation, sensitivity, temporal profiles, quality-score filtering, model-vs-Waze speed comparison). Run via `python experiments/run_all.py --config config/<city>/sim.yaml`.

### Free-flow speed model (frequently-tuned area)

Two-layer speed model per link, configured in `config/<city>/network_normalization.yaml`:

1. **Posted speed** (`normalization.posted_speed.source`): `import_then_fill` (default — keep OSM import, fill gaps from `defaults.speed_by_link_type`), `osm_maxspeed_then_fill` (overwrite from median OSM `maxspeed`, needs re-run of `build-network` enrichment), or `link_type_defaults_only`.
2. **Practical adjustment** (`normalization.practical_speed.mode`): `hcm` (`posted * base_factor - intersection_penalty_per_km * ipkm`, optional `max_ipkm` cap), `factor` (flat multiplier), or disabled (`enabled: false` / `mode: off`).

Minimum speed clamp is a single global floor: `normalization.thresholds.min_speed_kmh`. (`practical_speed.min_speed_kmh` is a legacy key and is ignored if present.) Keep baseline per-city speed tuning under `normalization.*`; only use `experiment_profiles` for deliberate A/B overrides, and avoid duplicating `practical_speed` in both places.

### Docker

Multi-stage `Dockerfile` builds a pipeline stage plus a `serve` stage with baked-in per-city results. `docker-compose.yml` exposes pre-built images per city (`docker compose up brno`, `--profile most`, etc.), each serving the API on `localhost:8000`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_CONFIG` | `config/brno/sim.yaml` | Config path when running uvicorn directly |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL for the frontend |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173` | Allowed origins for API CORS |
