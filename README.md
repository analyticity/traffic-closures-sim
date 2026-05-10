# Traffic Simulation Pipeline

Simulation pipeline for building, calibrating, and validating macroscopic transport models of Czech cities. Built on [AequilibraE](https://www.aequilibrae.com/) (static user-equilibrium assignment) with automated data acquisition from OpenStreetMap, Czech Statistical Office (SLDB 2021), Road and Motorway Directorate (CSD 2025), and NDIC/Police ČR closure feeds.

Developed as part of a diploma thesis at Brno University of Technology, Faculty of Information Technology.

## Project Structure

```
├── run.py                    # CLI entrypoint for all pipeline steps
├── src/sim/                  # Core library
│   ├── api.py                #   FastAPI REST server
│   ├── network/              #   OSM import, normalization, closures, export
│   ├── zoning/               #   TAZ zones, gateways, centroid connectors
│   ├── datasets/             #   External data download & preprocessing
│   ├── supernetwork/         #   National coarse network for through-traffic
│   ├── demand/               #   OD matrix construction & temporal profiles
│   ├── distribution/         #   Gravity model & IPF
│   ├── assignment/           #   User-equilibrium traffic assignment
│   ├── calibration/          #   ODME calibration, validation, supply audit
│   └── scenarios/            #   Scenario engine & delta analysis
├── frontend/                 # React/TypeScript map & report UI (Vite + Leaflet)
├── config/                   # Per-city YAML configurations
│   ├── brno/                 #   Brno (default)
│   ├── most/                 #   Most
│   └── olomouc/              #   Olomouc
├── experiments/              # Experiment drivers (exp01–exp10)
├── tests/                    # Unit & integration tests
├── scripts/                  # Utility scripts (config generator, audit, etc.)
├── scenarios/                # Example scenario YAML definitions
├── Dockerfile                # Multi-stage Docker build (pipeline + serve)
├── docker-compose.yml        # Pre-built city images
└── pyproject.toml            # Python packaging & dependencies
```

## Prerequisites

| Tool | Version | Note |
|------|---------|------|
| Python | >= 3.10 (tested on 3.11, 3.12) | |
| Node.js | >= 18 LTS | Only needed for the frontend |
| libspatialindex | system package | Required by `rtree` |

Fedora / RHEL:

```bash
sudo dnf install python3 python3-pip nodejs npm libspatialindex-devel
```

Ubuntu / Debian:

```bash
sudo apt install python3 python3-pip python3-venv nodejs npm libspatialindex-dev
```

## Installation

```bash
git clone <repo-url> && cd simulation

python3 -m venv .venv
source .venv/bin/activate

# Core install
pip install -e ".[dev]"

# For OSM-based network building (build-network, build-supernetwork)
pip install -e ".[dev,geo]"
```

Verify with:

```bash
pytest tests/unit -v
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

The frontend connects to the backend API at `http://localhost:8000` by default (override with `VITE_API_URL`).

## Running the Pipeline

All pipeline steps are executed through `run.py`:

```bash
python run.py --config config/<city>/sim.yaml <step>
```

### Full Workflow

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

Steps can be run independently once their prerequisites exist. The runner validates prerequisites and warns about stale outputs before execution.

### Pipeline Steps

| # | Step | Description |
|---|------|-------------|
| 0 | `init-city` | Generate minimal config for a new city |
| 1 | `clean` | Remove city-specific generated data (`--force` also removes shared sources) |
| 2 | `check` | Verify project bootstrap and config |
| 3 | `build-network` | Import road network from OpenStreetMap |
| 4 | `fetch-data` | Download external datasets (CSD, SLDB, closures, population) |
| 5 | `normalize-network` | Normalize attributes, compute BPR/capacity, apply closures |
| 6 | `build-zones` | Create TAZ zones, external gateways, centroid connectors |
| 7 | `build-supernetwork` | Build national coarse network for through-traffic classification |
| 8 | `build-demand` | Build seed OD matrix (commuting + synthetic + external segments) |
| 9 | `assign-warm-skims` | Short assignment to produce network skims for distribution |
| 10 | `distribute` | Gravity model calibration + IPF balancing |
| 11 | `assign` | Full user-equilibrium traffic assignment (biconjugate Frank-Wolfe) |
| 12 | `audit-supply` | Supply-side diagnostics (prerequisite for ODME) |
| 13 | `calibrate` | Spiess gradient ODME against observed link counts |
| 14 | `tune-supply` | Optional outer-loop supply parameter optimization |
| 15 | `validate` | Independent validation on holdout CSD sections |
| 16 | `learn-profile` | Extract temporal day-type factors from CSD |
| 17 | `strip-closures` | Restore pre-closure network for scenario analysis |
| 18 | `serve` | Start read-only FastAPI server |

## Multi-City Support

Each city has its own directory under `config/` with a minimal `sim.yaml` (~40 lines) containing only city-specific values (OSM place name, zoning sources, gateway whitelists). All methodology defaults are built into the code.

Add a new city interactively:

```bash
python run.py init-city
```

Or non-interactively:

```bash
python scripts/generate_city_config.py --city "Olomouc" --okres "Olomouc" --kraj "Olomoucký kraj"
```

Data is automatically separated per city (`data/<city>/`, `outputs/<city>/`, `project/<city>_aeq/`). Shared national downloads in `data/sources/` are reused across cities.

## Docker

Pre-built images with baked-in pipeline results are available via `docker-compose.yml`:

```bash
docker compose up brno           # Brno API at localhost:8000
docker compose --profile most up # Most
```

Build locally:

```bash
docker build --build-arg CITY=brno -t simulation-brno .
docker run --network host simulation-brno
```

## Experiments

Ten experiment drivers (`experiments/exp01`–`exp10`) cover baseline validation, free-flow speed comparison, congestion patterns, closure scenarios, geometry matching, event-link validation, accident correlation, sensitivity analysis, temporal profiles, and quality-score filtering.

Run all experiments for a city:

```bash
python experiments/run_all.py --config config/brno/sim.yaml
```

## Tests

```bash
pytest tests/unit -v                                        # unit tests
RUN_BUILD_NETWORK_E2E=1 pytest tests/integration -v -m integration --timeout=300  # integration
```

CI runs unit tests on Python 3.11 and 3.12 on every push; integration tests run on main.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_CONFIG` | `config/brno/sim.yaml` | Config path when running uvicorn directly |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL for the frontend |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173` | Allowed origins for API CORS |

## License

This software was developed as part of a diploma thesis. See the thesis text for full terms.
