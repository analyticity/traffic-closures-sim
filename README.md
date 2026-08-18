# Traffic Simulation Pipeline

Simulation pipeline for building, calibrating, and validating macroscopic transport models of Czech cities. Built on [AequilibraE](https://www.aequilibrae.com/) (static user-equilibrium assignment) with automated data acquisition from OpenStreetMap, Czech Statistical Office (SLDB 2021), Road and Motorway Directorate (CSD 2025), and NDIC/Police ČR closure feeds.

Developed as part of a diploma thesis at Brno University of Technology, Faculty of Information Technology.

> ## ⑂ This is a fork
>
> The pipeline was originally written by **[Adam Kaňkovský](https://github.com/adamkankovsky)**
> for his diploma thesis at BUT FIT — upstream:
> **[adamkankovsky/Trafic-sim-backend](https://github.com/adamkankovsky/Trafic-sim-backend)**.
>
> This fork is maintained by **Magdalena Ondrušková** under the *analyticity*
> organisation. It branches at commit `7ed13e2`; the complete upstream history is
> preserved, so every original commit keeps its authorship. See **[NOTICE](NOTICE)**
> for authorship and licence details.
>
> ```bash
> git log --oneline 7ed13e2..HEAD    # what this fork changed
> ```
>
> **What this fork adds**
>
> | | |
> |---|---|
> | [popis_simulacie.md](popis_simulacie.md) | Model explained from scratch + where the input vehicle count comes from |
> | [pipeline_detail.md](pipeline_detail.md) | Per-step technical review: inputs, algorithms, risks, defects found |
> | [PLAN_ZLEPSENIA.md](PLAN_ZLEPSENIA.md) | Prioritised improvement plan (P0–P2) with measurable milestones |
> | [KAMDOJIZDIME_PLAN.md](KAMDOJIZDIME_PLAN.md) | Plan for folding kamdojizdime.cz mobile-location data into the model |
>
> Model changes so far: gateways for the four boundary radials that previously had
> none (I/50, II/602, II/380, II/430), removal of the manual CSD exclusion lists,
> population mapping for zones that are themselves municipal parts, and
> `external_local` derived from mobile-location data instead of an unsupported estimate.

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
│   ├── diagnostics/          #   Parallel-corridor & bias diagnostics
│   ├── scenarios/            #   Scenario engine & delta analysis
│   └── sensitivity.py        #   Multi-parameter sensitivity analysis
├── frontend/                 # React/TypeScript map & report UI (Vite + Leaflet)
├── config/                   # Per-city YAML configurations
│   ├── brno/                 #   Brno (default)
│   ├── most/                 #   Most
│   ├── olomouc/              #   Olomouc
│   └── experiments.yaml      #   Shared experiment settings
├── experiments/              # Experiment drivers (exp01–exp03, exp05–exp11)
├── tests/                    # Unit & integration tests
├── scripts/                  # Utility scripts (config generator, benchmarks, etc.)
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

### Free-flow speed: posted vs practical

Free-flow in the SQLite / exported links is built in two conceptual layers:

1. **Posted speed** — what you treat as the legal / nominal limit before urban friction.
   Set `normalization.posted_speed.source` in `config/<city>/network_normalization.yaml`:

   - `import_then_fill` (default): keep speeds from the OSM import, fill only missing values from `defaults.speed_by_link_type`.
   - `osm_maxspeed_then_fill`: overwrite with the **median OSM `maxspeed`** joined at enrichment time (`osm_maxspeed_kmh` column; run `build-network` / enrichment after upgrading).
   - `link_type_defaults_only`: replace posted speeds entirely from `defaults.speed_by_link_type` (single lookup table).

2. **Practical adjustment** — optional second step on top of posted speeds.
   `normalization.practical_speed.mode`:

   - `hcm`: `posted * base_factor - intersection_penalty_per_km * ipkm` (optional `max_ipkm` cap).
   - `factor`: single multiplier `posted * speed_factor` (no intersection density term).
   - Disable with `enabled: false` or `mode: off`.

The **minimum speed clamp** is only `normalization.thresholds.min_speed_kmh` (one global floor). `practical_speed.min_speed_kmh` is ignored if present.

**Config hygiene:** keep baseline city speed tuning under `normalization.*`. Use `experiment_profiles` only for deliberate A/B scenarios — avoid duplicating `practical_speed` in both places unless you intend to override for a specific profile.

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
| 14 | `calibrate-odme` | Explicit alias for Spiess ODME (same as `calibrate` with `method: odme`) |
| 15 | `tune-supply` | Optional outer-loop supply parameter optimization |
| 16 | `validate` | Independent validation on holdout CSD sections |
| 17 | `sensitivity` | Multi-parameter sensitivity analysis |
| 18 | `learn-profile` | Extract temporal day-type factors from CSD |
| 19 | `strip-closures` | Restore pre-closure network for scenario analysis |
| 20 | `serve` | Start read-only FastAPI server |

## Multi-City Support

Each city has its own directory under `config/` with a `sim.yaml` containing city-specific values (OSM place name, zoning sources, gateway whitelists, calibration tuning). Each city directory also includes `network_normalization.yaml`, `screenlines.yaml`, and `locale.yaml`. All methodology defaults are built into the code.

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

Experiment drivers (`experiments/exp01`–`exp03`, `exp05`–`exp11`) cover baseline validation, congestion patterns, closure scenarios, geometry matching, event-link validation, accident correlation, sensitivity analysis, temporal profiles, quality-score filtering, and **model vs Waze free-flow speeds** (`exp11_waze_speeds`, needs PostgreSQL jams from `fetch-data`). A `package_experiment_results.py` utility collects outputs into a single archive.

Run all experiments for a city:

```bash
python experiments/run_all.py --config config/brno/sim.yaml
```

Waze speed comparison only (after `assign` and `fetch-data`):

```bash
python experiments/exp11_waze_speeds.py --config config/brno/sim.yaml
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
