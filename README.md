# Simulation Pipeline

This project provides a step-by-step simulation pipeline for building, calibrating, validating, and serving a transport model.

The pipeline is executed through a single runner script, where each step represents one logical phase of the workflow. The individual phases are designed so they can be run separately, which makes debugging, iteration, and experimentation easier.

---

## Overview

The pipeline supports the following workflow:

0. **init-city** – generate a minimal config directory for a new city (interactive or CLI)  
1. **clean** – remove generated data and start from a clean state  
2. **check** – verify that the AequilibraE project is correctly bootstrapped  
3. **build-network** – import the road network from OpenStreetMap  
4. **fetch-data** – download and preprocess external datasets (closures, counts, population, etc.)  
5. **normalize-network** – clean and normalize network attributes; optionally apply baseline road closures  
6. **build-zones** – create TAZ zones, external gateway zones, and centroid connectors; auto-remaps population  
7. **build-supernetwork** – coarse national network for external / through traffic at gateways  
8. **build-demand** – build the seed OD matrix (commuting, gateways, synthetic segments)  
9. **assign-warm-skims** – optional short assignment that always saves `skims.aem` for trip distribution  
10. **distribute** – gravity calibration + IPF on the seed matrix (network skims or Euclidean; see below)  
11. **assign** – full traffic assignment on the detailed network  
12. **calibrate** / **calibrate-odme** – Spiess gradient ODME against link counts (default); set `calibration.method` for alternatives  
13. **tune-supply** – optional outer loop on supply-side factors after demand calibration  
14. **validate** – independent checks vs. CSD (not used in calibration)  
15. **learn-profile** – temporal day-type factors from CSD  
16. **strip-closures** – restore pre-closure network attributes (clean baseline after validation)  
17. **serve** – read-only REST API for results  

---

## Runner

The pipeline is controlled by the main runner script:

```bash
python run.py --config config/brno/sim.yaml <step>
```

Example:

```bash
python run.py --config config/brno/sim.yaml build-network
```

If `--config` is not provided, the default configuration file is:

```bash
config/brno/sim.yaml
```

---

## Multi-City Configuration

The pipeline is designed to run for **any Czech city** without code changes.
Each city has its own directory under `config/` containing only city-specific
values; all methodology defaults (speeds, capacities, BPR parameters, Czech
national datasets) are built into the code.

```
config/
  brno/           # Brno-specific configs
    sim.yaml
    locale.yaml
    screenlines.yaml
    network_normalization.yaml
  most/           # Most-specific configs
    sim.yaml
    locale.yaml
    screenlines.yaml
  experiments.yaml  # shared tuning policy
```

### Adding a new city

The fastest way is to use the built-in config generator:

```bash
# Interactive — answers a few questions (city, district, region):
python scripts/generate_city_config.py

# Or non-interactive:
python scripts/generate_city_config.py \
    --city "Olomouc" \
    --okres "Olomouc" \
    --kraj "Olomoucký kraj"
```

This creates `config/olomouc/sim.yaml`, `locale.yaml`, and `screenlines.yaml`.
The generator is also available as a pipeline step:

```bash
python run.py init-city
```

### What goes into a city config

A minimal `sim.yaml` (~40 lines) only contains values unique to that city:

| Key | Purpose | Example |
|-----|---------|---------|
| `project_path` | AequilibraE project directory | `project/olomouc_aeq` |
| `osm.place_name` | OSM Nominatim query for the network extent | `Olomouc, Czechia` |
| `osm.buffer_km` | *(optional)* When using `place_name` without `model_bbox`, buffer the geocoded polygon by this many km for import/trim (default `0` = no extra buffer; e.g. `2` for a 2 km belt) | `2` |
| `zoning.sources` | Admin boundaries for TAZ zones | `Okres Olomouc, Czechia` |
| `zoning.external_gateways.whitelist` | Explicit gateway roads (empty = auto-discover) | `[]` |
| `demand.segments.external_local.total_daily_trips` | Scale of external traffic (`"auto"` = estimate from CSD data on gateway roads; or explicit number) | `"auto"` |
| `datasets.enabled` | Turn off all dataset fetching when `false` (omit or `true` to run `fetch-data`) | `true` |
| `datasets.sources.commuting_sldb2021.filter` | SLDB commuting district filter | `Olomouc` |
| `datasets.sources.validation_csd2025_v2.usage.area_filter.region_hint` | CSD region | `Olomoucký kraj` |

Everything else — BPR alpha/beta tables, daily capacity factors, normalization
defaults, multi-class settings, Czech dataset URLs, holidays, calibration
parameters — comes from code defaults and does not need to appear in YAML.

### Running the pipeline for a specific city

Always pass `--config` pointing to the city's `sim.yaml`:

```bash
# Brno (default if --config is omitted):
python run.py --config config/brno/sim.yaml build-network

# Most:
python run.py --config config/most/sim.yaml build-network

# A new city:
python run.py --config config/olomouc/sim.yaml build-network
```

### Data layout and separation

Data is automatically separated per city. The city slug is derived from the
config directory name (e.g. `config/brno/` → `brno`). All derived paths are
namespaced under that slug:

```
data/
  sources/                    # shared national downloads (never deleted by clean)
    osm/                      #   czech-republic-latest.osm.pbf  (~900 MB)
    csu/sldb2021/             #   commuting CSV + derived ``*_full_cr.parquet`` (same for all cities)
    rsd/csd2025/              #   traffic count XLSX + derived ``v2_csd2025.parquet``
    cz/places/                #   RUIAN ZIP + derived ``cz_place_centroids.parquet``
    brno/intensity/           #   optional city-specific sources (e.g. pentlogram GeoJSON)
  brno/                       # Brno-specific derived data
    cache/                    #   filtered parquets, closures, centroids, manifest
      supernetwork/           #   national graph cache for through-traffic
    demand/                   #   od_matrix.aem
  most/                       # Most-specific derived data
    cache/
      supernetwork/
    demand/
outputs/
  brno/baseline/              # Brno outputs (network, zones, maps, demand, supernetwork)
  most/baseline/              # Most outputs
project/
  brno_aeq/                   # AequilibraE project (already city-specific)
  most_aeq/
```

Running `clean` for one city only removes its `data/<city>/`, `outputs/<city>/`,
and `project/<city>_aeq/` directories. Shared downloads in `data/sources/` are
preserved.

National derived files (full-CR SLDB, CSD parquet, RUIAN centroids) are written
next to their downloads under `data/sources/` so one `fetch-data` run serves
every city; only city-filtered commuting parquet and other model artefacts live
under `data/<city>/cache/`.

---

## Typical Full Workflow

A standard end-to-end workflow for a city looks like this:

```bash
CFG=config/brno/sim.yaml          # change to your city

python run.py --config $CFG clean
python run.py --config $CFG check
python run.py --config $CFG build-network
python run.py --config $CFG fetch-data                # downloads closures, counts, CSD, population
python run.py --config $CFG normalize-network         # applies baseline closures if configured
python run.py --config $CFG build-zones               # auto-remaps population to zones
python run.py --config $CFG build-supernetwork
python run.py --config $CFG build-demand
python run.py --config $CFG assign-warm-skims
python run.py --config $CFG distribute
python run.py --config $CFG assign
python run.py --config $CFG calibrate                 # Spiess ODME by default
# python run.py --config $CFG tune-supply             # optional; disabled by default
python run.py --config $CFG validate
python run.py --config $CFG learn-profile
python run.py --config $CFG strip-closures            # remove closures for clean baseline
python run.py --config $CFG serve
```

In practice, not every run has to execute all steps. Once intermediate artifacts are generated, later steps can usually be rerun independently.

**Population auto-remap.** `fetch-data` downloads population data and produces `zone_population.parquet`. If zones do not exist yet at that point, the parquet uses a fallback `zone_id=0`. When `build-zones` runs later, it detects this stale fallback and automatically regenerates `zone_population.parquet` with correct zone IDs — no second `fetch-data` call is needed.

**Skim-driven trip distribution.** The first run of `distribute` has no `skims.aem` unless you assigned traffic first. With `demand.distribution.impedance: auto` (default), gravity/IPF then uses **Euclidean distance** between zone centroids. To use **network travel times** as impedance, run **`assign-warm-skims`** (or a full `assign` with `calibration.save_skims: true`) **before** `distribute`, then `distribute`, then run **`assign` again** for production link volumes on the IPF-adjusted matrix. A warm pass overwrites `assignment_results.parquet` with an intermediate result; the final assignment pass replaces it.

---

## Pipeline Steps

### 1. `clean`

Deletes **city-specific** generated data so the next run starts from scratch.

```bash
python run.py --config config/brno/sim.yaml clean          # city data only
python run.py --config config/brno/sim.yaml --force clean   # + shared sources
```

This step removes:

* the AequilibraE project directory (`project/<city>_aeq/`)
* city-specific outputs (`outputs/<city>/`)
* city-specific cache and demand data (`data/<city>/`)

**Shared national data** in `data/sources/` (OSM PBF ~900 MB, SLDB, CSD,
RUIAN) is **preserved by default** — it does not need to be re-downloaded when
cleaning a single city. Use `--force` to also remove `data/sources/` when you
truly need a from-scratch state including re-downloading all sources.

---

### 2. `check`

Verifies that the project bootstrap is correct.

This is intended as an early sanity check before running the full workflow. It helps confirm that the configuration, project structure, and AequilibraE setup are valid.

---

### 3. `build-network`

Imports the transport network from OpenStreetMap into the AequilibraE project.

This is the first real model-building step. It creates the base network representation that all later steps depend on.

The model bbox in config is used for maps and metadata; the full OSM extract is kept in the project. **`network.drivable_network`** (enabled by default) removes non-road OSM classes (`footway`, `path`, `cycleway`, …) and, if `require_mode_car` is true, any link whose `modes` string does not contain car mode `c`. Set `enabled: false` to keep the full multimodal extract. Override `excluded_link_types` to change the drop list.

**`network.isolated_components`** (enabled by default) keeps only the largest undirected connected component by link count and deletes all other fragments, then prunes orphan nodes. Set `enabled: false` to retain every disconnected subgraph from the extract.

**`osm.buffer_km`** (optional, default `0`): Only applies when the network is built from **`osm.place_name`** and you do **not** set `model_bbox` / `osm.bbox`. If **`buffer_km` > 0**, the place is geocoded to a polygon, then buffered by that many kilometres (in a metric CRS) to define the **model extent**; the OSM download area adds a fixed **5 km** margin on top for connectivity, after which the usual urban-core trim still runs. If **`buffer_km` is 0** (default), the importer uses the place polygon / name query without this configurable outer buffer.

---

### 4. `normalize-network`

Cleans and normalizes the imported network.

Typical tasks in this phase include:

* standardizing attribute values
* cleaning inconsistent link metadata
* computing BPR function parameters, capacities, and free-flow speeds
* applying baseline road closures (if configured)
* preparing the network for zoning and assignment
* exporting a normalized version for downstream processing

This step is important because raw OSM data is usually not directly suitable for assignment.

**Baseline closures** (`baseline_closures` in the city's `sim.yaml`): when enabled, road closures from the Police ČR XML feed (fetched by `fetch-data`) are spatially matched to network links. Affected links receive reduced capacity and speed (via a configurable `capacity_reduction_factor`). Original values are stored in `_preclosure_*` columns so they can be restored later by `strip-closures`.

**Configuration**

* `network.normalization_config` in the city's `sim.yaml` points to a YAML file (e.g. `config/brno/network_normalization.yaml`) with:
  * `normalization.defaults` — per–`link_type` fallbacks for speed, lanes, and capacity per lane when OSM/AequilibraE leaves gaps
  * `normalization.thresholds` — global floors (minimum speed/capacity/travel time, generic capacity per lane, fallback speed)
  * `experiment_profiles` — named sets of speed caps/floors, capacity multipliers, and time penalties; `network.experiment_profile` selects one (`baseline` applies no extra tweaks)
* You can override or extend any subsection inline under `network.normalization` / `network.experiment_profiles` in `sim.yaml`; values merge on top of the file.

**Exports**

* **`network_links.gpkg`** / **`network_nodes.gpkg`** use **`crs_epsg`** from `sim.yaml` (same CRS as the AequilibraE project). Geometries are in metres and are consistent with the `distance` column (also metres).
* **`network_links.geojson`** / **`network_nodes.geojson`** are reprojected to **WGS84** for web maps; do not use raw GeoJSON geometry length as metres—use `distance` or the GPKG layer for metric analysis.

**Data sources**

* Observed speeds and lane counts should come from **OpenStreetMap** (`maxspeed`, `lanes`, directional variants) where tagged; they are imported with `build-network` and only filled from defaults where missing.
* For **Czechia**, statutory limits depend on context (built-up vs. outside, motorway, signed residential zone). OSM `highway=residential` means typical access streets in housing areas, **not** the legal Czech *obytná zóna* (20 km/h); that is closer to `highway=living_street` or explicit `zone` tagging. OSM `highway=*` alone does not capture built-up vs. outside. Defaults in `network_normalization.yaml` are documented there relative to Act No. 361/2000 Sb.; use `maxspeed` / `zone` in OSM wherever possible so the model is not tied to a single heuristic table.
* **Capacity:** `capacity_ab` / `capacity_ba` are **total veh/h in that direction** (per-lane rates from YAML × directional lane count). AequilibraE uses one value per directed arc; unused direction on one-way links is ignored. The YAML table is **order-of-magnitude / uninterrupted-flow style**, not a full signalised-intersection model—calibrate against local counts (e.g. Czech intensity datasets) if you need realistic absolute volumes; assignment may apply separate time multipliers on urban classes.
* Finer speed calibration (e.g. probe or commercial speed layers) is optional and not required for this step.

---

### 5. `build-zones`

Builds transport analysis zones (TAZ), external gateway zones, and centroid connectors.

This phase creates the zoning system used for demand modeling and connects zone centroids to the road network so OD flows can enter and leave the network during assignment.

**External gateways** represent entry/exit points at the model boundary for traffic to/from outside the study area. They are created from two sources:

* **Whitelist** (`zoning.external_gateways.whitelist`) — explicit list of road refs (e.g. `D1`, `I/50`). These are always kept in the supernetwork for through-traffic routing.
* **Auto-discover** (`zoning.external_gateways.auto_discover`) — additional named roads near the model boundary not on the whitelist. These secondary gateways receive only local external demand, not through-traffic.

**Class-aware merging**: gateways close to each other on the model boundary are merged, but only within the same road class group (motorway/trunk, primary, secondary). This prevents merging of different corridors that happen to cross the boundary near the same point (e.g. D1 motorway and I/50 primary on the east).

**`max_anchor_distance_m`** (default 2000): gateways whose network anchor node is farther than this from the boundary are skipped, preventing phantom connectors for roads that don't actually reach the model edge.

---

### 6. `fetch-data`

Downloads and preprocesses external datasets required by the model.

These datasets may include demand, count, validation, temporal reference, or road closure inputs (e.g. Police ČR XML feed). The step prepares them into a consistent internal format for the following phases.

Set `datasets.enabled: false` in `sim.yaml` only if you want to skip this step entirely (default is to run whenever `fetch-data` is invoked).

---

### 7. `build-supernetwork`

Builds a coarse national road network for classifying external (through) traffic.

Gateways from `build-zones` are snapped to the national graph (built from the Czech OSM `.pbf`). For each external municipality, the system determines the best gateway(s) by shortest-path cost. Through-traffic demand between gateway pairs is computed from SLDB commuting data filtered by detour/time thresholds.

**`supernetwork.eligible_gateway_types`**: only gateways with matching road types participate in national routing. Auto-discovered secondary gateways are always excluded to prevent distorted place assignments. Whitelist gateways are always kept regardless of their OSM road class.

**Demand distribution** for `external_local` trips uses road-class-based weights derived automatically from `gateway_diagnostics.csv`: motorway/trunk gateways get weight 1.0, primary 0.6, secondary 0.2. Explicit overrides can be set in `corridor_weights`.

---

### 8. `build-demand`

Builds the initial OD matrix from SLDB commuting data and synthetic demand segments.

This produces the base origin-destination demand representation, which acts as the seed for later distribution and calibration. The matrix includes:

* **Commuting** — from SLDB (Czech census) origin-destination flows, converted to vehicles per period
* **Other** — synthetic gravity-model trips for non-commuting purposes
* **External local** — residual gateway-to-internal trips, weighted by road class
* **External through** — data-driven through-traffic from supernetwork gateway pairs

When `demand.sldb.external_processing.enabled` is true, the runner checks for supernetwork outputs (gateway lookup parquet, through pairs if configured, and `supernetwork_summary.json`) before building demand. With external processing off, only zones and centroid mapping from `build-zones` are required.

---

### 9. `assign-warm-skims`

Optional shorter traffic assignment whose main purpose is to write **`skims.aem`** under `demand.output_dir` for use as impedance in `distribute`.

* Uses `assignment.warm_skim_pass` in the city's `sim.yaml` for `algorithm`, `max_iter`, and `rgap_target` (defaults are lighter than full `calibrate` / `assign` settings).
* Always saves skims (`save_skims` is forced on for this step).
* Writes `assignment_results.parquet` like `assign`; treat it as an intermediate artifact if you run a full `assign` afterward.

---

### 10. `distribute`

Runs demand distribution and balancing.

This step typically includes:

* gravity-model calibration
* iterative proportional fitting (IPF)
* adjustment of the seed OD matrix

The goal is to transform the initial demand into a network-ready OD matrix consistent with constraints and observed structure.

**`demand.distribution.impedance`**

* **`auto`** (default): load `skims.aem` from `demand.output_dir` if it exists; otherwise use Euclidean distance between zone centroids.
* **`skim`**: require `skims.aem`; fail with a clear error if it is missing (run `assign-warm-skims` or `assign` with `calibration.save_skims: true` first).

---

### 11. `assign`

Runs traffic assignment.

This step loads the OD matrix onto the network using a selected assignment procedure, such as:

* shortest-path assignment
* equilibrium assignment

The result is an estimate of flows on individual network links.

---

### 12. `calibrate` / `calibrate-odme`

Performs iterative bi-level OD matrix calibration against observed link counts.

The default method is **Spiess gradient ODME** — a bi-level loop where the lower level runs AequilibraE equilibrium assignment and the upper level adjusts the OD matrix using select-link proportions from screenlines:

1. run equilibrium assignment (+ select-link OD extraction per screenline)
2. match assigned volumes to observed counts (CSD 2025 or pentlogram)
3. compute weighted SSE objective Z and per-screenline obs/mod ratios
4. apply Spiess multiplicative OD update across screenlines (multiple inner gradient steps)
5. apply global residual scaling, per-road-class residual, and gateway calibration
6. repeat until Z converges, daily criteria are met, or stall patience is exhausted

Both `calibrate` and `calibrate-odme` run ODME by default. To select a different method, set `calibration.method` in your city's `sim.yaml`:

| `calibration.method` | Runner function | Description |
|---|---|---|
| `odme` (default) | `run_odme_calibration` | Spiess gradient ODME with adaptive damping and best-snapshot recovery |
| `entropy_odme` | `run_entropy_odme` | Entropy-maximization variant (log-ratio update, better seed structure preservation) |
| `multistage` | `run_multistage_calibration` | 4-stage pipeline: gravity re-fit, gateway pre-calib, tight-bounds ODME, screenline fine-tuning |
| `fsm` | `run_calibration` | Legacy iterative method (1 inner step per iteration, no explicit Z objective) |

**Configuration (excerpt)**

* `calibration.aggregate_corridor` — when `true`, volumes on parallel divided-highway links are summed for comparison to a single count station (recommended for motorways).
* `calibration.odme.max_outer_iterations` — maximum ODME iterations (default 40).
* `calibration.odme.gradient_descent_iterations` — inner Spiess steps per outer iteration (default 8).
* `calibration.odme.max_deviation` — max multiplicative deviation from seed OD (default 6.0).
* `calibration.odme.convergence_tol` — relative Z change threshold for convergence (default 0.001).
* `calibration.count_source` — `csd_split` (default, CSD 2025) or `pentlogram` (Brno ArcGIS layer).

---

### 13. `tune-supply`

Runs outer-loop optimization of supply-side parameters.

While `calibrate` focuses on repeated internal adjustment, this step searches for better supply-related parameters at a higher level. Enable with `calibration.supply_tuning.enabled`. The composite objective weights GEH, screenline fit, and journey-time checks (fixed weights in code).

---

### 14. `validate`

Runs independent validation using CSD 2025.

Unlike calibration, this phase tests the model on separate validation data to assess generalization and robustness. It helps confirm whether the calibrated model behaves reasonably outside the calibration target.

---

### 15. `learn-profile`

Learns temporal day-type factors from CSD 2025.

This step extracts temporal behavior patterns that can later be used to derive profiles for different types of days or time periods.

It is especially useful when moving from a static baseline toward more realistic temporal interpretation of demand.

---

### 16. `strip-closures`

Restores the network to its pre-closure state after validation.

During `normalize-network`, baseline closures (from the Police ČR feed or manual input) can reduce link capacity and speed to match real-world conditions during the measurement period. After validation confirms the model against observed data, `strip-closures` reverts these reductions by restoring values from `_preclosure_*` columns, yielding a clean baseline network suitable for scenario analysis.

---

### 17. `serve`

Starts a read-only REST API server.

This makes the processed results accessible through an API, which is useful for:

* visualization layers
* external dashboards
* downstream consumers
* scenario inspection tools

This step assumes the model outputs already exist.

Always pass the **same** `sim.yaml` as for the pipeline (e.g. `python run.py --config config/most/sim.yaml serve`), so the API reads `outputs/<city>/…`. The runner’s default `--config` is **`config/brno/sim.yaml`**, so `serve` without `--config` will look under `outputs/brno/…` even if you built another city.

If you start the app with **`uvicorn sim.api:app`** instead, set **`SIM_CONFIG`** to that file (defaults to `config/brno/sim.yaml`), e.g. `SIM_CONFIG=config/most/sim.yaml uvicorn sim.api:app`, and run from the repository root.

---

## Command-Line Interface

The runner exposes the following steps:

* `init-city`
* `clean`
* `check`
* `build-network`
* `normalize-network`
* `build-zones`
* `fetch-data`
* `build-supernetwork`
* `build-demand`
* `assign-warm-skims`
* `distribute`
* `assign`
* `calibrate`
* `calibrate-odme`
* `tune-supply`
* `validate`
* `learn-profile`
* `strip-closures`
* `serve`

Basic usage:

```bash
python run.py <step>
```

With explicit config:

```bash
python run.py --config config/brno/sim.yaml <step>
```

---

## Full pipeline audit

For an **end-to-end smoke test** of the runner, use [`scripts/full_pipeline_audit.py`](scripts/full_pipeline_audit.py). It runs the main pipeline steps in a fixed order (each step is invoked as `python run.py --config … <step>`), checks that expected files exist after each step, scans logs for tracebacks and common warnings, and writes a report bundle under the audit output directory.

**Run from the repository root:**

```bash
python scripts/full_pipeline_audit.py
```

**Useful options:**

| Option | Default | Meaning |
|--------|---------|---------|
| `--config` | `config/brno/sim.yaml` | Config passed to every `run.py` step |
| `--out-root` | `outputs/audit/full_pipeline_audit` | Directory for all audit outputs (under repo root) |
| `--timeout-s` | `1200` | Per-step timeout in seconds (heavy steps use longer overrides inside the script) |
| `--stop-on-fail` | off | Stop after the first step with status `fail` or `blocked` instead of continuing |

**Outputs** (under `--out-root`):

* `pipeline_audit.csv` / `pipeline_audit.json` — per-step exit code, elapsed time, status (`ok` / `warning` / `fail` / `blocked`), missing artifacts, warning tags
* `step_logs/<step>.log` — captured stdout/stderr for each step
* `config_snapshot.yaml` — copy of the config used for the run
* `magic_constants_inventory.csv`, `metrics_consistency_report.md`, `data_driven_refactor_proposal.md`, `unused_policy_keys.md`, `before_after_compare.json` — auxiliary reports used for consistency / documentation reviews

The list of steps and expected artifact paths is maintained in the script; it assumes layout consistent with the default paths in the city's `sim.yaml` (for example the AequilibraE project under `project_path`). After a successful `clean`, the audit directory may be removed and is recreated as the run continues.

---

## Recommended Execution Strategy

For development, it is usually best to run the workflow incrementally:

* run `clean` only when necessary
* use `check` before rebuilding everything
* rebuild the network only when OSM-related inputs change
* rerun `build-zones` when zoning logic or gateway config changes; then rerun `build-demand` onward (the OD matrix dimension must match the zone set)
* rerun `build-supernetwork` when gateway or supernetwork config changes
* rerun `build-demand` or `distribute` when demand inputs change
* use `assign-warm-skims` before `distribute` when you want IPF/gravity driven by network skims instead of Euclidean distance
* rerun `assign`, `calibrate`, and `validate` frequently during model tuning
* run `strip-closures` after validation to produce a clean baseline for scenario work
* use `serve` only after the required outputs have been prepared

This keeps iteration fast and avoids recomputing expensive earlier phases unnecessarily.

---

## Notes

* Many steps depend on outputs generated by previous steps.
* The pipeline is designed to support both full rebuilds and partial reruns.
* Each city has its own config directory (e.g. `config/brno/`, `config/most/`). See **Multi-City Configuration** above for details and the config generator.
* The script inserts `src/` into `sys.path`, so project modules are loaded directly from the source tree.
* Each `run.py` invocation accepts exactly one step: `python run.py build-demand`. Chain calls in a shell script or run them sequentially.

---

## Entry Point

The workflow is implemented in the main script and dispatches individual steps to dedicated modules such as:

* `sim.network` — OSM import, filtering, CRS helpers, map export, attribute normalization, connectivity repair, baseline closures, network export
* `sim.zoning` — TAZ and gateway zone generation
* `sim.datasets` — external data download & preprocessing (CSD, SLDB, closures, ArcGIS, RUIAN centroids)
* `sim.supernetwork` — national coarse network for through-traffic
* `sim.demand` — OD matrix construction
* `sim.distribution` — gravity model and IPF
* `sim.assignment` — traffic assignment
* `sim.calibration` — calibration, ODME, validation, supply tuning
* `sim.scenarios` — scenario engine and delta analysis
* `sim.temporal` — temporal profile learning
* `sim.api` — REST API server

This keeps the runner lightweight while the domain logic remains separated into dedicated components.

Batch validation of the same steps is described under **Full pipeline audit** above ([`scripts/full_pipeline_audit.py`](scripts/full_pipeline_audit.py)).
