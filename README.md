# Simulation Pipeline

This project provides a step-by-step simulation pipeline for building, calibrating, validating, and serving a transport model.

The pipeline is executed through a single runner script, where each step represents one logical phase of the workflow. The individual phases are designed so they can be run separately, which makes debugging, iteration, and experimentation easier.

---

## Overview

The pipeline supports the following workflow:

0. **clean** – remove generated data and start from a clean state  
1. **check** – verify that the AequilibraE project is correctly bootstrapped  
2. **build-network** – import the road network from OpenStreetMap  
3. **fetch-data** – download and preprocess external datasets (closures, counts, population, etc.)  
4. **normalize-network** – clean and normalize network attributes; optionally apply baseline road closures  
5. **build-zones** – create TAZ zones, external gateway zones, and centroid connectors; auto-remaps population  
6. **build-supernetwork** – coarse national network for external / through traffic at gateways  
7. **build-demand** – build the seed OD matrix (commuting, gateways, synthetic segments)  
8. **assign-warm-skims** – optional short assignment that always saves `skims.aem` for trip distribution  
9. **distribute** – gravity calibration + IPF on the seed matrix (network skims or Euclidean; see below)  
10. **assign** – full traffic assignment on the detailed network  
11. **calibrate** / **calibrate-odme** – iterative demand scaling vs. link counts (pentlogram)  
12. **tune-supply** – optional outer loop on supply-side factors after demand calibration  
13. **validate** – independent checks vs. CSD (not used in calibration)  
14. **learn-profile** – temporal day-type factors from CSD  
15. **strip-closures** – restore pre-closure network attributes (clean baseline after validation)  
16. **serve** – read-only REST API for results  

---

## Runner

The pipeline is controlled by the main runner script:

```bash
python run.py --config config/sim.yaml <step>
```

Example:

```bash
python run.py --config config/sim.yaml build-network
```

If `--config` is not provided, the default configuration file is:

```bash
config/sim.yaml
```

---

## Typical Full Workflow

A standard end-to-end workflow usually looks like this:

```bash
python run.py clean
python run.py check
python run.py build-network
python run.py fetch-data                # downloads closures, counts, CSD, population
python run.py normalize-network         # applies baseline closures if configured
python run.py build-zones               # auto-remaps population to zones
python run.py build-supernetwork
python run.py build-demand
python run.py assign-warm-skims
python run.py distribute
python run.py assign
python run.py calibrate                 # or calibrate-odme for ODME method
python run.py tune-supply
python run.py validate
python run.py learn-profile
python run.py strip-closures            # remove closures for clean baseline
python run.py serve
```

In practice, not every run has to execute all steps. Once intermediate artifacts are generated, later steps can usually be rerun independently.

**Population auto-remap.** `fetch-data` downloads population data and produces `zone_population.parquet`. If zones do not exist yet at that point, the parquet uses a fallback `zone_id=0`. When `build-zones` runs later, it detects this stale fallback and automatically regenerates `zone_population.parquet` with correct zone IDs — no second `fetch-data` call is needed.

**Skim-driven trip distribution.** The first run of `distribute` has no `skims.aem` unless you assigned traffic first. With `demand.distribution.impedance: auto` (default), gravity/IPF then uses **Euclidean distance** between zone centroids. To use **network travel times** as impedance, run **`assign-warm-skims`** (or a full `assign` with `calibration.save_skims: true`) **before** `distribute`, then `distribute`, then run **`assign` again** for production link volumes on the IPF-adjusted matrix. A warm pass overwrites `assignment_results.parquet` with an intermediate result; the final assignment pass replaces it.

---

## Pipeline Steps

### 1. `clean`

Deletes generated data so the next run starts from scratch.

This step removes:

* the AequilibraE project directory
* output files
* cached data
* generated demand data
* generated zone data
* downloaded source data

Use this step when you want a completely fresh rebuild of the model.

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

**Baseline closures** (`baseline_closures` in `config/sim.yaml`): when enabled, road closures from the Police ČR XML feed (fetched by `fetch-data`) are spatially matched to network links. Affected links receive reduced capacity and speed (via a configurable `capacity_reduction_factor`). Original values are stored in `_preclosure_*` columns so they can be restored later by `strip-closures`.

**Configuration**

* `network.normalization_config` in `config/sim.yaml` points to a YAML file (default: `config/network_normalization.yaml`) with:
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

* Uses `assignment.warm_skim_pass` in `config/sim.yaml` for `algorithm`, `max_iter`, and `rgap_target` (defaults are lighter than full `calibrate` / `assign` settings).
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

Performs iterative model calibration.

This phase typically follows the loop:

1. assign traffic
2. compare assigned flows with observed data
3. scale or adjust model components
4. repeat until convergence or stopping criteria are reached

The objective is to reduce the difference between simulated and observed traffic patterns.

Two methods are available:

* **`calibrate`** — iterative demand scaling vs. link counts (pentlogram data)
* **`calibrate-odme`** — origin-destination matrix estimation (ODME) against link counts

**Configuration (excerpt)**

* `calibration.aggregate_corridor` — when `true`, volumes on parallel divided-highway links are summed for comparison to a single count station (recommended for motorways).

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

---

## Command-Line Interface

The runner exposes the following steps:

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
python run.py --config config/sim.yaml <step>
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
| `--config` | `config/sim.yaml` | Config passed to every `run.py` step |
| `--out-root` | `outputs/audit/full_pipeline_audit` | Directory for all audit outputs (under repo root) |
| `--timeout-s` | `1200` | Per-step timeout in seconds (heavy steps use longer overrides inside the script) |
| `--stop-on-fail` | off | Stop after the first step with status `fail` or `blocked` instead of continuing |

**Outputs** (under `--out-root`):

* `pipeline_audit.csv` / `pipeline_audit.json` — per-step exit code, elapsed time, status (`ok` / `warning` / `fail` / `blocked`), missing artifacts, warning tags
* `step_logs/<step>.log` — captured stdout/stderr for each step
* `config_snapshot.yaml` — copy of the config used for the run
* `magic_constants_inventory.csv`, `metrics_consistency_report.md`, `data_driven_refactor_proposal.md`, `unused_policy_keys.md`, `before_after_compare.json` — auxiliary reports used for consistency / documentation reviews

The list of steps and expected artifact paths is maintained in the script; it assumes layout consistent with the default paths in `config/sim.yaml` (for example the AequilibraE project under `project_path`). After a successful `clean`, the audit directory may be removed and is recreated as the run continues.

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
* The primary configuration is in `config/sim.yaml`. Supplementary configs: `config/network_normalization.yaml` (speed/capacity defaults), `config/screenlines.yaml` (calibration/validation count stations), `config/locale.yaml` (localization).
* The script inserts `src/` into `sys.path`, so project modules are loaded directly from the source tree.
* Each `run.py` invocation accepts exactly one step: `python run.py build-demand`. Chain calls in a shell script or run them sequentially.

---

## Entry Point

The workflow is implemented in the main script and dispatches individual steps to dedicated modules such as:

* `sim.network_pipeline` — OSM import
* `sim.network_normalization` — attribute normalization, baseline closures
* `sim.zoning` — TAZ and gateway zone generation
* `sim.fetch_datasets` — external data download (CSD, SLDB, closures)
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
