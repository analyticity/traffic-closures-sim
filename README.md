# Simulation Pipeline

This project provides a step-by-step simulation pipeline for building, calibrating, validating, and serving a transport model.

The pipeline is executed through a single runner script, where each step represents one logical phase of the workflow. The individual phases are designed so they can be run separately, which makes debugging, iteration, and experimentation easier.

---

## Overview

The pipeline supports the following workflow:

0. **clean** – remove generated data and start from a clean state  
1. **check** – verify that the AequilibraE project is correctly bootstrapped  
2. **build-network** – import the road network from OpenStreetMap  
3. **normalize-network** – clean and normalize network attributes  
4. **build-zones** – create TAZ zones and centroid connectors  
5. **fetch-data** – download and preprocess external datasets (optional; `datasets.enabled`)  
6. **build-supernetwork** – coarse national network for external / through traffic at gateways  
7. **build-demand** – build the seed OD matrix (commuting, gateways, synthetic segments)  
8. **assign-warm-skims** – optional short assignment that always saves `skims.aem` for trip distribution  
9. **distribute** – gravity calibration + IPF on the seed matrix (network skims or Euclidean; see below)  
10. **assign** – full traffic assignment on the detailed network  
11. **calibrate** – iterative demand scaling vs. link counts (pentlogram)  
12. **tune-supply** – optional outer loop on supply-side factors after demand calibration  
13. **validate** – independent checks vs. CSD (not used in calibration)  
14. **learn-profile** – temporal day-type factors from CSD  
15. **serve** – read-only REST API for results  

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
python run.py normalize-network
python run.py build-zones
python run.py fetch-data
python run.py build-supernetwork
python run.py build-demand
python run.py assign-warm-skims
python run.py distribute
python run.py assign
python run.py calibrate
python run.py tune-supply
python run.py validate
python run.py learn-profile
python run.py serve
```

In practice, not every run has to execute all steps. Once intermediate artifacts are generated, later steps can usually be rerun independently.

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
* preparing the network for zoning and assignment
* exporting a normalized version for downstream processing

This step is important because raw OSM data is usually not directly suitable for assignment.

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

Builds transport analysis zones (TAZ) and centroid connectors.

This phase creates the zoning system used for demand modeling and connects zone centroids to the road network so OD flows can enter and leave the network during assignment.

---

### 6. `fetch-data`

Downloads and preprocesses external datasets required by the model.

These datasets may include demand, count, validation, or temporal reference inputs. The step prepares them into a consistent internal format for the following phases.

---

### 7. `build-demand`

Builds the initial OD matrix from SLDB commuting data.

This produces the base origin-destination demand representation, which acts as the seed for later distribution and calibration.

When `demand.sldb.external_processing.enabled` is true, the runner checks for supernetwork outputs (gateway lookup parquet, through pairs if configured, and `supernetwork_summary.json`) before building demand. With external processing off, only zones and centroid mapping from `build-zones` are required.

---

### 8. `assign-warm-skims`

Optional shorter traffic assignment whose main purpose is to write **`skims.aem`** under `demand.output_dir` for use as impedance in `distribute`.

* Uses `assignment.warm_skim_pass` in `config/sim.yaml` for `algorithm`, `max_iter`, and `rgap_target` (defaults are lighter than full `calibrate` / `assign` settings).
* Always saves skims (`save_skims` is forced on for this step).
* Writes `assignment_results.parquet` like `assign`; treat it as an intermediate artifact if you run a full `assign` afterward.

---

### 9. `distribute`

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

### 10. `assign`

Runs traffic assignment.

This step loads the OD matrix onto the network using a selected assignment procedure, such as:

* shortest-path assignment
* equilibrium assignment

The result is an estimate of flows on individual network links.

---

### 11. `calibrate`

Performs iterative model calibration.

This phase typically follows the loop:

1. assign traffic
2. compare assigned flows with observed data
3. scale or adjust model components
4. repeat until convergence or stopping criteria are reached

The objective is to reduce the difference between simulated and observed traffic patterns.

**Configuration (excerpt)**

* `calibration.aggregate_corridor` — when `true`, volumes on parallel divided-highway links are summed for comparison to a single count station (recommended for motorways).

---

### 12. `tune-supply`

Runs outer-loop optimization of supply-side parameters.

While `calibrate` focuses on repeated internal adjustment, this step searches for better supply-related parameters at a higher level. Enable with `calibration.supply_tuning.enabled`. The composite objective weights GEH, screenline fit, and journey-time checks (fixed weights in code).

---

### 13. `validate`

Runs independent validation using CSD2020.

Unlike calibration, this phase tests the model on separate validation data to assess generalization and robustness. It helps confirm whether the calibrated model behaves reasonably outside the calibration target.

---

### 14. `learn-profile`

Learns temporal day-type factors from CSD2020.

This step extracts temporal behavior patterns that can later be used to derive profiles for different types of days or time periods.

It is especially useful when moving from a static baseline toward more realistic temporal interpretation of demand.

---

### 15. `serve`

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
* `tune-supply`
* `validate`
* `learn-profile`
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
* rerun `build-zones` when zoning logic changes
* rerun `build-demand` or `distribute` when demand inputs change
* use `assign-warm-skims` before `distribute` when you want IPF/gravity driven by network skims instead of Euclidean distance
* rerun `assign`, `calibrate`, and `validate` frequently during model tuning
* use `serve` only after the required outputs have been prepared

This keeps iteration fast and avoids recomputing expensive earlier phases unnecessarily.

---

## Notes

* Many steps depend on outputs generated by previous steps.
* The pipeline is designed to support both full rebuilds and partial reruns.
* The configuration is centralized in `config/sim.yaml`.
* The script inserts `src/` into `sys.path`, so project modules are loaded directly from the source tree.

---

## Entry Point

The workflow is implemented in the main script and dispatches individual steps to dedicated modules such as:

* `sim.network_pipeline`
* `sim.network_normalization`
* `sim.zoning`
* `sim.fetch_datasets`
* `sim.demand`
* `sim.distribution`
* `sim.assignment`
* `sim.calibration`
* `sim.temporal`
* `sim.api`

This keeps the runner lightweight while the domain logic remains separated into dedicated components.

Batch validation of the same steps is described under **Full pipeline audit** above ([`scripts/full_pipeline_audit.py`](scripts/full_pipeline_audit.py)).
