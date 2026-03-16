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
5. **fetch-data** – download and preprocess external datasets  
6. **build-demand** – build the initial OD matrix from commuting data  
7. **distribute** – perform distribution and balancing of demand  
8. **assign** – run traffic assignment on the network  
9. **calibrate** – iteratively calibrate the model against observations  
10. **tune-supply** – optimize supply-side parameters in an outer loop  
11. **validate** – run independent validation using CSD2020 data  
12. **learn-profile** – learn temporal day-type profiles from CSD2020  
13. **serve** – start a read-only REST API for exposing results  

---

## Runner

The pipeline is controlled by the main runner script:

```bash
python run.py --config config/sim.yaml <step>
````

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
python run.py build-demand
python run.py distribute
python run.py assign
python run.py calibrate
python run.py tune-supply
python run.py validate
python run.py learn-profile
python run.py serve
```

In practice, not every run has to execute all steps. Once intermediate artifacts are generated, later steps can usually be rerun independently.

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

---

### 4. `normalize-network`

Cleans and normalizes the imported network.

Typical tasks in this phase include:

* standardizing attribute values
* cleaning inconsistent link metadata
* preparing the network for zoning and assignment
* exporting a normalized version for downstream processing

This step is important because raw OSM data is usually not directly suitable for assignment.

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

---

### 8. `distribute`

Runs demand distribution and balancing.

This step typically includes:

* gravity-model calibration
* iterative proportional fitting (IPF)
* adjustment of the seed OD matrix

The goal is to transform the initial demand into a network-ready OD matrix consistent with constraints and observed structure.

---

### 9. `assign`

Runs traffic assignment.

This step loads the OD matrix onto the network using a selected assignment procedure, such as:

* shortest-path assignment
* equilibrium assignment

The result is an estimate of flows on individual network links.

---

### 10. `calibrate`

Performs iterative model calibration.

This phase typically follows the loop:

1. assign traffic
2. compare assigned flows with observed data
3. scale or adjust model components
4. repeat until convergence or stopping criteria are reached

The objective is to reduce the difference between simulated and observed traffic patterns.

---

### 11. `tune-supply`

Runs outer-loop optimization of supply-side parameters.

While `calibrate` focuses on repeated internal adjustment, this step searches for better supply-related parameters at a higher level. It is intended for more systematic tuning of the network model itself.

---

### 12. `validate`

Runs independent validation using CSD2020.

Unlike calibration, this phase tests the model on separate validation data to assess generalization and robustness. It helps confirm whether the calibrated model behaves reasonably outside the calibration target.

---

### 13. `learn-profile`

Learns temporal day-type factors from CSD2020.

This step extracts temporal behavior patterns that can later be used to derive profiles for different types of days or time periods.

It is especially useful when moving from a static baseline toward more realistic temporal interpretation of demand.

---

### 14. `serve`

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
* `build-demand`
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

## Recommended Execution Strategy

For development, it is usually best to run the workflow incrementally:

* run `clean` only when necessary
* use `check` before rebuilding everything
* rebuild the network only when OSM-related inputs change
* rerun `build-zones` when zoning logic changes
* rerun `build-demand` or `distribute` when demand inputs change
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
