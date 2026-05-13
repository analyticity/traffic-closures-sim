# Olomouc: GEH, screenlines, external demand, supply, CSD coverage

This note implements the diagnostic checklist for the Olomouc baseline: map
verification of worst Pentlogram matches, corridor aggregation behaviour on
I/46, auto screenline counts vs. usable counts, external demand sensitivity,
link capacities on worst-GEH links, and how to read low CSD coverage in
validation.

## Relation to the last two commits (mainline history)

**029ba2c** — *Divided-highway / one-way cleanup*: `normalize_network_attributes`
zeros reverse ``*_ba`` (or ``*_ab`` for ``direction == -1``) capacity, speed,
and travel time on true one-way links. That removes **phantom** reverse arcs;
it does **not** zero the forward drivable direction. If a link still shows
``total_vehicles_tot == 0`` with large observed traffic, the cause is almost
certainly **no car-feasible path** from the demand / connector topology to that
link (clipping, SCC boundary, or wrong match), not this normalization step.

**72e9b0d** — *Thesis experiments + Olomouc ``sim.yaml`` expansion*: Olomouc
gained an explicit ``external_local`` total, ``through_traffic_scale``,
``I/150`` on the gateway whitelist, and explicit calibration / ODME / CSD-split
keys (many mirror defaults and can be trimmed in YAML). **Screenlines** gained
``max-capacity`` style aggregation for some CSD cuts; validation may omit
screenlines with ``observed_total <= 0``.

**72e9b0d** — *``repair_boundary_scc``*: one-way **trunk** (including
re-classified Czech I-roads modelled as trunk) can be boundary-bidirectionalized
to attach gateways to the largest SCC. **Mainline motorway is still excluded**
from that repair. A clipped **D1** / **D46** *motorway* stub inside the OSM
buffer can therefore remain outside the car-reachable core even though the raw
SQLite graph shows degree-2 endpoints on the link — the disconnect is relative
to **zones / connectors / directed SCC**, not necessarily “missing OSM
geometry”.

## Recommended fix plan (structural, not shortcuts)

1. **Motorway zero-flow warnings (D1 38954, D46 4851, …)**  
   Treat as **topology / scope**: confirm in QGIS + connector reachability
   whether those links sit on a component that receives internal + external
   car demand. Remedies are **wider OSM buffer**, **gateway / supernetwork
   coverage**, or **network import fixes** — not ad-hoc inflation of screenline
   thresholds or arbitrary matcher hacks.

2. **Corridor inflation on one-way trunk (e.g. 7986)**  
   Keep map-based checks (export script). If corridor pairing is wrong, tune
   **buffer / bearing / weights** in ``defaults.calibration.matching`` with
   measured evidence, or exclude the station from hard metrics when excluded.

3. **External seed (147466 / 0.30)**  
   Use the documented sensitivity chain **only** to test a hypothesis after (1)
   is understood; changing seeds without fixing disconnected motorways mostly
   moves stress elsewhere.

4. **Auto screenlines vs. ODME stress**  
   Raising ``csd_min_aadt`` is a legitimate **workload / identifiability** knob
   (fewer CSD radials). It does **not** repair disconnected links.

5. **Low CSD ``coverage_ratio``**  
   Keep as **diagnostic / reporting policy** (section 6); fix by geometry scope
   (CSD clip vs. model extent), not by tightening GEH gates blindly.

6. **Matching (code)** — *bugfix, not a heuristic*: ``match_counts_to_links`` must
   run ``_apply_exclusion_scoring`` on ``_corridor_volume`` with the same
   ``match_quality_min`` as the pre-corridor pass. Passing ``0.0`` there
   effectively **disabled** post-corridor exclusion and let inflated corridor
   totals leak into metrics. This is corrected in ``src/sim/calibration/matching.py``.

## 1) Top GEH and map verification

Source: `outputs/olomouc/baseline/demand/matching_diagnostics.csv` (column
`GEH`, descending).

Programmatic export for QGIS (count points + matched link geometries for the
worst rows):

```bash
python scripts/export_olomouc_matching_geojson.py \
  --csv outputs/olomouc/baseline/demand/matching_diagnostics.csv \
  --db project/olomouc_aeq/project_database.sqlite \
  --out-dir outputs/olomouc/baseline/demand/map_export \
  --top 12
```

Open `matching_counts_top_geh.geojson` together with
`matching_links_top_geh.geojson`. Check: `osm_ref` vs. signage on the map,
`_dist` (metres; large values on motorways often mean a parallel slip / ramp),
`_bearing_diff` where present, and `_excluded` (post-filters — do not treat as
hard calibration failures).

**Priority rows from the latest baseline export**

| objectid | ref   | link_id | `_dist` (m) | `_bearing_diff` | `_corridor_volume` | Notes |
|----------|-------|---------|-------------|-----------------|--------------------|-------|
| 8000047  | D1    | 38954   | ~55         | 0               | 0                  | One-way motorway (`direction=1`, `capacity_ba=0`); **zero** model volume on the assigned link while observation is large → treat as **unreachable / wrong SCC** relative to car OD (see “Relation to the last two commits”), then verify count vs. carriageway in GIS. |
| 8000017  | 46    | 7986    | ~10         | ~26°            | ~38949             | One-way trunk; `_aggregate_corridor_volumes` adds the best opposing carriageway in-buffer — large ratio vs. observation suggests wrong second arc or geometry. |
| 8000003  | 46    | 1144    | ~0.4        | 0               | ~47842             | **Bidirectional** trunk (`direction=0`); corridor helper **does not** sum a twin — `_corridor_volume` is the raw model total on that link (not a doubled corridor). |
| 8000008  | 150A  | 9264    | ~63         | —               | —                  | Check side road vs. main I/150 alignment. |
| 8000001  | D35   | 18181   | ~112        | —               | —                  | Likely under-model / geometry offset on divided motorway. |

Database snapshot (capacities / direction): query `links` for the relevant
`link_id` values in `project/olomouc_aeq/project_database.sqlite`.

## 2) I/46 and `_aggregate_corridor_volumes`

Implementation: `src/sim/calibration/matching.py` — `_aggregate_corridor_volumes`.

- For **one-way** links (`direction != 0`), the Pentlogram observation is
  treated as bidirectional cross-section traffic; the matcher adds the volume
  from a **single** best opposing link of the same type family within the match
  buffer, scored by distance, bearing opposite (~180°), and optional name match.
- For **bidirectional** links (`direction == 0`), `_corridor_volume` stays
  equal to the base period column (no second link is added).

So the two high-GEH stations on ref **46** have different explanations: **7986**
is sensitive to corridor pairing; **1144** is dominated by raw assignment on a
two-way trunk, not corridor doubling.

## 3) Auto screenlines vs. Pentlogram counts

Defaults: `src/sim/defaults.py` → `calibration.auto_screenlines` (`csd_min_aadt`
5000, `gateway_screenlines` and `csd_screenlines` true).

Olomouc overrides: `config/olomouc/sim.yaml` → `calibration.auto_screenlines`
(raised `csd_min_aadt` to reduce CSD-derived radial screenlines while keeping
gateway radials). Manual cuts remain in `config/olomouc/screenlines.yaml`
(currently empty = no manual screenlines).

Loader merge: `src/sim/calibration/screenlines.py` reads
`cfg["calibration"]["auto_screenlines"]`.

After `calibration.auto_screenlines.csd_min_aadt: 10000`, a fresh `validate`
run wrote **30** screenline entries (**27** `auto_gw_*`, **3** `auto_csd_*`:
`150A`, `35`, `44934`) — down from the default-threshold mix that previously
included **7** CSD-derived radials.

**Do not mix two “usable” counts:** the **CSD split calibration** pass in
``validate`` reports about **17** usable stations on the **calibration subset**
of CSD-linked Pentlogram rows (~23 matched before exclusions). The **full**
Pentlogram match diagnostics on all count stations are looser (e.g. **33**
usable of **47** in a typical Olomouc baseline log). Use the **17** figure when
relating screenline count to **ODME / CSD-split** stress only.

## 4) External demand and gateway flows

Config: `config/olomouc/sim.yaml` — `demand.segments.external_local` and
`demand.sldb.external_processing.through_traffic_scale` (e.g. **147466**
external-local trips with **0.18** through scale — gross through-pair sums are
near **73k vehicles/day** in `through_gateway_pairs.parquet`, scaled linearly).

Aggregates: `outputs/olomouc/baseline/demand/od_summary.json` — external
segments and `combined_daily` (values depend on last `build-demand`).

Compare gateway radials in `validation_report.json` → `screenlines` (keys
`auto_gw_*`) to intuition and CSD/gateway observations. If lowering external
seed improves GEH on D35/D46 and on poor count matches, the miss is likely
supply-independent and driven by the external seed.

**Short sensitivity chain** (writes a temp YAML under `outputs/`, then runs
demand → warm skims → distribute → assign → validate):

```bash
chmod +x scripts/run_olomouc_external_sensitivity_chain.sh
# baseline Olomouc uses through_traffic_scale 0.18; try e.g. 0.22 for upward sensitivity
OLOMOUC_SENS_THROUGH_SCALE=0.22 scripts/run_olomouc_external_sensitivity_chain.sh
# or
OLOMOUC_SENS_EXTERNAL_LOCAL=120000 scripts/run_olomouc_external_sensitivity_chain.sh
```

Or emit a one-off config only:

```bash
python scripts/write_olomouc_sensitivity_sim_yaml.py \
  --out outputs/olomouc/sensitivity/sim_custom.yaml \
  --through-scale 0.22 --external-local-trips 120000
SIM_CONFIG=outputs/olomouc/sensitivity/sim_custom.yaml python run.py build-demand
```

## 5) Supply (worst-GEH links and `tune-supply`)

Worst-GEH examples use motorway/trunk with imputed BPR capacities (see
`outputs/olomouc/baseline/demand/supply_audit.json` for imputation share).

SQLite snapshot (`links` table) for priority `link_id` values:

| link_id | type      | dir | osm_ref | capacity_ab / capacity_ba | lanes_ab / lanes_ba |
|---------|-----------|-----|---------|---------------------------|---------------------|
| 38954   | motorway  | 1   | D1      | 4400 / 0                  | 2 / 1               |
| 7986    | trunk     | 1   | 46      | 1800 / 0                  | 1 / 1               |
| 1144    | trunk     | 0   | 46      | 1800 / 1800               | 1 / 1               |
| 9264    | secondary | 0   | 150A    | 1000 / 1000               | 1 / 1               |
| 18181   | motorway  | 1   | D35     | 4400 / 0                  | 2 / 1               |

Optional outer loop: `python run.py tune-supply` with
`calibration.supply_tuning` in defaults — applies class-level speed/capacity
factors from `_base_*` columns in SQLite (`src/sim/calibration/supply_tuning.py`).
Run after `audit-supply` when exploring capacity bias on main roads.

Network priors for I-class motorways: `config/olomouc/network_normalization.yaml`.

## 6) CSD `coverage_ratio` — diagnostic, not a pass/fail benchmark

`validation_report.json` → `csd_link_matching` lists per-road rows with
`coverage_ratio` and `partial_coverage`. When `coverage_ratio` is low (e.g.
I/47 trunk row with **~0.11** in the baseline export: model road length << CSD
section length), the per-road GEH is dominated by **geometric mismatch between
CSD aggregation length and the modelled network extent**, not only assignment
error.

**Policy for reporting:** for `partial_coverage: true` or
`coverage_ratio < ~0.5`, treat per-road GEH as a **diagnostic** (network extent,
buffer, or CSD clip), not a hard pass/fail benchmark. Prefer holdout, clip CSD
to the modelled corridor, or extend the network before using that road class in
strict gates.

Computation context: `src/sim/calibration/validation.py` (CSD–link matching and
summary assembly).
