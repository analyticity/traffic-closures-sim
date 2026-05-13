#!/usr/bin/env bash
# Full pipeline for Most, then full pipeline for Olomouc (calibrate uses
# outputs/_scratch/olomouc_smoke_calib.yaml — one ODME outer iteration).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

STEPS=(
  check
  build-network
  normalize-network
  build-zones
  fetch-data
  build-supernetwork
  build-demand
  assign-warm-skims
  distribute
  assign
  audit-supply
  calibrate
  validate
)

LOG_MOST="${LOG_MOST:-/tmp/pipeline_most_full.log}"
LOG_OLO="${LOG_OLO:-/tmp/pipeline_olomouc_full.log}"

{
  echo "=== MOST full pipeline $(date -Iseconds) ==="
  for s in "${STEPS[@]}"; do
    echo ""
    echo ">>>> MOST step: $s $(date -Iseconds)"
    python run.py --config config/most/sim.yaml "$s"
  done
  echo "=== MOST DONE $(date -Iseconds) ==="
} 2>&1 | tee "$LOG_MOST"

{
  echo "=== OLOMOUC full pipeline (calibrate: 1 ODME iter) $(date -Iseconds) ==="
  for s in "${STEPS[@]}"; do
    echo ""
    echo ">>>> OLOMOUC step: $s $(date -Iseconds)"
    python run.py --config outputs/_scratch/olomouc_smoke_calib.yaml "$s"
  done
  echo "=== OLOMOUC DONE $(date -Iseconds) ==="
} 2>&1 | tee "$LOG_OLO"
