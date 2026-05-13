#!/usr/bin/env bash
# Olomouc: short external-demand sensitivity chain (see docs/olomouc_calibration_diagnostics.md).
# Produces a temporary sim.yaml under outputs/ (gitignored) and runs demand → skims → distribute → assign → validate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

THROUGH="${OLOMOUC_SENS_THROUGH_SCALE:-}"
LOCAL="${OLOMOUC_SENS_EXTERNAL_LOCAL:-}"
if [[ -z "$THROUGH" && -z "$LOCAL" ]]; then
  echo "Set OLOMOUC_SENS_THROUGH_SCALE and/or OLOMOUC_SENS_EXTERNAL_LOCAL (int trips), e.g.:" >&2
  echo "  OLOMOUC_SENS_THROUGH_SCALE=0.22 $0" >&2
  exit 1
fi

TAG="sens"
[[ -n "$THROUGH" ]] && TAG="${TAG}_through_${THROUGH}"
[[ -n "$LOCAL" ]] && TAG="${TAG}_extloc_${LOCAL}"
OUT_YAML="$ROOT/outputs/olomouc/sensitivity/sim_${TAG}.yaml"
mkdir -p "$(dirname "$OUT_YAML")"

ARGS=(python scripts/write_olomouc_sensitivity_sim_yaml.py --out "$OUT_YAML")
[[ -n "$THROUGH" ]] && ARGS+=(--through-scale "$THROUGH")
[[ -n "$LOCAL" ]] && ARGS+=(--external-local-trips "$LOCAL")
"${ARGS[@]}"

export SIM_CONFIG="$OUT_YAML"
python run.py build-demand
python run.py assign-warm-skims
python run.py distribute
python run.py assign
python run.py validate
echo "Done. Config used: $OUT_YAML"
