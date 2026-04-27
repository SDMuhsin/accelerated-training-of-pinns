#!/usr/bin/env bash
# Parity smoke for V4 baseline vs rebuilt SAGE flow trainer.
#
# Runs the baseline and SAGE flow trainers with an identical per-stage
# step-count override, capturing per-stage step-0 losses for comparison.
# Preserves (and restores) the partner-precomputed flow JSON each time.
#
# Usage:
#   scripts/run_v4_parity_smoke.sh <steps_per_stage> <baseline_log> <sage_log>
# Example:
#   scripts/run_v4_parity_smoke.sh 2 /tmp/baseline_smoke.log /tmp/sage_smoke.log
#
# Output dirs:
#   /tmp/parity_baseline/flow/, /tmp/parity_sage/flow/
# (temporary; overwritten on each invocation)

set -euo pipefail

STEPS="${1:-2}"
BASE_LOG="${2:-/tmp/baseline_smoke.log}"
SAGE_LOG="${3:-/tmp/sage_smoke.log}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Self-contained env activation so ``nohup bash scripts/...sh`` works
# without the caller having sourced the venv first.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env/bin/activate"
fi

JSON_PATH="./data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json"
JSON_BACKUP="/tmp/parity_backup/flow_json_backup.json"

mkdir -p /tmp/parity_backup /tmp/parity_baseline /tmp/parity_sage

# Back up the partner-precomputed flow JSON if we haven't yet.
if [[ ! -f "$JSON_BACKUP" ]]; then
    cp "$JSON_PATH" "$JSON_BACKUP"
fi

export PCS_CAD_PATH="./data/partner_v4/designs/Study_Model_B_1st_4p3T.step"
export PCS_GEOM_JSON_PATH="./data/partner_v4/pipe_three_class_fixed.json"

OVERRIDES=(
    hydra.job.chdir=False
    flow.training.k_flow_init="$STEPS"
    flow.training.k_flow_bc="$STEPS"
    "flow.training.k_flow_per_stage=[$STEPS,$STEPS,$STEPS]"
)

echo "[parity] baseline run -> $BASE_LOG"
python -u src/partner_v4_flow.py \
    flow.network_dir=/tmp/parity_baseline/flow \
    "${OVERRIDES[@]}" \
    > "$BASE_LOG" 2>&1
cp "$JSON_BACKUP" "$JSON_PATH"

echo "[parity] sage run -> $SAGE_LOG"
python -u src/partner_v4_flow_sage.py \
    flow.network_dir=/tmp/parity_sage/flow \
    "${OVERRIDES[@]}" \
    > "$SAGE_LOG" 2>&1
cp "$JSON_BACKUP" "$JSON_PATH"

echo "[parity] done. compare via scripts/compare_smoke_step1.py"
