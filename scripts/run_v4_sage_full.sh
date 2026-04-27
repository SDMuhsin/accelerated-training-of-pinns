#!/usr/bin/env bash
# Full V4 SAGE end-to-end run (30 K flow + 12 K temp).
#
# This is the production SAGE run after all Level-2 / step-1 / 100-step
# parity checks have passed. Outputs go to results/partner_v4_sage_v2/.
#
# Flow trains against the same pipe_three_class_fixed.json geometry as
# the baseline and, after flow training, writes its own inference JSON
# at data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json
# (overwriting whatever was there). We restore the partner-precomputed
# reference JSON after flow training so the temp trainer sees the same
# flow input as results/partner_v4_baseline_retrain/temp was trained on.
# That keeps the baseline-vs-SAGE temp comparison apples-to-apples at
# the temp stage: same flow field in, only the temp gradient engine
# differs.
#
# Usage:
#   scripts/run_v4_sage_full.sh
#
# Outputs:
#   /workspace/dt-pinn/results/partner_v4_sage_v2/flow/stage_*/flow_network.0.pth
#   /workspace/dt-pinn/results/partner_v4_sage_v2/flow/full_run.log
#   /workspace/dt-pinn/results/partner_v4_sage_v2/temp/temperature_net.pt
#   /workspace/dt-pinn/results/partner_v4_sage_v2/temp/sage_run.log
#   /workspace/dt-pinn/results/partner_v4_sage_v2/temp/temperature_predictions.h5
#   /workspace/dt-pinn/results/partner_v4_sage_v2/temp/visualizations/temperature.gif

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Self-contained env activation so ``nohup bash scripts/...sh`` works
# without the caller having already sourced the venv.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/env/bin/activate"
fi
echo "[sage-full] using python: $(command -v python)"

OUT_ROOT="$REPO_ROOT/results/partner_v4_sage_v2"
FLOW_DIR="$OUT_ROOT/flow"
TEMP_DIR="$OUT_ROOT/temp"
mkdir -p "$FLOW_DIR" "$TEMP_DIR"

FLOW_LOG="$FLOW_DIR/sage_run.log"
TEMP_LOG="$TEMP_DIR/sage_run.log"

JSON_PATH="./data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json"
JSON_BACKUP="$OUT_ROOT/backup_partner_flow_json.json"

# Back up the partner-precomputed flow JSON (md5 5b04e983). The SAGE
# flow trainer will overwrite this at its inference step; we restore
# before running SAGE temp so temp sees the same flow input baseline
# temp saw.
if [[ ! -f "$JSON_BACKUP" ]]; then
    cp "$JSON_PATH" "$JSON_BACKUP"
    echo "[sage-full] backed up partner flow JSON -> $JSON_BACKUP"
fi
md5sum "$JSON_PATH" "$JSON_BACKUP" || true

export PCS_CAD_PATH="./data/partner_v4/designs/Study_Model_B_1st_4p3T.step"
export PCS_GEOM_JSON_PATH="./data/partner_v4/pipe_three_class_fixed.json"

# Pin GPU per caller's CUDA_VISIBLE_DEVICES, or default to GPU 1 which
# has more free memory on this box.
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES
echo "[sage-full] using GPU(s): $CUDA_VISIBLE_DEVICES"

# ---------- SAGE FLOW (30 000 steps: 3 K + 2 K + 5 K + 5 K + 10 K) ----------
echo "[sage-full] starting SAGE flow -> $FLOW_LOG"
python -u src/partner_v4_flow_sage.py \
    hydra.job.chdir=False \
    flow.network_dir="$FLOW_DIR" \
    > "$FLOW_LOG" 2>&1

# Restore partner-precomputed flow JSON so SAGE temp trains on the
# same flow input as the baseline-retrain temp did (apples-to-apples
# AT THE TEMP STAGE).
cp "$JSON_BACKUP" "$JSON_PATH"
echo "[sage-full] restored partner flow JSON"
md5sum "$JSON_PATH"

# ---------- SAGE TEMP (12 000 steps) ----------
echo "[sage-full] starting SAGE temp -> $TEMP_LOG"
python -u src/partner_v4_temp_sage.py \
    hydra.job.chdir=False \
    temp.network_dir="$TEMP_DIR" \
    > "$TEMP_LOG" 2>&1

echo "[sage-full] ALL DONE"
