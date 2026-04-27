#!/usr/bin/env bash
# Full V4 JAX-SAGE end-to-end run (30 K flow + 12 K temp).
#
# Apples-to-apples counterpart of ``run_v4_sage_full.sh``. Outputs go to
# results/partner_v4_jax_sage/ — does not touch baseline or PyTorch-SAGE
# result dirs. Flow trainer writes its own inference JSON at
# ``data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json`` at the
# end of flow training; we back up the partner-precomputed JSON before
# the run and restore it before temp, so temp trains against the same
# flow field the baseline / PyTorch-SAGE temp saw.
#
# Usage:
#   scripts/run_v4_jax_sage_full.sh              # full 30K+12K
#   SEED=2345 scripts/run_v4_jax_sage_full.sh   # alternate seed
#   OUT_SUFFIX=_seed2 SEED=2345 scripts/run_v4_jax_sage_full.sh
#
# Outputs:
#   results/partner_v4_jax_sage${OUT_SUFFIX}/flow/stage_*/flow_network.pkl
#   results/partner_v4_jax_sage${OUT_SUFFIX}/flow/jax_sage_run.log
#   results/partner_v4_jax_sage${OUT_SUFFIX}/temp/temperature_net.pkl
#   results/partner_v4_jax_sage${OUT_SUFFIX}/temp/jax_sage_run.log
#   results/partner_v4_jax_sage${OUT_SUFFIX}/temp/temperature_predictions.h5
#   results/partner_v4_jax_sage${OUT_SUFFIX}/temp/visualizations/temperature.gif

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$REPO_ROOT/env/bin/activate"
fi
echo "[jax-sage-full] using python: $(command -v python)"

SEED="${SEED:-1234}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
OUT_ROOT="$REPO_ROOT/results/partner_v4_jax_sage${OUT_SUFFIX}"
FLOW_DIR="$OUT_ROOT/flow"
TEMP_DIR="$OUT_ROOT/temp"
mkdir -p "$FLOW_DIR" "$TEMP_DIR"

FLOW_LOG="$FLOW_DIR/jax_sage_run.log"
TEMP_LOG="$TEMP_DIR/jax_sage_run.log"

JSON_PATH="./data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json"
JSON_BACKUP="$OUT_ROOT/backup_partner_flow_json.json"

if [[ ! -f "$JSON_BACKUP" ]]; then
    cp "$JSON_PATH" "$JSON_BACKUP"
    echo "[jax-sage-full] backed up partner flow JSON -> $JSON_BACKUP"
fi
md5sum "$JSON_PATH" "$JSON_BACKUP" || true

export PCS_CAD_PATH="./data/partner_v4/designs/Study_Model_B_1st_4p3T.step"
export PCS_GEOM_JSON_PATH="./data/partner_v4/pipe_three_class_fixed.json"

# Don't preallocate 90% of GPU memory — shared tenancy.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"

: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES
echo "[jax-sage-full] using GPU(s): $CUDA_VISIBLE_DEVICES"
echo "[jax-sage-full] SEED=$SEED  OUT_SUFFIX=$OUT_SUFFIX"

# ---------- JAX-SAGE FLOW (30 000 steps) ----------
echo "[jax-sage-full] starting JAX-SAGE flow -> $FLOW_LOG"
python -u src/partner_v4_flow_jax_sage.py \
    hydra.job.chdir=False \
    flow.network_dir="$FLOW_DIR" \
    +flow.training.seed="$SEED" \
    > "$FLOW_LOG" 2>&1

# Restore partner-precomputed flow JSON so JAX-SAGE temp trains against
# the same flow input as the baseline-retrain temp and PyTorch-SAGE temp.
cp "$JSON_BACKUP" "$JSON_PATH"
echo "[jax-sage-full] restored partner flow JSON"
md5sum "$JSON_PATH"

# ---------- JAX-SAGE TEMP (12 000 steps) ----------
echo "[jax-sage-full] starting JAX-SAGE temp -> $TEMP_LOG"
python -u src/partner_v4_temp_jax_sage.py \
    hydra.job.chdir=False \
    temp.network_dir="$TEMP_DIR" \
    temp.training.seed="$SEED" \
    > "$TEMP_LOG" 2>&1

echo "[jax-sage-full] ALL DONE"
