#!/usr/bin/env bash
# V4.1 full run: 3D flow (25K steps) + 3D temp (12K steps).
# Expected runtime: several hours on A40.
set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Build 3D geometry if missing
if [[ ! -f data/partner_v4_1/pipe_three_class_3d.json ]]; then
    echo "[full] 3D geometry not found, building..."
    python scripts/build_v4_1_geometry.py
fi

mkdir -p results/partner_v4_1/flow results/partner_v4_1/temp

echo "[full] running V4.1 E2E (3D)..."
python src/partner_v4_1_e2e.py \
    --delay-seconds 5 \
    "$@"

echo "[full] V4.1 run complete."
