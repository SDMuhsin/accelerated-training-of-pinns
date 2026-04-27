#!/usr/bin/env bash
# V4.1 smoke test: reduced-scale 3D end-to-end run for fast verification.
# Full run takes hours on A40; this should complete in ~20 minutes.
set -euo pipefail

# Always run from project root
cd "$(dirname "$0")/.."

# Activate env
source env/bin/activate

# Use first GPU unless caller overrides
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SMOKE_GEOM="data/partner_v4_1/smoke_geom.json"
SMOKE_FLOW="data/partner_v4_1/smoke_geom_pred_flow_steady.json"

echo "[smoke] building downsampled 3D geometry..."
python scripts/build_v4_1_geometry.py \
    --downsample-xy 4 \
    --z-slices 5 \
    --output "${SMOKE_GEOM}"

echo "[smoke] running 3D flow trainer (200 steps per stage)..."
python src/partner_v4_1_flow.py \
    hydra.job.chdir=False \
    flow.problem.geom_json_path="${SMOKE_GEOM}" \
    flow.network_dir=./results/partner_v4_1_smoke/flow \
    flow.training.k_flow_init=200 \
    flow.training.k_flow_bc=200 \
    "flow.training.k_flow_per_stage=[200,200,200]" \
    "flow.training.nu_schedule=[1.0e-2,5.0e-3,1.0e-3]" \
    flow.training.flow_pde_batch_size=1024 \
    flow.training.flow_init_batch_size=1024 \
    flow.training.flow_soft_init_batch_size=1024 \
    flow.training.flow_wall_batch_size=512 \
    flow.training.flow_bc_batch_size=128 \
    flow.training.flow_pde_points_target=4000 \
    flow.training.wall_guard_points=500 \
    flow.training.wall_guard_separator_points=400 \
    flow.training.geo_guidance_max_points=4000 \
    flow.training.save_network_freq=100000 \
    flow.training.print_stats_freq=25

echo "[smoke] running 3D temperature trainer (200 steps)..."
python src/partner_v4_1_temp.py \
    hydra.job.chdir=False \
    temp.problem.geom_json_path="${SMOKE_GEOM}" \
    temp.problem.flow_json_path="${SMOKE_FLOW}" \
    temp.network_dir=./results/partner_v4_1_smoke/temp \
    temp.training.max_steps=200 \
    temp.training.pde_batch_size=512 \
    temp.training.bc_batch_size=128 \
    temp.training.print_stats_freq=25 \
    temp.inference.batch_size=8192 \
    temp.problem.infer_t_end=10.0 \
    temp.problem.infer_dt=2.0

echo "[smoke] V4.1 smoke test complete. See results/partner_v4_1_smoke/"
