#!/bin/bash
# ============================================================================
# Faithful CAN-PINN Replication - SLURM Submission Script (EXP-11)
# ============================================================================
#
# Submits the faithful Chiu et al. 2022 CAN-PINN replication sweep across all
# three benchmark problems (cavity, Kovasznay, elasticity) and all three
# architectures (mlp, tsa-pinn, pirate-net) used by tab:main_results in
# paper/v2_tetci/results.tex. The "can-pinn-faithful" method dispatches to
# train_can_pinn_faithful{,_kovasznay,_elasticity} in src/lid_benchmark.py
# (Phases 2-5 of the 2026-04-29 replication; Phase 4 verdict in
# llmdocs/trackers/can_pinn_replication_2026-04-29.md §14 confirms direction-
# of-effect parity with the paper).
#
# Tracker entry: paper_rewrite/deferred_experiments.md, EXP-11.
# Body prose does not yet reference this run; integration into Table III is
# a post-sweep decision (REPLACE the Spectral-AD label or AUGMENT with a
# separate column / appendix).
#
# Scope (apples-to-apples with the rest of tab:main_results):
#   - Method:    can-pinn-faithful                              (1)
#   - Problems:  cavity, kovasznay, elasticity                  (3)
#   - Models:    mlp, tsa-pinn, pirate-net                      (3)
#   - Seeds:     0 1 7 23 42                                    (5)
#   - Total:     1 x 3 x 3 x 5 = 45 GPU jobs.
#
# Why no PIELM: PIELM is a separate method with its own dispatch path; it
# does not share the can-pinn-faithful pipeline. The can-pinn sweep is
# strictly the new method on the existing 9 (problem x arch) cells.
#
# Hardware:
#   - SLURM GPU type: H100 80GB MIG 2g.20gb (matches the rest of the
#     paper-quality sweeps: multiseed_20260427, canpinn_hpc_20260428).
#     Edit GPU_TYPE if your cluster uses a different GRES name.
#
# Output CSV:
#   - Default: results/lid_benchmark_results.csv (the main paper CSV;
#     distinguishable by tag = can_pinn_faithful_<date>).
#   - Override with --output-csv.
#
# Tag:
#   - Default: can_pinn_faithful_<date>. Override with --tag.
#
# Aggregation (after the sweep):
#   - Filter rows by tag = can_pinn_faithful_<date>, group by
#     (problem, model), aggregate via scripts/aggregate_l2_and_protocol.py
#     (or the appropriate aggregator) to produce per-cell mean +/- std PDE
#     RMS, train_time_s, ms/epoch.
#   - Compare to the existing Spectral-AD (chebyshev-pinn) rows at tag
#     canpinn_hpc_20260428: same problems / archs / seeds / hardware,
#     different gradient method.
#
# Usage (run on the SLURM login node):
#   ./sbatch/run_can_pinn_faithful.sh
#   ./sbatch/run_can_pinn_faithful.sh --account def-myprof
#   ./sbatch/run_can_pinn_faithful.sh --seeds "0 1 7 23 42" \
#       --tag can_pinn_faithful_2026_05_02
#   ./sbatch/run_can_pinn_faithful.sh \
#       --output-csv results/can_pinn_faithful_only.csv
#
# Local syntax check (no submission):
#   bash -n sbatch/run_can_pinn_faithful.sh
#
# Dev-box note: do NOT run this script on the dev box. It submits SLURM
# jobs and is meant for the HPC login node only. (CONTEXT.md §0/4.1.7:
# never run python on HPC login nodes; never submit sbatch from dev box.)
#
# Concurrent CSV safety: src/lid_benchmark.py:append_csv_row uses fcntl
# locking + fstat-inside-lock header decision (fixed 2026-04-26). The
# script also pre-creates the CSV header below as a belt-and-braces
# precaution against unreliable fcntl on NFS.
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT="def-seokbum"
SEEDS_OVERRIDE=""
TAG_OVERRIDE=""
OUTPUT_CSV_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --seeds)
            # Space-separated list, e.g. --seeds "0 1 7 23 42"
            SEEDS_OVERRIDE="$2"
            shift 2
            ;;
        --tag)
            TAG_OVERRIDE="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--seeds 'S1 S2 ...'] [--tag TAG] [--output-csv PATH]"
            exit 1
            ;;
    esac
done

ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

# The faithful CAN-PINN replication is a single-method sweep. Phases 2-5
# (2026-04-29) added train_can_pinn_faithful, train_can_pinn_faithful_kovasznay,
# and train_can_pinn_faithful_elasticity to src/lid_benchmark.py; the dispatch
# string is "can-pinn-faithful" for all three problems.
methods_per_problem=(
    can-pinn-faithful
)

# All three benchmark problems (matches the cells in tab:main_results).
problems=(
    cavity
    kovasznay
    elasticity
)

# All three architectures from the headline table.
models=(
    mlp
    tsa-pinn
    pirate-net
)

# Random seeds - the standard 5-seed audit set. Matches multiseed_20260427
# and canpinn_hpc_20260428, so this faithful CAN-PINN sweep can be compared
# row-for-row to the Spectral-AD (chebyshev-pinn) rows currently in
# tab:main_results.
seeds=(
    0
    1
    7
    23
    42
)

if [[ -n "$SEEDS_OVERRIDE" ]]; then
    read -ra seeds <<< "$SEEDS_OVERRIDE"
fi

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Matched-protocol Adam settings - identical to multiseed_20260427 and
# canpinn_hpc_20260428 for apples-to-apples comparison against the Spectral-AD
# / SAGE / autodiff rows in tab:main_results. The paper-faithful 200k-iter
# protocol used in scripts/can_pinn_paper_validation.py is for the §3.3
# replication gate (Phase 4 verdict, §6 of the spec); the harness comparison
# here uses 30k iter to preserve apples-to-apples with the rest of the table.
EPOCHS=30000
LR="1e-3"
OPTIMIZER="adam"
DTYPE="fp32"

TECHNIQUE="none"
TAG="${TAG_OVERRIDE:-can_pinn_faithful_$(date +%Y%m%d)}"
TRACK_INTERVAL=100

# Per-problem grid sizes are picked automatically inside src/lid_benchmark.py
# main() (~line 5246-5258) for can-pinn-faithful:
#   cavity:     50x50 (matches the rest of the table; default for cavity)
#   kovasznay:  51x51 (uniform FD stencil; dx=Lx/50=0.03, dy=Ly/50=0.04)
#   elasticity: 51x51 (uniform FD stencil; dx=dy=0.02 on [0,1]^2)
# These are fixed by the method-side defaults; no --grid-size flag is needed.
# Elasticity uses CosineAnnealingLR (T_max=n_epochs, eta_min=1e-5) inside the
# trainer; no harness change needed.

# ============================================================================
# SLURM RESOURCE ALLOCATION
# ============================================================================

# H100 80GB MIG 2g.20gb (matches multiseed_20260427 / canpinn_hpc_20260428).
# Edit GPU_TYPE if your cluster uses a different GRES name.
#
# Time budget: CAN-PINN's per-iter cost is ~1.24x the autodiff baseline on
# cavity (Phase 4 verdict §14: 25 iter/sec vs autodiff's 20 iter/sec, single
# A40 seed). On H100 MIG 2g.20gb the autodiff cavity row is ~22 min (MLP) and
# ~48 min (PirateNet) at 30k iter. CAN-PINN should land in the same ballpark
# (~15-25 min MLP, ~30-50 min PirateNet). The 3 h budget mirrors
# run_lid_benchmark.sh and gives ~3-4x safety margin on the slowest expected
# cell (PirateNet x cavity). Smaller MLP/Kov/Elas cells will finish well
# under the budget.
GPU_TIME="0-03:00:00"
GPU_TYPE="nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_MEM="16000M"
GPU_CPUS=2

# ============================================================================
# SETUP
# ============================================================================

mkdir -p ./logs
mkdir -p ./results

OUTPUT_CSV="${OUTPUT_CSV_OVERRIDE:-results/lid_benchmark_results.csv}"

# Pre-create the output CSV with its header row (same scheme as
# run_lid_benchmark.sh - keeps parallel jobs from racing on first-write).
# The header MUST stay in lockstep with CSV_COLUMNS in src/lid_benchmark.py
# (~line 2436). We hardcode it in bash so the sbatch submitter does not
# need to import the project (numpy/torch on the login node is unavailable).
CSV_HEADER='timestamp,problem,method,model,optimizer,lr,epochs,seed,grid_size,technique,tag,train_time_s,train_time_min,peak_gpu_memory_mb,gpu_memory_reserved_mb,ms_per_epoch,n_params,pde_rms,continuity_rms,momentum_rms,final_loss,best_epoch,status,device,gpu_name,pytorch_version'

if [[ ! -f "$OUTPUT_CSV" ]] || [[ ! -s "$OUTPUT_CSV" ]]; then
    echo "Pre-creating $OUTPUT_CSV with header row..."
    mkdir -p "$(dirname "$OUTPUT_CSV")"
    if ! printf '%s\n' "$CSV_HEADER" > "$OUTPUT_CSV"; then
        echo "ERROR: failed to write $OUTPUT_CSV"
        exit 1
    fi
    echo "Wrote header ($(awk -F',' '{print NF; exit}' "$OUTPUT_CSV") cols) to $OUTPUT_CSV"
else
    echo "Reusing existing $OUTPUT_CSV ($(wc -l < "$OUTPUT_CSV") lines)."
fi

echo "========================================================================"
echo "EXP-11: Faithful CAN-PINN Replication HPC Sweep"
echo "========================================================================"
echo ""
echo "Method:     ${methods_per_problem[*]}"
echo "Problems:   ${problems[*]}"
echo "Models:     ${models[*]}"
echo "Seeds:      ${seeds[*]} (${#seeds[@]} seeds)"
echo "Epochs:     $EPOCHS  ($OPTIMIZER, lr=$LR, $DTYPE)"
echo "Tag:        $TAG"
echo "Output:     $OUTPUT_CSV"
echo "GPU type:   $GPU_TYPE"
echo "Time/job:   $GPU_TIME"
if [[ -n "$ACCOUNT" ]]; then
    echo "Account:    $ACCOUNT"
fi
echo ""

job_count=0

# ============================================================================
# JOB SUBMISSION LOOP
# ============================================================================

for problem in "${problems[@]}"; do
    case "$problem" in
        cavity)     prefix="cav" ;;
        kovasznay)  prefix="kov" ;;
        elasticity) prefix="ela" ;;
        *)
            echo "ERROR: unknown problem '$problem'"
            exit 1
            ;;
    esac

    echo "=============================================="
    echo "Problem: $problem"
    echo "=============================================="

    for method in "${methods_per_problem[@]}"; do
        for model in "${models[@]}"; do
            for seed in "${seeds[@]}"; do
                job_name="${prefix}_canfaith_${model}_s${seed}"
                log_file="./logs/${job_name}"

                echo "Submitting: $job_name"
                sbatch \
                    $ACCOUNT_FLAG \
                    --nodes=1 \
                    --ntasks-per-node=1 \
                    --cpus-per-task=$GPU_CPUS \
                    --gpus=$GPU_TYPE \
                    --mem=$GPU_MEM \
                    --time=$GPU_TIME \
                    --job-name=$job_name \
                    --output=${log_file}-%N-%j.out \
                    --error=${log_file}-%N-%j.err \
                    --wrap="
                        module load scipy-stack cuda cudnn
                        module load arrow
                        source ./env/bin/activate
                        echo '========================================'
                        echo 'Job: $job_name'
                        echo 'Problem: $problem'
                        echo 'Method: $method'
                        echo 'Model: $model'
                        echo 'Seed: $seed'
                        echo 'Started: '\$(date)
                        echo '========================================'
                        nvidia-smi
                        python3 -u src/lid_benchmark.py \
                            --problem=$problem \
                            --method=$method \
                            --model=$model \
                            --optimizer=$OPTIMIZER \
                            --lr=$LR \
                            --epochs=$EPOCHS \
                            --dtype=$DTYPE \
                            --seed=$seed \
                            --technique=$TECHNIQUE \
                            --output-csv=$OUTPUT_CSV \
                            --tag=$TAG \
                            --track \
                            --track-interval=$TRACK_INTERVAL
                        echo '========================================'
                        echo 'Finished: '\$(date)
                        echo '========================================'
                    "
                ((job_count++))
            done
        done
    done
done

# ============================================================================
# SUMMARY
# ============================================================================

n_per_problem=$((${#methods_per_problem[@]} * ${#models[@]} * ${#seeds[@]}))
n_total=$(( ${#problems[@]} * n_per_problem ))

echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED"
echo "========================================================================"
echo ""
echo "Per problem: ${#methods_per_problem[@]} method x ${#models[@]} models x ${#seeds[@]} seeds = $n_per_problem jobs"
echo "Problems:    ${#problems[@]} (${problems[*]})"
echo "TOTAL JOBS:  $job_count   (expected $n_total)"
echo ""
echo "Tag:      $TAG"
echo "Results:  ./$OUTPUT_CSV"
echo "Logs:     ./logs/cav_canfaith_*  ./logs/kov_canfaith_*  ./logs/ela_canfaith_*"
echo "========================================================================"
