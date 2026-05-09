#!/bin/bash
# ============================================================================
# Channel-Flow PINN Benchmark - SLURM Submission Script
# ============================================================================
#
# 2026-05-09 ACTIVE CONFIGURATION: Option (D-prime) — full-budget Phase B run
# at Re_tau=180 with PirateNet + SOAP + causal-mean (eps=1.0, chunks=16).
# Total budget = 32 windows x 6250 epochs/window = 200,000 epochs (matches
# Wang/Perdikaris 2025 arXiv:2507.08972 budget for 3D Kolmogorov turbulence).
# This is the full Wang/Perdikaris recipe; the 32k-epoch dev-box smoke
# (tag c1_re180_piratenet_soap_causal_2026_05_09) descended cleanly to
# pde=1.81e-3 but produced an over-regularised laminar-like solution
# because 32k = 16% of Wang's published budget. (D-prime) extends to the
# full 200k budget on HPC; if turbulence still fails to develop here, the
# channel-flow direction itself needs reopening (option G in the tracker).
#
# Submits jobs to SLURM for the 3D turbulent channel flow benchmark
# (src/channel_flow_benchmark.py — plain PyTorch, no PhysicsNeMo dep).
#
# Channel-flow benchmark scope:
#   problem  : 3D incompressible Navier-Stokes, channel box
#              Lx x 2h x Lz, periodic in (x,z), no-slip walls at y=+/-h
#   reference: Moser-Kim-Mansour 1999 (chan180/chan590, public DNS).
#              Files at data/mkm99/chan{180,590}.{means,reystress}.
#   methods  : autodiff   (only one currently implemented; C3 will add
#                          ropinn / chebyshev-pinn / sk-pinn /
#                          can-pinn-faithful / dtpinn / sage as the
#                          methods sweep is ported)
#   models   : mlp, pirate-net   (tsa-pinn 2D-only at present)
#
# Output CSV columns (53):
#   timestamp, method, model, Re_tau, Lx, Lz, num_windows, window_size,
#   epochs_per_window, total_epochs, lr, lr_decay_rate, lr_decay_steps,
#   ic_weight, y_stretch, ic_perturb_amp, batch_interior, batch_ic,
#   hidden_dim, num_layers, n_params, seed, tag, wall_time_s, wall_time_min,
#   ms_per_epoch, peak_gpu_memory_mb, final_loss, final_pde_loss,
#   final_ic_loss, nan_windows, y_plus_grid, mean_u_profile, urms_profile,
#   vrms_profile, wrms_profile, uv_profile, log_law_match_log_layer,
#   linear_match_sublayer, u_mkm99_log, urms_mkm99_log, vrms_mkm99_log,
#   wrms_mkm99_log, uv_mkm99_log, optimizer, causal_eps, causal_chunks,
#   soap_betas, soap_precondition_frequency, status, device, gpu_name,
#   pytorch_version
#
# Usage:
#   ./sbatch/run_channel_flow.sh
#   ./sbatch/run_channel_flow.sh --account def-myprof
#   ./sbatch/run_channel_flow.sh --seeds "0 1 7 23 42" --tag chan_dprime_2026_05_09
#   ./sbatch/run_channel_flow.sh --output-csv results/chan_dprime.csv --seeds "0"
#
# Concurrent CSV safety:
#   src/channel_flow_benchmark.py:append_csv_row uses fcntl.LOCK_EX +
#   fstat-inside-lock header decision (mirrors src/lid_benchmark.py and
#   src/taylor_green_benchmark.py) so parallel jobs writing to the same
#   --output-csv won't double-write headers or interleave rows. This script
#   also pre-creates the CSV with a header before submission as a belt-and-
#   braces precaution against filesystems where fcntl is unreliable (NFS).
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

# Build account flag for sbatch (empty string if not specified)
ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# ============================================================================
# CONFIGURATION - Modify these arrays to control what runs
# ============================================================================

# Channel-flow methods. Currently only `autodiff` is implemented in
# src/channel_flow_benchmark.py (the --method choice list there is
# {"autodiff"}). Phase C3 will add ropinn / chebyshev-pinn / sk-pinn /
# can-pinn-faithful / dtpinn / sage; this array is the extension point for
# that future sweep. Listing a method that isn't a valid choice will fail
# at python parse time, NOT at sbatch submission, so test the python CLI
# locally before adding a new method here.
channel_methods=(
    autodiff
    # ropinn               # C3
    # chebyshev-pinn       # C3 (hybrid 2D-FFT-xz x 1D-Cheb-y)
    # sk-pinn              # C3
    # can-pinn-faithful    # C3
    # dtpinn               # C3
    # sage                 # C3
)

# Network architectures. mlp + pirate-net are ported and bit-equiv-locked
# in src/channel_flow_benchmark.py; tsa-pinn is 2D-only at present.
channel_models=(
    mlp
    pirate-net
)

# Random seeds. Default = 1 seed for the (D-prime) headline run (~12-21h
# per job on H100 MIG); the multiseed_20260427-style 5-seed sweep can
# follow once the headline lands turbulence and we know SAGE wins.
# Override with --seeds "0 1 7 23 42" for a multi-seed sweep.
channel_seeds=(
    0
    # 1
    # 7
    # 23
    # 42
)

# Apply --seeds CLI override (space-separated string -> bash array)
if [[ -n "$SEEDS_OVERRIDE" ]]; then
    read -ra channel_seeds <<< "$SEEDS_OVERRIDE"
fi

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Channel-flow (D-prime) Phase B paperscale recipe — full Wang/Perdikaris
# 2025 (arXiv:2507.08972) budget at Re_tau=180:
#
#   total epochs    = num_windows x epochs_per_window = 32 x 6250 = 200,000
#                     (Wang/Perdikaris '25 used 32 windows; same total budget
#                     they ran for 3D Kolmogorov turbulence)
#   total time      = num_windows x window_size = 32 x 1.0 = 32 wall units
#   architecture    = PirateNet 3 layers x 256 hidden, alpha-init = 0.0
#                     (physics-informed identity init, Wang et al. 2024)
#   loss            = causal-mean (Wang et al. 2022 form, eps=1.0, 16 chunks)
#                     -- mean-per-chunk + strict-prefix cumsum + exp(-eps*ps).
#                     NOT the PhysicsNeMo CausalLossNorm sum-per-chunk + w/w[0]
#                     form, which fp32-NaNs at PirateNet's alpha=0 init
#                     (see src/channel_flow_benchmark.py train loop comment).
#   optimizer       = SOAP (Vyas et al. 2024); src/soap.py vendored from
#                     github.com/nikhilvyas/SOAP, Apache-2.0.
#   reference data  = data/mkm99/chan180.{means,reystress}
#                     (Re_tau_actual = 178.12; loader at
#                      src/channel_flow_benchmark.py:load_mkm99 dispatches
#                      on closest nominal Re_tau in {180,395,550,590,1000}).
#
# Bit-equivalence guard: --optimizer=adam --causal-eps=0 with --re-tau=590,
# defaults, --num-windows=16 --epochs-per-window=2000, MUST byte-reproduce
# the locked anchor mean_u[y+=262.21]=27.75303. Verified post-port
# 2026-05-09 (tag c1_mlp_re590_bitequiv_postB_2026_05_09).
RE_TAU=180
NUM_WINDOWS=32
WINDOW_SIZE=1.0
EPOCHS_PER_WINDOW=6250
BATCH_INTERIOR=4096
BATCH_IC=4096
EVAL_NX=32
EVAL_NY=64
EVAL_NZ=32
EVAL_TIMES_PER_WINDOW=4
LR="1e-3"
LR_DECAY_RATE=0.95
LR_DECAY_STEPS=3000
IC_WEIGHT=100
Y_STRETCH=2.5
IC_PERTURB_AMP=0.1
HIDDEN_DIM=256
NUM_LAYERS=6
PIRATE_NUM_LAYERS=3
PIRATE_HIDDEN_DIM=256
PIRATE_NONLINEARITY=0.0
OPTIMIZER=soap
CAUSAL_EPS=1.0
CAUSAL_CHUNKS=16
SOAP_BETAS="0.9,0.999"
SOAP_SHAMPOO_BETA=-1.0
SOAP_EPS=1e-8
SOAP_WEIGHT_DECAY=0.0
SOAP_PRECONDITION_FREQUENCY=10

# Tag default. dprime_<date> distinguishes this run from the dev-box
# 32k-epoch smoke (c1_re180_piratenet_soap_causal_2026_05_09) and from
# the bit-equivalence verification rows. Override with --tag.
TAG="${TAG_OVERRIDE:-channel_dprime_re${RE_TAU}_$(date +%Y%m%d)}"

# Output CSV (shared across all jobs in this sweep, concurrent-safe via
# fcntl locking). Different schema from results/lid_benchmark_results.csv
# and results/tgv_phase3_results.csv, so kept in its own file.
OUTPUT_CSV="${OUTPUT_CSV_OVERRIDE:-results/channel_flow_results.csv}"

# ============================================================================
# SLURM RESOURCE ALLOCATION
# ============================================================================

# Channel-flow D-prime jobs (GPU). Wall time projection:
#   2026-05-09 dev-box smoke (Re_tau=180, PirateNet+SOAP+causal, 32k ep):
#       201.08 min wall on heavily-contended A40 (377 ms/epoch).
#   200k epochs at the same rate = ~21 h on contended A40.
#   Free A40 (no contention) would be ~12-14 h.
#   H100 MIG 2g.20gb is ~1.5-1.8x faster than A40 (per the run_lid_benchmark.sh
#   GPU_TIME comment), so paperscale on H100 ~= 7-10 h.
#   Allocate 1 day with a generous safety margin -> 2 days. SLURM defaults
#   often cap at 7 days; check your cluster.
GPU_TIME="2-00:00:00"
GPU_TYPE="nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_MEM="24000M"
GPU_CPUS=4

# ============================================================================
# SETUP
# ============================================================================

mkdir -p ./logs
mkdir -p ./results

# ----------------------------------------------------------------------------
# Pre-create the output CSV with its 53-col header row, so parallel jobs
# racing on a fresh file never both decide to write a header. The Python
# writer (src/channel_flow_benchmark.py:append_csv_row) checks file size
# inside the fcntl lock and skips the header when the file is non-empty.
#
# The header below MUST stay in lockstep with CHANNEL_FLOW_CSV_COLUMNS in
# src/channel_flow_benchmark.py (~line 880). We hardcode it in bash so the
# sbatch submitter does not need to import the project (numpy/torch on the
# login node is unavailable on standard HPC setups).
# ----------------------------------------------------------------------------
CSV_HEADER='timestamp,method,model,Re_tau,Lx,Lz,num_windows,window_size,epochs_per_window,total_epochs,lr,lr_decay_rate,lr_decay_steps,ic_weight,y_stretch,ic_perturb_amp,batch_interior,batch_ic,hidden_dim,num_layers,n_params,seed,tag,wall_time_s,wall_time_min,ms_per_epoch,peak_gpu_memory_mb,final_loss,final_pde_loss,final_ic_loss,nan_windows,y_plus_grid,mean_u_profile,urms_profile,vrms_profile,wrms_profile,uv_profile,log_law_match_log_layer,linear_match_sublayer,u_mkm99_log,urms_mkm99_log,vrms_mkm99_log,wrms_mkm99_log,uv_mkm99_log,optimizer,causal_eps,causal_chunks,soap_betas,soap_precondition_frequency,status,device,gpu_name,pytorch_version'

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
echo "CHANNEL-FLOW PINN BENCHMARK: SLURM Job Submission"
echo "========================================================================"
echo ""
echo "Mode:              D-prime (full 200k-epoch Phase B paperscale)"
echo "Methods:           ${channel_methods[*]}"
echo "Models:            ${channel_models[*]}"
echo "Seeds:             ${channel_seeds[*]} (${#channel_seeds[@]} seeds)"
echo "Re_tau:            $RE_TAU"
echo "Windows x epochs:  $NUM_WINDOWS x $EPOCHS_PER_WINDOW = $((NUM_WINDOWS * EPOCHS_PER_WINDOW)) total epochs"
echo "Window size:       $WINDOW_SIZE wall units (total simulated time = $((NUM_WINDOWS * 1)) wall units)"
echo "Network:           PirateNet ${PIRATE_NUM_LAYERS} blocks x ${PIRATE_HIDDEN_DIM} hidden, alpha-init=$PIRATE_NONLINEARITY"
echo "Loss:              causal-mean, eps=$CAUSAL_EPS, chunks=$CAUSAL_CHUNKS"
echo "Optimizer:         $OPTIMIZER (betas=$SOAP_BETAS, precond freq=$SOAP_PRECONDITION_FREQUENCY)"
echo "lr:                $LR  (decay $LR_DECAY_RATE every $LR_DECAY_STEPS steps)"
echo "Batch:             interior=$BATCH_INTERIOR, ic=$BATCH_IC"
echo "Eval grid:         $EVAL_NX x $EVAL_NY (CGL) x $EVAL_NZ, $EVAL_TIMES_PER_WINDOW t-samples per window"
echo "Tag:               $TAG"
echo "Output:            $OUTPUT_CSV"
if [[ -n "$ACCOUNT" ]]; then
    echo "Account:           $ACCOUNT"
fi
echo ""

job_count=0

# ============================================================================
# SECTION 1: CHANNEL FLOW - GRADIENT-BASED METHODS (GPU)
# ============================================================================
# Single section because channel flow is one problem (Re_tau parametrised).
# Per-(method, model, seed) job; each job is one full training run on a
# dedicated GPU and writes a single row to $OUTPUT_CSV.

echo "=============================================="
echo "Section 1: Channel Flow - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${channel_methods[@]}"; do
    for model in "${channel_models[@]}"; do
        for seed in "${channel_seeds[@]}"; do
            job_name="chan_${method}_${model}_s${seed}"
            log_file="./logs/${job_name}"

            # PirateNet flags only matter when --model=pirate-net; passing
            # them on the MLP path is harmless (parser accepts them, model
            # constructor ignores). Mirrors the TGV submission pattern in
            # run_lid_benchmark.sh:798-828.
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
                    echo 'Problem: channel flow (3D, Re_tau=$RE_TAU)'
                    echo 'Method: $method'
                    echo 'Model: $model'
                    echo 'Seed: $seed'
                    echo 'Tag: $TAG'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    python3 -u src/channel_flow_benchmark.py \
                        --method=$method \
                        --model=$model \
                        --re-tau=$RE_TAU \
                        --num-windows=$NUM_WINDOWS \
                        --window-size=$WINDOW_SIZE \
                        --epochs-per-window=$EPOCHS_PER_WINDOW \
                        --batch-interior=$BATCH_INTERIOR \
                        --batch-ic=$BATCH_IC \
                        --eval-nx=$EVAL_NX \
                        --eval-ny=$EVAL_NY \
                        --eval-nz=$EVAL_NZ \
                        --eval-times-per-window=$EVAL_TIMES_PER_WINDOW \
                        --lr=$LR \
                        --lr-decay-rate=$LR_DECAY_RATE \
                        --lr-decay-steps=$LR_DECAY_STEPS \
                        --ic-weight=$IC_WEIGHT \
                        --y-stretch=$Y_STRETCH \
                        --ic-perturb-amp=$IC_PERTURB_AMP \
                        --hidden-dim=$HIDDEN_DIM \
                        --num-layers=$NUM_LAYERS \
                        --pirate-num-layers=$PIRATE_NUM_LAYERS \
                        --pirate-hidden-dim=$PIRATE_HIDDEN_DIM \
                        --pirate-nonlinearity=$PIRATE_NONLINEARITY \
                        --optimizer=$OPTIMIZER \
                        --causal-eps=$CAUSAL_EPS \
                        --causal-chunks=$CAUSAL_CHUNKS \
                        --soap-betas=$SOAP_BETAS \
                        --soap-shampoo-beta=$SOAP_SHAMPOO_BETA \
                        --soap-eps=$SOAP_EPS \
                        --soap-weight-decay=$SOAP_WEIGHT_DECAY \
                        --soap-precondition-frequency=$SOAP_PRECONDITION_FREQUENCY \
                        --seed=$seed \
                        --output-csv=$OUTPUT_CSV \
                        --tag=${TAG}_s${seed}
                    echo '========================================'
                    echo 'Finished: '\$(date)
                    echo '========================================'
                "
            ((job_count++))
        done
    done
done

# ============================================================================
# SUMMARY
# ============================================================================

n_channel_jobs=$((${#channel_methods[@]} * ${#channel_models[@]} * ${#channel_seeds[@]}))

echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED"
echo "========================================================================"
echo ""
echo "CHANNEL FLOW - GRADIENT-BASED (GPU):"
for method in "${channel_methods[@]}"; do
    echo "  - $method  x  ${#channel_models[@]} models  x  ${#channel_seeds[@]} seeds  =  $((${#channel_models[@]} * ${#channel_seeds[@]})) jobs"
done
echo "  Total: $n_channel_jobs jobs (each $NUM_WINDOWS win x $EPOCHS_PER_WINDOW ep = $((NUM_WINDOWS * EPOCHS_PER_WINDOW)) epochs, ~12-14 h on H100 MIG 2g.20gb projected)"
echo ""
echo "TOTAL JOBS: $job_count"
echo ""
echo "Results:  ./$OUTPUT_CSV"
echo "Logs:     ./logs/chan_*"
echo "========================================================================"
