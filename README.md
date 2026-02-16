# SAGE: Symbolic Analytical Gradient Engine for Accelerating Physics-Informed Neural Network Training

> **Paper Status:** Currently under review at *Springer Applied Intelligence*. For the BibTeX citation or a copy of the paper, please contact **sdmuhsin@gmail.com**.

SAGE is a tracing-based reverse-mode automatic differentiation engine that generates optimized backward functions for PDE residual computations in Physics-Informed Neural Networks (PINNs). It traces the forward PDE residual symbolically, walks the resulting tape in reverse applying vector-Jacobian product (VJP) rules, and emits an explicit backward function that eliminates AD graph construction and traversal at every training step.

## Key Results

| Problem | SAGE Speedup vs AutoDiff | Accuracy |
|---------|--------------------------|----------|
| Lid-Driven Cavity (NS + Smagorinsky) | **10.6-18.1x** | Matches or beats AutoDiff |
| Kovasznay Flow (constant-viscosity NS) | **7.2-13.4x** | Matches or beats AutoDiff |
| 2D Linear Elasticity (Navier-Cauchy) | **7.6-13.8x** | ~10% gap (float32 D² conditioning) |

Speedup scales with architecture complexity: MLP (7-11x) → TSA-PINN (9-12x) → PirateNet (14-18x).

## Repository Structure

```
├── src/
│   ├── lid_benchmark.py              # Main experiment runner (~3500 lines)
│   ├── symbolic_vjp.py               # SAGE engine (579 lines, 10 VJP rules)
│   └── experiment_dt_elm_pinn/
│       └── models/
│           ├── tsa_pinn.py            # TSA-PINN architecture
│           ├── pirate_net.py          # PirateNet architecture
│           └── pielm_navier_stokes.py # PIELM model
├── scripts/
│   ├── plot_training_curves.py        # Training convergence figures
│   └── plot_pareto_frontiers.py       # Accuracy-speed Pareto frontier figures
├── sbatch/
│   ├── run_lid_benchmark.sh           # SLURM script: main benchmarks
│   └── run_tracking.sh               # SLURM script: per-epoch tracking
├── results/                           # Benchmark output CSVs
│   ├── lid_benchmark_results.csv      # Main results (48 runs)
│   └── tracking_*.csv                 # Per-run epoch-level tracking
└── requirements.txt
```

## Requirements

- Python 3.10+
- NVIDIA GPU (tested on H100 80GB and A40 48GB)
- CUDA-compatible PyTorch

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `scipy`, `matplotlib`

## Reproducing Paper Results

### 1. Environment Setup

```bash
git clone <repository-url>
cd dt-pinn
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Running Individual Experiments

The unified CLI supports all combinations of problems, methods, models, and configurations:

```bash
python -u src/lid_benchmark.py \
    --problem <problem> \
    --method <method> \
    --model <model> \
    --epochs 30000 \
    --seed 42 \
    --track \
    --track-interval 100
```

**Problems:** `cavity` (Navier-Stokes + Smagorinsky, Re=1000), `kovasznay` (constant-viscosity NS, Re=40), `elasticity` (Navier-Cauchy)

**Methods:** `sage`, `autodiff`, `dtpinn`, `ropinn`, `sk-pinn`, `analytical` (cavity only), `pielm` (cavity only)

**Models:** `mlp`, `tsa-pinn`, `pirate-net`

**Examples:**

```bash
# SAGE on lid-driven cavity with MLP
python -u src/lid_benchmark.py --problem cavity --method sage --model mlp --epochs 30000 --seed 42

# AutoDiff baseline on Kovasznay flow with PirateNet
python -u src/lid_benchmark.py --problem kovasznay --method autodiff --model pirate-net --epochs 30000 --seed 42

# SAGE on linear elasticity with TSA-PINN
python -u src/lid_benchmark.py --problem elasticity --method sage --model tsa-pinn --epochs 30000 --seed 42
```

### 3. Full Benchmark Suite

To reproduce the complete set of 48 experiments (3 problems × up to 6 methods × 3 models) reported in the paper, use the provided SLURM scripts on an HPC cluster:

```bash
sbatch sbatch/run_lid_benchmark.sh    # Main benchmarks
sbatch sbatch/run_tracking.sh         # With per-epoch tracking
```

For non-SLURM environments, the individual commands can be extracted from these scripts and run sequentially.

### 4. Generating Figures

After benchmark results are saved to `results/`:

```bash
python scripts/plot_training_curves.py      # Figure 1: Training convergence
python scripts/plot_pareto_frontiers.py     # Figure 2: Pareto frontiers
```

### 5. Results

Results are written to `results/lid_benchmark_results.csv` with columns for problem, method, model, PDE RMS, training time, and speedup. Per-epoch tracking data (when `--track` is enabled) is saved as individual CSV files in `results/`.

## How SAGE Works

1. **Trace**: The forward PDE residual function is executed with symbolic proxy variables (`TracedVar`) that record every operation on an ordered tape.
2. **Reverse**: The tape is traversed in reverse, applying 10 VJP rules to accumulate adjoint expressions symbolically.
3. **Emit**: The accumulated expressions are assembled into an explicit Python function, compiled once via `exec()`, and cached.
4. **Train**: The generated backward function replaces the AD engine for the PDE residual portion, computing the upstream gradient as straight-line arithmetic with no graph overhead.

The key optimization is *adjoint accumulation before matrix multiplication*: all adjoint contributions to an intermediate variable are summed before the transpose matrix-vector product, reducing the backward matmul count to exactly the number of forward matmuls (Proposition 1 in the paper).

## Authors

- Sayed Muhsin (University of Saskatchewan) — sdmuhsin@gmail.com
- Chul B. Park (University of Toronto)
- Seokbum Ko (University of Saskatchewan, corresponding author)

## License

Please contact the authors for licensing information.
