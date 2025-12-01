"""
Main comparison script that runs both vanilla and DT-PINN from project root.
Compares training time and task performance.
"""

import subprocess
import json
import os
import sys

def run_vanilla():
    """Run vanilla PINN training"""
    print("="*70)
    print("RUNNING VANILLA PINN")
    print("="*70)
    result = subprocess.run(
        [sys.executable, "src/run_vanilla_small.py"],
        cwd="/workspace/dt-pinn",
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        return None
    return result.returncode == 0

def run_dtpinn():
    """Run DT-PINN training"""
    print("\n" + "="*70)
    print("RUNNING DT-PINN")
    print("="*70)
    result = subprocess.run(
        [sys.executable, "src/run_dtpinn_small.py"],
        cwd="/workspace/dt-pinn",
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        return None
    return result.returncode == 0

def compare_results():
    """Compare results from both methods"""
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)

    # Load results
    vanilla_file = "results/vanilla_pinn/2/582/results.json"
    dtpinn_file = "results/dtpinn/2/582/results.json"

    with open(vanilla_file, 'r') as f:
        vanilla_results = json.load(f)

    with open(dtpinn_file, 'r') as f:
        dtpinn_results = json.load(f)

    # Extract metrics
    vanilla_time = vanilla_results['epoch_time'][-1]
    dtpinn_time = dtpinn_results['epoch_time'][-1]
    speedup = vanilla_time / dtpinn_time

    vanilla_l2 = vanilla_results['test_l2_losses'][-1]
    dtpinn_l2 = dtpinn_results['test_l2_losses'][-1]

    print(f"\nVanilla PINN (fp32):")
    print(f"  Training time: {vanilla_time:.2f}s")
    print(f"  Final test L2 error: {vanilla_l2:.6f}")

    print(f"\nDT-PINN (fp64):")
    print(f"  Training time: {dtpinn_time:.2f}s")
    print(f"  Final test L2 error: {dtpinn_l2:.6f}")

    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Task performance cost: {dtpinn_l2/vanilla_l2:.2f}x worse" if dtpinn_l2 > vanilla_l2 else f"Task performance improvement: {vanilla_l2/dtpinn_l2:.2f}x better")

    # Save comparative analysis
    analysis = {
        'vanilla_time_s': vanilla_time,
        'dtpinn_time_s': dtpinn_time,
        'speedup': speedup,
        'vanilla_test_l2': vanilla_l2,
        'dtpinn_test_l2': dtpinn_l2,
        'performance_ratio': dtpinn_l2 / vanilla_l2 if vanilla_l2 > 0 else float('inf')
    }

    os.makedirs("llmdocs", exist_ok=True)
    with open("llmdocs/comparison.json", 'w') as f:
        json.dump(analysis, f, indent=4)

    return analysis

if __name__ == "__main__":
    # Run both trainings
    vanilla_success = run_vanilla()
    dtpinn_success = run_dtpinn()

    if vanilla_success and dtpinn_success:
        analysis = compare_results()
        print(f"\n\nComparative analysis saved to llmdocs/comparison.json")
    else:
        print("\nOne or both trainings failed!")
