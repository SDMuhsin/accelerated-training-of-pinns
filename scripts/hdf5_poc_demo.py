#!/usr/bin/env python3
"""
HDF5 Compression POC — Benchmark & Demo Script.

Demonstrates CSV/JSON → structured HDF5 (gzip+shuffle) as a drop-in
replacement for UofW's flat-file PINN/CFD exports.

Sections:
  1. Real data: Convert partner JSON files → HDF5
  2. Scale test: Synthetic CSVs at 5 grid sizes → HDF5
  3. PyTorch loading: Load HDF5 → tensors in 3 lines
  4. Selective read: Load single timestep (chunked I/O)
  5. Summary table: Compression ratios and I/O times

Usage:
    source env/bin/activate
    python scripts/hdf5_poc_demo.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdf5_compress import (
    csv_to_hdf5,
    generate_uofw_csv,
    hdf5_to_csv,
    json_to_hdf5,
    load_field_at_timestep,
    load_hdf5_to_torch,
    print_comparison_table,
)

RESULTS_DIR = "results/hdf5_poc"
os.makedirs(RESULTS_DIR, exist_ok=True)


def section(n, title):
    print(f"\n{'='*60}")
    print(f"  Section {n}: {title}")
    print(f"{'='*60}\n")


# ------------------------------------------------------------------ #
# Section 1: Real data — partner JSON → HDF5
# ------------------------------------------------------------------ #
def run_real_data():
    section(1, "Real Data Conversion (Partner JSON → HDF5)")

    results = []

    # Flow fields
    ff_json = "results/partner_pinn/flow_fields.json"
    ff_hdf5 = f"{RESULTS_DIR}/flow_fields.h5"
    if os.path.exists(ff_json):
        stats = json_to_hdf5(ff_json, ff_hdf5, source="partner_pinn_ns", units="nondim")
        print(f"flow_fields.json: {stats['json_size']/1e6:.1f} MB → "
              f"{stats['hdf5_size']/1e6:.2f} MB  "
              f"({stats['ratio']:.1f}x compression, {stats['write_time']:.2f}s)")
        print(f"  Fields: {stats['fields']}")
        results.append({
            "label": "flow_fields.json",
            "original_size": stats["json_size"],
            "hdf5_size": stats["hdf5_size"],
            "ratio": stats["ratio"],
            "write_time": stats["write_time"],
        })

        # Verify roundtrip correctness
        print("\n  Roundtrip verification (JSON → HDF5 → torch vs JSON → numpy):")
        with open(ff_json) as f:
            orig = json.load(f)
        loaded = load_hdf5_to_torch(ff_hdf5)
        for src_name, canon_name in [("u", "velocity_x"), ("v", "velocity_y"), ("p", "pressure")]:
            orig_arr = np.array(orig["fields"][src_name], dtype=np.float32)
            hdf5_arr = loaded["fields"][canon_name].numpy()
            # HDF5 has extra z-dim: (nt, 1, ny, nx) vs orig (nt, ny, nx)
            hdf5_arr = hdf5_arr.squeeze(1)
            max_err = np.max(np.abs(orig_arr - hdf5_arr))
            print(f"    {src_name} → {canon_name}: max error = {max_err:.2e}  "
                  f"{'PASS' if max_err < 1e-6 else 'FAIL'}")
    else:
        print(f"  [skip] {ff_json} not found")

    # Temperature
    t_json = "results/partner_pinn/temperature.json"
    t_hdf5 = f"{RESULTS_DIR}/temperature.h5"
    if os.path.exists(t_json):
        stats = json_to_hdf5(t_json, t_hdf5, source="partner_pinn_energy", units="nondim")
        print(f"\ntemperature.json: {stats['json_size']/1e6:.1f} MB → "
              f"{stats['hdf5_size']/1e6:.2f} MB  "
              f"({stats['ratio']:.1f}x compression, {stats['write_time']:.2f}s)")
        print(f"  Fields: {stats['fields']}")
        results.append({
            "label": "temperature.json",
            "original_size": stats["json_size"],
            "hdf5_size": stats["hdf5_size"],
            "ratio": stats["ratio"],
            "write_time": stats["write_time"],
        })

        # Verify roundtrip
        print("\n  Roundtrip verification:")
        with open(t_json) as f:
            orig = json.load(f)
        loaded = load_hdf5_to_torch(t_hdf5)
        orig_arr = np.array(orig["fields"]["T"], dtype=np.float32)
        hdf5_arr = loaded["fields"]["temperature"].numpy().squeeze(1)
        max_err = np.max(np.abs(orig_arr - hdf5_arr))
        print(f"    T → temperature: max error = {max_err:.2e}  "
              f"{'PASS' if max_err < 1e-6 else 'FAIL'}")
    else:
        print(f"  [skip] {t_json} not found")

    return results


# ------------------------------------------------------------------ #
# Section 2: Scale test — synthetic CSVs at various grid sizes
# ------------------------------------------------------------------ #
def run_scale_test():
    section(2, "Scale Test (Synthetic CSV → HDF5)")

    configs = [
        {"label": "4K rows",    "nx": 20,  "ny": 10,  "nz": 1, "nt": 20},
        {"label": "40K rows",   "nx": 50,  "ny": 20,  "nz": 1, "nt": 40},
        {"label": "400K rows",  "nx": 100, "ny": 50,  "nz": 1, "nt": 80},
        {"label": "2M rows",    "nx": 200, "ny": 100, "nz": 1, "nt": 100},
        {"label": "16M rows",   "nx": 400, "ny": 200, "nz": 1, "nt": 200},
    ]

    results = []
    for cfg in configs:
        label = cfg["label"]
        csv_path = f"{RESULTS_DIR}/synth_{label.replace(' ', '_')}.csv"
        h5_path  = f"{RESULTS_DIR}/synth_{label.replace(' ', '_')}.h5"

        gen = generate_uofw_csv(
            csv_path, nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"], nt=cfg["nt"],
            fields=["Pressure_bar"],
        )
        print(f"  Generated {label}: {gen['n_rows']:>10,} rows, "
              f"{gen['csv_size']/1e6:.1f} MB CSV")

        stats = csv_to_hdf5(csv_path, h5_path)
        print(f"    → HDF5: {stats['hdf5_size']/1e6:.2f} MB, "
              f"{stats['ratio']:.1f}x compression, {stats['write_time']:.1f}s")

        results.append({
            "label": f"Synthetic {label}",
            "original_size": stats["csv_size"],
            "hdf5_size": stats["hdf5_size"],
            "ratio": stats["ratio"],
            "write_time": stats["write_time"],
        })

        # CSV roundtrip check on smallest size
        if cfg == configs[0]:
            csv_rt_path = f"{RESULTS_DIR}/roundtrip_check.csv"
            hdf5_to_csv(h5_path, csv_rt_path)
            orig_data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            rt_data = np.genfromtxt(csv_rt_path, delimiter=",", skip_header=1)
            # Sort both by all columns to handle ordering
            orig_data = orig_data[np.lexsort(orig_data.T)]
            rt_data = rt_data[np.lexsort(rt_data.T)]
            max_err = np.max(np.abs(orig_data - rt_data))
            print(f"    CSV roundtrip check: max error = {max_err:.2e}  "
                  f"{'PASS' if max_err < 1e-4 else 'FAIL'}")

        # Clean up CSV to save disk
        os.remove(csv_path)

    return results


# ------------------------------------------------------------------ #
# Section 3: PyTorch loading
# ------------------------------------------------------------------ #
def run_torch_loading():
    section(3, "PyTorch Loading Demo")

    h5_path = f"{RESULTS_DIR}/flow_fields.h5"
    if not os.path.exists(h5_path):
        print("  [skip] No flow_fields.h5 — run Section 1 first")
        return

    import torch

    print("  # Load entire dataset as PyTorch tensors:")
    print("  >>> from src.hdf5_compress import load_hdf5_to_torch")
    print(f'  >>> data = load_hdf5_to_torch("{h5_path}")')
    print('  >>> pressure = data["fields"]["pressure"]  # (Nt, Nz, Ny, Nx) tensor')
    print()

    t0 = time.perf_counter()
    data = load_hdf5_to_torch(h5_path)
    load_time = time.perf_counter() - t0

    print(f"  Loaded in {load_time:.3f}s")
    print(f"  Coordinates: {list(data['coords'].keys())}")
    for name, tensor in data["fields"].items():
        print(f"  Field '{name}': shape={list(tensor.shape)}, dtype={tensor.dtype}")
    print(f"  Meta: {dict(data['meta'])}")


# ------------------------------------------------------------------ #
# Section 4: Selective read (single timestep)
# ------------------------------------------------------------------ #
def run_selective_read():
    section(4, "Selective Read (Single Timestep)")

    h5_path = f"{RESULTS_DIR}/flow_fields.h5"
    if not os.path.exists(h5_path):
        print("  [skip] No flow_fields.h5 — run Section 1 first")
        return

    print("  # Load just one timestep (O(chunk) I/O):")
    print("  >>> from src.hdf5_compress import load_field_at_timestep")
    print(f'  >>> p_t5 = load_field_at_timestep("{h5_path}", "pressure", timestep=5)')
    print()

    t0 = time.perf_counter()
    p_t5 = load_field_at_timestep(h5_path, "pressure", timestep=5)
    sel_time = time.perf_counter() - t0
    print(f"  Shape: {list(p_t5.shape)}, dtype: {p_t5.dtype}")
    print(f"  Time: {sel_time*1000:.1f} ms")

    # Compare: full load time
    t0 = time.perf_counter()
    data = load_hdf5_to_torch(h5_path)
    full_time = time.perf_counter() - t0
    print(f"  Full load time: {full_time*1000:.1f} ms")
    print(f"  Selective read is {full_time/sel_time:.1f}x faster than full load")


# ------------------------------------------------------------------ #
# Section 5: Summary table
# ------------------------------------------------------------------ #
def run_summary(real_results, scale_results):
    section(5, "Summary")
    all_results = real_results + scale_results
    if all_results:
        print_comparison_table(all_results, "HDF5 Compression Results (gzip level 4, shuffle=True)")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("HDF5 Compression POC — Benchmark & Demo")
    print(f"Output directory: {RESULTS_DIR}/")

    real_results = run_real_data()
    scale_results = run_scale_test()
    run_torch_loading()
    run_selective_read()
    run_summary(real_results, scale_results)

    print("\nDone. HDF5 files saved in:", RESULTS_DIR)
