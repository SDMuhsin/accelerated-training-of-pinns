#!/usr/bin/env python3
"""
Test script for Partner's Part 2 code: pressure_velocity_temperature_to_csv

This script runs the pressure/velocity/temperature simulation from the partner's
PINN_Battery.ipynb notebook to understand what it produces and how long it takes.

IMPORTANT FINDING:
==================
Part 2 is NOT a PINN - it's a simple pixel-marching simulation that:
  - Propagates pressure from inlet to outlet pixel-by-pixel (BFS-like flood fill)
  - Computes velocity as |grad(P)| (NOT from momentum equations)
  - Simulates temperature cooling with a simple decay model (NOT heat equation)

The actual PINN is in Part 1 - it solves lid-driven cavity flow with:
  - Full Navier-Stokes with Smagorinsky turbulence model
  - Neural network trained to minimize PDE residuals
  - 30,000 training epochs
"""

import numpy as np
from PIL import Image
import csv
import time
import os

# Change to source directory where Pipe6.png is located
SOURCE_DIR = "/workspace/dt-pinn/from_partner_team/SourceCode"
OUTPUT_DIR = "/workspace/dt-pinn/scripts/output_part2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_domain(picture_name, width_mm, height_mm, inlet_xy_mm, outlet_xy_mm):
    """Analyze the domain image and print statistics."""
    img = Image.open(picture_name).convert("L")
    gray = np.asarray(img, dtype=np.float64) / 255.0
    inside = gray < 0.8  # dark = inside, white-ish = outside

    nrows, ncols = inside.shape
    n_inside = np.sum(inside)

    dx = width_mm / (ncols - 1)
    dy = height_mm / (nrows - 1)

    print(f"Image dimensions: {ncols} x {nrows} pixels")
    print(f"Physical domain: {width_mm} x {height_mm} mm")
    print(f"Grid spacing: dx={dx:.3f} mm, dy={dy:.3f} mm")
    print(f"Inside (fluid) pixels: {n_inside:,} of {nrows*ncols:,} total ({100*n_inside/(nrows*ncols):.1f}%)")

    return inside, nrows, ncols, dx, dy


def run_pressure_velocity_only(
    picture_name,
    width_mm,
    height_mm,
    inlet_xy_mm,
    outlet_xy_mm,
    pressure_csv="pressure_full_simulation.csv",
    velocity_csv="velocity_full_simulation.csv",
):
    """
    Run only the pressure/velocity simulation (Part 2a) - skip temperature.

    This demonstrates the algorithm without the massive temperature output.
    """
    # ============================================================
    #                 (A) PRESSURE + VELOCITY SETTINGS
    # ============================================================
    P_in  = 0.8   # [bar]
    P_out = 0.1   # [bar]
    t_final_P = 30.0
    max_iters_est_P = 2000
    dt_P = t_final_P / max_iters_est_P
    rand_sigmaP = 0.005

    # ============================================================
    #                 LOAD IMAGE
    # ============================================================
    img = Image.open(picture_name).convert("L")
    gray = np.asarray(img, dtype=np.float64) / 255.0
    inside = gray < 0.8

    nrows, ncols = inside.shape
    inside_rc_all = np.argwhere(inside)

    dx = width_mm / (ncols - 1)
    dy = height_mm / (nrows - 1)

    x_pix = np.arange(ncols, dtype=np.float64)
    y_pix = np.arange(nrows, dtype=np.float64)
    Xmm, Ymm = np.meshgrid(x_pix * dx, y_pix * dy)

    # ============================================================
    #     MAP inlet/outlet
    # ============================================================
    def xy_mm_to_nearest_inside_rc(xy_mm):
        x, y = float(xy_mm[0]), float(xy_mm[1])
        x = min(max(x, 0.0), width_mm)
        y = min(max(y, 0.0), height_mm)

        c0 = int(np.round(x / dx))
        r0 = int(np.round(y / dy))
        c0 = min(max(c0, 0), ncols - 1)
        r0 = min(max(r0, 0), nrows - 1)

        if inside[r0, c0]:
            return r0, c0

        rr = inside_rc_all[:, 0].astype(np.float64)
        cc = inside_rc_all[:, 1].astype(np.float64)
        x_all = cc * dx
        y_all = rr * dy
        d2 = (x_all - x)**2 + (y_all - y)**2
        k = int(np.argmin(d2))
        return int(rr[k]), int(cc[k])

    in_r, in_c = xy_mm_to_nearest_inside_rc(inlet_xy_mm)
    out_r, out_c = xy_mm_to_nearest_inside_rc(outlet_xy_mm)

    print(f"Inlet  (x={inlet_xy_mm[0]}, y={inlet_xy_mm[1]} mm) -> pixel (row={in_r}, col={in_c})")
    print(f"Outlet (x={outlet_xy_mm[0]}, y={outlet_xy_mm[1]} mm) -> pixel (row={out_r}, col={out_c})")

    inlet_mask = np.zeros((nrows, ncols), dtype=bool)
    outlet_mask = np.zeros((nrows, ncols), dtype=bool)
    inlet_mask[in_r, in_c] = True
    outlet_mask[out_r, out_c] = True

    # ============================================================
    #                 CSV HEADERS
    # ============================================================
    with open(pressure_csv, "w", newline="") as f:
        csv.writer(f).writerow(["x_mm", "y_mm", "time_s", "Pressure_bar"])

    with open(velocity_csv, "w", newline="") as f:
        csv.writer(f).writerow(["x_mm", "y_mm", "time_s", "v"])

    # ============================================================
    #                 PRESSURE + VELOCITY SIMULATION
    # ============================================================
    print("\n--- Pressure + Velocity Simulation (Flood-Fill Algorithm) ---")
    start_time = time.time()

    pres = np.full((nrows, ncols), np.nan, dtype=np.float64)
    state = np.zeros((nrows, ncols), dtype=np.uint8)  # 0=unreached, 1=front, 2=reached

    state[inlet_mask] = 1
    pres[inlet_mask] = P_in
    pres[outlet_mask] = P_out

    next_save_time = 1.0
    pres_prev_snapshot = None

    t_elapsed = 0.0
    done = False
    iteration = 0
    time_snapshots = []

    while not done:
        iteration += 1
        old_state = state.copy()
        new_state = old_state.copy()

        rs, cs = np.where(old_state == 1)

        for i, j in zip(rs, cs):
            r1 = max(0, i - 1)
            r2 = min(nrows - 1, i + 1)
            c1 = max(0, j - 1)
            c2 = min(ncols - 1, j + 1)

            for ii in range(r1, r2 + 1):
                for jj in range(c1, c2 + 1):
                    if inside[ii, jj] and old_state[ii, jj] == 0:
                        new_state[ii, jj] = 1
                        P_new = P_in - (P_in - P_out) * (t_elapsed / t_final_P)
                        P_new = min(max(P_new, P_out), P_in)
                        pres[ii, jj] = P_out if outlet_mask[ii, jj] else P_new

            new_state[i, j] = 2

        state = new_state

        if (not np.any(state[inside] == 0)) or (t_elapsed >= (t_final_P - 1e-9)):
            done = True

        t_elapsed += dt_P

        while (next_save_time <= t_final_P + 1e-9) and (t_elapsed >= next_save_time - 1e-9):
            t_snap = next_save_time

            mask_reached = inside & (~np.isnan(pres))
            n_reached = np.sum(mask_reached)
            time_snapshots.append((t_snap, n_reached))

            x_vec = Xmm[mask_reached]
            y_vec = Ymm[mask_reached]
            P_vec = pres[mask_reached]

            P_out_vec = P_vec + np.random.randn(P_vec.size) * rand_sigmaP
            with open(pressure_csv, "a", newline="") as f:
                w = csv.writer(f)
                for x_mm, y_mm, p in zip(x_vec, y_vec, P_out_vec):
                    w.writerow([float(x_mm), float(y_mm), float(t_snap), float(p)])

            if pres_prev_snapshot is None:
                v_vec = np.zeros_like(P_vec)
            else:
                filled = pres_prev_snapshot.copy()
                filled[~inside] = np.nan
                filled = np.nan_to_num(filled, nan=P_out)
                filled[inlet_mask] = P_in
                filled[outlet_mask] = P_out

                dP_dy, dP_dx = np.gradient(filled, dy, dx)
                v_full = np.sqrt(dP_dx**2 + dP_dy**2)
                v_vec = v_full[mask_reached]

            with open(velocity_csv, "a", newline="") as f:
                w = csv.writer(f)
                for x_mm, y_mm, v in zip(x_vec, y_vec, v_vec):
                    w.writerow([float(x_mm), float(y_mm), float(t_snap), float(v)])

            pres_prev_snapshot = pres.copy()
            next_save_time += 1.0

    end_time = time.time()
    duration = end_time - start_time

    return duration, iteration, time_snapshots, pres, inside


def main():
    print("=" * 70)
    print("Partner's Part 2: Pressure/Velocity/Temperature Simulation Analysis")
    print("=" * 70)
    print()

    # Set up file paths
    image_path = os.path.join(SOURCE_DIR, "Pipe6.png")
    pressure_csv = os.path.join(OUTPUT_DIR, "pressure_full_simulation.csv")
    velocity_csv = os.path.join(OUTPUT_DIR, "velocity_full_simulation.csv")

    # Default parameters from the notebook
    width_mm = 2000.0
    height_mm = 1000.0
    inlet_xy_mm = (10.0, 50.0)
    outlet_xy_mm = (1900.0, 900.0)

    print("IMPORTANT: This Part 2 code is NOT a PINN!")
    print("-" * 70)
    print("It's a simple heuristic simulation (NOT physics-based):")
    print("  1. Pressure: Flood-fill from inlet, linear interpolation to outlet")
    print("  2. Velocity: |grad(P)| (NOT from Navier-Stokes momentum equations)")
    print("  3. Temperature: Simple decay model (NOT from heat equation)")
    print()
    print("The actual PINN is in Part 1 - lid-driven cavity with Navier-Stokes.")
    print("=" * 70)
    print()

    # Analyze domain
    print("--- Domain Analysis ---")
    inside, nrows, ncols, dx, dy = analyze_domain(
        image_path, width_mm, height_mm, inlet_xy_mm, outlet_xy_mm
    )
    print()

    # Run pressure/velocity only (skip temperature - it's huge)
    duration, iterations, time_snapshots, final_pres, inside_mask = run_pressure_velocity_only(
        picture_name=image_path,
        width_mm=width_mm,
        height_mm=height_mm,
        inlet_xy_mm=inlet_xy_mm,
        outlet_xy_mm=outlet_xy_mm,
        pressure_csv=pressure_csv,
        velocity_csv=velocity_csv,
    )

    # Report results
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Pressure/Velocity simulation:")
    print(f"  - Duration: {duration:.2f} seconds")
    print(f"  - Iterations: {iterations}")
    print(f"  - Time steps saved: {len(time_snapshots)}")
    print()

    print("Flood-fill propagation over time:")
    for t, n_pts in time_snapshots[:5]:
        print(f"  t={t:.0f}s: {n_pts:,} points reached")
    if len(time_snapshots) > 5:
        print(f"  ...")
        for t, n_pts in time_snapshots[-2:]:
            print(f"  t={t:.0f}s: {n_pts:,} points reached")
    print()

    # Check output files
    print("Output files:")
    for csv_file in [pressure_csv, velocity_csv]:
        if os.path.exists(csv_file):
            size_kb = os.path.getsize(csv_file) / 1024
            with open(csv_file, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  {os.path.basename(csv_file)}: {size_kb:.1f} KB, {line_count:,} lines")
    print()

    # Summary statistics
    valid_pres = final_pres[inside_mask & ~np.isnan(final_pres)]
    print("Final pressure field statistics:")
    print(f"  Min:  {np.min(valid_pres):.4f} bar")
    print(f"  Max:  {np.max(valid_pres):.4f} bar")
    print(f"  Mean: {np.mean(valid_pres):.4f} bar")
    print()

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Part 2 is NOT solving any PDEs - it's a simple flood-fill algorithm

2. The "velocity" is just |grad(P)|, not from momentum equations
   - This is NOT physical fluid velocity
   - Real velocity requires solving Navier-Stokes

3. The temperature is a simple decay model, not heat equation
   - No advection (fluid transport)
   - No diffusion (conduction)

4. This produces time-series CSV data for visualization purposes only

5. The actual PINN (Part 1) solves:
   - Steady Navier-Stokes with Smagorinsky turbulence
   - Lid-driven cavity benchmark (different from this pipe geometry)
   - Requires 30,000 epochs of training

CONCLUSION: Part 2 is a post-processing visualization tool, NOT a physics solver.
The partner's PINN (Part 1) is what we should compare against for Navier-Stokes.
""")


if __name__ == "__main__":
    main()
