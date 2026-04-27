#!/usr/bin/env python3
"""
Generate a demonstration GIF showing heat diffusion through a battery cooling
heat sink geometry, comparing three simulation speeds:

1. FEM (Traditional) - ~1 day simulation → barely progresses in 60 min
2. PINN (UofW partners) - ~1 hour simulation → completes around frame 60
3. SAGE (Ours) - ~10 minutes → completes quickly

Approach: compute the steady-state temperature field, then animate the
transient approach to it using an exponential schedule. Each method
progresses at a different rate based on its simulation speed.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar
from PIL import Image
import os
import io


# =============================================================================
# Geometry
# =============================================================================

def create_battery_heatsink_geometry(Nx=300, Ny=150):
    """
    Binary mask for a battery cooling plate.
    1 = solid (heat conducts), 0 = void (coolant channel / hole).

    Serpentine channels with solid bridges, pin-fin plenum, mounting holes.
    """
    mask = np.ones((Ny, Nx), dtype=np.float64)
    Y, X = np.meshgrid(np.linspace(0, 1, Ny), np.linspace(0, 2, Nx), indexing='ij')

    # Outer border
    m = 0.013
    mask[(Y < m) | (Y > 1 - m) | (X < m) | (X > 2 - m)] = 0

    # Serpentine channels (narrow, alternating, with wide bridges)
    ch_w = 0.013
    n_ch = 9
    y_ch = np.linspace(0.08, 0.92, n_ch)

    for i, yc in enumerate(y_ch):
        x0 = 0.05 if i % 2 == 0 else 0.15
        x1 = 1.85 if i % 2 == 0 else 1.95
        mask[(np.abs(Y - yc) < ch_w) & (X > x0) & (X < x1)] = 0

        # Wide solid bridges so heat flows across channels
        for bx in np.linspace(x0 + 0.08, x1 - 0.08, 10):
            mask[(np.abs(X - bx) < 0.020) & (np.abs(Y - yc) < ch_w + 0.004)] = 1

    # U-turn connections
    for i in range(n_ch - 1):
        xc = 1.85 if i % 2 == 0 else 0.15
        mask[(np.abs(X - xc) < ch_w) & (Y > y_ch[i]) & (Y < y_ch[i+1])] = 0

    # Pin-fin plenum (center)
    pl = (X > 0.78) & (X < 1.22) & (Y > 0.38) & (Y < 0.62)
    mask[pl] = 0
    for px in np.linspace(0.82, 1.18, 6):
        for j, py in enumerate(np.linspace(0.41, 0.59, 4)):
            off = 0.030 if j % 2 == 1 else 0
            mask[np.sqrt((X - px - off)**2 + (Y - py)**2) < 0.018] = 1

    # Small holes between channels (sparser to keep connectivity)
    for i in range(0, n_ch - 1, 2):
        ym = (y_ch[i] + y_ch[i+1]) / 2
        for hx in np.linspace(0.30, 1.70, 8):
            if abs(hx - 1.0) > 0.30 or abs(ym - 0.50) > 0.18:
                mask[np.sqrt((X - hx)**2 + (Y - ym)**2) < 0.012] = 0

    # Mounting holes
    for cx, cy in [(0.07, 0.05), (1.93, 0.05), (0.07, 0.95), (1.93, 0.95),
                   (1.00, 0.05), (1.00, 0.95)]:
        mask[np.sqrt((X - cx)**2 + (Y - cy)**2) < 0.022] = 0

    # Inlet / outlet
    mask[(X < 0.04) & (np.abs(Y - y_ch[0]) < 0.025)] = 0
    mask[(X > 1.96) & (np.abs(Y - y_ch[-1]) < 0.025)] = 0

    return mask, X, Y


# =============================================================================
# Steady-state solver (Jacobi iteration)
# =============================================================================

def compute_steady_state(mask, Nx, Ny, max_iter=150000, tol=1e-6):
    """
    Solve steady-state heat equation:
      -lap(T) = Q(x,y)    in solid
      T = 0                at void/channel cells (coolant sink)

    Battery cells generate heat across the entire plate surface (they sit
    on top). Coolant channels underneath absorb it. The result is a rich
    temperature field with gradients around every channel and hole.

    Uses SOR (Successive Over-Relaxation) for faster convergence.
    """
    Y, X = np.meshgrid(np.linspace(0, 1, Ny), np.linspace(0, 2, Nx), indexing='ij')

    # --- Heat source: distributed across plate (battery cells on top) ---
    source = np.zeros((Ny, Nx), dtype=np.float64)

    # Base heat generation everywhere on solid
    source[mask > 0] = 0.6

    # Hotter near battery cell centers (3x2 grid of rectangular cells)
    for cx in [0.35, 1.00, 1.65]:
        for cy in [0.30, 0.70]:
            cell = (np.abs(X - cx) < 0.22) & (np.abs(Y - cy) < 0.12) & (mask > 0)
            source[cell] = 1.0

    # Extra hot spots at cell centers
    for cx in [0.35, 1.00, 1.65]:
        for cy in [0.30, 0.70]:
            hot = (np.abs(X - cx) < 0.08) & (np.abs(Y - cy) < 0.05) & (mask > 0)
            source[hot] = 1.5

    # Scale for discrete Poisson (h² factor)
    dx = 2.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    h2 = 0.5 * (dx**2 + dy**2)
    Q = source * h2 * 60  # tuned for nice range after normalization

    # --- Jacobi iteration (unconditionally stable) ---
    T = np.zeros((Ny, Nx), dtype=np.float64)

    solid = mask > 0
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = solid[1:-1, 1:-1]

    print(f"    Solving steady state (Jacobi, max {max_iter} iter)...")
    for it in range(max_iter):
        T_new = T.copy()
        T_new[1:-1, 1:-1] = 0.25 * (
            T[1:-1, 2:] + T[1:-1, :-2] +
            T[2:, 1:-1] + T[:-2, 1:-1] +
            Q[1:-1, 1:-1]
        )

        # Enforce void = 0
        T_new *= mask

        if (it + 1) % 5000 == 0:
            diff = np.abs(T_new - T)[interior].max()
            if (it + 1) % 20000 == 0:
                print(f"      iter {it+1:6d}: Δ={diff:.2e}, "
                      f"max_T={T_new[interior].max():.4f}, "
                      f"mean_T={T_new[interior].mean():.4f}")
            if diff < tol:
                print(f"    Converged at iter {it+1} (Δ={diff:.2e})")
                T = T_new
                break

        T = T_new
    else:
        print(f"    Reached max iterations ({max_iter})")

    # Normalize to [0, 1]
    T_max = T.max()
    if T_max > 0:
        T /= T_max
    print(f"    Steady-state: max={T.max():.3f}, "
          f"mean(solid)={T[solid].mean():.3f}, "
          f"median(solid)={np.median(T[solid]):.3f}")

    return T


# =============================================================================
# Build animation frames from steady state
# =============================================================================

def build_frames(T_steady, mask, n_frames=60):
    """
    Build transient frames by interpolating from T=0 to T_steady using
    an exponential approach schedule:
        T(t) = T_steady * (1 - exp(-lambda * t))

    Lambda is chosen so that at t=1.0 (full simulation), T reaches ~99%
    of steady state.

    Returns n_frames frames spanning t from 0 to 1.
    """
    lam = 4.6  # exp(-4.6) ≈ 0.01, so ~99% at t=1

    frames = []
    for i in range(n_frames):
        # t goes from a small initial value to 1.0
        t = (i + 1) / n_frames  # 1/60 to 1.0
        frac = 1.0 - np.exp(-lam * t)
        frame = T_steady * frac
        frames.append(frame)

    return frames


# =============================================================================
# Rendering
# =============================================================================

def make_colormap():
    """Custom thermal: black → deep blue → purple → red → orange → yellow → white."""
    stops = [
        (0.00, '#1a1a40'),
        (0.08, '#1e1060'),
        (0.18, '#3b1578'),
        (0.30, '#6b1d6e'),
        (0.42, '#b42045'),
        (0.54, '#e04520'),
        (0.66, '#ee7518'),
        (0.77, '#f4a818'),
        (0.87, '#f8d848'),
        (0.95, '#fef4a0'),
        (1.00, '#ffffff'),
    ]
    positions = [s[0] for s in stops]
    rgbs = [mcolors.to_rgb(s[1]) for s in stops]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, (r, g, b) in zip(positions, rgbs):
        cdict['red'].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue'].append((pos, b, b))
    return mcolors.LinearSegmentedColormap('thermal', cdict, N=512)


def render_frame(mask, all_frames, frame_idx, cmap, void_rgba):
    """Render one animation frame with three side-by-side panels."""
    BG = '#ffffff'
    fig = plt.figure(figsize=(19.2, 8.8), facecolor=BG)

    # Title
    fig.text(0.5, 0.965, 'Battery Cooling Plate — Thermal Simulation',
             ha='center', va='center', fontsize=24, fontweight='bold',
             color='#1a1a1a', fontfamily='sans-serif')

    elapsed = frame_idx + 1
    fig.text(0.5, 0.92, f'Wall-Clock Time Elapsed:  {elapsed:02d} min',
             ha='center', va='center', fontsize=15, color='#555555',
             fontfamily='monospace')

    # Methods
    methods = [
        ('Traditional FEM', 'Full numerical solve  ·  Est. ~1 day',
         '#cc2222', 24 * 60),
        ('PINN  (UofW)', 'Neural PDE surrogate  ·  ~1 hour',
         '#2266cc', 60),
        ('Optimized PINN  (Ours)', 'Accelerated PINN inference  ·  ~10 min',
         '#118844', 10),
    ]

    n_frames = len(all_frames)

    for i, (name, subtitle, color, sim_total) in enumerate(methods):
        ax = fig.add_axes([0.022 + i * 0.325, 0.145, 0.315, 0.71])

        # How far has this method progressed?
        frac = min(1.0, elapsed / sim_total)
        fidx = min(n_frames - 1, int(frac * (n_frames - 1)))
        temp = all_frames[fidx]

        # Display solid cells; NaN for voids
        display = np.where(mask > 0, temp, np.nan)

        ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect='equal',
                  interpolation='bilinear', origin='lower',
                  extent=[0, 2, 0, 1])
        ax.imshow(void_rgba, aspect='equal', origin='lower',
                  extent=[0, 2, 0, 1], interpolation='nearest')

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2.5)

        badge = '⚡  ' if i == 2 else ''
        ax.set_title(f'{badge}{name}', fontsize=17, fontweight='bold',
                     color=color, pad=12, fontfamily='sans-serif')
        ax.text(0.5, -0.055, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, color='#666666',
                fontfamily='sans-serif')

        # Progress bar
        bar_y, bar_h = -0.115, 0.032
        bg_rect = plt.Rectangle((0.04, bar_y), 0.92, bar_h,
                                transform=ax.transAxes, fc='#e0e0e0',
                                ec='#bbbbbb', lw=0.8, clip_on=False)
        ax.add_patch(bg_rect)
        fill = plt.Rectangle((0.04, bar_y), 0.92 * frac, bar_h,
                              transform=ax.transAxes, fc=color,
                              alpha=0.85, clip_on=False)
        ax.add_patch(fill)
        lbl = 'COMPLETE ✓' if frac >= 1.0 else f'{frac*100:.1f}%'
        ax.text(0.5, bar_y + bar_h / 2, lbl, transform=ax.transAxes,
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color='white', fontfamily='monospace', clip_on=False)

    # Colorbar
    cbar_ax = fig.add_axes([0.36, 0.065, 0.28, 0.013])
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                          norm=mcolors.Normalize(0, 1),
                                          orientation='horizontal')
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['Cool', '', 'Hot'])
    cb.ax.tick_params(colors='#555555', labelsize=9)
    cb.outline.set_edgecolor('#aaaaaa')

    # Footnote
    fig.text(0.5, 0.008,
             '*Illustrative simulation. Actual FEM for production geometries takes days–weeks. '
             'PINN inference time depends on model, geometry, and hardware. '
             'Relative speedups are representative of observed results.',
             ha='center', va='bottom', fontsize=7.5, color='#888888',
             style='italic', fontfamily='sans-serif')

    # To PIL
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=BG,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Battery Cooling Plate — Heat Diffusion Demo GIF")
    print("=" * 60)

    outdir = 'results/heat_diffusion_demo'
    os.makedirs(outdir, exist_ok=True)

    # 1. Geometry
    print("\n[1/4] Creating geometry...")
    Nx, Ny = 300, 150
    mask, X, Y = create_battery_heatsink_geometry(Nx, Ny)
    sf = mask.sum() / mask.size
    print(f"  {Nx}x{Ny} grid, {sf:.0%} solid, {1-sf:.0%} void")

    # 2. Steady-state solution
    print("\n[2/4] Computing steady-state temperature field...")
    T_ss = compute_steady_state(mask, Nx, Ny, max_iter=80000, tol=1e-5)

    # 3. Build animation frames
    print("\n[3/4] Building animation frames...")
    frames = build_frames(T_ss, mask, n_frames=60)
    for fi in [0, 14, 29, 44, 59]:
        vals = frames[fi][mask > 0]
        print(f"  Frame {fi:2d}: mean={vals.mean():.3f}, max={vals.max():.3f}")

    # 4. Render
    print("\n[4/4] Rendering GIF frames...")
    cmap = make_colormap()

    # Precompute void overlay
    void_rgba = np.zeros((*mask.shape, 4))
    void_rgba[mask == 0] = [0.85, 0.85, 0.87, 1.0]

    pil_frames = []
    for i in range(60):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i+1}/60")
        pil_frames.append(render_frame(mask, frames, i, cmap, void_rgba))

    # Save GIF
    gif_path = os.path.join(outdir, 'heat_diffusion_comparison.gif')
    print(f"\n  Saving → {gif_path}")
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=167, loop=0)
    mb = os.path.getsize(gif_path) / 1024**2
    print(f"  Size: {mb:.1f} MB")

    # Key frames
    for idx, name in [(0, 'frame_01'), (29, 'frame_30'), (59, 'frame_60')]:
        pil_frames[idx].save(os.path.join(outdir, f'{name}.png'))
    print("  Key frame PNGs saved")

    print(f"\n{'='*60}\nDONE!\n{'='*60}")


if __name__ == '__main__':
    main()
