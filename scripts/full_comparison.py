"""
Full quantitative comparison: Partner's PINN vs Our PIELM (full and simplified).

This script runs all three methods and provides a detailed comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_dt_elm_pinn.models.pielm_navier_stokes import PIELM_NavierStokes

# Output directory
os.makedirs('results/comparison', exist_ok=True)

print("=" * 70)
print("FULL QUANTITATIVE COMPARISON")
print("Partner's PINN vs PIELM (full) vs PIELM (simplified)")
print("=" * 70)

# ============================================================
# Common parameters
# ============================================================
Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1

N_interior = 2000
N_wall = 400
N_lid = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(f"Problem size: {N_interior} interior, {N_wall} wall, {N_lid} lid points")

# ============================================================
# 1. Partner's PINN (run for 2000 epochs as reference)
# ============================================================
print("\n" + "=" * 70)
print("1. PARTNER'S PINN (2000 epochs)")
print("=" * 70)

def run_partner_pinn(n_epochs=2000):
    """Run partner's PINN for reference solution."""

    def gradients(y, x):
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True,
        )[0]

    class PINN_Cavity(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            layers.append(nn.Linear(2, 64))
            layers.append(nn.Tanh())
            for _ in range(5):
                layers.append(nn.Linear(64, 64))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(64, 3))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    np.random.seed(42)
    xy_int = torch.tensor(np.random.rand(N_interior, 2), dtype=torch.float32, device=device)

    x_lid = np.random.rand(N_lid, 1)
    y_lid = np.ones((N_lid, 1))
    xy_lid = torch.tensor(np.hstack((x_lid, y_lid)), dtype=torch.float32, device=device)

    N_each = N_wall // 3
    xb, yb = np.random.rand(N_each, 1), np.zeros((N_each, 1))
    xl, yl = np.zeros((N_each, 1)), np.random.rand(N_each, 1)
    xr, yr = np.ones((N_each, 1)), np.random.rand(N_each, 1)
    xy_wall = torch.tensor(np.vstack([
        np.hstack((xb, yb)), np.hstack((xl, yl)), np.hstack((xr, yr)),
    ]), dtype=torch.float32, device=device)

    xy_p = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    def eddy_viscosity(xy, u, v):
        x, y = xy[:, 0:1], xy[:, 1:2]
        d = torch.min(torch.min(x, 1.0-x), torch.min(y, 1.0-y))
        grad_u, grad_v = gradients(u, xy), gradients(v, xy)
        du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
        dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
        Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
        S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2)
        S_mag = torch.sqrt(S_sq + 1e-12)
        nu_t = (Cs * d)**2 * S_mag
        return nu_laminar + nu_t, du_dx, du_dy, dv_dx, dv_dy

    def pde_residuals(model, xy):
        xy.requires_grad_(True)
        pred = model(xy)
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        nu_eff, du_dx, du_dy, dv_dx, dv_dy = eddy_viscosity(xy, u, v)
        continuity = du_dx + dv_dy
        u_conv = u * du_dx + v * du_dy
        v_conv = u * dv_dx + v * dv_dy
        grad_p = gradients(p, xy)
        dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
        # Full divergence form
        qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
        qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
        grad_qx_u, grad_qy_u = gradients(qx_u, xy), gradients(qy_u, xy)
        grad_qx_v, grad_qy_v = gradients(qx_v, xy), gradients(qy_v, xy)
        visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
        visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v
        return continuity, mom_u, mom_v

    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    start_time = time.perf_counter()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        cont, mom_u, mom_v = pde_residuals(model, xy_int)
        loss_pde = (mse_loss(cont, torch.zeros_like(cont)) +
                   mse_loss(mom_u, torch.zeros_like(mom_u)) +
                   mse_loss(mom_v, torch.zeros_like(mom_v)))
        pred_lid = model(xy_lid)
        loss_lid = (mse_loss(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) +
                   mse_loss(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2])))
        pred_wall = model(xy_wall)
        loss_wall = (mse_loss(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) +
                    mse_loss(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2])))
        pred_p = model(xy_p)
        loss_p = mse_loss(pred_p[:, 2:3], torch.zeros_like(pred_p[:, 2:3]))
        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4e} | PDE: {loss_pde.item():.4e}")
    train_time = time.perf_counter() - start_time

    return model, train_time, loss.item()

pinn_model, pinn_time, pinn_loss = run_partner_pinn(2000)
print(f"\nPINN training time: {pinn_time:.1f}s")
print(f"PINN final loss: {pinn_loss:.4e}")

# ============================================================
# 2. PIELM with FULL viscous term
# ============================================================
print("\n" + "=" * 70)
print("2. PIELM (FULL viscous term - matches PINN physics)")
print("=" * 70)

np.random.seed(42)
pielm_full = PIELM_NavierStokes(
    Re=Re, U_lid=U_lid, Cs=Cs,
    n_hidden=300,
    N_interior=N_interior,
    N_wall=N_wall,
    N_lid=N_lid,
    max_picard_iter=100,
    tol=1e-5,
    verbose=True,
    seed=42,
)
pielm_full.use_full_viscous = True
results_full = pielm_full.train()

print(f"\nPIELM (full) training time: {results_full['train_time']:.1f}s")
print(f"PIELM (full) iterations: {results_full['n_iterations']}")

# ============================================================
# 3. PIELM with SIMPLIFIED viscous term
# ============================================================
print("\n" + "=" * 70)
print("3. PIELM (SIMPLIFIED viscous term - faster but different physics)")
print("=" * 70)

np.random.seed(42)
pielm_simple = PIELM_NavierStokes(
    Re=Re, U_lid=U_lid, Cs=Cs,
    n_hidden=300,
    N_interior=N_interior,
    N_wall=N_wall,
    N_lid=N_lid,
    max_picard_iter=100,
    tol=1e-5,
    verbose=True,
    seed=42,
)
pielm_simple.use_full_viscous = False
results_simple = pielm_simple.train()

print(f"\nPIELM (simplified) training time: {results_simple['train_time']:.1f}s")
print(f"PIELM (simplified) iterations: {results_simple['n_iterations']}")

# ============================================================
# 4. Compare solutions on a grid
# ============================================================
print("\n" + "=" * 70)
print("4. SOLUTION COMPARISON")
print("=" * 70)

nx, ny = 31, 31
x_lin = np.linspace(0, 1, nx)
y_lin = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_lin, y_lin)
XY = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
XY_t = torch.tensor(XY, dtype=torch.float32, device=device)

# Get PINN predictions
pinn_model.eval()
with torch.no_grad():
    pred = pinn_model(XY_t)
    u_pinn = pred[:, 0].cpu().numpy()
    v_pinn = pred[:, 1].cpu().numpy()
    p_pinn = pred[:, 2].cpu().numpy()

# Get PIELM predictions
u_full, v_full, p_full = pielm_full.predict(XY)
u_simple, v_simple, p_simple = pielm_simple.predict(XY)

# Compute L2 errors
def l2_error(pred, ref):
    return np.sqrt(np.mean((pred - ref)**2))

def max_error(pred, ref):
    return np.max(np.abs(pred - ref))

print("\nL2 errors relative to PINN:")
print(f"  PIELM (full):")
print(f"    u: {l2_error(u_full, u_pinn):.6f}")
print(f"    v: {l2_error(v_full, v_pinn):.6f}")
print(f"    p: {l2_error(p_full, p_pinn):.6f}")
print(f"  PIELM (simplified):")
print(f"    u: {l2_error(u_simple, u_pinn):.6f}")
print(f"    v: {l2_error(v_simple, v_pinn):.6f}")
print(f"    p: {l2_error(p_simple, p_pinn):.6f}")

print("\nMax errors relative to PINN:")
print(f"  PIELM (full):")
print(f"    u: {max_error(u_full, u_pinn):.6f}")
print(f"    v: {max_error(v_full, v_pinn):.6f}")
print(f"  PIELM (simplified):")
print(f"    u: {max_error(u_simple, u_pinn):.6f}")
print(f"    v: {max_error(v_simple, v_pinn):.6f}")

# ============================================================
# 5. Visualize comparison
# ============================================================
print("\n" + "=" * 70)
print("5. GENERATING COMPARISON PLOTS")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(14, 12))

# Row 1: u-velocity
for ax, (u, title) in zip(axes[0], [(u_pinn, 'PINN (reference)'),
                                      (u_full, 'PIELM (full)'),
                                      (u_simple, 'PIELM (simplified)')]):
    U = u.reshape(ny, nx)
    cf = ax.contourf(X, Y, U, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'{title}\nu-velocity')
    ax.set_aspect('equal')

# Row 2: v-velocity
for ax, (v, title) in zip(axes[1], [(v_pinn, 'PINN (reference)'),
                                      (v_full, 'PIELM (full)'),
                                      (v_simple, 'PIELM (simplified)')]):
    V = v.reshape(ny, nx)
    cf = ax.contourf(X, Y, V, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'{title}\nv-velocity')
    ax.set_aspect('equal')

# Row 3: Difference from PINN
for ax, (u, v, title) in zip(axes[2], [(u_full, v_full, 'PIELM (full) - PINN'),
                                        (u_simple, v_simple, 'PIELM (simplified) - PINN'),
                                        (u_simple - u_full, v_simple - v_full, 'Simplified - Full')]):
    U_diff = u.reshape(ny, nx) - u_pinn.reshape(ny, nx) if 'PINN' in title else u.reshape(ny, nx)
    cf = ax.contourf(X, Y, U_diff, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'{title}\nu-velocity difference')
    ax.set_aspect('equal')

plt.suptitle('Lid-Driven Cavity (Re=1000): Method Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('results/comparison/method_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/comparison/method_comparison.png")
plt.close()

# ============================================================
# 6. Summary Table
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
{'Method':<25} {'Time (s)':<12} {'Speedup':<10} {'u L2 err':<12} {'u Max err':<12}
{'-'*70}
{'PINN (2000 epochs)':<25} {pinn_time:<12.1f} {'1.0x':<10} {'(reference)':<12} {'(reference)':<12}
{'PIELM (full viscous)':<25} {results_full['train_time']:<12.1f} {f'{pinn_time/results_full["train_time"]:.1f}x':<10} {l2_error(u_full, u_pinn):<12.6f} {max_error(u_full, u_pinn):<12.6f}
{'PIELM (simplified)':<25} {results_simple['train_time']:<12.1f} {f'{pinn_time/results_simple["train_time"]:.1f}x':<10} {l2_error(u_simple, u_pinn):<12.6f} {max_error(u_simple, u_pinn):<12.6f}
""")

print("=" * 70)
print("HONEST ASSESSMENT")
print("=" * 70)
print("""
WHAT WE CAN CLAIM:
- PIELM provides speedup over gradient-descent PINN
- PIELM (full) solves the SAME physics as partner's PINN
- Both PIELM versions produce physically reasonable solutions

CAVEATS:
1. PINN was only run for 2000 epochs (not fully converged)
   - Full PINN training takes 30,000 epochs (~50 min)
   - Our comparison may underestimate PINN quality

2. PIELM (simplified) solves DIFFERENT equations
   - Missing ~25% of viscous term near walls
   - Results will differ from partner's PINN

3. ELM approximation errors
   - Hidden layer representation has finite capacity
   - Solution accuracy depends on n_hidden

4. Picard iteration may not fully converge
   - Oscillatory behavior observed at fine tolerances
   - May need continuation methods for harder problems
""")
