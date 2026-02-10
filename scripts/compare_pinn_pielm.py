"""
Comprehensive comparison of partner's PINN vs our PIELM.

This script:
1. Extracts and compares the physics equations term-by-term
2. Runs both methods on the same problem
3. Identifies any differences or issues
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("CRITICAL COMPARISON: Partner's PINN vs Our PIELM")
print("=" * 70)

# ============================================================
# PART 1: Physics Equations Analysis (Term-by-Term)
# ============================================================

print("\n" + "=" * 70)
print("PART 1: PHYSICS EQUATIONS COMPARISON")
print("=" * 70)

print("""
Partner's PINN Equations (from their code):
-------------------------------------------

1. CONTINUITY:
   ∂u/∂x + ∂v/∂y = 0

2. MOMENTUM-X:
   u·∂u/∂x + v·∂u/∂y + ∂p/∂x - visc_u = 0

   where visc_u = ∇·(ν_eff·∇u)
               = ∂(ν_eff·∂u/∂x)/∂x + ∂(ν_eff·∂u/∂y)/∂y

   Expanding: visc_u = ν_eff·∇²u + (∂ν_eff/∂x)·(∂u/∂x) + (∂ν_eff/∂y)·(∂u/∂y)

3. MOMENTUM-Y:
   u·∂v/∂x + v·∂v/∂y + ∂p/∂y - visc_v = 0

   where visc_v = ∇·(ν_eff·∇v)

4. SMAGORINSKY MODEL:
   ν_eff = ν_laminar + (Cs·d)²·|S|
   where:
   - d = distance to nearest wall
   - |S| = sqrt(2·(S_xx² + S_yy² + 2·S_xy²))
   - S_xx = ∂u/∂x, S_yy = ∂v/∂y, S_xy = 0.5·(∂u/∂y + ∂v/∂x)

""")

print("""
Our PIELM Implementation (CURRENT):
------------------------------------

1. CONTINUITY: ∂u/∂x + ∂v/∂y = 0  ✓ MATCHES

2. MOMENTUM-X:
   u^k·∂u/∂x + v^k·∂u/∂y + ∂p/∂x - ν_eff·∇²u = 0

   ⚠️ PROBLEM: We use simplified viscous term!

   We have:    ν_eff·∇²u
   They have:  ∇·(ν_eff·∇u) = ν_eff·∇²u + ∇ν_eff·∇u

   MISSING TERM: ∇ν_eff·∇u = (∂ν_eff/∂x)·(∂u/∂x) + (∂ν_eff/∂y)·(∂u/∂y)

3. MOMENTUM-Y: Same issue as momentum-x

4. SMAGORINSKY MODEL: ✓ MATCHES

""")

print("""
============================================================
⚠️  CRITICAL DIFFERENCE IDENTIFIED!
============================================================

The partner's code uses the FULL divergence form of viscous stress:
   ∇·(ν_eff·∇u)

Our PIELM uses the SIMPLIFIED form:
   ν_eff·∇²u

The missing term (∇ν_eff·∇u) is NON-ZERO because:
1. ν_eff depends on wall distance d (varies spatially)
2. ν_eff depends on strain rate |S| (varies with velocity gradients)

This difference is SIGNIFICANT near walls where:
- Wall distance d changes rapidly
- Velocity gradients are large (boundary layer)

""")

# ============================================================
# PART 2: Quantify the Missing Term
# ============================================================

print("\n" + "=" * 70)
print("PART 2: ESTIMATING MAGNITUDE OF MISSING TERM")
print("=" * 70)

def estimate_missing_term_magnitude():
    """
    Estimate how large the missing ∇ν_eff·∇u term is compared to ν_eff·∇²u.
    """
    # Physical parameters
    Re = 1000.0
    U_lid = 1.0
    nu_laminar = U_lid / Re  # = 0.001
    Cs = 0.1

    # Near a wall (say y = 0.01, near bottom)
    y = 0.01  # Distance to wall
    d = y

    # Typical velocity gradients in boundary layer
    # u ≈ U_lid * (y / delta) where delta ~ 1/sqrt(Re) ~ 0.03
    delta = 1.0 / np.sqrt(Re)  # ~ 0.03
    du_dy = U_lid / delta  # ~ 30
    du_dx = 0.0  # roughly

    # Strain rate magnitude
    S_mag = np.sqrt(2 * du_dy**2)  # ~ 42

    # Eddy viscosity
    nu_turb = (Cs * d)**2 * S_mag  # = (0.1 * 0.01)^2 * 42 ~ 4.2e-5
    nu_eff = nu_laminar + nu_turb  # ~ 0.001

    # Gradient of nu_eff w.r.t. y (dominant component)
    # ν_turb = (Cs·d)²·|S|
    # ∂ν_turb/∂y ≈ 2·Cs²·d·|S| (from d = y)
    dnu_eff_dy = 2 * Cs**2 * d * S_mag  # ~ 0.0084

    # Laplacian approximation
    # ∂²u/∂y² ≈ -U_lid / delta² ~ -1000
    d2u_dy2 = -U_lid / (delta**2)

    # Compare terms:
    term1 = abs(nu_eff * d2u_dy2)  # ν_eff·∇²u ~ |0.001 * -1000| = 1.0
    term2 = abs(dnu_eff_dy * du_dy)  # (∂ν_eff/∂y)·(∂u/∂y) ~ |0.0084 * 30| = 0.25

    print(f"Near-wall estimates (y = {y}):")
    print(f"  ν_laminar = {nu_laminar:.4f}")
    print(f"  ν_turb = {nu_turb:.6f}")
    print(f"  ν_eff = {nu_eff:.4f}")
    print(f"  ∂ν_eff/∂y ≈ {dnu_eff_dy:.4f}")
    print(f"  ∂u/∂y ≈ {du_dy:.1f}")
    print(f"  ∂²u/∂y² ≈ {d2u_dy2:.1f}")
    print()
    print(f"Term magnitudes:")
    print(f"  |ν_eff·∇²u| ≈ {term1:.3f}")
    print(f"  |∇ν_eff·∇u| ≈ {term2:.3f}")
    print(f"  Ratio (missing/included): {term2/term1:.1%}")
    print()
    print(f"⚠️  Missing term is ~{term2/term1:.0%} of the included term near walls!")

estimate_missing_term_magnitude()

# ============================================================
# PART 3: Boundary Conditions Comparison
# ============================================================

print("\n" + "=" * 70)
print("PART 3: BOUNDARY CONDITIONS COMPARISON")
print("=" * 70)

print("""
Partner's BCs:
--------------
1. Lid (y=1):     u = U_lid = 1.0, v = 0  ✓
2. Bottom (y=0):  u = 0, v = 0            ✓
3. Left (x=0):    u = 0, v = 0            ✓
4. Right (x=1):   u = 0, v = 0            ✓
5. Pressure:      p(0.5, 0.5) = 0         ✓

Our PIELM BCs:
--------------
1. Lid (y=1):     u = U_lid = 1.0, v = 0  ✓
2. Bottom (y=0):  u = 0, v = 0            ✓
3. Left (x=0):    u = 0, v = 0            ✓
4. Right (x=1):   u = 0, v = 0            ✓
5. Pressure:      p(0.5, 0.5) = 0         ✓

✓ All boundary conditions MATCH!
""")

# ============================================================
# PART 4: Run Partner's PINN (short test)
# ============================================================

print("\n" + "=" * 70)
print("PART 4: RUNNING PARTNER'S PINN (SHORT TEST)")
print("=" * 70)

def run_partner_pinn_short(n_epochs=500):
    """Run partner's PINN for a short training to verify it works."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parameters
    Re = 1000.0
    U_lid = 1.0
    nu_laminar = U_lid / Re
    Cs = 0.1

    N_interior = 2000
    N_wall = 400
    N_lid = 400

    # Helper functions
    def gradients(y, x):
        return torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]

    # PINN model
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

    # Sampling
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
        np.hstack((xb, yb)),
        np.hstack((xl, yl)),
        np.hstack((xr, yr)),
    ]), dtype=torch.float32, device=device)

    xy_p = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    # Eddy viscosity function
    def eddy_viscosity(xy, u, v):
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        d = torch.min(torch.min(x, 1.0-x), torch.min(y, 1.0-y))

        grad_u = gradients(u, xy)
        grad_v = gradients(v, xy)
        du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
        dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2)
        S_mag = torch.sqrt(S_sq + 1e-12)

        nu_t = (Cs * d)**2 * S_mag
        nu_eff = nu_laminar + nu_t
        return nu_eff, du_dx, du_dy, dv_dx, dv_dy

    # PDE residuals (THEIR version with full divergence)
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

        # FULL divergence form
        qx_u = nu_eff * du_dx
        qy_u = nu_eff * du_dy
        qx_v = nu_eff * dv_dx
        qy_v = nu_eff * dv_dy

        grad_qx_u = gradients(qx_u, xy)
        grad_qy_u = gradients(qy_u, xy)
        grad_qx_v = gradients(qx_v, xy)
        grad_qy_v = gradients(qy_v, xy)

        visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
        visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]

        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v

        return continuity, mom_u, mom_v

    # Model and optimizer
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # Training
    print(f"\nTraining for {n_epochs} epochs...")
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

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4e} | PDE: {loss_pde.item():.4e}")

    print(f"\nFinal loss: {loss.item():.4e}")
    return model, loss.item()

try:
    pinn_model, pinn_loss = run_partner_pinn_short(500)
    print("✓ Partner's PINN runs successfully!")
except Exception as e:
    print(f"✗ Partner's PINN failed: {e}")
    pinn_model = None

# ============================================================
# PART 5: Summary of Issues
# ============================================================

print("\n" + "=" * 70)
print("PART 5: SUMMARY OF ISSUES AND CAVEATS")
print("=" * 70)

print("""
============================================================
⚠️  CRITICAL ISSUES FOUND:
============================================================

1. VISCOUS TERM MISMATCH (SERIOUS)
   ---------------------------------
   Partner's code: visc = ∇·(ν_eff·∇u) = ν_eff·∇²u + ∇ν_eff·∇u
   Our PIELM:      visc = ν_eff·∇²u

   Missing: ∇ν_eff·∇u term (~25% of viscous term near walls)

   Impact: Our solution will be LESS ACCURATE near walls where
   the Smagorinsky turbulence model is most active.

2. LINEARIZATION APPROACH
   -----------------------
   Our Picard iteration linearizes the convective terms by
   using previous iteration velocities. This is standard and OK.

   BUT: The eddy viscosity also depends on velocity gradients!
   We update ν_eff at each iteration, which is correct, but the
   missing ∇ν_eff·∇u term means we're not fully consistent.

3. NUMERICAL SCHEME DIFFERENCE
   ----------------------------
   Partner: Uses autodiff for ALL derivatives (exact)
   Us: Uses analytical derivatives of sigmoid/tanh basis

   This should be OK mathematically, but introduces approximation
   error from the ELM representation.

============================================================
WHAT THIS MEANS FOR THE PARTNER TEAM:
============================================================

We CANNOT claim our PIELM is a "drop-in replacement" because:

1. We solve a SIMPLIFIED version of their equations
2. The missing viscous term is significant (~25% near walls)
3. Results will differ, especially near boundaries

============================================================
OPTIONS TO FIX:
============================================================

Option A: Add the missing ∇ν_eff·∇u term to our PIELM
         - Requires computing ∂ν_eff/∂x and ∂ν_eff/∂y
         - More complex but makes equations match

Option B: Acknowledge the simplification
         - State that we use constant-coefficient approximation
         - Compare results to show difference magnitude
         - May still be useful for faster prototyping

Option C: Use Stokes (drop all nonlinear terms)
         - Much simpler, converges in 1 iteration
         - But loses the convective physics entirely

""")

print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
Before telling the partner team anything, we should:

1. FIX the viscous term to match theirs exactly
2. Run both methods on the same problem
3. Compare quantitatively (L2 error between solutions)
4. Only then make claims about speedup

Let me implement the fix now...
""")
