"""
Verify JVP-trained model: compare JVP derivatives vs autograd derivatives.
Also compare with the old DT-PINN spectral derivative mismatch.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sage_partner_ns import (
    FNN_NS, build_3d_grid, compute_pde_ns_3d, compute_pde_ns_3d_jvp,
    NU, V0, evaluate_ns,
)

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load both models
print("Loading JVP-trained model...")
model_jvp = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
model_jvp.load_state_dict(torch.load('results/jvp_full/model_ns_jvp.pt',
                                       map_location=device, weights_only=True))
model_jvp.eval()

print("Loading old DT-PINN model...")
model_dtpinn = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
model_dtpinn.load_state_dict(torch.load('results/sage_partner/model_ns_dtpinn.pt',
                                          map_location=device, weights_only=True))
model_dtpinn.eval()

# Build grid
g = build_3d_grid(55, 15, 30, device)
ii = g['interior_idx']

def measure_mismatch(model, name):
    """Measure spectral vs autograd derivative mismatch."""
    with torch.no_grad():
        pred = model(g['xyt_all'])
        u = pred[:, 0:1]
        v = pred[:, 1:2]

        # Spectral
        cont_sp = torch.sparse.mm(g['Dx'], u) + torch.sparse.mm(g['Dy'], v)
        mom_u_sp = (torch.sparse.mm(g['Dt'], u) +
                    u * torch.sparse.mm(g['Dx'], u) +
                    v * torch.sparse.mm(g['Dy'], u) +
                    torch.sparse.mm(g['Dx'], pred[:, 2:3]) -
                    NU * (torch.sparse.mm(g['Dxx'], u) + torch.sparse.mm(g['Dyy'], u)))

    # Autograd
    xyt = g['xyt_all'].detach().requires_grad_(True)
    pred_ag = model(xyt)
    u_ag, v_ag, p_ag = pred_ag[:, 0:1], pred_ag[:, 1:2], pred_ag[:, 2:3]

    grad_u = torch.autograd.grad(u_ag.sum(), xyt, create_graph=True)[0]
    grad_v = torch.autograd.grad(v_ag.sum(), xyt, create_graph=True)[0]
    grad_p = torch.autograd.grad(p_ag.sum(), xyt, create_graph=True)[0]

    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
    p_x = grad_p[:, 0:1]

    grad_ux = torch.autograd.grad(u_x.sum(), xyt, create_graph=False, retain_graph=True)[0]
    grad_uy = torch.autograd.grad(u_y.sum(), xyt, create_graph=False, retain_graph=True)[0]
    u_xx = grad_ux[:, 0:1]
    u_yy = grad_uy[:, 1:2]

    cont_ag = u_x + v_y
    mom_u_ag = u_t + u_ag * u_x + v_ag * u_y + p_x - NU * (u_xx + u_yy)

    # Compare
    cont_sp_int = cont_sp[ii].detach().cpu().numpy().flatten()
    cont_ag_int = cont_ag[ii].detach().cpu().numpy().flatten()
    mu_sp_int = mom_u_sp[ii].detach().cpu().numpy().flatten()
    mu_ag_int = mom_u_ag[ii].detach().cpu().numpy().flatten()

    sp_pde = np.sqrt(np.mean(cont_sp_int**2 + mu_sp_int**2))
    ag_pde = np.sqrt(np.mean(cont_ag_int**2 + mu_ag_int**2))
    sp_loss = np.mean(cont_sp_int**2) + np.mean(mu_sp_int**2)
    ag_loss = np.mean(cont_ag_int**2) + np.mean(mu_ag_int**2)

    print(f"\n  {name}:")
    print(f"    Spectral PDE RMS (interior): {sp_pde:.6f}")
    print(f"    Autograd PDE RMS (interior): {ag_pde:.6f}")
    print(f"    Ratio (ag/sp):               {ag_pde/max(sp_pde, 1e-15):.2f}x")
    print(f"    Spectral PDE loss (interior): {sp_loss:.6e}")
    print(f"    Autograd PDE loss (interior): {ag_loss:.6e}")
    print(f"    Loss ratio (ag/sp):           {ag_loss/max(sp_loss, 1e-30):.2f}x")

    return sp_pde, ag_pde

print("\n" + "=" * 70)
print("SPECTRAL vs AUTOGRAD DERIVATIVE COMPARISON")
print("=" * 70)

sp1, ag1 = measure_mismatch(model_dtpinn, "Old DT-PINN (spectral training)")
sp2, ag2 = measure_mismatch(model_jvp, "New JVP (exact derivative training)")

print("\n" + "=" * 70)
print("EVALUATION (161x81x20 uniform grid, autograd)")
print("=" * 70)

eval_dt = evaluate_ns(model_dtpinn, device)
eval_jvp = evaluate_ns(model_jvp, device)

print(f"\n  Old DT-PINN:  PDE RMS = {eval_dt['pde_rms']:.6f}")
print(f"  New JVP:      PDE RMS = {eval_jvp['pde_rms']:.6f}")
print(f"  Improvement:  {eval_dt['pde_rms']/eval_jvp['pde_rms']:.1f}x better accuracy")

print("\nDone.")
