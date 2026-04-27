"""
Honest apples-to-apples comparison: DeepXDE baseline vs JVP methods.

Runs three methods sequentially on the same GPU:
1. DeepXDE (partner's code, exactly as-is)
2. JVP Chebyshev (our code, spectral grid)
3. JVP Random (our code, random collocation — fairer vs DeepXDE)

All evaluated with identical autograd PDE residuals on 161x81x20 uniform grid.

Usage:
  source env/bin/activate
  python -u src/compare_methods.py --outdir results/comparison_fresh
"""

import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ── Constants matching both codebases ──
NU = 1e-3
V0 = 1.0
X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = 0.0, 0.5
T_MIN, T_MAX = 0.0, 1.0
NX_EVAL, NY_EVAL, NT_EVAL = 161, 81, 20


def gradients(y, x):
    """Compute dy/dx via autograd."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]


def evaluate_ns_pde(model, device):
    """
    Evaluate PDE residuals on 161x81x20 uniform grid via autograd.
    This is the GROUND TRUTH evaluation used for all methods.
    Returns dict with pde_rms, continuity_rms, momentum_rms, and per-component details.
    """
    xs = np.linspace(X_MIN, X_MAX, NX_EVAL)
    ys = np.linspace(Y_MIN, Y_MAX, NY_EVAL)
    ts = np.linspace(T_MIN, T_MAX, NT_EVAL)

    all_cont, all_mu, all_mv = [], [], []
    # Also track BC/IC errors
    bc_inlet_u, bc_inlet_v = [], []
    bc_wall_u, bc_wall_v = [], []
    bc_outlet_p = []
    ic_u, ic_v, ic_p = [], [], []

    model.eval()

    for t_idx, t_val in enumerate(ts):
        X, Y = np.meshgrid(xs, ys)
        T_arr = np.full_like(X, t_val)
        xyt_np = np.column_stack([X.ravel(), Y.ravel(), T_arr.ravel()])
        xyt_t = torch.tensor(xyt_np, dtype=torch.float32, device=device,
                             requires_grad=True)

        pred = model(xyt_t)
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        grad_u = gradients(u, xyt_t)
        grad_v = gradients(v, xyt_t)
        grad_p = gradients(p, xyt_t)

        u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        grad_u_x = gradients(u_x, xyt_t)
        grad_u_y = gradients(u_y, xyt_t)
        grad_v_x = gradients(v_x, xyt_t)
        grad_v_y = gradients(v_y, xyt_t)
        u_xx = grad_u_x[:, 0:1]
        u_yy = grad_u_y[:, 1:2]
        v_xx = grad_v_x[:, 0:1]
        v_yy = grad_v_y[:, 1:2]

        cont = u_x + v_y
        mom_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        mom_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

        all_cont.append(cont.detach().cpu().numpy().flatten())
        all_mu.append(mom_u.detach().cpu().numpy().flatten())
        all_mv.append(mom_v.detach().cpu().numpy().flatten())

        # BC/IC errors (detach for evaluation)
        u_np = u.detach().cpu().numpy().reshape(NY_EVAL, NX_EVAL)
        v_np = v.detach().cpu().numpy().reshape(NY_EVAL, NX_EVAL)
        p_np = p.detach().cpu().numpy().reshape(NY_EVAL, NX_EVAL)

        # Inlet: x=0, u=V0, v=0
        bc_inlet_u.append(u_np[:, 0] - V0)
        bc_inlet_v.append(v_np[:, 0])

        # Walls: y=0 and y=0.5, u=0, v=0
        bc_wall_u.extend([u_np[0, :], u_np[-1, :]])
        bc_wall_v.extend([v_np[0, :], v_np[-1, :]])

        # Outlet: x=2, p=0
        bc_outlet_p.append(p_np[:, -1])

        # IC: t=0, u=0, v=0, p=0
        if t_idx == 0:
            ic_u.append(u_np.flatten())
            ic_v.append(v_np.flatten())
            ic_p.append(p_np.flatten())

    cont_all = np.concatenate(all_cont)
    mu_all = np.concatenate(all_mu)
    mv_all = np.concatenate(all_mv)

    pde_rms = float(np.sqrt(np.mean(cont_all**2 + mu_all**2 + mv_all**2)))
    cont_rms = float(np.sqrt(np.mean(cont_all**2)))
    mom_u_rms = float(np.sqrt(np.mean(mu_all**2)))
    mom_v_rms = float(np.sqrt(np.mean(mv_all**2)))
    mom_rms = float(np.sqrt(np.mean(mu_all**2 + mv_all**2)))

    # BC/IC RMS
    inlet_u_rms = float(np.sqrt(np.mean(np.concatenate(bc_inlet_u)**2)))
    inlet_v_rms = float(np.sqrt(np.mean(np.concatenate(bc_inlet_v)**2)))
    wall_u_rms = float(np.sqrt(np.mean(np.concatenate(bc_wall_u)**2)))
    wall_v_rms = float(np.sqrt(np.mean(np.concatenate(bc_wall_v)**2)))
    outlet_p_rms = float(np.sqrt(np.mean(np.concatenate(bc_outlet_p)**2)))
    ic_u_rms = float(np.sqrt(np.mean(np.concatenate(ic_u)**2)))
    ic_v_rms = float(np.sqrt(np.mean(np.concatenate(ic_v)**2)))
    ic_p_rms = float(np.sqrt(np.mean(np.concatenate(ic_p)**2)))

    model.train()

    return {
        'pde_rms': pde_rms,
        'continuity_rms': cont_rms,
        'momentum_u_rms': mom_u_rms,
        'momentum_v_rms': mom_v_rms,
        'momentum_rms': mom_rms,
        'inlet_u_rms': inlet_u_rms,
        'inlet_v_rms': inlet_v_rms,
        'wall_u_rms': wall_u_rms,
        'wall_v_rms': wall_v_rms,
        'outlet_p_rms': outlet_p_rms,
        'ic_u_rms': ic_u_rms,
        'ic_v_rms': ic_v_rms,
        'ic_p_rms': ic_p_rms,
    }


# ═══════════════════════════════════════════════════════════════════════
# Method 1: DeepXDE baseline (partner code)
# ═══════════════════════════════════════════════════════════════════════
def run_deepxde(outdir, seed=0):
    """Run partner's DeepXDE code and evaluate with our PDE residual function."""
    print("\n" + "=" * 70)
    print("METHOD 1: DeepXDE BASELINE (partner code)")
    print("=" * 70)

    # Set backend before importing
    os.environ["DDEBACKEND"] = "pytorch"
    import deepxde as dde

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.reset_peak_memory_stats(device)

    # Build geometry (matching partner code exactly)
    rect = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])
    timedomain = dde.geometry.TimeDomain(T_MIN, T_MAX)
    geomtime = dde.geometry.GeometryXTime(rect, timedomain)

    # BCs (matching partner code exactly)
    def inlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], X_MIN)

    def outlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], X_MAX)

    def wall_bottom(x, on_boundary):
        return on_boundary and np.isclose(x[1], Y_MIN)

    def wall_top(x, on_boundary):
        return on_boundary and np.isclose(x[1], Y_MAX)

    bc_in_u = dde.icbc.DirichletBC(geomtime, lambda x: V0, inlet, component=0)
    bc_in_v = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, inlet, component=1)
    bc_wb_u = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_bottom, component=0)
    bc_wb_v = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_bottom, component=1)
    bc_wt_u = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_top, component=0)
    bc_wt_v = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_top, component=1)
    bc_out_p = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, outlet, component=2)
    ic_u = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_initial: on_initial, component=0)
    ic_v = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_initial: on_initial, component=1)
    ic_p = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_initial: on_initial, component=2)

    # PDE
    def navier_stokes_pde(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        p = y[:, 2:3]
        u_x = dde.grad.jacobian(y, x, i=0, j=0)
        u_y = dde.grad.jacobian(y, x, i=0, j=1)
        u_t = dde.grad.jacobian(y, x, i=0, j=2)
        v_x = dde.grad.jacobian(y, x, i=1, j=0)
        v_y = dde.grad.jacobian(y, x, i=1, j=1)
        v_t = dde.grad.jacobian(y, x, i=1, j=2)
        p_x = dde.grad.jacobian(y, x, i=2, j=0)
        p_y = dde.grad.jacobian(y, x, i=2, j=1)
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        mom_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        mom_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        cont = u_x + v_y
        return [mom_u, mom_v, cont]

    data = dde.data.TimePDE(
        geomtime,
        navier_stokes_pde,
        [bc_in_u, bc_in_v, bc_wb_u, bc_wb_v, bc_wt_u, bc_wt_v, bc_out_p,
         ic_u, ic_v, ic_p],
        num_domain=20000,
        num_boundary=4000,
        num_initial=4000,
    )

    net = dde.nn.FNN([3] + [128] * 6 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"[DeepXDE] Parameters: {n_params}")
    print(f"[DeepXDE] Points: 20K domain + 4K boundary + 4K IC = 28K")

    # ── Adam phase ──
    print(f"\n[DeepXDE] Starting Adam (20000 iterations, lr=1e-3)...")
    model.compile("adam", lr=1e-3)
    t0 = time.time()
    _, train_state = model.train(iterations=20000, display_every=1000)
    adam_time = time.time() - t0
    print(f"[DeepXDE] Adam done: {adam_time:.1f}s ({adam_time/60:.2f} min)")

    # Get per-component training losses after Adam
    adam_losses = train_state.loss_train[-1] if hasattr(train_state, 'loss_train') and len(train_state.loss_train) > 0 else None
    if adam_losses is not None:
        print(f"[DeepXDE] Adam final losses (13 terms): {adam_losses}")

    # ── L-BFGS phase ──
    print(f"\n[DeepXDE] Starting L-BFGS (DeepXDE defaults: maxiter=15000)...")
    model.compile("L-BFGS")
    t1 = time.time()
    _, train_state = model.train()
    lbfgs_time = time.time() - t1
    print(f"[DeepXDE] L-BFGS done: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")

    total_time = adam_time + lbfgs_time
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)

    # Get final training losses
    final_losses = None
    total_loss = None
    if hasattr(train_state, 'loss_train') and len(train_state.loss_train) > 0:
        final_losses = train_state.loss_train[-1]
        if hasattr(final_losses, '__iter__'):
            total_loss = float(sum(final_losses))
            print(f"[DeepXDE] Final losses (13 terms): {final_losses}")
        else:
            total_loss = float(final_losses)
            print(f"[DeepXDE] Final loss (scalar): {total_loss:.6f}")
        print(f"[DeepXDE] Total final loss: {total_loss:.6f}")

    # Save model
    save_path = os.path.join(outdir, "deepxde_model")
    model.save(save_path)
    print(f"[DeepXDE] Model saved to {save_path}")

    # ── Evaluate with our standard PDE residual function ──
    print(f"\n[DeepXDE] Evaluating PDE residuals on {NX_EVAL}x{NY_EVAL}x{NT_EVAL} grid...")
    # Extract the PyTorch network from DeepXDE model
    pytorch_net = model.net
    pytorch_net.to(device)
    eval_results = evaluate_ns_pde(pytorch_net, device)

    print(f"\n[DeepXDE] ── RESULTS ──")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Adam time:  {adam_time:.1f}s ({adam_time/60:.2f} min)")
    print(f"  L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    print(f"  Parameters: {n_params}")
    print(f"  PDE RMS:        {eval_results['pde_rms']:.6f}")
    print(f"  Continuity RMS: {eval_results['continuity_rms']:.6f}")
    print(f"  Momentum-u RMS: {eval_results['momentum_u_rms']:.6f}")
    print(f"  Momentum-v RMS: {eval_results['momentum_v_rms']:.6f}")
    print(f"  Momentum RMS:   {eval_results['momentum_rms']:.6f}")
    print(f"  Inlet u RMS:    {eval_results['inlet_u_rms']:.6f}")
    print(f"  Inlet v RMS:    {eval_results['inlet_v_rms']:.6f}")
    print(f"  Wall u RMS:     {eval_results['wall_u_rms']:.6f}")
    print(f"  Wall v RMS:     {eval_results['wall_v_rms']:.6f}")
    print(f"  Outlet p RMS:   {eval_results['outlet_p_rms']:.6f}")
    print(f"  IC u RMS:       {eval_results['ic_u_rms']:.6f}")
    print(f"  IC v RMS:       {eval_results['ic_v_rms']:.6f}")
    print(f"  IC p RMS:       {eval_results['ic_p_rms']:.6f}")

    return {
        'method': 'DeepXDE',
        'total_time_s': total_time,
        'adam_time_s': adam_time,
        'lbfgs_time_s': lbfgs_time,
        'peak_mem_gb': peak_mem,
        'n_params': n_params,
        'n_points': 28000,
        'point_type': 'random (DeepXDE internal)',
        'final_loss': float(total_loss) if total_loss is not None else None,
        **eval_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Method 2 & 3: JVP (Chebyshev and Random)
# ═══════════════════════════════════════════════════════════════════════
def run_jvp(method, outdir, seed=0):
    """Run our JVP code (Chebyshev or Random) and evaluate."""
    method_name = "JVP Chebyshev" if method == "jvp" else "JVP Random"
    print("\n" + "=" * 70)
    print(f"METHOD: {method_name}")
    print("=" * 70)

    # Import our code
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

    # We need to run sage_partner_ns.py as a subprocess to get clean timing
    # But for evaluation, we need the model in this process
    # Solution: import the relevant functions
    import subprocess

    jvp_outdir = os.path.join(outdir, method)
    os.makedirs(jvp_outdir, exist_ok=True)

    cmd = [
        sys.executable, '-u', 'src/sage_partner_ns.py',
        '--method', method,
        '--stage', 'ns',
        '--adam_epochs', '20000',
        '--lbfgs',
        '--lbfgs_steps', '15000',
        '--seed', str(seed),
        '--outdir', jvp_outdir,
    ]

    if method == 'jvp':
        cmd.extend(['--Nx', '55', '--Ny', '15', '--Nt', '30'])
    elif method == 'jvp_random':
        cmd.extend(['--sampling', 'hammersley'])

    print(f"[{method_name}] Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd='/workspace/dt-pinn')
    wall_time = time.time() - t0

    if result.returncode != 0:
        print(f"[{method_name}] FAILED with return code {result.returncode}")
        return None

    print(f"\n[{method_name}] Subprocess complete in {wall_time:.1f}s ({wall_time/60:.2f} min)")

    # Load the saved model and evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check for results CSV
    results_csv = os.path.join(jvp_outdir, 'results.csv')
    timing = {}
    if os.path.exists(results_csv):
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timing = row
                break  # First row

    print(f"[{method_name}] Results from training subprocess:")
    for k, v in timing.items():
        print(f"  {k}: {v}")

    # Load model for our own evaluation
    # JVP saves as model_ns_jvp.pt or model_ns_jvp_random.pt with plain state_dict()
    model_path = os.path.join(jvp_outdir, f'model_ns_{method}.pt')
    if not os.path.exists(model_path):
        # Search for any .pt file
        pt_files = [f for f in os.listdir(jvp_outdir) if f.endswith('.pt')]
        if pt_files:
            model_path = os.path.join(jvp_outdir, pt_files[0])

    if os.path.exists(model_path):
        # Import model class
        from sage_partner_ns import FNN_NS
        model = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"\n[{method_name}] Re-evaluating PDE residuals on {NX_EVAL}x{NY_EVAL}x{NT_EVAL} grid...")
        eval_results = evaluate_ns_pde(model, device)

        print(f"\n[{method_name}] ── RESULTS ──")
        print(f"  PDE RMS:        {eval_results['pde_rms']:.6f}")
        print(f"  Continuity RMS: {eval_results['continuity_rms']:.6f}")
        print(f"  Momentum-u RMS: {eval_results['momentum_u_rms']:.6f}")
        print(f"  Momentum-v RMS: {eval_results['momentum_v_rms']:.6f}")
        print(f"  Momentum RMS:   {eval_results['momentum_rms']:.6f}")
        print(f"  Inlet u RMS:    {eval_results['inlet_u_rms']:.6f}")
        print(f"  Inlet v RMS:    {eval_results['inlet_v_rms']:.6f}")
        print(f"  Wall u RMS:     {eval_results['wall_u_rms']:.6f}")
        print(f"  Wall v RMS:     {eval_results['wall_v_rms']:.6f}")
        print(f"  Outlet p RMS:   {eval_results['outlet_p_rms']:.6f}")
        print(f"  IC u RMS:       {eval_results['ic_u_rms']:.6f}")
        print(f"  IC v RMS:       {eval_results['ic_v_rms']:.6f}")
        print(f"  IC p RMS:       {eval_results['ic_p_rms']:.6f}")

        return {
            'method': method_name,
            'timing_from_subprocess': timing,
            **eval_results,
        }
    else:
        print(f"[{method_name}] WARNING: No model file found at {model_path}")
        print(f"[{method_name}] Available files: {os.listdir(jvp_outdir)}")
        return {'method': method_name, 'timing_from_subprocess': timing}


def print_comparison(results):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("HONEST COMPARISON — ALL METRICS")
    print("=" * 90)

    metrics = [
        ('PDE RMS (total)', 'pde_rms'),
        ('Continuity RMS', 'continuity_rms'),
        ('Momentum-u RMS', 'momentum_u_rms'),
        ('Momentum-v RMS', 'momentum_v_rms'),
        ('Momentum RMS', 'momentum_rms'),
        ('Inlet u RMS', 'inlet_u_rms'),
        ('Inlet v RMS', 'inlet_v_rms'),
        ('Wall u RMS', 'wall_u_rms'),
        ('Wall v RMS', 'wall_v_rms'),
        ('Outlet p RMS', 'outlet_p_rms'),
        ('IC u RMS', 'ic_u_rms'),
        ('IC v RMS', 'ic_v_rms'),
        ('IC p RMS', 'ic_p_rms'),
    ]

    # Header
    header = f"{'Metric':<25}"
    for r in results:
        if r is not None:
            header += f"  {r['method']:<20}"
    print(header)
    print("-" * (25 + 22 * len(results)))

    for name, key in metrics:
        row = f"{name:<25}"
        for r in results:
            if r is not None and key in r:
                row += f"  {r[key]:<20.6f}"
            else:
                row += f"  {'N/A':<20}"
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description='Honest DeepXDE vs JVP comparison')
    parser.add_argument('--outdir', type=str, default='results/comparison_fresh',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-deepxde', action='store_true',
                        help='Skip DeepXDE run (for testing)')
    parser.add_argument('--skip-jvp', action='store_true',
                        help='Skip JVP Chebyshev run')
    parser.add_argument('--skip-jvp-random', action='store_true',
                        help='Skip JVP Random run')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    results = []

    # ── Run DeepXDE ──
    if not args.skip_deepxde:
        deepxde_result = run_deepxde(args.outdir, seed=args.seed)
        results.append(deepxde_result)
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("\n[SKIP] DeepXDE baseline skipped")

    # ── Run JVP Chebyshev ──
    if not args.skip_jvp:
        jvp_result = run_jvp('jvp', args.outdir, seed=args.seed)
        results.append(jvp_result)
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("\n[SKIP] JVP Chebyshev skipped")

    # ── Run JVP Random ──
    if not args.skip_jvp_random:
        jvp_random_result = run_jvp('jvp_random', args.outdir, seed=args.seed)
        results.append(jvp_random_result)
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("\n[SKIP] JVP Random skipped")

    # ── Print comparison ──
    print_comparison(results)

    # ── Save results ──
    results_file = os.path.join(args.outdir, 'comparison_results.json')
    # Make serializable
    serializable = []
    for r in results:
        if r is not None:
            s = {}
            for k, v in r.items():
                if isinstance(v, (int, float, str, bool, list, type(None))):
                    s[k] = v
                elif isinstance(v, dict):
                    s[k] = {str(kk): str(vv) for kk, vv in v.items()}
                else:
                    s[k] = str(v)
            serializable.append(s)

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
