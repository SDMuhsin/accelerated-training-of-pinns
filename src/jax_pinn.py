"""JAX+JIT PINN baseline for the SAGE benchmark.

This module implements the `jaxpinn` method referenced from
`src/lid_benchmark.py`. It uses the standard JAX PINN approach:
- Flax MLP / PirateNet architectures mirroring the PyTorch versions.
- PDE residuals via `jax.grad`/`jax.hessian` on a per-point neural net,
  vmapped over collocation points (NOT spectral matrices).
- Adam via optax.
- `@jax.jit` on the whole training step.
- Timing uses one warmup jit call, then `block_until_ready` on the
  returned loss to force completion before stopping the clock.

After training the Flax parameters are copied into an equivalently-shaped
PyTorch model so that the existing `evaluate_*` routines (which rely on
PyTorch autograd) can compute PDE RMS metrics without reimplementation.
"""
from __future__ import annotations

import math
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import jax
# Force full fp32 matmul precision. JAX/XLA defaults to TF32 (19-bit mantissa)
# on Ampere+ GPUs, which breaks SAGE-JAX training on Kovasznay and Elasticity:
# the under-regularized spectral-Chebyshev loss converges to a non-smooth
# minimum with high-frequency artifacts that TF32 cannot distinguish from
# the true smooth minimum. Applied module-wide so jaxpinn and sage-jax are
# measured under identical precision — PyTorch already uses full fp32 matmul
# by default (torch.backends.cuda.matmul.allow_tf32=False since 1.12).
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import random, jit, vmap, grad, hessian
import flax.linen as fnn
import optax

# ----------------------------------------------------------------------------
# Physics constants (mirrors src/lid_benchmark.py)
# ----------------------------------------------------------------------------
Re = 1000.0
U_lid = 1.0
NU_LAM = U_lid / Re  # 0.001
CS = 0.1

Re_KOV = 40.0
NU_KOV = 1.0 / Re_KOV
LAMBDA_KOV = Re_KOV / 2.0 - math.sqrt(Re_KOV ** 2 / 4.0 + 4.0 * math.pi ** 2)

LAM_E = 1.0
MU_E = 0.5
Q_E = 4.0


# ----------------------------------------------------------------------------
# Flax architectures — param counts matched to the PyTorch versions.
# ----------------------------------------------------------------------------
class MlpFlax(fnn.Module):
    """6-hidden-layer tanh MLP. Matches PINN_Cavity (21,187 params for out_dim=3)."""
    hidden_dim: int = 64
    num_hidden: int = 6
    out_dim: int = 3

    @fnn.compact
    def __call__(self, x):
        for _ in range(self.num_hidden):
            x = fnn.Dense(features=self.hidden_dim)(x)
            x = jnp.tanh(x)
        x = fnn.Dense(features=self.out_dim)(x)
        return x


class PIModifiedBottleneckFlax(fnn.Module):
    hidden_dim: int
    nonlinearity: float = 0.0

    @fnn.compact
    def __call__(self, x, u, v):
        identity = x
        z = jnp.tanh(fnn.Dense(features=self.hidden_dim)(x))
        z = z * u + (1.0 - z) * v
        z = jnp.tanh(fnn.Dense(features=self.hidden_dim)(z))
        z = z * u + (1.0 - z) * v
        z = jnp.tanh(fnn.Dense(features=self.hidden_dim)(z))
        alpha = self.param("alpha", lambda key: jnp.asarray(self.nonlinearity))
        return alpha * z + (1.0 - alpha) * identity


class PirateNetFlax(fnn.Module):
    """PirateNet matching src/experiment_dt_elm_pinn/models/pirate_net.py.

    hidden_dim=38, num_blocks=4. Param count: 20,983 for out_dim=3, 20,944 for
    out_dim=2. Uses a learned linear projection (tanh) instead of Fourier
    embeddings, matching the PyTorch port.
    """
    hidden_dim: int = 38
    num_blocks: int = 4
    out_dim: int = 3

    @fnn.compact
    def __call__(self, x):
        h = jnp.tanh(fnn.Dense(features=self.hidden_dim, name="input_proj")(x))
        u = jnp.tanh(fnn.Dense(features=self.hidden_dim, name="encoder_u")(h))
        v = jnp.tanh(fnn.Dense(features=self.hidden_dim, name="encoder_v")(h))
        for i in range(self.num_blocks):
            h = PIModifiedBottleneckFlax(
                hidden_dim=self.hidden_dim, name=f"block_{i}"
            )(h, u, v)
        return fnn.Dense(features=self.out_dim, name="output_layer")(h)


def make_jax_model(model_name: str, out_dim: int):
    if model_name == "mlp":
        return MlpFlax(hidden_dim=64, num_hidden=6, out_dim=out_dim)
    if model_name == "pirate-net":
        return PirateNetFlax(hidden_dim=38, num_blocks=4, out_dim=out_dim)
    raise ValueError(f"JAX PINN supports model_name in {{mlp, pirate-net}}, got {model_name!r}")


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# ----------------------------------------------------------------------------
# Flax -> PyTorch parameter transfer (for the shared eval routines)
# ----------------------------------------------------------------------------
def _flax_dense_to_torch(linear: nn.Linear, flax_dense: dict):
    """Copy a Flax Dense layer's parameters into a torch Linear."""
    kernel = np.array(flax_dense["kernel"], copy=True)  # (in, out)
    bias = np.array(flax_dense["bias"], copy=True)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(kernel.T.copy()).to(linear.weight.device))
        linear.bias.copy_(torch.from_numpy(bias).to(linear.bias.device))


def flax_params_to_torch(model_name: str, params, out_dim: int, device: torch.device):
    """Build a fresh PyTorch model and copy Flax params in. Returns the torch model."""
    # Import here to avoid circular imports via src.lid_benchmark.
    from src.lid_benchmark import make_model
    torch_model = make_model(model_name, output_dim=out_dim).to(device)
    params = params["params"]  # Flax nests under 'params'
    if model_name == "mlp":
        # Flax: Dense_0..Dense_6 (6 hidden + output). Torch: self.net = Sequential(Linear, Tanh, Linear, Tanh, ..., Linear)
        # Torch layer order inside self.net: idx 0,2,4,6,8,10,12 are Linear (7 total).
        torch_linears = [m for m in torch_model.net if isinstance(m, nn.Linear)]
        flax_keys = sorted(params.keys(), key=lambda k: int(k.split("_")[1]))
        assert len(torch_linears) == len(flax_keys), (
            f"Dense count mismatch: torch {len(torch_linears)} vs flax {len(flax_keys)}")
        for t, k in zip(torch_linears, flax_keys):
            _flax_dense_to_torch(t, params[k])
    elif model_name == "pirate-net":
        _flax_dense_to_torch(torch_model.input_proj, params["input_proj"])
        _flax_dense_to_torch(torch_model.encoder_u, params["encoder_u"])
        _flax_dense_to_torch(torch_model.encoder_v, params["encoder_v"])
        _flax_dense_to_torch(torch_model.output_layer, params["output_layer"])
        for i, block in enumerate(torch_model.blocks):
            bparams = params[f"block_{i}"]
            # Flax auto-names: Dense_0, Dense_1, Dense_2 inside each bottleneck.
            dense_keys = sorted([k for k in bparams.keys() if k.startswith("Dense_")],
                                key=lambda k: int(k.split("_")[1]))
            _flax_dense_to_torch(block.fc1, bparams[dense_keys[0]])
            _flax_dense_to_torch(block.fc2, bparams[dense_keys[1]])
            _flax_dense_to_torch(block.fc3, bparams[dense_keys[2]])
            with torch.no_grad():
                block.alpha.copy_(torch.tensor(float(np.asarray(bparams["alpha"]).reshape(()))))
    else:
        raise ValueError(model_name)
    return torch_model


# ----------------------------------------------------------------------------
# Collocation samplers — Chebyshev grids matching PyTorch build_collocation_points_*
# ----------------------------------------------------------------------------
def _chebyshev_points(N):
    return np.cos(np.pi * np.arange(N) / (N - 1))


def cavity_points(N_grid):
    x_ref = _chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    eps = 1e-10
    xc, yc = xy[:, 0], xy[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    is_lid = (yc > 1 - eps)
    is_wall = is_boundary & ~is_lid
    return {
        "xy_int": xy[~is_boundary].astype(np.float32),
        "xy_lid": xy[is_lid].astype(np.float32),
        "xy_wall": xy[is_wall].astype(np.float32),
        "xy_center": np.array([[0.5, 0.5]], dtype=np.float32),
    }


def kovasznay_exact_np(x, y):
    lam = LAMBDA_KOV
    u = 1.0 - np.exp(lam * x) * np.cos(2.0 * math.pi * y)
    v = (lam / (2.0 * math.pi)) * np.exp(lam * x) * np.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * x))
    return u, v, p


def kovasznay_points(N_grid):
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5
    x_ref = _chebyshev_points(N_grid)
    x_phys = x0 + Lx * 0.5 * (x_ref + 1.0)
    y_phys = y0 + Ly * 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, y_phys, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    eps = 1e-10
    xc, yc = xy[:, 0], xy[:, 1]
    is_boundary = ((xc < x0 + eps) | (xc > x0 + Lx - eps) |
                   (yc < y0 + eps) | (yc > y0 + Ly - eps))
    xy_int = xy[~is_boundary].astype(np.float32)
    xy_bc = xy[is_boundary].astype(np.float32)
    u_ex, v_ex, p_ex = kovasznay_exact_np(xy_bc[:, 0], xy_bc[:, 1])
    bc_target = np.stack([u_ex, v_ex, p_ex], axis=1).astype(np.float32)
    x_center = np.array([[-0.5 + Lx / 2, -0.5 + Ly / 2]], dtype=np.float32)
    _, _, p_ctr = kovasznay_exact_np(x_center[:, 0], x_center[:, 1])
    return {
        "xy_int": xy_int,
        "xy_bc": xy_bc,
        "bc_target": bc_target,
        "xy_center": x_center,
        "p_center_exact": float(p_ctr[0]),
    }


def elasticity_body_forces_np(x, y):
    pi = math.pi
    ux_xx = -(2 * pi) ** 2 * np.cos(2 * pi * x) * np.sin(pi * y)
    ux_yy = -(pi ** 2) * np.cos(2 * pi * x) * np.sin(pi * y)
    ux_xy = -2 * pi ** 2 * np.sin(2 * pi * x) * np.cos(pi * y)
    uy_xx = -(pi ** 2) * np.sin(pi * x) * Q_E * y ** 4 / 4.0
    uy_yy = np.sin(pi * x) * Q_E * 3.0 * y ** 2
    uy_xy = pi * np.cos(pi * x) * Q_E * y ** 3
    fx = -((LAM_E + 2 * MU_E) * ux_xx + MU_E * ux_yy + (LAM_E + MU_E) * uy_xy)
    fy = -(MU_E * uy_xx + (LAM_E + 2 * MU_E) * uy_yy + (LAM_E + MU_E) * ux_xy)
    return fx, fy


def elasticity_exact_np(x, y):
    pi = math.pi
    ux = np.cos(2 * pi * x) * np.sin(pi * y)
    uy = np.sin(pi * x) * Q_E * y ** 4 / 4.0
    return ux, uy


def elasticity_points(N_grid):
    x_ref = _chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    eps = 1e-10
    xc, yc = xy[:, 0], xy[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    xy_int = xy[~is_boundary].astype(np.float32)
    xy_bc = xy[is_boundary].astype(np.float32)
    ux_ex, uy_ex = elasticity_exact_np(xy_bc[:, 0], xy_bc[:, 1])
    bc_target = np.stack([ux_ex, uy_ex], axis=1).astype(np.float32)
    fx_int, fy_int = elasticity_body_forces_np(xy_int[:, 0], xy_int[:, 1])
    return {
        "xy_int": xy_int,
        "xy_bc": xy_bc,
        "bc_target": bc_target,
        "fx_int": fx_int.astype(np.float32),
        "fy_int": fy_int.astype(np.float32),
    }


# ----------------------------------------------------------------------------
# PDE residual builders.
#
# Two strategies are used:
#
#  (a) `_cavity_residuals` — per-point vmap(jacfwd) because the Smagorinsky
#      viscous term requires the divergence of a field that itself depends on
#      gradients of (u, v). The 2D input dimension keeps jacfwd competitive.
#
#  (b) `_kovasznay_residuals`, `_elasticity_residuals` — per-point vmap with
#      `jax.hessian`. Second derivatives via the Hessian are cleaner than
#      nested jax.grad and compile to a single XLA computation.
#
# All three strategies use per-point computation that `vmap` replicates across
# the full collocation grid. Each outer call goes through ONE `jax.value_and_grad`
# inside the JIT-compiled training step, so XLA can fuse everything.
# ----------------------------------------------------------------------------
def _per_point(apply_fn):
    """Per-point wrapper: accept a (2,) vector, return (out_dim,)."""
    def f(params, xy):
        return apply_fn(params, xy)
    return f


def _cavity_residuals(apply_fn):
    f = _per_point(apply_fn)

    def neural(params, xy):
        return f(params, xy)  # (3,)

    def visc_flux(params, xy):
        J = jax.jacfwd(neural, argnums=1)(params, xy)  # (3, 2)
        du_dx, du_dy = J[0, 0], J[0, 1]
        dv_dx, dv_dy = J[1, 0], J[1, 1]
        x, y = xy[0], xy[1]
        d = jnp.minimum(jnp.minimum(x, 1.0 - x), jnp.minimum(y, 1.0 - y))
        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_mag = jnp.sqrt(2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2) + 1e-12)
        nu_eff = NU_LAM + (CS * d) ** 2 * S_mag
        return jnp.stack([nu_eff * du_dx, nu_eff * du_dy,
                          nu_eff * dv_dx, nu_eff * dv_dy])

    def residual_single(params, xy):
        pred = neural(params, xy)
        u, v = pred[0], pred[1]
        J = jax.jacfwd(neural, argnums=1)(params, xy)  # (3, 2)
        du_dx, du_dy = J[0, 0], J[0, 1]
        dv_dx, dv_dy = J[1, 0], J[1, 1]
        dp_dx, dp_dy = J[2, 0], J[2, 1]
        Jflux = jax.jacfwd(visc_flux, argnums=1)(params, xy)  # (4, 2)
        visc_u = Jflux[0, 0] + Jflux[1, 1]
        visc_v = Jflux[2, 0] + Jflux[3, 1]
        cont = du_dx + dv_dy
        mom_u = u * du_dx + v * du_dy + dp_dx - visc_u
        mom_v = u * dv_dx + v * dv_dy + dp_dy - visc_v
        return jnp.stack([cont, mom_u, mom_v])

    return vmap(residual_single, in_axes=(None, 0))


def _kovasznay_residuals(apply_fn):
    f = _per_point(apply_fn)

    def u_fn(params, xy): return f(params, xy)[0]
    def v_fn(params, xy): return f(params, xy)[1]
    def p_fn(params, xy): return f(params, xy)[2]

    def residual_single(params, xy):
        u = u_fn(params, xy); v = v_fn(params, xy)
        gu = jax.grad(u_fn, argnums=1)(params, xy)
        gv = jax.grad(v_fn, argnums=1)(params, xy)
        gp = jax.grad(p_fn, argnums=1)(params, xy)
        du_dx, du_dy = gu[0], gu[1]
        dv_dx, dv_dy = gv[0], gv[1]
        dp_dx, dp_dy = gp[0], gp[1]
        hu = jax.hessian(u_fn, argnums=1)(params, xy)  # (2,2)
        hv = jax.hessian(v_fn, argnums=1)(params, xy)
        lap_u = hu[0, 0] + hu[1, 1]
        lap_v = hv[0, 0] + hv[1, 1]
        cont = du_dx + dv_dy
        mom_u = u * du_dx + v * du_dy + dp_dx - NU_KOV * lap_u
        mom_v = u * dv_dx + v * dv_dy + dp_dy - NU_KOV * lap_v
        return jnp.stack([cont, mom_u, mom_v])

    return vmap(residual_single, in_axes=(None, 0))


def _elasticity_residuals(apply_fn):
    f = _per_point(apply_fn)

    def ux_fn(params, xy): return f(params, xy)[0]
    def uy_fn(params, xy): return f(params, xy)[1]

    def residual_single(params, xy, fx, fy):
        hx = jax.hessian(ux_fn, argnums=1)(params, xy)  # (2,2)
        hy = jax.hessian(uy_fn, argnums=1)(params, xy)
        d2ux_dx2 = hx[0, 0]
        d2ux_dy2 = hx[1, 1]
        d2ux_dxdy = hx[0, 1]
        d2uy_dx2 = hy[0, 0]
        d2uy_dy2 = hy[1, 1]
        d2uy_dxdy = hy[0, 1]
        eq_x = ((LAM_E + 2 * MU_E) * d2ux_dx2 + MU_E * d2ux_dy2
                + (LAM_E + MU_E) * d2uy_dxdy + fx)
        eq_y = (MU_E * d2uy_dx2 + (LAM_E + 2 * MU_E) * d2uy_dy2
                + (LAM_E + MU_E) * d2ux_dxdy + fy)
        return jnp.stack([eq_x, eq_y])

    return vmap(residual_single, in_axes=(None, 0, 0, 0))


# ----------------------------------------------------------------------------
# Loss functions per problem — normalization matches PyTorch nn.MSELoss() which
# divides by numel(). PyTorch code uses per-component MSE and sums them, which
# for a (N, k) tensor equals (1/N) * sum_components mean_over_samples(sq).
# ----------------------------------------------------------------------------
def _mse_scalar(arr):
    return jnp.mean(arr ** 2)


def make_cavity_train_step(model, lr: float):
    apply_fn = model.apply
    res_fn = _cavity_residuals(apply_fn)

    def loss_fn(params, xy_int, xy_lid, xy_wall, xy_center):
        r = res_fn(params, xy_int)  # (N_int, 3)
        loss_pde = _mse_scalar(r[:, 0]) + _mse_scalar(r[:, 1]) + _mse_scalar(r[:, 2])
        pred_lid = apply_fn(params, xy_lid)
        loss_lid = _mse_scalar(pred_lid[:, 0] - 1.0) + _mse_scalar(pred_lid[:, 1])
        pred_wall = apply_fn(params, xy_wall)
        loss_wall = _mse_scalar(pred_wall[:, 0]) + _mse_scalar(pred_wall[:, 1])
        pred_c = apply_fn(params, xy_center)
        loss_p = _mse_scalar(pred_c[:, 2])
        return loss_pde + loss_lid + loss_wall + loss_p

    optimizer = optax.adam(lr)

    @jit
    def train_step(params, opt_state, xy_int, xy_lid, xy_wall, xy_center):
        loss, grads = jax.value_and_grad(loss_fn)(params, xy_int, xy_lid, xy_wall, xy_center)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step, optimizer


def make_kovasznay_train_step(model, lr: float):
    apply_fn = model.apply
    res_fn = _kovasznay_residuals(apply_fn)

    def loss_fn(params, xy_int, xy_bc, bc_target, xy_center, p_center):
        r = res_fn(params, xy_int)
        loss_pde = _mse_scalar(r[:, 0]) + _mse_scalar(r[:, 1]) + _mse_scalar(r[:, 2])
        pred_bc = apply_fn(params, xy_bc)
        # PyTorch: mse(pred_bc, bc_target) — combined mean over all elements
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        pred_c = apply_fn(params, xy_center)
        loss_pc = _mse_scalar(pred_c[:, 2] - p_center)
        return loss_pde + loss_bc + loss_pc

    optimizer = optax.adam(lr)

    @jit
    def train_step(params, opt_state, xy_int, xy_bc, bc_target, xy_center, p_center):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, xy_int, xy_bc, bc_target, xy_center, p_center)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step, optimizer


def make_elasticity_train_step(model, lr: float, n_epochs: int):
    apply_fn = model.apply
    res_fn = _elasticity_residuals(apply_fn)

    def loss_fn(params, xy_int, fx_int, fy_int, xy_bc, bc_target):
        r = res_fn(params, xy_int, fx_int, fy_int)
        loss_pde = _mse_scalar(r[:, 0]) + _mse_scalar(r[:, 1])
        pred_bc = apply_fn(params, xy_bc)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        return loss_pde + loss_bc

    # Cosine schedule to match PyTorch train_autodiff_elasticity (which uses
    # CosineAnnealingLR to eta_min=1e-5).
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs, alpha=1e-5 / lr)
    optimizer = optax.adam(learning_rate=schedule)

    @jit
    def train_step(params, opt_state, xy_int, fx_int, fy_int, xy_bc, bc_target):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, xy_int, fx_int, fy_int, xy_bc, bc_target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step, optimizer


# ----------------------------------------------------------------------------
# Top-level entry points called from src/lid_benchmark.py
# ----------------------------------------------------------------------------
def _init_model(model_name: str, out_dim: int, seed: int):
    net = make_jax_model(model_name, out_dim=out_dim)
    key = random.PRNGKey(seed)
    dummy = jnp.zeros((1, 2), dtype=jnp.float32)
    params = net.init(key, dummy)
    return net, params


def _to_jax(np_arr):
    return jnp.asarray(np_arr)


def train_jaxpinn_cavity(seed: int, device: torch.device, n_epochs: int, lr: float,
                         grid_size: int, model_name: str):
    """Train a JAX+JIT PINN on the lid-driven cavity. Returns (torch_model, train_time_s, final_loss)."""
    pts = cavity_points(grid_size)
    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [jaxpinn] Model params: {n_params} ({model_name})")

    train_step, optimizer = make_cavity_train_step(net, lr=lr)
    opt_state = optimizer.init(params)

    xy_int = _to_jax(pts["xy_int"])
    xy_lid = _to_jax(pts["xy_lid"])
    xy_wall = _to_jax(pts["xy_wall"])
    xy_center = _to_jax(pts["xy_center"])

    # Warmup (JIT compile) — not counted in training time, mirroring CONTEXT.md guidance.
    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state, xy_int, xy_lid, xy_wall, xy_center)
    loss.block_until_ready()
    print(f"  [jaxpinn] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):  # already did one step in warmup
        params, opt_state, loss = train_step(params, opt_state, xy_int, xy_lid, xy_wall, xy_center)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


def train_jaxpinn_kovasznay(seed: int, device: torch.device, n_epochs: int, lr: float,
                            grid_size: int, model_name: str):
    pts = kovasznay_points(grid_size)
    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [jaxpinn] Model params: {n_params} ({model_name})")

    train_step, optimizer = make_kovasznay_train_step(net, lr=lr)
    opt_state = optimizer.init(params)

    xy_int = _to_jax(pts["xy_int"])
    xy_bc = _to_jax(pts["xy_bc"])
    bc_target = _to_jax(pts["bc_target"])
    xy_center = _to_jax(pts["xy_center"])
    p_center = jnp.float32(pts["p_center_exact"])

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(
        params, opt_state, xy_int, xy_bc, bc_target, xy_center, p_center)
    loss.block_until_ready()
    print(f"  [jaxpinn] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(
            params, opt_state, xy_int, xy_bc, bc_target, xy_center, p_center)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


def train_jaxpinn_elasticity(seed: int, device: torch.device, n_epochs: int, lr: float,
                             grid_size: int, model_name: str):
    pts = elasticity_points(grid_size)
    net, params = _init_model(model_name, out_dim=2, seed=seed)
    n_params = count_params(params)
    print(f"  [jaxpinn] Model params: {n_params} ({model_name})")

    train_step, optimizer = make_elasticity_train_step(net, lr=lr, n_epochs=n_epochs)
    opt_state = optimizer.init(params)

    xy_int = _to_jax(pts["xy_int"])
    fx_int = _to_jax(pts["fx_int"])
    fy_int = _to_jax(pts["fy_int"])
    xy_bc = _to_jax(pts["xy_bc"])
    bc_target = _to_jax(pts["bc_target"])

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(
        params, opt_state, xy_int, fx_int, fy_int, xy_bc, bc_target)
    loss.block_until_ready()
    print(f"  [jaxpinn] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(
            params, opt_state, xy_int, fx_int, fy_int, xy_bc, bc_target)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=2, device=device)
    return torch_model, train_time, final_loss_val


# ----------------------------------------------------------------------------
# SAGE-JAX: explicit pre-accumulated PDE backward generated by symbolic_vjp
# ----------------------------------------------------------------------------
#
# This path runs the SAME Chebyshev-spectral PDE residual forward and backward
# as `train_sage_*` in src/lid_benchmark.py, but executes them in JAX rather
# than PyTorch, so the entire training step (network forward, SAGE PDE
# backward, BC adjoint, network backprop via `jax.vjp`, and optax Adam update)
# fuses into one `@jit` XLA program. The generated SAGE backward uses
# `jnp`-native matmul/mul/sqrt etc. emitted by src/symbolic_vjp.py with
# backend="jax".
# ----------------------------------------------------------------------------

def _torch_g_to_jax(g_torch, keys):
    """Convert a torch grid_data dict into a jnp dict, keeping only needed keys
    (so jit traces a minimal pytree). Scalar ints are kept as Python ints so they
    become static via closure."""
    g_jax = {}
    for k in keys:
        v = g_torch[k]
        if torch.is_tensor(v):
            g_jax[k] = jnp.asarray(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            g_jax[k] = jnp.asarray(v)
        else:
            g_jax[k] = v
    return g_jax


def _reparam_kovasznay_grid_(g_torch, re_param: float):
    """Rebuild Kov BC targets + pressure center for a new Reynolds number.

    The Kovasznay exact solution has a Re-dependent parameter
    lambda = Re/2 - sqrt(Re^2/4 + 4*pi^2). Changing Re changes lambda,
    which changes u_ex, v_ex, p_ex used as Dirichlet BCs.
    """
    lam = re_param / 2.0 - math.sqrt(re_param ** 2 / 4.0 + 4.0 * math.pi ** 2)
    x_bc = g_torch['xy_bc'][:, 0:1]
    y_bc = g_torch['xy_bc'][:, 1:2]
    u_ex = 1.0 - torch.exp(lam * x_bc) * torch.cos(2.0 * math.pi * y_bc)
    v_ex = (lam / (2.0 * math.pi)) * torch.exp(lam * x_bc) * torch.sin(2.0 * math.pi * y_bc)
    p_ex = 0.5 * (1.0 - torch.exp(2.0 * lam * x_bc))
    g_torch['bc_target'] = torch.cat([u_ex, v_ex, p_ex], dim=1)
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5
    xc = torch.tensor([[x0 + Lx / 2]], dtype=torch.float32, device=g_torch['xy_bc'].device)
    yc = torch.tensor([[y0 + Ly / 2]], dtype=torch.float32, device=g_torch['xy_bc'].device)
    p_ctr = 0.5 * (1.0 - torch.exp(2.0 * lam * xc))
    g_torch['p_center_exact'] = float(p_ctr.item())
    g_torch['nu_kov'] = float(1.0 / re_param)
    return g_torch


def _reparam_elasticity_grid_(g_torch, E_ratio: float, nu_poisson: float):
    """Rebuild elasticity body forces + BC targets for new (E/E_0, nu).

    Reference state (E_0, nu_0) is the legacy configuration with lam_e=1.0,
    mu_e=0.5, i.e. E_0 = mu(3*lam+2*mu)/(lam+mu) = 0.5*(3+1)/(1+0.5) = 4/3,
    nu_0 = lam/(2*(lam+mu)) = 1/3. The parametric family uses:
        E = E_ratio * E_0
        nu = nu_poisson  (absolute, in [0.2, 0.45])
        mu = E / (2*(1+nu))
        lam = E*nu / ((1+nu)*(1-2*nu))

    Body forces fx, fy follow from applying Navier-Cauchy to the same
    manufactured solution under new (lam, mu); BC targets are unchanged
    (the manufactured solution is fixed; only material params vary).
    """
    E_0 = 4.0 / 3.0
    E = E_ratio * E_0
    mu_new = E / (2.0 * (1.0 + nu_poisson))
    lam_new = E * nu_poisson / ((1.0 + nu_poisson) * (1.0 - 2.0 * nu_poisson))
    g_torch['lam_e'] = float(lam_new)
    g_torch['mu_e'] = float(mu_new)

    # Recompute body forces for the manufactured solution
    # ux_exact(x,y) = cos(2πx) sin(πy),  uy_exact(x,y) = sin(πx) Q_E y⁴/4
    x = g_torch['xy_all'][:, 0:1]
    y = g_torch['xy_all'][:, 1:2]
    pi = math.pi
    Q_E_val = 4.0
    ux_xx = -(2 * pi) ** 2 * torch.cos(2 * pi * x) * torch.sin(pi * y)
    ux_yy = -(pi ** 2) * torch.cos(2 * pi * x) * torch.sin(pi * y)
    ux_xy = -2 * pi ** 2 * torch.sin(2 * pi * x) * torch.cos(pi * y)
    uy_xx = -(pi ** 2) * torch.sin(pi * x) * Q_E_val * y ** 4 / 4.0
    uy_yy = torch.sin(pi * x) * Q_E_val * 3.0 * y ** 2
    uy_xy = pi * torch.cos(pi * x) * Q_E_val * y ** 3
    fx = -((lam_new + 2 * mu_new) * ux_xx + mu_new * ux_yy + (lam_new + mu_new) * uy_xy)
    fy = -(mu_new * uy_xx + (lam_new + 2 * mu_new) * uy_yy + (lam_new + mu_new) * ux_xy)
    g_torch['fx'] = fx
    g_torch['fy'] = fy
    return g_torch


def _build_sage_backward(problem: str):
    """Lazily import src.symbolic_vjp and produce a jax-backend SAGE backward."""
    from src.symbolic_vjp import generate_backward
    _, fn = generate_backward(sparse=False, problem=problem, backend="jax")
    return fn


def _build_bfsa_backward(problem: str):
    """Build a BFSA backward: Kronecker-structured transpose matmuls."""
    from src.symbolic_vjp import generate_backward
    _, fn = generate_backward(sparse=False, problem=problem, backend="jax",
                              kronecker=True)
    return fn


def _enrich_g_jax_with_1d_matrices(g_jax, N_grid, Lx=1.0, Ly=1.0):
    """Add per-axis 1D Chebyshev diff matrices to g_jax for BFSA Kronecker ops.

    For a domain of size Lx × Ly, the physical 1D derivative matrices are:
        Dx_1d = D1d_ref * (2/Lx)
        Dy_1d = D1d_ref * (2/Ly)
    where D1d_ref is the [-1,1] Chebyshev diff matrix.
    """
    from src.lid_benchmark import chebyshev_diff_matrix
    D1d_ref = chebyshev_diff_matrix(N_grid)
    D1d_x = jnp.asarray((D1d_ref * (2.0 / Lx)).astype(np.float32))
    D1d_y = jnp.asarray((D1d_ref * (2.0 / Ly)).astype(np.float32))
    g_jax['D1d_x'] = D1d_x
    g_jax['D1dT_x'] = D1d_x.T
    g_jax['D1d_y'] = D1d_y
    g_jax['D1dT_y'] = D1d_y.T
    g_jax['D1d_sq_x'] = D1d_x @ D1d_x
    g_jax['D1dT_sq_x'] = D1d_x.T @ D1d_x.T
    g_jax['D1d_sq_y'] = D1d_y @ D1d_y
    g_jax['D1dT_sq_y'] = D1d_y.T @ D1d_y.T
    g_jax['N_grid'] = N_grid
    return g_jax


# ---------- Cavity ----------
def _compute_pde_cavity_jax(pred, g):
    """Mirror of src.lid_benchmark.compute_pde_terms, in jnp.

    Laminar viscosity is read from g['nu_lam'] if threaded (Program-B F1),
    otherwise falls back to the module-level NU_LAM (Re=1000 legacy).
    """
    nu_lam = g['nu_lam'] if 'nu_lam' in g else NU_LAM
    u = pred[:, 0:1]; v = pred[:, 1:2]; p = pred[:, 2:3]
    Dx, Dy = g['Dx'], g['Dy']
    du_dx = Dx @ u; du_dy = Dy @ u
    dv_dx = Dx @ v; dv_dy = Dy @ v
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = jnp.sqrt(2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2) + 1e-12)
    nu_eff = nu_lam + g['Cs_d_sq'] * S_mag
    cont = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    dp_dx = Dx @ p; dp_dy = Dy @ p
    visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
    visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return cont, mom_u, mom_v


def _make_sage_jax_cavity_train_step(model, lr: float, g_jax, sage_backward,
                                     N_all, off_lid, off_wall, off_center,
                                     N_lid, N_wall, M):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    interior_mask = g_jax['interior_mask']  # shape (N_all, 1)
    optimizer = optax.adam(lr)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)

        pred_pde = pred_batch[:N_all]
        pred_lid = pred_batch[off_lid:off_wall]
        pred_wall = pred_batch[off_wall:off_center]
        pred_c = pred_batch[off_center:]

        # PDE adjoint via SAGE-generated JAX backward (closes over g_jax matrices)
        grad_pde = sage_backward(pred_pde, g_jax)

        # BC adjoints — identical formulas to src/lid_benchmark.py train_sage
        gl0 = 2.0 * (pred_lid[:, 0:1] - 1.0) / N_lid
        gl1 = 2.0 * pred_lid[:, 1:2] / N_lid
        gl2 = jnp.zeros_like(pred_lid[:, 2:3])
        grad_lid = jnp.concatenate([gl0, gl1, gl2], axis=1)

        gw0 = 2.0 * pred_wall[:, 0:1] / N_wall
        gw1 = 2.0 * pred_wall[:, 1:2] / N_wall
        gw2 = jnp.zeros_like(pred_wall[:, 2:3])
        grad_wall = jnp.concatenate([gw0, gw1, gw2], axis=1)

        gc0 = jnp.zeros_like(pred_c[:, 0:1])
        gc1 = jnp.zeros_like(pred_c[:, 1:2])
        gc2 = 2.0 * pred_c[:, 2:3]
        grad_center = jnp.concatenate([gc0, gc1, gc2], axis=1)

        upstream = jnp.concatenate(
            [grad_pde, grad_lid, grad_wall, grad_center], axis=0)
        (param_grads,) = vjp_fn(upstream)

        # Loss value for logging — matches PyTorch (masked-mean over interior).
        cont, mom_u, mom_v = _compute_pde_cavity_jax(pred_pde, g_jax)
        loss_pde = (jnp.sum(cont ** 2 * interior_mask) / M
                    + jnp.sum(mom_u ** 2 * interior_mask) / M
                    + jnp.sum(mom_v ** 2 * interior_mask) / M)
        loss_lid = jnp.mean((pred_lid[:, 0:1] - 1.0) ** 2) + jnp.mean(pred_lid[:, 1:2] ** 2)
        loss_wall = jnp.mean(pred_wall[:, 0:1] ** 2) + jnp.mean(pred_wall[:, 1:2] ** 2)
        loss_p = jnp.mean(pred_c[:, 2:3] ** 2)
        loss_val = loss_pde + loss_lid + loss_wall + loss_p
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def train_sage_jax_cavity(seed: int, device: torch.device, n_epochs: int, lr: float,
                          grid_size: int, model_name: str):
    """Train a SAGE-JAX PINN on the lid-driven cavity.

    Uses the Chebyshev-spectral backward generated by symbolic_vjp.py with
    backend='jax', fused with a `jax.vjp` network backprop inside a single
    `@jit` training step.
    """
    from src.lid_benchmark import build_grid_data
    g_torch = build_grid_data(grid_size, device)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Cs_d_sq', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_lid', 'xy_wall']
    g_jax = _torch_g_to_jax(g_torch, keys)
    # Static ints stay as Python ints (go through closure, not pytree)
    N_all = int(g_torch['N_all'])
    N_lid = int(g_torch['N_lid'])
    N_wall = int(g_torch['N_wall'])
    M = int(g_torch['M'])
    off_lid = int(g_torch['off_lid'])
    off_wall = int(g_torch['off_wall'])
    off_center = int(g_torch['off_center'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [sage-jax] Model params: {n_params} ({model_name})")

    sage_backward = _build_sage_backward('cavity')

    train_step, optimizer = _make_sage_jax_cavity_train_step(
        net, lr, g_jax, sage_backward,
        N_all, off_lid, off_wall, off_center, N_lid, N_wall, M)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [sage-jax] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


# ---------- Kovasznay ----------
def _compute_pde_kovasznay_jax(pred, g):
    """Kovasznay spectral PDE residuals.

    Viscosity read from g['nu_kov'] (Program-B F2 parametric family), falling
    back to module-level NU_KOV (Re=40 legacy) when the key is absent.
    """
    nu = g['nu_kov'] if 'nu_kov' in g else NU_KOV
    u = pred[:, 0:1]; v = pred[:, 1:2]; p = pred[:, 2:3]
    Dx, Dy = g['Dx'], g['Dy']
    du_dx = Dx @ u; du_dy = Dy @ u
    dv_dx = Dx @ v; dv_dy = Dy @ v
    dp_dx = Dx @ p; dp_dy = Dy @ p
    d2u_dx2 = Dx @ du_dx; d2u_dy2 = Dy @ du_dy
    d2v_dx2 = Dx @ dv_dx; d2v_dy2 = Dy @ dv_dy
    cont = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
    return cont, mom_u, mom_v


def _make_sage_jax_kovasznay_train_step(model, lr, g_jax, sage_backward,
                                        N_all, off_bc, off_center, N_bc, M,
                                        p_center_exact):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    bc_target = g_jax['bc_target']
    interior_mask = g_jax['interior_mask']
    optimizer = optax.adam(lr)
    p_center_f = jnp.float32(p_center_exact)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)

        pred_pde = pred_batch[:N_all]
        pred_bc = pred_batch[off_bc:off_center]
        pred_c = pred_batch[off_center:]

        grad_pde = sage_backward(pred_pde, g_jax)

        n_out = 3  # (u, v, p) — matches src/lid_benchmark.py:1947
        grad_bc = 2.0 * (pred_bc - bc_target) / (N_bc * n_out)

        gc_p = 2.0 * (pred_c[:, 2:3] - p_center_f)
        gc_uv = jnp.zeros_like(pred_c[:, 0:2])
        grad_center = jnp.concatenate([gc_uv, gc_p], axis=1)

        upstream = jnp.concatenate([grad_pde, grad_bc, grad_center], axis=0)
        (param_grads,) = vjp_fn(upstream)

        cont, mom_u, mom_v = _compute_pde_kovasznay_jax(pred_pde, g_jax)
        loss_pde = (jnp.sum(cont ** 2 * interior_mask) / M
                    + jnp.sum(mom_u ** 2 * interior_mask) / M
                    + jnp.sum(mom_v ** 2 * interior_mask) / M)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        loss_p = jnp.mean((pred_c[:, 2:3] - p_center_f) ** 2)
        loss_val = loss_pde + loss_bc + loss_p
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def train_sage_jax_kovasznay(seed: int, device: torch.device, n_epochs: int, lr: float,
                             grid_size: int, model_name: str):
    from src.lid_benchmark import build_grid_data_kovasznay
    g_torch = build_grid_data_kovasznay(grid_size, device)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    off_center = int(g_torch['off_center'])
    p_center_exact = float(g_torch['p_center_exact'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [sage-jax] Model params: {n_params} ({model_name})")

    sage_backward = _build_sage_backward('kovasznay')

    train_step, optimizer = _make_sage_jax_kovasznay_train_step(
        net, lr, g_jax, sage_backward,
        N_all, off_bc, off_center, N_bc, M, p_center_exact)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [sage-jax] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


# ---------- Elasticity ----------
def _compute_pde_elasticity_jax(pred, g):
    """Elasticity Navier-Cauchy spectral residuals.

    Lamé constants read from g['lam_e'] / g['mu_e'] (Program-B F3 parametric
    family). Body forces g['fx'], g['fy'] must be regenerated per-instance
    to stay consistent with the manufactured solution under varying (E, nu).
    """
    lam = g['lam_e'] if 'lam_e' in g else LAM_E
    mu = g['mu_e'] if 'mu_e' in g else MU_E
    ux = pred[:, 0:1]; uy = pred[:, 1:2]
    Dxx, Dyy, Dxy = g['Dxx'], g['Dyy'], g['Dxy']
    d2ux_dx2 = Dxx @ ux
    d2ux_dy2 = Dyy @ ux
    d2uy_dx2 = Dxx @ uy
    d2uy_dy2 = Dyy @ uy
    d2uy_dxdy = Dxy @ uy
    d2ux_dxdy = Dxy @ ux
    eq_x = ((lam + 2 * mu) * d2ux_dx2 + mu * d2ux_dy2
            + (lam + mu) * d2uy_dxdy + g['fx'])
    eq_y = (mu * d2uy_dx2 + (lam + 2 * mu) * d2uy_dy2
            + (lam + mu) * d2ux_dxdy + g['fy'])
    return eq_x, eq_y


def _make_sage_jax_elasticity_train_step(model, lr, n_epochs, g_jax, sage_backward,
                                         N_all, off_bc, N_bc, M):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    bc_target = g_jax['bc_target']
    interior_mask = g_jax['interior_mask']
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs,
                                           alpha=1e-5 / lr)
    optimizer = optax.adam(learning_rate=schedule)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)

        pred_pde = pred_batch[:N_all]
        pred_bc = pred_batch[off_bc:off_bc + N_bc]

        grad_pde = sage_backward(pred_pde, g_jax)

        n_out = 2  # (ux, uy)
        grad_bc = 2.0 * (pred_bc - bc_target) / (N_bc * n_out)

        upstream = jnp.concatenate([grad_pde, grad_bc], axis=0)
        (param_grads,) = vjp_fn(upstream)

        eq_x, eq_y = _compute_pde_elasticity_jax(pred_pde, g_jax)
        loss_pde = (jnp.sum(eq_x ** 2 * interior_mask) / M
                    + jnp.sum(eq_y ** 2 * interior_mask) / M)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        loss_val = loss_pde + loss_bc
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def train_sage_jax_elasticity(seed: int, device: torch.device, n_epochs: int, lr: float,
                              grid_size: int, model_name: str):
    from src.lid_benchmark import build_grid_data_elasticity
    g_torch = build_grid_data_elasticity(grid_size, device)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Dxx', 'Dyy', 'Dxy',
            'DxxT', 'DyyT', 'DxyT', 'interior_mask', 'fx', 'fy',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M

    net, params = _init_model(model_name, out_dim=2, seed=seed)
    n_params = count_params(params)
    print(f"  [sage-jax] Model params: {n_params} ({model_name})")

    sage_backward = _build_sage_backward('elasticity')

    train_step, optimizer = _make_sage_jax_elasticity_train_step(
        net, lr, n_epochs, g_jax, sage_backward, N_all, off_bc, N_bc, M)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [sage-jax] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=2, device=device)
    return torch_model, train_time, final_loss_val


# ============================================================================
# SLRM-JAX: Static Linear Residual Map surrogate gradient, JAX backend
#
# Research phase 5 artefact — see llmdocs/research/research_log/04_design.md.
# At problem setup, materialise the residual DAG's Jacobian at a fixed
# reference input as a single constant matrix M_ref, then every jit-fused
# training step multiplies the current residual tensor by M_ref once. No
# chain-rule replay through the residual DAG ever happens inside the loop.
# ============================================================================

def _slrm_build_M_jax(residual_fn, pred_ref):
    """Build the (N*K, N*k) static linear map via forward-mode Jacobian.

    residual_fn : callable (N,K) -> (N,k) written in jnp.
    pred_ref    : (N,K) jnp array at which to linearise.
    """
    J = jax.jacfwd(residual_fn)(pred_ref)  # (N, k, N, K)
    N, k, _, K = J.shape
    J_flat = J.reshape(N * k, N * K)
    return J_flat.T  # (N*K, N*k)


# ============================================================================
# BFSA: Butterfly-Factored Spectral Adjoint (Kronecker-structured backward)
# ============================================================================

def train_bfsa_cavity(seed: int, device: torch.device, n_epochs: int, lr: float,
                      grid_size: int, model_name: str, re_param: float = None):
    """BFSA cavity: exact forward + Kronecker-structured backward.

    If re_param is None, uses the legacy Re=1000 cavity. Otherwise sets
    nu_lam = U_lid / re_param and threads it through g_jax for Program-B F1.
    """
    from src.lid_benchmark import build_grid_data, U_lid
    g_torch = build_grid_data(grid_size, device)
    if re_param is not None:
        g_torch['nu_lam'] = float(U_lid / re_param)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Cs_d_sq', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_lid', 'xy_wall', 'nu_lam']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_lid = int(g_torch['N_lid'])
    N_wall = int(g_torch['N_wall'])
    M = int(g_torch['M'])
    off_lid = int(g_torch['off_lid'])
    off_wall = int(g_torch['off_wall'])
    off_center = int(g_torch['off_center'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M
    g_jax = _enrich_g_jax_with_1d_matrices(g_jax, grid_size, Lx=1.0, Ly=1.0)

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [bfsa] Model params: {n_params} ({model_name})")

    bfsa_backward = _build_bfsa_backward('cavity')

    train_step, optimizer = _make_sage_jax_cavity_train_step(
        net, lr, g_jax, bfsa_backward,
        N_all, off_lid, off_wall, off_center, N_lid, N_wall, M)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [bfsa] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


def train_bfsa_kovasznay(seed: int, device: torch.device, n_epochs: int, lr: float,
                         grid_size: int, model_name: str, re_param: float = None):
    """BFSA kovasznay: exact forward + Kronecker-structured backward.

    If re_param is None, uses the legacy Re=40 Kovasznay. Otherwise sets
    nu_kov = 1/re_param AND regenerates the BC exact solution + pressure
    center for the new Re (Program-B F2).
    """
    from src.lid_benchmark import build_grid_data_kovasznay
    g_torch = build_grid_data_kovasznay(grid_size, device)
    if re_param is not None:
        g_torch = _reparam_kovasznay_grid_(g_torch, float(re_param))
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target', 'nu_kov']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    off_center = int(g_torch['off_center'])
    p_center_exact = float(g_torch['p_center_exact'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M
    g_jax = _enrich_g_jax_with_1d_matrices(g_jax, grid_size, Lx=1.5, Ly=2.0)

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [bfsa] Model params: {n_params} ({model_name})")

    bfsa_backward = _build_bfsa_backward('kovasznay')

    train_step, optimizer = _make_sage_jax_kovasznay_train_step(
        net, lr, g_jax, bfsa_backward,
        N_all, off_bc, off_center, N_bc, M, p_center_exact)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [bfsa] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


def train_bfsa_elasticity(seed: int, device: torch.device, n_epochs: int, lr: float,
                          grid_size: int, model_name: str,
                          E_ratio: float = None, nu_poisson: float = None):
    """BFSA elasticity: exact forward + Kronecker-structured backward.

    If (E_ratio, nu_poisson) is None, uses the legacy E_0=1.5, nu=1/3
    elasticity (lam_e=1, mu_e=0.5). Otherwise computes Lamé constants
    from (E = E_ratio * E_0, nu = nu_poisson) and regenerates body
    forces fx, fy + BC targets consistent with the manufactured solution
    under the new (lam, mu) (Program-B F3).
    """
    from src.lid_benchmark import build_grid_data_elasticity
    g_torch = build_grid_data_elasticity(grid_size, device)
    if E_ratio is not None or nu_poisson is not None:
        assert E_ratio is not None and nu_poisson is not None, \
            "E_ratio and nu_poisson must both be set or both be None"
        g_torch = _reparam_elasticity_grid_(g_torch, float(E_ratio), float(nu_poisson))
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Dxx', 'Dyy', 'Dxy',
            'DxxT', 'DyyT', 'DxyT', 'interior_mask', 'fx', 'fy',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target', 'lam_e', 'mu_e']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M
    g_jax = _enrich_g_jax_with_1d_matrices(g_jax, grid_size, Lx=1.0, Ly=1.0)

    net, params = _init_model(model_name, out_dim=2, seed=seed)
    n_params = count_params(params)
    print(f"  [bfsa] Model params: {n_params} ({model_name})")

    bfsa_backward = _build_bfsa_backward('elasticity')

    train_step, optimizer = _make_sage_jax_elasticity_train_step(
        net, lr, n_epochs, g_jax, bfsa_backward, N_all, off_bc, N_bc, M)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [bfsa] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=2, device=device)
    return torch_model, train_time, final_loss_val


# ============================================================================
# SDCCG: Spectral Defect-Corrected Coarse Gradient
#
# Compute PDE upstream gradient on a coarse (m×m) Chebyshev grid every step;
# every k-th step (anchor) also compute fine-grid (n×n) gradient and cache
# the spectral defect. Add cached defect to coarse gradient between anchors.
# ============================================================================

def _chebyshev_interp_1d(n_fine, n_coarse):
    """Barycentric Chebyshev interpolation matrix from n_fine to n_coarse points.
    Returns numpy (n_coarse, n_fine) array.
    """
    x_f = np.cos(np.pi * np.arange(n_fine) / (n_fine - 1))
    x_c = np.cos(np.pi * np.arange(n_coarse) / (n_coarse - 1))
    w = np.ones(n_fine)
    w[0] = 0.5
    for j in range(1, n_fine):
        w[j] = (-1) ** j
    w[-1] = 0.5 * ((-1) ** (n_fine - 1))
    P = np.zeros((n_coarse, n_fine))
    for i in range(n_coarse):
        diffs = np.abs(x_c[i] - x_f)
        match = np.where(diffs < 1e-14)[0]
        if len(match) > 0:
            P[i, match[0]] = 1.0
        else:
            tmp = w / (x_c[i] - x_f)
            P[i, :] = tmp / np.sum(tmp)
    return P


def _restrict_2d(g_fine, R_1d, n_fine, n_coarse):
    """Restrict 2D field from fine to coarse grid via Kronecker R_1d ⊗ R_1d.
    g_fine: (n_fine², K), R_1d: (n_coarse, n_fine) JAX array.
    Returns: (n_coarse², K).
    """
    K = g_fine.shape[1]
    G = g_fine.reshape(n_fine, n_fine, K).transpose(2, 0, 1)
    tmp = G @ R_1d.T
    out = R_1d @ tmp
    return out.transpose(1, 2, 0).reshape(n_coarse * n_coarse, K)


# ---------- SDCCG Cavity ----------

def _make_sdccg_cavity_steps(model, lr, g_f, g_c, bfsa_bwd, R_1d,
                              nf, nc,
                              Na_f, ol_f, ow_f, oc_f, Nl_f, Nw_f, M_f,
                              Na_c, ol_c, ow_c, oc_c, Nl_c, Nw_c, M_c):
    apply_fn = model.apply
    xy_f = g_f['xy_batched']
    xy_c = g_c['xy_batched']
    imask_c = g_c['interior_mask']
    optimizer = optax.adam(lr)

    def _coarse_lag(params, defect):
        def fwd(p):
            return apply_fn(p, xy_c)
        pred, vjp_fn = jax.vjp(fwd, params)
        pp = pred[:Na_c]
        pl = pred[ol_c:ow_c]
        pw = pred[ow_c:oc_c]
        pc = pred[oc_c:]
        gp = bfsa_bwd(pp, g_c) + defect
        gl = jnp.concatenate([2.0*(pl[:,0:1]-1.0)/Nl_c,
                              2.0*pl[:,1:2]/Nl_c,
                              jnp.zeros_like(pl[:,2:3])], axis=1)
        gw = jnp.concatenate([2.0*pw[:,0:1]/Nw_c,
                              2.0*pw[:,1:2]/Nw_c,
                              jnp.zeros_like(pw[:,2:3])], axis=1)
        gc = jnp.concatenate([jnp.zeros_like(pc[:,0:1]),
                              jnp.zeros_like(pc[:,1:2]),
                              2.0*pc[:,2:3]], axis=1)
        us = jnp.concatenate([gp, gl, gw, gc], axis=0)
        (pg,) = vjp_fn(us)
        c, mu, mv = _compute_pde_cavity_jax(pp, g_c)
        lp = (jnp.sum(c**2*imask_c)/M_c + jnp.sum(mu**2*imask_c)/M_c
              + jnp.sum(mv**2*imask_c)/M_c)
        ll = jnp.mean((pl[:,0:1]-1.0)**2) + jnp.mean(pl[:,1:2]**2)
        lw = jnp.mean(pw[:,0:1]**2) + jnp.mean(pw[:,1:2]**2)
        lpc = jnp.mean(pc[:,2:3]**2)
        return lp+ll+lw+lpc, pg

    @jit
    def coarse_step(params, opt_state, defect):
        loss, pg = _coarse_lag(params, defect)
        upd, opt_state = optimizer.update(pg, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss

    def _anchor_lag(params, defect):
        # Coarse-grid NN backward (same as coarse step, with current defect)
        def fwd_c(p):
            return apply_fn(p, xy_c)
        pred_c, vjp_fn = jax.vjp(fwd_c, params)
        pp = pred_c[:Na_c]
        pl = pred_c[ol_c:ow_c]
        pw = pred_c[ow_c:oc_c]
        pc = pred_c[oc_c:]
        gpc = bfsa_bwd(pp, g_c)
        gp = gpc + defect
        gl = jnp.concatenate([2.0*(pl[:,0:1]-1.0)/Nl_c,
                              2.0*pl[:,1:2]/Nl_c,
                              jnp.zeros_like(pl[:,2:3])], axis=1)
        gw = jnp.concatenate([2.0*pw[:,0:1]/Nw_c,
                              2.0*pw[:,1:2]/Nw_c,
                              jnp.zeros_like(pw[:,2:3])], axis=1)
        gc = jnp.concatenate([jnp.zeros_like(pc[:,0:1]),
                              jnp.zeros_like(pc[:,1:2]),
                              2.0*pc[:,2:3]], axis=1)
        us = jnp.concatenate([gp, gl, gw, gc], axis=0)
        (pg,) = vjp_fn(us)
        # Fine-grid evaluation for defect update (no vjp needed)
        pred_f = apply_fn(params, xy_f)
        gpf = bfsa_bwd(pred_f[:Na_f], g_f)
        gpf_r = _restrict_2d(gpf, R_1d, nf, nc) * (M_c / M_f)
        new_defect = (gpf_r - gpc) * imask_c
        c, mu, mv = _compute_pde_cavity_jax(pp, g_c)
        lp = (jnp.sum(c**2*imask_c)/M_c + jnp.sum(mu**2*imask_c)/M_c
              + jnp.sum(mv**2*imask_c)/M_c)
        ll = jnp.mean((pl[:,0:1]-1.0)**2) + jnp.mean(pl[:,1:2]**2)
        lw = jnp.mean(pw[:,0:1]**2) + jnp.mean(pw[:,1:2]**2)
        lpc = jnp.mean(pc[:,2:3]**2)
        return lp+ll+lw+lpc, pg, new_defect

    @jit
    def anchor_step(params, opt_state, defect):
        loss, pg, new_defect = _anchor_lag(params, defect)
        upd, opt_state = optimizer.update(pg, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, new_defect

    return coarse_step, anchor_step, optimizer


def train_sdccg_cavity(seed: int, device: torch.device, n_epochs: int, lr: float,
                        grid_size: int, model_name: str,
                        m_coarse: int = 40, k_anchor: int = 10):
    from src.lid_benchmark import build_grid_data
    g_tf = build_grid_data(grid_size, device)
    keys = ['Dx','Dy','DxT','DyT','Cs_d_sq','interior_mask',
            'xy_batched','xy_all','xy_lid','xy_wall']
    g_f = _torch_g_to_jax(g_tf, keys)
    Na_f = int(g_tf['N_all']); Nl_f = int(g_tf['N_lid']); Nw_f = int(g_tf['N_wall'])
    M_f = int(g_tf['M']); ol_f = int(g_tf['off_lid']); ow_f = int(g_tf['off_wall'])
    oc_f = int(g_tf['off_center'])
    g_f['N_all'] = Na_f; g_f['M'] = M_f
    g_f = _enrich_g_jax_with_1d_matrices(g_f, grid_size, Lx=1.0, Ly=1.0)

    g_tc = build_grid_data(m_coarse, device)
    g_c = _torch_g_to_jax(g_tc, keys)
    Na_c = int(g_tc['N_all']); Nl_c = int(g_tc['N_lid']); Nw_c = int(g_tc['N_wall'])
    M_c = int(g_tc['M']); ol_c = int(g_tc['off_lid']); ow_c = int(g_tc['off_wall'])
    oc_c = int(g_tc['off_center'])
    g_c['N_all'] = Na_c; g_c['M'] = M_c
    g_c = _enrich_g_jax_with_1d_matrices(g_c, m_coarse, Lx=1.0, Ly=1.0)

    R_1d = jnp.asarray(_chebyshev_interp_1d(grid_size, m_coarse).astype(np.float32))

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    print(f"  [sdccg] Model params: {count_params(params)} ({model_name})")
    bfsa_bwd = _build_bfsa_backward('cavity')

    coarse_step, anchor_step, optimizer = _make_sdccg_cavity_steps(
        net, lr, g_f, g_c, bfsa_bwd, R_1d, grid_size, m_coarse,
        Na_f, ol_f, ow_f, oc_f, Nl_f, Nw_f, M_f,
        Na_c, ol_c, ow_c, oc_c, Nl_c, Nw_c, M_c)
    opt_state = optimizer.init(params)
    defect = jnp.zeros((Na_c, 3))

    t_w = time.perf_counter()
    params, opt_state, loss, defect = anchor_step(params, opt_state, defect)
    loss.block_until_ready()
    params, opt_state, loss = coarse_step(params, opt_state, defect)
    loss.block_until_ready()
    print(f"  [sdccg] JIT warmup: {time.perf_counter()-t_w:.2f}s, loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(2, n_epochs):
        if epoch % k_anchor == 0:
            params, opt_state, loss, defect = anchor_step(params, opt_state, defect)
        else:
            params, opt_state, loss = coarse_step(params, opt_state, defect)
        if (epoch+1) % 5000 == 0 or (epoch+1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)
    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


# ---------- SDCCG Kovasznay ----------

def _make_sdccg_kovasznay_steps(model, lr, g_f, g_c, bfsa_bwd, R_1d,
                                 nf, nc,
                                 Na_f, ob_f, oc_f, Nb_f, M_f, pc_f,
                                 Na_c, ob_c, oc_c, Nb_c, M_c, pc_c):
    apply_fn = model.apply
    xy_f = g_f['xy_batched']
    xy_c = g_c['xy_batched']
    imask_c = g_c['interior_mask']
    bc_tgt_c = g_c['bc_target']
    optimizer = optax.adam(lr)
    pc_f_j = jnp.float32(pc_f)
    pc_c_j = jnp.float32(pc_c)

    def _coarse_lag(params, defect):
        def fwd(p):
            return apply_fn(p, xy_c)
        pred, vjp_fn = jax.vjp(fwd, params)
        pp = pred[:Na_c]
        pb = pred[ob_c:oc_c]
        pc = pred[oc_c:]
        gp = bfsa_bwd(pp, g_c) + defect
        gb = 2.0*(pb - bc_tgt_c)/(Nb_c*3)
        gc_p = 2.0*(pc[:,2:3]-pc_c_j)
        gc_uv = jnp.zeros_like(pc[:,0:2])
        gc = jnp.concatenate([gc_uv, gc_p], axis=1)
        us = jnp.concatenate([gp, gb, gc], axis=0)
        (pg,) = vjp_fn(us)
        c,mu,mv = _compute_pde_kovasznay_jax(pp, g_c)
        lp = (jnp.sum(c**2*imask_c)/M_c + jnp.sum(mu**2*imask_c)/M_c
              + jnp.sum(mv**2*imask_c)/M_c)
        lb = jnp.mean((pb-bc_tgt_c)**2)
        lpc = jnp.mean((pc[:,2:3]-pc_c_j)**2)
        return lp+lb+lpc, pg

    @jit
    def coarse_step(params, opt_state, defect):
        loss, pg = _coarse_lag(params, defect)
        upd, opt_state = optimizer.update(pg, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss

    def _anchor_lag(params, defect):
        def fwd_c(p):
            return apply_fn(p, xy_c)
        pred_c, vjp_fn = jax.vjp(fwd_c, params)
        pp = pred_c[:Na_c]
        pb = pred_c[ob_c:oc_c]
        pc = pred_c[oc_c:]
        gpc = bfsa_bwd(pp, g_c)
        gp = gpc + defect
        gb = 2.0*(pb-bc_tgt_c)/(Nb_c*3)
        gc_p = 2.0*(pc[:,2:3]-pc_c_j)
        gc_uv = jnp.zeros_like(pc[:,0:2])
        gc = jnp.concatenate([gc_uv, gc_p], axis=1)
        us = jnp.concatenate([gp, gb, gc], axis=0)
        (pg,) = vjp_fn(us)
        pred_f = apply_fn(params, xy_f)
        gpf = bfsa_bwd(pred_f[:Na_f], g_f)
        gpf_r = _restrict_2d(gpf, R_1d, nf, nc) * (M_c / M_f)
        new_defect = (gpf_r - gpc)*imask_c
        c,mu,mv = _compute_pde_kovasznay_jax(pp, g_c)
        lp = (jnp.sum(c**2*imask_c)/M_c + jnp.sum(mu**2*imask_c)/M_c
              + jnp.sum(mv**2*imask_c)/M_c)
        lb = jnp.mean((pb-bc_tgt_c)**2)
        lpc = jnp.mean((pc[:,2:3]-pc_c_j)**2)
        return lp+lb+lpc, pg, new_defect

    @jit
    def anchor_step(params, opt_state, defect):
        loss, pg, new_defect = _anchor_lag(params, defect)
        upd, opt_state = optimizer.update(pg, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, new_defect

    return coarse_step, anchor_step, optimizer


def train_sdccg_kovasznay(seed: int, device: torch.device, n_epochs: int, lr: float,
                           grid_size: int, model_name: str,
                           m_coarse: int = 40, k_anchor: int = 10):
    from src.lid_benchmark import build_grid_data_kovasznay
    g_tf = build_grid_data_kovasznay(grid_size, device)
    keys = ['Dx','Dy','DxT','DyT','interior_mask',
            'xy_batched','xy_all','xy_bc','bc_target']
    g_f = _torch_g_to_jax(g_tf, keys)
    Na_f = int(g_tf['N_all']); Nb_f = int(g_tf['N_bc']); M_f = int(g_tf['M'])
    ob_f = int(g_tf['off_bc']); oc_f = int(g_tf['off_center'])
    pc_f = float(g_tf['p_center_exact'])
    g_f['N_all'] = Na_f; g_f['M'] = M_f
    g_f = _enrich_g_jax_with_1d_matrices(g_f, grid_size, Lx=1.5, Ly=2.0)

    g_tc = build_grid_data_kovasznay(m_coarse, device)
    g_c = _torch_g_to_jax(g_tc, keys)
    Na_c = int(g_tc['N_all']); Nb_c = int(g_tc['N_bc']); M_c = int(g_tc['M'])
    ob_c = int(g_tc['off_bc']); oc_c = int(g_tc['off_center'])
    pc_c = float(g_tc['p_center_exact'])
    g_c['N_all'] = Na_c; g_c['M'] = M_c
    g_c = _enrich_g_jax_with_1d_matrices(g_c, m_coarse, Lx=1.5, Ly=2.0)

    R_1d = jnp.asarray(_chebyshev_interp_1d(grid_size, m_coarse).astype(np.float32))

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    print(f"  [sdccg] Model params: {count_params(params)} ({model_name})")
    bfsa_bwd = _build_bfsa_backward('kovasznay')

    coarse_step, anchor_step, optimizer = _make_sdccg_kovasznay_steps(
        net, lr, g_f, g_c, bfsa_bwd, R_1d, grid_size, m_coarse,
        Na_f, ob_f, oc_f, Nb_f, M_f, pc_f,
        Na_c, ob_c, oc_c, Nb_c, M_c, pc_c)
    opt_state = optimizer.init(params)
    defect = jnp.zeros((Na_c, 3))

    t_w = time.perf_counter()
    params, opt_state, loss, defect = anchor_step(params, opt_state, defect)
    loss.block_until_ready()
    params, opt_state, loss = coarse_step(params, opt_state, defect)
    loss.block_until_ready()
    print(f"  [sdccg] JIT warmup: {time.perf_counter()-t_w:.2f}s, loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(2, n_epochs):
        if epoch % k_anchor == 0:
            params, opt_state, loss, defect = anchor_step(params, opt_state, defect)
        else:
            params, opt_state, loss = coarse_step(params, opt_state, defect)
        if (epoch+1) % 5000 == 0 or (epoch+1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)
    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


# ---------- SDCCG Elasticity ----------

def _make_sdccg_elasticity_steps(model, lr, n_epochs, g_f, g_c, bfsa_bwd, R_1d,
                                  nf, nc,
                                  Na_f, ob_f, Nb_f, M_f,
                                  Na_c, ob_c, Nb_c, M_c):
    apply_fn = model.apply
    xy_f = g_f['xy_batched']
    xy_c = g_c['xy_batched']
    imask_c = g_c['interior_mask']
    bctgt_c = g_c['bc_target']
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs,
                                           alpha=1e-5/lr)
    optimizer = optax.adam(learning_rate=schedule)

    def _coarse_lag(params, defect):
        def fwd(p):
            return apply_fn(p, xy_c)
        pred, vjp_fn = jax.vjp(fwd, params)
        pp = pred[:Na_c]
        pb = pred[ob_c:ob_c+Nb_c]
        gp = bfsa_bwd(pp, g_c) + defect
        gb = 2.0*(pb-bctgt_c)/(Nb_c*2)
        us = jnp.concatenate([gp, gb], axis=0)
        (pg,) = vjp_fn(us)
        ex,ey = _compute_pde_elasticity_jax(pp, g_c)
        lp = jnp.sum(ex**2*imask_c)/M_c + jnp.sum(ey**2*imask_c)/M_c
        lb = jnp.mean((pb-bctgt_c)**2)
        return lp+lb, pg

    @jit
    def coarse_step(params, opt_state, defect):
        loss, pg = _coarse_lag(params, defect)
        upd, opt_state = optimizer.update(pg, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss

    def _anchor_lag(params, defect):
        def fwd_c(p):
            return apply_fn(p, xy_c)
        pred_c, vjp_fn = jax.vjp(fwd_c, params)
        pp = pred_c[:Na_c]
        pb = pred_c[ob_c:ob_c+Nb_c]
        gpc = bfsa_bwd(pp, g_c)
        gp = gpc + defect
        gb = 2.0*(pb-bctgt_c)/(Nb_c*2)
        us = jnp.concatenate([gp, gb], axis=0)
        (pg,) = vjp_fn(us)
        pred_f = apply_fn(params, xy_f)
        gpf = bfsa_bwd(pred_f[:Na_f], g_f)
        gpf_r = _restrict_2d(gpf, R_1d, nf, nc) * (M_c / M_f)
        new_defect = (gpf_r - gpc)*imask_c
        ex,ey = _compute_pde_elasticity_jax(pp, g_c)
        lp = jnp.sum(ex**2*imask_c)/M_c + jnp.sum(ey**2*imask_c)/M_c
        lb = jnp.mean((pb-bctgt_c)**2)
        return lp+lb, pg, new_defect

    @jit
    def anchor_step(params, opt_state, defect):
        loss, pg, new_defect = _anchor_lag(params, defect)
        upd, opt_state = optimizer.update(pg, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, new_defect

    return coarse_step, anchor_step, optimizer


def train_sdccg_elasticity(seed: int, device: torch.device, n_epochs: int, lr: float,
                            grid_size: int, model_name: str,
                            m_coarse: int = 40, k_anchor: int = 10):
    from src.lid_benchmark import build_grid_data_elasticity
    g_tf = build_grid_data_elasticity(grid_size, device)
    keys = ['Dx','Dy','DxT','DyT','Dxx','Dyy','Dxy',
            'DxxT','DyyT','DxyT','interior_mask','fx','fy',
            'xy_batched','xy_all','xy_bc','bc_target']
    g_f = _torch_g_to_jax(g_tf, keys)
    Na_f = int(g_tf['N_all']); Nb_f = int(g_tf['N_bc']); M_f = int(g_tf['M'])
    ob_f = int(g_tf['off_bc'])
    g_f['N_all'] = Na_f; g_f['M'] = M_f
    g_f = _enrich_g_jax_with_1d_matrices(g_f, grid_size, Lx=1.0, Ly=1.0)

    g_tc = build_grid_data_elasticity(m_coarse, device)
    g_c = _torch_g_to_jax(g_tc, keys)
    Na_c = int(g_tc['N_all']); Nb_c = int(g_tc['N_bc']); M_c = int(g_tc['M'])
    ob_c = int(g_tc['off_bc'])
    g_c['N_all'] = Na_c; g_c['M'] = M_c
    g_c = _enrich_g_jax_with_1d_matrices(g_c, m_coarse, Lx=1.0, Ly=1.0)

    R_1d = jnp.asarray(_chebyshev_interp_1d(grid_size, m_coarse).astype(np.float32))

    net, params = _init_model(model_name, out_dim=2, seed=seed)
    print(f"  [sdccg] Model params: {count_params(params)} ({model_name})")
    bfsa_bwd = _build_bfsa_backward('elasticity')

    coarse_step, anchor_step, optimizer = _make_sdccg_elasticity_steps(
        net, lr, n_epochs, g_f, g_c, bfsa_bwd, R_1d, grid_size, m_coarse,
        Na_f, ob_f, Nb_f, M_f,
        Na_c, ob_c, Nb_c, M_c)
    opt_state = optimizer.init(params)
    defect = jnp.zeros((Na_c, 2))

    t_w = time.perf_counter()
    params, opt_state, loss, defect = anchor_step(params, opt_state, defect)
    loss.block_until_ready()
    params, opt_state, loss = coarse_step(params, opt_state, defect)
    loss.block_until_ready()
    print(f"  [sdccg] JIT warmup: {time.perf_counter()-t_w:.2f}s, loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(2, n_epochs):
        if epoch % k_anchor == 0:
            params, opt_state, loss, defect = anchor_step(params, opt_state, defect)
        else:
            params, opt_state, loss = coarse_step(params, opt_state, defect)
        if (epoch+1) % 5000 == 0 or (epoch+1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)
    torch_model = flax_params_to_torch(model_name, params, out_dim=2, device=device)
    return torch_model, train_time, final_loss_val


# ---------- SLRM-JAX Cavity ----------
def _make_slrm_jax_cavity_train_step(model, lr, g_jax, M_ref,
                                     N_all, off_lid, off_wall, off_center,
                                     N_lid, N_wall, M):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    interior_mask = g_jax['interior_mask']
    optimizer = optax.adam(lr)
    K_pred = 3
    two_over_M = jnp.float32(2.0 / M)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)

        pred_pde = pred_batch[:N_all]
        pred_lid = pred_batch[off_lid:off_wall]
        pred_wall = pred_batch[off_wall:off_center]
        pred_c = pred_batch[off_center:]

        cont, mom_u, mom_v = _compute_pde_cavity_jax(pred_pde, g_jax)
        r = jnp.concatenate([cont, mom_u, mom_v], axis=1)
        r = r * interior_mask
        r_flat = r.reshape(-1)
        grad_pde_flat = two_over_M * (M_ref @ r_flat)
        grad_pde = grad_pde_flat.reshape(N_all, K_pred)

        gl0 = 2.0 * (pred_lid[:, 0:1] - 1.0) / N_lid
        gl1 = 2.0 * pred_lid[:, 1:2] / N_lid
        gl2 = jnp.zeros_like(pred_lid[:, 2:3])
        grad_lid = jnp.concatenate([gl0, gl1, gl2], axis=1)

        gw0 = 2.0 * pred_wall[:, 0:1] / N_wall
        gw1 = 2.0 * pred_wall[:, 1:2] / N_wall
        gw2 = jnp.zeros_like(pred_wall[:, 2:3])
        grad_wall = jnp.concatenate([gw0, gw1, gw2], axis=1)

        gc0 = jnp.zeros_like(pred_c[:, 0:1])
        gc1 = jnp.zeros_like(pred_c[:, 1:2])
        gc2 = 2.0 * pred_c[:, 2:3]
        grad_center = jnp.concatenate([gc0, gc1, gc2], axis=1)

        upstream = jnp.concatenate(
            [grad_pde, grad_lid, grad_wall, grad_center], axis=0)
        (param_grads,) = vjp_fn(upstream)

        loss_pde = (jnp.sum(cont ** 2 * interior_mask) / M
                    + jnp.sum(mom_u ** 2 * interior_mask) / M
                    + jnp.sum(mom_v ** 2 * interior_mask) / M)
        loss_lid = jnp.mean((pred_lid[:, 0:1] - 1.0) ** 2) + jnp.mean(pred_lid[:, 1:2] ** 2)
        loss_wall = jnp.mean(pred_wall[:, 0:1] ** 2) + jnp.mean(pred_wall[:, 1:2] ** 2)
        loss_p = jnp.mean(pred_c[:, 2:3] ** 2)
        loss_val = loss_pde + loss_lid + loss_wall + loss_p
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def train_slrm_jax_cavity(seed: int, device: torch.device, n_epochs: int, lr: float,
                          grid_size: int, model_name: str):
    from src.lid_benchmark import build_grid_data
    g_torch = build_grid_data(grid_size, device)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Cs_d_sq', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_lid', 'xy_wall']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_lid = int(g_torch['N_lid'])
    N_wall = int(g_torch['N_wall'])
    M = int(g_torch['M'])
    off_lid = int(g_torch['off_lid'])
    off_wall = int(g_torch['off_wall'])
    off_center = int(g_torch['off_center'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M

    net, lrm_params = _init_model(model_name, out_dim=3, seed=seed)
    params = lrm_params
    n_params = count_params(params)
    print(f"  [slrm-jax] Model params: {n_params} ({model_name})")

    # Reference input for SLRM linearisation.
    # Two strategies tested:
    #   SLRM_REF=nn_init (default): use the NN's own initial full-grid
    #     output. Near-zero magnitudes → advection terms in J vanish at
    #     the ref point. Diverges on nonlinear NS cells.
    #   SLRM_REF=shear: use u = y (shear profile satisfying u=0 on
    #     bottom wall, u=1 on top lid), v=0, p=0. Non-zero u captures
    #     the "u * D_x" coupling in advection Jacobian.
    import os as _os
    _ref_mode = _os.environ.get("SLRM_REF", "nn_init")
    if _ref_mode == "shear":
        y_full = g_jax['xy_all'][:, 1:2]
        pred_ref = jnp.concatenate(
            [y_full, jnp.zeros_like(y_full), jnp.zeros_like(y_full)], axis=1)
        print(f"  [slrm-jax] pred_ref: shear (u=y, v=0, p=0)")
    else:
        pred_ref = net.apply(params, g_jax['xy_all'])
        print(f"  [slrm-jax] pred_ref: NN init output")
    def res_fn_for_M(pred):
        c, mu, mv = _compute_pde_cavity_jax(pred, g_jax)
        return jnp.concatenate([c, mu, mv], axis=1)
    t_mb0 = time.perf_counter()
    M_ref = _slrm_build_M_jax(res_fn_for_M, pred_ref)
    M_ref.block_until_ready()
    print(f"  [slrm-jax] Built M_ref {M_ref.shape} in {time.perf_counter() - t_mb0:.2f}s")

    train_step, optimizer = _make_slrm_jax_cavity_train_step(
        net, lr, g_jax, M_ref,
        N_all, off_lid, off_wall, off_center, N_lid, N_wall, M)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [slrm-jax] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


# ---------- SLRM-JAX Kovasznay ----------
def _make_slrm_jax_kovasznay_train_step(model, lr, g_jax, M_ref,
                                        N_all, off_bc, off_center, N_bc, M,
                                        p_center_exact):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    bc_target = g_jax['bc_target']
    interior_mask = g_jax['interior_mask']
    optimizer = optax.adam(lr)
    p_center_f = jnp.float32(p_center_exact)
    K_pred = 3
    two_over_M = jnp.float32(2.0 / M)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)

        pred_pde = pred_batch[:N_all]
        pred_bc = pred_batch[off_bc:off_center]
        pred_c = pred_batch[off_center:]

        cont, mom_u, mom_v = _compute_pde_kovasznay_jax(pred_pde, g_jax)
        r = jnp.concatenate([cont, mom_u, mom_v], axis=1)
        r = r * interior_mask
        r_flat = r.reshape(-1)
        grad_pde_flat = two_over_M * (M_ref @ r_flat)
        grad_pde = grad_pde_flat.reshape(N_all, K_pred)

        n_out = 3
        grad_bc = 2.0 * (pred_bc - bc_target) / (N_bc * n_out)

        gc_p = 2.0 * (pred_c[:, 2:3] - p_center_f)
        gc_uv = jnp.zeros_like(pred_c[:, 0:2])
        grad_center = jnp.concatenate([gc_uv, gc_p], axis=1)

        upstream = jnp.concatenate([grad_pde, grad_bc, grad_center], axis=0)
        (param_grads,) = vjp_fn(upstream)

        loss_pde = (jnp.sum(cont ** 2 * interior_mask) / M
                    + jnp.sum(mom_u ** 2 * interior_mask) / M
                    + jnp.sum(mom_v ** 2 * interior_mask) / M)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        loss_p = jnp.mean((pred_c[:, 2:3] - p_center_f) ** 2)
        loss_val = loss_pde + loss_bc + loss_p
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def train_slrm_jax_kovasznay(seed: int, device: torch.device, n_epochs: int, lr: float,
                             grid_size: int, model_name: str):
    from src.lid_benchmark import build_grid_data_kovasznay
    g_torch = build_grid_data_kovasznay(grid_size, device)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    off_center = int(g_torch['off_center'])
    p_center_exact = float(g_torch['p_center_exact'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    n_params = count_params(params)
    print(f"  [slrm-jax] Model params: {n_params} ({model_name})")

    import os as _os
    _ref_mode = _os.environ.get("SLRM_REF", "nn_init")
    if _ref_mode == "exact":
        # Use the Kovasznay exact solution as pred_ref — this linearises at
        # the true minimum, giving an ideal upper-bound on SLRM accuracy.
        x_all = g_jax['xy_all'][:, 0:1]
        y_all = g_jax['xy_all'][:, 1:2]
        lam = LAMBDA_KOV
        u_ex = 1.0 - jnp.exp(lam * x_all) * jnp.cos(2.0 * jnp.pi * y_all)
        v_ex = (lam / (2.0 * jnp.pi)) * jnp.exp(lam * x_all) * jnp.sin(2.0 * jnp.pi * y_all)
        p_ex = 0.5 * (1.0 - jnp.exp(2.0 * lam * x_all))
        pred_ref = jnp.concatenate([u_ex, v_ex, p_ex], axis=1)
        print(f"  [slrm-jax] pred_ref: exact Kovasznay solution (upper-bound test)")
    else:
        pred_ref = net.apply(params, g_jax['xy_all'])
        print(f"  [slrm-jax] pred_ref: NN init output")
    def res_fn_for_M(pred):
        c, mu, mv = _compute_pde_kovasznay_jax(pred, g_jax)
        return jnp.concatenate([c, mu, mv], axis=1)
    t_mb0 = time.perf_counter()
    M_ref = _slrm_build_M_jax(res_fn_for_M, pred_ref)
    M_ref.block_until_ready()
    print(f"  [slrm-jax] Built M_ref {M_ref.shape} in {time.perf_counter() - t_mb0:.2f}s")

    train_step, optimizer = _make_slrm_jax_kovasznay_train_step(
        net, lr, g_jax, M_ref,
        N_all, off_bc, off_center, N_bc, M, p_center_exact)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [slrm-jax] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=3, device=device)
    return torch_model, train_time, final_loss_val


# ---------- SLRM-JAX Elasticity ----------
def _make_slrm_jax_elasticity_train_step(model, lr, n_epochs, g_jax, M_ref,
                                         N_all, off_bc, N_bc, M):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    bc_target = g_jax['bc_target']
    interior_mask = g_jax['interior_mask']
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs,
                                           alpha=1e-5 / lr)
    optimizer = optax.adam(learning_rate=schedule)
    K_pred = 2
    two_over_M = jnp.float32(2.0 / M)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)

        pred_pde = pred_batch[:N_all]
        pred_bc = pred_batch[off_bc:off_bc + N_bc]

        eq_x, eq_y = _compute_pde_elasticity_jax(pred_pde, g_jax)
        r = jnp.concatenate([eq_x, eq_y], axis=1)
        r = r * interior_mask
        r_flat = r.reshape(-1)
        grad_pde_flat = two_over_M * (M_ref @ r_flat)
        grad_pde = grad_pde_flat.reshape(N_all, K_pred)

        n_out = 2
        grad_bc = 2.0 * (pred_bc - bc_target) / (N_bc * n_out)

        upstream = jnp.concatenate([grad_pde, grad_bc], axis=0)
        (param_grads,) = vjp_fn(upstream)

        loss_pde = (jnp.sum(eq_x ** 2 * interior_mask) / M
                    + jnp.sum(eq_y ** 2 * interior_mask) / M)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        loss_val = loss_pde + loss_bc
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def train_slrm_jax_elasticity(seed: int, device: torch.device, n_epochs: int, lr: float,
                              grid_size: int, model_name: str):
    from src.lid_benchmark import build_grid_data_elasticity
    g_torch = build_grid_data_elasticity(grid_size, device)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Dxx', 'Dyy', 'Dxy',
            'DxxT', 'DyyT', 'DxyT', 'interior_mask', 'fx', 'fy',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M

    net, params = _init_model(model_name, out_dim=2, seed=seed)
    n_params = count_params(params)
    print(f"  [slrm-jax] Model params: {n_params} ({model_name})")

    pred_ref = net.apply(params, g_jax['xy_all'])
    def res_fn_for_M(pred):
        ex, ey = _compute_pde_elasticity_jax(pred, g_jax)
        return jnp.concatenate([ex, ey], axis=1)
    t_mb0 = time.perf_counter()
    M_ref = _slrm_build_M_jax(res_fn_for_M, pred_ref)
    M_ref.block_until_ready()
    print(f"  [slrm-jax] Built M_ref {M_ref.shape} in {time.perf_counter() - t_mb0:.2f}s")

    train_step, optimizer = _make_slrm_jax_elasticity_train_step(
        net, lr, n_epochs, g_jax, M_ref, N_all, off_bc, N_bc, M)
    opt_state = optimizer.init(params)

    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    print(f"  [slrm-jax] JIT warmup: {time.perf_counter() - t_warm0:.2f}s, "
          f"initial loss={float(loss):.6f}")

    start = time.perf_counter()
    final_loss_val = float(loss)
    for epoch in range(1, n_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        if (epoch + 1) % 5000 == 0 or (epoch + 1) == n_epochs:
            final_loss_val = float(loss)
            print(f"  Epoch {epoch+1}: loss={final_loss_val:.6f}")
    loss.block_until_ready()
    train_time = time.perf_counter() - start
    final_loss_val = float(loss)

    torch_model = flax_params_to_torch(model_name, params, out_dim=2, device=device)
    return torch_model, train_time, final_loss_val
