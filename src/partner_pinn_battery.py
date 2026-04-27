"""
Battery cooling channel PINN — Two-stage DeepXDE pipeline.

Ported from partner team's main.py (Feb 2026, partner_code_v2).

Stage A: Time-dependent 2D incompressible Navier-Stokes.
         Input (x,y,t) -> output (u,v,p). FNN [3, 128x6, 3], tanh.
         Adam 20K + L-BFGS. Rectangle [0,2]x[0,0.5], T_final=1.0.

Stage B: Temperature advection-diffusion using frozen flow from Stage A.
         Input (x,y,t) -> output T. FNN [3, 128x6, 1], tanh.
         Adam 20K + L-BFGS. Inlet T=25C, adiabatic walls.

Known gap: load_domain_json() reads pipe geometry but is never called.
The PINN training uses simple rectangle geometry, not the complex pipe.
This is reproduced as-is from the partner's code.

All logic, defaults, and architecture identical to partner's original.
"""

import os
import json
import argparse
import numpy as np

# IMPORTANT: set backend BEFORE importing deepxde
os.environ["DDEBACKEND"] = "pytorch"
import deepxde as dde
from deepxde.backend import torch as bkd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# -----------------------------
# Utilities: JSON (indent=4) I/O
# -----------------------------
def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Geometry helpers
# -----------------------------
def build_geometry_rect_or_cyl(
    x_min, x_max, y_min, y_max,
    obstacle_circle=None
):
    """
    Rectangle channel with optional circular obstacle:
    obstacle_circle = (cx, cy, r) or None
    """
    rect = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    if obstacle_circle is None:
        return rect

    cx, cy, r = obstacle_circle
    circle = dde.geometry.Disk([cx, cy], r)
    geom = dde.geometry.CSGDifference(rect, circle)
    return geom


# -----------------------------
# Stage A: Navier-Stokes PINN
# -----------------------------
def navier_stokes_pde(nu: float):
    """
    Incompressible 2D NS in primitive variables:
      u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy) = 0
      v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy) = 0
      u_x + v_y = 0

    Network output: [u, v, p]
    Input: [x, y, t]
    """
    def pde(x, y):
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

        mom_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        mom_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        cont  = u_x + v_y

        return [mom_u, mom_v, cont]

    return pde


def train_navier_stokes(
    geom,
    T_final: float,
    nu: float,
    V0: float,
    save_model_path: str,
    n_domain=20000,
    n_boundary=4000,
    n_initial=4000,
    lr=1e-3,
    adam_epochs=20000,
    lbfgs=True,
):
    """
    Example BC/IC:
      - Inlet (x = x_min): u=V0, v=0
      - Walls (y=y_min or y=y_max): u=0, v=0 (no-slip)
      - Outlet (x = x_max): p=0
      - Initial (t=0): u=0, v=0, p=0  (you can change)
    """
    timedomain = dde.geometry.TimeDomain(0.0, T_final)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    x_min = geom.bbox[0][0]
    x_max = geom.bbox[1][0]
    y_min = geom.bbox[0][1]
    y_max = geom.bbox[1][1]

    def inlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_min)

    def outlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_max)

    def wall_bottom(x, on_boundary):
        return on_boundary and np.isclose(x[1], y_min)

    def wall_top(x, on_boundary):
        return on_boundary and np.isclose(x[1], y_max)

    # Inlet BC: u=V0, v=0
    bc_in_u = dde.icbc.DirichletBC(geomtime, lambda x: V0, inlet, component=0)
    bc_in_v = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, inlet, component=1)

    # No-slip walls
    bc_wb_u = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_bottom, component=0)
    bc_wb_v = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_bottom, component=1)
    bc_wt_u = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_top, component=0)
    bc_wt_v = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, wall_top, component=1)

    # Outlet: p=0
    bc_out_p = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, outlet, component=2)

    # Initial: u=0, v=0, p=0 at t=0
    ic_u = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_initial: on_initial, component=0)
    ic_v = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_initial: on_initial, component=1)
    ic_p = dde.icbc.IC(geomtime, lambda x: 0.0, lambda x, on_initial: on_initial, component=2)

    data = dde.data.TimePDE(
        geomtime,
        navier_stokes_pde(nu),
        [bc_in_u, bc_in_v, bc_wb_u, bc_wb_v, bc_wt_u, bc_wt_v, bc_out_p, ic_u, ic_v, ic_p],
        num_domain=n_domain,
        num_boundary=n_boundary,
        num_initial=n_initial,
    )

    net = dde.nn.FNN([3] + [128] * 6 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    model.train(iterations=adam_epochs, display_every=200)

    if lbfgs:
        model.compile("L-BFGS")
        model.train()

    # Save DeepXDE model weights/checkpoint
    model.save(save_model_path)
    return model


def export_flow_fields_json(
    model,
    geom,
    T_final: float,
    nx: int,
    ny: int,
    nt: int,
    out_json: str,
):
    """
    Samples (u,v,p) on a regular grid and saves to JSON (indent=4).
    WARNING: JSON can get huge. Keep nx,ny,nt moderate (e.g., 161x81x20).
    """
    x_min, y_min = geom.bbox[0]
    x_max, y_max = geom.bbox[1]

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    ts = np.linspace(0.0, T_final, nt)

    # grid for each time
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape (ny, nx)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)  # (ny*nx, 2)

    flow = {
        "meta": {
            "nx": nx, "ny": ny, "nt": nt,
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
            "t_min": 0.0, "t_max": float(T_final),
        },
        "grid": {
            "x": xs.tolist(),
            "y": ys.tolist(),
            "t": ts.tolist(),
        },
        "fields": {
            "u": [],
            "v": [],
            "p": [],
        }
    }

    for t in ts:
        tcol = np.full((XY.shape[0], 1), t)
        inp = np.hstack([XY, tcol])  # (N, 3)
        pred = model.predict(inp)    # (N, 3)
        u = pred[:, 0].reshape(ny, nx)
        v = pred[:, 1].reshape(ny, nx)
        p = pred[:, 2].reshape(ny, nx)

        flow["fields"]["u"].append(u.tolist())
        flow["fields"]["v"].append(v.tolist())
        flow["fields"]["p"].append(p.tolist())

    save_json(out_json, flow)
    print(f"[save] flow fields JSON -> {out_json}")


# -----------------------------
# Stage B: Temperature PINN
# -----------------------------
def make_flow_interpolator_from_json(flow_json: dict):
    """
    Builds simple trilinear interpolators u(x,y,t), v(x,y,t) from saved grids.
    For speed/quality you can replace this with SciPy RegularGridInterpolator.
    """
    xs = np.array(flow_json["grid"]["x"], dtype=np.float64)
    ys = np.array(flow_json["grid"]["y"], dtype=np.float64)
    ts = np.array(flow_json["grid"]["t"], dtype=np.float64)

    u_arr = np.array(flow_json["fields"]["u"], dtype=np.float64)  # (nt, ny, nx)
    v_arr = np.array(flow_json["fields"]["v"], dtype=np.float64)  # (nt, ny, nx)

    # Precompute for bounds
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    t_min, t_max = ts[0], ts[-1]

    def clamp(a, lo, hi):
        return np.minimum(np.maximum(a, lo), hi)

    def trilerp(field, x, y, t):
        """
        field: (nt, ny, nx)
        x,y,t: (N,)
        returns: (N,)
        """
        x = clamp(x, x_min, x_max)
        y = clamp(y, y_min, y_max)
        t = clamp(t, t_min, t_max)

        ix = np.searchsorted(xs, x) - 1
        iy = np.searchsorted(ys, y) - 1
        it = np.searchsorted(ts, t) - 1

        ix = np.clip(ix, 0, len(xs) - 2)
        iy = np.clip(iy, 0, len(ys) - 2)
        it = np.clip(it, 0, len(ts) - 2)

        x0, x1 = xs[ix], xs[ix + 1]
        y0, y1 = ys[iy], ys[iy + 1]
        t0, t1 = ts[it], ts[it + 1]

        xd = (x - x0) / (x1 - x0 + 1e-12)
        yd = (y - y0) / (y1 - y0 + 1e-12)
        td = (t - t0) / (t1 - t0 + 1e-12)

        # gather 8 corners
        def F(tt, yy, xx):
            return field[tt, yy, xx]

        c000 = F(it,     iy,     ix)
        c001 = F(it,     iy,     ix + 1)
        c010 = F(it,     iy + 1, ix)
        c011 = F(it,     iy + 1, ix + 1)
        c100 = F(it + 1, iy,     ix)
        c101 = F(it + 1, iy,     ix + 1)
        c110 = F(it + 1, iy + 1, ix)
        c111 = F(it + 1, iy + 1, ix + 1)

        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        c = c0 * (1 - td) + c1 * td
        return c

    def u_func(xyt: np.ndarray) -> np.ndarray:
        x = xyt[:, 0]
        y = xyt[:, 1]
        t = xyt[:, 2]
        return trilerp(u_arr, x, y, t).reshape(-1, 1)

    def v_func(xyt: np.ndarray) -> np.ndarray:
        x = xyt[:, 0]
        y = xyt[:, 1]
        t = xyt[:, 2]
        return trilerp(v_arr, x, y, t).reshape(-1, 1)

    return u_func, v_func


def temperature_pde(alpha: float, u_func, v_func):
    """
    Advection-diffusion:
      T_t + u(x,y,t) T_x + v(x,y,t) T_y - alpha (T_xx + T_yy) = 0

    Network output: [T]
    Input: [x,y,t]
    """
    def pde(x, y):
        T = y[:, 0:1]

        T_x = dde.grad.jacobian(y, x, i=0, j=0)
        T_y = dde.grad.jacobian(y, x, i=0, j=1)
        T_t = dde.grad.jacobian(y, x, i=0, j=2)

        T_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        T_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

        # u,v from exported NS fields (numpy) -> convert to torch tensor
        x_np = x.detach().cpu().numpy()
        u_np = u_func(x_np)
        v_np = v_func(x_np)
        u = bkd.from_numpy(u_np.astype(np.float32)).to(x.device)
        v = bkd.from_numpy(v_np.astype(np.float32)).to(x.device)

        return T_t + u * T_x + v * T_y - alpha * (T_xx + T_yy)

    return pde


def train_temperature(
    geom,
    T_final: float,
    alpha: float,
    inlet_T: float,
    flow_json_path: str,
    save_model_path: str,
    n_domain=20000,
    n_boundary=4000,
    n_initial=4000,
    lr=1e-3,
    adam_epochs=20000,
    lbfgs=True,
):
    flow = load_json(flow_json_path)
    u_func, v_func = make_flow_interpolator_from_json(flow)

    timedomain = dde.geometry.TimeDomain(0.0, T_final)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    x_min = geom.bbox[0][0]
    x_max = geom.bbox[1][0]
    y_min = geom.bbox[0][1]
    y_max = geom.bbox[1][1]

    def inlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_min)

    def outlet(x, on_boundary):
        return on_boundary and np.isclose(x[0], x_max)

    def wall_bottom(x, on_boundary):
        return on_boundary and np.isclose(x[1], y_min)

    def wall_top(x, on_boundary):
        return on_boundary and np.isclose(x[1], y_max)

    # Temperature BC/IC examples:
    # Inlet fixed temperature
    bc_in_T = dde.icbc.DirichletBC(geomtime, lambda x: inlet_T, inlet, component=0)

    # Walls: adiabatic (Neumann dT/dn = 0) -> for horizontal walls, use dT/dy = 0
    bc_wall_bot = dde.icbc.NeumannBC(geomtime, lambda x: 0.0, wall_bottom, component=0)
    bc_wall_top = dde.icbc.NeumannBC(geomtime, lambda x: 0.0, wall_top, component=0)

    # Outlet: zero-gradient (approx)
    bc_out_T = dde.icbc.NeumannBC(geomtime, lambda x: 0.0, outlet, component=0)

    # Initial condition: everywhere 25C (or your value)
    ic_T = dde.icbc.IC(geomtime, lambda x: inlet_T, lambda x, on_initial: on_initial, component=0)

    data = dde.data.TimePDE(
        geomtime,
        temperature_pde(alpha, u_func, v_func),
        [bc_in_T, bc_wall_bot, bc_wall_top, bc_out_T, ic_T],
        num_domain=n_domain,
        num_boundary=n_boundary,
        num_initial=n_initial,
    )

    net = dde.nn.FNN([3] + [128] * 6 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    model.train(iterations=adam_epochs, display_every=200)

    if lbfgs:
        model.compile("L-BFGS")
        model.train()

    model.save(save_model_path)
    return model


def export_temperature_json_and_gif(
    model,
    geom,
    T_final: float,
    nx: int,
    ny: int,
    nt: int,
    out_json: str,
    out_gif: str,
    fps: int = 6,
):
    x_min, y_min = geom.bbox[0]
    x_max, y_max = geom.bbox[1]

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    ts = np.linspace(0.0, T_final, nt)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)

    out = {
        "meta": {
            "nx": nx, "ny": ny, "nt": nt,
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
            "t_min": 0.0, "t_max": float(T_final),
        },
        "grid": {"x": xs.tolist(), "y": ys.tolist(), "t": ts.tolist()},
        "fields": {"T": []},
    }

    frames = []
    for k, t in enumerate(ts):
        tcol = np.full((XY.shape[0], 1), t)
        inp = np.hstack([XY, tcol])
        pred = model.predict(inp)[:, 0].reshape(ny, nx)

        out["fields"]["T"].append(pred.tolist())

        # frame
        fig = plt.figure()
        plt.imshow(
            pred,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
        )
        plt.title(f"T(x,y) at t={t:.4f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(label="Temperature")

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        frames.append(img)

    save_json(out_json, out)
    imageio.mimsave(out_gif, frames, fps=fps)

    print(f"[save] temperature JSON -> {out_json}")
    print(f"[save] temperature GIF  -> {out_gif}")


# -- Read the json coordinates (UNUSED — this is the known gap) --

def load_domain_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pts = np.asarray(cfg["points"], dtype=np.float32)  # shape [N, 3] : x,y,flag
    xy = pts[:, :2]
    flag = pts[:, 2].astype(np.int32)

    boundary_xy  = xy[flag == 1]
    interior_xy  = xy[flag == 0]

    inlet_xy  = np.array([cfg["inlet"]["x"],  cfg["inlet"]["y"]],  dtype=np.float32)
    outlet_xy = np.array([cfg["outlet"]["x"], cfg["outlet"]["y"]], dtype=np.float32)

    W = float(cfg["width"])
    H = float(cfg["height"])

    return {
        "width": W, "height": H,
        "interior_xy": interior_xy,
        "boundary_xy": boundary_xy,
        "inlet_xy": inlet_xy,
        "outlet_xy": outlet_xy,
        "all_xy": xy,
        "flag": flag,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ns", "temp", "all"], default="all")
    parser.add_argument("--outdir", type=str, default="results/partner_pinn")

    # domain
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=2.0)
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=0.5)
    parser.add_argument("--T_final", type=float, default=1.0)

    # obstacle (optional)
    parser.add_argument("--obs_cx", type=float, default=None)
    parser.add_argument("--obs_cy", type=float, default=None)
    parser.add_argument("--obs_r", type=float, default=None)

    # physics
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--V0", type=float, default=1.0)
    parser.add_argument("--Tin", type=float, default=25.0)

    # export resolution
    parser.add_argument("--nx", type=int, default=161)
    parser.add_argument("--ny", type=int, default=81)
    parser.add_argument("--nt", type=int, default=20)

    # training
    parser.add_argument("--adam_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    obstacle = None
    if args.obs_cx is not None and args.obs_cy is not None and args.obs_r is not None:
        obstacle = (args.obs_cx, args.obs_cy, args.obs_r)

    geom = build_geometry_rect_or_cyl(
        args.x_min, args.x_max, args.y_min, args.y_max,
        obstacle_circle=obstacle
    )

    ns_ckpt = os.path.join(args.outdir, "ns_model")
    flow_json = os.path.join(args.outdir, "flow_fields.json")

    temp_ckpt = os.path.join(args.outdir, "temp_model")
    temp_json = os.path.join(args.outdir, "temperature.json")
    temp_gif = os.path.join(args.outdir, "temperature.gif")

    if args.stage in ["ns", "all"]:
        ns_model = train_navier_stokes(
            geom=geom,
            T_final=args.T_final,
            nu=args.nu,
            V0=args.V0,
            save_model_path=ns_ckpt,
            lr=args.lr,
            adam_epochs=args.adam_epochs,
            lbfgs=True,
        )
        export_flow_fields_json(
            model=ns_model,
            geom=geom,
            T_final=args.T_final,
            nx=args.nx, ny=args.ny, nt=args.nt,
            out_json=flow_json,
        )

    if args.stage in ["temp", "all"]:
        # If you want to skip training and only load previous NS JSON, just run stage=temp
        temp_model = train_temperature(
            geom=geom,
            T_final=args.T_final,
            alpha=args.alpha,
            inlet_T=args.Tin,
            flow_json_path=flow_json,
            save_model_path=temp_ckpt,
            lr=args.lr,
            adam_epochs=args.adam_epochs,
            lbfgs=True,
        )
        export_temperature_json_and_gif(
            model=temp_model,
            geom=geom,
            T_final=args.T_final,
            nx=args.nx, ny=args.ny, nt=args.nt,
            out_json=temp_json,
            out_gif=temp_gif,
            fps=6,
        )


if __name__ == "__main__":
    main()
