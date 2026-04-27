"""Verify V4.1 3D PDE classes wire up end-to-end under PhysicsNeMo-Sym.

Runs six independent checks covering instantiation, equation-key
contracts, node emission, autograd wiring for the 3D Navier-Stokes
class, wall-normal no-penetration evaluation, and a numerical-identity
check on the 3D cross product inside FlowTrajectoryGuidance3D.

Runnable on CPU in under 60 seconds. Exits with non-zero status on any
failure.

    python scripts/verify_v4_1_flow.py
"""

import os  # for path munging
import sys  # for exit codes
import traceback  # for failure reporting


# -----------------------------
# Make src/ importable
# -----------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))  # repo root
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))  # src/ on path


import torch  # autograd + tensor ops
torch.manual_seed(0)  # reproducibility

from physicsnemo.sym.key import Key  # network keys
from physicsnemo.sym.models.fully_connected import FullyConnectedArch  # tiny MLP
from physicsnemo.sym.graph import Graph  # PDE-graph evaluator

from partner_v4_1_physics import (
    SteadyNavierStokes3DScaled,
    WallNormalNoPenetration3D,
    FlowTrajectoryGuidance3D,
)  # the three V4.1 PDE classes


# -----------------------------
# helpers
# -----------------------------
def _ok(msg: str) -> None:
    print(f"[ok] {msg}")  # uniform ok line


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)  # uniform fail line


def _pick_device() -> torch.device:
    # Prefer CUDA when present, fall back to CPU. The verify script must
    # run green on CPU-only machines so we never hard-require CUDA.
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")  # probe
            return torch.device("cuda")  # GPU path
        except Exception:
            pass  # fall back silently
    return torch.device("cpu")  # CPU path


def _rand_inputs(
    device: torch.device, n: int, keys: list[str]
) -> dict[str, torch.Tensor]:
    # Random Nx1 tensors for each input key, float32 on the chosen device.
    out = {}  # invar dict to return
    for k in keys:
        t = torch.rand((n, 1), device=device, dtype=torch.float32)  # random
        t.requires_grad_(True)  # autograd leaves for spatial coords
        out[k] = t  # attach
    return out  # ready to feed to Graph


def _build_flow_net(
    input_keys_str: list[str], output_keys_str: list[str]
) -> FullyConnectedArch:
    # 2 layers x 32 width tiny net matching the spec.
    return FullyConnectedArch(
        input_keys=[Key(k) for k in input_keys_str],  # inputs as Key objects
        output_keys=[Key(k) for k in output_keys_str],  # outputs as Key objects
        layer_size=32,  # narrow for speed
        nr_layers=2,  # shallow for speed
    )  # tiny MLP


# -----------------------------
# Checks
# -----------------------------
def check_1_imports_and_instantiation() -> None:
    SteadyNavierStokes3DScaled(rho=1.0, nu=1.0e-3, Lx=2.0, Ly=1.5, Lz=0.2)  # instantiate NS
    WallNormalNoPenetration3D()  # instantiate wall
    FlowTrajectoryGuidance3D()  # instantiate guidance
    _ok("imports + instantiation")  # pass


def check_2_equation_keys() -> None:
    ns = SteadyNavierStokes3DScaled(rho=1.0, nu=1.0e-3, Lx=2.0, Ly=1.5, Lz=0.2)  # NS
    assert set(ns.equations.keys()) == {
        "continuity",
        "momentum_x",
        "momentum_y",
        "momentum_z",
    }, f"NS eq keys mismatch: {set(ns.equations.keys())}"  # four eqs

    wnp = WallNormalNoPenetration3D()  # wall
    assert set(wnp.equations.keys()) == {
        "wall_normal_velocity",
    }, f"wall eq keys mismatch: {set(wnp.equations.keys())}"  # one eq

    geo = FlowTrajectoryGuidance3D()  # guidance
    assert set(geo.equations.keys()) == {
        "flow_geo_parallel",
        "flow_geo_cosine",
        "flow_geo_cross_x",
        "flow_geo_cross_y",
        "flow_geo_cross_z",
        "flow_geo_speed",
    }, f"guidance eq keys mismatch: {set(geo.equations.keys())}"  # six eqs

    _ok("equation keys")  # pass


def check_3_make_nodes_non_empty() -> None:
    ns = SteadyNavierStokes3DScaled(rho=1.0, nu=1.0e-3, Lx=2.0, Ly=1.5, Lz=0.2)  # NS
    ns_nodes = ns.make_nodes()  # emit nodes
    assert isinstance(ns_nodes, list) and len(ns_nodes) >= 4, (
        f"expected >=4 NS nodes, got {len(ns_nodes)}"
    )  # NS emits 4

    wnp = WallNormalNoPenetration3D()  # wall
    wnp_nodes = wnp.make_nodes()  # emit nodes
    assert isinstance(wnp_nodes, list) and len(wnp_nodes) >= 1, (
        f"expected >=1 wall node, got {len(wnp_nodes)}"
    )  # wall emits 1

    geo = FlowTrajectoryGuidance3D()  # guidance
    geo_nodes = geo.make_nodes()  # emit nodes
    assert isinstance(geo_nodes, list) and len(geo_nodes) >= 6, (
        f"expected >=6 guidance nodes, got {len(geo_nodes)}"
    )  # guidance emits 6

    _ok("make_nodes non-empty")  # pass


def check_4_autograd_flow_ns(device: torch.device) -> None:
    # Build a tiny flow net + NS nodes, evaluate residuals, backprop.
    input_keys = ["x", "y", "z", "dw", "sin", "sout"]  # net inputs
    output_keys = ["u", "v", "w", "p"]  # net outputs
    net = _build_flow_net(input_keys, output_keys).to(device)  # tiny MLP
    ns = SteadyNavierStokes3DScaled(rho=1.0, nu=1.0e-3, Lx=2.0, Ly=1.5, Lz=0.2)  # 3D NS
    nodes = ns.make_nodes() + [net.make_node(name="flow_network")]  # assemble

    req_keys = [Key("continuity"), Key("momentum_x"), Key("momentum_y"), Key("momentum_z")]  # residuals
    invar_keys = [Key(k) for k in input_keys]  # PhysicsNeMo expects Keys
    graph = Graph(nodes=nodes, invar=invar_keys, req_names=req_keys)  # assemble graph
    graph = graph.to(device)  # move to device

    invar = _rand_inputs(device, n=64, keys=input_keys)  # 64 random points
    out = graph(invar)  # evaluate residuals

    loss = sum((out[k] ** 2).mean() for k in ["continuity", "momentum_x", "momentum_y", "momentum_z"])  # SSR
    assert torch.isfinite(loss).item(), f"non-finite residual loss: {loss}"  # sanity

    loss.backward()  # backprop through the graph

    any_nonzero = False  # flag
    any_nonfinite = False  # flag
    for p in net.parameters():
        if p.grad is None:
            continue  # not all params are hit necessarily; that's fine
        if not torch.isfinite(p.grad).all().item():
            any_nonfinite = True  # detected non-finite gradient
            break  # no need to continue
        if float(p.grad.abs().sum().item()) > 0.0:
            any_nonzero = True  # at least one gradient flows

    assert not any_nonfinite, "at least one gradient was non-finite"  # autograd health
    assert any_nonzero, "no non-zero gradients found — autograd graph is not wired"  # must have signal
    _ok(f"3D NS autograd on {device} (loss={float(loss.detach()):.4e})")  # pass


def check_5_wall_normal_eval(device: torch.device) -> None:
    # Feed x,y,z + nx,ny,nz through the wall-normal node and confirm finiteness.
    input_keys = ["x", "y", "z", "n_x", "n_y", "n_z"]  # wall inputs
    net = _build_flow_net(
        input_keys_str=["x", "y", "z"],  # net only needs spatial inputs
        output_keys_str=["u", "v", "w", "p"],  # outputs
    ).to(device)  # tiny MLP

    wnp = WallNormalNoPenetration3D()  # wall class
    nodes = wnp.make_nodes() + [net.make_node(name="flow_network")]  # combine

    req_keys = [Key("wall_normal_velocity")]  # single residual
    invar_keys = [Key(k) for k in input_keys]  # as Keys
    graph = Graph(nodes=nodes, invar=invar_keys, req_names=req_keys)  # assemble
    graph = graph.to(device)  # move

    # spatial coords require grad for the net's forward pass
    invar = {
        "x": torch.rand((32, 1), device=device, dtype=torch.float32, requires_grad=True),
        "y": torch.rand((32, 1), device=device, dtype=torch.float32, requires_grad=True),
        "z": torch.rand((32, 1), device=device, dtype=torch.float32, requires_grad=True),
        "n_x": torch.rand((32, 1), device=device, dtype=torch.float32),
        "n_y": torch.rand((32, 1), device=device, dtype=torch.float32),
        "n_z": torch.rand((32, 1), device=device, dtype=torch.float32),
    }  # mixed invar dict

    out = graph(invar)  # evaluate
    val = out["wall_normal_velocity"]  # grab residual
    assert torch.isfinite(val).all().item(), "non-finite wall_normal_velocity"  # must be finite
    _ok(f"wall_normal_velocity finite on {device} (shape={tuple(val.shape)})")  # pass


def check_6_guidance_eval_and_cross_identity(device: torch.device) -> None:
    # Evaluate all six guidance equations, then numerically verify
    # cross_x = v*gz - w*gy on a forced input (u=0, v=1, w=0, g=(0,0,1)).
    input_keys = ["x", "y", "z", "gx", "gy", "gz"]  # guidance inputs
    net = _build_flow_net(
        input_keys_str=["x", "y", "z"],  # net only needs spatial
        output_keys_str=["u", "v", "w", "p"],  # outputs
    ).to(device)  # tiny MLP

    geo = FlowTrajectoryGuidance3D(speed_eps=1.0e-4)  # guidance class
    nodes = geo.make_nodes() + [net.make_node(name="flow_network")]  # combine

    req_keys = [
        Key("flow_geo_parallel"),
        Key("flow_geo_cosine"),
        Key("flow_geo_cross_x"),
        Key("flow_geo_cross_y"),
        Key("flow_geo_cross_z"),
        Key("flow_geo_speed"),
    ]  # all six outputs
    invar_keys = [Key(k) for k in input_keys]  # as Keys
    graph = Graph(nodes=nodes, invar=invar_keys, req_names=req_keys)  # assemble
    graph = graph.to(device)  # move

    n = 16  # small batch
    invar = {
        "x": torch.rand((n, 1), device=device, dtype=torch.float32, requires_grad=True),
        "y": torch.rand((n, 1), device=device, dtype=torch.float32, requires_grad=True),
        "z": torch.rand((n, 1), device=device, dtype=torch.float32, requires_grad=True),
        "gx": torch.rand((n, 1), device=device, dtype=torch.float32),
        "gy": torch.rand((n, 1), device=device, dtype=torch.float32),
        "gz": torch.rand((n, 1), device=device, dtype=torch.float32),
    }  # random inputs
    out = graph(invar)  # evaluate
    for k in [
        "flow_geo_parallel",
        "flow_geo_cosine",
        "flow_geo_cross_x",
        "flow_geo_cross_y",
        "flow_geo_cross_z",
        "flow_geo_speed",
    ]:
        assert torch.isfinite(out[k]).all().item(), f"non-finite {k}"  # each output finite
        assert out[k].shape == (n, 1), f"unexpected shape for {k}: {out[k].shape}"  # (n, 1)

    # Identity check: with u=0, v=1, w=0, g=(0,0,1) we expect
    #   cross_x = v*gz - w*gy = 1*1 - 0*0 = 1
    #   cross_y = w*gx - u*gz = 0
    #   cross_z = u*gy - v*gx = 0
    # We cannot set u,v,w directly through a neural net; instead we
    # build a tiny graph that bypasses the net and feeds u,v,w via an
    # identity node, reusing only the guidance PDE nodes.
    from physicsnemo.sym.node import Node  # local import keeps global scope clean

    # Identity nodes: each output key is just a pass-through of an input tensor.
    # PhysicsNeMo wraps nodes in an nn.ModuleList so the evaluator has to be an
    # nn.Module, not a plain Python function.
    class _EchoModule(torch.nn.Module):
        def __init__(self, name: str):
            super().__init__()  # init nn.Module
            self._out_name = name  # which key to echo

        def forward(self, invar_local):
            return {self._out_name: invar_local[self._out_name]}  # echo

    def _make_identity_node(name: str) -> Node:
        # A node whose evaluate returns {name: invar[name]}. Lets us feed
        # u, v, w directly as "input" keys and read them back as if they came
        # from the network.
        from physicsnemo.sym.key import Key as _Key  # alias

        mod = _EchoModule(name)  # nn.Module echoing the named input
        return Node(
            inputs=[_Key(name)],  # single input of that name
            outputs=[_Key(name)],  # same name as output
            evaluate=mod,  # Module echoer (required by ModuleList)
            name=f"identity_{name}",  # readable node name
        )  # identity pass-through

    identity_nodes = [_make_identity_node(k) for k in ("u", "v", "w")]  # echo u,v,w
    nodes2 = geo.make_nodes() + identity_nodes  # guidance + echoes
    req_keys2 = [Key("flow_geo_cross_x"), Key("flow_geo_cross_y"), Key("flow_geo_cross_z")]  # cross only
    invar_keys2 = [Key(k) for k in ("u", "v", "w", "gx", "gy", "gz")]  # direct-feed inputs
    graph2 = Graph(nodes=nodes2, invar=invar_keys2, req_names=req_keys2)  # assemble
    graph2 = graph2.to(device)  # move

    m = 4  # tiny batch
    invar2 = {
        "u": torch.zeros((m, 1), device=device, dtype=torch.float32),  # u = 0
        "v": torch.ones((m, 1), device=device, dtype=torch.float32),  # v = 1
        "w": torch.zeros((m, 1), device=device, dtype=torch.float32),  # w = 0
        "gx": torch.zeros((m, 1), device=device, dtype=torch.float32),  # gx = 0
        "gy": torch.zeros((m, 1), device=device, dtype=torch.float32),  # gy = 0
        "gz": torch.ones((m, 1), device=device, dtype=torch.float32),  # gz = 1
    }  # forced inputs
    out2 = graph2(invar2)  # evaluate
    cx = out2["flow_geo_cross_x"].detach().cpu()  # pull x-component
    cy = out2["flow_geo_cross_y"].detach().cpu()  # pull y-component
    cz = out2["flow_geo_cross_z"].detach().cpu()  # pull z-component

    assert torch.allclose(cx, torch.ones_like(cx), atol=1.0e-6), (
        f"expected cross_x == 1, got {cx.flatten().tolist()}"
    )  # v*gz - w*gy = 1*1 - 0 = 1
    assert torch.allclose(cy, torch.zeros_like(cy), atol=1.0e-6), (
        f"expected cross_y == 0, got {cy.flatten().tolist()}"
    )  # w*gx - u*gz = 0
    assert torch.allclose(cz, torch.zeros_like(cz), atol=1.0e-6), (
        f"expected cross_z == 0, got {cz.flatten().tolist()}"
    )  # u*gy - v*gx = 0
    _ok("guidance outputs finite + cross identity (u=0,v=1,w=0,g=z) holds")  # pass


# -----------------------------
# main
# -----------------------------
def main() -> int:
    device = _pick_device()  # CPU or CUDA
    try:
        check_1_imports_and_instantiation()  # 1
        check_2_equation_keys()  # 2
        check_3_make_nodes_non_empty()  # 3
        check_4_autograd_flow_ns(device)  # 4
        check_5_wall_normal_eval(device)  # 5
        check_6_guidance_eval_and_cross_identity(device)  # 6
    except Exception as exc:
        _fail(f"verification failed: {exc}")  # log
        traceback.print_exc()  # show traceback
        return 1  # non-zero exit
    print("verify_v4_1_flow.py: ALL CHECKS PASSED")  # success marker
    return 0  # success


if __name__ == "__main__":
    sys.exit(main())  # forward exit code
