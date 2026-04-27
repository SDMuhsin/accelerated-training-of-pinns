"""V4.1 physics: 3D PDE classes for PhysicsNeMo-Sym.

This module defines the three sympy-backed PDE classes used by the V4.1
3D steady Navier-Stokes flow trainer:

    - SteadyNavierStokes3DScaled: 3D incompressible steady NS residuals.
    - WallNormalNoPenetration3D:  u*nx + v*ny + w*nz = 0 at walls.
    - FlowTrajectoryGuidance3D:   per-point 3D tangent-guidance scalars.

The module is the direct 3D extension of the 2D classes that live inline
in `src/partner_v4_flow.py` (SteadyNavierStokes2DScaled,
WallNormalNoPenetration2D, FlowTrajectoryGuidance2D). The style, the
per-line-comment convention, and the invL-based derivative scaling are
matched to the 2D originals so that V4.1 differs from V4 only by the 3D
lift.

References:
    - `./llmdocs/stream_battery_consortium/V4_1_DESIGN.md` section 3.
    - `src/partner_v4_flow.py` lines 867-950 (the 2D originals).
"""

from sympy import Symbol, Function, Number, sqrt  # sympy primitives
from sympy import Derivative as D  # shorter alias, matches V4
from physicsnemo.sym.eq.pde import PDE  # PhysicsNeMo PDE base class


# -----------------------------
# Steady 3D Navier-Stokes
# -----------------------------
class SteadyNavierStokes3DScaled(PDE):
    """Steady incompressible 3D Navier-Stokes in normalised coordinates.

    Residual equations (all set equal to 0 by the loss):

        continuity  = ux + vy + wz
        momentum_x  = u*ux + v*uy + w*uz + (1/rho)*px - nu*(uxx + uyy + uzz)
        momentum_y  = u*vx + v*vy + w*vz + (1/rho)*py - nu*(vxx + vyy + vzz)
        momentum_z  = u*wx + v*wy + w*wz + (1/rho)*pz - nu*(wxx + wyy + wzz)

    Derivatives are scaled by invLx, invLy, invLz so that derivatives
    evaluated in the normalised [0, 1]-ish coordinates match physical
    derivatives, mirroring the 2D pattern used in V4.
    """

    name = "SteadyNavierStokes3DScaled"  # class-level name for PhysicsNeMo

    def __init__(
        self,
        rho: float = 1.0,
        nu: float = 1.0e-3,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Lz: float = 1.0,
    ):
        super().__init__()  # init PDE base class

        x = Symbol("x")  # normalized x coordinate
        y = Symbol("y")  # normalized y coordinate
        z = Symbol("z")  # normalized z coordinate

        rhoN = Number(float(rho))  # density constant as sympy Number
        nuN = Number(float(nu))  # kinematic viscosity as sympy Number
        invLx = Number(1.0 / max(float(Lx), 1.0e-12))  # x derivative scale
        invLy = Number(1.0 / max(float(Ly), 1.0e-12))  # y derivative scale
        invLz = Number(1.0 / max(float(Lz), 1.0e-12))  # z derivative scale
        invLx2 = Number(float(invLx) ** 2)  # x second derivative scale
        invLy2 = Number(float(invLy) ** 2)  # y second derivative scale
        invLz2 = Number(float(invLz) ** 2)  # z second derivative scale

        u = Function("u")(x, y, z)  # u velocity field
        v = Function("v")(x, y, z)  # v velocity field
        w = Function("w")(x, y, z)  # w velocity field
        p = Function("p")(x, y, z)  # pressure field

        ux = invLx * D(u, x)  # du/dx_phys
        uy = invLy * D(u, y)  # du/dy_phys
        uz = invLz * D(u, z)  # du/dz_phys
        vx = invLx * D(v, x)  # dv/dx_phys
        vy = invLy * D(v, y)  # dv/dy_phys
        vz = invLz * D(v, z)  # dv/dz_phys
        wx = invLx * D(w, x)  # dw/dx_phys
        wy = invLy * D(w, y)  # dw/dy_phys
        wz = invLz * D(w, z)  # dw/dz_phys
        px = invLx * D(p, x)  # dp/dx_phys
        py = invLy * D(p, y)  # dp/dy_phys
        pz = invLz * D(p, z)  # dp/dz_phys

        uxx = invLx2 * D(u, x, 2)  # d2u/dx_phys2
        uyy = invLy2 * D(u, y, 2)  # d2u/dy_phys2
        uzz = invLz2 * D(u, z, 2)  # d2u/dz_phys2
        vxx = invLx2 * D(v, x, 2)  # d2v/dx_phys2
        vyy = invLy2 * D(v, y, 2)  # d2v/dy_phys2
        vzz = invLz2 * D(v, z, 2)  # d2v/dz_phys2
        wxx = invLx2 * D(w, x, 2)  # d2w/dx_phys2
        wyy = invLy2 * D(w, y, 2)  # d2w/dy_phys2
        wzz = invLz2 * D(w, z, 2)  # d2w/dz_phys2

        inv_rho = Number(1.0) / rhoN  # 1/rho factor for pressure term

        self.equations = {
            "continuity": ux + vy + wz,  # mass conservation
            "momentum_x": u * ux + v * uy + w * uz + inv_rho * px - nuN * (uxx + uyy + uzz),  # x-momentum
            "momentum_y": u * vx + v * vy + w * vz + inv_rho * py - nuN * (vxx + vyy + vzz),  # y-momentum
            "momentum_z": u * wx + v * wy + w * wz + inv_rho * pz - nuN * (wxx + wyy + wzz),  # z-momentum
        }  # four-equation steady 3D NS residual set


# -----------------------------
# Wall-normal no-penetration (3D)
# -----------------------------
class WallNormalNoPenetration3D(PDE):
    """Enforces u*nx + v*ny + w*nz = 0 at a wall or wall-adjacent point.

    The wall-normal components (nx, ny, nz) are sympy Symbols (not
    Functions of x, y, z): they are per-point features supplied via the
    PhysicsNeMo PointwiseConstraint invar dict, matching how the 2D
    WallNormalNoPenetration2D handles (n_x, n_y).
    """

    name = "WallNormalNoPenetration3D"  # class-level name

    def __init__(self, eq_name: str = "wall_normal_velocity"):
        super().__init__()  # init PDE base class

        x = Symbol("x")  # normalized x coordinate
        y = Symbol("y")  # normalized y coordinate
        z = Symbol("z")  # normalized z coordinate

        u = Function("u")(x, y, z)  # u velocity field
        v = Function("v")(x, y, z)  # v velocity field
        w = Function("w")(x, y, z)  # w velocity field

        n_x = Symbol("n_x")  # wall normal x component (point feature)
        n_y = Symbol("n_y")  # wall normal y component (point feature)
        n_z = Symbol("n_z")  # wall normal z component (point feature)

        self.equations = {
            str(eq_name): u * n_x + v * n_y + w * n_z,  # no-penetration residual
        }  # single-equation wall-normal constraint


# -----------------------------
# Flow trajectory guidance (3D)
# -----------------------------
class FlowTrajectoryGuidance3D(PDE):
    """Per-point 3D guidance scalars along a unit tangent g = (gx, gy, gz).

    Emits six scalar equations that the trainer maps onto independent
    PointwiseConstraint losses (each with its own target value):

        flow_geo_parallel  = u*gx + v*gy + w*gz                       (scalar)
        flow_geo_speed     = sqrt(u*u + v*v + w*w + speed_eps*speed_eps)
        flow_geo_cosine    = parallel / speed                         (in [-1, 1])
        flow_geo_cross_x   = v*gz - w*gy                              (cross-x)
        flow_geo_cross_y   = w*gx - u*gz                              (cross-y)
        flow_geo_cross_z   = u*gy - v*gx                              (cross-z)

    The 3D cross product is split into its three Cartesian components
    because PhysicsNeMo's PointwiseConstraint drives scalar outputs to
    targets; splitting keeps the loss mapping as clean as the 2D case,
    where FlowTrajectoryGuidance2D emits a single scalar flow_geo_cross
    (there was only one non-trivial cross component in 2D).

    `gx, gy, gz` are sympy Symbols (per-point features from the invar
    dict), not Functions of (x, y, z) - they are spatial constants at a
    single sample and are supplied directly by the trainer.
    """

    name = "FlowTrajectoryGuidance3D"  # class-level name

    def __init__(
        self,
        eq_parallel: str = "flow_geo_parallel",
        eq_speed: str = "flow_geo_speed",
        eq_cosine: str = "flow_geo_cosine",
        eq_cross_x: str = "flow_geo_cross_x",
        eq_cross_y: str = "flow_geo_cross_y",
        eq_cross_z: str = "flow_geo_cross_z",
        speed_eps: float = 1.0e-4,
    ):
        super().__init__()  # init PDE base class

        x = Symbol("x")  # normalized x coordinate
        y = Symbol("y")  # normalized y coordinate
        z = Symbol("z")  # normalized z coordinate

        u = Function("u")(x, y, z)  # u velocity field
        v = Function("v")(x, y, z)  # v velocity field
        w = Function("w")(x, y, z)  # w velocity field

        g_x = Symbol("gx")  # desired direction x (point feature)
        g_y = Symbol("gy")  # desired direction y (point feature)
        g_z = Symbol("gz")  # desired direction z (point feature)

        eps = Number(float(max(speed_eps, 1.0e-12)))  # speed epsilon as sympy Number
        speed = sqrt(u * u + v * v + w * w + eps * eps)  # speed magnitude with eps
        parallel = u * g_x + v * g_y + w * g_z  # along-tangent projection
        cosine = parallel / speed  # cosine alignment with tangent

        cross_x = v * g_z - w * g_y  # cross product x component
        cross_y = w * g_x - u * g_z  # cross product y component
        cross_z = u * g_y - v * g_x  # cross product z component

        self.equations = {
            str(eq_parallel): parallel,  # scalar along-tangent component
            str(eq_cosine): cosine,  # cosine alignment in [-1, 1]
            str(eq_cross_x): cross_x,  # cross x (target 0)
            str(eq_cross_y): cross_y,  # cross y (target 0)
            str(eq_cross_z): cross_z,  # cross z (target 0)
            str(eq_speed): speed,  # speed magnitude (> 0)
        }  # six-equation 3D trajectory-guidance set


# -----------------------------
# In-module self-test
# -----------------------------
if __name__ == "__main__":
    # Minimal sanity-check: instantiate every class and assert equation keys.
    ns = SteadyNavierStokes3DScaled(rho=1.0, nu=1.0e-3, Lx=2.0, Ly=1.5, Lz=0.2)  # 3D NS
    assert set(ns.equations.keys()) == {
        "continuity",
        "momentum_x",
        "momentum_y",
        "momentum_z",
    }, f"unexpected NS keys: {set(ns.equations.keys())}"  # four-equation check

    wnp = WallNormalNoPenetration3D()  # wall no-penetration
    assert set(wnp.equations.keys()) == {"wall_normal_velocity"}, (
        f"unexpected wall-normal keys: {set(wnp.equations.keys())}"
    )  # single-equation check

    geo = FlowTrajectoryGuidance3D()  # guidance scalars
    assert set(geo.equations.keys()) == {
        "flow_geo_parallel",
        "flow_geo_cosine",
        "flow_geo_cross_x",
        "flow_geo_cross_y",
        "flow_geo_cross_z",
        "flow_geo_speed",
    }, f"unexpected guidance keys: {set(geo.equations.keys())}"  # six-equation check

    print("physics module OK")  # success marker
