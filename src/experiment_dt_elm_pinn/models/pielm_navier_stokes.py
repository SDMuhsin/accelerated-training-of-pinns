"""
PIELM for Navier-Stokes: Physics-Informed ELM for Incompressible Flow

Solves the steady-state incompressible Navier-Stokes equations with
Smagorinsky turbulence model using Picard iteration.

The key insight: at each Picard iteration, the linearized system is solved
in ONE least-squares step (no gradient descent), giving ~1000x speedup over PINN.

Equations (steady incompressible NS with Smagorinsky):
    Continuity: ∂u/∂x + ∂v/∂y = 0
    Momentum-x: u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ∇·(ν_eff·∇u) = 0
    Momentum-y: u·∂v/∂x + v·∂v/∂y + ∂p/∂y - ∇·(ν_eff·∇v) = 0

Where:
    ν_eff = ν_laminar + ν_turb (Smagorinsky)
    ν_turb = (Cs·d)² · |S|
    |S| = sqrt(2·(S_xx² + S_yy² + 2·S_xy²))
    S_xx = ∂u/∂x, S_yy = ∂v/∂y, S_xy = 0.5·(∂u/∂y + ∂v/∂x)
    d = distance to nearest wall

Picard Linearization:
    At iteration k, given (u^k, v^k), solve for (u^{k+1}, v^{k+1}, p^{k+1}):
    - Convective terms use KNOWN velocities: u^k·∂u^{k+1}/∂x + v^k·∂u^{k+1}/∂y
    - ν_eff^k computed from (u^k, v^k) gradients
    - Result is LINEAR in unknowns → single least-squares solve
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time
from scipy.sparse.linalg import lsqr


class PIELM_NavierStokes:
    """
    Physics-Informed Extreme Learning Machine for Navier-Stokes.

    Drop-in replacement for partner team's PINN_Cavity class.
    Same interface: initialize, train, predict(xy) -> (u, v, p)

    Attributes:
        Re: Reynolds number
        U_lid: Lid velocity
        nu_laminar: Kinematic viscosity (= U_lid / Re)
        Cs: Smagorinsky constant
        n_hidden: Number of hidden neurons
    """

    def __init__(
        self,
        Re: float = 1000.0,
        U_lid: float = 1.0,
        Cs: float = 0.1,
        n_hidden: int = 500,
        activation: str = 'tanh',
        weight_range: float = 1.0,
        max_picard_iter: int = 50,
        tol: float = 1e-6,
        seed: int = 42,
        N_interior: int = 6000,
        N_wall: int = 800,
        N_lid: int = 800,
        bc_weight: float = 10.0,
        verbose: bool = False,
    ):
        """
        Args:
            Re: Reynolds number
            U_lid: Lid velocity (boundary condition at y=1)
            Cs: Smagorinsky constant
            n_hidden: Number of hidden neurons
            activation: Activation function ('tanh', 'sigmoid', 'sin')
            weight_range: Random weights in [-range, range]
            max_picard_iter: Maximum Picard iterations
            tol: Convergence tolerance for Picard iteration
            seed: Random seed
            N_interior: Number of interior collocation points
            N_wall: Number of wall boundary points (bottom, left, right)
            N_lid: Number of lid boundary points (top)
            bc_weight: Boundary condition weight in least squares
            verbose: Print progress
        """
        self.Re = Re
        self.U_lid = U_lid
        self.nu_laminar = U_lid / Re
        self.Cs = Cs

        self.n_hidden = n_hidden
        self.activation = activation
        self.weight_range = weight_range
        self.max_picard_iter = max_picard_iter
        self.tol = tol
        self.seed = seed

        self.N_interior = N_interior
        self.N_wall = N_wall
        self.N_lid = N_lid
        self.bc_weight = bc_weight
        self.verbose = verbose

        # Under-relaxation parameter for Picard iteration
        self.relaxation = 0.7  # 0 < relaxation <= 1 (1 = no relaxation)

        # Solver options: 'direct' (lstsq) or 'iterative' (lsqr)
        self.solver = 'direct'  # 'direct' for small problems, 'iterative' for large
        self.lsqr_iter_lim = 2000  # Max iterations for iterative solver

        # Full viscous term: include ∇ν_eff·∇u (matches partner's PINN exactly)
        # Set to False for simplified model (faster but less accurate)
        self.use_full_viscous = True

        # Network weights (to be initialized)
        self.W = None  # Input weights (2, n_hidden)
        self.b = None  # Biases (n_hidden,)

        # Output weights for each field (solved during training)
        self.beta_u = None  # (n_hidden,)
        self.beta_v = None  # (n_hidden,)
        self.beta_p = None  # (n_hidden,)

        # Training points (to be sampled)
        self.xy_interior = None
        self.xy_lid = None
        self.xy_wall = None
        self.xy_p_anchor = None

        # Training state
        self.is_trained = False
        self.train_time = None
        self.n_iterations = None
        self.residual_history = []

    def _sample_domain(self):
        """Sample collocation points in the lid-driven cavity domain [0,1]²."""
        np.random.seed(self.seed)

        # Interior points
        x = np.random.rand(self.N_interior, 1)
        y = np.random.rand(self.N_interior, 1)
        self.xy_interior = np.hstack((x, y))

        # Lid (y=1)
        x_lid = np.random.rand(self.N_lid, 1)
        y_lid = np.ones((self.N_lid, 1))
        self.xy_lid = np.hstack((x_lid, y_lid))

        # Walls (bottom, left, right)
        N_each = self.N_wall // 3

        # Bottom (y=0)
        xb = np.random.rand(N_each, 1)
        yb = np.zeros((N_each, 1))

        # Left (x=0)
        xl = np.zeros((N_each, 1))
        yl = np.random.rand(N_each, 1)

        # Right (x=1)
        xr = np.ones((N_each, 1))
        yr = np.random.rand(N_each, 1)

        self.xy_wall = np.vstack([
            np.hstack((xb, yb)),
            np.hstack((xl, yl)),
            np.hstack((xr, yr)),
        ])

        # Pressure anchor (center point where p=0)
        self.xy_p_anchor = np.array([[0.5, 0.5]])

    def _initialize_weights(self):
        """Initialize random hidden layer weights."""
        np.random.seed(self.seed + 1)  # Different seed from sampling

        self.W = np.random.uniform(
            -self.weight_range, self.weight_range,
            (2, self.n_hidden)
        )

        self.b = np.random.uniform(
            -self.weight_range, self.weight_range,
            self.n_hidden
        )

    def _activation_fn(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'sin':
            return np.sin(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """First derivative of activation function."""
        if self.activation == 'tanh':
            t = np.tanh(z)
            return 1 - t**2
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif self.activation == 'sin':
            return np.cos(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_second_derivative(self, z: np.ndarray) -> np.ndarray:
        """Second derivative of activation function."""
        if self.activation == 'tanh':
            t = np.tanh(z)
            return -2 * t * (1 - t**2)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s) * (1 - 2*s)
        elif self.activation == 'sin':
            return -np.sin(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _compute_features(self, xy: np.ndarray) -> np.ndarray:
        """
        Compute hidden layer features H.

        H[i,j] = σ(W[:,j]·xy[i,:] + b[j])

        Args:
            xy: Input coordinates (N, 2)

        Returns:
            H: Hidden features (N, n_hidden)
        """
        z = xy @ self.W + self.b  # (N, n_hidden)
        return self._activation_fn(z)

    def _compute_gradient_features(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of hidden layer features.

        ∂H_j/∂x = σ'(z_j) * W[0,j]
        ∂H_j/∂y = σ'(z_j) * W[1,j]

        Args:
            xy: Input coordinates (N, 2)

        Returns:
            dH_dx, dH_dy: (N, n_hidden) each
        """
        z = xy @ self.W + self.b
        sigma_p = self._activation_derivative(z)

        dH_dx = sigma_p * self.W[0, :]
        dH_dy = sigma_p * self.W[1, :]

        return dH_dx, dH_dy

    def _compute_laplacian_features(self, xy: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of hidden layer features.

        ∇²H_j = σ''(z_j) * (W[0,j]² + W[1,j]²)

        Args:
            xy: Input coordinates (N, 2)

        Returns:
            LapH: (N, n_hidden)
        """
        z = xy @ self.W + self.b
        sigma_pp = self._activation_second_derivative(z)
        W_norm_sq = self.W[0, :]**2 + self.W[1, :]**2

        return sigma_pp * W_norm_sq

    def _compute_second_derivative_features(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute second derivatives of hidden layer features.

        ∂²H_j/∂x² = σ''(z_j) * W[0,j]²
        ∂²H_j/∂x∂y = σ''(z_j) * W[0,j] * W[1,j]
        ∂²H_j/∂y² = σ''(z_j) * W[1,j]²

        Args:
            xy: Input coordinates (N, 2)

        Returns:
            d2H_dxx, d2H_dxy, d2H_dyy: (N, n_hidden) each
        """
        z = xy @ self.W + self.b
        sigma_pp = self._activation_second_derivative(z)

        d2H_dxx = sigma_pp * (self.W[0, :] ** 2)
        d2H_dxy = sigma_pp * (self.W[0, :] * self.W[1, :])
        d2H_dyy = sigma_pp * (self.W[1, :] ** 2)

        return d2H_dxx, d2H_dxy, d2H_dyy

    def _compute_wall_distance(self, xy: np.ndarray) -> np.ndarray:
        """
        Compute distance to nearest wall for Smagorinsky model.
        Domain is [0,1]² with walls at x=0, x=1, y=0, y=1.

        Args:
            xy: Coordinates (N, 2)

        Returns:
            d: Distance to nearest wall (N,)
        """
        x = xy[:, 0]
        y = xy[:, 1]

        d_left = x
        d_right = 1.0 - x
        d_bottom = y
        d_top = 1.0 - y

        d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        return d

    def _compute_wall_distance_gradient(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of wall distance function.

        For unit square domain, d = min(x, 1-x, y, 1-y)

        Args:
            xy: Coordinates (N, 2)

        Returns:
            dd_dx, dd_dy: Gradient of wall distance (N,) each
        """
        x = xy[:, 0]
        y = xy[:, 1]

        d_left = x
        d_right = 1.0 - x
        d_bottom = y
        d_top = 1.0 - y

        # Find which wall is closest
        d_x = np.where(d_left < d_right, d_left, d_right)
        d_y = np.where(d_bottom < d_top, d_bottom, d_top)

        # Gradient depends on which wall is closest
        dd_dx = np.where(d_x < d_y,
                        np.where(d_left < d_right, 1.0, -1.0),
                        0.0)
        dd_dy = np.where(d_y <= d_x,
                        np.where(d_bottom < d_top, 1.0, -1.0),
                        0.0)

        return dd_dx, dd_dy

    def _compute_eddy_viscosity(
        self,
        xy: np.ndarray,
        du_dx: np.ndarray,
        du_dy: np.ndarray,
        dv_dx: np.ndarray,
        dv_dy: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Smagorinsky eddy viscosity.

        ν_eff = ν_laminar + (Cs·d)² · |S|

        where:
            |S| = sqrt(2·(S_xx² + S_yy² + 2·S_xy²))
            S_xx = ∂u/∂x, S_yy = ∂v/∂y, S_xy = 0.5·(∂u/∂y + ∂v/∂x)
            d = distance to nearest wall

        Args:
            xy: Coordinates
            du_dx, du_dy, dv_dx, dv_dy: Velocity gradients

        Returns:
            nu_eff: Effective viscosity (N,)
        """
        d = self._compute_wall_distance(xy)

        # Strain rate components
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)

        # Strain rate magnitude
        S_sq = 2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2)
        S_mag = np.sqrt(S_sq + 1e-12)

        # Smagorinsky turbulent viscosity
        nu_turb = (self.Cs * d)**2 * S_mag

        nu_eff = self.nu_laminar + nu_turb
        return nu_eff

    def _compute_eddy_viscosity_gradient(
        self,
        xy: np.ndarray,
        du_dx: np.ndarray,
        du_dy: np.ndarray,
        dv_dx: np.ndarray,
        dv_dy: np.ndarray,
        d2u_dxx: np.ndarray,
        d2u_dxy: np.ndarray,
        d2u_dyy: np.ndarray,
        d2v_dxx: np.ndarray,
        d2v_dxy: np.ndarray,
        d2v_dyy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of eddy viscosity ∇ν_eff.

        ν_turb = (Cs·d)² · |S|
        ∂ν_turb/∂x = 2·Cs²·d·(∂d/∂x)·|S| + (Cs·d)²·(∂|S|/∂x)

        This is needed for the full divergence form: ∇·(ν_eff·∇u) = ν_eff·∇²u + ∇ν_eff·∇u

        Args:
            xy: Coordinates
            du_dx, du_dy, dv_dx, dv_dy: First derivatives of velocity
            d2u_dxx, etc.: Second derivatives of velocity

        Returns:
            dnu_dx, dnu_dy: Gradient of effective viscosity (N,) each
        """
        d = self._compute_wall_distance(xy)
        dd_dx, dd_dy = self._compute_wall_distance_gradient(xy)

        # Strain rate components
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)

        # Strain rate magnitude
        S_sq = 2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2)
        S_mag = np.sqrt(S_sq + 1e-12)

        # Derivatives of strain rate components
        dS_xx_dx = d2u_dxx
        dS_xx_dy = d2u_dxy
        dS_yy_dx = d2v_dxy
        dS_yy_dy = d2v_dyy
        dS_xy_dx = 0.5 * (d2u_dxy + d2v_dxx)
        dS_xy_dy = 0.5 * (d2u_dyy + d2v_dxy)

        # Derivative of |S|
        # |S| = sqrt(2*(S_xx^2 + S_yy^2 + 2*S_xy^2))
        # d|S|/dx = (1/|S|) * (2*S_xx*dS_xx/dx + 2*S_yy*dS_yy/dx + 4*S_xy*dS_xy/dx)
        dS_mag_dx = (1.0 / (S_mag + 1e-12)) * (
            2.0 * S_xx * dS_xx_dx + 2.0 * S_yy * dS_yy_dx + 4.0 * S_xy * dS_xy_dx
        )
        dS_mag_dy = (1.0 / (S_mag + 1e-12)) * (
            2.0 * S_xx * dS_xx_dy + 2.0 * S_yy * dS_yy_dy + 4.0 * S_xy * dS_xy_dy
        )

        # Derivative of ν_turb = (Cs*d)^2 * |S|
        # dν_turb/dx = 2*Cs^2*d*(dd/dx)*|S| + (Cs*d)^2*(d|S|/dx)
        Cs2 = self.Cs ** 2
        dnu_turb_dx = 2.0 * Cs2 * d * dd_dx * S_mag + Cs2 * d**2 * dS_mag_dx
        dnu_turb_dy = 2.0 * Cs2 * d * dd_dy * S_mag + Cs2 * d**2 * dS_mag_dy

        # ν_eff = ν_laminar + ν_turb, and ν_laminar is constant
        return dnu_turb_dx, dnu_turb_dy

    def _build_linearized_system(
        self,
        u_k: np.ndarray,
        v_k: np.ndarray,
        du_dx_k: np.ndarray,
        du_dy_k: np.ndarray,
        dv_dx_k: np.ndarray,
        dv_dy_k: np.ndarray,
        nu_eff_k: np.ndarray,
        dnu_dx_k: np.ndarray = None,
        dnu_dy_k: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the linearized system matrix for Picard iteration.

        The system solves for [beta_u, beta_v, beta_p] jointly.

        Linearized equations (using known u^k, v^k, ν_eff^k, ∇ν_eff^k):
            Continuity: ∂u^{k+1}/∂x + ∂v^{k+1}/∂y = 0

            Momentum-x (FULL divergence form):
            u^k·∂u/∂x + v^k·∂u/∂y + ∂p/∂x - ∇·(ν_eff·∇u) = 0
            where ∇·(ν_eff·∇u) = ν_eff·∇²u + (∂ν_eff/∂x)·(∂u/∂x) + (∂ν_eff/∂y)·(∂u/∂y)

            Rearranged:
            [u^k - ∂ν_eff/∂x]·∂u/∂x + [v^k - ∂ν_eff/∂y]·∂u/∂y + ∂p/∂x - ν_eff·∇²u = 0

        Returns:
            A: System matrix
            b: Right-hand side
        """
        N_int = self.xy_interior.shape[0]
        N_lid = self.xy_lid.shape[0]
        N_wall = self.xy_wall.shape[0]
        N_anchor = 1

        # Compute features and derivatives at interior points
        H_int = self._compute_features(self.xy_interior)
        dH_dx_int, dH_dy_int = self._compute_gradient_features(self.xy_interior)
        LapH_int = self._compute_laplacian_features(self.xy_interior)

        # Boundary features
        H_lid = self._compute_features(self.xy_lid)
        H_wall = self._compute_features(self.xy_wall)
        H_anchor = self._compute_features(self.xy_p_anchor)

        n = self.n_hidden
        sqrt_w = np.sqrt(self.bc_weight)  # BC weight

        # --- Build system matrix blocks ---
        # We solve for [beta_u; beta_v; beta_p] of size (3*n_hidden,)

        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        # → dH_dx @ beta_u + dH_dy @ beta_v = 0
        A_cont_u = dH_dx_int
        A_cont_v = dH_dy_int
        A_cont_p = np.zeros((N_int, n))

        # Prepare coefficient arrays
        u_k_col = u_k[:, np.newaxis] if u_k.ndim == 1 else u_k
        v_k_col = v_k[:, np.newaxis] if v_k.ndim == 1 else v_k
        nu_col = nu_eff_k[:, np.newaxis] if nu_eff_k.ndim == 1 else nu_eff_k

        # Handle ∇ν_eff gradient (use zero if not provided - simplified model)
        if dnu_dx_k is not None and dnu_dy_k is not None:
            dnu_dx_col = dnu_dx_k[:, np.newaxis] if dnu_dx_k.ndim == 1 else dnu_dx_k
            dnu_dy_col = dnu_dy_k[:, np.newaxis] if dnu_dy_k.ndim == 1 else dnu_dy_k
        else:
            # Fall back to simplified model (no ∇ν_eff term)
            dnu_dx_col = np.zeros((N_int, 1))
            dnu_dy_col = np.zeros((N_int, 1))

        # Momentum-x (FULL form):
        # [u^k - ∂ν_eff/∂x]·∂u/∂x + [v^k - ∂ν_eff/∂y]·∂u/∂y + ∂p/∂x - ν_eff·∇²u = 0
        coeff_dudx = u_k_col - dnu_dx_col
        coeff_dudy = v_k_col - dnu_dy_col

        A_momx_u = coeff_dudx * dH_dx_int + coeff_dudy * dH_dy_int - nu_col * LapH_int
        A_momx_v = np.zeros((N_int, n))
        A_momx_p = dH_dx_int

        # Momentum-y (FULL form):
        # [u^k - ∂ν_eff/∂x]·∂v/∂x + [v^k - ∂ν_eff/∂y]·∂v/∂y + ∂p/∂y - ν_eff·∇²v = 0
        A_momy_u = np.zeros((N_int, n))
        A_momy_v = coeff_dudx * dH_dx_int + coeff_dudy * dH_dy_int - nu_col * LapH_int
        A_momy_p = dH_dy_int

        # --- Boundary conditions ---
        # Lid (y=1): u = U_lid, v = 0
        A_lid_u_bc = H_lid * sqrt_w
        A_lid_u_v = np.zeros((N_lid, n))
        A_lid_u_p = np.zeros((N_lid, n))

        A_lid_v_u = np.zeros((N_lid, n))
        A_lid_v_bc = H_lid * sqrt_w
        A_lid_v_p = np.zeros((N_lid, n))

        # Walls (bottom, left, right): u = 0, v = 0
        A_wall_u_bc = H_wall * sqrt_w
        A_wall_u_v = np.zeros((N_wall, n))
        A_wall_u_p = np.zeros((N_wall, n))

        A_wall_v_u = np.zeros((N_wall, n))
        A_wall_v_bc = H_wall * sqrt_w
        A_wall_v_p = np.zeros((N_wall, n))

        # Pressure anchor: p(0.5, 0.5) = 0
        A_anchor_u = np.zeros((N_anchor, n))
        A_anchor_v = np.zeros((N_anchor, n))
        A_anchor_p = H_anchor * sqrt_w

        # --- Assemble full system ---
        # Stack rows for PDE constraints
        A_pde = np.vstack([
            np.hstack([A_cont_u, A_cont_v, A_cont_p]),
            np.hstack([A_momx_u, A_momx_v, A_momx_p]),
            np.hstack([A_momy_u, A_momy_v, A_momy_p]),
        ])

        # Stack rows for boundary conditions
        A_bc = np.vstack([
            np.hstack([A_lid_u_bc, A_lid_u_v, A_lid_u_p]),
            np.hstack([A_lid_v_u, A_lid_v_bc, A_lid_v_p]),
            np.hstack([A_wall_u_bc, A_wall_u_v, A_wall_u_p]),
            np.hstack([A_wall_v_u, A_wall_v_bc, A_wall_v_p]),
            np.hstack([A_anchor_u, A_anchor_v, A_anchor_p]),
        ])

        A = np.vstack([A_pde, A_bc])

        # --- Right-hand side ---
        # PDE residuals = 0
        b_cont = np.zeros(N_int)
        b_momx = np.zeros(N_int)
        b_momy = np.zeros(N_int)

        # Boundary values
        b_lid_u = np.full(N_lid, self.U_lid) * sqrt_w
        b_lid_v = np.zeros(N_lid) * sqrt_w
        b_wall_u = np.zeros(N_wall) * sqrt_w
        b_wall_v = np.zeros(N_wall) * sqrt_w
        b_anchor = np.zeros(N_anchor) * sqrt_w

        b = np.concatenate([
            b_cont, b_momx, b_momy,
            b_lid_u, b_lid_v, b_wall_u, b_wall_v, b_anchor
        ])

        return A, b

    def train(self) -> Dict[str, Any]:
        """
        Train the PIELM using Picard iteration.

        Returns:
            Dictionary with training results (time, iterations, residual)
        """
        start_time = time.perf_counter()

        # Initialize
        self._sample_domain()
        self._initialize_weights()

        N_int = self.N_interior
        n = self.n_hidden

        # Initial guess: zero velocity, zero pressure
        # We store the coefficients beta_u, beta_v, beta_p
        self.beta_u = np.zeros(n)
        self.beta_v = np.zeros(n)
        self.beta_p = np.zeros(n)

        # For the first iteration, use constant initial guess
        u_k = np.zeros(N_int)
        v_k = np.zeros(N_int)

        # Initial velocity gradients (zero for first iteration)
        du_dx_k = np.zeros(N_int)
        du_dy_k = np.zeros(N_int)
        dv_dx_k = np.zeros(N_int)
        dv_dy_k = np.zeros(N_int)

        # Initial eddy viscosity (just laminar)
        nu_eff_k = np.full(N_int, self.nu_laminar)

        # Initial ∇ν_eff (zero for laminar viscosity)
        dnu_dx_k = np.zeros(N_int)
        dnu_dy_k = np.zeros(N_int)

        # Second derivatives (needed for ∇ν_eff computation)
        d2u_dxx_k = np.zeros(N_int)
        d2u_dxy_k = np.zeros(N_int)
        d2u_dyy_k = np.zeros(N_int)
        d2v_dxx_k = np.zeros(N_int)
        d2v_dxy_k = np.zeros(N_int)
        d2v_dyy_k = np.zeros(N_int)

        self.residual_history = []

        for k in range(self.max_picard_iter):
            # Build linearized system (with or without full viscous term)
            if self.use_full_viscous:
                A, b = self._build_linearized_system(
                    u_k, v_k, du_dx_k, du_dy_k, dv_dx_k, dv_dy_k, nu_eff_k,
                    dnu_dx_k, dnu_dy_k
                )
            else:
                A, b = self._build_linearized_system(
                    u_k, v_k, du_dx_k, du_dy_k, dv_dx_k, dv_dy_k, nu_eff_k
                )

            # Solve least squares
            if self.solver == 'iterative':
                # Use LSQR for large problems (better scaling)
                result = lsqr(A, b, iter_lim=self.lsqr_iter_lim)
                beta = result[0]
            else:
                # Direct solver (lstsq) - more accurate but O(mn^2)
                beta, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            # Extract weights
            self.beta_u = beta[:n]
            self.beta_v = beta[n:2*n]
            self.beta_p = beta[2*n:]

            # Compute new solution at interior points
            H_int = self._compute_features(self.xy_interior)
            dH_dx_int, dH_dy_int = self._compute_gradient_features(self.xy_interior)

            u_new = H_int @ self.beta_u
            v_new = H_int @ self.beta_v

            # Compute velocity gradients for next iteration
            du_dx_new = dH_dx_int @ self.beta_u
            du_dy_new = dH_dy_int @ self.beta_u
            dv_dx_new = dH_dx_int @ self.beta_v
            dv_dy_new = dH_dy_int @ self.beta_v

            # Check convergence
            delta_u = np.linalg.norm(u_new - u_k)
            delta_v = np.linalg.norm(v_new - v_k)
            delta = np.sqrt(delta_u**2 + delta_v**2)

            # Relative change
            norm_new = np.sqrt(np.linalg.norm(u_new)**2 + np.linalg.norm(v_new)**2) + 1e-10
            rel_change = delta / norm_new

            self.residual_history.append(rel_change)

            if self.verbose:
                print(f"  Picard iter {k+1}: rel_change = {rel_change:.4e}")

            if rel_change < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {k+1}")
                break

            # Update for next iteration (with under-relaxation)
            omega = self.relaxation
            u_k = omega * u_new + (1 - omega) * u_k
            v_k = omega * v_new + (1 - omega) * v_k
            du_dx_k = omega * du_dx_new + (1 - omega) * du_dx_k
            du_dy_k = omega * du_dy_new + (1 - omega) * du_dy_k
            dv_dx_k = omega * dv_dx_new + (1 - omega) * dv_dx_k
            dv_dy_k = omega * dv_dy_new + (1 - omega) * dv_dy_k

            # Update eddy viscosity
            nu_eff_k = self._compute_eddy_viscosity(
                self.xy_interior, du_dx_k, du_dy_k, dv_dx_k, dv_dy_k
            )

            # Compute ∇ν_eff for full viscous term (requires second derivatives)
            if self.use_full_viscous:
                # Compute second derivatives of velocity
                d2H_dxx, d2H_dxy, d2H_dyy = self._compute_second_derivative_features(self.xy_interior)
                d2u_dxx_k = d2H_dxx @ self.beta_u
                d2u_dxy_k = d2H_dxy @ self.beta_u
                d2u_dyy_k = d2H_dyy @ self.beta_u
                d2v_dxx_k = d2H_dxx @ self.beta_v
                d2v_dxy_k = d2H_dxy @ self.beta_v
                d2v_dyy_k = d2H_dyy @ self.beta_v

                # Compute gradient of eddy viscosity
                dnu_dx_k, dnu_dy_k = self._compute_eddy_viscosity_gradient(
                    self.xy_interior,
                    du_dx_k, du_dy_k, dv_dx_k, dv_dy_k,
                    d2u_dxx_k, d2u_dxy_k, d2u_dyy_k,
                    d2v_dxx_k, d2v_dxy_k, d2v_dyy_k
                )

        self.train_time = time.perf_counter() - start_time
        self.n_iterations = k + 1
        self.is_trained = True

        results = {
            'train_time': self.train_time,
            'n_iterations': self.n_iterations,
            'final_residual': self.residual_history[-1] if self.residual_history else None,
            'converged': rel_change < self.tol,
        }

        if self.verbose:
            print(f"Training completed in {self.train_time:.3f}s ({self.n_iterations} iterations)")

        return results

    def predict(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict velocity and pressure at given coordinates.

        This is the DROP-IN REPLACEMENT interface for partner's PINN.

        Args:
            xy: Coordinates (N, 2)

        Returns:
            u, v, p: Velocity components and pressure, each (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        H = self._compute_features(xy)
        u = H @ self.beta_u
        v = H @ self.beta_v
        p = H @ self.beta_p

        return u, v, p

    def predict_with_gradients(
        self, xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict velocity, pressure, and velocity gradients.

        Args:
            xy: Coordinates (N, 2)

        Returns:
            u, v, p: Fields
            grads: Dictionary with 'du_dx', 'du_dy', 'dv_dx', 'dv_dy', 'dp_dx', 'dp_dy'
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        H = self._compute_features(xy)
        dH_dx, dH_dy = self._compute_gradient_features(xy)

        u = H @ self.beta_u
        v = H @ self.beta_v
        p = H @ self.beta_p

        grads = {
            'du_dx': dH_dx @ self.beta_u,
            'du_dy': dH_dy @ self.beta_u,
            'dv_dx': dH_dx @ self.beta_v,
            'dv_dy': dH_dy @ self.beta_v,
            'dp_dx': dH_dx @ self.beta_p,
            'dp_dy': dH_dy @ self.beta_p,
        }

        return u, v, p, grads

    def compute_pde_residuals(self, xy: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute PDE residuals for validation.

        Returns:
            Dictionary with 'continuity', 'momentum_x', 'momentum_y' residuals
        """
        u, v, p, grads = self.predict_with_gradients(xy)

        # Eddy viscosity
        nu_eff = self._compute_eddy_viscosity(
            xy, grads['du_dx'], grads['du_dy'], grads['dv_dx'], grads['dv_dy']
        )

        # Laplacian (approximate via second derivatives)
        LapH = self._compute_laplacian_features(xy)
        Lap_u = LapH @ self.beta_u
        Lap_v = LapH @ self.beta_v

        # Continuity: ∂u/∂x + ∂v/∂y = 0
        continuity = grads['du_dx'] + grads['dv_dy']

        # Momentum-x: u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν_eff·∇²u = 0
        momentum_x = (u * grads['du_dx'] + v * grads['du_dy'] +
                      grads['dp_dx'] - nu_eff * Lap_u)

        # Momentum-y: u·∂v/∂x + v·∂v/∂y + ∂p/∂y - ν_eff·∇²v = 0
        momentum_y = (u * grads['dv_dx'] + v * grads['dv_dy'] +
                      grads['dp_dy'] - nu_eff * Lap_v)

        return {
            'continuity': continuity,
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
        }


def test_pielm_navier_stokes(quick=True):
    """Quick test of PIELM_NavierStokes on lid-driven cavity."""
    print("=" * 60)
    print("Testing PIELM_NavierStokes on Lid-Driven Cavity")
    print("=" * 60)

    if quick:
        # Smaller problem for quick testing
        n_hidden = 200
        N_interior = 1000
        N_wall = 200
        N_lid = 200
        print("Using QUICK test settings (smaller problem size)")
    else:
        # Full problem matching partner's PINN
        n_hidden = 500
        N_interior = 6000
        N_wall = 800
        N_lid = 800
        print("Using FULL problem settings")

    # Create model with same parameters as partner's PINN
    model = PIELM_NavierStokes(
        Re=1000.0,
        U_lid=1.0,
        Cs=0.1,
        n_hidden=n_hidden,
        activation='tanh',
        max_picard_iter=50,
        tol=1e-6,
        N_interior=N_interior,
        N_wall=N_wall,
        N_lid=N_lid,
        bc_weight=10.0,
        verbose=True,
    )

    # Train
    print("\nTraining...")
    results = model.train()

    print(f"\n--- Results ---")
    print(f"Training time: {results['train_time']:.3f} seconds")
    print(f"Iterations: {results['n_iterations']}")
    print(f"Converged: {results['converged']}")

    # Test predictions
    print("\nTesting predictions...")
    test_xy = np.array([
        [0.5, 0.5],  # Center
        [0.5, 1.0],  # Lid (should have u=1)
        [0.0, 0.5],  # Left wall (should have u=0, v=0)
        [0.5, 0.0],  # Bottom wall (should have u=0, v=0)
    ])

    u, v, p = model.predict(test_xy)

    print("\nPredictions at test points:")
    print(f"  Center (0.5, 0.5): u={u[0]:.4f}, v={v[0]:.4f}, p={p[0]:.4f}")
    print(f"  Lid (0.5, 1.0):    u={u[1]:.4f}, v={v[1]:.4f}, p={p[1]:.4f}")
    print(f"  Left (0.0, 0.5):   u={u[2]:.4f}, v={v[2]:.4f}, p={p[2]:.4f}")
    print(f"  Bottom (0.5, 0.0): u={u[3]:.4f}, v={v[3]:.4f}, p={p[3]:.4f}")

    # Check boundary conditions
    print("\nBoundary condition checks:")
    print(f"  Lid u error: |u - 1| = {abs(u[1] - 1.0):.6f}")
    print(f"  Lid v error: |v - 0| = {abs(v[1]):.6f}")
    print(f"  Left wall u error: |u| = {abs(u[2]):.6f}")
    print(f"  Bottom wall v error: |v| = {abs(v[3]):.6f}")

    # Compute PDE residuals
    print("\nPDE residuals at interior test points...")
    test_interior = np.array([[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]])
    residuals = model.compute_pde_residuals(test_interior)

    print(f"  Continuity RMS: {np.sqrt(np.mean(residuals['continuity']**2)):.6f}")
    print(f"  Momentum-x RMS: {np.sqrt(np.mean(residuals['momentum_x']**2)):.6f}")
    print(f"  Momentum-y RMS: {np.sqrt(np.mean(residuals['momentum_y']**2)):.6f}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

    return model, results


if __name__ == "__main__":
    test_pielm_navier_stokes()
