"""
Nonlinear Poisson equation task.

PDE: ∇²u = f + exp(u)   in Ω
BC:  αu_n + βu = g       on ∂Ω

This is based on the DT-PINN benchmark dataset with precomputed RBF-FD operators.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from typing import Dict, Any, Optional

from .base import BaseTask, TaskData


class NonlinearPoissonTask(BaseTask):
    """
    Nonlinear Poisson equation on an L-shaped or irregular domain.

    Equation: ∇²u = f + exp(u)
    """

    name = "nonlinear-poisson"

    def __init__(
        self,
        data_path: str = None,
        order: int = 2,
        size: int = 2236,
        precision: str = "float64",
        **kwargs
    ):
        """
        Args:
            data_path: Path to data directory. Default: project_root/nonlinear
            order: RBF-FD stencil order (2-5)
            size: Number of training points (e.g., 582, 828, 1663, 2236, ...)
            precision: 'float32' or 'float64'
        """
        super().__init__(**kwargs)
        self.order = order
        self.size = size
        self.precision = np.float64 if precision == "float64" else np.float32

        # Find data path
        if data_path is None:
            # Try to find the nonlinear directory relative to this file
            # This file is at: src/experiment_dt_elm_pinn/tasks/nonlinear_poisson.py
            # Project root is 4 levels up: nonlinear_poisson.py -> tasks -> experiment_dt_elm_pinn -> src -> dt-pinn
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))
            data_path = os.path.join(project_root, "data", "nonlinear")

        self.data_path = data_path
        self.file_name = f"{order}_{size}"
        self.data_dir = os.path.join(data_path, f"files_{self.file_name}")

    def load_data(self) -> TaskData:
        """Load nonlinear Poisson data from .mat files."""
        d = self.data_dir

        # Load collocation points
        X_i = loadmat(f"{d}/Xi.mat")["Xi"].astype(self.precision)
        X_b = loadmat(f"{d}/Xb.mat")["Xb"].astype(self.precision)
        X_g = loadmat(f"{d}/Xg.mat")["X_g"].astype(self.precision)

        # Load source and BC
        f = loadmat(f"{d}/f.mat")["f"].astype(self.precision)
        g = loadmat(f"{d}/g.mat")["g"].astype(self.precision)

        # Load ground truth
        u_true = loadmat(f"{d}/u.mat")["u"].astype(self.precision)

        # Load discrete operators
        L = csr_matrix(loadmat(f"{d}/L1.mat")["L1"], dtype=self.precision)
        B = csr_matrix(loadmat(f"{d}/B1.mat")["B1"], dtype=self.precision)

        # Load Robin BC coefficients and normals
        alpha = loadmat(f"{d}/alpha.mat")["Neucoeff"].astype(self.precision)
        beta = loadmat(f"{d}/beta.mat")["Dircoeff"].astype(self.precision)
        n = loadmat(f"{d}/n.mat")["n"].astype(self.precision)

        # Get interior+boundary index
        ib_idx = X_i.shape[0] + X_b.shape[0]

        return TaskData(
            X_interior=X_i,
            X_boundary=X_b,
            X_ghost=X_g,
            f=f[:ib_idx].flatten(),  # f only defined at interior+boundary
            g=g.flatten(),
            u_true=u_true.flatten(),
            L=L,
            B=B,
            alpha=alpha.flatten(),
            beta=beta.flatten(),
            n=n,
        )

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """
        Compute PDE residual: ∇²u - f - exp(u) = 0

        Args:
            u: Solution at interior+boundary points (N_ib,)
            laplacian_u: Laplacian of u at interior+boundary points (N_ib,)

        Returns:
            Residual at interior+boundary points
        """
        return laplacian_u - self.data.f - np.exp(u)

    def compute_bc_residual(self, u: np.ndarray, u_full: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boundary condition residual.

        For pure Dirichlet: u - g = 0
        For Robin: α*∂u/∂n + β*u - g = 0

        Since we're using discrete operators, BC is enforced via B @ u_full = g.

        Args:
            u: Solution at boundary points (N_b,)
            u_full: Full solution including ghost points (for operator application)

        Returns:
            BC residual at boundary points
        """
        if u_full is not None and self.data.B is not None:
            # Use discrete operator
            return (self.data.B @ u_full).flatten() - self.data.g
        else:
            # Simple Dirichlet (extract boundary values)
            u_boundary = u[self.data.N_interior:]
            return u_boundary - self.data.g

    def compute_jacobian_exp(self, u: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of Jacobian for exp(u) term.

        d/du[exp(u)] = exp(u)

        Args:
            u: Solution at interior+boundary points

        Returns:
            Diagonal elements (exp(u))
        """
        return np.exp(u)

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        """Return default arguments for this task."""
        return {
            'order': 2,
            'size': 2236,
            'precision': 'float64',
        }

    @classmethod
    def list_available_sizes(cls) -> list:
        """List available problem sizes."""
        return [582, 828, 1663, 2236, 3196, 4977, 6114, 8767, 19638]
