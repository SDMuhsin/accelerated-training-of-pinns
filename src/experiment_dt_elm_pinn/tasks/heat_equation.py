"""
Heat (Laplace) equation task.

PDE: ∇²u = f        in Ω
BC:  αu_n + βu = g  on ∂Ω (Robin BC)

This is a LINEAR equation, unlike the nonlinear Poisson which has exp(u).
Based on the DT-PINN benchmark dataset with precomputed RBF-FD operators.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from typing import Dict, Any, Optional

from .base import BaseTask, TaskData


class HeatEquationTask(BaseTask):
    """
    Heat (Laplace) equation on an irregular domain.

    Equation: ∇²u = f  (linear!)
    BC: αu_n + βu = g (Robin)

    Unlike nonlinear Poisson, this has NO exp(u) term.
    """

    name = "heat-equation"

    def __init__(
        self,
        data_path: str = None,
        order: int = 2,
        size: int = 828,
        precision: str = "float64",
        **kwargs
    ):
        """
        Args:
            data_path: Path to data directory. Default: project_root/data/heat
            order: RBF-FD stencil order (2-5)
            size: Number of training points (e.g., 828)
            precision: 'float32' or 'float64'
        """
        super().__init__(**kwargs)
        self.order = order
        self.size = size
        self.precision = np.float64 if precision == "float64" else np.float32

        # Find data path
        if data_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))
            data_path = os.path.join(project_root, "data", "heat")

        self.data_path = data_path
        self.file_name = f"{order}_{size}"
        self.data_dir = os.path.join(data_path, f"files_{self.file_name}")

    def load_data(self) -> TaskData:
        """Load heat equation data from .mat files."""
        d = self.data_dir

        # Load collocation points
        X_i = loadmat(f"{d}/Xi.mat")["Xi"].astype(self.precision)
        X_b = loadmat(f"{d}/Xb.mat")["Xb"].astype(self.precision)
        X_g = loadmat(f"{d}/Xg.mat")["X_g"].astype(self.precision)

        # Load source and BC - NOTE: uses _heat suffix
        f = loadmat(f"{d}/f_heat.mat")["f_heat"].astype(self.precision)
        g = loadmat(f"{d}/g_heat.mat")["g_heat"].astype(self.precision)

        # Load ground truth
        u_true = loadmat(f"{d}/u_heat.mat")["u_heat"].astype(self.precision)

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
            f=f[:ib_idx].flatten(),
            g=g.flatten(),
            u_true=u_true.flatten()[:ib_idx],  # Only interior+boundary, not ghost
            L=L,
            B=B,
            alpha=alpha.flatten(),
            beta=beta.flatten(),
            n=n,
        )

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """
        Compute PDE residual: ∇²u - f = 0 (LINEAR!)

        Args:
            u: Solution at interior+boundary points (N_ib,)
            laplacian_u: Laplacian of u at interior+boundary points (N_ib,)

        Returns:
            Residual at interior+boundary points
        """
        return laplacian_u - self.data.f

    def compute_bc_residual(self, u: np.ndarray, u_full: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute boundary condition residual.

        Robin BC: α*∂u/∂n + β*u - g = 0

        Args:
            u: Solution at boundary points (N_b,)
            u_full: Full solution including ghost points (for operator application)

        Returns:
            BC residual at boundary points
        """
        if u_full is not None and self.data.B is not None:
            return (self.data.B @ u_full).flatten() - self.data.g
        else:
            u_boundary = u[self.data.N_interior:]
            return u_boundary - self.data.g

    def is_linear(self) -> bool:
        """Heat equation is LINEAR (no nonlinear term)."""
        return True

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        """Return default arguments for this task."""
        return {
            'order': 2,
            'size': 828,
            'precision': 'float64',
        }

    @classmethod
    def list_available_sizes(cls) -> list:
        """List available problem sizes."""
        return [828]  # Currently only one size available
