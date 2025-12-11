"""
Real NVP Normalizing Flow for Deep Adaptive Sampling (DAS).

PyTorch implementation of Real-valued Non-Volume Preserving (Real NVP) flow
for adaptive collocation point sampling based on PDE residual distributions.

Reference: Dinh et al., "Density estimation using Real-NVP" (2017)
           Tang et al., "DAS: A Deep Adaptive Sampling Method..." (2022)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class MLP(nn.Module):
    """Simple MLP for scale/shift networks in coupling layers."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Initialize last layer to near-zero for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActNorm(nn.Module):
    """
    Activation Normalization layer.

    Data-dependent initialization: First batch sets scale and bias
    such that outputs have zero mean and unit variance.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Data-dependent initialization."""
        with torch.no_grad():
            # Set bias to center data
            self.bias.data = -x.mean(dim=0, keepdim=True)
            # Set scale to normalize variance
            std = x.std(dim=0, keepdim=True) + 1e-6
            self.log_scale.data = -torch.log(std)
        self.initialized = True

    def forward(self, x: torch.Tensor, reverse: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: x -> (x + bias) * exp(log_scale)
        Reverse: x -> x * exp(-log_scale) - bias

        Returns: (output, log_det_jacobian)
        """
        if not self.initialized:
            self.initialize(x)

        # Clamp log_scale for numerical stability
        log_scale = torch.clamp(self.log_scale, -5.0, 5.0)

        if not reverse:
            y = (x + self.bias) * torch.exp(log_scale)
            log_det = log_scale.sum() * torch.ones(x.shape[0], device=x.device)
        else:
            y = x * torch.exp(-log_scale) - self.bias
            log_det = -log_scale.sum() * torch.ones(x.shape[0], device=x.device)

        return y, log_det


class AffineCoupling(nn.Module):
    """
    Affine Coupling Layer for Real NVP.

    Splits input into two parts [x1, x2]:
    - x1 passes through unchanged
    - x2 is transformed: y2 = x2 * (1 + alpha * tanh(s)) + t
      where s, t = NN(x1)

    The transformation is invertible and has tractable Jacobian.
    """

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor):
        """
        Args:
            dim: Input dimension
            hidden_dim: Hidden layer width for scale/shift networks
            mask: Binary mask indicating which dimensions to keep fixed
        """
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)

        # Number of dimensions that are kept fixed vs transformed
        n_fixed = int(mask.sum().item())
        n_transform = dim - n_fixed

        # Network outputs both scale and shift
        self.net = MLP(n_fixed, hidden_dim, n_transform * 2)

        # Learnable scaling factor (from DAS code)
        self.log_gamma = nn.Parameter(torch.zeros(1, n_transform))

        # Scaling factor for stability (from DAS code: alpha = 0.6)
        self.alpha = 0.6

    def forward(self, x: torch.Tensor, reverse: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: Apply affine transformation to masked dimensions.
        Reverse: Invert the affine transformation.

        Returns: (output, log_det_jacobian)
        """
        # Split based on mask
        mask = self.mask.bool()
        x_fixed = x[:, mask]
        x_transform = x[:, ~mask]

        # Compute scale and shift from fixed part
        h = self.net(x_fixed)
        n_transform = x_transform.shape[1]
        shift = h[:, :n_transform]
        scale_raw = h[:, n_transform:]

        # Apply stabilized scaling (from DAS code)
        log_gamma_clamped = torch.clamp(self.log_gamma, -5.0, 5.0)
        shift = torch.exp(log_gamma_clamped) * torch.tanh(shift)
        scale = self.alpha * torch.tanh(scale_raw)

        if not reverse:
            # Forward: y = x * (1 + scale) + shift
            y_transform = x_transform * (1 + scale) + shift
            log_det = torch.sum(torch.log(torch.abs(1 + scale) + 1e-8), dim=1)
        else:
            # Reverse: x = (y - shift) / (1 + scale)
            y_transform = (x_transform - shift) / (1 + scale)
            log_det = -torch.sum(torch.log(torch.abs(1 + scale) + 1e-8), dim=1)

        # Reconstruct output
        y = torch.zeros_like(x)
        y[:, mask] = x_fixed
        y[:, ~mask] = y_transform

        return y, log_det


class RealNVP(nn.Module):
    """
    Real NVP Normalizing Flow.

    Stacks ActNorm and AffineCoupling layers with alternating masks
    to create an invertible transformation with tractable density.

    Used in DAS to learn the distribution of PDE residuals and
    generate new collocation points in high-residual regions.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        hidden_dim: int = 64,
        device: torch.device = None
    ):
        """
        Args:
            dim: Input/output dimension
            n_layers: Number of coupling layers (should be even)
            hidden_dim: Hidden dimension for coupling networks
            device: Device to place the model on
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        if device is None:
            device = torch.device('cpu')
        self.device = device

        # Must have even number of layers for full mixing
        assert n_layers % 2 == 0, "n_layers must be even"

        # Build alternating masks
        # Mask pattern: [1,0,1,0,...] and [0,1,0,1,...]
        mask1 = torch.zeros(dim, device=device)
        mask1[::2] = 1  # Even indices
        mask2 = 1 - mask1  # Odd indices

        # Build layers
        self.actnorms = nn.ModuleList()
        self.couplings = nn.ModuleList()

        for i in range(n_layers):
            self.actnorms.append(ActNorm(dim))
            mask = mask1 if i % 2 == 0 else mask2
            self.couplings.append(AffineCoupling(dim, hidden_dim, mask))

        # Prior distribution: standard Gaussian
        self.prior_mean = torch.zeros(dim, device=device)
        self.prior_std = torch.ones(dim, device=device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from data space to prior space.

        Args:
            x: Input samples [batch_size, dim]

        Returns:
            z: Transformed samples in prior space
            log_det: Log determinant of Jacobian
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for i in range(self.n_layers):
            # ActNorm
            z, log_det = self.actnorms[i](z, reverse=False)
            log_det_total = log_det_total + log_det

            # Affine coupling
            z, log_det = self.couplings[i](z, reverse=False)
            log_det_total = log_det_total + log_det

            # Reverse order (from DAS code: z = z[:,::-1])
            z = z.flip(dims=[1])

        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from prior space to data space.

        Args:
            z: Samples in prior space [batch_size, dim]

        Returns:
            x: Transformed samples in data space
            log_det: Log determinant of inverse Jacobian
        """
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)

        for i in reversed(range(self.n_layers)):
            # Reverse order first
            x = x.flip(dims=[1])

            # Inverse affine coupling
            x, log_det = self.couplings[i](x, reverse=True)
            log_det_total = log_det_total + log_det

            # Inverse ActNorm
            x, log_det = self.actnorms[i](x, reverse=True)
            log_det_total = log_det_total + log_det

        return x, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples under the flow.

        log p(x) = log p(z) + log |det(dz/dx)|

        Args:
            x: Input samples [batch_size, dim]

        Returns:
            log_prob: Log probability [batch_size]
        """
        z, log_det = self.forward(x)

        # Log probability under standard Gaussian prior
        log_prior = -0.5 * (
            self.dim * np.log(2 * np.pi) +
            torch.sum(z ** 2, dim=1)
        )

        return log_prior + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the learned distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            x: Samples in data space [n_samples, dim]
        """
        # Sample from prior
        z = torch.randn(n_samples, self.dim, device=self.device)

        # Transform to data space
        x, _ = self.inverse(z)

        return x

    def reset_actnorm(self):
        """Reset ActNorm data-dependent initialization."""
        for actnorm in self.actnorms:
            actnorm.initialized = False


class BoundedRealNVP(nn.Module):
    """
    Real NVP with bounded domain support.

    Wraps RealNVP with logistic transformations to handle
    bounded domains like [0, 1]^d or [a, b]^d.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        hidden_dim: int = 64,
        lb: float = 0.0,
        ub: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """
        Args:
            dim: Input/output dimension
            n_layers: Number of coupling layers
            hidden_dim: Hidden dimension for coupling networks
            lb: Lower bound of domain
            ub: Upper bound of domain
            device: Device to place the model on
            dtype: Data type (torch.float32 or torch.float64)
        """
        super().__init__()
        self.dim = dim
        self.lb = lb
        self.ub = ub

        if device is None:
            device = torch.device('cpu')
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Core flow on unbounded space
        self.flow = RealNVP(dim, n_layers, hidden_dim, device)

    def _to_unbounded(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map from bounded [lb, ub] to unbounded (-inf, inf) via logistic.

        y = logit((x - lb) / (ub - lb))
          = log(x') - log(1 - x')  where x' in (0, 1)
        """
        # Normalize to (0, 1)
        x_norm = (x - self.lb) / (self.ub - self.lb)
        x_norm = torch.clamp(x_norm, 1e-6, 1 - 1e-6)

        # Logit transform
        y = torch.log(x_norm) - torch.log(1 - x_norm)

        # Log determinant: sum of log |dy/dx| = sum of -log(x') - log(1-x')
        log_det = -torch.log(x_norm) - torch.log(1 - x_norm)
        log_det = log_det.sum(dim=1)

        return y, log_det

    def _to_bounded(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map from unbounded (-inf, inf) to bounded [lb, ub] via sigmoid.

        x = lb + (ub - lb) * sigmoid(y)
        """
        # Sigmoid transform
        x_norm = torch.sigmoid(y)

        # Scale to [lb, ub]
        x = self.lb + (self.ub - self.lb) * x_norm

        # Log determinant: sum of log |dx/dy| = sum of log(sigmoid'(y))
        # sigmoid'(y) = sigmoid(y) * (1 - sigmoid(y))
        log_det = torch.log(x_norm + 1e-8) + torch.log(1 - x_norm + 1e-8)
        log_det = log_det.sum(dim=1)

        return x, log_det

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from bounded data space to prior space."""
        # Bounded -> unbounded
        y, log_det1 = self._to_unbounded(x)

        # Flow transform
        z, log_det2 = self.flow.forward(y)

        return z, log_det1 + log_det2

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from prior space to bounded data space."""
        # Inverse flow
        y, log_det1 = self.flow.inverse(z)

        # Unbounded -> bounded
        x, log_det2 = self._to_bounded(y)

        return x, log_det1 + log_det2

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability under the bounded flow."""
        z, log_det = self.forward(x)

        # Log prior (standard Gaussian)
        log_prior = -0.5 * (
            self.dim * np.log(2 * np.pi) +
            torch.sum(z ** 2, dim=1)
        )

        return log_prior + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the bounded distribution."""
        # Sample from prior with correct dtype
        z = torch.randn(n_samples, self.dim, device=self.device, dtype=self.dtype)

        # Transform to bounded data space
        x, _ = self.inverse(z)

        return x

    def reset_actnorm(self):
        """Reset ActNorm initialization."""
        self.flow.reset_actnorm()
