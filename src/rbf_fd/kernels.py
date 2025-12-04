"""
RBF Kernel Functions for RBF-FD

Implements various radial basis functions and their Laplacians for 2D problems.
"""

import numpy as np
from typing import Union


def phs(r: np.ndarray, m: int = 3) -> np.ndarray:
    """
    Polyharmonic Spline (PHS) kernel.

    phi(r) = r^m for odd m
    phi(r) = r^m * log(r) for even m (with 0*log(0) = 0)

    Common choices: m=3 (r^3), m=5 (r^5), m=7 (r^7)

    Args:
        r: Distance array (non-negative)
        m: Order of PHS (positive integer)

    Returns:
        Kernel values phi(r)
    """
    if m % 2 == 1:
        # Odd m: phi(r) = r^m
        return np.power(r, m)
    else:
        # Even m: phi(r) = r^m * log(r)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.power(r, m) * np.log(r)
            result[r == 0] = 0.0
        return result


def phs_laplacian(r: np.ndarray, m: int = 3, dim: int = 2) -> np.ndarray:
    """
    Laplacian of PHS kernel in dim dimensions.

    For odd m in 2D:
        Lap(r^m) = m^2 * r^(m-2)

    For even m in 2D:
        Lap(r^m * log(r)) = m^2 * r^(m-2) * log(r) + (2m-2+dim) * r^(m-2)

    Args:
        r: Distance array
        m: Order of PHS
        dim: Spatial dimension (default 2)

    Returns:
        Laplacian values
    """
    if m < 2:
        raise ValueError(f"PHS order m must be >= 2 for Laplacian, got {m}")

    if m % 2 == 1:
        # Odd m: Lap(r^m) = m*(m-2+dim)*r^(m-2)
        # In 2D: Lap(r^m) = m*(m)*r^(m-2) = m^2 * r^(m-2)
        coeff = m * (m - 2 + dim)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = coeff * np.power(r, m - 2)
            if m == 2:
                result[r == 0] = coeff  # r^0 = 1
            else:
                result[r == 0] = 0.0
        return result
    else:
        # Even m in 2D
        with np.errstate(divide='ignore', invalid='ignore'):
            r_pow = np.power(r, m - 2)
            log_r = np.log(r)
            result = m * (m - 2 + dim) * r_pow * log_r + (2 * m - 2 + dim) * r_pow
            result[r == 0] = 0.0
        return result


def gaussian(r: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Gaussian kernel: phi(r) = exp(-eps^2 * r^2)

    Args:
        r: Distance array
        eps: Shape parameter

    Returns:
        Kernel values
    """
    return np.exp(-eps**2 * r**2)


def gaussian_laplacian(r: np.ndarray, eps: float = 1.0, dim: int = 2) -> np.ndarray:
    """
    Laplacian of Gaussian kernel in dim dimensions.

    Lap(exp(-eps^2*r^2)) = 2*eps^2*(2*eps^2*r^2 - dim) * exp(-eps^2*r^2)

    Args:
        r: Distance array
        eps: Shape parameter
        dim: Spatial dimension

    Returns:
        Laplacian values
    """
    return 2 * eps**2 * (2 * eps**2 * r**2 - dim) * np.exp(-eps**2 * r**2)


def multiquadric(r: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Multiquadric (MQ) kernel: phi(r) = sqrt(1 + eps^2 * r^2)

    Args:
        r: Distance array
        eps: Shape parameter

    Returns:
        Kernel values
    """
    return np.sqrt(1 + eps**2 * r**2)


def multiquadric_laplacian(r: np.ndarray, eps: float = 1.0, dim: int = 2) -> np.ndarray:
    """
    Laplacian of MQ kernel in dim dimensions.

    Lap(sqrt(1+eps^2*r^2)) = (dim-1)*eps^2 / (1+eps^2*r^2)^(1/2)
                            + eps^2 / (1+eps^2*r^2)^(3/2)

    In 2D: = eps^2 / (1+eps^2*r^2)^(1/2) + eps^2 / (1+eps^2*r^2)^(3/2)
         = eps^2 * (2 + eps^2*r^2) / (1+eps^2*r^2)^(3/2)

    Args:
        r: Distance array
        eps: Shape parameter
        dim: Spatial dimension

    Returns:
        Laplacian values
    """
    s = 1 + eps**2 * r**2
    return eps**2 * (dim - 1 + 1/s) / np.sqrt(s)


def inverse_multiquadric(r: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Inverse Multiquadric (IMQ) kernel: phi(r) = 1/sqrt(1 + eps^2 * r^2)

    Args:
        r: Distance array
        eps: Shape parameter

    Returns:
        Kernel values
    """
    return 1.0 / np.sqrt(1 + eps**2 * r**2)


def inverse_multiquadric_laplacian(r: np.ndarray, eps: float = 1.0, dim: int = 2) -> np.ndarray:
    """
    Laplacian of IMQ kernel in dim dimensions.

    Args:
        r: Distance array
        eps: Shape parameter
        dim: Spatial dimension

    Returns:
        Laplacian values
    """
    s = 1 + eps**2 * r**2
    return eps**2 * ((3 * eps**2 * r**2 - (dim - 1) * s) / s**2.5)


# Convenience dictionary for kernel selection
KERNELS = {
    'phs': (phs, phs_laplacian),
    'gaussian': (gaussian, gaussian_laplacian),
    'mq': (multiquadric, multiquadric_laplacian),
    'imq': (inverse_multiquadric, inverse_multiquadric_laplacian),
}


def get_kernel(name: str):
    """Get kernel function and its Laplacian by name."""
    if name not in KERNELS:
        raise ValueError(f"Unknown kernel: {name}. Available: {list(KERNELS.keys())}")
    return KERNELS[name]
