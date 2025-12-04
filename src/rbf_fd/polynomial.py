"""
Polynomial Basis Functions for RBF-FD

Implements polynomial basis and their Laplacians for augmenting RBF systems.
"""

import numpy as np
from typing import Tuple
from math import comb


def polynomial_basis_2d(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate 2D polynomial basis at given points.

    Degree 0: [1]                                    (1 term)
    Degree 1: [1, x, y]                              (3 terms)
    Degree 2: [1, x, y, x^2, xy, y^2]                (6 terms)
    Degree 3: [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]  (10 terms)

    Args:
        X: Point coordinates (n_points, 2)
        degree: Maximum polynomial degree

    Returns:
        P: Polynomial basis matrix (n_points, n_terms)
    """
    n = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]

    # Number of terms for degree d in 2D: (d+1)(d+2)/2
    n_terms = (degree + 1) * (degree + 2) // 2
    P = np.zeros((n, n_terms), dtype=X.dtype)

    col = 0
    for d in range(degree + 1):
        for i in range(d + 1):
            # x^(d-i) * y^i
            P[:, col] = np.power(x, d - i) * np.power(y, i)
            col += 1

    return P


def polynomial_laplacian_2d(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate Laplacian of 2D polynomial basis at given points.

    Lap(x^a * y^b) = a*(a-1)*x^(a-2)*y^b + b*(b-1)*x^a*y^(b-2)

    Args:
        X: Point coordinates (n_points, 2)
        degree: Maximum polynomial degree

    Returns:
        LP: Laplacian of polynomial basis (n_points, n_terms)
    """
    n = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]

    n_terms = (degree + 1) * (degree + 2) // 2
    LP = np.zeros((n, n_terms), dtype=X.dtype)

    col = 0
    for d in range(degree + 1):
        for i in range(d + 1):
            a = d - i  # x exponent
            b = i      # y exponent

            # Lap(x^a * y^b) = a*(a-1)*x^(a-2)*y^b + b*(b-1)*x^a*y^(b-2)
            term = np.zeros(n, dtype=X.dtype)

            if a >= 2:
                term += a * (a - 1) * np.power(x, a - 2) * np.power(y, b)
            if b >= 2:
                term += b * (b - 1) * np.power(x, a) * np.power(y, b - 2)

            LP[:, col] = term
            col += 1

    return LP


def polynomial_gradient_2d(X: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate gradient of 2D polynomial basis at given points.

    d/dx(x^a * y^b) = a*x^(a-1)*y^b
    d/dy(x^a * y^b) = b*x^a*y^(b-1)

    Args:
        X: Point coordinates (n_points, 2)
        degree: Maximum polynomial degree

    Returns:
        Px: x-derivative of polynomial basis (n_points, n_terms)
        Py: y-derivative of polynomial basis (n_points, n_terms)
    """
    n = X.shape[0]
    x = X[:, 0]
    y = X[:, 1]

    n_terms = (degree + 1) * (degree + 2) // 2
    Px = np.zeros((n, n_terms), dtype=X.dtype)
    Py = np.zeros((n, n_terms), dtype=X.dtype)

    col = 0
    for d in range(degree + 1):
        for i in range(d + 1):
            a = d - i
            b = i

            if a >= 1:
                Px[:, col] = a * np.power(x, a - 1) * np.power(y, b)
            if b >= 1:
                Py[:, col] = b * np.power(x, a) * np.power(y, b - 1)

            col += 1

    return Px, Py


def n_poly_terms_2d(degree: int) -> int:
    """Number of polynomial terms for given degree in 2D."""
    return (degree + 1) * (degree + 2) // 2


def recommended_poly_degree(stencil_size: int) -> int:
    """
    Recommend polynomial degree for given stencil size.

    Rule of thumb: n_neighbors >= 2 * n_poly_terms for stable solve.

    Args:
        stencil_size: Number of neighbors in stencil

    Returns:
        degree: Recommended polynomial augmentation degree
    """
    # Degree 2 needs 6 terms, degree 3 needs 10, degree 4 needs 15
    for d in range(10):
        if n_poly_terms_2d(d + 1) * 2 > stencil_size:
            return d
    return 9  # Maximum reasonable degree
