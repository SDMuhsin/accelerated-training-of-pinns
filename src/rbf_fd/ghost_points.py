"""
Ghost Point Generation for RBF-FD

Generates ghost points outside the domain for boundary stencil support.
Ghost points are placed along the outward normal at a fixed distance.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional


def estimate_normals_from_curve(X_boundary: np.ndarray) -> np.ndarray:
    """
    Estimate outward normals for a 2D boundary curve.

    Assumes boundary points are ordered (counter-clockwise for outward normal).
    Uses central differences to estimate tangent, then rotates 90 degrees.

    Args:
        X_boundary: Boundary points (N_b, 2), ordered along the curve

    Returns:
        normals: Outward unit normal vectors (N_b, 2)
    """
    n = len(X_boundary)

    # Compute tangent vectors using central differences (with periodic wrapping)
    tangents = np.zeros_like(X_boundary)
    for i in range(n):
        # Central difference with wrapping
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        tangents[i] = X_boundary[next_idx] - X_boundary[prev_idx]

    # Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / tangent_norms

    # Rotate 90 degrees clockwise to get outward normal (for CCW ordered boundary)
    # [tx, ty] -> [ty, -tx]
    normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])

    return normals


def estimate_normals_from_interior(
    X_boundary: np.ndarray,
    X_interior: np.ndarray,
    k_neighbors: int = 5,
) -> np.ndarray:
    """
    Estimate outward normals by pointing away from interior centroid.

    More robust for unordered boundary points.

    Args:
        X_boundary: Boundary points (N_b, 2)
        X_interior: Interior points (N_i, 2)
        k_neighbors: Number of interior neighbors to use for direction

    Returns:
        normals: Outward unit normal vectors (N_b, 2)
    """
    # Use centroid of interior points as reference
    interior_centroid = X_interior.mean(axis=0)

    # Direction from centroid to each boundary point
    directions = X_boundary - interior_centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normals = directions / norms

    return normals


def estimate_normals_radial(
    X_boundary: np.ndarray,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate outward normals assuming a convex domain with known center.

    For star-convex domains (like circles, ellipses), this gives exact normals.

    Args:
        X_boundary: Boundary points (N_b, 2)
        center: Center point. If None, uses mean of boundary points.

    Returns:
        normals: Outward unit normal vectors (N_b, 2)
    """
    if center is None:
        center = X_boundary.mean(axis=0)

    # Direction from center to boundary
    directions = X_boundary - center
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normals = directions / norms

    return normals


def generate_ghost_points(
    X_boundary: np.ndarray,
    normals: np.ndarray,
    offset: float,
) -> np.ndarray:
    """
    Generate ghost points at fixed offset along normals.

    Args:
        X_boundary: Boundary points (N_b, 2)
        normals: Outward normal vectors (N_b, 2)
        offset: Distance from boundary to ghost points

    Returns:
        X_ghost: Ghost points (N_b, 2)
    """
    return X_boundary + offset * normals


def estimate_ghost_offset(X_interior: np.ndarray, X_boundary: np.ndarray) -> float:
    """
    Estimate appropriate ghost point offset based on point spacing.

    Uses the average minimum distance between interior points as reference.

    Args:
        X_interior: Interior points (N_i, 2)
        X_boundary: Boundary points (N_b, 2)

    Returns:
        offset: Recommended ghost point offset
    """
    # Build KD-tree for interior points
    tree = cKDTree(X_interior)

    # Find average spacing (distance to nearest neighbor)
    distances, _ = tree.query(X_interior, k=2)  # k=2 because first is self
    avg_spacing = distances[:, 1].mean()

    # Ghost offset is typically a fraction of the spacing
    # MATLAB uses ~0.16 * spacing based on our analysis
    return 0.16 * avg_spacing


class GhostPointGenerator:
    """
    Ghost point generator for RBF-FD.

    Generates ghost points outside the domain for stable boundary stencils.
    """

    def __init__(
        self,
        offset: Optional[float] = None,
        normal_method: str = 'interior',
    ):
        """
        Initialize ghost point generator.

        Args:
            offset: Fixed ghost point offset. If None, auto-estimate from point spacing.
            normal_method: Method for estimating normals ('curve' or 'interior')
        """
        self.offset = offset
        self.normal_method = normal_method

    def generate(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ghost points for the given domain.

        Args:
            X_interior: Interior points (N_i, 2)
            X_boundary: Boundary points (N_b, 2)

        Returns:
            X_ghost: Ghost points (N_b, 2)
            normals: Outward normal vectors used (N_b, 2)
        """
        # Estimate offset if not provided
        offset = self.offset
        if offset is None:
            offset = estimate_ghost_offset(X_interior, X_boundary)

        # Estimate normals
        if self.normal_method == 'curve':
            normals = estimate_normals_from_curve(X_boundary)
        else:  # 'interior'
            normals = estimate_normals_from_interior(X_boundary, X_interior)

        # Generate ghost points
        X_ghost = generate_ghost_points(X_boundary, normals, offset)

        return X_ghost, normals

    def validate_against_matlab(
        self,
        X_ghost_python: np.ndarray,
        X_ghost_matlab: np.ndarray,
    ) -> dict:
        """
        Validate Python-generated ghost points against MATLAB reference.

        Args:
            X_ghost_python: Python-generated ghost points
            X_ghost_matlab: MATLAB-generated ghost points

        Returns:
            dict with validation metrics
        """
        diff = X_ghost_python - X_ghost_matlab
        max_diff = np.abs(diff).max()
        mean_diff = np.abs(diff).mean()
        rel_error = np.linalg.norm(diff) / np.linalg.norm(X_ghost_matlab)

        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'relative_error': rel_error,
            'pass': rel_error < 1e-6,
        }
