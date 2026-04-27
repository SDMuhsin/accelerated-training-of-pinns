"""Faithful RBF-FD operator construction for DT-PINN.

Port of Sharma & Shankar 2022 (arXiv:2205.09332) MatlabSolver:
  - PHS kernel φ(r) = (r+ε)^m with m clamped to [5,11].
  - Tensor-product orthonormal Legendre polynomial basis (Jacobi α=β=0).
  - Saddle-point stencil weights via direct LU on (n+polyM) × (n+polyM) systems.
  - Ghost-node augmentation: Xg = Xb + 0.25·h·normal.
  - Returns sparse operators of shape (Ni+Nb, Ni+Nb+Ng) over Xf=[Xi;Xb;Xg].

Faithfulness to paper conventions:
  - Gradient operators (Dx, Dy) use scaled polynomial coords pc=(p−c)/w (FormGradients.m).
  - Second derivatives (Dxx, Dyy, Dxy, Lap) use unscaled physical coords (FormLaplacian.m).
  - Stencil size n = 2·polyM + 1, polyM = C(ell+d,d), ell = p + θ − 1.

This module is consumed by src/lid_benchmark.py train_dtpinn_* functions (Phase 4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
import scipy.sparse
from scipy.spatial import cKDTree

# Machine epsilon used in (r+ε) — matches Matlab `eps`.
_EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# 1. Polynomial helpers (port of MatlabSolver/{jacobi_recurrence,poly_eval,
#    mpoly_eval,total_degree_indices}.m)
# ---------------------------------------------------------------------------


def total_degree_indices(d: int, k: int) -> np.ndarray:
    """Return all multi-indices α∈ℕ^d with |α|≤k, ordered by total degree.

    Output: (M, d) int array, M = C(k+d, d). Ordering matches
    MatlabSolver/total_degree_indices.m (degree-graded, with within-degree
    ordering produced by the "traveling ones-man" recurrence).
    """
    if k < 0:
        return np.zeros((0, d), dtype=np.int64)

    M = math.comb(k + d, d)
    a = np.zeros((M, d), dtype=np.int64)
    if k == 0:
        return a

    row = 1
    a[0] = 0
    for q in range(1, k + 1):
        current = np.zeros(d, dtype=np.int64)
        current[0] = q
        a[row] = current.copy()
        row += 1

        # "traveling ones-man" enumeration of all (d,q) compositions
        onesman_home = 0  # 0-based index of the column hosting the moving 1
        onesman_loc = 0
        finished = False
        while not finished:
            # walk the ones-man from onesman_loc → end
            while onesman_loc < d - 1:
                onesman_loc += 1
                current[onesman_loc - 1] -= 1
                current[onesman_loc] += 1
                a[row] = current.copy()
                row += 1
            # if home was at column d-2, drain remaining at column d-2
            if onesman_home + 1 == d - 1:
                while current[onesman_home] > 0:
                    current[-1] += 1
                    current[-2] -= 1
                    a[row] = current.copy()
                    row += 1
            if current[-1] == q:
                finished = True
                break
            # find rightmost non-zero column (in 1-based: 'last 2 nonzeros')
            nonzero_cols = np.flatnonzero(current)
            if len(nonzero_cols) >= 2:
                col = nonzero_cols[-2]
            else:
                col = nonzero_cols[-1]  # shouldn't happen if not finished
            current[col] -= 1
            current[col + 1] = current[-1] + 1
            current[-1] = 0
            a[row] = current.copy()
            row += 1
            onesman_home = col + 1
            onesman_loc = col + 1
    if row != M:
        raise RuntimeError(
            f"total_degree_indices(d={d}, k={k}) generated {row} rows, expected {M}."
        )
    return a


def jacobi_recurrence(N: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Three-term recurrence coefs for Jacobi polynomials with parameters α, β.

    Returns (a, b) of length N. For α=β=0 → Legendre on [-1,1].
    Matches MatlabSolver/jacobi_recurrence.m.
    """
    n = np.arange(N, dtype=np.float64)  # 0..N-1
    a = (beta**2 - alpha**2) * np.ones_like(n)
    b = np.ones_like(n)

    # n == 0
    mask0 = n == 0
    if np.any(mask0):
        a[mask0] = (beta - alpha) / (alpha + beta + 2.0)
        b[mask0] = math.exp(
            (alpha + beta + 1.0) * math.log(2.0)
            + math.lgamma(alpha + 1.0)
            + math.lgamma(beta + 1.0)
            - math.lgamma(alpha + beta + 2.0)
        )

    # n == 1
    mask1 = n == 1
    if np.any(mask1):
        a[mask1] = a[mask1] / ((2.0 + alpha + beta) * (4.0 + alpha + beta))
        b[mask1] = (
            4.0
            * (1.0 + alpha)
            * (1.0 + beta)
            / ((2.0 + alpha + beta) ** 2 * (3.0 + alpha + beta))
        )

    mask = ~(mask0 | mask1)
    n_m = n[mask]
    a[mask] = a[mask] / ((2 * n_m + alpha + beta) * (2 * n_m + alpha + beta + 2))
    b[mask] = (
        4
        * n_m
        * (n_m + alpha)
        * (n_m + beta)
        * (n_m + alpha + beta)
        / (
            (2 * n_m + alpha + beta) ** 2
            * (2 * n_m + alpha + beta + 1)
            * (2 * n_m + alpha + beta - 1)
        )
    )
    return a, b


def poly_eval(a: np.ndarray, b: np.ndarray, x: np.ndarray, N: int, d: int = 0) -> np.ndarray:
    """Evaluate the d-th derivative of the orthonormal Jacobi polynomials p_n
    (n = 0..N) at points x using the recurrence

        sqrt(b_{n+1}) p_{n+1} = (x − a_n) p_n − sqrt(b_n) p_{n−1}.

    Returns array of shape (nx, N+1). Matches MatlabSolver/poly_eval.m.
    """
    if d < 0:
        raise ValueError("d must be ≥ 0")
    if N < 0:
        raise ValueError("N must be ≥ 0")
    if N + 1 > len(a) or N + 1 > len(b):
        raise ValueError(f"need at least {N + 1} recurrence coefs; got len(a)={len(a)}, len(b)={len(b)}")
    xf = np.asarray(x, dtype=np.float64).reshape(-1)
    nx = xf.size

    p = np.zeros((nx, N + 1), dtype=np.float64)
    p[:, 0] = 1.0 / math.sqrt(b[0])
    if N > 0:
        p[:, 1] = 1.0 / math.sqrt(b[1]) * (xf - a[0]) * p[:, 0]
    for q in range(2, N + 1):
        p[:, q] = (xf - a[q - 1]) * p[:, q - 1] - math.sqrt(b[q - 1]) * p[:, q - 2]
        p[:, q] /= math.sqrt(b[q])

    if d == 0:
        return p

    # successively differentiate using the same recurrence with extra +qd*p[:,q] term
    for qd in range(1, d + 1):
        pd = np.zeros_like(p)
        for q in range(qd, N + 1):
            if q == qd:
                pd[:, q] = math.exp(
                    math.lgamma(qd + 1) - 0.5 * float(np.sum(np.log(b[: q + 1])))
                )
            else:
                pd[:, q] = (
                    (xf - a[q - 1]) * pd[:, q - 1]
                    - math.sqrt(b[q - 1]) * pd[:, q - 2]
                    + qd * p[:, q - 1]
                )
                pd[:, q] /= math.sqrt(b[q])
        p = pd
    return p


def mpoly_eval(
    x: np.ndarray,
    alpha: np.ndarray,
    recurrence_fn: Callable[[int], Tuple[np.ndarray, np.ndarray]],
    deriv: Sequence[int] | None = None,
) -> np.ndarray:
    """Evaluate orthonormal tensor-product polynomials at points x.

    x:       (M, dim)
    alpha:   (N, dim) multi-index rows; each row is a tensor-product index
    deriv:   (dim,) or None — partial derivative multi-index (default: 0).

    Returns (M, N) matrix of polynomial values (or derivatives).

    Uses recurrence b(0) (the Legendre normalisation 2 for α=β=0) implicitly.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    M, dim = x.shape
    N, dim_a = alpha.shape
    if dim_a != dim:
        raise ValueError("x and alpha must have the same number of columns")
    if deriv is None:
        deriv = np.zeros(dim, dtype=np.int64)
    deriv = np.asarray(deriv, dtype=np.int64)
    if deriv.shape != (dim,):
        raise ValueError(f"deriv must be ({dim},); got {deriv.shape}")

    a_rec, b_rec = recurrence_fn(int(np.max(alpha)) + 1)
    p = np.ones((M, N), dtype=np.float64) / math.sqrt(b_rec[0] ** dim)
    a_pos = alpha > 0  # for the "skip zero alpha" optimisation in deriv=0 case

    for qdim in range(dim):
        max_alpha = int(np.max(alpha[:, qdim]))
        temp = poly_eval(a_rec, b_rec, x[:, qdim], max_alpha, int(deriv[qdim]))
        # temp has shape (M, max_alpha+1) — pick columns for each row of alpha
        if deriv[qdim] > 0:
            # multiply ALL columns
            p = p * temp[:, alpha[:, qdim]] * math.sqrt(b_rec[0])
        else:
            # multiply only columns whose alpha[:, qdim] > 0; for alpha==0,
            # poly_eval returns 1/sqrt(b[0]) which combines with the leading
            # 1/sqrt(b[0]^dim) to give 1/sqrt(b[0]^(dim+1)). The Matlab code
            # only multiplies for positive alpha; multiplied-by-1 columns
            # implicitly stay at 1/sqrt(b[0]^dim).
            mask = a_pos[:, qdim]
            p[:, mask] = p[:, mask] * temp[:, alpha[mask, qdim]] * math.sqrt(b_rec[0])
    return p


# ---------------------------------------------------------------------------
# 2. PHS kernel and its derivatives in 2D
# ---------------------------------------------------------------------------

# We use 2D throughout. For 3D (sphere) extension, see the paper §3.
def _phs(r: np.ndarray, m: int) -> np.ndarray:
    return (r + _EPS) ** m


def _phs_drbf_over_r(r: np.ndarray, m: int) -> np.ndarray:
    """Returns (dφ/dr) / r = m · (r+ε)^(m−2). Used so that
    (xj − xk) · _phs_drbf_over_r gives ∂φ/∂x directly.
    """
    return m * (r + _EPS) ** (m - 2)


def _phs_d2rbf(r: np.ndarray, m: int) -> np.ndarray:
    """d²φ/dr² = m·(m−1)·(r+ε)^(m−2)."""
    return m * (m - 1) * (r + _EPS) ** (m - 2)


def _phs_laplacian(r: np.ndarray, m: int, d: int) -> np.ndarray:
    """Δφ(r) = d²φ/dr² + (d−1)/r · dφ/dr = m·(m+d−2)·(r+ε)^(m−2). Matches
    MatlabSolver/lrbf.m for k=1.
    """
    return m * (m + d - 2) * (r + _EPS) ** (m - 2)


# ---------------------------------------------------------------------------
# 3. RBF-FD parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RBFFDParams:
    """Mirrors MatlabSolver/rbffdop.m for hyperviscosity_flag=0."""

    s_dim: int  # spatial dimension (2 or 3)
    p: int  # RBF-FD order ξ ∈ {2,3,4,5}
    theta: int  # operator order (1=∇, 2=Δ)
    ell: int  # polynomial degree = p + θ − 1
    m: int  # PHS exponent
    poly_count: int  # = C(ell+d, d)
    stencil_size: int  # = 2·poly_count + 1

    @classmethod
    def from_orders(cls, s_dim: int, p: int, theta: int) -> "RBFFDParams":
        if s_dim not in (2, 3):
            raise ValueError("s_dim must be 2 or 3")
        if p < 2:
            raise ValueError("p (xi) must be ≥ 2")
        if theta not in (1, 2):
            raise ValueError("theta must be 1 (∇) or 2 (Δ)")
        ell = p + theta - 1
        m = ell - 1 if (ell % 2 == 0) else ell
        m = max(min(m, 11), 5)
        poly_count = math.comb(ell + s_dim, s_dim)
        stencil_size = 2 * poly_count + 1
        return cls(s_dim, p, theta, ell, m, poly_count, stencil_size)


# ---------------------------------------------------------------------------
# 4. Local stencil-weight construction (per-stencil saddle-point solve)
# ---------------------------------------------------------------------------


def _legendre_recurrence(N: int) -> Tuple[np.ndarray, np.ndarray]:
    return jacobi_recurrence(N, 0.0, 0.0)


def _build_local_weights_one(
    stencil_xy: np.ndarray,
    deriv: Tuple[int, int] | str,
    params: RBFFDParams,
    alpha_indices: np.ndarray,
    *,
    scale_polynomial: bool,
) -> np.ndarray:
    """Compute RBF-FD weights at the stencil centre (= stencil_xy[0]).

    stencil_xy:       (n, 2) coordinates with stencil_xy[0] the centre.
    deriv:            (i, j) multi-index (i+j ≤ ell) or 'lap' for the Laplacian.
    params:           RBFFDParams.
    alpha_indices:    (poly_count, 2) precomputed total-degree indices.
    scale_polynomial: if True, evaluate the polynomial basis at scaled coords
                      pc=(p−c)/w (FormGradients.m); else at physical p
                      (FormLaplacian.m).

    Returns: w of shape (n,) with (Lu)(c) ≈ Σ_j w_j u(stencil_xy[j]).
    """
    n = params.stencil_size
    if stencil_xy.shape != (n, 2):
        raise ValueError(f"expected stencil ({n},2), got {stencil_xy.shape}")
    p = stencil_xy
    centre = p[0]
    # pairwise distances within stencil
    diffs = p[:, None, :] - p[None, :, :]  # (n, n, 2)
    rd = np.sqrt(np.maximum(np.sum(diffs * diffs, axis=2), 0.0))  # (n, n)

    A_rbf = _phs(rd, params.m)  # (n, n)

    # polynomial-basis coordinates
    if scale_polynomial:
        w_scale = rd[0, n - 1]  # ‖x_n − x_0‖
        if w_scale <= 0:
            raise RuntimeError("degenerate stencil: zero scaling distance")
        pc = (p - centre) / w_scale
    else:
        w_scale = 1.0
        pc = p

    v = mpoly_eval(pc, alpha_indices, _legendre_recurrence)  # (n, poly_count)

    # augmented saddle-point matrix
    A = np.block(
        [
            [A_rbf, v],
            [v.T, np.zeros((params.poly_count, params.poly_count))],
        ]
    )

    # build RHS = derivative-of-(kernel,polynomial) at the centre
    rhs = np.empty(n + params.poly_count, dtype=np.float64)

    if deriv == "lap":
        # Δφ(|c − x_j|) for each x_j — symmetric in distance
        rhs[:n] = _phs_laplacian(rd[0, :], params.m, params.s_dim)
        # Δp_α(c)
        if params.s_dim == 2:
            lp_xx = mpoly_eval(pc[:1], alpha_indices, _legendre_recurrence, [2, 0])[0]
            lp_yy = mpoly_eval(pc[:1], alpha_indices, _legendre_recurrence, [0, 2])[0]
        else:
            raise NotImplementedError("Laplacian polynomial-derivative for s_dim=3")
        lp = lp_xx + lp_yy
        if scale_polynomial:
            lp = lp / (w_scale ** 2)
        rhs[n:] = lp
    else:
        i, j = deriv
        order = i + j
        if order < 1 or order > 2:
            raise NotImplementedError(f"deriv {deriv} not supported")
        # ∂_α φ at the centre (with respect to physical x = stencil[0])
        dx = centre[0] - p[:, 0]  # = -(p[:,0] - centre[0]); sign matters!
        dy = centre[1] - p[:, 1]
        # NOTE: ∂φ(|c-xj|)/∂c_x = (dφ/dr) · (c_x - xj_x)/r = (xj_x - c_x) · _phs_drbf_over_r
        #   But we want the derivative WRT the centre c (the evaluation point), not WRT xj.
        #   φ depends on |c-xj|, so ∂_{c_x} = -∂_{xj_x}. Sign matters for second
        #   derivatives; for first derivatives, the (c_x - xj_x) sign convention here
        #   assigns weights such that Σ w_j u(xj) ≈ ∂u/∂x at c.
        #
        # Reproducing the Matlab convention (FormGradients.m line 87):
        #   Bx = [(xj(1,:) − xk(1,:)) .* D, gpx(1,:)]
        #   where xj(1,:) = c_x repeated and xk(1,:) = stencil x's.
        # So the convention is (c_x - xj_x), matching our `dx = centre[0] - p[:,0]`.
        D = _phs_drbf_over_r(rd[0, :], params.m)
        if order == 1:
            if i == 1 and j == 0:  # ∂_x
                rhs[:n] = dx * D
            elif i == 0 and j == 1:  # ∂_y
                rhs[:n] = dy * D
            else:
                raise NotImplementedError(deriv)
        else:  # order == 2
            # second-derivative kernel formulas with φ = (r+ε)^m:
            #   ∂²φ/∂x² = m·r^{m-2} + m·(m-2)·r^{m-4}·dx²
            #   ∂²φ/∂y² = m·r^{m-2} + m·(m-2)·r^{m-4}·dy²
            #   ∂²φ/∂x∂y =          m·(m-2)·r^{m-4}·dx·dy
            r_safe = rd[0, :] + _EPS
            t_a = params.m * r_safe ** (params.m - 2)
            t_b = params.m * (params.m - 2) * r_safe ** (params.m - 4)
            if (i, j) == (2, 0):
                rhs[:n] = t_a + t_b * dx * dx
            elif (i, j) == (0, 2):
                rhs[:n] = t_a + t_b * dy * dy
            elif (i, j) == (1, 1):
                rhs[:n] = t_b * dx * dy
            else:
                raise NotImplementedError(deriv)

        # ∂_α p_α evaluated at the centre
        d_p = mpoly_eval(pc[:1], alpha_indices, _legendre_recurrence, deriv=[i, j])[0]
        if scale_polynomial:
            d_p = d_p / (w_scale ** order)
        rhs[n:] = d_p

    sol = np.linalg.solve(A, rhs)
    weights = sol[:n]
    return weights


# ---------------------------------------------------------------------------
# 5. High-level operator-builder
# ---------------------------------------------------------------------------


def build_operators(
    Xi: np.ndarray,
    Xb: np.ndarray,
    normals: np.ndarray,
    p: int,
    *,
    derivs: Iterable[Tuple[int, int] | str] = (
        (1, 0),
        (0, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        "lap",
    ),
    centres_kind: str = "int_bd",
    s_dim: int = 2,
) -> Dict[Tuple[int, int] | str, scipy.sparse.csr_matrix]:
    """Build sparse RBF-FD operators on Xf=[Xi;Xb;Xg] for the requested derivatives.

    centres_kind ∈ {"int_bd", "full"}:
      - "int_bd" (default, paper convention): rows are [Xi;Xb] → operator
        shape (Ni+Nb, Ni+Nb+Ng).
      - "full": rows are Xf → operator shape (Nf, Nf). Required when chaining
        derivatives (e.g., NS viscous-flux divergence ∂_x(ν_eff·∂u/∂x)) so
        that the inner-derivative result lives on the same point set the
        outer derivative consumes.

    derivs: any subset of {(1,0), (0,1), (2,0), (0,2), (1,1), 'lap'}.
    p: RBF-FD order (≥2).

    Returns dict mapping each derivative to a CSR sparse matrix in fp64.
    Per paper convention: first derivatives use scaled polynomial coords;
    second derivatives + Laplacian use unscaled.
    """
    Xi = np.asarray(Xi, dtype=np.float64)
    Xb = np.asarray(Xb, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if Xi.shape[1] != s_dim or Xb.shape[1] != s_dim or normals.shape[1] != s_dim:
        raise ValueError("Xi, Xb, normals must all have s_dim columns")
    if Xb.shape[0] != normals.shape[0]:
        raise ValueError("Xb and normals must have the same row count")

    Ni, Nb = Xi.shape[0], Xb.shape[0]
    Nf_no_ghost = Ni + Nb
    h = 1.0 / (Nf_no_ghost ** (1.0 / s_dim))
    Xg = Xb + 0.25 * h * normals
    Xf = np.vstack([Xi, Xb, Xg])
    if centres_kind == "int_bd":
        centres = np.vstack([Xi, Xb])
    elif centres_kind == "full":
        centres = Xf
    else:
        raise ValueError(f"unknown centres_kind {centres_kind!r}")

    # KD-tree on Xf (paper convention)
    tree = cKDTree(Xf)

    # Group derivs by (theta, scale_polynomial). Different (theta,scale) imply
    # different RBFFDParams and different A-matrix structures, so they need
    # separate per-stencil solves. Per-derivative `deriv` shares the A-matrix
    # within a group (only RHS differs).
    groups: Dict[Tuple[int, bool], list] = {}
    for d in derivs:
        if isinstance(d, str):
            if d != "lap":
                raise ValueError(f"unknown deriv string {d!r}")
            theta = 2
            scale_poly = False
        else:
            i, j = d
            if i + j == 1:
                theta = 1
                scale_poly = True
            elif i + j == 2:
                theta = 2
                scale_poly = False
            else:
                raise ValueError(f"unsupported deriv order {d}")
        groups.setdefault((theta, scale_poly), []).append(d)

    out: Dict[Tuple[int, int] | str, scipy.sparse.csr_matrix] = {}

    for (theta, scale_poly), derivs_in_group in groups.items():
        params = RBFFDParams.from_orders(s_dim, p, theta)
        n = params.stencil_size
        alpha_indices = total_degree_indices(s_dim, params.ell)
        # COO assemblies, one per derivative in this group
        rows_acc: Dict[Tuple[int, int] | str, list] = {d: [] for d in derivs_in_group}
        cols_acc: Dict[Tuple[int, int] | str, list] = {d: [] for d in derivs_in_group}
        vals_acc: Dict[Tuple[int, int] | str, list] = {d: [] for d in derivs_in_group}

        # Need n nearest neighbours from Xf for every centre point.
        # We query in one batch, then per-stencil solve.
        _, idx_all = tree.query(centres, k=n)  # (Ni+Nb, n)

        for ic in range(centres.shape[0]):
            je = idx_all[ic]
            stencil_xy = Xf[je]
            # Build A once per stencil: factor and reuse for all derivatives in this group
            # We share the full matrix A but solve with different RHS for each deriv.
            # Construct A explicitly to factor and reuse:
            diffs = stencil_xy[:, None, :] - stencil_xy[None, :, :]
            rd = np.sqrt(np.maximum(np.sum(diffs * diffs, axis=2), 0.0))
            A_rbf = _phs(rd, params.m)

            if scale_poly:
                w_scale = rd[0, n - 1]
                if w_scale <= 0:
                    raise RuntimeError(
                        f"degenerate stencil at centre {ic}: zero scaling distance"
                    )
                pc = (stencil_xy - stencil_xy[0]) / w_scale
            else:
                w_scale = 1.0
                pc = stencil_xy

            v = mpoly_eval(pc, alpha_indices, _legendre_recurrence)
            A = np.block(
                [
                    [A_rbf, v],
                    [v.T, np.zeros((params.poly_count, params.poly_count))],
                ]
            )
            try:
                lu_piv = scipy.linalg.lu_factor(A)
            except Exception:
                # rare degenerate stencil — skip with one-stencil-per-iter fallback
                lu_piv = None

            for d in derivs_in_group:
                rhs = np.empty(n + params.poly_count, dtype=np.float64)
                if d == "lap":
                    rhs[:n] = _phs_laplacian(rd[0, :], params.m, params.s_dim)
                    if params.s_dim == 2:
                        lp_xx = mpoly_eval(
                            pc[:1], alpha_indices, _legendre_recurrence, [2, 0]
                        )[0]
                        lp_yy = mpoly_eval(
                            pc[:1], alpha_indices, _legendre_recurrence, [0, 2]
                        )[0]
                        lp = lp_xx + lp_yy
                    else:
                        raise NotImplementedError("3D Laplacian of polynomial basis")
                    if scale_poly:
                        lp = lp / (w_scale ** 2)
                    rhs[n:] = lp
                else:
                    i, j = d
                    order = i + j
                    dx = stencil_xy[0, 0] - stencil_xy[:, 0]
                    dy = stencil_xy[0, 1] - stencil_xy[:, 1]
                    if order == 1:
                        D = _phs_drbf_over_r(rd[0, :], params.m)
                        rhs[:n] = (dx * D) if (i, j) == (1, 0) else (dy * D)
                    else:
                        r_safe = rd[0, :] + _EPS
                        t_a = params.m * r_safe ** (params.m - 2)
                        t_b = (
                            params.m
                            * (params.m - 2)
                            * r_safe ** (params.m - 4)
                        )
                        if (i, j) == (2, 0):
                            rhs[:n] = t_a + t_b * dx * dx
                        elif (i, j) == (0, 2):
                            rhs[:n] = t_a + t_b * dy * dy
                        elif (i, j) == (1, 1):
                            rhs[:n] = t_b * dx * dy
                    d_p = mpoly_eval(
                        pc[:1], alpha_indices, _legendre_recurrence, deriv=[i, j]
                    )[0]
                    if scale_poly:
                        d_p = d_p / (w_scale ** order)
                    rhs[n:] = d_p

                if lu_piv is not None:
                    sol = scipy.linalg.lu_solve(lu_piv, rhs)
                else:
                    sol = np.linalg.solve(A, rhs)
                w = sol[:n]
                rows_acc[d].append(np.full(n, ic, dtype=np.int64))
                cols_acc[d].append(je.astype(np.int64))
                vals_acc[d].append(w)

        for d in derivs_in_group:
            rows = np.concatenate(rows_acc[d])
            cols = np.concatenate(cols_acc[d])
            vals = np.concatenate(vals_acc[d])
            shape = (centres.shape[0], Xf.shape[0])
            mat = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
            out[d] = mat

    out["__metadata__"] = dict(
        Xi=Xi, Xb=Xb, Xg=Xg, Xf=Xf, normals=normals, h=h, p=p, s_dim=s_dim,
        Ni=Ni, Nb=Nb, Ng=Xg.shape[0], centres_kind=centres_kind,
    )
    return out


# Lazy import for scipy.linalg.lu_factor in build_operators above.
import scipy.linalg  # noqa: E402  (placed here to keep the top of the file lean)


# ---------------------------------------------------------------------------
# 6. Quasi-uniform node generators for our test domains
# ---------------------------------------------------------------------------


def _halton(n: int, base: int, skip: int = 1) -> np.ndarray:
    """Halton sequence of length n in base `base`, starting at index `skip`."""
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = i + skip
        f = 1.0
        v = 0.0
        while idx > 0:
            f /= base
            v += f * (idx % base)
            idx //= base
        out[i] = v
    return out


def gen_rectangle_nodes(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    N_target: int,
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate quasi-uniform interior + boundary nodes on a rectangle.

    N_target is approximate (Ni+Nb). Uses Halton (bases 2,3) for interior with
    a small jitter for genericity, plus equispaced boundary nodes.

    Returns (Xi, Xb, normals, h).
    """
    rng = np.random.default_rng(seed)
    Lx, Ly = xmax - xmin, ymax - ymin
    area = Lx * Ly
    h = math.sqrt(area / N_target)

    # Boundary first — equispaced along ∂Ω, with Nb_target ≈ Perimeter / h
    perim = 2 * (Lx + Ly)
    Nb_target = max(int(round(perim / h)), 4)
    # Distribute to four edges proportional to length
    n_x = max(int(round(Nb_target * Lx / perim)), 2)
    n_y = max(int(round(Nb_target * Ly / perim)), 2)
    # Place Nb so corners are visited exactly once
    pts = []
    nrm = []
    # bottom edge y=ymin: (x, ymin), normal (0, -1)
    xs = np.linspace(xmin, xmax, n_x + 1, endpoint=False)
    pts.append(np.column_stack([xs, np.full_like(xs, ymin)]))
    nrm.append(np.tile([0.0, -1.0], (n_x + 1, 1)))
    # right edge x=xmax: (xmax, y), normal (1, 0)
    ys = np.linspace(ymin, ymax, n_y + 1, endpoint=False)
    pts.append(np.column_stack([np.full_like(ys, xmax), ys]))
    nrm.append(np.tile([1.0, 0.0], (n_y + 1, 1)))
    # top edge y=ymax: (x, ymax), normal (0, 1)
    xs2 = np.linspace(xmax, xmin, n_x + 1, endpoint=False)
    pts.append(np.column_stack([xs2, np.full_like(xs2, ymax)]))
    nrm.append(np.tile([0.0, 1.0], (n_x + 1, 1)))
    # left edge x=xmin: (xmin, y), normal (-1, 0)
    ys2 = np.linspace(ymax, ymin, n_y + 1, endpoint=False)
    pts.append(np.column_stack([np.full_like(ys2, xmin), ys2]))
    nrm.append(np.tile([-1.0, 0.0], (n_y + 1, 1)))
    Xb = np.vstack(pts)
    normals = np.vstack(nrm)
    Nb = Xb.shape[0]

    # Interior — Halton in (xmin+0.5h, xmax-0.5h) × analogous, then keep only
    # those at distance > 0.5h from the boundary.
    Ni_target = max(N_target - Nb, 16)
    # Halton produces points in (0,1); we want them in (xmin+buf, xmax-buf)
    buf = 0.5 * h
    inner_Lx = Lx - 2 * buf
    inner_Ly = Ly - 2 * buf
    if inner_Lx <= 0 or inner_Ly <= 0:
        raise ValueError("buffer too large for rectangle")
    # Generate ~1.4× target to allow rejection
    n_gen = int(1.4 * Ni_target) + 16
    h2 = _halton(n_gen, 2, skip=1)
    h3 = _halton(n_gen, 3, skip=1)
    # tiny jitter for genericity (does not break quasi-uniformity for h_jit ≪ h)
    jitter = (rng.random((n_gen, 2)) - 0.5) * 0.1 * h
    cand_x = xmin + buf + h2 * inner_Lx + jitter[:, 0]
    cand_y = ymin + buf + h3 * inner_Ly + jitter[:, 1]
    # Clamp into interior (rare jitter overflow)
    cand_x = np.clip(cand_x, xmin + 0.4 * h, xmax - 0.4 * h)
    cand_y = np.clip(cand_y, ymin + 0.4 * h, ymax - 0.4 * h)
    Xi = np.column_stack([cand_x, cand_y])[:Ni_target]

    return Xi, Xb, normals, h


def load_disk_nodes(
    mat_path: str, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the paper's pre-stored quasi-uniform unit-disk node set.

    mat_path: path to DiskPoissonNodes.mat or DiskPoissonNodesLarge.mat.
    k: 1-based set index (sizes vary per file; see the README in this package).

    Returns (Xi, Xb, normals, h).
    """
    from scipy.io import loadmat

    m = loadmat(mat_path)
    Xi = np.asarray(m["fullintnodes"][k - 1, 0], dtype=np.float64)
    Xb = np.asarray(m["bdrynodes"][k - 1, 0], dtype=np.float64)
    normals = np.asarray(m["normals"][k - 1, 0], dtype=np.float64)
    h = 1.0 / math.sqrt(Xi.shape[0] + Xb.shape[0])  # 2D
    return Xi, Xb, normals, h


# ---------------------------------------------------------------------------
# 7. Polynomial-recovery validation (Phase 1 gate)
# ---------------------------------------------------------------------------


def validate_polynomial_recovery(
    Xi: np.ndarray,
    Xb: np.ndarray,
    normals: np.ndarray,
    p: int,
    *,
    s_dim: int = 2,
    derivs: Sequence[Tuple[int, int] | str] = (
        (1, 0),
        (0, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        "lap",
    ),
    tol: float = 1e-9,
) -> Dict[str, Dict[str, float]]:
    """Apply each operator to every monomial f(x,y) = x^a · y^b with a+b ≤ ell,
    check the constructed operator output matches the analytic derivative.

    Returns a nested dict keyed by deriv label → {polynomial → max_abs_error}.
    Raises AssertionError if any error exceeds tol.
    """
    ops = build_operators(Xi, Xb, normals, p, derivs=tuple(derivs), s_dim=s_dim)
    Xf = ops["__metadata__"]["Xf"]
    centres = np.vstack([Xi, Xb])

    params_grad = RBFFDParams.from_orders(s_dim, p, 1)
    params_lap = RBFFDParams.from_orders(s_dim, p, 2)
    # ell can differ between gradient and laplacian groups; iterate all valid (a,b)
    # for the larger ell.
    ell_max = max(params_grad.ell, params_lap.ell)

    out: Dict[str, Dict[str, float]] = {}
    failures: list[Tuple[str, str, float]] = []

    for d in derivs:
        d_label = "lap" if d == "lap" else f"({d[0]},{d[1]})"
        out[d_label] = {}
        # determine operator's "effective ell" for this derivative
        if d == "lap" or (isinstance(d, tuple) and (d[0] + d[1]) == 2):
            ell_eff = params_lap.ell
        else:
            ell_eff = params_grad.ell
        for a in range(ell_eff + 1):
            for b in range(ell_eff + 1 - a):
                f = (Xf[:, 0] ** a) * (Xf[:, 1] ** b)
                if d == "lap":
                    # ∂xx + ∂yy of x^a · y^b
                    d2x = (
                        (a * (a - 1)) * (centres[:, 0] ** max(a - 2, 0)) * (centres[:, 1] ** b)
                        if a >= 2
                        else np.zeros(centres.shape[0])
                    )
                    d2y = (
                        (b * (b - 1)) * (centres[:, 0] ** a) * (centres[:, 1] ** max(b - 2, 0))
                        if b >= 2
                        else np.zeros(centres.shape[0])
                    )
                    analytic = d2x + d2y
                elif d == (1, 0):
                    analytic = (
                        a * (centres[:, 0] ** max(a - 1, 0)) * (centres[:, 1] ** b)
                        if a >= 1
                        else np.zeros(centres.shape[0])
                    )
                elif d == (0, 1):
                    analytic = (
                        b * (centres[:, 0] ** a) * (centres[:, 1] ** max(b - 1, 0))
                        if b >= 1
                        else np.zeros(centres.shape[0])
                    )
                elif d == (2, 0):
                    analytic = (
                        a * (a - 1) * (centres[:, 0] ** max(a - 2, 0)) * (centres[:, 1] ** b)
                        if a >= 2
                        else np.zeros(centres.shape[0])
                    )
                elif d == (0, 2):
                    analytic = (
                        b * (b - 1) * (centres[:, 0] ** a) * (centres[:, 1] ** max(b - 2, 0))
                        if b >= 2
                        else np.zeros(centres.shape[0])
                    )
                elif d == (1, 1):
                    analytic = (
                        a * b * (centres[:, 0] ** max(a - 1, 0)) * (centres[:, 1] ** max(b - 1, 0))
                        if a >= 1 and b >= 1
                        else np.zeros(centres.shape[0])
                    )
                else:
                    raise NotImplementedError(d)

                pred = ops[d] @ f
                err = float(np.max(np.abs(pred - analytic)))
                key = f"x^{a}·y^{b}"
                out[d_label][key] = err
                if not np.isfinite(err) or err > tol:
                    failures.append((d_label, key, err))

    if failures:
        msg = "\n".join(f"  {d} f={k}: max_err={e:.3e} > tol={tol:.1e}" for d, k, e in failures)
        raise AssertionError(
            f"Polynomial-recovery validation failed for {len(failures)} (deriv, polynomial) pairs:\n{msg}"
        )

    return out


# ---------------------------------------------------------------------------
# 8. PyTorch sparse wrapping (consumed by training loops in Phase 4)
# ---------------------------------------------------------------------------


def to_torch_sparse(
    op: scipy.sparse.spmatrix,
    *,
    dtype=None,
    device=None,
):
    """Convert a SciPy sparse matrix to a PyTorch sparse_coo_tensor in fp64.

    Lazily imports torch so this module can be imported without torch installed.
    """
    import torch as _torch

    if dtype is None:
        dtype = _torch.float64
    op = op.tocoo()
    indices = np.vstack([op.row, op.col])
    return _torch.sparse_coo_tensor(
        _torch.from_numpy(indices.astype(np.int64)),
        _torch.from_numpy(op.data.astype(np.float64)).to(dtype=dtype),
        size=op.shape,
        device=device,
    ).coalesce()
