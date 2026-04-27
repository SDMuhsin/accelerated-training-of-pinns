"""Static Linear Residual Map (SLRM) surrogate gradient.

Research phase 5 artefact — see llmdocs/research/research_log/04_design.md.

At problem setup, pick a reference input pred_ref and materialise the
Jacobian of the PDE residual DAG at pred_ref as a single constant
matrix. During every training step the surrogate "PDE gradient" is
that constant matrix applied once to the current (masked, flattened)
residual tensor. No tape is built and no chain rule is replayed at
any training step.

See also: llmdocs/research/research_log/03_decomposition.md for the
paradigm (design-space #4 surrogate scalar functional / #5 iterative
refinement, violating assumptions A1 and A6).
"""

from __future__ import annotations

from typing import Callable

import torch


def build_slrm_operator(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    pred_ref: torch.Tensor,
) -> torch.Tensor:
    """Materialise J(pred_ref)^T as a single dense matrix.

    Parameters
    ----------
    residual_fn : callable
        Takes a tensor of shape (N, K) and returns a tensor of shape
        (N, k) representing the PDE residual at every collocation
        point. Must be torch-autograd-traceable (no torch.no_grad).
    pred_ref : torch.Tensor
        Reference input of shape (N, K) at which to linearise the
        residual DAG. Its magnitude controls which nonlinear coupling
        terms are captured in the Jacobian.

    Returns
    -------
    M_ref : torch.Tensor
        Dense matrix of shape (N*K, N*k) containing the transpose of
        the flattened residual Jacobian at pred_ref. During training,
        a surrogate gradient is computed as
            g_flat = M_ref @ r_masked_flat,
        reshaped back to (N, K).
    """
    pred_ref = pred_ref.detach().clone()

    # torch.autograd.functional.jacobian with vectorize=True internally
    # uses vmap to compute all output-entry gradients in one shot.
    J = torch.autograd.functional.jacobian(
        residual_fn, pred_ref, vectorize=True, create_graph=False
    )
    # J has shape (N, k, N, K). Flatten to (N*k, N*K), then transpose
    # to (N*K, N*k) so that the right-hand side is a flattened residual.
    N, k, _, K = J.shape
    J_flat = J.reshape(N * k, N * K)
    M_ref = J_flat.T.contiguous()  # (N*K, N*k)
    return M_ref


def slrm_surrogate_grad(
    pred_pde: torch.Tensor,
    residual_fn_nograd: Callable[[torch.Tensor], torch.Tensor],
    M_ref: torch.Tensor,
    interior_mask: torch.Tensor,
    M_int: int,
) -> torch.Tensor:
    """Compute SLRM surrogate gradient of the PDE loss wrt pred_pde.

    The PDE loss the surrogate is targeting is the interior-mean
    squared residual, scaled per channel by 1/M_int:
        L_pde = (1/M_int) * sum_{i in interior, j} r[i,j]^2,
    whose exact gradient is (2/M_int) * J(pred_pde)^T r_masked. SLRM
    replaces J(pred_pde) by J(pred_ref) and never recomputes it.

    Parameters
    ----------
    pred_pde : torch.Tensor
        (N, K) current PDE-grid prediction, detached from the graph.
    residual_fn_nograd : callable
        Identical to the `residual_fn` passed to `build_slrm_operator`
        but called inside a `torch.no_grad()` context.
    M_ref : torch.Tensor
        Precomputed (N*K, N*k) matrix from `build_slrm_operator`.
    interior_mask : torch.Tensor
        (N, 1) broadcastable mask of 0/1 marking interior rows.
    M_int : int
        Number of interior points (denominator in L_pde).

    Returns
    -------
    grad_pde : torch.Tensor
        (N, K) surrogate gradient tensor to be used as the upstream
        gradient at pred_pde during `pred_batch.backward(...)`.
    """
    with torch.no_grad():
        r = residual_fn_nograd(pred_pde)  # (N, k)
        r = r * interior_mask  # zero boundary rows
        r_flat = r.reshape(-1)  # (N*k,)
        grad_flat = M_ref @ r_flat  # (N*K,)
        N, K = pred_pde.shape
        grad = grad_flat.reshape(N, K)
        return (2.0 / M_int) * grad
