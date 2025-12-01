"""
Metric functions for evaluating PINN solutions.
"""

import numpy as np
import torch


def compute_l2_error(pred, true, as_numpy=False):
    """
    Compute relative L2 error: ||pred - true|| / ||true||

    Args:
        pred: Predicted solution (torch.Tensor or np.ndarray)
        true: Ground truth solution (torch.Tensor or np.ndarray)
        as_numpy: If True, convert tensors to numpy before computation

    Returns:
        float: Relative L2 error
    """
    if isinstance(pred, torch.Tensor):
        if as_numpy:
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy() if isinstance(true, torch.Tensor) else true
        else:
            pred = pred.flatten()
            true = true.flatten() if isinstance(true, torch.Tensor) else torch.tensor(true).flatten()
            diff = pred - true
            return (torch.linalg.norm(diff) / torch.linalg.norm(true)).item()

    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def compute_mse(pred, true):
    """
    Compute mean squared error.

    Args:
        pred: Predicted solution
        true: Ground truth solution

    Returns:
        float: MSE value
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.flatten()
        true = true.flatten() if isinstance(true, torch.Tensor) else torch.tensor(true).flatten()
        return torch.nn.MSELoss()(pred, true).item()

    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    return np.mean((pred - true) ** 2)


def compute_linf(arr):
    """
    Compute L-infinity (max absolute) norm.

    Args:
        arr: Input array or tensor

    Returns:
        float: L-infinity norm
    """
    if isinstance(arr, torch.Tensor):
        return torch.linalg.norm(arr.flatten(), ord=float('inf')).item()
    return np.max(np.abs(arr))
