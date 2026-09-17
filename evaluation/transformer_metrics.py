"""Regression metrics for the NeuroScale Transformer.

All functions operate on plain NumPy arrays so they are reusable outside
the training loop, in evaluation scripts, and in tests.

Target layout
-------------
Predictions and ground-truth arrays both have shape
``(N, prediction_horizon, num_targets)`` where:

  - axis 0: samples
  - axis 1: prediction steps
  - axis 2: ``[cpu_target, memory_target]``

The convenience functions :func:`cpu_metrics` and :func:`memory_metrics`
collapse axes 0 and 1 to produce per-target scalar metrics.
"""

from typing import Dict

import numpy as np

# Column indices within the target axis-2 dimension
CPU_IDX = 0
MEMORY_IDX = 1


# ---------------------------------------------------------------------------
# Core metric functions (operate on 1-D or N-D flattened arrays)
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    diff = y_true.ravel() - y_pred.ravel()
    return float(np.mean(diff ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true.ravel() - y_pred.ravel())))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Per-target convenience helpers
# ---------------------------------------------------------------------------

def cpu_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE / MAE / RMSE for the CPU target column.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape ``(N, prediction_horizon, num_targets)`` or ``(N, num_targets)``.
    """
    t = y_true[..., CPU_IDX]
    p = y_pred[..., CPU_IDX]
    return {"mse": mse(t, p), "mae": mae(t, p), "rmse": rmse(t, p)}


def memory_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE / MAE / RMSE for the memory target column.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape ``(N, prediction_horizon, num_targets)`` or ``(N, num_targets)``.
    """
    t = y_true[..., MEMORY_IDX]
    p = y_pred[..., MEMORY_IDX]
    return {"mse": mse(t, p), "mae": mae(t, p), "rmse": rmse(t, p)}


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute overall MSE / MAE / RMSE across all targets.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Any broadcastable shape; all elements are compared.
    """
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def full_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return a nested dict of metrics for cpu, memory, and overall.

    >>> report = full_report(y_true, y_pred)
    >>> report["cpu"]["mae"]
    """
    return {
        "cpu": cpu_metrics(y_true, y_pred),
        "memory": memory_metrics(y_true, y_pred),
        "overall": overall_metrics(y_true, y_pred),
    }
