"""Windowing utilities for NeuroScale.

Provides a function to generate sliding input/target windows from feature
and target arrays based on the configuration.
"""

import numpy as np
from typing import List, Tuple

from data.config import PipelineConfig


def generate_windows(
    features: np.ndarray,  # shape (N, num_features)
    targets: np.ndarray,   # shape (N, num_targets)
    cfg: PipelineConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate a list of (input_window, target_window) tuples.

    Parameters
    ----------
    features : np.ndarray
        2‑D array of feature values ordered chronologically.
    targets : np.ndarray
        2‑D array of target values ordered chronologically.
    cfg : PipelineConfig
        Configuration specifying ``input_window`` and ``prediction_horizon``.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Each tuple contains an input window of shape ``(input_window, F)``
        and a target window of shape ``(prediction_horizon, T)`` where ``F``
        and ``T`` are the numbers of feature and target columns.
    """
    iw = cfg.input_window
    ph = cfg.prediction_horizon
    total_len = features.shape[0]
    if total_len < iw + ph:
        return []  # Not enough data for a full window; return empty list instead of raising
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(total_len - iw - ph + 1):
        inp = features[start : start + iw]
        tgt = targets[start + iw : start + iw + ph]
        windows.append((inp, tgt))
    return windows
