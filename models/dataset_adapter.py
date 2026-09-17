"""PyTorch Dataset adapter for the NeuroScale data pipeline output.

The :class:`WindowDataset` wraps the list of ``(X, y)`` tuples produced by
:func:`data.dataset.load_dataset` into a standard ``torch.utils.data.Dataset``
so they can be consumed by a :class:`torch.utils.data.DataLoader`.

No window-generation logic is duplicated here.  All pre-processing (resampling,
normalisation, window creation) remains in the ``data`` package.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Dataset adapter wrapping ``List[(X, y)]`` tuples from the data pipeline.

    Parameters
    ----------
    windows : List[Tuple[np.ndarray, np.ndarray]]
        List of ``(input_window, target_window)`` pairs as returned by
        :func:`data.windows.generate_windows`.  ``X`` has shape
        ``(seq_len, feature_count)`` and ``y`` has shape
        ``(prediction_horizon, num_targets)``.
    """

    def __init__(self, windows: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        if not windows:
            raise ValueError("WindowDataset received an empty window list.")
        self._windows = windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_np, y_np = self._windows[idx]
        x = torch.tensor(x_np, dtype=torch.float32)  # (seq_len, feature_count)
        y = torch.tensor(y_np, dtype=torch.float32)  # (prediction_horizon, num_targets)
        return x, y
