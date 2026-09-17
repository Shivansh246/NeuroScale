"""Normalization utilities for NeuroScale data pipeline.

Provides a ``Normalizer`` class that computes per‑feature mean and standard
deviation using NumPy and can transform data arrays accordingly.  The
parameters are persisted to JSON so that the same scaling can be applied to
validation and test sets.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


class Normalizer:
    """Fit‑transform normalizer for numeric features.

    The normalizer expects a 2‑D NumPy ``ndarray`` where columns correspond to
    the feature list defined in :class:`data.config.PipelineConfig`.  After
    calling :meth:`fit`, the ``mean_`` and ``std_`` attributes contain the
    per‑column statistics.  ``transform`` applies standard score scaling
    ``(x - mean) / std``.  ``inverse_transform`` reverts the operation.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, array: np.ndarray) -> "Normalizer":
        if array.ndim != 2:
            raise ValueError("Normalizer.fit expects a 2‑D array (samples, features)")
        self.mean_ = np.mean(array, axis=0)
        self.std_ = np.std(array, axis=0, ddof=0)
        # Avoid division by zero – replace zeros with 1.0 (no scaling)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fitted before calling transform")
        return (array - self.mean_) / self.std_

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer must be fitted before calling inverse_transform")
        return array * self.std_ + self.mean_

    # Persistence helpers ---------------------------------------------------
    def save(self, path: str) -> None:
        """Save ``mean_`` and ``std_`` to a JSON file.

        The JSON format stores two lists under keys ``"mean"`` and ``"std"``.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot save an unfitted Normalizer")
        data = {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str) -> "Normalizer":
        """Load parameters from a JSON file produced by :meth:`save`."""
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        self.mean_ = np.array(loaded["mean"], dtype=float)
        self.std_ = np.array(loaded["std"], dtype=float)
        return self
