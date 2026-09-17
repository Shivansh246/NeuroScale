"""Inference API for the NeuroScale Autoencoder anomaly detector.

Public contract
---------------
::

    detector = AutoencoderAnomalyDetector.from_checkpoint(
        "checkpoints/best_autoencoder.pt"
    )
    result = detector.score(sequence)
    # → {"score": float, "is_anomaly": bool}

Input contract
--------------
``sequence`` is a NumPy array of shape ``(seq_len, feature_count)`` in the
**original (un-normalised)** scale, with columns matching
``AutoencoderConfig.feature_columns`` in order.

The detector internally:
1. Normalises the window using the feature statistics stored in the checkpoint.
2. Flattens the normalised window to shape ``(seq_len * feature_count,)``.
3. Feeds it through the autoencoder.
4. Computes the per-sample mean-squared reconstruction error.
5. Compares the error against the stored threshold.

Output contract
---------------
``score`` is the mean-squared reconstruction error of the normalised input
in normalised feature space.  It is non-negative.  Its scale depends on how
well the autoencoder learned the training distribution.

``is_anomaly`` is ``True`` when ``score > threshold``.

The ``score`` is **not** a calibrated probability.  It is a deterministic
distance measure in the normalised feature space.

Threshold
---------
Stored in the checkpoint as:
    ``threshold = mean_val_error + k * std_val_error``
where the statistics are computed from reconstruction errors on *normal*
validation data after training.

The threshold is fixed at checkpoint save time and never recomputed at
inference.
"""

from typing import Dict, List

import numpy as np
import torch

from anomaly.autoencoder import Autoencoder
from anomaly.config import AutoencoderConfig
from anomaly.trainer import load_checkpoint
from data.normalization import Normalizer


class AutoencoderAnomalyDetector:
    """Anomaly detector wrapping a trained Autoencoder checkpoint.

    Parameters
    ----------
    model : Autoencoder
        Loaded eval-mode model.
    feat_normalizer : Normalizer
        Feature normalizer fitted during training.
    feature_columns : list[str]
        Expected feature column names in the correct order.
    seq_len : int
        Expected number of time-steps per window.
    threshold : float
        Anomaly threshold (reconstruction error above this → anomaly).
    """

    def __init__(
        self,
        model: Autoencoder,
        feat_normalizer: Normalizer,
        feature_columns: List[str],
        seq_len: int,
        threshold: float,
    ) -> None:
        self._model = model
        self._model.eval()
        self._feat_normalizer = feat_normalizer
        self._feature_columns = list(feature_columns)
        self._seq_len = seq_len
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device = torch.device("cpu"),
    ) -> "AutoencoderAnomalyDetector":
        """Instantiate from a checkpoint produced by the trainer.

        Parameters
        ----------
        path : str
            Path to the ``.pt`` checkpoint file.
        device : torch.device
            Inference device (default: CPU).
        """
        model, ckpt = load_checkpoint(path, device)

        # ---- Feature normalizer ----
        feat_normalizer = Normalizer()
        fn = ckpt.get("feat_normalization", {})
        if fn:
            feat_normalizer.mean_ = np.array(fn["mean"], dtype=float)
            feat_normalizer.std_ = np.array(fn["std"], dtype=float)

        feature_columns: List[str] = ckpt.get(
            "feature_columns",
            ["cpu_percent", "cpu_usage_ns", "memory_usage_mb", "memory_percent"],
        )
        seq_len: int = ckpt.get("seq_len", 12)
        threshold: float = ckpt.get("threshold", float("inf"))

        return cls(model, feat_normalizer, feature_columns, seq_len, threshold)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def score(self, sequence: np.ndarray) -> Dict[str, object]:
        """Compute the anomaly score for a single metric window.

        Parameters
        ----------
        sequence : np.ndarray
            Shape ``(seq_len, feature_count)`` in **original (un-normalised)**
            scale, with columns in the same order as ``feature_columns``.

        Returns
        -------
        dict
            ``{"score": float, "is_anomaly": bool}``

            - ``score``: mean-squared reconstruction error in normalised
              feature space.  Non-negative.  Not a probability.
            - ``is_anomaly``: ``True`` when ``score > threshold``.
        """
        seq = np.array(sequence, dtype=float)
        if seq.ndim != 2:
            raise ValueError(
                f"sequence must be 2-D (seq_len, feature_count), got shape {seq.shape}"
            )
        expected_cols = len(self._feature_columns)
        if seq.shape[1] != expected_cols:
            raise ValueError(
                f"sequence has {seq.shape[1]} columns but detector expects "
                f"{expected_cols} ({self._feature_columns})"
            )

        # 1. Normalise using training statistics
        if self._feat_normalizer.mean_ is not None:
            seq_norm = self._feat_normalizer.transform(seq)
        else:
            seq_norm = seq

        # 2. Flatten to (1, input_dim)
        x = torch.tensor(seq_norm.reshape(1, -1), dtype=torch.float32)

        # 3. Reconstruction error
        error_tensor = self._model.reconstruction_error(x)  # shape (1,)
        error = float(error_tensor[0].item())

        return {
            "score": error,
            "is_anomaly": error > self._threshold,
        }

    # ------------------------------------------------------------------
    # Accessors (useful for testing and logging)
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """The anomaly threshold stored in the checkpoint."""
        return self._threshold

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names expected by this detector."""
        return list(self._feature_columns)

    @property
    def seq_len(self) -> int:
        """Expected number of time-steps per input window."""
        return self._seq_len
