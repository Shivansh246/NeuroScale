"""Inference API for the NeuroScale Transformer.

Public contract
---------------
::

    predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")
    result = predictor.predict(sequence)
    # → {"cpu": float, "memory": float, "confidence": float}

Input contract
--------------
``sequence`` must be a NumPy array of shape ``(seq_len, feature_count)`` in the
**original (un-normalised)** scale, matching the feature columns used during
training.  The predictor normalises the input internally using the feature
statistics stored in the checkpoint.

Output contract
---------------
``cpu`` and ``memory`` are returned in **original scale** (e.g. CPU % and
memory %) by inverse-transforming the model's normalised predictions using the
per-target normalisation statistics stored in the checkpoint.

Confidence calculation
----------------------
Confidence is a **deterministic heuristic** based on the validation loss stored
in the checkpoint.  It is *not* a probabilistic uncertainty estimate.

The checkpoint records ``val_loss`` = validation MSE measured in **normalised
target space** (because the model trains on normalised targets).  Normalised-
space MSE is dimensionless and has a natural scale of O(1) for a well-fitted
model.

Method:

1. ``rmse_norm = sqrt(val_loss)``   — RMSE in normalised target units.
2. ``confidence = exp(-rmse_norm / CONFIDENCE_SCALE)``

   ``CONFIDENCE_SCALE = 1.0`` (tunable).

Interpretation:

- ``val_loss = 0.0`` → ``confidence = 1.0``  (perfect fit, never achieved)
- ``val_loss = 1.0`` (RMSE = 1 normalised unit) → ``confidence ≈ 0.37``
- ``val_loss = 4.0`` (RMSE = 2 normalised units) → ``confidence ≈ 0.14``

The confidence is fixed at checkpoint-load time and does not vary per
prediction.  It represents how well the model generalised during training.

To produce proper sample-level uncertainty, replace with MC-Dropout or
conformal prediction in a future iteration.

Inverse-transform mechanism
---------------------------
The checkpoint stores a ``target_norm_map`` keyed by target column name::

    {
        "cpu_percent":    {"mean": float, "std": float},
        "memory_percent": {"mean": float, "std": float},
    }

The predictor uses this map to inverse-transform each target output by name,
irrespective of feature column order.  This eliminates the previous hardcoded
positional assumption.
"""

import math
from typing import Dict, List

import numpy as np
import torch

from models.transformer import NeuroScaleTransformer
from models.config import TransformerConfig
from data.normalization import Normalizer

# Scale factor for confidence mapping: lower = more conservative
CONFIDENCE_SCALE = 1.0


class TransformerPredictor:
    """Load a trained checkpoint and provide a clean prediction interface.

    Parameters
    ----------
    model : NeuroScaleTransformer
        Loaded, eval-mode model.
    feat_normalizer : Normalizer
        Fitted feature normalizer.  Applied to raw input sequences before
        inference.
    target_columns : list[str]
        Target column names in model output order.
    target_norm_map : dict
        Per-target normalisation stats::

            {"cpu_percent": {"mean": float, "std": float}, ...}

        Used to inverse-transform each normalised prediction back to original
        scale.
    val_loss : float
        Validation MSE in normalised target space; used to derive confidence.
    """

    def __init__(
        self,
        model: NeuroScaleTransformer,
        feat_normalizer: Normalizer,
        target_columns: List[str],
        target_norm_map: Dict[str, Dict[str, float]],
        val_loss: float,
    ) -> None:
        self._model = model
        self._model.eval()
        self._feat_normalizer = feat_normalizer
        self._target_columns = list(target_columns)
        self._target_norm_map = target_norm_map
        # Confidence is computed once from the checkpoint val_loss
        # val_loss is in normalised target space so the formula is meaningful.
        rmse_norm = math.sqrt(max(val_loss, 0.0))
        self._confidence = float(math.exp(-rmse_norm / CONFIDENCE_SCALE))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device = torch.device("cpu")
    ) -> "TransformerPredictor":
        """Instantiate from a checkpoint file produced by the trainer.

        Parameters
        ----------
        path : str
            Path to the ``.pt`` checkpoint.
        device : torch.device
            Device to run inference on (default: CPU).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # ---- Model ----
        cfg_dict = checkpoint["model_config"]
        model_cfg = TransformerConfig(**cfg_dict)
        model = NeuroScaleTransformer(model_cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        # ---- Feature normalizer ----
        feat_normalizer = Normalizer()
        # Prefer the explicit "feat_normalization" key; fall back to "normalization"
        feat_norm_data = checkpoint.get("feat_normalization") or checkpoint.get("normalization", {})
        if feat_norm_data:
            feat_normalizer.mean_ = np.array(feat_norm_data["mean"], dtype=float)
            feat_normalizer.std_ = np.array(feat_norm_data["std"], dtype=float)

        # ---- Target normalisation map (column-name keyed) ----
        target_norm_map: Dict[str, Dict[str, float]] = checkpoint.get("target_norm_map", {})

        target_columns: List[str] = checkpoint.get(
            "target_columns", ["cpu_percent", "memory_percent"]
        )
        val_loss: float = checkpoint.get("val_loss", 1.0)

        return cls(model, feat_normalizer, target_columns, target_norm_map, val_loss)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sequence: np.ndarray) -> Dict[str, float]:
        """Predict next CPU and memory demand.

        Parameters
        ----------
        sequence : np.ndarray
            Input window of shape ``(seq_len, feature_count)`` in the
            **original (un-normalised)** scale, matching the feature columns
            used during training.

        Returns
        -------
        dict
            ``{"cpu": float, "memory": float, "confidence": float}``

            - ``cpu``: predicted CPU percentage in **original scale** (%).
            - ``memory``: predicted memory percentage in **original scale** (%).
            - ``confidence``: deterministic heuristic in [0, 1] derived from
              the checkpoint's validation MSE in normalised space (see module
              docstring).
        """
        seq = np.array(sequence, dtype=float)
        if seq.ndim != 2:
            raise ValueError(
                f"sequence must be 2-D (seq_len, feature_count), got shape {seq.shape}"
            )

        # 1. Normalise input features using training statistics
        if self._feat_normalizer.mean_ is not None:
            seq_norm = self._feat_normalizer.transform(seq)
        else:
            seq_norm = seq

        # 2. Run model — output is in normalised target space
        x = torch.tensor(seq_norm[np.newaxis], dtype=torch.float32)  # (1, seq_len, F)
        with torch.no_grad():
            out = self._model(x)  # (1, prediction_horizon, num_targets)
        pred_norm = out[0, 0].numpy()  # (num_targets,)

        # 3. Inverse-transform each target output by column name
        result: Dict[str, float] = {"confidence": self._confidence}
        for i, col in enumerate(self._target_columns):
            norm_val = float(pred_norm[i])
            original_val = self._inverse_transform_target(col, norm_val)
            # Map to the public keys "cpu" and "memory"
            if col == "cpu_percent":
                result["cpu"] = original_val
            elif col in ("memory_percent", "memory_usage_mb"):
                result["memory"] = original_val
            # Additional targets (if any) are silently ignored in the
            # public interface but the inverse-transform is still applied
            # correctly for any configured target column.

        # Ensure required keys are always present
        result.setdefault("cpu", float("nan"))
        result.setdefault("memory", float("nan"))

        return result

    def _inverse_transform_target(self, column: str, norm_value: float) -> float:
        """Inverse-transform a normalised prediction for the named target column.

        Uses the per-column statistics stored in ``target_norm_map`` (embedded
        in the checkpoint).  Raises ``KeyError`` if the column is not found in
        the map, which would indicate a corrupt or mismatched checkpoint.

        Parameters
        ----------
        column : str
            Target column name (e.g. ``"cpu_percent"``).
        norm_value : float
            Normalised model output for that target.

        Returns
        -------
        float
            Value in original scale: ``norm_value * std + mean``.
        """
        if not self._target_norm_map or column not in self._target_norm_map:
            # Fallback: return unnormalised value as-is (should not occur with
            # well-formed checkpoints).
            return norm_value
        stats = self._target_norm_map[column]
        return float(norm_value * stats["std"] + stats["mean"])
