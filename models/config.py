"""Transformer model configuration for NeuroScale."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TransformerConfig:
    """Configuration for the NeuroScale Transformer Encoder model.

    All fields are keyword arguments with defaults chosen for low-resource
    (CPU-only) experimentation.  Override as needed for larger datasets.

    Attributes
    ----------
    d_model : int
        Internal embedding dimension of the Transformer.
    nhead : int
        Number of attention heads.  Must divide ``d_model`` evenly.
    num_encoder_layers : int
        Number of stacked Transformer Encoder layers.
    dim_feedforward : int
        Hidden dimension of the feed-forward sub-layer.
    dropout : float
        Dropout probability applied inside the Transformer.
    prediction_horizon : int
        Number of future steps to predict.  Must match
        :attr:`data.config.PipelineConfig.prediction_horizon`.
    num_targets : int
        Number of target variables (default 2: CPU %, memory %).
    feature_count : int
        Number of input features per time-step.  Derived at runtime from
        :attr:`data.config.PipelineConfig.feature_columns`.
    seed : int
        Global random seed for reproducibility.
    checkpoint_dir : str
        Directory where model checkpoints are written.
    """

    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    prediction_horizon: int = 1
    num_targets: int = 2
    feature_count: int = 4          # len(PipelineConfig.feature_columns)
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
