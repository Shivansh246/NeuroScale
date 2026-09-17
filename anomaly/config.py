"""Configuration for the NeuroScale Autoencoder anomaly detector."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AutoencoderConfig:
    """Configuration for the fully-connected autoencoder.

    Attributes
    ----------
    input_dim : int
        Flattened window size = ``seq_len * feature_count``.
        Must be set by the caller to match the data pipeline window shape.
    latent_dim : int
        Size of the bottleneck (encoder output / decoder input).
        Smaller = more compression, higher sensitivity to small deviations.
    hidden_dims : list[int]
        Hidden layer widths in the encoder (decoder mirrors these in reverse).
        E.g. ``[128, 64]`` → encoder: input → 128 → 64 → latent;
        decoder: latent → 64 → 128 → input.
    dropout : float
        Dropout probability applied inside the encoder and decoder.
    seed : int
        Random seed for reproducibility.
    checkpoint_dir : str
        Directory for saving model checkpoints.
    threshold_k : float
        Multiplier for the standard deviation in the anomaly threshold rule:
        ``threshold = mean_val_error + k * std_val_error``.
        Higher k → fewer false positives, lower sensitivity.
    feature_columns : list[str]
        Names of the feature columns in order, mirroring ``PipelineConfig``.
        Stored in the checkpoint to detect column-order mismatches.
    seq_len : int
        Number of time-steps per window (``PipelineConfig.input_window``).
    """

    input_dim: int = 48                 # 12 steps × 4 features
    latent_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.0
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    threshold_k: float = 3.0
    feature_columns: List[str] = field(default_factory=lambda: [
        "cpu_percent",
        "cpu_usage_ns",
        "memory_usage_mb",
        "memory_percent",
    ])
    seq_len: int = 12
