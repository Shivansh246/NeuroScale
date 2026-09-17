from dataclasses import dataclass, field
from typing import List

@dataclass
class PipelineConfig:
    """Configuration for the NeuroScale data pipeline.

    All defaults are chosen for quick local experimentation.
    """
    sampling_interval_seconds: int = 5  # how often to sample Docker metrics
    input_window: int = 12  # number of past samples in an input sequence (12*5s = 60s)
    prediction_horizon: int = 1  # steps ahead to predict (default next sample)
    feature_columns: List[str] = field(default_factory=lambda: [
        "cpu_percent",
        "cpu_usage_ns",
        "memory_usage_mb",
        "memory_percent",
    ])
    target_columns: List[str] = field(default_factory=lambda: [
        "cpu_percent",
        "memory_percent",
    ])
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Paths are relative to the repository root
    raw_data_path: str = "data/raw_metrics.jsonl"
    processed_data_path: str = "data/processed_metrics.jsonl"
    normalization_path: str = "data/normalization_params.json"
    """Note: callers should join these with the project root as needed."""
