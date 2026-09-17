"""Dataset construction utilities for NeuroScale.

The :func:`load_dataset` function orchestrates the full pipeline:

1. Load raw JSONL records.
2. Validate and resample to a deterministic interval.
3. Extract feature and target arrays based on :class:`PipelineConfig`.
4. Chronologically split into train/val/test according to ratios.
5. Fit a :class:`Normalizer` on the training features and persist the
   parameters.
6. Transform all splits using the fitted normalizer.
7. Generate sliding windows for each split.

The function returns a mapping ``{"train": [...], "val": [...], "test": [...]}``
where each list item is a ``(input_window, target_window)`` tuple of NumPy
arrays ready for a Transformer model.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data.config import PipelineConfig
from data.normalization import Normalizer
from data.preprocessing import load_jsonl, resample_records
from data.windows import generate_windows


def _extract_arrays(records, cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of validated records into feature and target arrays.

    Parameters
    ----------
    records : List[Dict]
        Validated metric dictionaries.
    cfg : PipelineConfig
        Configuration specifying which columns are features/targets.
    """
    # Preserve chronological order (records are already sorted after resampling)
    features = []
    targets = []
    for rec in records:
        features.append([rec[col] for col in cfg.feature_columns])
        targets.append([rec[col] for col in cfg.target_columns])
    return np.array(features, dtype=float), np.array(targets, dtype=float)


def _chronological_split(
    features: np.ndarray, targets: np.ndarray, cfg: PipelineConfig
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split feature/target arrays into train/val/test based on ratios.

    The split is performed on the first dimension (samples) preserving order.
    """
    n = features.shape[0]
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.val_ratio)
    # Ensure at least one sample per split when possible
    train_feat, train_tgt = features[:train_end], targets[:train_end]
    val_feat, val_tgt = features[train_end:val_end], targets[train_end:val_end]
    test_feat, test_tgt = features[val_end:], targets[val_end:]
    return (train_feat, train_tgt), (val_feat, val_tgt), (test_feat, test_tgt)


def load_dataset(cfg: PipelineConfig = None) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load the full dataset according to the pipeline configuration.

    Returns a dictionary with keys ``"train"``, ``"val"`` and ``"test"``.
    Each value is a list of ``(input_window, target_window)`` tuples.
    """
    cfg = cfg or PipelineConfig()
    # 1. Load raw data
    raw_path = Path(cfg.raw_data_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    records = load_jsonl(str(raw_path))

    # 2. Validate and resample
    resampled = resample_records(records, interval_seconds=cfg.sampling_interval_seconds)

    # 3. Extract feature/target arrays
    features, targets = _extract_arrays(resampled, cfg)

    # 4. Split
    (train_f, train_t), (val_f, val_t), (test_f, test_t) = _chronological_split(features, targets, cfg)

    # 5. Fit normalizer on training features only and persist
    normalizer = Normalizer().fit(train_f)
    # Ensure directory exists
    Path(cfg.normalization_path).parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(cfg.normalization_path)

    # 6. Transform all splits
    train_f_norm = normalizer.transform(train_f)
    val_f_norm = normalizer.transform(val_f)
    test_f_norm = normalizer.transform(test_f)

    # 7. Generate windows
    train_windows = generate_windows(train_f_norm, train_t, cfg)
    val_windows = generate_windows(val_f_norm, val_t, cfg)
    test_windows = generate_windows(test_f_norm, test_t, cfg)

    return {"train": train_windows, "val": val_windows, "test": test_windows}
