"""Dataset construction utilities for NeuroScale.

Normalization contract
----------------------
``load_dataset`` fits a **feature** Normalizer on training features only,
and a separate **target** Normalizer on training targets only.  Both are
persisted to JSON and used consistently across train / val / test splits.

The model is therefore trained on:

  X: normalised features  (mean~0, std~1)
  y: normalised targets   (mean~0, std~1)

Predictions from the model are in normalised target space.  The predictor
is responsible for inverse-transforming them back to original scale before
returning CPU% and memory% values.

The :func:`load_dataset` function orchestrates the full pipeline:

1. Load raw JSONL records.
2. Validate and resample to a deterministic interval.
3. Extract feature and target arrays based on :class:`PipelineConfig`.
4. Chronologically split into train/val/test according to ratios.
5. Fit a feature :class:`Normalizer` on training features only and persist.
6. Fit a target :class:`Normalizer` on training targets only and persist.
7. Transform all splits (features and targets) using the fitted normalizers.
8. Generate sliding windows for each split.

The function returns a mapping ``{"train": [...], "val": [...], "test": [...]}``
where each list item is a ``(input_window, target_window)`` tuple of NumPy
arrays.  Both arrays are in *normalised* space.
"""

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
    train_feat, train_tgt = features[:train_end], targets[:train_end]
    val_feat, val_tgt = features[train_end:val_end], targets[train_end:val_end]
    test_feat, test_tgt = features[val_end:], targets[val_end:]
    return (train_feat, train_tgt), (val_feat, val_tgt), (test_feat, test_tgt)


def _target_norm_path(normalization_path: str) -> str:
    """Derive the target normalizer path from the feature normalizer path."""
    p = Path(normalization_path)
    return str(p.parent / (p.stem + "_targets" + p.suffix))


def load_dataset(cfg: PipelineConfig = None) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load the full dataset according to the pipeline configuration.

    Returns a dictionary with keys ``"train"``, ``"val"`` and ``"test"``.
    Each value is a list of ``(input_window, target_window)`` tuples where
    **both X and y are in normalised space** (zero mean, unit variance per
    column, statistics fitted on the training split only).
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

    # 5. Fit feature normalizer on training features only and persist
    feat_normalizer = Normalizer().fit(train_f)
    Path(cfg.normalization_path).parent.mkdir(parents=True, exist_ok=True)
    feat_normalizer.save(cfg.normalization_path)

    # 6. Fit target normalizer on training targets only and persist
    tgt_normalizer = Normalizer().fit(train_t)
    tgt_normalizer.save(_target_norm_path(cfg.normalization_path))

    # 7. Transform all splits (features AND targets)
    train_f_norm = feat_normalizer.transform(train_f)
    val_f_norm = feat_normalizer.transform(val_f)
    test_f_norm = feat_normalizer.transform(test_f)

    train_t_norm = tgt_normalizer.transform(train_t)
    val_t_norm = tgt_normalizer.transform(val_t)
    test_t_norm = tgt_normalizer.transform(test_t)

    # 8. Generate windows — both X and y are normalised
    train_windows = generate_windows(train_f_norm, train_t_norm, cfg)
    val_windows = generate_windows(val_f_norm, val_t_norm, cfg)
    test_windows = generate_windows(test_f_norm, test_t_norm, cfg)

    return {"train": train_windows, "val": val_windows, "test": test_windows}
