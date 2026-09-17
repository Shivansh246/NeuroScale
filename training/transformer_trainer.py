"""Reproducible training pipeline for the NeuroScale Transformer.

Normalization contract
----------------------
The trainer consumes the output of :func:`data.dataset.load_dataset`, which
returns windows where **both X (features) and y (targets) are in normalised
space** (zero mean, unit variance, statistics fitted on the training split).

The model therefore learns:

    normalised_X → normalised_y

Validation loss (MSE) is computed in normalised target space.  This makes the
confidence heuristic (``exp(-sqrt(val_loss))``) meaningful and bounded:

- Perfectly calibrated model: val_loss → 0, confidence → 1.0
- RMSE of 1 normalised unit: val_loss = 1.0, confidence ≈ 0.37
- RMSE of 2 normalised units: val_loss = 4.0, confidence ≈ 0.14

The checkpoint stores separate per-target normalizer statistics so the
predictor can inverse-transform each target column by name, independent of
feature column ordering.

Usage (CLI)
-----------
::

    python -m training.transformer_trainer \\
        --raw-data data/raw_metrics.jsonl  \\
        --epochs 50                         \\
        --batch-size 32                     \\
        --lr 1e-3                           \\
        --checkpoint-dir checkpoints        \\
        --seed 42
"""

import argparse
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.config import PipelineConfig
from data.dataset import load_dataset, _target_norm_path
from data.normalization import Normalizer
from models.config import TransformerConfig
from models.dataset_adapter import WindowDataset
from models.transformer import NeuroScaleTransformer
from evaluation.transformer_metrics import full_report


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    CPU-only runs are fully deterministic.
    CUDA runs may still be non-deterministic unless ``CUBLAS_WORKSPACE_CONFIG``
    is set in the environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

class TrainingConfig:
    """Simple container for training hyper-parameters."""

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir


# ---------------------------------------------------------------------------
# Core train/val loops
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> float:
    """Run one epoch of training or validation.

    Returns the average loss over all batches.
    X and y fed through the loader are both in normalised space.
    """
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)           # (batch, ph, num_targets) — normalised
            loss = criterion(pred, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _build_target_norm_map(
    target_columns: List[str],
    tgt_normalizer: Normalizer,
) -> Dict[str, Dict[str, float]]:
    """Build a per-target-column normalisation stats mapping.

    Returns a dict keyed by target column name, each containing the scalar
    ``mean`` and ``std`` for that target.  This is the authoritative source
    for inverse-transforming model predictions in the predictor.

    Example::

        {
            "cpu_percent":    {"mean": 42.3, "std": 14.1},
            "memory_percent": {"mean": 12.7, "std": 3.8},
        }
    """
    if tgt_normalizer.mean_ is None:
        return {}
    return {
        col: {
            "mean": float(tgt_normalizer.mean_[i]),
            "std": float(tgt_normalizer.std_[i]),
        }
        for i, col in enumerate(target_columns)
    }


def save_checkpoint(
    path: str,
    model: NeuroScaleTransformer,
    model_cfg: TransformerConfig,
    pipeline_cfg: PipelineConfig,
    feat_normalizer: Normalizer,
    tgt_normalizer: Normalizer,
    epoch: int,
    val_loss: float,
    train_loss: float,
) -> None:
    """Persist the model and all metadata needed for inference.

    Checkpoint keys
    ---------------
    ``state_dict``
        PyTorch ``model.state_dict()``.
    ``model_config``
        :class:`models.config.TransformerConfig` as a plain dict.
    ``feature_columns``
        Ordered list of input feature names.
    ``target_columns``
        Ordered list of target names (model output order).
    ``feat_normalization``
        Feature normalizer: ``{"mean": [...], "std": [...]}`` (one entry per
        feature column).  Used to normalise input sequences at inference time.
    ``target_norm_map``
        Per-target normaliser stats keyed by column name::

            {"cpu_percent": {"mean": float, "std": float}, ...}

        Used to inverse-transform each target prediction independently,
        regardless of feature column ordering.
    ``epoch``
        Epoch number when this checkpoint was saved.
    ``val_loss``
        Validation MSE in **normalised target space** at that epoch.
    ``train_loss``
        Training MSE in normalised target space at that epoch.
    """
    feat_norm_data: Dict = {}
    if feat_normalizer.mean_ is not None:
        feat_norm_data = {
            "mean": feat_normalizer.mean_.tolist(),
            "std": feat_normalizer.std_.tolist(),
        }

    tgt_norm_map = _build_target_norm_map(pipeline_cfg.target_columns, tgt_normalizer)

    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "feature_columns": list(pipeline_cfg.feature_columns),
        "target_columns": list(pipeline_cfg.target_columns),
        "feat_normalization": feat_norm_data,
        # Legacy alias so old code still gets something from "normalization"
        "normalization": feat_norm_data,
        "target_norm_map": tgt_norm_map,
        "epoch": epoch,
        "val_loss": val_loss,
        "train_loss": train_loss,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")):
    """Load a checkpoint and return (model, checkpoint_dict).

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    tuple[NeuroScaleTransformer, dict]
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = checkpoint["model_config"]
    model_cfg = TransformerConfig(**cfg_dict)
    model = NeuroScaleTransformer(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    pipeline_cfg: PipelineConfig,
    model_cfg: TransformerConfig,
    training_cfg: TrainingConfig,
    windows: Optional[Dict] = None,
    feat_normalizer: Optional[Normalizer] = None,
    tgt_normalizer: Optional[Normalizer] = None,
) -> Tuple[NeuroScaleTransformer, str]:
    """Train the Transformer and return the best model and checkpoint path.

    Data contract
    ~~~~~~~~~~~~~
    The ``windows`` dict (or the output of :func:`data.dataset.load_dataset`)
    must contain windows where **both X and y are normalised**.  If you supply
    ``windows`` directly you must also supply the corresponding
    ``feat_normalizer`` and ``tgt_normalizer`` so they can be embedded in the
    checkpoint.

    Parameters
    ----------
    pipeline_cfg : PipelineConfig
        Used to load the dataset if ``windows`` is not supplied.
    model_cfg : TransformerConfig
        Architecture configuration.
    training_cfg : TrainingConfig
        Hyper-parameter configuration.
    windows : dict, optional
        Pre-loaded ``{"train": [...], "val": [...], "test": [...]}`` dict
        where both X and y windows are in normalised space.
        If ``None``, the function calls :func:`data.dataset.load_dataset`.
    feat_normalizer : Normalizer, optional
        Feature normalizer used to produce ``windows``.  Required when
        ``windows`` is supplied.
    tgt_normalizer : Normalizer, optional
        Target normalizer used to produce ``windows``.  Required when
        ``windows`` is supplied.

    Returns
    -------
    model : NeuroScaleTransformer
        The best model (selected by lowest validation loss).
    checkpoint_path : str
        Path to the saved checkpoint file.
    """
    set_seed(training_cfg.seed)
    device = torch.device("cpu")

    # ---- Data ----
    if windows is None:
        windows = load_dataset(pipeline_cfg)
        # Load normalizers that were fitted and saved by load_dataset
        feat_normalizer = Normalizer()
        if os.path.isfile(pipeline_cfg.normalization_path):
            feat_normalizer.load(pipeline_cfg.normalization_path)
        tgt_normalizer = Normalizer()
        tgt_norm_path = _target_norm_path(pipeline_cfg.normalization_path)
        if os.path.isfile(tgt_norm_path):
            tgt_normalizer.load(tgt_norm_path)

    # Fall back to empty Normalizers if callers didn't provide them
    if feat_normalizer is None:
        feat_normalizer = Normalizer()
    if tgt_normalizer is None:
        tgt_normalizer = Normalizer()

    train_windows = windows["train"]
    val_windows = windows.get("val", [])

    if not train_windows:
        raise ValueError("Training dataset is empty; cannot train.")

    train_ds = WindowDataset(train_windows)
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    has_val = len(val_windows) > 0
    if has_val:
        val_ds = WindowDataset(val_windows)
        val_loader = DataLoader(val_ds, batch_size=training_cfg.batch_size, shuffle=False)
    else:
        val_loader = None

    # ---- Model ----
    model = NeuroScaleTransformer(model_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.lr)

    # ---- Training loop ----
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(training_cfg.checkpoint_dir, "best_transformer.pt")
    os.makedirs(training_cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(1, training_cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        if has_val:
            val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False)
        else:
            val_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path,
                model,
                model_cfg,
                pipeline_cfg,
                feat_normalizer,
                tgt_normalizer,
                epoch=epoch,
                val_loss=val_loss,
                train_loss=train_loss,
            )

        if epoch % max(1, training_cfg.epochs // 10) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{training_cfg.epochs}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"{'(best)' if val_loss == best_val_loss else ''}"
            )

    # Reload best model
    best_model, _ = load_checkpoint(checkpoint_path, device)
    print(f"\nBest val_loss={best_val_loss:.6f} (normalised target space) | checkpoint: {checkpoint_path}")
    return best_model, checkpoint_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NeuroScale Transformer")
    p.add_argument(
        "--raw-data",
        default="data/raw_metrics.jsonl",
        help="Path to the raw JSONL metrics file (default: data/raw_metrics.jsonl)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    pipeline_cfg = PipelineConfig(raw_data_path=args.raw_data)

    model_cfg = TransformerConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        prediction_horizon=pipeline_cfg.prediction_horizon,
        num_targets=len(pipeline_cfg.target_columns),
        feature_count=len(pipeline_cfg.feature_columns),
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    training_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    model, ckpt_path = train(pipeline_cfg, model_cfg, training_cfg)
    print(f"Training complete. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
