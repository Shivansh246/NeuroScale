"""Training pipeline for the NeuroScale Autoencoder anomaly detector.

Normalization contract
----------------------
The trainer expects windows that are **already normalised** (produced by
:func:`data.dataset.load_dataset`).  Each window ``(X, y)`` contains a
feature array ``X`` of shape ``(seq_len, feature_count)``.  The autoencoder
operates solely on ``X``; ``y`` (the target) is not used.

The flattened input to the autoencoder is:

    x_flat = X.reshape(-1)  shape: (seq_len * feature_count,)

Threshold rule
--------------
After training, reconstruction errors on the **validation** split are
collected.  The anomaly threshold is:

    threshold = mean_val_error + k * std_val_error

where ``k = AutoencoderConfig.threshold_k`` (default 3.0).  This is a
standard-deviation fence — samples whose reconstruction error exceeds the
threshold are flagged as anomalies.

**Important**: this threshold is derived from *normal* training/validation
data.  It is not a calibrated probability.  Its sensitivity depends on how
representative the training data is of normal behaviour.

Checkpoint format
-----------------
::

    {
        "state_dict":    {...},
        "model_config":  {...},          # AutoencoderConfig as dict
        "feature_columns": [...],
        "seq_len":       int,
        "input_dim":     int,
        "feat_normalization": {          # feature normalizer stats
            "mean": [...], "std": [...]
        },
        "threshold":     float,          # mean_val_err + k * std_val_err
        "threshold_k":   float,
        "mean_val_error": float,
        "std_val_error":  float,
        "best_val_loss":  float,
        "seed":          int,
    }
"""

import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from anomaly.autoencoder import Autoencoder
from anomaly.config import AutoencoderConfig
from data.normalization import Normalizer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Window → flat tensor helpers
# ---------------------------------------------------------------------------

def windows_to_tensor(windows: List[Tuple[np.ndarray, np.ndarray]]) -> torch.Tensor:
    """Flatten and stack (X, y) window tuples into a single 2-D tensor.

    Only the feature array ``X`` is used; targets are ignored.

    Parameters
    ----------
    windows : list of (X, y) tuples
        X shape: ``(seq_len, feature_count)``

    Returns
    -------
    torch.Tensor
        Shape ``(N, seq_len * feature_count)`` — one flattened window per row.
    """
    flat = [x.reshape(-1) for x, _ in windows]
    return torch.tensor(np.stack(flat), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Threshold calculation
# ---------------------------------------------------------------------------

def compute_threshold(
    errors: np.ndarray,
    k: float,
) -> Tuple[float, float, float]:
    """Compute the anomaly threshold from a set of reconstruction errors.

    Rule: ``threshold = mean + k * std``

    Parameters
    ----------
    errors : np.ndarray
        1-D array of per-sample reconstruction errors on normal data.
    k : float
        Standard-deviation multiplier.

    Returns
    -------
    threshold : float
    mean_error : float
    std_error : float
    """
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors, ddof=0))
    threshold = mean_err + k * std_err
    return threshold, mean_err, std_err


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: Autoencoder,
    cfg: AutoencoderConfig,
    feat_normalizer: Normalizer,
    threshold: float,
    mean_val_error: float,
    std_val_error: float,
    best_val_loss: float,
) -> None:
    """Save model and all metadata needed for deterministic inference.

    Parameters
    ----------
    path : str
        Destination file path (must end with ``.pt`` for convention).
    model : Autoencoder
        Trained model.
    cfg : AutoencoderConfig
        Model and training configuration.
    feat_normalizer : Normalizer
        Feature normalizer fitted on training data.
    threshold : float
        Anomaly threshold.
    mean_val_error, std_val_error : float
        Statistics used to derive the threshold.
    best_val_loss : float
        Best epoch validation loss (MSE, normalised space).
    """
    feat_norm_data: Dict = {}
    if feat_normalizer.mean_ is not None:
        feat_norm_data = {
            "mean": feat_normalizer.mean_.tolist(),
            "std": feat_normalizer.std_.tolist(),
        }

    cfg_dict = asdict(cfg)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": cfg_dict,
        "feature_columns": list(cfg.feature_columns),
        "seq_len": cfg.seq_len,
        "input_dim": cfg.input_dim,
        "feat_normalization": feat_norm_data,
        "threshold": threshold,
        "threshold_k": cfg.threshold_k,
        "mean_val_error": mean_val_error,
        "std_val_error": std_val_error,
        "best_val_loss": best_val_loss,
        "seed": cfg.seed,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Autoencoder, dict]:
    """Load checkpoint; return (model, checkpoint_dict).

    Parameters
    ----------
    path : str
        Path to ``.pt`` checkpoint.
    device : torch.device

    Returns
    -------
    tuple[Autoencoder, dict]
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = checkpoint["model_config"]
    cfg = AutoencoderConfig(**cfg_dict)
    model = Autoencoder(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class AETrainingConfig:
    """Hyper-parameters for autoencoder training."""

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir


def _run_epoch(
    model: Autoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_hat = model(x_batch)
            loss = criterion(x_hat, x_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_autoencoder(
    ae_cfg: AutoencoderConfig,
    training_cfg: AETrainingConfig,
    train_windows: List[Tuple[np.ndarray, np.ndarray]],
    val_windows: List[Tuple[np.ndarray, np.ndarray]],
    feat_normalizer: Normalizer,
) -> Tuple[Autoencoder, str]:
    """Train the autoencoder and return (best_model, checkpoint_path).

    The model trains to minimise MSE reconstruction loss on normalised
    flattened windows.  After training, the anomaly threshold is computed
    from validation reconstruction errors and embedded in the checkpoint.

    Parameters
    ----------
    ae_cfg : AutoencoderConfig
        Model architecture and anomaly threshold configuration.
    training_cfg : AETrainingConfig
        Epoch count, batch size, learning rate.
    train_windows : list of (X, y)
        Normalised training windows; only X is used.
    val_windows : list of (X, y)
        Normalised validation windows; used for threshold computation.
    feat_normalizer : Normalizer
        Fitted feature normalizer (embedded in checkpoint for inference).

    Returns
    -------
    model : Autoencoder
        Best model (lowest val loss).
    checkpoint_path : str
    """
    set_seed(training_cfg.seed)
    device = torch.device("cpu")

    if not train_windows:
        raise ValueError("train_windows is empty; cannot train autoencoder.")

    # ---- Datasets ----
    X_train = windows_to_tensor(train_windows).to(device)
    train_ds = TensorDataset(X_train)
    train_loader = DataLoader(
        train_ds, batch_size=training_cfg.batch_size, shuffle=True, drop_last=False
    )

    has_val = len(val_windows) > 0
    if has_val:
        X_val = windows_to_tensor(val_windows).to(device)
        val_ds = TensorDataset(X_val)
        val_loader = DataLoader(val_ds, batch_size=training_cfg.batch_size, shuffle=False)
    else:
        X_val = X_train
        val_loader = DataLoader(TensorDataset(X_train), batch_size=training_cfg.batch_size)

    # ---- Model ----
    model = Autoencoder(ae_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.lr)

    # ---- Training loop ----
    best_val_loss = float("inf")
    ckpt_path = os.path.join(training_cfg.checkpoint_dir, "best_autoencoder.pt")
    os.makedirs(training_cfg.checkpoint_dir, exist_ok=True)

    # Placeholder saves will be replaced; we need at least one save
    _threshold, _mean_err, _std_err = 0.0, 0.0, 0.0

    for epoch in range(1, training_cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Compute threshold from current val errors
            model.eval()
            with torch.no_grad():
                errors = model.reconstruction_error(X_val).cpu().numpy()
            _threshold, _mean_err, _std_err = compute_threshold(errors, ae_cfg.threshold_k)
            save_checkpoint(
                ckpt_path, model, ae_cfg, feat_normalizer,
                _threshold, _mean_err, _std_err, best_val_loss,
            )

        if epoch % max(1, training_cfg.epochs // 10) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{training_cfg.epochs}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"{'(best)' if val_loss == best_val_loss else ''}"
            )

    best_model, _ = load_checkpoint(ckpt_path, device)
    print(
        f"\nBest val_loss={best_val_loss:.6f} | "
        f"threshold={_threshold:.6f} (mean={_mean_err:.4f}, "
        f"std={_std_err:.4f}, k={ae_cfg.threshold_k}) | "
        f"checkpoint: {ckpt_path}"
    )
    return best_model, ckpt_path
