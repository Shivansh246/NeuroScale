"""Lightweight Transformer Encoder for CPU/memory demand prediction.

Architecture
------------
input features  (batch, seq_len, feature_count)
→ Linear projection          → (batch, seq_len, d_model)
→ Positional encoding        → (batch, seq_len, d_model)
→ Transformer Encoder stack  → (batch, seq_len, d_model)
→ Take last ``prediction_horizon`` time-steps
→ Linear prediction head     → (batch, prediction_horizon, num_targets)

The model is sequence-length agnostic: ``seq_len`` is read from the
input tensor at each forward pass, so no architectural change is needed
when ``PipelineConfig.input_window`` changes.
"""

import math

import torch
import torch.nn as nn

from models.config import TransformerConfig


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float
        Dropout probability applied after adding the positional encoding.
    max_len : int
        Maximum sequence length supported.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encoding table (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class NeuroScaleTransformer(nn.Module):
    """Transformer Encoder model for demand prediction.

    Parameters
    ----------
    cfg : TransformerConfig
        Full model configuration.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- Input projection ---
        self.input_projection = nn.Linear(cfg.feature_count, cfg.d_model)

        # --- Positional encoding ---
        self.positional_encoding = PositionalEncoding(cfg.d_model, cfg.dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)

        # --- Prediction head ---
        # Takes the last ``prediction_horizon`` encoder outputs and maps them
        # to ``num_targets`` predictions each.
        self.prediction_head = nn.Linear(cfg.d_model, cfg.num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, feature_count)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(batch, prediction_horizon, num_targets)``.
        """
        # (batch, seq_len, feature_count) → (batch, seq_len, d_model)
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # (batch, seq_len, d_model)
        encoded = self.encoder(x)

        # Use the last ``prediction_horizon`` time-steps as predictions.
        # For prediction_horizon=1 this is just the final step.
        ph = self.cfg.prediction_horizon
        encoded_tail = encoded[:, -ph:, :]          # (batch, ph, d_model)
        output = self.prediction_head(encoded_tail)  # (batch, ph, num_targets)
        return output
