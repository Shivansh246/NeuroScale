"""Lightweight fully-connected autoencoder for NeuroScale anomaly detection.

Architecture
------------
The autoencoder operates on **flattened** normalised metric windows:

    input: (batch, input_dim)  where input_dim = seq_len × feature_count

Encoder:
    input_dim → hidden_dims[0] → hidden_dims[1] → ... → latent_dim

Decoder (mirrors encoder):
    latent_dim → hidden_dims[-1] → ... → hidden_dims[0] → input_dim

Each hidden layer uses ReLU activation.  The final decoder layer has no
activation so that the output can represent arbitrary real values in the
normalised feature space.

Dropout (if configured) is applied after every hidden ReLU in both encoder
and decoder.

Design rationale
----------------
A fully-connected autoencoder is the simplest architecture that can capture
cross-feature and cross-timestep correlations without the complexity of a
convolutional or recurrent design.  For lightweight anomaly detection on a
small number of metric channels this is the right trade-off.
"""

from typing import List

import torch
import torch.nn as nn

from anomaly.config import AutoencoderConfig


def _make_layers(
    dims: List[int],
    add_activation: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a sequential stack of Linear → ReLU (→ Dropout) blocks.

    Parameters
    ----------
    dims : list[int]
        Layer widths [in, h1, h2, ..., out].
    add_activation : bool
        If True, add ReLU after every layer except the last.
    dropout : float
        Dropout probability; 0.0 disables dropout.
    """
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if add_activation and not is_last:
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Fully-connected autoencoder for metric-window reconstruction.

    Parameters
    ----------
    cfg : AutoencoderConfig
        Full model configuration.
    """

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder: input_dim → hidden_dims → latent_dim
        enc_dims = [cfg.input_dim] + list(cfg.hidden_dims) + [cfg.latent_dim]
        self.encoder = _make_layers(enc_dims, add_activation=True, dropout=cfg.dropout)

        # Decoder: latent_dim → reversed(hidden_dims) → input_dim
        dec_dims = [cfg.latent_dim] + list(reversed(cfg.hidden_dims)) + [cfg.input_dim]
        self.decoder = _make_layers(dec_dims, add_activation=True, dropout=cfg.dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of flattened windows to latent space.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent batch back to input space.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error (no gradient).

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` — one scalar error per sample.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            # Mean over the input_dim axis → one value per sample
            errors = ((x - x_hat) ** 2).mean(dim=1)
        return errors
