"""Deep Q-Network Architecture for NeuroScale."""

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class DQNConfig:
    """Configuration for the DQN Architecture."""
    state_dim: int = 11
    action_count: int = 16
    hidden_dim: int = 64


class DQN(nn.Module):
    """
    Lightweight Multi-Layer Perceptron for Deep Q-Learning.
    Maps a state vector to Q-values for each discrete action.
    """
    def __init__(self, config: DQNConfig):
        super(DQN, self).__init__()
        self.config = config
        
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_count)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values.
        Args:
            x: Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            Q-values of shape (batch_size, action_count) or (action_count,)
        """
        return self.network(x)
