"""Factorized Deep Q-Network Architecture for NeuroScale Resource Allocation."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ResourceFactorizedDQNConfig:
    """Configuration for Resource-Factorized DQN."""
    state_dim: int = 11
    cpu_action_count: int = 4
    mem_action_count: int = 4
    hidden_dim: int = 64


class ResourceFactorizedDQN(nn.Module):
    """
    Factorized network with shared state representation trunk and two specialized Q heads:
    Input dimension: 11 (state vector)
    Shared trunk: Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim) -> ReLU
    Two heads:
        - Q_cpu: Linear(hidden_dim, 4)   # CPU actions [0.25, 0.5, 1.0, 2.0]
        - Q_mem: Linear(hidden_dim, 4)   # Mem actions [128, 256, 512, 1024]
    """

    def __init__(
        self,
        config: Optional[ResourceFactorizedDQNConfig] = None,
        state_dim: int = 11,
        hidden_dim: int = 64,
        cpu_action_count: int = 4,
        mem_action_count: int = 4,
    ):
        super(ResourceFactorizedDQN, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ResourceFactorizedDQNConfig(
                state_dim=state_dim,
                cpu_action_count=cpu_action_count,
                mem_action_count=mem_action_count,
                hidden_dim=hidden_dim,
            )

        self.trunk = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )
        self.q_cpu = nn.Linear(self.config.hidden_dim, self.config.cpu_action_count)
        self.q_mem = nn.Linear(self.config.hidden_dim, self.config.mem_action_count)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute factorized Q-values.
        Args:
            x: Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            Tuple of:
            - q_cpu: (batch_size, 4) or (4,)
            - q_mem: (batch_size, 4) or (4,)
        """
        features = self.trunk(x)
        return self.q_cpu(features), self.q_mem(features)
