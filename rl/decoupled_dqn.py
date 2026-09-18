"""Decoupled (Action Branching) Deep Q-Network Architecture for NeuroScale."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DecoupledDQNConfig:
    """Configuration for the Decoupled DQN Architecture."""
    state_dim: int = 11
    cpu_action_count: int = 4
    mem_action_count: int = 4
    hidden_dim: int = 64


class DecoupledDQN(nn.Module):
    """
    Action Branching Multi-Layer Perceptron.
    Shares a common state representation trunk, then splits into two specialized Q heads:
    - cpu_head: outputs Q-values for CPU actions [0.25, 0.5, 1.0, 2.0]
    - mem_head: outputs Q-values for Memory actions [128, 256, 512, 1024]
    """

    def __init__(self, config: DecoupledDQNConfig):
        super(DecoupledDQN, self).__init__()
        self.config = config

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Specialized branch heads
        self.cpu_head = nn.Linear(config.hidden_dim, config.cpu_action_count)
        self.mem_head = nn.Linear(config.hidden_dim, config.mem_action_count)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute branch-specific Q-values.
        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            Tuple of:
            - q_cpu: Q-values of shape (batch_size, cpu_action_count)
            - q_mem: Q-values of shape (batch_size, mem_action_count)
        """
        features = self.trunk(x)
        q_cpu = self.cpu_head(features)
        q_mem = self.mem_head(features)
        return q_cpu, q_mem
