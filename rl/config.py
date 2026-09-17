"""Configuration for the NeuroScale DRL Environment."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ActionConfig:
    """Discrete action space configurations."""
    cpu_options: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    memory_options: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])


@dataclass
class RewardConfig:
    """Weights and parameters for the reward function."""
    sla_penalty_weight: float = 10.0
    sla_latency_threshold_ms: float = 200.0
    resource_waste_weight: float = 1.0
    reallocation_penalty: float = 0.5
    under_provision_penalty: float = 5.0
    anomaly_penalty: float = 2.0


@dataclass
class StateConfig:
    """State space configuration and normalization bounds."""
    # Used for clipping/normalizing state components
    max_cpu_util: float = 400.0   # %
    max_mem_util: float = 2048.0  # MB
    max_cpu_alloc: float = 4.0    # cores
    max_mem_alloc: float = 4096.0 # MB
    max_latency: float = 1000.0   # ms
    max_anomaly_score: float = 10.0

@dataclass
class EnvConfig:
    """Overall environment configuration."""
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    state: StateConfig = field(default_factory=StateConfig)
    max_steps: int = 100

