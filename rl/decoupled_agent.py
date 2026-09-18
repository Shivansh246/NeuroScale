"""Decoupled (Action Branching) DQN Agent for NeuroScale Resource Allocation."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

from .decoupled_dqn import DecoupledDQN, DecoupledDQNConfig
from .replay_buffer import ReplayBuffer
from .actions import DiscreteActionSpace
from .config import ActionConfig


@dataclass
class DecoupledDQNAgentConfig:
    """Hyperparameters for the Decoupled DQN Agent."""
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 4000
    batch_size: int = 64
    replay_capacity: int = 50000
    target_update_freq: int = 100
    decoupled_dqn_config: DecoupledDQNConfig = field(default_factory=DecoupledDQNConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    seed: int = 42


class DecoupledDQNAgent:
    """
    Action Branching Deep Q-Network Agent.
    Decomposes the discrete resource allocation into independent CPU and Memory branches,
    mitigating the exponential growth and coupling of the joint action space.
    """

    def __init__(self, config: DecoupledDQNAgentConfig):
        self.config = config

        # Set seeds
        torch.manual_seed(config.seed)
        self.rng = np.random.RandomState(config.seed)

        # Action space mapping
        self.action_space = DiscreteActionSpace(config.action_config)
        self.cpu_options: List[float] = list(config.action_config.cpu_options)
        self.memory_options: List[int] = list(config.action_config.memory_options)
        self.num_cpu_actions = len(self.cpu_options)
        self.num_mem_actions = len(self.memory_options)

        self.config.decoupled_dqn_config.cpu_action_count = self.num_cpu_actions
        self.config.decoupled_dqn_config.mem_action_count = self.num_mem_actions

        # Networks
        self.online_net = DecoupledDQN(config.decoupled_dqn_config)
        self.target_net = DecoupledDQN(config.decoupled_dqn_config)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer & Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay Buffer
        self.memory = ReplayBuffer(
            capacity=config.replay_capacity,
            state_dim=config.decoupled_dqn_config.state_dim,
            seed=config.seed
        )

        # Training state
        self.steps_done = 0
        self.epsilon = config.epsilon_start

    def choose_branch_indices(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, int]:
        """
        Select branch action indices (cpu_idx, mem_idx) using epsilon-greedy policy.
        """
        if not evaluate and self.rng.rand() < self.epsilon:
            cpu_idx = self.rng.randint(self.num_cpu_actions)
            mem_idx = self.rng.randint(self.num_mem_actions)
            return cpu_idx, mem_idx

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_cpu, q_mem = self.online_net(state_tensor)
            cpu_idx = int(torch.argmax(q_cpu, dim=1).item())
            mem_idx = int(torch.argmax(q_mem, dim=1).item())
            return cpu_idx, mem_idx

    def choose_action_index(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Maps branch decisions to joint discrete action index for environment compatibility:
        joint_index = cpu_idx * num_mem_actions + mem_idx
        """
        cpu_idx, mem_idx = self.choose_branch_indices(state, evaluate=evaluate)
        return cpu_idx * self.num_mem_actions + mem_idx

    def choose_action(self, state: np.ndarray) -> Dict[str, float]:
        """
        Public integration contract: Choose an action deterministically for inference.
        Returns:
            Dict containing 'cpu' and 'memory' allocations.
        """
        cpu_idx, mem_idx = self.choose_branch_indices(state, evaluate=True)
        return {
            "cpu": float(self.cpu_options[cpu_idx]),
            "memory": int(self.memory_options[mem_idx])
        }

    def train_step(self) -> Optional[float]:
        """
        Perform one gradient descent step on a mini-batch using Action Branching Q-learning.
        """
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        state_batch = torch.FloatTensor(states)
        action_batch = torch.LongTensor(actions)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_states)
        done_batch = torch.BoolTensor(dones).unsqueeze(1)

        # Decompose joint action indices into branch indices
        cpu_actions = (action_batch // self.num_mem_actions).unsqueeze(1)
        mem_actions = (action_batch % self.num_mem_actions).unsqueeze(1)

        # Current Q-values from online network
        q_cpu, q_mem = self.online_net(state_batch)
        q_cpu_val = q_cpu.gather(1, cpu_actions)
        q_mem_val = q_mem.gather(1, mem_actions)

        # Action Branching target calculation:
        # y = r + gamma * 0.5 * (max_a' Q_cpu_target(s', a') + max_a' Q_mem_target(s', a'))
        with torch.no_grad():
            next_q_cpu, next_q_mem = self.target_net(next_state_batch)
            max_next_cpu = next_q_cpu.max(1)[0].unsqueeze(1)
            max_next_mem = next_q_mem.max(1)[0].unsqueeze(1)
            mean_max_next = 0.5 * (max_next_cpu + max_next_mem)
            expected_q = reward_batch + (self.config.gamma * mean_max_next * (~done_batch))

        loss_cpu = self.criterion(q_cpu_val, expected_q)
        loss_mem = self.criterion(q_mem_val, expected_q)
        total_loss = loss_cpu + loss_mem

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        # Epsilon decay
        decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
        self.epsilon = max(self.config.epsilon_end, self.epsilon - decay_rate)

        # Target network update
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return total_loss.item()

    def save(self, filepath: str):
        """Save the decoupled agent checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
            "config": self.config
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DecoupledDQNAgent":
        """Load a decoupled agent from a checkpoint."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        agent = cls(checkpoint["config"])
        agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = checkpoint["steps_done"]
        agent.epsilon = checkpoint["epsilon"]
        return agent
