"""Resource-Factorized DQN Agent with Independent CPU and Memory TD Learning."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

from .resource_factorized_dqn import ResourceFactorizedDQN, ResourceFactorizedDQNConfig
from .actions import DiscreteActionSpace
from .config import ActionConfig


class ResourceFactorizedReplayBuffer:
    """Fixed-capacity replay buffer storing decomposed resource transitions."""

    def __init__(self, capacity: int, state_dim: int, seed: Optional[int] = None):
        self.capacity = capacity
        self.state_dim = state_dim

        # Separate pre-allocated arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.cpu_actions = np.zeros(capacity, dtype=np.int64)
        self.mem_actions = np.zeros(capacity, dtype=np.int64)
        self.r_cpus = np.zeros(capacity, dtype=np.float32)
        self.r_mems = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.RandomState(seed)

    def push(
        self,
        state: np.ndarray,
        cpu_action: int,
        mem_action: int,
        r_cpu: float,
        r_mem: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store decomposed transition with separate r_cpu and r_mem."""
        self.states[self.ptr] = state
        self.cpu_actions[self.ptr] = cpu_action
        self.mem_actions[self.ptr] = mem_action
        self.r_cpus[self.ptr] = r_cpu
        self.r_mems[self.ptr] = r_mem
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample mini-batch returning separate CPU and Memory components."""
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer.")

        indices = self.rng.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.cpu_actions[indices],
            self.mem_actions[indices],
            self.r_cpus[indices],
            self.r_mems[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class ResourceFactorizedAgentConfig:
    """Hyperparameters for Resource-Factorized DQN Agent."""
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 4000
    batch_size: int = 64
    replay_capacity: int = 50000
    target_update_freq: int = 100
    dqn_config: ResourceFactorizedDQNConfig = field(default_factory=ResourceFactorizedDQNConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    seed: int = 42


class ResourceFactorizedDQNAgent:
    """
    Resource-Factorized Deep Q-Network Agent.
    Maintains a shared trunk and two separate Q heads (CPU and Memory).
    Learns via separate TD targets:
        y_cpu = r_cpu + gamma * max_a' Q_cpu_target(s', a')
        y_mem = r_mem + gamma * max_a' Q_mem_target(s', a')
    Optimizes combined Huber loss:
        loss = loss_cpu + loss_mem
    """

    def __init__(self, config: Optional[ResourceFactorizedAgentConfig] = None):
        self.config = config or ResourceFactorizedAgentConfig()

        torch.manual_seed(self.config.seed)
        self.rng = np.random.RandomState(self.config.seed)

        self.action_space = DiscreteActionSpace(self.config.action_config)
        self.cpu_options: List[float] = list(self.config.action_config.cpu_options)
        self.memory_options: List[int] = list(self.config.action_config.memory_options)
        self.num_cpu_actions = len(self.cpu_options)
        self.num_mem_actions = len(self.memory_options)

        self.config.dqn_config.cpu_action_count = self.num_cpu_actions
        self.config.dqn_config.mem_action_count = self.num_mem_actions

        # Networks
        self.online_net = ResourceFactorizedDQN(self.config.dqn_config)
        self.target_net = ResourceFactorizedDQN(self.config.dqn_config)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer & Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Factorized Replay Buffer storing r_cpu and r_mem separately
        self.memory = ResourceFactorizedReplayBuffer(
            capacity=self.config.replay_capacity,
            state_dim=self.config.dqn_config.state_dim,
            seed=self.config.seed,
        )

        self.steps_done = 0
        self.epsilon = self.config.epsilon_start

    def choose_branch_indices(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, int]:
        """Select branch action indices (cpu_idx, mem_idx) using epsilon-greedy policy."""
        if not evaluate and self.rng.rand() < self.epsilon:
            cpu_idx = int(self.rng.randint(self.num_cpu_actions))
            mem_idx = int(self.rng.randint(self.num_mem_actions))
            return cpu_idx, mem_idx

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_cpu, q_mem = self.online_net(state_tensor)
            cpu_idx = int(torch.argmax(q_cpu, dim=1).item())
            mem_idx = int(torch.argmax(q_mem, dim=1).item())
            return cpu_idx, mem_idx

    def choose_action_index(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Maps branch decisions to joint discrete action index:
        joint_index = cpu_idx * 4 + mem_idx
        """
        cpu_idx, mem_idx = self.choose_branch_indices(state, evaluate=evaluate)
        return cpu_idx * self.num_mem_actions + mem_idx

    def choose_action(self, state: np.ndarray) -> Dict[str, float]:
        """
        Public integration contract: Choose action deterministically for inference.
        Returns:
            Dict containing 'cpu' and 'memory' allocations.
        """
        cpu_idx, mem_idx = self.choose_branch_indices(state, evaluate=True)
        return {
            "cpu": float(self.cpu_options[cpu_idx]),
            "memory": int(self.memory_options[mem_idx]),
        }

    def train_step(self) -> Optional[float]:
        """
        Perform one gradient descent step on a mini-batch using Factorized TD Learning.
        y_cpu = r_cpu + gamma * max Q_cpu_target(s')
        y_mem = r_mem + gamma * max Q_mem_target(s')
        loss = loss_cpu + loss_mem
        """
        if len(self.memory) < self.config.batch_size:
            return None

        (
            states,
            cpu_actions,
            mem_actions,
            r_cpus,
            r_mems,
            next_states,
            dones,
        ) = self.memory.sample(self.config.batch_size)

        state_batch = torch.FloatTensor(states)
        cpu_action_batch = torch.LongTensor(cpu_actions).unsqueeze(1)
        mem_action_batch = torch.LongTensor(mem_actions).unsqueeze(1)
        r_cpu_batch = torch.FloatTensor(r_cpus).unsqueeze(1)
        r_mem_batch = torch.FloatTensor(r_mems).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_states)
        done_batch = torch.BoolTensor(dones).unsqueeze(1)

        # Current Q-values from online network
        q_cpu, q_mem = self.online_net(state_batch)
        q_cpu_val = q_cpu.gather(1, cpu_action_batch)
        q_mem_val = q_mem.gather(1, mem_action_batch)

        # Factorized TD targets
        with torch.no_grad():
            next_q_cpu, next_q_mem = self.target_net(next_state_batch)
            max_next_cpu = next_q_cpu.max(1)[0].unsqueeze(1)
            max_next_mem = next_q_mem.max(1)[0].unsqueeze(1)

            expected_q_cpu = r_cpu_batch + (self.config.gamma * max_next_cpu * (~done_batch))
            expected_q_mem = r_mem_batch + (self.config.gamma * max_next_mem * (~done_batch))

        # Separate Huber losses
        loss_cpu = self.criterion(q_cpu_val, expected_q_cpu)
        loss_mem = self.criterion(q_mem_val, expected_q_mem)
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
        """Save the factorized agent checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ResourceFactorizedDQNAgent":
        """Load a factorized agent from a checkpoint."""
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"), weights_only=False)
        agent = cls(checkpoint["config"])
        agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = checkpoint["steps_done"]
        agent.epsilon = checkpoint["epsilon"]
        return agent
