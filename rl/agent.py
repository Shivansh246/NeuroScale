"""DQN Agent for NeuroScale Resource Allocation."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .dqn import DQN, DQNConfig
from .replay_buffer import ReplayBuffer
from .actions import DiscreteActionSpace
from .config import ActionConfig

@dataclass
class DQNAgentConfig:
    """Hyperparameters for the DQN Agent."""
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    batch_size: int = 64
    replay_capacity: int = 50000
    target_update_freq: int = 100
    dqn_config: DQNConfig = field(default_factory=DQNConfig)
    action_config: ActionConfig = field(default_factory=ActionConfig)
    seed: int = 42

class DQNAgent:
    """Deep Q-Network Agent."""

    def __init__(self, config: DQNAgentConfig):
        self.config = config
        
        # Set seeds
        torch.manual_seed(config.seed)
        self.rng = np.random.RandomState(config.seed)
        
        # Action space mapping
        self.action_space = DiscreteActionSpace(config.action_config)
        self.config.dqn_config.action_count = self.action_space.n
        
        # Networks
        self.online_net = DQN(config.dqn_config)
        self.target_net = DQN(config.dqn_config)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer & Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.criterion = nn.SmoothL1Loss() # Huber loss
        
        # Replay Buffer
        self.memory = ReplayBuffer(
            capacity=config.replay_capacity,
            state_dim=config.dqn_config.state_dim,
            seed=config.seed
        )
        
        # Training state
        self.steps_done = 0
        self.epsilon = config.epsilon_start

    def choose_action_index(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an action index using epsilon-greedy policy."""
        if not evaluate and self.rng.rand() < self.epsilon:
            return self.rng.randint(self.action_space.n)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def choose_action(self, state: np.ndarray) -> Dict[str, float]:
        """
        Public integration contract: Choose an action deterministically for inference.
        Returns:
            Dict containing 'cpu' and 'memory' allocations.
        """
        action_idx = self.choose_action_index(state, evaluate=True)
        return self.action_space.get_action(action_idx)

    def train_step(self) -> Optional[float]:
        """Perform one gradient descent step on a mini-batch."""
        if len(self.memory) < self.config.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        state_batch = torch.FloatTensor(states)
        action_batch = torch.LongTensor(actions).unsqueeze(1)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_states)
        done_batch = torch.BoolTensor(dones).unsqueeze(1)
        
        # Q(s_t, a)
        q_values = self.online_net(state_batch).gather(1, action_batch)
        
        # max_a Q_target(s_{t+1}, a)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            # target = r + gamma * max Q (if not done)
            expected_q_values = reward_batch + (self.config.gamma * next_q_values * (~done_batch))
            
        loss = self.criterion(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Optional gradient clipping
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        # Epsilon decay
        decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
        self.epsilon = max(self.config.epsilon_end, self.epsilon - decay_rate)
        
        # Target network update
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        return loss.item()

    def save(self, filepath: str):
        """Save the agent checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
            # Just store raw config objects
            "config": self.config
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DQNAgent":
        """Load an agent from a checkpoint."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        agent = cls(checkpoint["config"])
        agent.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.steps_done = checkpoint["steps_done"]
        agent.epsilon = checkpoint["epsilon"]
        return agent
