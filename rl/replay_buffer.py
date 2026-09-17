"""Replay Buffer for DQN."""

import numpy as np
from typing import Tuple, List

class ReplayBuffer:
    """Fixed-capacity replay buffer for storing MDP transitions."""

    def __init__(self, capacity: int, state_dim: int, seed: int = None):
        self.capacity = capacity
        self.state_dim = state_dim
        
        # Preallocate numpy arrays for fast insertion and sampling
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        
        self.ptr = 0
        self.size = 0
        
        self.rng = np.random.RandomState(seed)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store a new transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch of transitions.
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer.")
            
        indices = self.rng.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        return self.size
