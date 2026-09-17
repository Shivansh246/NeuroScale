"""Action space definition for the NeuroScale DRL Environment."""

from typing import Dict, List, Tuple
from .config import ActionConfig


class DiscreteActionSpace:
    """Discrete Cartesian-product action space for CPU and memory allocation."""

    def __init__(self, config: ActionConfig):
        self.config = config
        self.cpu_options = sorted(self.config.cpu_options)
        self.memory_options = sorted(self.config.memory_options)
        
        # Build Cartesian product mapping
        self.index_to_action: List[Tuple[float, int]] = []
        for c in self.cpu_options:
            for m in self.memory_options:
                self.index_to_action.append((c, m))
                
        self.n = len(self.index_to_action)

    def get_action(self, action_index: int) -> Dict[str, float]:
        """Convert a discrete action index to a CPU/Memory resource dictionary."""
        if not (0 <= action_index < self.n):
            raise ValueError(f"Action index {action_index} out of bounds (0 to {self.n - 1})")
        
        cpu, mem = self.index_to_action[action_index]
        return {"cpu": cpu, "memory": mem}
    
    def get_closest_index(self, cpu: float, memory: int) -> int:
        """Find the index of the closest action to the given cpu/memory."""
        best_idx = 0
        min_dist = float('inf')
        for i, (c, m) in enumerate(self.index_to_action):
            # normalized L2 or L1 distance. Using simple L1 distance normalized roughly
            dist = abs(c - cpu) / max(self.cpu_options) + abs(m - memory) / max(self.memory_options)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx
