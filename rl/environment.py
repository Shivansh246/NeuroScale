"""Lightweight DRL Environment for NeuroScale."""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from .config import EnvConfig
from .actions import DiscreteActionSpace
from .state import StateBuilder, STATE_DIM
from .reward import RewardCalculator


class NeuroScaleEnv:
    """
    Lightweight, Gymnasium-compatible environment for Docker resource allocation.
    Operates on a deterministic trace of workload metrics and predictions.
    """

    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self.action_space = DiscreteActionSpace(self.config.action)
        self.state_builder = StateBuilder(self.config.state)
        self.reward_calculator = RewardCalculator(self.config.reward)
        
        self.trace: List[Dict[str, Any]] = []
        self.current_step = 0
        self.max_steps = self.config.max_steps
        
        self.current_raw_state: Dict[str, Any] = {}
        
    def load_trace(self, trace: List[Dict[str, Any]]) -> None:
        """Load a deterministic simulation trace."""
        self.trace = trace
        self.max_steps = min(len(trace), self.config.max_steps)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the beginning of the trace."""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        if not self.trace:
            raise ValueError("No trace loaded. Call load_trace() before reset().")
            
        # Initial state from trace
        trace_step = self.trace[self.current_step]
        
        self.current_raw_state = {
            "current_cpu_util": trace_step.get("cpu_util", 0.0),
            "current_mem_util": trace_step.get("mem_util", 0.0),
            "predicted_cpu_demand": trace_step.get("pred_cpu", 0.0),
            "predicted_mem_demand": trace_step.get("pred_mem", 0.0),
            "current_cpu_alloc": 1.0,  # Default starting
            "current_mem_alloc": 256,  # Default starting
            "sla_latency": 50.0,       # Baseline
            "anomaly_score": trace_step.get("anomaly_score", 0.0),
            "is_anomaly": trace_step.get("is_anomaly", False),
            "host_cpu_avail": 4.0,
            "host_mem_avail": 4096.0
        }
        
        state_vec = self.state_builder.build_state(self.current_raw_state)
        return state_vec, self.current_raw_state.copy()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action, returning (next_state, reward, terminated, truncated, info).
        """
        if self.current_step >= self.max_steps - 1:
            # Reached end of trace
            state_vec = self.state_builder.build_state(self.current_raw_state)
            return state_vec, 0.0, True, False, {"msg": "End of trace"}
            
        # 1. Map action index to allocation
        action_dict = self.action_space.get_action(action_index)
        cpu_alloc = action_dict["cpu"]
        mem_alloc = action_dict["memory"]
        
        # 2. Get next step's "target" demand from trace
        self.current_step += 1
        trace_step = self.trace[self.current_step]
        
        target_cpu_util = trace_step.get("cpu_util", 0.0) # in %
        target_mem_util = trace_step.get("mem_util", 0.0) # in MB
        
        # 3. Simulate SLA and actual utilization based on allocation limits
        actual_cpu_util = min(target_cpu_util, cpu_alloc * 100.0)
        actual_mem_util = min(target_mem_util, mem_alloc)
        
        # Simple SLA latency simulation
        base_latency = 50.0
        latency = base_latency
        
        # If CPU throttled, latency spikes proportionally
        if target_cpu_util > cpu_alloc * 100.0:
            cpu_deficit = target_cpu_util - (cpu_alloc * 100.0)
            latency += cpu_deficit * 5.0 # 5ms per 1% cpu deficit
            
        # If Memory exceeded, massive latency spike (OOM/Swap simulation)
        if target_mem_util > mem_alloc:
            mem_deficit = target_mem_util - mem_alloc
            latency += mem_deficit * 20.0 # 20ms per MB deficit
            
        # 4. Construct next raw state
        next_raw_state = {
            "current_cpu_util": actual_cpu_util,
            "current_mem_util": actual_mem_util,
            "predicted_cpu_demand": trace_step.get("pred_cpu", 0.0),
            "predicted_mem_demand": trace_step.get("pred_mem", 0.0),
            "current_cpu_alloc": cpu_alloc,
            "current_mem_alloc": mem_alloc,
            "sla_latency": latency,
            "anomaly_score": trace_step.get("anomaly_score", 0.0),
            "is_anomaly": trace_step.get("is_anomaly", False),
            "host_cpu_avail": 4.0,
            "host_mem_avail": 4096.0
        }
        
        # 5. Calculate reward
        reward = self.reward_calculator.calculate(
            prev_state=self.current_raw_state,
            action=action_dict,
            next_state=next_raw_state
        )
        
        self.current_raw_state = next_raw_state
        state_vec = self.state_builder.build_state(self.current_raw_state)
        
        terminated = (self.current_step >= self.max_steps - 1)
        truncated = False
        info = {
            "raw_state": next_raw_state.copy(),
            "action": action_dict.copy()
        }
        
        return state_vec, reward, terminated, truncated, info
