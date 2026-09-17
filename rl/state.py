"""State representation for the NeuroScale DRL Environment."""

import numpy as np
from typing import Dict, Any, List
from .config import StateConfig

# Define the state indices explicitly for deterministic dimensionality
STATE_INDICES = {
    "current_cpu_util": 0,
    "current_mem_util": 1,
    "predicted_cpu_demand": 2,
    "predicted_mem_demand": 3,
    "current_cpu_alloc": 4,
    "current_mem_alloc": 5,
    "sla_latency": 6,
    "anomaly_score": 7,
    "is_anomaly": 8,
    "host_cpu_avail": 9,
    "host_mem_avail": 10,
}

STATE_DIM = len(STATE_INDICES)

class StateBuilder:
    """Constructs and normalizes the state vector for the RL agent."""

    def __init__(self, config: StateConfig):
        self.config = config

    def build_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Builds a normalized 1D numpy array representing the state.
        Missing keys default to 0.0.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        
        # 0. Current CPU (normalize by max_cpu_util)
        cpu = raw_state.get("current_cpu_util", 0.0)
        state[STATE_INDICES["current_cpu_util"]] = np.clip(cpu / self.config.max_cpu_util, 0.0, 1.0)
        
        # 1. Current Mem (normalize by max_mem_util)
        mem = raw_state.get("current_mem_util", 0.0)
        state[STATE_INDICES["current_mem_util"]] = np.clip(mem / self.config.max_mem_util, 0.0, 1.0)
        
        # 2. Predicted CPU
        pred_cpu = raw_state.get("predicted_cpu_demand", 0.0)
        state[STATE_INDICES["predicted_cpu_demand"]] = np.clip(pred_cpu / self.config.max_cpu_util, 0.0, 1.0)
        
        # 3. Predicted Mem
        pred_mem = raw_state.get("predicted_mem_demand", 0.0)
        state[STATE_INDICES["predicted_mem_demand"]] = np.clip(pred_mem / self.config.max_mem_util, 0.0, 1.0)
        
        # 4. Current CPU allocation
        cpu_alloc = raw_state.get("current_cpu_alloc", 1.0)
        state[STATE_INDICES["current_cpu_alloc"]] = np.clip(cpu_alloc / self.config.max_cpu_alloc, 0.0, 1.0)
        
        # 5. Current Mem allocation
        mem_alloc = raw_state.get("current_mem_alloc", 256.0)
        state[STATE_INDICES["current_mem_alloc"]] = np.clip(mem_alloc / self.config.max_mem_alloc, 0.0, 1.0)
        
        # 6. SLA latency
        lat = raw_state.get("sla_latency", 0.0)
        state[STATE_INDICES["sla_latency"]] = np.clip(lat / self.config.max_latency, 0.0, 1.0)
        
        # 7. Anomaly Score
        score = raw_state.get("anomaly_score", 0.0)
        state[STATE_INDICES["anomaly_score"]] = np.clip(score / self.config.max_anomaly_score, 0.0, 1.0)
        
        # 8. Is Anomaly (Boolean to float)
        is_anomaly = raw_state.get("is_anomaly", False)
        state[STATE_INDICES["is_anomaly"]] = 1.0 if is_anomaly else 0.0
        
        # 9. Host CPU available
        host_cpu = raw_state.get("host_cpu_avail", self.config.max_cpu_alloc)
        state[STATE_INDICES["host_cpu_avail"]] = np.clip(host_cpu / self.config.max_cpu_alloc, 0.0, 1.0)
        
        # 10. Host Mem available
        host_mem = raw_state.get("host_mem_avail", self.config.max_mem_alloc)
        state[STATE_INDICES["host_mem_avail"]] = np.clip(host_mem / self.config.max_mem_alloc, 0.0, 1.0)
        
        return state
