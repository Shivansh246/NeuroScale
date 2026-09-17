"""Reward calculation for the NeuroScale DRL Environment."""

from typing import Dict, Any
from .config import RewardConfig


class RewardCalculator:
    """Calculates the reward for an action given the state transition."""

    def __init__(self, config: RewardConfig):
        self.config = config

    def calculate(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, float],
        next_state: Dict[str, Any],
    ) -> float:
        """
        Calculate the reward based on SLA, resource waste, reallocation, and underprovisioning.
        
        Formula:
        reward = - sla_penalty
                 - resource_waste
                 - reallocation_penalty
                 - under_provision_penalty
                 - anomaly_penalty
        
        Note: The reward is formulated primarily as a sum of penalties to be minimized.
        """
        reward = 0.0
        
        # 1. SLA Penalty
        latency = next_state.get("sla_latency", 0.0)
        if latency > self.config.sla_latency_threshold_ms:
            # Scale penalty by how much it exceeded
            violation_ratio = (latency - self.config.sla_latency_threshold_ms) / self.config.sla_latency_threshold_ms
            reward -= self.config.sla_penalty_weight * (1.0 + violation_ratio)
            
        # 2. Resource Waste Penalty
        # Penalty for allocating significantly more than utilized
        cpu_alloc = action["cpu"]
        mem_alloc = action["memory"]
        cpu_util = next_state.get("current_cpu_util", 0.0)
        # Assuming cpu_util is in % (e.g. 100% = 1 core). Convert to cores.
        cpu_util_cores = cpu_util / 100.0
        
        mem_util = next_state.get("current_mem_util", 0.0)
        
        cpu_waste = max(0.0, cpu_alloc - cpu_util_cores)
        # Normalize mem waste to roughly match CPU scale (e.g. 1024MB roughly = 1 core weight)
        mem_waste = max(0.0, (mem_alloc - mem_util) / 1024.0)
        
        reward -= self.config.resource_waste_weight * (cpu_waste + mem_waste)
        
        # 3. Under-provision Penalty
        # Penalty for allocating less than current utilization (leads to throttling/OOM)
        if cpu_alloc < cpu_util_cores:
            reward -= self.config.under_provision_penalty * (cpu_util_cores - cpu_alloc)
        if mem_alloc < mem_util:
            reward -= self.config.under_provision_penalty * ((mem_util - mem_alloc) / 1024.0)
            
        # 4. Reallocation Penalty
        # Penalize if allocation changed from previous state
        prev_cpu = prev_state.get("current_cpu_alloc", cpu_alloc)
        prev_mem = prev_state.get("current_mem_alloc", mem_alloc)
        
        if prev_cpu != cpu_alloc or prev_mem != mem_alloc:
            reward -= self.config.reallocation_penalty
            
        # 5. Anomaly Penalty
        # Slight penalty if an anomaly is happening and we haven't given max resources
        # This encourages the agent to allocate defensively during anomalies
        is_anomaly = next_state.get("is_anomaly", False)
        if is_anomaly:
            # If not allocating max, apply penalty
            # Simplistic approach: fixed penalty if not highest tier
            reward -= self.config.anomaly_penalty
            
        return reward
