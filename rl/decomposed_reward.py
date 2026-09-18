"""Resource-aware decomposed reward calculation for NeuroScale."""

from typing import Dict, Any, Tuple, Optional
from .config import RewardConfig


class DecomposedRewardCalculator:
    """
    Calculates resource-decomposed rewards for CPU and Memory dimensions
    while preserving the exact semantics and scaling of the environment reward.

    Formulation:
        r_cpu = - sla_pen_cpu - waste_pen_cpu - under_pen_cpu - realloc_cpu - anomaly_cpu
        r_mem = - sla_pen_mem - waste_pen_mem - under_pen_mem - realloc_mem - anomaly_mem
        r_experiment = r_cpu + r_mem

    Guarantees:
        r_experiment == r_cpu + r_mem
        r_experiment == r_original (exact mathematical equivalence, no double-counting)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def calculate_decomposed(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, float],
        next_state: Dict[str, Any],
        target_cpu_util: Optional[float] = None,
        target_mem_util: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculates (r_cpu, r_mem, r_experiment).
        """
        cpu_alloc = float(action["cpu"])
        mem_alloc = int(action["memory"])

        # 1. Resource Demands & Deficits
        # If explicit targets are not provided, fall back to next_state values
        target_cpu = target_cpu_util if target_cpu_util is not None else float(next_state.get("target_cpu_util", next_state.get("current_cpu_util", 0.0)))
        target_mem = target_mem_util if target_mem_util is not None else float(next_state.get("target_mem_util", next_state.get("current_mem_util", 0.0)))

        cpu_deficit = max(0.0, target_cpu - (cpu_alloc * 100.0))
        mem_deficit = max(0.0, target_mem - mem_alloc)

        # Simulator's explicit latency contributions:
        # 5.0 ms per 1% CPU deficit, 20.0 ms per 1 MB memory deficit
        cpu_latency_contrib = cpu_deficit * 5.0
        mem_latency_contrib = mem_deficit * 20.0
        total_latency_excess = cpu_latency_contrib + mem_latency_contrib

        # 2. SLA Penalty Decomposition
        latency = float(next_state.get("sla_latency", 50.0))
        total_sla_pen = 0.0
        if latency > self.config.sla_latency_threshold_ms:
            violation_ratio = (latency - self.config.sla_latency_threshold_ms) / self.config.sla_latency_threshold_ms
            raw_sla_pen = self.config.sla_penalty_weight * (1.0 + violation_ratio)
            sla_cap = getattr(self.config, "sla_penalty_cap", float("inf"))
            total_sla_pen = min(raw_sla_pen, sla_cap)

        if total_latency_excess > 0.0:
            alpha_cpu = cpu_latency_contrib / total_latency_excess
            alpha_mem = mem_latency_contrib / total_latency_excess
        else:
            alpha_cpu = 0.5
            alpha_mem = 0.5

        sla_pen_cpu = alpha_cpu * total_sla_pen
        sla_pen_mem = alpha_mem * total_sla_pen

        # 3. Resource Waste Decomposition
        cpu_util = float(next_state.get("current_cpu_util", 0.0))
        mem_util = float(next_state.get("current_mem_util", 0.0))
        cpu_util_cores = cpu_util / 100.0

        cpu_waste = max(0.0, cpu_alloc - cpu_util_cores)
        mem_divisor = getattr(self.config, "memory_waste_divisor", 1024.0)
        mem_waste = max(0.0, (mem_alloc - mem_util) / mem_divisor)

        waste_pen_cpu = self.config.resource_waste_weight * cpu_waste
        waste_pen_mem = self.config.resource_waste_weight * mem_waste

        # 4. Under-provision Penalty Decomposition
        # Evaluated on post-step utilization matching original RewardCalculator semantics
        under_pen_cpu = 0.0
        under_pen_mem = 0.0
        if cpu_alloc < cpu_util_cores:
            under_pen_cpu = self.config.under_provision_penalty * (cpu_util_cores - cpu_alloc)
        if mem_alloc < mem_util:
            under_pen_mem = self.config.under_provision_penalty * ((mem_util - mem_alloc) / mem_divisor)

        # 5. Reallocation Penalty Decomposition
        prev_cpu = float(prev_state.get("current_cpu_alloc", cpu_alloc))
        prev_mem = int(prev_state.get("current_mem_alloc", mem_alloc))

        cpu_realloc = (prev_cpu != cpu_alloc)
        mem_realloc = (prev_mem != mem_alloc)

        if cpu_realloc and mem_realloc:
            # Both changed: split 0.5 penalty equally (0.25 each)
            realloc_cpu = 0.5 * self.config.reallocation_penalty
            realloc_mem = 0.5 * self.config.reallocation_penalty
        elif cpu_realloc and not mem_realloc:
            realloc_cpu = self.config.reallocation_penalty
            realloc_mem = 0.0
        elif not cpu_realloc and mem_realloc:
            realloc_cpu = 0.0
            realloc_mem = self.config.reallocation_penalty
        else:
            realloc_cpu = 0.0
            realloc_mem = 0.0

        # 6. Anomaly Penalty Decomposition
        # Shared system-wide condition, split equally between both resource branches
        is_anomaly = bool(next_state.get("is_anomaly", False))
        if is_anomaly:
            anomaly_cpu = 0.5 * self.config.anomaly_penalty
            anomaly_mem = 0.5 * self.config.anomaly_penalty
        else:
            anomaly_cpu = 0.0
            anomaly_mem = 0.0

        # Assemble component rewards
        r_cpu = - (sla_pen_cpu + waste_pen_cpu + under_pen_cpu + realloc_cpu + anomaly_cpu)
        r_mem = - (sla_pen_mem + waste_pen_mem + under_pen_mem + realloc_mem + anomaly_mem)
        r_experiment = r_cpu + r_mem

        return float(r_cpu), float(r_mem), float(r_experiment)

    def calculate(
        self,
        prev_state: Dict[str, Any],
        action: Dict[str, float],
        next_state: Dict[str, Any],
        target_cpu_util: Optional[float] = None,
        target_mem_util: Optional[float] = None,
    ) -> float:
        """Returns total scalar reward r_experiment."""
        _, _, r_exp = self.calculate_decomposed(
            prev_state=prev_state,
            action=action,
            next_state=next_state,
            target_cpu_util=target_cpu_util,
            target_mem_util=target_mem_util,
        )
        return r_exp
