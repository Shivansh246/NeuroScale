"""Environment supporting resource-aware reward decomposition."""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from .environment import NeuroScaleEnv
from .config import EnvConfig
from .decomposed_reward import DecomposedRewardCalculator


class DecomposedRewardEnv(NeuroScaleEnv):
    """
    Experimental environment that computes decomposed resource rewards (r_cpu, r_mem, r_exp).
    Preserves exact dynamics and state transitions of NeuroScaleEnv.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        super(DecomposedRewardEnv, self).__init__(config)
        self.decomposed_calculator = DecomposedRewardCalculator(self.config.reward)

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.max_steps - 1:
            state_vec = self.state_builder.build_state(self.current_raw_state)
            return state_vec, 0.0, True, False, {"msg": "End of trace", "r_cpu": 0.0, "r_mem": 0.0, "r_exp": 0.0}

        action_dict = self.action_space.get_action(action_index)
        cpu_alloc = action_dict["cpu"]
        mem_alloc = action_dict["memory"]

        self.current_step += 1
        trace_step = self.trace[self.current_step]

        target_cpu_util = trace_step.get("cpu_util", 0.0)
        target_mem_util = trace_step.get("mem_util", 0.0)

        actual_cpu_util = min(target_cpu_util, cpu_alloc * 100.0)
        actual_mem_util = min(target_mem_util, mem_alloc)

        base_latency = 50.0
        latency = base_latency

        if target_cpu_util > cpu_alloc * 100.0:
            cpu_deficit = target_cpu_util - (cpu_alloc * 100.0)
            latency += cpu_deficit * 5.0

        if target_mem_util > mem_alloc:
            mem_deficit = target_mem_util - mem_alloc
            latency += mem_deficit * 20.0

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
            "host_mem_avail": 4096.0,
            "target_cpu_util": target_cpu_util,
            "target_mem_util": target_mem_util,
        }

        # Calculate decomposed rewards
        r_cpu, r_mem, r_exp = self.decomposed_calculator.calculate_decomposed(
            prev_state=self.current_raw_state,
            action=action_dict,
            next_state=next_raw_state,
            target_cpu_util=target_cpu_util,
            target_mem_util=target_mem_util,
        )

        self.current_raw_state = next_raw_state
        state_vec = self.state_builder.build_state(self.current_raw_state)

        terminated = (self.current_step >= self.max_steps - 1)
        truncated = False
        info = {
            "raw_state": next_raw_state.copy(),
            "action": action_dict.copy(),
            "r_cpu": r_cpu,
            "r_mem": r_mem,
            "r_exp": r_exp,
        }

        return state_vec, r_exp, terminated, truncated, info
