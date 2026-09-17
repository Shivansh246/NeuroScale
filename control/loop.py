"""Thin orchestration layer for NeuroScale Closed-Loop Control."""

import time
import numpy as np
from typing import Dict, Any, List, Optional
import collections

from .config import ControlLoopConfig
from .logger import CycleLogger

class Orchestrator:
    """End-to-End Orchestrator for NeuroScale."""
    
    def __init__(
        self,
        config: ControlLoopConfig,
        collector,
        predictor,
        detector,
        agent,
        controller,
        action_space,
        state_builder,
        reward_calculator
    ):
        self.config = config
        self.collector = collector
        self.predictor = predictor
        self.detector = detector
        self.agent = agent
        self.controller = controller
        self.action_space = action_space
        self.state_builder = state_builder
        self.reward_calc = reward_calculator
        self.logger = CycleLogger(self.config.log_file)
        
        self.cycle_count = 0
        self.rolling_window = collections.deque(maxlen=self.config.window_size)
        
        # State tracking
        self.current_alloc_cpu = 1.0
        self.current_alloc_mem = 256
        self.prev_raw_state = None
        self.last_action_dict = None
        
    def _extract_sequence(self) -> np.ndarray:
        """Convert the rolling window into a numpy array matching feature_columns."""
        data = []
        for metric_dict in self.rolling_window:
            row = [metric_dict.get(col, 0.0) for col in self.config.feature_columns]
            data.append(row)
        return np.array(data, dtype=np.float32)

    def run_cycle(self, container_id: str, workload_mode: str = "unknown") -> Dict[str, Any]:
        """Execute one complete observe->predict->decide->control cycle."""
        start_time = time.time()
        self.cycle_count += 1
        
        # 1. Collect real metrics
        metrics = self.collector.collect(container_id)
        # If collector returns None (e.g. container stopped), handle it
        if not metrics:
            metrics = {col: 0.0 for col in self.config.feature_columns}
            metrics["timestamp"] = time.time()
            metrics["container_id"] = container_id
            
        self.rolling_window.append(metrics)
        
        cycle_record = {
            "timestamp": time.time(),
            "cycle_number": self.cycle_count,
            "container_id": container_id,
            "workload_mode": workload_mode,
            "current_cpu": metrics.get("cpu_percent", 0.0),
            "current_memory": metrics.get("memory_usage_mb", 0.0),
            "current_cpu_alloc": self.current_alloc_cpu,
            "current_mem_alloc": self.current_alloc_mem,
            "control_latency_ms": 0.0,
            "simulation_mode": self.config.simulation_mode,
        }
        
        # Warmup check
        if len(self.rolling_window) < self.config.warmup_cycles:
            cycle_record["status"] = "warming_up"
            cycle_record["control_latency_ms"] = (time.time() - start_time) * 1000.0
            self.logger.log(cycle_record)
            return cycle_record
            
        sequence = self._extract_sequence()
        
        # 2. Predict & Detect
        prediction = self.predictor.predict(sequence)
        anomaly = self.detector.score(sequence)
        
        cycle_record.update({
            "predicted_cpu": prediction.get("cpu", 0.0),
            "predicted_memory": prediction.get("memory", 0.0),
            "prediction_confidence": prediction.get("confidence", 0.0),
            "anomaly_score": anomaly.get("score", 0.0),
            "is_anomaly": anomaly.get("is_anomaly", False),
        })
        
        # 3. SLA Approximation (if real latency isn't available)
        cpu_util = metrics.get("cpu_percent", 0.0)
        mem_util = metrics.get("memory_usage_mb", 0.0)
        
        latency = self.config.base_latency_ms
        if cpu_util >= (self.current_alloc_cpu * 100.0) * 0.95:
            # Throttling proxy
            latency += 150.0
        if mem_util >= self.current_alloc_mem * 0.95:
            latency += 500.0
            
        cycle_record["sla_latency"] = latency
        cycle_record["sla_violation"] = latency > 200.0 # From reward config
        
        # 4. Build RL State
        raw_state = {
            "current_cpu_util": cpu_util,
            "current_mem_util": mem_util,
            "predicted_cpu_demand": prediction.get("cpu", 0.0),
            "predicted_mem_demand": prediction.get("memory", 0.0),
            "current_cpu_alloc": self.current_alloc_cpu,
            "current_mem_alloc": self.current_alloc_mem,
            "sla_latency": latency,
            "anomaly_score": anomaly.get("score", 0.0),
            "is_anomaly": anomaly.get("is_anomaly", False),
            "host_cpu_avail": 4.0, # Placeholder, could be retrieved via psutil
            "host_mem_avail": 4096.0,
        }
        
        state_vec = self.state_builder.build_state(raw_state)
        
        # 5. Decide (DQN)
        action_idx = self.agent.choose_action_index(state_vec, evaluate=True)
        action_dict = self.action_space.get_action(action_idx)
        target_cpu = action_dict["cpu"]
        target_mem = action_dict["memory"]
        
        cycle_record.update({
            "selected_action_cpu": target_cpu,
            "selected_action_memory": target_mem,
        })
        
        # Calculate Reward for the previous transition
        reward = 0.0
        if self.prev_raw_state is not None and self.last_action_dict is not None:
            reward = self.reward_calc.calculate(
                prev_state=self.prev_raw_state,
                action=self.last_action_dict,
                next_state=raw_state
            )
        cycle_record["reward"] = reward
        
        self.prev_raw_state = raw_state
        self.last_action_dict = action_dict
        
        # 6. Control (Apply)
        if not self.config.simulation_mode:
            result = self.controller.apply(container_id, cpu=target_cpu, memory=target_mem)
            cycle_record["controller_success"] = result.get("success", False)
            cycle_record["controller_error"] = result.get("error", None)
            
            if result.get("success", False):
                self.current_alloc_cpu = target_cpu
                self.current_alloc_mem = target_mem
        else:
            # In simulation, assume success
            cycle_record["controller_success"] = True
            self.current_alloc_cpu = target_cpu
            self.current_alloc_mem = target_mem
            
        cycle_record["status"] = "active"
        cycle_record["control_latency_ms"] = (time.time() - start_time) * 1000.0
        
        # 7. Log
        self.logger.log(cycle_record)
        
        return cycle_record
