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
        self.prev_cpu_usage_ns = None
        self.current_session_id = None
        
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

    def _prepare_transformer_sequence(self, canonical_sequence: np.ndarray) -> np.ndarray:
        """Apply Transformer-specific preprocessing to the canonical telemetry."""
        seq = canonical_sequence.copy()
        if "cpu_usage_ns_delta" in self.config.feature_columns:
            idx = self.config.feature_columns.index("cpu_usage_ns_delta")
            # Transformer was trained on log1p(max(0, delta))
            seq[:, idx] = np.log1p(np.maximum(0, seq[:, idx]))
        return seq

    def _prepare_anomaly_sequence(self, canonical_sequence: np.ndarray) -> np.ndarray:
        """Apply Autoencoder-specific preprocessing to the canonical telemetry."""
        # Autoencoder was trained on raw delta, so return unaltered copy
        return canonical_sequence.copy()

    def run_cycle(self, container_id: str, workload_mode: str = "unknown") -> Dict[str, Any]:
        """Execute one complete observe->predict->decide->control cycle."""
        start_time = time.time()
        self.cycle_count += 1
        
        # 1. Collect real metrics
        t_col_start = time.time()
        metrics = self.collector.collect(container_id)
        collector_latency_ms = (time.time() - t_col_start) * 1000.0
        
        counter_reset = False
        
        # If collector returns None (e.g. container stopped), handle it
        if not metrics:
            metrics = {col: 0.0 for col in self.config.feature_columns}
            metrics["timestamp"] = time.time()
            metrics["container_id"] = container_id
            
        # Compute cpu_usage_ns_delta
        current_cpu_ns = metrics.get("cpu_usage_ns", 0)
        
        if self.current_session_id != container_id:
            # First sample or container changed
            self.current_session_id = container_id
            metrics["cpu_usage_ns_delta"] = 0
            self.prev_cpu_usage_ns = current_cpu_ns
        else:
            if current_cpu_ns < self.prev_cpu_usage_ns:
                # Docker counter reset!
                counter_reset = True
                metrics["cpu_usage_ns_delta"] = current_cpu_ns
            else:
                metrics["cpu_usage_ns_delta"] = current_cpu_ns - self.prev_cpu_usage_ns
            self.prev_cpu_usage_ns = current_cpu_ns
            
        self.rolling_window.append(metrics)
        
        cycle_record = {
            "timestamp": time.time(),
            "counter_reset": counter_reset,
            "cpu_usage_ns": current_cpu_ns,
            "cpu_usage_ns_delta": metrics.get("cpu_usage_ns_delta", 0),
            "cycle_number": self.cycle_count,
            "container_id": container_id,
            "workload_mode": workload_mode,
            "current_cpu": metrics.get("cpu_percent", 0.0),
            "current_memory": metrics.get("memory_usage_mb", 0.0),
            "current_cpu_alloc": self.current_alloc_cpu,
            "current_mem_alloc": self.current_alloc_mem,
            "collector_latency_ms": collector_latency_ms,
            "controller_operation_latency_ms": 0.0,
            "model_inference_latency_ms": 0.0,
            "control_latency_ms": 0.0,
            "cycle_wall_time_ms": 0.0,
            "sampling_interval_ms": 1000.0,
            "simulation_mode": self.config.simulation_mode,
        }
        
        # Warmup check
        if len(self.rolling_window) < self.config.warmup_cycles:
            cycle_record["status"] = "warming_up"
            wall_ms = (time.time() - start_time) * 1000.0
            cycle_record["control_latency_ms"] = wall_ms
            cycle_record["cycle_wall_time_ms"] = wall_ms
            self.logger.log(cycle_record)
            return cycle_record
            
        canonical_sequence = self._extract_sequence()
        transformer_sequence = self._prepare_transformer_sequence(canonical_sequence)
        anomaly_sequence = self._prepare_anomaly_sequence(canonical_sequence)
        
        # 2. Predict & Detect
        t_infer_start = time.time()
        prediction = self.predictor.predict(transformer_sequence)
        anomaly = self.detector.score(anomaly_sequence)
        
        cycle_record.update({
            "transformer_cpu_delta": float(transformer_sequence[-1, 1]) if "cpu_usage_ns_delta" in self.config.feature_columns else 0.0,
            "anomaly_cpu_delta": float(anomaly_sequence[-1, 1]) if "cpu_usage_ns_delta" in self.config.feature_columns else 0.0,
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
        pred_cpu = prediction.get("cpu", 0.0)
        pred_mem = prediction.get("memory", 0.0)
        # If predictor output is percentage, convert to MB based on current allocation/limit
        if "memory_percent" in getattr(self.predictor, "_target_columns", []):
            pred_mem_mb = (pred_mem / 100.0) * float(self.current_alloc_mem)
        else:
            pred_mem_mb = pred_mem

        raw_state = {
            "current_cpu_util": cpu_util,
            "current_mem_util": mem_util,
            "predicted_cpu_demand": pred_cpu,
            "predicted_mem_demand": pred_mem_mb,
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
        if hasattr(self.agent, "choose_branch_indices"):
            cpu_idx, mem_idx = self.agent.choose_branch_indices(state_vec, evaluate=True)
        else:
            cpu_idx = action_idx // 4
            mem_idx = action_idx % 4

        action_dict = self.action_space.get_action(action_idx)
        target_cpu = action_dict["cpu"]
        target_mem = action_dict["memory"]
        model_inference_latency_ms = (time.time() - t_infer_start) * 1000.0
        
        cycle_record.update({
            "selected_action_cpu": target_cpu,
            "selected_action_memory": target_mem,
            "selected_cpu_allocation": target_cpu,
            "selected_memory_allocation": target_mem,
            "action_index": action_idx,
            "cpu_action_index": cpu_idx,
            "mem_action_index": mem_idx,
            "current_cpu_allocation": self.current_alloc_cpu,
            "current_memory_allocation": self.current_alloc_mem,
            "model_inference_latency_ms": model_inference_latency_ms,
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
        docker_before = {}
        docker_after = {}
        docker_verified = False
        controller_operation_latency_ms = 0.0

        if not self.config.simulation_mode:
            # Query Docker state before reallocation
            if hasattr(self.controller, "client") and self.controller.client is not None:
                try:
                    c_inspect = self.controller.client.containers.get(container_id)
                    h_cfg = c_inspect.attrs.get("HostConfig", {})
                    docker_before = {
                        "cpu_quota": h_cfg.get("CpuQuota"),
                        "cpu_period": h_cfg.get("CpuPeriod"),
                        "memory_bytes": h_cfg.get("Memory"),
                    }
                except Exception:
                    pass

            t_ctrl_op_start = time.time()
            result = self.controller.apply(container_id, cpu=target_cpu, memory=target_mem)
            controller_operation_latency_ms = (time.time() - t_ctrl_op_start) * 1000.0
            cycle_record["controller_success"] = result.get("success", False)
            cycle_record["controller_error"] = result.get("error", None)
            cycle_record["error"] = result.get("error", None)
            
            if result.get("success", False):
                self.current_alloc_cpu = target_cpu
                self.current_alloc_mem = target_mem

                # Query Docker state after reallocation to verify actual change
                if hasattr(self.controller, "client") and self.controller.client is not None:
                    try:
                        c_inspect = self.controller.client.containers.get(container_id)
                        c_inspect.reload()
                        h_cfg = c_inspect.attrs.get("HostConfig", {})
                        docker_after = {
                            "cpu_quota": h_cfg.get("CpuQuota"),
                            "cpu_period": h_cfg.get("CpuPeriod"),
                            "memory_bytes": h_cfg.get("Memory"),
                        }
                        expected_quota = int(target_cpu * h_cfg.get("CpuPeriod", 100000))
                        expected_mem = int(target_mem * 1024 * 1024)
                        docker_verified = (
                            h_cfg.get("CpuQuota") == expected_quota and
                            h_cfg.get("Memory") == expected_mem
                        )
                    except Exception:
                        pass
        else:
            # In simulation, assume success
            cycle_record["controller_success"] = True
            cycle_record["error"] = None
            self.current_alloc_cpu = target_cpu
            self.current_alloc_mem = target_mem
            
        cycle_record["status"] = "active"
        wall_ms = (time.time() - start_time) * 1000.0
        cycle_record["control_latency_ms"] = wall_ms
        cycle_record["controller_latency"] = wall_ms
        cycle_record["controller_operation_latency_ms"] = controller_operation_latency_ms
        cycle_record["cycle_wall_time_ms"] = wall_ms
        cycle_record["docker_before"] = docker_before
        cycle_record["docker_after"] = docker_after
        cycle_record["docker_verified"] = docker_verified
        
        # 7. Log
        self.logger.log(cycle_record)
        
        return cycle_record
