"""Configuration for the NeuroScale closed-loop controller."""

from dataclasses import dataclass

@dataclass
class ControlLoopConfig:
    """Settings for the orchestration loop."""
    window_size: int = 12
    warmup_cycles: int = 12
    feature_columns: tuple = ("cpu_percent", "cpu_usage_ns_delta", "memory_usage_mb", "memory_percent")
    log_file: str = "data/control_loop.jsonl"
    simulation_mode: bool = False
    
    # Target latency approximation logic
    # In real mode, if we lack true latency, we estimate it from CPU throttling
    # just like the simulated RL environment does.
    base_latency_ms: float = 50.0
