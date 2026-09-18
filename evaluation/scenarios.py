"""Deterministic evaluation scenarios and trace normalization."""

import math
from typing import Dict, List, Any


def normalize_trace_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw workload step into the common evaluation trace schema."""
    cpu = float(step.get("current_cpu", step.get("cpu_util", step.get("cpu_percent", 0.0))))
    mem = float(step.get("current_memory", step.get("mem_util", step.get("memory_usage_mb", 0.0))))

    # Feature columns expected by Transformer & Autoencoder:
    # ["cpu_percent", "cpu_usage_ns", "memory_usage_mb", "memory_percent"]
    cpu_ns = int(step.get("cpu_usage_ns", int(cpu * 1e7)))
    mem_pct = float(step.get("memory_percent", (mem / 1024.0) * 100.0))

    return {
        "current_cpu": cpu,
        "current_memory": mem,
        "cpu_util": cpu,
        "mem_util": mem,
        "cpu_percent": cpu,
        "cpu_usage_ns": cpu_ns,
        "memory_usage_mb": mem,
        "memory_percent": mem_pct,
    }


def normalize_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize an entire workload trace into the common schema."""
    return [normalize_trace_step(step) for step in trace]


def get_evaluation_scenarios(steps_per_scenario: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate the 4 deterministic evaluation scenarios:
    - normal: low/steady workload with subtle deterministic oscillation
    - cpu_spike: steady warmup -> sharp CPU demand spike -> recovery
    - mem_spike: steady warmup -> heavy memory demand spike -> recovery
    - sustained: steady warmup -> sustained high CPU and memory demand

    All steps provide both human-readable (current_cpu, current_memory) and
    model-expected feature columns (cpu_percent, cpu_usage_ns, memory_usage_mb, memory_percent).
    """
    scenarios: Dict[str, List[Dict[str, Any]]] = {}

    # 1. Normal Workload (low/steady ~40% CPU, 128 MB RAM)
    normal = []
    for i in range(steps_per_scenario):
        cpu = 40.0 + 3.0 * math.sin(i * 0.4)
        mem = 128.0 + 10.0 * math.cos(i * 0.4)
        normal.append({"cpu_util": round(cpu, 2), "mem_util": round(mem, 1)})
    scenarios["normal"] = normalize_trace(normal)

    # 2. CPU Spike
    # Warmup (15 steps @ 40%), Spike (15 steps @ 180%), Recovery (20 steps @ 40%)
    warmup_len = max(15, int(steps_per_scenario * 0.3))
    spike_len = max(15, int(steps_per_scenario * 0.3))
    recovery_len = max(0, steps_per_scenario - warmup_len - spike_len)

    cpu_spike = (
        [{"cpu_util": 40.0, "mem_util": 128.0} for _ in range(warmup_len)]
        + [{"cpu_util": 180.0, "mem_util": 128.0} for _ in range(spike_len)]
        + [{"cpu_util": 40.0, "mem_util": 128.0} for _ in range(recovery_len)]
    )
    scenarios["cpu_spike"] = normalize_trace(cpu_spike)

    # 3. Memory Spike
    # Warmup (15 steps @ 128MB), Spike (15 steps @ 768MB), Recovery (20 steps @ 128MB)
    mem_spike = (
        [{"cpu_util": 40.0, "mem_util": 128.0} for _ in range(warmup_len)]
        + [{"cpu_util": 40.0, "mem_util": 768.0} for _ in range(spike_len)]
        + [{"cpu_util": 40.0, "mem_util": 128.0} for _ in range(recovery_len)]
    )
    scenarios["mem_spike"] = normalize_trace(mem_spike)

    # 4. Sustained High (overload / sustained stress)
    # Warmup (15 steps @ 40% CPU, 128MB), Sustained (35 steps @ 150% CPU, 512MB)
    sustained_len = max(0, steps_per_scenario - warmup_len)
    sustained = (
        [{"cpu_util": 40.0, "mem_util": 128.0} for _ in range(warmup_len)]
        + [{"cpu_util": 150.0, "mem_util": 512.0} for _ in range(sustained_len)]
    )
    scenarios["sustained"] = normalize_trace(sustained)

    return scenarios
