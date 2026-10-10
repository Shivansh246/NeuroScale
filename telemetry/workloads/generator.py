"""Workload generator and factory for constructing reproducible workload profiles."""

from typing import Dict, Any, Type, Optional
import random

from .base import BaseWorkload
from .profiles import (
    IdleWorkload,
    CPUIntensiveWorkload,
    MemoryIntensiveWorkload,
    DiskIOWorkload,
    NetworkIntensiveWorkload,
    MixedWorkload,
)


class WorkloadFactory:
    """Factory for instantiating and configuring workload profiles by name."""

    _REGISTRY: Dict[str, Type[BaseWorkload]] = {
        "idle": IdleWorkload,
        "cpu": CPUIntensiveWorkload,
        "cpu_intensive": CPUIntensiveWorkload,
        "memory": MemoryIntensiveWorkload,
        "memory_intensive": MemoryIntensiveWorkload,
        "disk": DiskIOWorkload,
        "disk_io": DiskIOWorkload,
        "disk_io_intensive": DiskIOWorkload,
        "network": NetworkIntensiveWorkload,
        "network_intensive": NetworkIntensiveWorkload,
        "mixed": MixedWorkload,
    }

    @classmethod
    def register(cls, name: str, workload_cls: Type[BaseWorkload]) -> None:
        """Register a new custom workload profile class."""
        cls._REGISTRY[name.lower()] = workload_cls

    @classmethod
    def create(cls, profile_name: str, params: Optional[Dict[str, Any]] = None) -> BaseWorkload:
        """Instantiate a workload profile by name with parameters."""
        key = profile_name.lower()
        if key not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ValueError(f"Unknown workload profile '{profile_name}'. Available: {available}")

        workload_cls = cls._REGISTRY[key]
        return workload_cls(params=params)

    @classmethod
    def create_randomized(
        cls,
        experiment_config: Any,
        seed: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> BaseWorkload:
        """Create a randomized workload profile based on experiment YAML bounds and seed."""
        rnd_cfg = experiment_config.workload_randomization
        rng_seed = seed if seed is not None else experiment_config.experiment.seed
        rng = random.Random(rng_seed)

        # Select profile based on weights
        profiles = list(rnd_cfg.profile_weights.keys())
        weights = [rnd_cfg.profile_weights[p] for p in profiles]
        chosen_profile = rng.choices(profiles, weights=weights, k=1)[0]

        # Determine phase duration
        dur = duration or rng.randint(
            rnd_cfg.min_phase_duration_seconds,
            rnd_cfg.max_phase_duration_seconds,
        )

        params: Dict[str, Any] = {"duration": dur, "seed": rng.randint(1, 100000)}

        # Sample intensity bounds from YAML config
        ranges = rnd_cfg.intensity_ranges
        if chosen_profile in ("cpu", "cpu_intensive"):
            cpu_min, cpu_max = ranges.get("cpu_utilization_target", [20, 100])
            params["intensity"] = rng.randint(int(cpu_min), int(cpu_max))
        elif chosen_profile in ("memory", "memory_intensive"):
            mem_min, mem_max = ranges.get("memory_allocation_mb", [128, 512])
            params["target_mb"] = rng.randint(int(mem_min), int(mem_max))
            params["chunk_mb"] = 16
        elif chosen_profile in ("disk", "disk_io", "disk_io_intensive"):
            bs_min, bs_max = ranges.get("io_block_size_kb", [4, 512])
            params["block_size_kb"] = rng.choice([4, 16, 64, 128, 256])
            params["write_rate_kbps"] = rng.randint(1024, 10240)
        elif chosen_profile in ("network", "network_intensive"):
            rate_min, rate_max = ranges.get("network_packet_rate_mbps", [5, 50])
            params["throughput_kbps"] = int(rng.uniform(rate_min, rate_max) * 128)
            params["target_host"] = "172.28.0.10"
        elif chosen_profile == "mixed":
            params["seed"] = rng.randint(1, 100000)

        return cls.create(chosen_profile, params)
