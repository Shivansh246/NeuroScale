"""Experiment configuration parser and dataclass models for NeuroScale telemetry collection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import re


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Lightweight, zero-dependency YAML parser for standard key-value and nested mappings.
    
    Falls back to PyYAML if installed, otherwise parses standard YAML structures using
    indentation and types (numbers, booleans, strings, lists, dicts).
    """
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    # Zero-dependency parser for standard YAML subsets
    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any, Optional[str]]] = [(-1, root, None)]

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))

        # Pop stack if current indent is less than or equal to current top
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent_indent, parent_obj, parent_key = stack[-1]

        # Check for list item
        if stripped.startswith("- "):
            val_str = stripped[2:].strip()
            val = _parse_scalar(val_str)
            if not isinstance(parent_obj, list):
                # Convert or assign parent to list
                new_list = [val]
                if parent_key and isinstance(stack[-2][1], dict):
                    stack[-2][1][parent_key] = new_list
                stack.append((indent, new_list, None))
            else:
                parent_obj.append(val)
            continue

        # Key-value pair
        if ":" in stripped:
            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()

            if not rest or rest.startswith("#"):
                # Nested mapping begins
                new_dict: Dict[str, Any] = {}
                if isinstance(parent_obj, dict):
                    parent_obj[key] = new_dict
                stack.append((indent, new_dict, key))
            else:
                # Inlined comment removal
                if " #" in rest:
                    rest = rest.split(" #", 1)[0].strip()
                val = _parse_scalar(rest)
                if isinstance(parent_obj, dict):
                    parent_obj[key] = val

    return root


def _parse_scalar(val_str: str) -> Any:
    """Parse scalar YAML string values into Python primitive types."""
    if val_str.lower() in ("true", "yes", "on"):
        return True
    if val_str.lower() in ("false", "no", "off"):
        return False
    if val_str.lower() in ("null", "none", "~", ""):
        return None
    # Inlined list like [20, 100]
    if val_str.startswith("[") and val_str.endswith("]"):
        items = val_str[1:-1].split(",")
        return [_parse_scalar(item.strip()) for item in items if item.strip()]
    # Integers
    if re.match(r"^-?\d+$", val_str):
        return int(val_str)
    # Floats
    if re.match(r"^-?\d+\.\d+$", val_str):
        return float(val_str)
    # Quoted string stripping
    if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    return val_str


@dataclass
class ExperimentConfig:
    name: str = "default_benchmark"
    seed: int = 42
    duration_seconds: int = 300
    sampling_interval_seconds: float = 1.0


@dataclass
class ContainerConfig:
    concurrent_instances: int = 2
    base_image: str = "neuroscale-workload:latest"
    network_name: str = "neuroscale_bench_net"
    cpus: float = 2.0
    memory_mb: int = 1024


@dataclass
class RandomizationConfig:
    enabled: bool = True
    min_phase_duration_seconds: int = 15
    max_phase_duration_seconds: int = 60
    profile_weights: Dict[str, float] = field(default_factory=lambda: {
        "idle": 0.15,
        "cpu_intensive": 0.25,
        "memory_intensive": 0.20,
        "disk_io_intensive": 0.15,
        "network_intensive": 0.10,
        "mixed": 0.15,
    })
    intensity_ranges: Dict[str, List[Any]] = field(default_factory=lambda: {
        "cpu_utilization_target": [20, 100],
        "memory_allocation_mb": [128, 896],
        "io_block_size_kb": [4, 1024],
        "network_packet_rate_mbps": [5, 100],
    })


@dataclass
class ExportConfig:
    export_csv: bool = True
    export_jsonl: bool = True
    sync_to_data_raw_metrics: bool = True
    datasets_dir: str = "datasets"


@dataclass
class TelemetryCollectionConfig:
    """Master configuration container for telemetry benchmark runs."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    containers: ContainerConfig = field(default_factory=ContainerConfig)
    workload_randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "TelemetryCollectionConfig":
        """Load and validate an experiment configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        data = _parse_simple_yaml(content)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryCollectionConfig":
        """Construct configuration object from raw dictionary, applying validations and defaults."""
        exp_data = data.get("experiment", {})
        cnt_data = data.get("containers", {})
        res_data = cnt_data.get("resource_constraints", {})
        rnd_data = data.get("workload_randomization", {})
        exp_cfg_data = data.get("export", {})

        experiment = ExperimentConfig(
            name=str(exp_data.get("name", "default_benchmark")),
            seed=int(exp_data.get("seed", 42)),
            duration_seconds=int(exp_data.get("duration_seconds", 300)),
            sampling_interval_seconds=float(exp_data.get("sampling_interval_seconds", 1.0)),
        )

        containers = ContainerConfig(
            concurrent_instances=int(cnt_data.get("concurrent_instances", 2)),
            base_image=str(cnt_data.get("base_image", "neuroscale-workload:latest")),
            network_name=str(cnt_data.get("network_name", "neuroscale_bench_net")),
            cpus=float(res_data.get("cpus", 2.0)),
            memory_mb=int(res_data.get("memory_mb", 1024)),
        )

        workload_randomization = RandomizationConfig(
            enabled=bool(rnd_data.get("enabled", True)),
            min_phase_duration_seconds=int(rnd_data.get("min_phase_duration_seconds", 15)),
            max_phase_duration_seconds=int(rnd_data.get("max_phase_duration_seconds", 60)),
            profile_weights=dict(rnd_data.get("profile_weights", {
                "idle": 0.15,
                "cpu_intensive": 0.25,
                "memory_intensive": 0.20,
                "disk_io_intensive": 0.15,
                "network_intensive": 0.10,
                "mixed": 0.15,
            })),
            intensity_ranges=dict(rnd_data.get("intensity_ranges", {
                "cpu_utilization_target": [20, 100],
                "memory_allocation_mb": [128, 896],
                "io_block_size_kb": [4, 1024],
                "network_packet_rate_mbps": [5, 100],
            })),
        )

        export = ExportConfig(
            export_csv=bool(exp_cfg_data.get("export_csv", True)),
            export_jsonl=bool(exp_cfg_data.get("export_jsonl", True)),
            sync_to_data_raw_metrics=bool(exp_cfg_data.get("sync_to_data_raw_metrics", True)),
            datasets_dir=str(exp_cfg_data.get("datasets_dir", "datasets")),
        )

        # Basic validations
        if experiment.duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be positive, got {experiment.duration_seconds}")
        if experiment.sampling_interval_seconds <= 0:
            raise ValueError(f"sampling_interval_seconds must be positive, got {experiment.sampling_interval_seconds}")
        if containers.concurrent_instances < 1:
            raise ValueError(f"concurrent_instances must be >= 1, got {containers.concurrent_instances}")

        return cls(
            experiment=experiment,
            containers=containers,
            workload_randomization=workload_randomization,
            export=export,
            raw_dict=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON/YAML serializable dictionary."""
        return {
            "experiment": {
                "name": self.experiment.name,
                "seed": self.experiment.seed,
                "duration_seconds": self.experiment.duration_seconds,
                "sampling_interval_seconds": self.experiment.sampling_interval_seconds,
            },
            "containers": {
                "concurrent_instances": self.containers.concurrent_instances,
                "base_image": self.containers.base_image,
                "network_name": self.containers.network_name,
                "resource_constraints": {
                    "cpus": self.containers.cpus,
                    "memory_mb": self.containers.memory_mb,
                },
            },
            "workload_randomization": {
                "enabled": self.workload_randomization.enabled,
                "min_phase_duration_seconds": self.workload_randomization.min_phase_duration_seconds,
                "max_phase_duration_seconds": self.workload_randomization.max_phase_duration_seconds,
                "profile_weights": self.workload_randomization.profile_weights,
                "intensity_ranges": self.workload_randomization.intensity_ranges,
            },
            "export": {
                "export_csv": self.export.export_csv,
                "export_jsonl": self.export.export_jsonl,
                "sync_to_data_raw_metrics": self.export.sync_to_data_raw_metrics,
                "datasets_dir": self.export.datasets_dir,
            },
        }
