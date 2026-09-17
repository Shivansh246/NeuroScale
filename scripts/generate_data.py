#!/usr/bin/env python3
import json
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.config import PipelineConfig

def generate_synthetic_jsonl(path, n_samples=1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.linspace(0, 10 * np.pi, n_samples)
    cpu_percent = 40 + 30 * np.sin(t) + np.random.normal(0, 5, n_samples)
    memory_mb = 256 + 128 * np.cos(t) + np.random.normal(0, 10, n_samples)
    
    cpu_percent = np.clip(cpu_percent, 1.0, 200.0)
    memory_mb = np.clip(memory_mb, 10.0, 1024.0)
    
    start_time = time.time() - (n_samples * 10)
    
    with open(path, "w") as f:
        for i in range(n_samples):
            record = {
                "timestamp": start_time + i * 10,
                "container_id": "synthetic",
                "container_name": "synthetic",
                "cpu_percent": float(cpu_percent[i]),
                "cpu_usage_ns": int(cpu_percent[i] * 1e7),
                "memory_usage_mb": float(memory_mb[i]),
                "memory_usage_bytes": int(memory_mb[i] * 1024 * 1024),
                "memory_limit_mb": 1024.0,
                "memory_limit_bytes": 1024 * 1024 * 1024,
                "memory_percent": float(memory_mb[i] / 10.24),
                "workload_mode": "burst"
            }
            f.write(json.dumps(record) + "\n")
    print(f"Generated {n_samples} synthetic records at {path}")

if __name__ == '__main__':
    cfg = PipelineConfig()
    generate_synthetic_jsonl(cfg.raw_data_path, 2000)
