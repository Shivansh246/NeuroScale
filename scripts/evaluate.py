#!/usr/bin/env python3
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.runner import run_evaluation_suite

def generate_spike_trace():
    trace = []
    # Normal
    for _ in range(10):
        trace.append({
            "cpu_util": 20.0, "mem_util": 128,
            "pred_cpu": 20.0, "pred_mem": 128,
            "anomaly_score": 0.0, "is_anomaly": False
        })
    # Spike
    for _ in range(10):
        trace.append({
            "cpu_util": 90.0, "mem_util": 256,
            "pred_cpu": 90.0, "pred_mem": 256,
            "anomaly_score": 0.8, "is_anomaly": True
        })
    # Recovery
    for _ in range(10):
        trace.append({
            "cpu_util": 20.0, "mem_util": 128,
            "pred_cpu": 20.0, "pred_mem": 128,
            "anomaly_score": 0.0, "is_anomaly": False
        })
    return trace

def main():
    trace = generate_spike_trace()
    results = run_evaluation_suite("spike", trace)
    
    os.makedirs("data", exist_ok=True)
    out_file = "data/evaluation_results.jsonl"
    
    with open(out_file, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Evaluated {len(results)} strategies on 'spike' scenario.")
    print("Metrics summary:")
    for r in results:
        print(f" - {r['strategy']:15s}: SLA Violations: {r['sla_violations']:2d}, CPU Wastage: {r['cpu_wastage']:6.1f}, Reward: {r['reward']:.2f}")

if __name__ == '__main__':
    main()
