import json
import sys
from collections import defaultdict
import statistics

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def validate(data, interval=5):
    print("--- VALIDATION ---")
    if not data:
        print("No data found.")
        return False
        
    print(f"Total samples: {len(data)}")
    
    # Check fields
    required = [
        "timestamp", "container_id", "container_name", "cpu_percent", 
        "memory_usage_mb", "memory_percent", "cpu_usage_ns", "cpu_usage_ns_delta",
        "network_rx_bytes", "network_tx_bytes", "block_read_bytes", 
        "block_write_bytes", "workload_mode", "cpu_limit", "memory_limit"
    ]
    
    missing_fields = set()
    for d in data:
        for r in required:
            if r not in d:
                missing_fields.add(r)
                
    if missing_fields:
        print(f"ERROR: Missing fields: {missing_fields}")
        return False
        
    # Check numeric bounds
    out_of_bounds = 0
    for d in data:
        if d["cpu_percent"] < 0 or d["memory_usage_mb"] < 0 or d["cpu_usage_ns_delta"] < 0:
            out_of_bounds += 1
    if out_of_bounds > 0:
        print(f"ERROR: {out_of_bounds} records with negative values.")
        
    # Check timestamps
    timestamps = [d["timestamp"] for d in data]
    monotonic = all(x <= y for x, y in zip(timestamps, timestamps[1:]))
    if not monotonic:
        print("WARNING: Timestamps are not strictly monotonic (expected if phases restart quickly, but check carefully).")
        
    # Check intervals
    intervals = [y - x for x, y in zip(timestamps, timestamps[1:])]
    if intervals:
        avg_int = sum(intervals) / len(intervals)
        print(f"Average interval: {avg_int:.2f}s (expected ~{interval}s)")
        
    print("Validation passed (with possible warnings).")
    return True

def analyze(data):
    print("\n--- DISTRIBUTION ANALYSIS ---")
    workloads = defaultdict(list)
    for d in data:
        workloads[d["workload_mode"]].append(d)
        
    print(f"{'Workload':<15} {'Count':<8} {'CPU % (Mean/Std)':<20} {'Mem MB (Mean/Std)':<20} {'CPU ns delta (Mean)':<20}")
    print("-" * 85)
    
    for mode, records in workloads.items():
        cpu_pct = [r["cpu_percent"] for r in records]
        mem_mb = [r["memory_usage_mb"] for r in records]
        cpu_delta = [r["cpu_usage_ns_delta"] for r in records]
        
        c_mean = statistics.mean(cpu_pct)
        c_std = statistics.stdev(cpu_pct) if len(cpu_pct) > 1 else 0.0
        
        m_mean = statistics.mean(mem_mb)
        m_std = statistics.stdev(mem_mb) if len(mem_mb) > 1 else 0.0
        
        d_mean = statistics.mean(cpu_delta)
        
        print(f"{mode:<15} {len(records):<8} {c_mean:>6.2f} / {c_std:<6.2f}      {m_mean:>7.2f} / {m_std:<6.2f}      {d_mean:>15.2e}")
        
    # Global correlations or timelines can be added here if requested
    print("-" * 85)
    print("Overall Timeline:")
    current_mode = None
    start_t = None
    count = 0
    for d in data:
        mode = d["workload_mode"]
        if mode != current_mode:
            if current_mode is not None:
                duration = d["timestamp"] - start_t
                print(f"  {current_mode:<10} : {count} samples over {duration:.1f}s")
            current_mode = mode
            start_t = d["timestamp"]
            count = 1
        else:
            count += 1
            
    if current_mode is not None:
        duration = data[-1]["timestamp"] - start_t
        print(f"  {current_mode:<10} : {count} samples over {duration:.1f}s")

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/prediction_experiment/real_workload_pilot.jsonl"
    print(f"Loading {filepath}...")
    try:
        data = load_data(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return
        
    validate(data)
    analyze(data)

if __name__ == "__main__":
    main()
