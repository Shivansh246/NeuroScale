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
    print("=== 1. DATA VALIDATION ===")
    if not data:
        print("No data found.")
        return False
        
    print(f"Total samples: {len(data)}")
    
    container_ids = set(d["container_id"] for d in data)
    print(f"Container IDs found: {len(container_ids)}")
    if len(container_ids) > 1:
        print("ERROR: Multiple container IDs found! Not a continuous time series.")
    else:
        print(f"Persistent Container ID: {list(container_ids)[0]}")
        
    timestamps = [d["timestamp"] for d in data]
    monotonic = all(x <= y for x, y in zip(timestamps, timestamps[1:]))
    print(f"Timestamp monotonicity: {'PASS' if monotonic else 'FAIL'}")
    
    intervals = [y - x for x, y in zip(timestamps, timestamps[1:])]
    if intervals:
        avg_int = statistics.mean(intervals)
        std_int = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
        print(f"Sampling interval: {avg_int:.3f}s (std: {std_int:.3f}s)")
    
    neg_deltas = sum(1 for d in data if d["cpu_usage_ns_delta"] < 0)
    print(f"Negative CPU deltas: {neg_deltas}")
    print()

def analyze_memory(data):
    print("=== 2. MEMORY BEHAVIOR ===")
    mem = [d["memory_usage_mb"] for d in data]
    if not mem:
        return
    print(f"Min memory: {min(mem):.2f} MB")
    print(f"Max memory: {max(mem):.2f} MB")
    print(f"Mean memory: {statistics.mean(mem):.2f} MB")
    print(f"Median memory: {statistics.median(mem):.2f} MB")
    print(f"Std memory: {statistics.stdev(mem):.2f} MB" if len(mem) > 1 else "Std: N/A")
    print()

def analyze_forecasting(data):
    print("=== 3. FORECASTING WINDOWS ===")
    history = 12
    # Because it is continuous, ANY contiguous 12+k block is valid.
    # Total samples = N. Valid for k-step future = N - history - k + 1
    N = len(data)
    for steps_future in [1, 3, 6]:
        valid = max(0, N - history - steps_future + 1)
        
        # How many windows contain transitions?
        # A window spans indices i to i + history + steps_future - 1.
        # It contains a transition if the workload_mode changes within this span.
        transitions = 0
        for i in range(valid):
            modes = set(d["workload_mode"] for d in data[i : i + history + steps_future])
            if len(modes) > 1:
                transitions += 1
        print(f"12-step -> {steps_future}-step future:")
        print(f"  Total valid windows: {valid}")
        print(f"  Windows containing transitions: {transitions}")
    print()
    
def get_workload_stats(data):
    print("=== 4. WORKLOAD SAMPLES ===")
    counts = defaultdict(int)
    for d in data:
        counts[d["workload_mode"]] += 1
    for mode, count in counts.items():
        print(f"  {mode}: {count} samples")
    print()

def analyze_transitions(data):
    print("=== 5. TRANSITION VERIFICATION ===")
    current_mode = None
    for i, d in enumerate(data):
        if current_mode is not None and d["workload_mode"] != current_mode:
            print(f"Transition: {current_mode} -> {d['workload_mode']} at sample {i}")
            # print 1 before and 2 after
            start_idx = max(0, i - 1)
            end_idx = min(len(data), i + 2)
            for j in range(start_idx, end_idx):
                prefix = "  *" if j == i else "   "
                print(f"{prefix} [{data[j]['workload_mode']:<6}] CPU: {data[j]['cpu_percent']:>6.2f}%, Mem: {data[j]['memory_usage_mb']:>7.2f} MB, CPU_ns: {data[j]['cpu_usage_ns_delta']:>10.2e}")
        current_mode = d["workload_mode"]
    print()

def main():
    filepath = sys.argv[1]
    data = load_data(filepath)
    validate(data)
    get_workload_stats(data)
    analyze_memory(data)
    analyze_forecasting(data)
    analyze_transitions(data)

if __name__ == "__main__":
    main()
