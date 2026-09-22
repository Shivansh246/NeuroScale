import json
import statistics
import sys
from collections import defaultdict

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def audit_timestamps(data):
    timestamps = [d["timestamp"] for d in data]
    diffs = [y - x for x, y in zip(timestamps, timestamps[1:])]
    
    print("=== 1. TIMESTAMP ANALYSIS ===")
    print(f"Min: {min(diffs):.3f}s")
    print(f"Max: {max(diffs):.3f}s")
    print(f"Mean: {statistics.mean(diffs):.3f}s")
    print(f"Median: {statistics.median(diffs):.3f}s")
    print(f"Std: {statistics.stdev(diffs):.3f}s")
    
    below_5 = sum(1 for d in diffs if d < 5)
    in_5_6 = sum(1 for d in diffs if 5 <= d < 6)
    in_6_7 = sum(1 for d in diffs if 6 <= d < 7)
    above_7 = sum(1 for d in diffs if d >= 7)
    
    print(f"Intervals < 5s: {below_5}")
    print(f"Intervals 5-6s: {in_5_6}")
    print(f"Intervals 6-7s: {in_6_7}")
    print(f"Intervals >= 7s: {above_7}")
    print()

def audit_memory(data):
    print("=== 2. MEMORY DISTRIBUTION ===")
    for phase in ["memory", "mixed"]:
        phase_data = [d for d in data if d["workload_mode"] == phase]
        if not phase_data:
            continue
        mem_mb = [d["memory_usage_mb"] for d in phase_data]
        limits = [d["memory_limit"] for d in phase_data]
        print(f"Phase: {phase}")
        print(f"  Min: {min(mem_mb):.2f} MB")
        print(f"  Max: {max(mem_mb):.2f} MB")
        print(f"  Mean: {statistics.mean(mem_mb):.2f} MB")
        print(f"  Median: {statistics.median(mem_mb):.2f} MB")
        print(f"  Std: {statistics.stdev(mem_mb):.2f} MB" if len(mem_mb) > 1 else "  Std: N/A")
        print(f"  First 10: {mem_mb[:10]}")
        print(f"  Last 10:  {mem_mb[-10:]}")
        print(f"  Docker Limit: {limits[0]:.2f} MB")
        print()

def audit_cpu(data):
    print("=== 3. CPU DISTRIBUTION ===")
    for phase in ["idle", "burst", "cpu", "memory", "mixed"]:
        phase_data = [d for d in data if d["workload_mode"] == phase]
        if not phase_data:
            continue
        cpu_pct = [d["cpu_percent"] for d in phase_data]
        print(f"Phase: {phase}")
        print(f"  Min: {min(cpu_pct):.2f}%")
        print(f"  Max: {max(cpu_pct):.2f}%")
        print(f"  Mean: {statistics.mean(cpu_pct):.2f}%")
        print(f"  Median: {statistics.median(cpu_pct):.2f}%")
        print(f"  Std: {statistics.stdev(cpu_pct):.2f}%" if len(cpu_pct) > 1 else "  Std: N/A")
        print(f"  First 5: {cpu_pct[:5]}")
        print(f"  Last 5:  {cpu_pct[-5:]}")
        print()

def audit_forecasting(data):
    print("=== 4. FORECASTING WINDOW ANALYSIS ===")
    history = 12
    # Count valid windows where phase doesn't change OR where we just count continuous data?
    # A valid supervised example requires a contiguous sequence of timestamps.
    # We will count how many windows exist overall (since all data is one sequence, though container restarts).
    # Wait, if container restarted, there's a jump in timestamp or container_id might change?
    # Our workload_schedule removes and restarts the container!
    # So container_name/id is new for each phase. A window shouldn't cross container boundary.
    
    # Split data by container_id (or phase blocks)
    blocks = []
    current_block = []
    current_id = None
    for d in data:
        if d["container_id"] != current_id:
            if current_block:
                blocks.append(current_block)
            current_block = [d]
            current_id = d["container_id"]
        else:
            current_block.append(d)
    if current_block:
        blocks.append(current_block)
        
    for steps_future in [1, 3, 6]:
        total_examples = 0
        phase_examples = defaultdict(int)
        for block in blocks:
            mode = block[0]["workload_mode"]
            # To predict `t + steps_future`, we need history `t-11` to `t`. So we need `12 + steps_future` samples minimum.
            # Number of valid windows in this block:
            valid = max(0, len(block) - history - steps_future + 1)
            total_examples += valid
            phase_examples[mode] += valid
            
        print(f"Horizon: {steps_future}-step future")
        print(f"  Overall: {total_examples} valid examples")
        for mode, count in phase_examples.items():
            print(f"  {mode:<10}: {count}")
        print()

def audit_transitions(data):
    print("=== 5. CHECK TRANSITIONS ===")
    print("Transitions are separated by container restarts. Let's look at the first 3 samples of each phase.")
    current_mode = None
    for i, d in enumerate(data):
        if d["workload_mode"] != current_mode:
            print(f"Transition to {d['workload_mode']}:")
            for j in range(i, min(i+3, len(data))):
                print(f"  t+{j-i}: CPU {data[j]['cpu_percent']:>6.2f}%, Mem {data[j]['memory_usage_mb']:>7.2f}MB, CPU_ns_delta {data[j]['cpu_usage_ns_delta']:>10.2e}")
            current_mode = d["workload_mode"]
    print()

def main():
    filepath = sys.argv[1]
    data = load_data(filepath)
    audit_timestamps(data)
    audit_memory(data)
    audit_cpu(data)
    audit_forecasting(data)
    audit_transitions(data)

if __name__ == "__main__":
    main()
