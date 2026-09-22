import argparse
import time
import docker
import json
import os
import threading

from experiments.prediction_data.collector import PilotCollector

# Configurable schedule
# Mode, Duration (s), Memory (MB)
PHASES = [
    ("idle", 300, 256),
    ("cpu", 600, 256), # Wait, NORMAL vs CPU STRESS. In previous code:
    # "normal states -> A1 (0.25 CPU / 256 MB), stress states -> A15 (2 CPU / 1024 MB)"
    # But workload_mode is just the mode. Let's use:
    # IDLE, NORMAL (maybe 'cpu' with small args? or 'idle' with some load), CPU STRESS ('cpu' high load).
    # The `workload.py` modes are "idle", "cpu", "memory", "burst", "mixed".
    # I will just map:
    # IDLE -> "idle"
    # NORMAL -> "burst" or "mixed"? Let's use "mixed" with low memory?
    # Let's map exactly as required: 
    # 1. IDLE (idle)
    # 2. NORMAL (burst)
    # 3. CPU STRESS (cpu)
    # 4. RECOVERY (idle)
    # 5. MEMORY STRESS (memory)
    # 6. RECOVERY (idle)
    # 7. MIXED CPU+MEM (mixed)
    # 8. CPU BURST (burst)
    # 9. FINAL RECOVERY (idle)
]

def get_phases(test_mode=False):
    if test_mode:
        return [
            ("idle", 10, 256),
            ("burst", 10, 256),
            ("cpu", 10, 256),
            ("idle", 10, 256),
            ("memory", 10, 1024),
            ("idle", 10, 256),
            ("mixed", 10, 1024),
            ("burst", 10, 256),
            ("idle", 10, 256),
        ]
    return [
        ("idle", 300, 256),
        ("burst", 300, 256),
        ("cpu", 300, 256),
        ("idle", 300, 256),
        ("memory", 300, 1024),
        ("idle", 300, 256),
        ("mixed", 300, 1024),
        ("burst", 300, 256),
        ("idle", 300, 256),
    ]

_stop_event = threading.Event()

def run_collector(client, container_name, output_file, interval, mode, duration):
    collector = PilotCollector(client=client)
    start_time = time.time()
    
    # Wait for container to be ready
    container = None
    for _ in range(10):
        try:
            container = client.containers.get(container_name)
            if container.status == "running":
                break
        except docker.errors.NotFound:
            pass
        time.sleep(1)
        
    if not container:
        print(f"Container {container_name} not found or not running.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "a") as f:
        while not _stop_event.is_set() and (time.time() - start_time) < duration:
            try:
                # Reload container to check status
                container.reload()
                if container.status != "running":
                    break
                metrics = collector.collect(container.id, workload_mode=mode)
                if metrics:
                    f.write(json.dumps(metrics) + "\n")
                    f.flush()
            except Exception as e:
                print(f"Collection error: {e}")
            
            time.sleep(interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run shortened phases for testing")
    parser.add_argument("--output", default="data/prediction_experiment/real_workload_pilot.jsonl")
    parser.add_argument("--interval", type=int, default=5, help="Sampling interval")
    parser.add_argument("--image", default="neuroscale-workload:latest")
    args = parser.parse_args()

    client = docker.from_env()
    phases = get_phases(args.test_mode)
    
    print(f"Starting Prediction Data Pilot...")
    print(f"Output file: {args.output}")
    print(f"Total phases: {len(phases)}")
    
    container_name = "neuroscale-workload-pilot"

    for i, (mode, duration, memory) in enumerate(phases):
        print(f"\n--- Phase {i+1}/{len(phases)}: {mode.upper()} ---")
        print(f"Duration: {duration}s, Memory: {memory}MB")
        
        # Cleanup existing container
        try:
            c = client.containers.get(container_name)
            c.remove(force=True)
        except docker.errors.NotFound:
            pass
            
        # Run container
        print(f"Starting container...")
        container = client.containers.run(
            args.image,
            name=container_name,
            command=["--mode", mode, "--duration", str(duration), "--memory", str(memory)],
            detach=True
        )
        
        # Run collector in thread so we can precisely wait
        _stop_event.clear()
        col_thread = threading.Thread(
            target=run_collector,
            args=(client, container_name, args.output, args.interval, mode, duration)
        )
        col_thread.start()
        
        # Wait for phase duration
        time.sleep(duration)
        
        # Stop collector and container
        _stop_event.set()
        col_thread.join()
        
        try:
            container.remove(force=True)
        except:
            pass

    print("\nPilot complete.")

if __name__ == "__main__":
    main()
