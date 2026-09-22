import argparse
import time
import docker
import json
import os
import threading
import signal

from experiments.prediction_data.collector import PilotCollector

PHASES = [
    ("idle", 180, 256),
    ("cpu", 180, 256),
    ("idle", 180, 256),
    ("memory", 180, 1024),
    ("idle", 180, 256),
    ("burst", 180, 256),
    ("idle", 180, 256),
    ("mixed", 180, 1024),
    ("idle", 180, 256),
    ("cpu", 180, 256),
    ("idle", 180, 256),
    ("memory", 180, 1024),
    ("idle", 180, 256),
    ("idle", 180, 256), # total 14 phases
]

def get_phases(test_mode=False):
    if test_mode:
        return [
            ("idle", 10, 256),
            ("cpu", 10, 256),
            ("idle", 10, 256),
            ("memory", 10, 1024),
            ("idle", 10, 256),
            ("burst", 10, 256),
            ("idle", 10, 256),
            ("mixed", 10, 1024),
            ("idle", 10, 256),
        ]
    return PHASES

_stop_event = threading.Event()
_current_mode = "idle"
_current_mode_lock = threading.Lock()

def set_mode(mode):
    global _current_mode
    with _current_mode_lock:
        _current_mode = mode

def get_mode():
    with _current_mode_lock:
        return _current_mode

def run_collector(client, container_id, output_file, interval):
    collector = PilotCollector(client=client)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "a") as f:
        while not _stop_event.is_set():
            start_t = time.monotonic()
            
            mode = get_mode()
            try:
                metrics = collector.collect(container_id, workload_mode=mode)
                if metrics:
                    f.write(json.dumps(metrics) + "\n")
                    f.flush()
            except Exception as e:
                print(f"Collection error: {e}")
                
            elapsed = time.monotonic() - start_t
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--output", default="data/prediction_experiment/real_workload_continuous_pilot.jsonl")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--image", default="neuroscale-workload:latest")
    args = parser.parse_args()

    client = docker.from_env()
    phases = get_phases(args.test_mode)
    
    container_name = "neuroscale-workload-continuous"

    # Cleanup existing container just in case
    try:
        c = client.containers.get(container_name)
        c.remove(force=True)
    except docker.errors.NotFound:
        pass
        
    print(f"Starting persistent container {container_name}...")
    container = client.containers.run(
        args.image,
        name=container_name,
        entrypoint=["tail", "-f", "/dev/null"],
        detach=True
    )
    
    # Ensure cleanup on interrupt
    def cleanup(signum, frame):
        print("\nInterrupt received, cleaning up...")
        _stop_event.set()
        try:
            container.remove(force=True)
        except:
            pass
        os._exit(1)
        
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"Container ID: {container.id}")
    
    col_thread = threading.Thread(
        target=run_collector,
        args=(client, container.id, args.output, args.interval)
    )
    col_thread.start()
    
    try:
        for i, (mode, duration, memory) in enumerate(phases):
            print(f"\n--- Phase {i+1}/{len(phases)}: {mode.upper()} ---")
            print(f"Duration: {duration}s, Memory: {memory}MB")
            
            set_mode(mode)
            
            if mode == "idle":
                # Ensure no other workload processes are running
                time.sleep(duration)
            else:
                cmd = f"python workload.py --mode {mode} --duration {duration} --memory {memory}"
                print(f"Running exec: {cmd}")
                # exec_run is blocking
                exit_code, output = container.exec_run(cmd, detach=False)
                if exit_code != 0:
                    print(f"Workload command failed with code {exit_code}: {output}")
                    
            # Verify container is still alive
            container.reload()
            if container.status != "running":
                print("ERROR: Persistent container died!")
                break
                
    finally:
        print("\nStopping collector and cleaning up container...")
        _stop_event.set()
        col_thread.join()
        container.remove(force=True)
        print("Done.")

if __name__ == "__main__":
    main()
