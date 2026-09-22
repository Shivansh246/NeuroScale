import argparse
import time
import docker
import json
import os
import threading
import signal

from experiments.prediction_data.collector import PilotCollector

PHASE_CYCLE = [
    ("idle", 200, 256),
    ("cpu", 200, 256),
    ("idle", 200, 256),
    ("memory", 200, 1024),
    ("idle", 200, 256),
    ("burst", 200, 256),
    ("idle", 200, 256),
    ("mixed", 200, 1024),
    ("idle", 200, 256),
]

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

def run_collector_exact(client, container_id, output_file, interval, target_samples):
    collector = PilotCollector(client=client)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    samples = 0
    with open(output_file, "a") as f:
        while not _stop_event.is_set() and samples < target_samples:
            start_t = time.monotonic()
            
            mode = get_mode()
            try:
                metrics = collector.collect(container_id, workload_mode=mode)
                if metrics:
                    f.write(json.dumps(metrics) + "\n")
                    f.flush()
                    samples += 1
            except Exception as e:
                print(f"Collection error: {e}")
                
            if samples >= target_samples:
                print(f"\nCollector reached target {target_samples} samples.")
                break
                
            elapsed = time.monotonic() - start_t
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)
            
    # Signal main thread to stop
    _stop_event.set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/prediction_experiment/real_workload_long_5000.jsonl")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--image", default="neuroscale-workload:latest")
    args = parser.parse_args()

    client = docker.from_env()
    
    container_name = "neuroscale-workload-long"

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
        target=run_collector_exact,
        args=(client, container.id, args.output, args.interval, args.target)
    )
    col_thread.start()
    
    cycle_idx = 0
    try:
        while not _stop_event.is_set():
            for i, (mode, duration, memory) in enumerate(PHASE_CYCLE):
                if _stop_event.is_set():
                    break
                    
                print(f"\n--- Cycle {cycle_idx+1}, Phase {i+1}/{len(PHASE_CYCLE)}: {mode.upper()} ---")
                print(f"Duration: {duration}s, Memory: {memory}MB")
                
                set_mode(mode)
                
                if mode == "idle":
                    for _ in range(duration):
                        if _stop_event.is_set():
                            break
                        time.sleep(1)
                else:
                    cmd = f"python workload.py --mode {mode} --duration {duration} --memory {memory}"
                    print(f"Running exec: {cmd}")
                    # exec_run is blocking but we can detach it or use a detached exec and wait
                    # To be able to interrupt, let's use a detached exec and poll
                    exec_instance = client.api.exec_create(container.id, cmd)
                    exec_id = exec_instance['Id']
                    client.api.exec_start(exec_id, detach=True)
                    
                    # wait for it to finish or for _stop_event
                    while True:
                        if _stop_event.is_set():
                            break
                        exec_info = client.api.exec_inspect(exec_id)
                        if not exec_info['Running']:
                            break
                        time.sleep(1)
                        
                # Verify container is still alive
                container.reload()
                if container.status != "running":
                    print("ERROR: Persistent container died!")
                    _stop_event.set()
                    break
                    
            cycle_idx += 1
                
    finally:
        print("\nStopping collector and cleaning up container...")
        _stop_event.set()
        col_thread.join()
        container.remove(force=True)
        print("Done.")

if __name__ == "__main__":
    main()
