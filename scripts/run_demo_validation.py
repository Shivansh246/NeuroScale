import time
import subprocess
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from control.config import ControlLoopConfig
from control.loop import Orchestrator
from controller.controller import Controller
from rl.state import StateBuilder, StateConfig
from rl.actions import DiscreteActionSpace, ActionConfig
from rl.resource_factorized_agent import ResourceFactorizedDQNAgent
from models.predictor import TransformerPredictor
from anomaly.detector import AutoencoderAnomalyDetector
from monitoring.collector import Collector

def stop_container(name):
    subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_container(name):
    subprocess.run([
        "docker", "run", "-d", "--name", name,
        "--cpu-period=100000", "--cpu-quota=100000", "--memory=256m",
        "--entrypoint", "tail",
        "neuroscale-workload", "-f", "/dev/null"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait for container to come up
    time.sleep(2)
    
class WorkloadManager:
    def __init__(self, container_name):
        self.container_name = container_name
        self.current_pid = None
        
    def get_workload_pids(self):
        # We can't use ps inside the container. We rely on tracking self.current_pid.
        # But we can check if the python process exists using /proc.
        if self.current_pid:
            result = subprocess.run(
                ["docker", "exec", self.container_name, "sh", "-c", f"test -d /proc/{self.current_pid} && ! grep -q 'State:.*Z (zombie)' /proc/{self.current_pid}/status && echo YES || echo NO"],
                capture_output=True, text=True
            )
            if "YES" in result.stdout:
                return [self.current_pid]
        return []

    def launch(self, mode):
        self.terminate()
        # Launch using sh -c and capture PID
        cmd = f"nohup python3 workload.py --mode {mode} --duration 3600 > /dev/null 2>&1 & echo $!"
        result = subprocess.run(
            ["docker", "exec", self.container_name, "sh", "-c", cmd],
            capture_output=True, text=True
        )
        pid_str = result.stdout.strip().split('\n')[-1]
        try:
            self.current_pid = int(pid_str)
        except ValueError:
            print(f"WARNING: Could not parse PID from: {result.stdout}")
            
    def terminate(self):
        if self.current_pid:
            subprocess.run(["docker", "exec", self.container_name, "sh", "-c", f"kill -9 {self.current_pid}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for exit
            time.sleep(1)
            remaining = self.get_workload_pids()
            if remaining:
                print(f"WARNING: Workload process {remaining} survived kill -9")
        self.current_pid = None

def main():
    container_name = "neuroscale-smoke"
    log_file = "experiments/closed_loop_evaluation/results/phase12a_validation.jsonl"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Clean file
    if os.path.exists(log_file):
        os.remove(log_file)
        
    stop_container(container_name)
    run_container(container_name)
    
    # 1. Setup components
    cfg = ControlLoopConfig()
    cfg.log_file = log_file
    cfg.warmup_cycles = 12
    
    collector = Collector()
    predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer_real.pt")
    detector = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")
    agent = ResourceFactorizedDQNAgent.load("checkpoints/best_resource_factorized_dqn.pt")
    controller = Controller()
    action_space = DiscreteActionSpace(ActionConfig())
    state_builder = StateBuilder(StateConfig())
    
    orchestrator = Orchestrator(cfg, collector, predictor, detector, agent, controller, action_space, state_builder, None)
    orchestrator.reward_calc = type("MockReward", (), {"calculate": lambda self, prev_state, action, next_state: 0.0})()
    
    container_id = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", container_name],
        capture_output=True, text=True
    ).stdout.strip()
    
    print(f"Container ID: {container_id}")
    
    wl_manager = WorkloadManager(container_name)
    
    phases = [
        {"mode": "idle", "duration_cycles": 2},
        {"mode": "cpu", "duration_cycles": 2}
    ]
    
    cycle_length = 5.0
    
    for p_idx, phase in enumerate(phases):
        mode = phase["mode"]
        n_cycles = phase["duration_cycles"]
        
        # Verify no overlap before launching
        wl_manager.terminate()
        pids_before = wl_manager.get_workload_pids()
        assert len(pids_before) == 0, f"Found surviving processes: {pids_before}"
        
        wl_manager.launch(mode)
        pids_after = wl_manager.get_workload_pids()
        
        # Idle mode has 1 process (workload.py). cpu mode has workload + 1 child process (total 2?). 
        # Actually workload.py multiprocessing might spawn children.
        # But our terminate kills them if they match '[w]orkload.py'
        print(f"--- Transitioned to {mode}. PIDs before: {pids_before}, PIDs after: {pids_after} ---")
        
        for i in range(n_cycles):
            t_start = time.time()
            record = orchestrator.run_cycle(container_id, mode)
            
            # Print diagnostic for analysis


            if record.get("status") == "active":
                print("-" * 80)
                print(f"CYCLE {record['cycle_number']}")
                print(f"Workload Mode      : {mode}")
                print(f"CPU %              : {record.get('current_cpu', 0):.2f}%")
                print(f"Memory MB          : {record.get('current_memory_mb', 0):.2f} MB")
                print(f"Raw CPU ns_delta   : {record.get('cpu_usage_ns_delta', 0)}")
                print(f"Trans. CPU_delta   : {record.get('transformer_cpu_delta', 0):.4f}")
                print(f"Predicted CPU      : {record.get('predicted_cpu', 0):.2f}%")
                print(f"Predicted Memory   : {record.get('predicted_memory', 0):.2f} MB")
                print(f"Anomaly Score      : {record.get('anomaly_score', 0):.4f}")
                print(f"Anomaly Flag       : {record.get('is_anomaly', False)}")
                
                # DQN Action
                print(f"DQN Action Index   : {record.get('action_index', 'N/A')}")
                print(f"DQN Target CPU     : {record.get('selected_action_cpu', 0):.2f} C")
                print(f"DQN Target Memory  : {record.get('selected_action_memory', 0):.2f} MB")
                
                # Docker verification
                dbefore = record.get("docker_before", {})
                dafter = record.get("docker_after", {})
                
                print("Docker Limits Before:")
                print(f"  CPU Quota : {dbefore.get('cpu_quota')}")
                print(f"  Mem Bytes : {dbefore.get('memory_bytes')}")
                print("Docker Limits After:")
                print(f"  CPU Quota : {dafter.get('cpu_quota')}")
                print(f"  Mem Bytes : {dafter.get('memory_bytes')}")
                print(f"Docker Change Verified : {record.get('docker_verified', False)}")
                print("-" * 80)
                print(f"CYCLE {record['cycle_number']}")
                print(f"Workload Mode      : {mode}")
                print(f"CPU %              : {record.get('current_cpu', 0):.2f}%")
                print(f"Memory MB          : {record.get('current_memory_mb', 0):.2f} MB")
                print(f"Raw CPU ns_delta   : {record.get('cpu_usage_ns_delta', 0)}")
                print(f"Trans. CPU_delta   : {record.get('transformer_cpu_delta', 0):.4f}")
                print(f"Predicted CPU      : {record.get('predicted_cpu', 0):.2f}%")
                print(f"Predicted Memory   : {record.get('predicted_memory', 0):.2f} MB")
                print(f"Anomaly Score      : {record.get('anomaly_score', 0):.4f}")
                print(f"Anomaly Flag       : {record.get('is_anomaly', False)}")
                
                # DQN Action
                print(f"DQN Action Index   : {record.get('action_index', 'N/A')}")
                print(f"DQN Target CPU     : {record.get('selected_action_cpu', 0):.2f} C")
                print(f"DQN Target Memory  : {record.get('selected_action_memory', 0):.2f} MB")
                
                # Docker verification
                dbefore = record.get("docker_before", {})
                dafter = record.get("docker_after", {})
                
                print("Docker Limits Before:")
                print(f"  CPU Quota : {dbefore.get('cpu_quota')} / {dbefore.get('cpu_period')}")
                print(f"  Mem Bytes : {dbefore.get('memory_bytes')}")
                print("Docker Limits After:")
                print(f"  CPU Quota : {dafter.get('cpu_quota')} / {dafter.get('cpu_period')}")
                print(f"  Mem Bytes : {dafter.get('memory_bytes')}")
                print(f"Docker Change Verified : {record.get('docker_verified', False)}")
                print("-" * 80)
            t_elapsed = time.time() - t_start
            t_sleep = cycle_length - t_elapsed
            if t_sleep > 0:
                time.sleep(t_sleep)

    wl_manager.terminate()
    stop_container(container_name)
    print("Done!")

if __name__ == "__main__":
    main()
