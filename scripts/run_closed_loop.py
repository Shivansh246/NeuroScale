#!/usr/bin/env python3
"""Run real NeuroScale closed-loop integration."""

import os
import time
import argparse
import subprocess
import json

from monitoring.collector import Collector
from controller.controller import Controller
from control.config import ControlLoopConfig
from control.loop import Orchestrator

# Models
from models.predictor import TransformerPredictor
from anomaly.detector import AutoencoderAnomalyDetector
from rl.agent import DQNAgent, DQNAgentConfig
from rl.actions import DiscreteActionSpace
from rl.state import StateBuilder
from rl.reward import RewardCalculator
from rl.config import ActionConfig, EnvConfig, StateConfig, RewardConfig


def run_container(image="neuroscale-workload", name="neuroscale-smoke", duration=30):
    print(f"Starting Docker container {name}...")
    subprocess.run([
        "docker", "run", "-d", "--name", name, "--rm",
        image, "--mode", "burst", "--duration", str(duration)
    ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
def stop_container(name="neuroscale-smoke"):
    print(f"Stopping Docker container {name}...")
    subprocess.run(["docker", "stop", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class StubPredictor:
    def predict(self, seq):
        return {"cpu": 50.0, "memory": 128.0, "confidence": 0.5}

class StubDetector:
    def score(self, seq):
        return {"score": 0.01, "is_anomaly": False}
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="neuroscale-smoke")
    parser.add_argument("--cycles", type=int, default=15)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    # Verify Docker is running
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: Docker daemon is not running or accessible.")
        return

    # Check for models
    use_simulation = False
    
    if os.path.exists("checkpoints/best_transformer.pt"):
        print("Loading Transformer...")
        predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")
    else:
        print("Transformer missing. Using Stub.")
        predictor = StubPredictor()
        use_simulation = True
        
    if os.path.exists("checkpoints/best_autoencoder.pt"):
        print("Loading Autoencoder...")
        detector = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")
    else:
        print("Autoencoder missing. Using Stub.")
        detector = StubDetector()
        use_simulation = True

    env_config = EnvConfig()
    
    if os.path.exists("checkpoints/best_dqn.pt"):
        print("Loading DQN...")
        agent = DQNAgent.load("checkpoints/best_dqn.pt")
        action_space = agent.action_space
    else:
        print("DQN missing. Initializing untrained Agent.")
        agent = DQNAgent(DQNAgentConfig())
        action_space = DiscreteActionSpace(ActionConfig())
        use_simulation = True
        
    print(f"Operating in {'SIMULATION' if use_simulation else 'REAL DOCKER'} model mode.")

    stop_container(args.container)
    run_container(name=args.container, duration=args.cycles * 2)
    time.sleep(2.0) # Let it start

    collector = Collector()
    controller = Controller()
    
    # Initialize initial resources just to be safe
    controller.apply(args.container, 1.0, 256)
    
    # Orchestrator
    config = ControlLoopConfig(
        window_size=4, # Small for smoke test
        warmup_cycles=4,
        simulation_mode=False # Still actually control Docker!
    )
    
    orchestrator = Orchestrator(
        config=config,
        collector=collector,
        predictor=predictor,
        detector=detector,
        agent=agent,
        controller=controller,
        action_space=action_space,
        state_builder=StateBuilder(StateConfig()),
        reward_calculator=RewardCalculator(RewardConfig())
    )

    print("\nStarting control loop...")
    print(f"{'Cycle':<6} | {'Status':<12} | {'CPU%':<8} | {'MemMB':<8} | {'ActCPU':<6} | {'ActMem':<6} | {'Success'}")
    print("-" * 70)
    
    for i in range(args.cycles):
        res = orchestrator.run_cycle(args.container, workload_mode="burst")
        
        status = res["status"]
        cpu = f"{res.get('current_cpu', 0.0):.1f}"
        mem = f"{res.get('current_memory', 0.0):.1f}"
        act_cpu = f"{res.get('selected_action_cpu', orchestrator.current_alloc_cpu):.2f}"
        act_mem = f"{res.get('selected_action_memory', orchestrator.current_alloc_mem)}"
        success = res.get('controller_success', '-')
        
        print(f"{i+1:<6} | {status:<12} | {cpu:<8} | {mem:<8} | {act_cpu:<6} | {act_mem:<6} | {success}")
        
        time.sleep(args.sleep)

    print("\nCleaning up...")
    stop_container(args.container)
    
    print(f"\nDone! Logs written to {config.log_file}")
    
    with open(config.log_file, "r") as f:
        lines = f.readlines()
        print(f"Total structured JSONL records: {len(lines)}")
        if len(lines) > 0:
            print("Sample record (latest):")
            print(json.dumps(json.loads(lines[-1]), indent=2))

if __name__ == '__main__':
    main()
