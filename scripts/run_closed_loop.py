#!/usr/bin/env python3
"""Run real NeuroScale closed-loop integration."""

import os
import time
import argparse
import subprocess
import json
import sys

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monitoring.collector import Collector
from controller.controller import Controller
from control.config import ControlLoopConfig
from control.loop import Orchestrator

# Models
from models.predictor import TransformerPredictor
from anomaly.detector import AutoencoderAnomalyDetector
from rl.agent import DQNAgent, DQNAgentConfig
from rl.resource_factorized_agent import ResourceFactorizedDQNAgent, ResourceFactorizedAgentConfig
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
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--mode", choices=["simulation", "real-model"], default="simulation", help="Model execution mode")
    parser.add_argument("--predictor", choices=["production", "real"], default="production", help="Which transformer predictor to use")
    parser.add_argument("--log-file", default="data/control_loop.jsonl", help="Output JSONL log path")
    args = parser.parse_args()

    # Verify Docker is running
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: Docker daemon is not running or accessible.")
        return

    # Check for models
    use_simulation = (args.mode == "simulation")

    transformer_checkpoint = "checkpoints/best_transformer.pt"
    if args.predictor == "real":
        transformer_checkpoint = "checkpoints/best_transformer_real.pt"

    missing_checkpoints = []
    required_cps = [
        transformer_checkpoint,
        "checkpoints/best_autoencoder.pt",
        "checkpoints/best_resource_factorized_dqn.pt"
    ]
    for cp in required_cps:
        if not os.path.exists(cp):
            missing_checkpoints.append(cp)

    if args.mode == "real-model" and missing_checkpoints:
        print(f"Error: --mode real-model specified but checkpoints are missing: {missing_checkpoints}")
        sys.exit(1)

    if use_simulation:
        print("Operating in SIMULATION model mode (using stubs).")
        predictor = StubPredictor()
        detector = StubDetector()
        agent = DQNAgent(DQNAgentConfig())
        action_space = DiscreteActionSpace(ActionConfig())
    else:
        print("Operating in REAL-MODEL mode with Resource-Factorized DQN.")
        print(f"Loading Transformer ({args.predictor} predictor)...")
        predictor = TransformerPredictor.from_checkpoint(transformer_checkpoint)
        print("Loading Autoencoder...")
        detector = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")
        print("Loading Resource-Factorized DQN...")
        agent = ResourceFactorizedDQNAgent.load("checkpoints/best_resource_factorized_dqn.pt")
        action_space = agent.action_space

    stop_container(args.container)
    run_container(name=args.container, duration=max(args.cycles * 5, 120))
    time.sleep(2.0) # Let it start

    collector = Collector()
    controller = Controller()

    # Initialize initial resources just to be safe
    controller.apply(args.container, 1.0, 256)

    # Orchestrator
    config = ControlLoopConfig(
        window_size=12,
        warmup_cycles=12,
        simulation_mode=use_simulation,
        log_file=args.log_file,
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
