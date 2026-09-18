#!/usr/bin/env python3
"""
Isolated Stimulus Experiment on Live Docker Container.

Tests whether the live Docker closed loop demonstrates partial factorized behavior:
Test A: CPU-only stress (CPU spikes, memory stays ~11MB)
        -> Observe CPU-head branch index vs Memory-head branch index
Test B: Memory-only stress (Memory spikes to ~400MB, CPU stays ~0%)
        -> Observe Memory-head branch index vs CPU-head branch index

Logs output to data/real_docker_isolated_demo.jsonl.
"""

import os
import sys
import time
import json
import argparse
import docker
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.collector import Collector
from controller.controller import Controller
from control.config import ControlLoopConfig
from control.loop import Orchestrator
from models.predictor import TransformerPredictor
from anomaly.detector import AutoencoderAnomalyDetector
from rl.resource_factorized_agent import ResourceFactorizedDQNAgent
from rl.state import StateBuilder
from rl.reward import RewardCalculator
from rl.config import StateConfig, RewardConfig


def run_isolated_test(test_name: str, mode: str, cmd: str, cycles: int = 18) -> List[Dict[str, Any]]:
    print(f"\n--- Running Isolated {test_name} ---")
    docker_client = docker.from_env()
    container_name = f"neuroscale-iso-{mode}"
    
    try:
        old = docker_client.containers.get(container_name)
        old.stop(timeout=2)
        old.remove(force=True)
    except Exception:
        pass

    container = docker_client.containers.run(
        "neuroscale-workload:latest",
        command=["--mode", "idle", "--duration", "300"],
        detach=True,
        name=container_name,
    )
    time.sleep(1.5)

    collector = Collector(client=docker_client)
    controller = Controller(client=docker_client)
    controller.apply(container.id, 1.0, 256)

    predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")
    detector = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")
    agent = ResourceFactorizedDQNAgent.load("checkpoints/best_resource_factorized_dqn.pt")

    log_path = f"data/isolated_{mode}.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    config = ControlLoopConfig(
        window_size=12,
        warmup_cycles=12,
        log_file=log_path,
        simulation_mode=False,
    )
    orchestrator = Orchestrator(
        config=config,
        collector=collector,
        predictor=predictor,
        detector=detector,
        agent=agent,
        controller=controller,
        action_space=agent.action_space,
        state_builder=StateBuilder(StateConfig()),
        reward_calculator=RewardCalculator(RewardConfig()),
    )
    orchestrator.current_alloc_cpu = 1.0
    orchestrator.current_alloc_mem = 256

    records = []
    try:
        for c in range(1, cycles + 1):
            if c == 13:
                print(f"  >>> Injecting stimulus: {cmd}")
                container.exec_run(cmd, detach=True)

            res = orchestrator.run_cycle(container.id, workload_mode=mode if c >= 13 else "idle")
            records.append(res)
            
            status = res.get("status", "-")
            cpu = res.get("current_cpu", 0.0)
            mem = res.get("current_memory", 0.0)
            c_idx = res.get("cpu_action_index", "-")
            m_idx = res.get("mem_action_index", "-")
            act = res.get("action_index", "-")
            c_alloc = res.get("selected_cpu_allocation", "-")
            m_alloc = res.get("selected_memory_allocation", "-")
            
            print(f"  Cycle {c:2d} | Status: {status:<10} | CPU: {cpu:5.1f}% | Mem: {mem:6.1f}MB | CPU Branch: {c_idx} ({c_alloc}C) | Mem Branch: {m_idx} ({m_alloc}MB) | Joint Action: {act}")
            time.sleep(1.0)
    finally:
        try:
            container.stop(timeout=2)
            container.remove(force=True)
        except Exception:
            pass

    return records


def main():
    print("=" * 80)
    print("NEUROSCALE ISOLATED STIMULUS EXPERIMENTS")
    print("=" * 80)

    records_cpu = run_isolated_test("Test A: CPU-Only Stress", "cpu", "python workload.py --mode cpu --duration 6", cycles=18)
    records_mem = run_isolated_test("Test B: Memory-Only Stress", "mem", "python workload.py --mode memory --memory 400 --duration 6", cycles=18)

    # Save combined
    os.makedirs("data", exist_ok=True)
    combined_path = "data/real_docker_isolated_demo.jsonl"
    with open(combined_path, "w") as f:
        for r in records_cpu + records_mem:
            f.write(json.dumps(r) + "\n")

    print(f"\nIsolated stimulus demonstration log saved to: {combined_path}")


if __name__ == "__main__":
    main()
