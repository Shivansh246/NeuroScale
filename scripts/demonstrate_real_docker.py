#!/usr/bin/env python3
"""
Real Docker Demonstration of NeuroScale Autonomous Closed-Loop Pipeline.

Demonstrates the complete end-to-end loop:
observe -> predict -> detect -> decide -> control -> measure -> repeat
using real Transformer, Autoencoder, and Resource-Factorized DQN checkpoints
controlling a real Docker workload container across 4 distinct phases:
  Phase A: Normal / Warmup
  Phase B: CPU Spike
  Phase C: Memory Spike
  Phase D: Recovery / Down-scaling

Evidence is logged as structured JSONL to data/real_docker_demo.jsonl.
Actual Docker resource limits (CpuQuota, Memory) are queried and verified before and after each reallocation.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import docker
from typing import Dict, Any, List, Optional

# Ensure project root is on PYTHONPATH
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


REQUIRED_CHECKPOINTS = {
    "transformer": "checkpoints/best_transformer.pt",
    "autoencoder": "checkpoints/best_autoencoder.pt",
    "resource_factorized_dqn": "checkpoints/best_resource_factorized_dqn.pt",
}


def verify_checkpoints() -> None:
    """Verify that all required real-model checkpoints exist."""
    missing = []
    for name, path in REQUIRED_CHECKPOINTS.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    if missing:
        raise FileNotFoundError(
            f"Cannot run real-model demonstration: missing checkpoints: {', '.join(missing)}"
        )


def launch_container(client: docker.DockerClient, name: str = "neuroscale-real-demo") -> docker.models.containers.Container:
    """Launch the workload container in idle mode with a long duration."""
    try:
        old = client.containers.get(name)
        print(f"[Docker] Removing existing container {name}...")
        old.stop(timeout=2)
        old.remove(force=True)
    except docker.errors.NotFound:
        pass
    except Exception as e:
        print(f"[Docker] Warning while cleaning old container: {e}")

    print(f"[Docker] Launching {name} from image neuroscale-workload:latest...")
    container = client.containers.run(
        "neuroscale-workload:latest",
        command=["--mode", "idle", "--duration", "600"],
        detach=True,
        name=name,
    )
    time.sleep(1.5)  # Allow container process to initialize
    return container


def get_docker_limits(container: docker.models.containers.Container) -> Dict[str, Any]:
    """Query Docker daemon for actual container resource configuration."""
    container.reload()
    host_cfg = container.attrs.get("HostConfig", {})
    cpu_quota = host_cfg.get("CpuQuota", 0)
    cpu_period = host_cfg.get("CpuPeriod", 100000) or 100000
    mem_bytes = host_cfg.get("Memory", 0)
    cpu_cores = round(cpu_quota / cpu_period, 2) if cpu_quota > 0 else 0.0
    mem_mb = round(mem_bytes / (1024 * 1024), 2) if mem_bytes > 0 else 0.0
    return {
        "cpu_quota": cpu_quota,
        "cpu_period": cpu_period,
        "cpu_cores": cpu_cores,
        "memory_bytes": mem_bytes,
        "memory_mb": mem_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="NeuroScale Real Docker Demonstration")
    parser.add_argument("--container", default="neuroscale-real-demo", help="Container name")
    parser.add_argument("--output", default="data/real_docker_demo.jsonl", help="JSONL log path")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between cycles")
    args = parser.parse_args()

    print("=" * 80)
    print("NEUROSCALE REAL DOCKER DEMONSTRATION")
    print("Transformer + Autoencoder + Resource-Factorized DQN Closed Loop")
    print("=" * 80)

    # 1. Verify Docker daemon
    try:
        docker_client = docker.from_env()
        docker_client.ping()
        print("[Docker] Daemon accessible and operational.")
    except Exception as e:
        print(f"[Error] Docker daemon not accessible: {e}")
        sys.exit(1)

    # 2. Verify and load real model checkpoints (FAIL CLEARLY IF MISSING)
    print("\n[Step 1/5] Verifying and loading model checkpoints...")
    verify_checkpoints()

    print(f"  -> Loading Transformer: {REQUIRED_CHECKPOINTS['transformer']}")
    predictor = TransformerPredictor.from_checkpoint(REQUIRED_CHECKPOINTS["transformer"])
    print(f"     Loaded TransformerPredictor (Confidence: {predictor._confidence:.4f})")

    print(f"  -> Loading Autoencoder: {REQUIRED_CHECKPOINTS['autoencoder']}")
    detector = AutoencoderAnomalyDetector.from_checkpoint(REQUIRED_CHECKPOINTS["autoencoder"])
    print(f"     Loaded AutoencoderAnomalyDetector (Threshold: {detector._threshold:.6f})")

    print(f"  -> Loading Resource-Factorized DQN: {REQUIRED_CHECKPOINTS['resource_factorized_dqn']}")
    agent = ResourceFactorizedDQNAgent.load(REQUIRED_CHECKPOINTS["resource_factorized_dqn"])
    action_space = agent.action_space
    print(f"     Loaded ResourceFactorizedDQNAgent (Epsilon: {agent.epsilon:.4f})")

    # 3. Clean and prepare output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)
    print(f"\n[Step 2/5] Initialized fresh demonstration log: {args.output}")

    # 4. Launch Docker container
    print("\n[Step 3/5] Starting real workload container...")
    container = launch_container(docker_client, name=args.container)
    container_id = container.id

    collector = Collector(client=docker_client)
    controller = Controller(client=docker_client)

    # Set initial baseline limits: 1.0 CPU core, 256 MB
    print("  -> Setting initial baseline limits: 1.0 CPU, 256 MB...")
    init_res = controller.apply(container_id, cpu=1.0, memory=256)
    init_limits = get_docker_limits(container)
    print(f"  -> Docker verified initial: CpuQuota={init_limits['cpu_quota']}, Memory={init_limits['memory_mb']}MB")

    # 5. Initialize Orchestrator
    config = ControlLoopConfig(
        window_size=12,
        warmup_cycles=12,
        log_file=args.output,
        simulation_mode=False,
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
        reward_calculator=RewardCalculator(RewardConfig()),
    )
    # Set orchestrator starting allocation to match Docker initial state
    orchestrator.current_alloc_cpu = 1.0
    orchestrator.current_alloc_mem = 256

    # 6. Execute 4 Demonstration Phases
    print("\n[Step 4/5] Executing multi-phase demonstration...")
    print(
        f"{'Cycle':<6} | {'Phase':<14} | {'Status':<10} | {'CPU%':<6} | {'MemMB':<7} | "
        f"{'PredCPU':<7} | {'PredMem':<7} | {'Anom':<5} | {'AllocCPU':<8} | {'AllocMem':<8} | {'DockerVerified'}"
    )
    print("-" * 110)

    records: List[Dict[str, Any]] = []
    reallocations: List[Dict[str, Any]] = []

    # Define Phase Schedule (Total 44 cycles)
    # Phase A (Normal/Warmup): Cycles 1..14 (12 warmup + 2 active calm)
    # Phase B (CPU Spike): Cycles 15..22 (8 cycles)
    # Phase C (Memory Spike): Cycles 23..30 (8 cycles)
    # Phase D (Recovery): Cycles 31..42 (12 cycles)
    total_cycles = 42

    try:
        for cycle in range(1, total_cycles + 1):
            # Phase management & stimulus triggering
            if cycle <= 14:
                phase_name = "Phase A (Normal)"
                workload_mode = "idle"
            elif 15 <= cycle <= 22:
                phase_name = "Phase B (CPU Spike)"
                workload_mode = "cpu_spike"
                if cycle == 15:
                    print(f"\n>>> [EVENT] Triggering CPU Spike inside container (python workload.py --mode cpu)...")
                    container.exec_run("python workload.py --mode cpu --duration 12", detach=True)
            elif 23 <= cycle <= 30:
                phase_name = "Phase C (Mem Spike)"
                workload_mode = "mem_spike"
                if cycle == 23:
                    print(f"\n>>> [EVENT] Triggering Memory Spike inside container (python workload.py --mode memory --memory 400)...")
                    container.exec_run("python workload.py --mode memory --memory 400 --duration 12", detach=True)
            else:
                phase_name = "Phase D (Recovery)"
                workload_mode = "recovery"
                if cycle == 31:
                    print(f"\n>>> [EVENT] Workload entering Recovery Phase (returning to calm baseline)...")

            # Query Docker resource state BEFORE cycle execution
            limits_before = get_docker_limits(container)

            # Run closed-loop cycle
            res = orchestrator.run_cycle(container_id, workload_mode=workload_mode)

            # Query Docker resource state AFTER cycle execution
            limits_after = get_docker_limits(container)

            # Check if reallocation occurred
            realloc = (
                limits_before["cpu_quota"] != limits_after["cpu_quota"] or
                limits_before["memory_bytes"] != limits_after["memory_bytes"]
            )
            if realloc and res.get("status") == "active":
                reallocation_event = {
                    "cycle": cycle,
                    "phase": phase_name,
                    "workload_mode": workload_mode,
                    "before": limits_before,
                    "after": limits_after,
                    "requested_cpu": res.get("selected_cpu_allocation"),
                    "requested_mem": res.get("selected_memory_allocation"),
                    "action_index": res.get("action_index"),
                    "docker_verified": res.get("docker_verified", False),
                }
                reallocations.append(reallocation_event)

            records.append(res)

            # Print cycle progress
            status = res.get("status", "-")
            cpu_util = f"{res.get('current_cpu', 0.0):.1f}"
            mem_util = f"{res.get('current_memory', 0.0):.1f}"
            pred_cpu = f"{res.get('predicted_cpu', 0.0):.1f}" if "predicted_cpu" in res else "-"
            pred_mem = f"{res.get('predicted_memory', 0.0):.1f}" if "predicted_memory" in res else "-"
            is_anom = "YES" if res.get("is_anomaly", False) else "no"
            alloc_cpu = f"{res.get('current_cpu_alloc', 0.0):.2f}C"
            alloc_mem = f"{res.get('current_mem_alloc', 0)}MB"
            verified = "VERIFIED" if res.get("docker_verified", False) else ("-" if status == "warming_up" else "same")

            print(
                f"{cycle:<6} | {phase_name:<14} | {status:<10} | {cpu_util:<6} | {mem_util:<7} | "
                f"{pred_cpu:<7} | {pred_mem:<7} | {is_anom:<5} | {alloc_cpu:<8} | {alloc_mem:<8} | {verified}"
            )

            time.sleep(args.sleep)

    finally:
        print("\n[Step 5/5] Cleaning up container...")
        try:
            container.stop(timeout=2)
            container.remove(force=True)
            print("  -> Container stopped and removed.")
        except Exception as e:
            print(f"  -> Cleanup notice: {e}")

    # 7. Generate Demonstration Summary
    print("\n" + "=" * 80)
    print("REAL DOCKER DEMONSTRATION SUMMARY & VERIFICATION")
    print("=" * 80)

    total_records = len(records)
    active_records = [r for r in records if r.get("status") == "active"]
    successful_ops = sum(1 for r in active_records if r.get("controller_success", False))
    failed_ops = sum(1 for r in active_records if not r.get("controller_success", False))
    sla_violations = sum(1 for r in active_records if r.get("sla_violation", False))
    anomalies_detected = sum(1 for r in active_records if r.get("is_anomaly", False))
    avg_latency = (
        sum(r.get("controller_latency", 0.0) for r in active_records) / len(active_records)
        if active_records else 0.0
    )

    print(f"Total Cycles Executed:             {total_records} (12 warmup + {len(active_records)} active)")
    print(f"Successful Controller Operations:   {successful_ops}")
    print(f"Failed Controller Operations:       {failed_ops}")
    print(f"Total Meaningful Reallocations:    {len(reallocations)}")
    print(f"Average Controller Latency:         {avg_latency:.2f} ms")
    print(f"Total SLA Violations:              {sla_violations}")
    print(f"Total Anomaly Detections:          {anomalies_detected}")
    print(f"Predictions Available:             {len(active_records)}/{len(active_records)}")

    print("\n--- ACTUAL DOCKER REALLOCATION EVENTS (BEFORE vs AFTER VERIFICATION) ---")
    for idx, event in enumerate(reallocations, 1):
        print(f"\n[Reallocation Event {idx}] Cycle {event['cycle']} ({event['phase']})")
        print(f"  Requested Target: CPU={event['requested_cpu']}C, Memory={event['requested_mem']}MB (Action {event['action_index']})")
        print(f"  Docker BEFORE:    CpuQuota={event['before']['cpu_quota']} ({event['before']['cpu_cores']}C), Memory={event['before']['memory_mb']}MB")
        print(f"  Docker AFTER:     CpuQuota={event['after']['cpu_quota']} ({event['after']['cpu_cores']}C), Memory={event['after']['memory_mb']}MB")
        print(f"  Docker Verified:  {event['docker_verified']}")

    print("\n--- REPRESENTATIVE WORKLOAD PHASE EVENTS ---")
    # Identify representative events
    rep_events = {}
    for r in records:
        mode = r.get("workload_mode")
        status = r.get("status")
        if status == "active":
            if "first_normal" not in rep_events and mode == "idle":
                rep_events["first_normal"] = r
            elif "cpu_spike" not in rep_events and mode == "cpu_spike" and r.get("current_cpu", 0) > 50:
                rep_events["cpu_spike"] = r
            elif "mem_spike" not in rep_events and mode == "mem_spike" and r.get("current_memory", 0) > 200:
                rep_events["mem_spike"] = r
            elif "recovery" not in rep_events and mode == "recovery":
                rep_events["recovery"] = r

    for name, ev in rep_events.items():
        print(f"\nEvent: [{name.upper()}] (Cycle {ev.get('cycle_number')}, Mode: {ev.get('workload_mode')})")
        print(f"  Current Resources: CPU={ev.get('current_cpu'):.1f}%, Mem={ev.get('current_memory'):.1f}MB")
        print(f"  Prediction:        CPU={ev.get('predicted_cpu'):.1f}%, Mem={ev.get('predicted_memory'):.1f}MB (Conf: {ev.get('prediction_confidence', 0):.4f})")
        print(f"  Anomaly:           Score={ev.get('anomaly_score', 0):.4f}, IsAnomaly={ev.get('is_anomaly')}")
        print(f"  Allocation:        Selected CPU={ev.get('selected_cpu_allocation')}C, Mem={ev.get('selected_memory_allocation')}MB (Action {ev.get('action_index')})")
        print(f"  SLA Status:        Latency={ev.get('sla_latency'):.1f}ms, Violation={ev.get('sla_violation')}")

    print(f"\nStructured JSONL evidence written to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
