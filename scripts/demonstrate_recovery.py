#!/usr/bin/env python3
"""
Dedicated Recovery Demonstration of NeuroScale Autonomous Closed-Loop Pipeline.

Goal: Test whether the live system down-scales after stress terminates and the
12-step rolling feature window clears.

Sequence:
1. Start normal.
2. Build 12-step warmup window.
3. Inject CPU stress.
4. Allow the system to scale up.
5. Terminate the stress.
6. Continue for 20+ cycles so the 12-step rolling window clears completely.
7. Observe whether the policy selects a lower allocation.
8. Verify whether Docker limits actually decrease.

Logs output to data/real_docker_recovery_demo.jsonl.
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
    parser = argparse.ArgumentParser(description="NeuroScale Recovery Demonstration")
    parser.add_argument("--container", default="neuroscale-recovery-demo")
    parser.add_argument("--output", default="data/real_docker_recovery_demo.jsonl")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 80)
    print("NEUROSCALE DEDICATED RECOVERY DEMONSTRATION")
    print("Testing Down-Scaling Behavior After Stress Clearance")
    print("=" * 80)

    docker_client = docker.from_env()

    # Load real models without stubs
    predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")
    detector = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")
    agent = ResourceFactorizedDQNAgent.load("checkpoints/best_resource_factorized_dqn.pt")
    action_space = agent.action_space

    # Clean previous output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    # Launch container
    try:
        old = docker_client.containers.get(args.container)
        old.stop(timeout=2)
        old.remove(force=True)
    except Exception:
        pass

    print(f"[Docker] Starting container {args.container}...")
    container = docker_client.containers.run(
        "neuroscale-workload:latest",
        command=["--mode", "idle", "--duration", "600"],
        detach=True,
        name=args.container,
    )
    time.sleep(1.5)

    collector = Collector(client=docker_client)
    controller = Controller(client=docker_client)

    # Set initial baseline: 1.0 CPU, 256 MB
    controller.apply(container.id, 1.0, 256)
    init_limits = get_docker_limits(container)
    print(f"[Docker] Initial baseline: CpuQuota={init_limits['cpu_quota']} ({init_limits['cpu_cores']}C), Memory={init_limits['memory_mb']}MB")

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
    orchestrator.current_alloc_cpu = 1.0
    orchestrator.current_alloc_mem = 256

    print("\nExecuting recovery protocol (38 cycles total):")
    print("  Cycles  1..12: Warmup (normal idle load)")
    print("  Cycles 13..18: CPU Stress injection (6 cycles)")
    print("  Cycles 19..38: Recovery observation (20 cycles to clear 12-step window)")
    print("-" * 110)
    print(
        f"{'Cycle':<6} | {'Phase':<14} | {'Status':<10} | {'CPU%':<6} | {'MemMB':<7} | "
        f"{'PredCPU':<7} | {'PredMem':<7} | {'Anom':<5} | {'AllocCPU':<8} | {'AllocMem':<8} | {'DockerVerified'}"
    )
    print("-" * 110)

    records: List[Dict[str, Any]] = []
    reallocations: List[Dict[str, Any]] = []

    try:
        for cycle in range(1, 39):
            if cycle <= 12:
                phase_name = "Warmup"
                mode = "idle"
            elif 13 <= cycle <= 18:
                phase_name = "CPU Stress"
                mode = "cpu_stress"
                if cycle == 13:
                    print(f"\n>>> [EVENT] Injecting CPU Stress (duration 6s)...")
                    container.exec_run("python workload.py --mode cpu --duration 6", detach=True)
            else:
                phase_name = "Recovery"
                mode = "recovery"
                if cycle == 19:
                    print(f"\n>>> [EVENT] CPU Stress ended. Entering 20-cycle Recovery Phase...")

            limits_before = get_docker_limits(container)
            res = orchestrator.run_cycle(container.id, workload_mode=mode)
            limits_after = get_docker_limits(container)

            realloc = (
                limits_before["cpu_quota"] != limits_after["cpu_quota"] or
                limits_before["memory_bytes"] != limits_after["memory_bytes"]
            )
            if realloc and res.get("status") == "active":
                event = {
                    "cycle": cycle,
                    "phase": phase_name,
                    "before": limits_before,
                    "after": limits_after,
                    "requested_cpu": res.get("selected_cpu_allocation"),
                    "requested_mem": res.get("selected_memory_allocation"),
                    "action_index": res.get("action_index"),
                    "docker_verified": res.get("docker_verified"),
                }
                reallocations.append(event)

            records.append(res)

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
        try:
            container.stop(timeout=2)
            container.remove(force=True)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("RECOVERY DEMONSTRATION SUMMARY")
    print("=" * 80)
    print(f"Total Cycles: {len(records)} (11 warmup + 27 active)")
    print(f"Reallocations Observed: {len(reallocations)}")
    for idx, ev in enumerate(reallocations, 1):
        print(f"  [{idx}] Cycle {ev['cycle']} ({ev['phase']}):")
        print(f"      Before: CpuQuota={ev['before']['cpu_quota']} ({ev['before']['cpu_cores']}C), Mem={ev['before']['memory_mb']}MB")
        print(f"      After:  CpuQuota={ev['after']['cpu_quota']} ({ev['after']['cpu_cores']}C), Mem={ev['after']['memory_mb']}MB")
        print(f"      Requested Action: {ev['action_index']} -> {ev['requested_cpu']}C / {ev['requested_mem']}MB")

    # Downscale analysis
    downscale_events = [
        ev for ev in reallocations
        if ev['after']['cpu_quota'] < ev['before']['cpu_quota'] or
           ev['after']['memory_bytes'] < ev['before']['memory_bytes']
    ]
    print(f"\nDownscale Events Observed: {len(downscale_events)}")
    if downscale_events:
        for ds in downscale_events:
            print(f"  Downscale at Cycle {ds['cycle']}: {ds['before']['cpu_cores']}C -> {ds['after']['cpu_cores']}C, {ds['before']['memory_mb']}MB -> {ds['after']['memory_mb']}MB")
    else:
        print("  NO down-scaling occurred. The system maintained its elevated allocation through recovery.")

    print(f"\nRecovery demonstration log written to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
