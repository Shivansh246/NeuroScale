#!/usr/bin/env python3
"""Evaluate and compare Existing 16-action DQN vs. Decoupled DQN."""

import os
import sys
import json
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.scenarios import get_evaluation_scenarios
from evaluation.runner import load_evaluation_models, run_simulation
from rl.agent import DQNAgent
from rl.decoupled_agent import DecoupledDQNAgent
from rl.actions import DiscreteActionSpace
from rl.config import EnvConfig, ActionConfig, StateConfig
from rl.state import StateBuilder
import torch
import numpy as np


def main():
    print("=" * 90)
    print("CONTROLLED EXPERIMENT: 16-ACTION DQN VS DECOUPLED CPU/MEMORY DQN")
    print("=" * 90)

    # 1. Load models
    predictor, detector, _ = load_evaluation_models()
    dqn_16 = DQNAgent.load("checkpoints/best_dqn.pt")
    decoupled_dqn = DecoupledDQNAgent.load("checkpoints/best_decoupled_dqn.pt")

    scenarios = get_evaluation_scenarios(steps_per_scenario=50)
    act_space = DiscreteActionSpace(ActionConfig())
    env_cfg = EnvConfig()

    agents = {
        "16-Action DQN": dqn_16,
        "Decoupled DQN": decoupled_dqn,
    }

    all_summaries = []
    all_step_records = defaultdict(lambda: defaultdict(list))

    for sc_name, trace in scenarios.items():
        for ag_name, ag in agents.items():
            metrics, records = run_simulation(
                strategy_name="NeuroScale",
                trace=trace,
                predictor=predictor,
                detector=detector,
                agent=ag,
                action_space=act_space,
                env_config=env_cfg,
                scenario_name=sc_name,
            )
            metrics["agent"] = ag_name
            metrics["scenario"] = sc_name
            all_summaries.append(metrics)
            all_step_records[sc_name][ag_name] = records

    # Print scenario comparison table
    header = (
        f"{'Scenario':<12} | {'Agent':<16} | {'CPU Alloc':<9} | {'Mem Alloc':<9} | "
        f"{'CPU Waste':<9} | {'Mem Waste':<9} | {'SLA Vio':<7} | {'Reward':<8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['scenario']:<12} | {s['agent']:<16} | {s['avg_cpu_allocation']:<9.2f} | "
            f"{s['avg_memory_allocation']:<9.0f} | {s['cpu_wastage']:<9.1f} | "
            f"{s['memory_wastage']:<9.1f} | {s['sla_violations']:<7d} | {s['reward']:<8.2f}"
        )

    # Detailed action distributions and reallocations
    print("\n" + "=" * 90)
    print("ACTION DISTRIBUTIONS & REALLOCATIONS PER SCENARIO")
    print("=" * 90)

    for sc_name in scenarios.keys():
        print(f"\n--- Scenario: {sc_name} (50 steps) ---")
        for ag_name in agents.keys():
            recs = all_step_records[sc_name][ag_name]
            actions = [act_space.get_closest_index(r["cpu_allocation"], r["memory_allocation"]) for r in recs]
            action_counts = Counter(actions)
            action_str = ", ".join(
                f"A{a} ({act_space.get_action(a)['cpu']}C/{act_space.get_action(a)['memory']}M): {cnt}"
                for a, cnt in sorted(action_counts.items())
            )
            # Reallocations
            reallocs = 0
            for i in range(1, len(recs)):
                if (recs[i]["cpu_allocation"] != recs[i-1]["cpu_allocation"] or
                    recs[i]["memory_allocation"] != recs[i-1]["memory_allocation"]):
                    reallocs += 1
            print(f"  {ag_name:<16s} | Reallocs: {reallocs:2d} | Actions: {action_str}")

    # Total 200-step distribution
    print("\n" + "=" * 90)
    print("TOTAL 200-STEP OVERALL ACTION DISTRIBUTION")
    print("=" * 90)
    for ag_name in agents.keys():
        total_counts = Counter()
        cpu_counts = Counter()
        mem_counts = Counter()
        for sc_name in scenarios.keys():
            recs = all_step_records[sc_name][ag_name]
            for r in recs:
                idx = act_space.get_closest_index(r["cpu_allocation"], r["memory_allocation"])
                total_counts[idx] += 1
                cpu_counts[r["cpu_allocation"]] += 1
                mem_counts[r["memory_allocation"]] += 1
        print(f"\n{ag_name}:")
        for a in range(16):
            if total_counts[a] > 0:
                act = act_space.get_action(a)
                pct = total_counts[a] / 200.0 * 100.0
                print(f"  Action {a:2d} ({act['cpu']} CPU / {act['memory']:4d} MB): {total_counts[a]:3d} steps ({pct:5.1f}%)")
        print("  CPU breakdown: " + ", ".join(f"{c}C: {cnt} ({cnt/2:.1f}%)" for c, cnt in sorted(cpu_counts.items())))
        print("  Mem breakdown: " + ", ".join(f"{m}MB: {cnt} ({cnt/2:.1f}%)" for m, cnt in sorted(mem_counts.items())))


if __name__ == '__main__':
    main()
