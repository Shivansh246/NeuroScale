#!/usr/bin/env python3
import os
import sys
import json
from collections import Counter, defaultdict
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.scenarios import get_evaluation_scenarios
from evaluation.runner import load_evaluation_models, run_simulation
from rl.agent import DQNAgent
from rl.actions import DiscreteActionSpace
from rl.config import EnvConfig, ActionConfig, StateConfig
from rl.state import StateBuilder
from rl.decomposed_reward import DecomposedRewardCalculator
from rl.reward import RewardCalculator


def main():
    print("=" * 90)
    print("COLLECTING METRICS FOR REWARD-AWARE DQN EXPERIMENT")
    print("=" * 90)

    # 1. Load models
    predictor, detector, _ = load_evaluation_models()
    dqn_baseline = DQNAgent.load("checkpoints/best_dqn.pt")
    dqn_exp = DQNAgent.load("checkpoints/best_reward_aware_dqn.pt")

    scenarios = get_evaluation_scenarios(steps_per_scenario=50)
    act_space = DiscreteActionSpace(ActionConfig())
    env_cfg = EnvConfig()

    agents = {
        "Baseline DQN": dqn_baseline,
        "Reward-Aware DQN": dqn_exp,
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
        f"{'Scenario':<12} | {'Agent':<18} | {'CPU Alloc':<9} | {'Mem Alloc':<9} | "
        f"{'CPU Waste':<9} | {'Mem Waste':<9} | {'SLA Vio':<7} | {'Reward':<8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['scenario']:<12} | {s['agent']:<18} | {s['avg_cpu_allocation']:<9.2f} | "
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
            reallocs = 0
            for i in range(1, len(recs)):
                if (recs[i]["cpu_allocation"] != recs[i-1]["cpu_allocation"] or
                    recs[i]["memory_allocation"] != recs[i-1]["memory_allocation"]):
                    reallocs += 1
            print(f"  {ag_name:<18s} | Reallocs: {reallocs:2d} | Actions: {action_str}")

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

    # 4 Representative States
    rep_states = {
        "Normal": np.array([10.0, 100.0, 0.25, 256.0, 10.0, 100.0, 0.0, 0.0, 50.0, 10.0, 100.0], dtype=np.float32),
        "CPU Spike": np.array([90.0, 100.0, 0.25, 256.0, 95.0, 100.0, 1.0, 0.5, 250.0, 90.0, 100.0], dtype=np.float32),
        "Memory Spike": np.array([10.0, 900.0, 0.25, 256.0, 10.0, 950.0, 1.0, 0.5, 250.0, 10.0, 900.0], dtype=np.float32),
        "Sustained": np.array([85.0, 800.0, 0.5, 512.0, 85.0, 800.0, 0.0, 0.0, 180.0, 85.0, 800.0], dtype=np.float32),
    }

    print("\n" + "=" * 90)
    print("4x4 Q-VALUE TABLES FOR REPRESENTATIVE STATES")
    print("=" * 90)

    for st_name, s_vec in rep_states.items():
        print(f"\n--- State: {st_name} ---")
        st_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            q_base = dqn_baseline.online_net(st_tensor).squeeze(0).cpu().numpy()
            q_exp = dqn_exp.online_net(st_tensor).squeeze(0).cpu().numpy()

        for label, q_vals in [("Baseline DQN", q_base), ("Reward-Aware DQN", q_exp)]:
            best_a = int(np.argmax(q_vals))
            print(f"\n  {label} (Best: A{best_a} - {act_space.get_action(best_a)}):")
            # 4x4 table: rows CPU [0.25, 0.5, 1.0, 2.0], cols Mem [128, 256, 512, 1024]
            cpus = [0.25, 0.5, 1.0, 2.0]
            mems = [128, 256, 512, 1024]
            header = f"    {'CPU / Mem':<10} | " + " | ".join(f"{m:4d} MB" for m in mems)
            print(header)
            print("    " + "-" * len(header))
            for i, c in enumerate(cpus):
                row_vals = []
                for j, m in enumerate(mems):
                    a_idx = i * 4 + j
                    val = q_vals[a_idx]
                    star = "*" if a_idx == best_a else " "
                    row_vals.append(f"{val:7.2f}{star}")
                print(f"    {c:4.2f} CPU   | " + " | ".join(row_vals))

    # Counterfactual Tests (Cases 1-6)
    print("\n" + "=" * 90)
    print("COUNTERFACTUAL TESTS (CASES 1-6)")
    print("=" * 90)

    cf_cases = {
        "Case 1: Normal (calm state)": np.array([10.0, 100.0, 0.25, 256.0, 10.0, 100.0, 0.0, 0.0, 50.0, 10.0, 100.0], dtype=np.float32),
        "Case 2: CPU spike only (CPU 90, Mem 100)": np.array([90.0, 100.0, 0.25, 256.0, 95.0, 100.0, 1.0, 0.5, 250.0, 90.0, 100.0], dtype=np.float32),
        "Case 3: Memory spike only (CPU 10, Mem 900)": np.array([10.0, 900.0, 0.25, 256.0, 10.0, 950.0, 1.0, 0.5, 250.0, 10.0, 900.0], dtype=np.float32),
        "Case 4: Joint CPU + Memory stress": np.array([90.0, 900.0, 0.25, 256.0, 95.0, 950.0, 1.0, 0.8, 300.0, 90.0, 900.0], dtype=np.float32),
        "Case 5: High current load, low prediction": np.array([90.0, 900.0, 0.25, 256.0, 15.0, 150.0, 0.0, 0.0, 100.0, 90.0, 900.0], dtype=np.float32),
        "Case 6: Low current load, high prediction": np.array([15.0, 150.0, 0.25, 256.0, 90.0, 900.0, 1.0, 0.7, 50.0, 15.0, 150.0], dtype=np.float32),
    }

    for c_name, vec in cf_cases.items():
        st_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qb = dqn_baseline.online_net(st_tensor).squeeze(0).cpu().numpy()
            qe = dqn_exp.online_net(st_tensor).squeeze(0).cpu().numpy()
        ab = int(np.argmax(qb))
        ae = int(np.argmax(qe))
        act_b = act_space.get_action(ab)
        act_e = act_space.get_action(ae)
        print(f"\n{c_name}:")
        print(f"  Baseline DQN:     Action {ab:2d} ({act_b['cpu']} CPU / {act_b['memory']:4d} MB)  Q={qb[ab]:.2f}")
        print(f"  Reward-Aware DQN: Action {ae:2d} ({act_e['cpu']} CPU / {act_e['memory']:4d} MB)  Q={qe[ae]:.2f}")

    # Reward Accounting Validation: 3 Hand-checkable cases
    print("\n" + "=" * 90)
    print("REWARD ACCOUNTING VALIDATION: 3 HAND-CHECKABLE CASES")
    print("=" * 90)

    from rl.config import RewardConfig
    rew_cfg = RewardConfig()
    orig_calc = RewardCalculator(rew_cfg)
    decomp_calc = DecomposedRewardCalculator(rew_cfg)

    test_cases = [
        ("CPU-Heavy Stress", {
            "current_cpu": 150.0, "current_memory": 200.0, "target_cpu": 150.0, "target_memory": 200.0,
            "cpu_alloc": 0.5, "mem_alloc": 256.0, "prev_cpu_alloc": 0.5, "prev_mem_alloc": 256.0,
            "is_anomaly": False, "step_count": 10
        }),
        ("Memory-Heavy Stress", {
            "current_cpu": 20.0, "current_memory": 800.0, "target_cpu": 20.0, "target_memory": 800.0,
            "cpu_alloc": 0.25, "mem_alloc": 256.0, "prev_cpu_alloc": 0.25, "prev_mem_alloc": 256.0,
            "is_anomaly": False, "step_count": 10
        }),
        ("Joint CPU+Memory Stress with Reallocation & Anomaly", {
            "current_cpu": 150.0, "current_memory": 800.0, "target_cpu": 150.0, "target_memory": 800.0,
            "cpu_alloc": 1.0, "mem_alloc": 512.0, "prev_cpu_alloc": 0.5, "prev_mem_alloc": 256.0,
            "is_anomaly": True, "step_count": 10
        }),
    ]

    for name, kw in test_cases:
        s_cpu = min(kw["current_cpu"], kw["prev_cpu_alloc"] * 100.0)
        s_mem = min(kw["current_memory"], kw["prev_mem_alloc"])
        ns_cpu = min(kw["target_cpu"], kw["cpu_alloc"] * 100.0)
        ns_mem = min(kw["target_memory"], kw["mem_alloc"])
        
        lat_cpu = max(0.0, kw["target_cpu"] - kw["cpu_alloc"] * 100.0) * 5.0
        lat_mem = max(0.0, kw["target_memory"] - kw["mem_alloc"]) * 20.0
        latency = 50.0 + lat_cpu + lat_mem

        prev_state = {
            "current_cpu_alloc": kw["prev_cpu_alloc"],
            "current_mem_alloc": kw["prev_mem_alloc"],
        }
        next_state = {
            "current_cpu_util": ns_cpu,
            "current_mem_util": ns_mem,
            "sla_latency": latency,
            "is_anomaly": kw["is_anomaly"],
            "target_cpu_util": kw["target_cpu"],
            "target_mem_util": kw["target_memory"],
        }
        action = {"cpu": kw["cpu_alloc"], "memory": kw["mem_alloc"]}

        r_orig = orig_calc.calculate(prev_state, action, next_state)
        r_cpu, r_mem, r_exp = decomp_calc.calculate_decomposed(
            prev_state, action, next_state,
            target_cpu_util=kw["target_cpu"],
            target_mem_util=kw["target_memory"]
        )

        print(f"\n--- Case: {name} ---")
        print(f"  Target: CPU {kw['target_cpu']}%, Mem {kw['target_memory']}MB | Alloc: CPU {kw['cpu_alloc']}, Mem {kw['mem_alloc']}MB")
        print(f"  Latency: Total {latency:.1f}ms (CPU deficit: {max(0.0, kw['target_cpu'] - kw['cpu_alloc']*100):.1f}%, Mem deficit: {max(0.0, kw['target_memory'] - kw['mem_alloc']):.1f}MB)")
        print(f"  r_cpu:            {r_cpu:.4f}")
        print(f"  r_mem:            {r_mem:.4f}")
        print(f"  r_exp (sum):      {r_exp:.4f}")
        print(f"  r_original:       {r_orig:.4f}")
        print(f"  Difference:       {abs(r_exp - r_orig):.6e}")
        assert abs(r_exp - r_orig) < 1e-5, f"Mismatch: r_exp={r_exp}, r_orig={r_orig}"

if __name__ == '__main__':
    main()
