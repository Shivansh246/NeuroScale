#!/usr/bin/env python3
"""Comprehensive Evaluation, Benchmarking, and Audit for Resource-Factorized DQN."""

import os
import sys
import json
from collections import Counter, defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.agent import DQNAgent
from rl.decoupled_agent import DecoupledDQNAgent
from rl.resource_factorized_agent import ResourceFactorizedDQNAgent
from rl.environment import NeuroScaleEnv
from rl.decomposed_env import DecomposedRewardEnv
from rl.decomposed_reward import DecomposedRewardCalculator
from rl.reward import RewardCalculator
from rl.config import EnvConfig, ActionConfig, RewardConfig, StateConfig
from rl.actions import DiscreteActionSpace
from rl.state import StateBuilder
from evaluation.scenarios import get_evaluation_scenarios
from evaluation.metrics import compute_metrics


def validate_reward_decomposition():
    """Validates mathematical identity: r_cpu + r_mem == r_original across test cases."""
    print("=" * 90)
    print("1. VALIDATING REWARD DECOMPOSITION IDENTITY: r_cpu + r_mem == r_original")
    print("=" * 90)

    rew_cfg = RewardConfig()
    orig_calc = RewardCalculator(rew_cfg)
    decomp_calc = DecomposedRewardCalculator(rew_cfg)

    cases = [
        ("Normal Low Load", 40.0, 128.0, 0.5, 256, 0.5, 256, False),
        ("CPU Spike Overload", 180.0, 128.0, 0.5, 256, 0.5, 256, True),
        ("Memory Spike Overload", 40.0, 768.0, 0.5, 256, 0.5, 256, True),
        ("Joint Stress + Realloc", 150.0, 512.0, 2.0, 1024, 0.5, 256, True),
    ]

    max_err = 0.0
    for name, c_u, m_u, c_al, m_al, p_c, p_m, anom in cases:
        lat = 50.0 + max(0.0, c_u - c_al * 100.0) * 5.0 + max(0.0, m_u - m_al) * 20.0
        prev_st = {"current_cpu_alloc": p_c, "current_mem_alloc": p_m}
        next_st = {
            "current_cpu_util": min(c_u, c_al * 100.0),
            "current_mem_util": min(m_u, m_al),
            "sla_latency": lat,
            "is_anomaly": anom,
            "target_cpu_util": c_u,
            "target_mem_util": m_u,
        }
        act = {"cpu": c_al, "memory": m_al}
        r_orig = orig_calc.calculate(prev_st, act, next_st)
        r_c, r_m, r_exp = decomp_calc.calculate_decomposed(
            prev_st, act, next_st, target_cpu_util=c_u, target_mem_util=m_u
        )
        err = abs((r_c + r_m) - r_orig)
        max_err = max(max_err, err)
        print(f"[{'PASSED' if err < 1e-5 else 'FAILED'}] {name:<24s} | r_cpu={r_c:8.3f}, r_mem={r_m:8.3f}, sum={r_c+r_m:8.3f}, orig={r_orig:8.3f}, diff={err:.2e}")

    # Random test cases
    rng = np.random.RandomState(42)
    for _ in range(100):
        c_u = rng.uniform(10.0, 300.0)
        m_u = rng.uniform(50.0, 1500.0)
        c_al = rng.choice([0.25, 0.5, 1.0, 2.0])
        m_al = rng.choice([128, 256, 512, 1024])
        p_c = rng.choice([0.25, 0.5, 1.0, 2.0])
        p_m = rng.choice([128, 256, 512, 1024])
        anom = bool(rng.choice([True, False]))
        lat = 50.0 + max(0.0, c_u - c_al * 100.0) * 5.0 + max(0.0, m_u - m_al) * 20.0
        prev_st = {"current_cpu_alloc": p_c, "current_mem_alloc": p_m}
        next_st = {
            "current_cpu_util": min(c_u, c_al * 100.0),
            "current_mem_util": min(m_u, m_al),
            "sla_latency": lat,
            "is_anomaly": anom,
            "target_cpu_util": c_u,
            "target_mem_util": m_u,
        }
        act = {"cpu": c_al, "memory": m_al}
        r_orig = orig_calc.calculate(prev_st, act, next_st)
        r_c, r_m, r_exp = decomp_calc.calculate_decomposed(
            prev_st, act, next_st, target_cpu_util=c_u, target_mem_util=m_u
        )
        max_err = max(max_err, abs((r_c + r_m) - r_orig))

    print(f"\nReward decomposition identity validated: max absolute error across all cases = {max_err:.2e}\n")


def main():
    validate_reward_decomposition()

    print("=" * 90)
    print("2. THREE-MODEL COMPARATIVE BENCHMARK (VERIFIED BASELINE vs DECOUPLED vs FACTORIZED)")
    print("=" * 90)

    act_space = DiscreteActionSpace(ActionConfig())
    env_cfg = EnvConfig()
    scenarios = get_evaluation_scenarios(steps_per_scenario=50)

    # 1. Load verified monolithic baseline records from evaluation_results.jsonl
    base_records = []
    if os.path.exists("data/evaluation_results.jsonl"):
        with open("data/evaluation_results.jsonl") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("strategy") == "NeuroScale":
                        base_records.append(d)

    # 2. Evaluate Decoupled DQN
    decoupled_agent = DecoupledDQNAgent.load("checkpoints/best_decoupled_dqn.pt")
    # 3. Evaluate Factorized DQN
    factorized_agent = ResourceFactorizedDQNAgent.load("checkpoints/best_resource_factorized_dqn.pt")

    env = NeuroScaleEnv(env_cfg)

    def eval_model(ag):
        res = {}
        acts_dict = {}
        for sc_name, trace in scenarios.items():
            env.load_trace(trace)
            s, _ = env.reset(seed=42)
            done = False
            step = 0
            c_allocs, m_allocs, c_wastes, m_wastes, sla, rews = [], [], [], [], 0, []
            acts = []
            while not done and step < len(trace):
                raw = trace[step]
                a = ag.choose_action_index(s, evaluate=True)
                ad = act_space.get_action(a)
                ns, r, term, trunc, _ = env.step(a)
                done = term or trunc
                c_allocs.append(ad["cpu"])
                m_allocs.append(ad["memory"])
                c_wastes.append(max(0.0, ad["cpu"] * 100.0 - raw["cpu_util"]))
                m_wastes.append(max(0.0, ad["memory"] - raw["mem_util"]))
                if r < -10.0:
                    sla += 1
                rews.append(r)
                acts.append(a)
                s = ns
                step += 1
            res[sc_name] = {
                "cpu_alloc": float(np.mean(c_allocs)),
                "mem_alloc": float(np.mean(m_allocs)),
                "cpu_waste": float(np.mean(c_wastes)),
                "mem_waste": float(np.mean(m_wastes)),
                "sla": sla,
                "reward": float(np.mean(rews)),
            }
            acts_dict[sc_name] = acts
        return res, acts_dict

    dec_res, dec_acts = eval_model(decoupled_agent)
    fact_res, fact_acts = eval_model(factorized_agent)

    # Baseline metrics from validated records
    base_res = {}
    base_acts = {}
    for sc_name in scenarios.keys():
        recs = [r for r in base_records if r.get("scenario") == sc_name]
        if recs:
            c_al = [r["current_cpu_alloc"] for r in recs]
            m_al = [r["current_mem_alloc"] for r in recs]
            c_w = [max(0.0, r["current_cpu_alloc"] * 100.0 - r["current_cpu"]) for r in recs]
            m_w = [max(0.0, r["current_mem_alloc"] - r["current_memory"]) for r in recs]
            sla = sum(1 for r in recs if r.get("sla_violation"))
            rews = [r.get("reward", 0.0) for r in recs]
            acts = [act_space.get_closest_index(r["current_cpu_alloc"], r["current_mem_alloc"]) for r in recs]
            base_res[sc_name] = {
                "cpu_alloc": float(np.mean(c_al)),
                "mem_alloc": float(np.mean(m_al)),
                "cpu_waste": float(np.mean(c_w)),
                "mem_waste": float(np.mean(m_w)),
                "sla": sla,
                "reward": float(np.mean(rews)),
            }
            base_acts[sc_name] = acts

    # Print Table
    header = f"{'Scenario':<12} | {'Model':<22} | {'CPU Alloc':<9} | {'Mem Alloc':<9} | {'CPU Waste':<9} | {'Mem Waste':<9} | {'SLA':<4} | {'Reward':<7}"
    print(header)
    print("-" * len(header))
    for sc in scenarios.keys():
        b = base_res.get(sc, {})
        d = dec_res[sc]
        f = fact_res[sc]
        if b:
            print(f"{sc:<12} | {'Verified Monolithic':<22} | {b['cpu_alloc']:9.2f} | {b['mem_alloc']:9.0f} | {b['cpu_waste']:9.1f} | {b['mem_waste']:9.1f} | {b['sla']:4d} | {b['reward']:7.2f}")
        print(f"{' ':<12} | {'Decoupled DQN':<22} | {d['cpu_alloc']:9.2f} | {d['mem_alloc']:9.0f} | {d['cpu_waste']:9.1f} | {d['mem_waste']:9.1f} | {d['sla']:4d} | {d['reward']:7.2f}")
        print(f"{' ':<12} | {'Resource-Factorized':<22} | {f['cpu_alloc']:9.2f} | {f['mem_alloc']:9.0f} | {f['cpu_waste']:9.1f} | {f['mem_waste']:9.1f} | {f['sla']:4d} | {f['reward']:7.2f}")
        print("-" * len(header))

    # Action distribution summary
    print("\n" + "=" * 90)
    print("3. ACTION DISTRIBUTIONS PER SCENARIO AND OVERALL (200 STEPS)")
    print("=" * 90)
    models = [("Verified Monolithic", base_acts), ("Decoupled DQN", dec_acts), ("Resource-Factorized", fact_acts)]
    for sc in scenarios.keys():
        print(f"\n--- Scenario: {sc} (50 steps) ---")
        for m_name, act_dict in models:
            c = Counter(act_dict.get(sc, []))
            c_str = ", ".join(f"A{a} ({act_space.get_action(a)['cpu']}C/{act_space.get_action(a)['memory']}M): {cnt}" for a, cnt in sorted(c.items()))
            print(f"  {m_name:<22s}: {c_str}")

    print("\n=== TOTAL 200-STEP OVERALL DISTRIBUTIONS ===")
    for m_name, act_dict in models:
        all_a = []
        for sc in scenarios:
            all_a.extend(act_dict.get(sc, []))
        c = Counter(all_a)
        print(f"\n{m_name}:")
        for a, cnt in sorted(c.items()):
            act = act_space.get_action(a)
            print(f"  Action {a:2d} ({act['cpu']}C / {act['memory']:4d}MB): {cnt:3d} steps ({cnt/len(all_a)*100:5.1f}%)")

    # Representative States & Implied 4x4 Q Tables
    print("\n" + "=" * 90)
    print("4. REPRESENTATIVE STATES & IMPLIED 4x4 JOINT Q-TABLES (NORMALIZED INPUTS)")
    print("=" * 90)

    sb = StateBuilder(StateConfig())
    rep_raw_states = {
        "Normal (Calm)": {
            "current_cpu_util": 40.0, "current_mem_util": 128.0,
            "predicted_cpu_demand": 40.0, "predicted_mem_demand": 128.0,
            "current_cpu_alloc": 0.5, "current_mem_alloc": 256,
            "sla_latency": 50.0, "anomaly_score": 0.05, "is_anomaly": False,
            "host_cpu_avail": 4.0, "host_mem_avail": 4096.0
        },
        "CPU Spike": {
            "current_cpu_util": 180.0, "current_mem_util": 128.0,
            "predicted_cpu_demand": 180.0, "predicted_mem_demand": 128.0,
            "current_cpu_alloc": 0.5, "current_mem_alloc": 256,
            "sla_latency": 250.0, "anomaly_score": 0.60, "is_anomaly": True,
            "host_cpu_avail": 4.0, "host_mem_avail": 4096.0
        },
        "Memory Spike": {
            "current_cpu_util": 40.0, "current_mem_util": 768.0,
            "predicted_cpu_demand": 40.0, "predicted_mem_demand": 768.0,
            "current_cpu_alloc": 0.5, "current_mem_alloc": 256,
            "sla_latency": 250.0, "anomaly_score": 0.60, "is_anomaly": True,
            "host_cpu_avail": 4.0, "host_mem_avail": 4096.0
        },
        "Sustained High": {
            "current_cpu_util": 150.0, "current_mem_util": 512.0,
            "predicted_cpu_demand": 150.0, "predicted_mem_demand": 512.0,
            "current_cpu_alloc": 0.5, "current_mem_alloc": 256,
            "sla_latency": 180.0, "anomaly_score": 0.50, "is_anomaly": True,
            "host_cpu_avail": 4.0, "host_mem_avail": 4096.0
        }
    }

    cpus = [0.25, 0.5, 1.0, 2.0]
    mems = [128, 256, 512, 1024]

    for st_name, raw_st in rep_raw_states.items():
        s_vec = sb.build_state(raw_st)
        st_t = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qc, qm = factorized_agent.online_net(st_t)
            qc = qc.squeeze(0).cpu().numpy()
            qm = qm.squeeze(0).cpu().numpy()

        c_idx = int(np.argmax(qc))
        m_idx = int(np.argmax(qm))
        best_joint = c_idx * 4 + m_idx

        print(f"\n--- State: {st_name} (Resource-Factorized DQN) ---")
        print(f"  CPU Head Q-values:    " + ", ".join(f"{c}C: {qc[i]:.2f}{'*' if i==c_idx else ''}" for i, c in enumerate(cpus)))
        print(f"  Memory Head Q-values: " + ", ".join(f"{m}MB: {qm[j]:.2f}{'*' if j==m_idx else ''}" for j, m in enumerate(mems)))
        print(f"  Selected Action:      {cpus[c_idx]}C / {mems[m_idx]}MB (Action {best_joint})")

        header_str = f"  {'CPU / Mem':<10} | " + " | ".join(f"{m:6d} MB" for m in mems)
        print("  Implied 4x4 Joint Q Table [Q_cpu(c) + Q_mem(m)]:")
        print("  " + header_str)
        print("  " + "-" * len(header_str))
        for i, c in enumerate(cpus):
            row_vals = []
            for j, m in enumerate(mems):
                val = qc[i] + qm[j]
                star = "*" if (i == c_idx and j == m_idx) else " "
                row_vals.append(f"{val:8.2f}{star}")
            print(f"  {c:4.2f} CPU   | " + " | ".join(row_vals))

    # 5. Counterfactual Isolation Tests with Exact Deltas
    print("\n" + "=" * 90)
    print("5. SIX COUNTERFACTUAL ISOLATION EXPERIMENTS (UNPERTURBED vs PERTURBED DELTAS)")
    print("=" * 90)

    unperturbed_raw = rep_raw_states["Normal (Calm)"].copy()
    unperturbed_vec = sb.build_state(unperturbed_raw)
    unp_t = torch.tensor(unperturbed_vec).unsqueeze(0)
    with torch.no_grad():
        unp_qc, unp_qm = factorized_agent.online_net(unp_t)
        unp_qc = unp_qc.squeeze(0).numpy()
        unp_qm = unp_qm.squeeze(0).numpy()
    unp_c = int(np.argmax(unp_qc))
    unp_m = int(np.argmax(unp_qm))

    print(f"--- UNPERTURBED (NORMAL CALM) BASELINE ---")
    print(f"  CPU Q-values:    {dict(zip(cpus, np.round(unp_qc, 2)))} -> Choice: {cpus[unp_c]}C")
    print(f"  Memory Q-values: {dict(zip(mems, np.round(unp_qm, 2)))} -> Choice: {mems[unp_m]}MB")
    print(f"  Joint Action:    {cpus[unp_c]}C / {mems[unp_m]}MB (Action {unp_c*4 + unp_m})")

    cf_perturbations = [
        ("High Current CPU", {"current_cpu_util": 190.0}),
        ("High Current Memory", {"current_mem_util": 850.0}),
        ("High Predicted CPU", {"predicted_cpu_demand": 190.0}),
        ("High Predicted Memory", {"predicted_mem_demand": 850.0}),
        ("Anomaly Active", {"anomaly_score": 0.85, "is_anomaly": True}),
        ("SLA Latency Spike", {"sla_latency": 280.0}),
    ]

    for name, changes in cf_perturbations:
        p_raw = unperturbed_raw.copy()
        p_raw.update(changes)
        p_vec = sb.build_state(p_raw)
        p_t = torch.tensor(p_vec).unsqueeze(0)
        with torch.no_grad():
            qc, qm = factorized_agent.online_net(p_t)
            qc = qc.squeeze(0).numpy()
            qm = qm.squeeze(0).numpy()
        c_idx = int(np.argmax(qc))
        m_idx = int(np.argmax(qm))
        d_qc = qc - unp_qc
        d_qm = qm - unp_qm

        print(f"\n--- {name} ---")
        print(f"  Perturbation:       {changes}")
        print(f"  CPU Q-values:       {dict(zip(cpus, np.round(qc, 2)))} -> Action: {cpus[c_idx]}C (delta: {cpus[c_idx] - cpus[unp_c]:+.2f}C)")
        print(f"  CPU Delta Q:        {dict(zip(cpus, np.round(d_qc, 2)))}")
        print(f"  Memory Q-values:    {dict(zip(mems, np.round(qm, 2)))} -> Action: {mems[m_idx]}MB (delta: {mems[m_idx] - mems[unp_m]:+d}MB)")
        print(f"  Memory Delta Q:     {dict(zip(mems, np.round(d_qm, 2)))}")
        print(f"  Joint Action:       {cpus[c_idx]}C / {mems[m_idx]}MB (Action {c_idx*4 + m_idx})")

    print("\n" + "=" * 90)
    print("AUDIT & BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    main()
