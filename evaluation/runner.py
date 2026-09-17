"""Runner for simulation comparison."""

import time
import json
from .baselines import StaticController, ThresholdController, PredictionOnlyController
from .metrics import compute_metrics
from rl.environment import NeuroScaleEnv
from rl.config import EnvConfig
from rl.agent import DQNAgent, DQNAgentConfig
from rl.actions import DiscreteActionSpace
from rl.config import ActionConfig
import numpy as np

def run_simulation(strategy_name: str, trace, agent=None, action_space=None):
    env = NeuroScaleEnv(EnvConfig())
    env.load_trace(trace)
    
    state, _ = env.reset()
    done = False
    
    records = []
    
    static_ctrl = StaticController()
    thresh_ctrl = ThresholdController()
    pred_ctrl = PredictionOnlyController()
    
    current_cpu = 1.0
    current_mem = 256
    
    # We step the environment manually here instead of env.step() to apply custom baselines easily
    step_idx = 0
    while not done and step_idx < len(trace):
        raw_state = trace[step_idx]
        
        # Decide action
        if strategy_name == "Static":
            tgt_cpu, tgt_mem = static_ctrl.decide(raw_state)
        elif strategy_name == "Threshold":
            tgt_cpu, tgt_mem = thresh_ctrl.decide(raw_state, current_cpu, current_mem)
        elif strategy_name == "Prediction-only":
            tgt_cpu, tgt_mem = pred_ctrl.decide(raw_state)
        elif strategy_name == "NeuroScale":
            action_idx = agent.choose_action_index(state, evaluate=True)
            action_dict = action_space.get_action(action_idx)
            tgt_cpu = action_dict["cpu"]
            tgt_mem = action_dict["memory"]
        else:
            tgt_cpu, tgt_mem = current_cpu, current_mem
            
        current_cpu = tgt_cpu
        current_mem = tgt_mem
        
        # Apply step in env to get reward and next state
        # First map to closest discrete action for env.step()
        best_idx = 0
        if strategy_name != "NeuroScale":
            # approximate discrete index
            best_dist = float('inf')
            for i in range(action_space.n):
                act = action_space.get_action(i)
                dist = abs(act["cpu"] - tgt_cpu) + abs(act["memory"] - tgt_mem)/100.0
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        else:
            best_idx = action_idx
            
        next_state, reward, terminated, truncated, _ = env.step(best_idx)
        done = terminated or truncated
        
        record = {
            "timestamp": time.time(),
            "cycle_number": step_idx,
            "current_cpu": raw_state["cpu_util"],
            "current_memory": raw_state["mem_util"],
            "current_cpu_alloc": current_cpu,
            "current_mem_alloc": current_mem,
            "sla_violation": reward < -10, # rough approximation based on reward
            "reward": reward,
            "predicted_cpu": raw_state.get("pred_cpu", raw_state["cpu_util"]),
            "predicted_memory": raw_state.get("pred_mem", raw_state["mem_util"]),
        }
        records.append(record)
        state = next_state
        step_idx += 1
        
    return compute_metrics(records)

def run_evaluation_suite(trace_name: str, trace):
    agent = None
    action_space = DiscreteActionSpace(ActionConfig())
    try:
        agent = DQNAgent.load("checkpoints/best_dqn.pt")
    except:
        agent = DQNAgent(DQNAgentConfig())
        
    results = []
    
    for strategy in ["Static", "Threshold", "Prediction-only", "NeuroScale"]:
        metrics = run_simulation(strategy, trace, agent, action_space)
        metrics["scenario"] = trace_name
        metrics["strategy"] = strategy
        metrics["timestamp"] = time.time()
        results.append(metrics)
        
    return results
