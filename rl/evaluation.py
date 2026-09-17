"""Evaluation procedures for the DQN Agent."""

from typing import Dict, Any, List
import numpy as np

from .agent import DQNAgent
from .environment import NeuroScaleEnv


def evaluate_agent(agent: DQNAgent, env: NeuroScaleEnv, traces: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Evaluates the agent on a dictionary of deterministic traces (scenarios).
    Returns a dictionary of mean rewards and selected action statistics.
    """
    results = {}
    
    for scenario_name, trace in traces.items():
        env.load_trace(trace)
        state, _ = env.reset(seed=42)
        
        episode_reward = 0.0
        actions_taken = []
        done = False
        
        while not done:
            # Always evaluate greedly
            action_idx = agent.choose_action_index(state, evaluate=True)
            actions_taken.append(action_idx)
            
            next_state, reward, terminated, truncated, _ = env.step(action_idx)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            
        results[scenario_name] = {
            "reward": episode_reward,
            "mean_reward": episode_reward / len(trace),
            "actions": actions_taken
        }
        
    return results

def get_synthetic_evaluation_scenarios() -> Dict[str, List[Dict[str, Any]]]:
    """Returns a set of synthetic scenarios for evaluation."""
    
    # 1. Normal Workload (low/steady)
    normal = [{"cpu_util": 40.0, "mem_util": 128} for _ in range(20)]
    
    # 2. CPU Spike
    cpu_spike = [{"cpu_util": 40.0, "mem_util": 128} for _ in range(5)] + \
                [{"cpu_util": 180.0, "mem_util": 128} for _ in range(5)] + \
                [{"cpu_util": 40.0, "mem_util": 128} for _ in range(10)]
                
    # 3. Memory Spike
    mem_spike = [{"cpu_util": 40.0, "mem_util": 128} for _ in range(5)] + \
                [{"cpu_util": 40.0, "mem_util": 768} for _ in range(5)] + \
                [{"cpu_util": 40.0, "mem_util": 128} for _ in range(10)]
                
    # 4. Sustained High (over-provision scenario tester)
    sustained = [{"cpu_util": 150.0, "mem_util": 512} for _ in range(20)]
    
    return {
        "normal": normal,
        "cpu_spike": cpu_spike,
        "mem_spike": mem_spike,
        "sustained": sustained
    }
