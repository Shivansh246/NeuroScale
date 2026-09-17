"""Training loop for DQN Agent."""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from .agent import DQNAgent, DQNAgentConfig
from .environment import NeuroScaleEnv
from .config import EnvConfig
from .evaluation import evaluate_agent, get_synthetic_evaluation_scenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNTrainer:
    """Trains the DQN Agent."""
    
    def __init__(self, agent_config: DQNAgentConfig, env_config: EnvConfig):
        self.agent = DQNAgent(agent_config)
        self.env = NeuroScaleEnv(env_config)
        self.eval_scenarios = get_synthetic_evaluation_scenarios()
        
        self.best_eval_reward = float('-inf')
        self.checkpoint_path = "checkpoints/best_dqn.pt"
        
    def generate_random_trace(self, length: int = 50) -> List[Dict[str, Any]]:
        """Generate a random synthetic trace for training."""
        trace = []
        cpu_base = np.random.uniform(20.0, 100.0)
        mem_base = np.random.choice([128, 256, 512])
        
        for _ in range(length):
            # random walk
            cpu_base = np.clip(cpu_base + np.random.normal(0, 10.0), 10.0, 300.0)
            mem_base = np.clip(mem_base + np.random.normal(0, 50.0), 64, 1024)
            
            trace.append({
                "cpu_util": cpu_base,
                "mem_util": int(mem_base),
                "pred_cpu": cpu_base, # perfect prediction for simplicity
                "pred_mem": int(mem_base),
                "anomaly_score": 0.0,
                "is_anomaly": False
            })
        return trace
        
    def train(self, num_episodes: int = 200, max_steps_per_episode: int = 50, eval_freq: int = 10) -> Dict[str, Any]:
        """Run the DQN training loop."""
        
        history = {
            "episode_rewards": [],
            "losses": [],
            "eval_rewards": []
        }
        
        for ep in range(1, num_episodes + 1):
            trace = self.generate_random_trace(max_steps_per_episode)
            self.env.load_trace(trace)
            state, _ = self.env.reset()
            
            ep_reward = 0.0
            ep_losses = []
            
            done = False
            while not done:
                action = self.agent.choose_action_index(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                
                loss = self.agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                    
            history["episode_rewards"].append(ep_reward)
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0
            history["losses"].append(avg_loss)
            
            # Evaluate periodically
            if ep % eval_freq == 0 or ep == num_episodes:
                eval_results = evaluate_agent(self.agent, self.env, self.eval_scenarios)
                mean_eval_reward = np.mean([res["mean_reward"] for res in eval_results.values()])
                history["eval_rewards"].append(mean_eval_reward)
                
                logger.info(f"Ep {ep}/{num_episodes} | Train Rew: {ep_reward:.2f} | Eval Rew: {mean_eval_reward:.2f} | Eps: {self.agent.epsilon:.3f}")
                
                if mean_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = mean_eval_reward
                    self.agent.save(self.checkpoint_path)
                    logger.info(f"Saved new best model with Eval Rew: {mean_eval_reward:.2f}")
                    
        return history
