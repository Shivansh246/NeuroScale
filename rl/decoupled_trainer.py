"""Training loop for Decoupled (Action Branching) DQN Agent."""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from .decoupled_agent import DecoupledDQNAgent, DecoupledDQNAgentConfig
from .environment import NeuroScaleEnv
from .config import EnvConfig
from .evaluation import evaluate_agent, get_synthetic_evaluation_scenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecoupledDQNTrainer:
    """Trains the Decoupled DQN Agent."""

    def __init__(self, agent_config: DecoupledDQNAgentConfig, env_config: EnvConfig):
        self.agent = DecoupledDQNAgent(agent_config)
        self.env = NeuroScaleEnv(env_config)
        self.eval_scenarios = get_synthetic_evaluation_scenarios()

        self.best_eval_reward = float('-inf')
        self.checkpoint_path = "checkpoints/best_decoupled_dqn.pt"

    def generate_balanced_trace(
        self,
        episode: int,
        length: int = 50,
        regime: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate a deterministic, balanced synthetic trace for training."""
        regimes = ["normal", "cpu_spike", "mem_spike", "sustained"]
        selected_regime = regime if regime in regimes else regimes[episode % len(regimes)]

        rng = np.random.RandomState(42 + episode * 100)
        trace = []

        warmup_len = max(5, int(length * 0.3))
        spike_len = max(5, int(length * 0.3))
        recovery_len = max(0, length - warmup_len - spike_len)

        for step_idx in range(length):
            if selected_regime == "normal":
                cpu = float(np.clip(rng.normal(40.0, 5.0), 15.0, 75.0))
                mem = float(np.clip(rng.normal(128.0, 15.0), 64.0, 250.0))
                is_anomaly = False
                anomaly_score = float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.2))
            elif selected_regime == "cpu_spike":
                if warmup_len <= step_idx < warmup_len + spike_len:
                    cpu = float(np.clip(rng.normal(180.0, 15.0), 120.0, 260.0))
                    mem = float(np.clip(rng.normal(128.0, 15.0), 64.0, 250.0))
                    is_anomaly = True
                    anomaly_score = float(np.clip(rng.normal(0.5, 0.05), 0.35, 0.8))
                else:
                    cpu = float(np.clip(rng.normal(40.0, 5.0), 15.0, 75.0))
                    mem = float(np.clip(rng.normal(128.0, 15.0), 64.0, 250.0))
                    is_anomaly = False
                    anomaly_score = float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.2))
            elif selected_regime == "mem_spike":
                if warmup_len <= step_idx < warmup_len + spike_len:
                    cpu = float(np.clip(rng.normal(40.0, 5.0), 15.0, 75.0))
                    mem = float(np.clip(rng.normal(768.0, 40.0), 550.0, 950.0))
                    is_anomaly = True
                    anomaly_score = float(np.clip(rng.normal(0.5, 0.05), 0.35, 0.8))
                else:
                    cpu = float(np.clip(rng.normal(40.0, 5.0), 15.0, 75.0))
                    mem = float(np.clip(rng.normal(128.0, 15.0), 64.0, 250.0))
                    is_anomaly = False
                    anomaly_score = float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.2))
            else:  # sustained
                if step_idx < warmup_len:
                    cpu = float(np.clip(rng.normal(40.0, 5.0), 15.0, 75.0))
                    mem = float(np.clip(rng.normal(128.0, 15.0), 64.0, 250.0))
                    is_anomaly = False
                    anomaly_score = float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.2))
                else:
                    cpu = float(np.clip(rng.normal(150.0, 15.0), 110.0, 220.0))
                    mem = float(np.clip(rng.normal(512.0, 30.0), 380.0, 650.0))
                    is_anomaly = True
                    anomaly_score = float(np.clip(rng.normal(0.45, 0.05), 0.3, 0.7))

            pred_cpu = float(np.clip(cpu + rng.normal(0.0, 3.0), 10.0, 300.0))
            pred_mem = float(np.clip(mem + rng.normal(0.0, 10.0), 50.0, 1024.0))

            trace.append({
                "cpu_util": round(cpu, 2),
                "mem_util": int(round(mem)),
                "pred_cpu": round(pred_cpu, 2),
                "pred_mem": int(round(pred_mem)),
                "anomaly_score": round(anomaly_score, 3),
                "is_anomaly": is_anomaly,
                "regime": selected_regime,
            })

        return trace

    def train(self, num_episodes: int = 100, max_steps_per_episode: int = 50, eval_freq: int = 10) -> Dict[str, Any]:
        """Run the Decoupled DQN training loop."""
        history = {
            "episode_rewards": [],
            "losses": [],
            "eval_rewards": [],
            "eval_actions": []
        }

        for ep in range(1, num_episodes + 1):
            trace = self.generate_balanced_trace(episode=ep, length=max_steps_per_episode)
            self.env.load_trace(trace)
            state, _ = self.env.reset()

            ep_reward = 0.0
            ep_losses = []

            done = False
            while not done:
                # Decoupled agent returns joint action index for environment interaction
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

                mean_cpus = []
                mean_mems = []
                all_actions = []
                for res in eval_results.values():
                    actions = res.get("actions", [])
                    all_actions.extend(actions)
                    for a in actions:
                        act = self.agent.action_space.get_action(a)
                        mean_cpus.append(act["cpu"])
                        mean_mems.append(act["memory"])
                mean_cpu = np.mean(mean_cpus) if mean_cpus else 0.0
                mean_mem = np.mean(mean_mems) if mean_mems else 0.0

                action_counts = {a: all_actions.count(a) for a in range(self.agent.action_space.n)}
                history["eval_actions"].append(action_counts)

                action_dist_str = ", ".join(
                    f"A{a}:{count}" for a, count in sorted(action_counts.items()) if count > 0
                )

                logger.info(
                    f"Ep {ep:3d}/{num_episodes} | Train Rew: {ep_reward:7.2f} | Eval Rew: {mean_eval_reward:6.2f} | "
                    f"Eps: {self.agent.epsilon:.3f} | Action: {mean_cpu:.2f}C/{mean_mem:.0f}MB | "
                    f"Actions: [{action_dist_str}]"
                )

                if mean_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = mean_eval_reward
                    self.agent.save(self.checkpoint_path)
                    logger.info(f"Saved new best decoupled model with Eval Rew: {mean_eval_reward:.2f}")

        return history
