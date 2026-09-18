#!/usr/bin/env python3
"""Train the Resource-Factorized DQN Agent with independent CPU and Memory TD learning."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.resource_factorized_trainer import ResourceFactorizedDQNTrainer
from rl.resource_factorized_agent import ResourceFactorizedAgentConfig
from rl.config import EnvConfig


def main():
    agent_config = ResourceFactorizedAgentConfig()
    env_config = EnvConfig()

    os.makedirs("checkpoints", exist_ok=True)

    trainer = ResourceFactorizedDQNTrainer(agent_config, env_config)
    trainer.checkpoint_path = "checkpoints/best_resource_factorized_dqn.pt"

    print("=" * 80)
    print("TRAINING RESOURCE-FACTORIZED DQN AGENT")
    print("Architecture: Shared Trunk (11->64->64) + Q_cpu (4 actions) + Q_mem (4 actions)")
    print("TD Targets:   y_cpu = r_cpu + gamma * max Q_cpu_target(s')")
    print("              y_mem = r_mem + gamma * max Q_mem_target(s')")
    print("Loss:         loss_cpu + loss_mem (Huber)")
    print("Checkpoint:   checkpoints/best_resource_factorized_dqn.pt")
    print("=" * 80)

    history = trainer.train(num_episodes=100, max_steps_per_episode=50, eval_freq=10)

    print("=" * 80)
    print(f"Training completed successfully!")
    print(f"Best Eval Reward: {trainer.best_eval_reward:.2f}")
    print(f"Model saved to:   {trainer.checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
