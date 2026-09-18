#!/usr/bin/env python3
"""Train the monolithic 16-Action DQN with Resource-Aware Decomposed Reward feedback."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.reward_aware_trainer import RewardAwareDQNTrainer
from rl.agent import DQNAgentConfig
from rl.config import EnvConfig


def main():
    agent_config = DQNAgentConfig()
    env_config = EnvConfig()

    os.makedirs("checkpoints", exist_ok=True)

    trainer = RewardAwareDQNTrainer(agent_config, env_config)
    trainer.checkpoint_path = "checkpoints/best_reward_aware_dqn.pt"

    print("Training Monolithic 16-Action DQN with Resource-Aware Decomposed Reward...")
    history = trainer.train(num_episodes=100, max_steps_per_episode=50, eval_freq=10)
    print(f"Reward-aware DQN training complete. Best model saved to: {trainer.checkpoint_path}")


if __name__ == '__main__':
    main()
