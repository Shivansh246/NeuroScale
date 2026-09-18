#!/usr/bin/env python3
"""Train the experimental Decoupled (Action Branching) DQN Agent."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.decoupled_trainer import DecoupledDQNTrainer
from rl.decoupled_agent import DecoupledDQNAgentConfig
from rl.config import EnvConfig


def main():
    agent_config = DecoupledDQNAgentConfig()
    env_config = EnvConfig()

    os.makedirs("checkpoints", exist_ok=True)

    trainer = DecoupledDQNTrainer(agent_config, env_config)
    trainer.checkpoint_path = "checkpoints/best_decoupled_dqn.pt"

    print("Training Decoupled (Action Branching) DQN...")
    # Exact apples-to-apples training: 100 episodes * 50 steps = 5,000 environment steps
    history = trainer.train(num_episodes=100, max_steps_per_episode=50, eval_freq=10)
    print(f"Decoupled DQN training complete. Best model saved to: {trainer.checkpoint_path}")


if __name__ == '__main__':
    main()
