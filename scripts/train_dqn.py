#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.trainer import DQNTrainer
from rl.agent import DQNAgentConfig
from rl.config import EnvConfig

def main():
    agent_config = DQNAgentConfig()
    env_config = EnvConfig()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    trainer = DQNTrainer(agent_config, env_config)
    trainer.checkpoint_path = "checkpoints/best_dqn.pt"
    
    print("Training DQN...")
    # Train for fewer episodes to keep it fast, enough to demonstrate
    history = trainer.train(num_episodes=50, max_steps_per_episode=50, eval_freq=10)
    print(f"DQN training complete. Best model saved to: {trainer.checkpoint_path}")

if __name__ == '__main__':
    main()
