"""Tests for the Decoupled (Action Branching) DQN Agent and Architecture."""

import unittest
import numpy as np
import torch
import os
import shutil

from rl.decoupled_dqn import DecoupledDQN, DecoupledDQNConfig
from rl.decoupled_agent import DecoupledDQNAgent, DecoupledDQNAgentConfig
from rl.decoupled_trainer import DecoupledDQNTrainer
from rl.environment import NeuroScaleEnv
from rl.config import EnvConfig
from rl.evaluation import get_synthetic_evaluation_scenarios, evaluate_agent


class TestDecoupledDQN(unittest.TestCase):
    def setUp(self):
        self.config = DecoupledDQNConfig(
            state_dim=11,
            cpu_action_count=4,
            mem_action_count=4,
            hidden_dim=32
        )
        self.net = DecoupledDQN(self.config)

    def test_branch_output_shapes(self):
        batch = torch.rand(5, 11)
        q_cpu, q_mem = self.net(batch)

        # CPU-head output shape = (B, 4)
        self.assertEqual(q_cpu.shape, (5, 4))
        # Memory-head output shape = (B, 4)
        self.assertEqual(q_mem.shape, (5, 4))

    def test_single_vector_output_shapes(self):
        single = torch.rand(11)
        q_cpu, q_mem = self.net(single)
        self.assertEqual(q_cpu.shape[-1], 4)
        self.assertEqual(q_mem.shape[-1], 4)


class TestDecoupledDQNAgent(unittest.TestCase):
    def setUp(self):
        self.config = DecoupledDQNAgentConfig(
            batch_size=4,
            replay_capacity=100,
            epsilon_start=1.0,
            epsilon_end=0.1,
            seed=42
        )
        self.agent = DecoupledDQNAgent(self.config)
        self.state = np.random.rand(11).astype(np.float32)

    def test_deterministic_action_selection(self):
        # In evaluate mode, repeated calls on identical state must yield identical action indices
        act1 = self.agent.choose_action_index(self.state, evaluate=True)
        act2 = self.agent.choose_action_index(self.state, evaluate=True)
        self.assertEqual(act1, act2)

        c1, m1 = self.agent.choose_branch_indices(self.state, evaluate=True)
        c2, m2 = self.agent.choose_branch_indices(self.state, evaluate=True)
        self.assertEqual(c1, c2)
        self.assertEqual(m1, m2)

    def test_valid_cpu_memory_combinations(self):
        action_dict = self.agent.choose_action(self.state)
        self.assertIn("cpu", action_dict)
        self.assertIn("memory", action_dict)

        # Valid CPU option and valid Memory option
        self.assertIn(action_dict["cpu"], [0.25, 0.5, 1.0, 2.0])
        self.assertIn(action_dict["memory"], [128, 256, 512, 1024])

    def test_checkpoint_save_and_load(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        path = "test_checkpoints/test_decoupled_dqn.pt"

        self.agent.epsilon = 0.45
        self.agent.steps_done = 120
        self.agent.save(path)

        loaded = DecoupledDQNAgent.load(path)
        self.assertEqual(loaded.epsilon, 0.45)
        self.assertEqual(loaded.steps_done, 120)

        # Predictions after reload must match exactly
        act_orig = self.agent.choose_action_index(self.state, evaluate=True)
        act_loaded = loaded.choose_action_index(self.state, evaluate=True)
        self.assertEqual(act_orig, act_loaded)

        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")

    def test_training_step(self):
        # Empty buffer returns None
        self.assertIsNone(self.agent.train_step())

        # Populate buffer
        for i in range(10):
            self.agent.memory.push(self.state, i % 16, 1.0, self.state, False)

        loss = self.agent.train_step()
        self.assertIsNotNone(loss)
        self.assertTrue(loss >= 0.0)

    def test_representative_state_action_selection(self):
        # Verify agent can run forward pass on representative state representations
        from rl.state import StateBuilder
        from rl.config import StateConfig
        sb = StateBuilder(StateConfig())

        norm_st = {
            "current_cpu_util": 40.0, "current_mem_util": 128.0,
            "predicted_cpu_demand": 40.0, "predicted_mem_demand": 128.0,
            "current_cpu_alloc": 0.5, "current_mem_alloc": 128,
            "sla_latency": 50.0, "anomaly_score": 0.0, "is_anomaly": False,
            "host_cpu_avail": 4.0, "host_mem_avail": 4096.0
        }
        s_vec = sb.build_state(norm_st)
        act_dict = self.agent.choose_action(s_vec)
        self.assertIsInstance(act_dict["cpu"], float)
        self.assertIsInstance(act_dict["memory"], int)

    def test_trainer_integration(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        trainer = DecoupledDQNTrainer(
            DecoupledDQNAgentConfig(batch_size=4, seed=42),
            EnvConfig()
        )
        trainer.checkpoint_path = "test_checkpoints/test_decoupled.pt"
        history = trainer.train(num_episodes=2, max_steps_per_episode=5, eval_freq=2)
        self.assertEqual(len(history["episode_rewards"]), 2)
        self.assertEqual(len(history["eval_rewards"]), 1)
        self.assertTrue(os.path.exists(trainer.checkpoint_path))

        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")


if __name__ == '__main__':
    unittest.main()
