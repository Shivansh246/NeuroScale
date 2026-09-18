"""Tests for the DQN Agent and Replay Buffer."""

import unittest
import numpy as np
import torch
import os
import shutil

from rl.dqn import DQN, DQNConfig
from rl.replay_buffer import ReplayBuffer
from rl.agent import DQNAgent, DQNAgentConfig
from rl.environment import NeuroScaleEnv
from rl.config import EnvConfig
from rl.evaluation import evaluate_agent, get_synthetic_evaluation_scenarios
from rl.trainer import DQNTrainer

class TestDQN(unittest.TestCase):
    def test_network_construction_and_shape(self):
        config = DQNConfig(state_dim=11, action_count=16, hidden_dim=32)
        net = DQN(config)
        
        state = torch.rand(5, 11)
        q_values = net(state)
        
        self.assertEqual(q_values.shape, (5, 16))

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100, state_dim=11, seed=42)

    def test_insertion_and_capacity(self):
        self.assertEqual(len(self.buffer), 0)
        
        state = np.ones(11, dtype=np.float32)
        next_state = np.zeros(11, dtype=np.float32)
        
        for i in range(150):
            self.buffer.push(state, i % 16, 1.0, next_state, False)
            
        self.assertEqual(len(self.buffer), 100)
        # Check overwrite
        self.assertEqual(self.buffer.actions[self.buffer.ptr - 1], 149 % 16)

    def test_sampling_shape_and_determinism(self):
        for i in range(20):
            self.buffer.push(np.ones(11), 1, 1.0, np.zeros(11), False)
            
        s, a, r, n, d = self.buffer.sample(10)
        self.assertEqual(s.shape, (10, 11))
        self.assertEqual(a.shape, (10,))
        self.assertEqual(r.shape, (10,))
        self.assertEqual(n.shape, (10, 11))
        self.assertEqual(d.shape, (10,))
        
        # Test determinism
        buf1 = ReplayBuffer(capacity=50, state_dim=11, seed=123)
        buf2 = ReplayBuffer(capacity=50, state_dim=11, seed=123)
        for i in range(20):
            buf1.push(np.ones(11)*i, i%5, 1.0, np.zeros(11), False)
            buf2.push(np.ones(11)*i, i%5, 1.0, np.zeros(11), False)
            
        s1, a1, _, _, _ = buf1.sample(5)
        s2, a2, _, _, _ = buf2.sample(5)
        
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(a1, a2)

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.config = DQNAgentConfig(
            batch_size=4,
            replay_capacity=100,
            epsilon_start=1.0,
            epsilon_end=0.1,
            seed=42
        )
        self.agent = DQNAgent(self.config)
        self.state = np.random.rand(11).astype(np.float32)

    def test_epsilon_greedy(self):
        # Epsilon is 1.0 initially, should return random actions
        actions = [self.agent.choose_action_index(self.state) for _ in range(50)]
        self.assertTrue(len(set(actions)) > 1)
        
        # Evaluate mode (epsilon=0) should be deterministic
        act1 = self.agent.choose_action_index(self.state, evaluate=True)
        act2 = self.agent.choose_action_index(self.state, evaluate=True)
        self.assertEqual(act1, act2)

    def test_public_choose_action(self):
        action_dict = self.agent.choose_action(self.state)
        self.assertIn("cpu", action_dict)
        self.assertIn("memory", action_dict)
        self.assertTrue(isinstance(action_dict["cpu"], float))
        self.assertTrue(isinstance(action_dict["memory"], int))
        
        # Bounds check
        self.assertTrue(0.1 <= action_dict["cpu"] <= 16.0)
        self.assertTrue(64 <= action_dict["memory"] <= 32768)

    def test_train_step(self):
        # Empty buffer
        self.assertIsNone(self.agent.train_step())
        
        for _ in range(10):
            self.agent.memory.push(self.state, 0, 1.0, self.state, False)
            
        loss = self.agent.train_step()
        self.assertIsNotNone(loss)
        self.assertTrue(loss >= 0.0)

    def test_save_load(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        path = "test_checkpoints/test_dqn.pt"
        
        # Change epsilon and step to verify it loads
        self.agent.epsilon = 0.5
        self.agent.steps_done = 55
        self.agent.save(path)
        
        loaded_agent = DQNAgent.load(path)
        self.assertEqual(loaded_agent.epsilon, 0.5)
        self.assertEqual(loaded_agent.steps_done, 55)
        
        # Output should be identical
        act1 = self.agent.choose_action_index(self.state, evaluate=True)
        act2 = loaded_agent.choose_action_index(self.state, evaluate=True)
        self.assertEqual(act1, act2)
        
        shutil.rmtree("test_checkpoints")

class TestTrainerAndEval(unittest.TestCase):
    def test_evaluation(self):
        agent_config = DQNAgentConfig(seed=42)
        agent = DQNAgent(agent_config)
        env = NeuroScaleEnv()
        scenarios = get_synthetic_evaluation_scenarios()
        
        results = evaluate_agent(agent, env, scenarios)
        self.assertIn("normal", results)
        self.assertIn("cpu_spike", results)
        
        self.assertTrue(len(results["normal"]["actions"]) > 0)
        
    def test_trainer_integration(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        agent_config = DQNAgentConfig(batch_size=8, seed=42)
        env_config = EnvConfig()
        trainer = DQNTrainer(agent_config, env_config)
        trainer.checkpoint_path = "test_checkpoints/test_dqn_trainer.pt"
        
        # Run very short training
        history = trainer.train(num_episodes=2, max_steps_per_episode=10, eval_freq=2)
        
        self.assertEqual(len(history["episode_rewards"]), 2)
        self.assertEqual(len(history["eval_rewards"]), 1)
        self.assertTrue(os.path.exists(trainer.checkpoint_path))
        
        # Cleanup
        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")

if __name__ == '__main__':
    unittest.main()
