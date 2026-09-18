"""Tests for the Resource-Factorized DQN Architecture, Agent, and Training Pipeline."""

import unittest
import numpy as np
import torch
import os
import shutil

from rl.resource_factorized_dqn import ResourceFactorizedDQN, ResourceFactorizedDQNConfig
from rl.resource_factorized_agent import (
    ResourceFactorizedDQNAgent,
    ResourceFactorizedAgentConfig,
    ResourceFactorizedReplayBuffer,
)
from rl.resource_factorized_trainer import ResourceFactorizedDQNTrainer
from rl.decomposed_env import DecomposedRewardEnv
from rl.reward import RewardCalculator
from rl.config import EnvConfig, RewardConfig, ActionConfig


class TestResourceFactorizedDQN(unittest.TestCase):
    def setUp(self):
        self.config = ResourceFactorizedDQNConfig(
            state_dim=11,
            cpu_action_count=4,
            mem_action_count=4,
            hidden_dim=32,
        )
        self.net = ResourceFactorizedDQN(self.config)

    def test_branch_output_shapes_batch(self):
        batch = torch.rand(5, 11)
        q_cpu, q_mem = self.net(batch)
        self.assertEqual(q_cpu.shape, (5, 4))
        self.assertEqual(q_mem.shape, (5, 4))

    def test_single_vector_output_shapes(self):
        single = torch.rand(11)
        q_cpu, q_mem = self.net(single)
        self.assertEqual(q_cpu.shape[-1], 4)
        self.assertEqual(q_mem.shape[-1], 4)

    def test_direct_args_constructor(self):
        net = ResourceFactorizedDQN(state_dim=11, hidden_dim=48, cpu_action_count=4, mem_action_count=4)
        x = torch.rand(3, 11)
        qc, qm = net(x)
        self.assertEqual(qc.shape, (3, 4))
        self.assertEqual(qm.shape, (3, 4))


class TestResourceFactorizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ResourceFactorizedReplayBuffer(capacity=10, state_dim=11, seed=42)

    def test_push_and_sample_separate_rewards(self):
        s1 = np.ones(11, dtype=np.float32)
        s2 = np.ones(11, dtype=np.float32) * 2

        self.buffer.push(
            state=s1,
            cpu_action=2,
            mem_action=1,
            r_cpu=-3.5,
            r_mem=-1.2,
            next_state=s2,
            done=False,
        )
        self.assertEqual(len(self.buffer), 1)

        states, cpu_acts, mem_acts, r_cpus, r_mems, next_states, dones = self.buffer.sample(1)
        self.assertEqual(cpu_acts[0], 2)
        self.assertEqual(mem_acts[0], 1)
        self.assertAlmostEqual(r_cpus[0], -3.5, places=5)
        self.assertAlmostEqual(r_mems[0], -1.2, places=5)
        self.assertFalse(dones[0])


class TestResourceFactorizedDQNAgent(unittest.TestCase):
    def setUp(self):
        self.config = ResourceFactorizedAgentConfig(
            batch_size=4,
            replay_capacity=100,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
            seed=42,
        )
        self.agent = ResourceFactorizedDQNAgent(self.config)
        self.state = np.random.rand(11).astype(np.float32)

    def test_deterministic_greedy_action_selection(self):
        c1, m1 = self.agent.choose_branch_indices(self.state, evaluate=True)
        c2, m2 = self.agent.choose_branch_indices(self.state, evaluate=True)
        self.assertEqual(c1, c2)
        self.assertEqual(m1, m2)

        idx1 = self.agent.choose_action_index(self.state, evaluate=True)
        idx2 = self.agent.choose_action_index(self.state, evaluate=True)
        self.assertEqual(idx1, idx2)
        self.assertEqual(idx1, c1 * 4 + m1)

    def test_choose_action_dict(self):
        act_dict = self.agent.choose_action(self.state)
        self.assertIn("cpu", act_dict)
        self.assertIn("memory", act_dict)
        self.assertIn(act_dict["cpu"], [0.25, 0.5, 1.0, 2.0])
        self.assertIn(act_dict["memory"], [128, 256, 512, 1024])

    def test_train_step_gradient_flow(self):
        # Empty buffer returns None
        self.assertIsNone(self.agent.train_step())

        # Populate buffer with factorized transitions
        for i in range(10):
            self.agent.memory.push(
                state=self.state,
                cpu_action=i % 4,
                mem_action=(i + 1) % 4,
                r_cpu=-float(i),
                r_mem=-float(i * 0.5),
                next_state=self.state,
                done=(i == 9),
            )

        loss = self.agent.train_step()
        self.assertIsNotNone(loss)
        self.assertTrue(loss >= 0.0)

        # Check gradients exist on trunk and both heads
        for param in self.agent.online_net.trunk.parameters():
            self.assertIsNotNone(param.grad)
        for param in self.agent.online_net.q_cpu.parameters():
            self.assertIsNotNone(param.grad)
        for param in self.agent.online_net.q_mem.parameters():
            self.assertIsNotNone(param.grad)

    def test_checkpoint_save_and_load(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        path = "test_checkpoints/test_rf_dqn.pt"

        self.agent.steps_done = 55
        self.agent.epsilon = 0.35
        self.agent.save(path)

        loaded = ResourceFactorizedDQNAgent.load(path)
        self.assertEqual(loaded.steps_done, 55)
        self.assertAlmostEqual(loaded.epsilon, 0.35, places=5)

        act_orig = self.agent.choose_action_index(self.state, evaluate=True)
        act_loaded = loaded.choose_action_index(self.state, evaluate=True)
        self.assertEqual(act_orig, act_loaded)

        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")

    def test_reward_identity_in_environment(self):
        env = DecomposedRewardEnv(EnvConfig())
        trace = [
            {"cpu_util": 50.0, "mem_util": 200.0},
            {"cpu_util": 180.0, "mem_util": 768.0, "is_anomaly": True},
        ]
        env.load_trace(trace)
        s, _ = env.reset()

        orig_calc = RewardCalculator(RewardConfig())

        # Step 1
        act_idx = 10  # cpu 1.0, mem 512
        act_dict = env.action_space.get_action(act_idx)
        prev_raw = env.current_raw_state.copy()

        next_s, r_exp, term, trunc, info = env.step(act_idx)
        next_raw = env.current_raw_state.copy()

        r_orig = orig_calc.calculate(prev_raw, act_dict, next_raw)

        self.assertAlmostEqual(info["r_cpu"] + info["r_mem"], r_exp, places=5)
        self.assertAlmostEqual(r_exp, r_orig, places=5)


class TestResourceFactorizedTrainerIntegration(unittest.TestCase):
    def test_trainer_short_run(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        trainer = ResourceFactorizedDQNTrainer(
            agent_config=ResourceFactorizedAgentConfig(batch_size=4, seed=42),
            env_config=EnvConfig(),
        )
        trainer.checkpoint_path = "test_checkpoints/test_trainer_rf.pt"
        history = trainer.train(num_episodes=2, max_steps_per_episode=5, eval_freq=2)

        self.assertEqual(len(history["episode_rewards"]), 2)
        self.assertEqual(len(history["eval_rewards"]), 1)
        self.assertTrue(os.path.exists(trainer.checkpoint_path))

        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")


if __name__ == "__main__":
    unittest.main()
