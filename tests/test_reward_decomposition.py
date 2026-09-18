"""Tests for Resource-Aware Reward Decomposition and Learning."""

import unittest
import numpy as np
import os
import shutil

from rl.decomposed_reward import DecomposedRewardCalculator
from rl.decomposed_env import DecomposedRewardEnv
from rl.reward_aware_trainer import RewardAwareDQNTrainer
from rl.agent import DQNAgent, DQNAgentConfig
from rl.config import EnvConfig, RewardConfig, ActionConfig
from rl.actions import DiscreteActionSpace
from rl.reward import RewardCalculator


class TestRewardDecomposition(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig()
        self.decomposed = DecomposedRewardCalculator(self.config)
        self.original = RewardCalculator(self.config)
        self.actions = DiscreteActionSpace(ActionConfig())

    def test_r_experiment_equals_sum_of_components(self):
        # r_experiment must equal r_cpu + r_mem for any state transition
        prev_st = {"current_cpu_alloc": 1.0, "current_mem_alloc": 256, "sla_latency": 50.0}
        act = {"cpu": 1.0, "memory": 256}
        next_st = {
            "current_cpu_util": 50.0, "current_mem_util": 200.0,
            "current_cpu_alloc": 1.0, "current_mem_alloc": 256,
            "sla_latency": 50.0, "is_anomaly": False
        }
        r_cpu, r_mem, r_exp = self.decomposed.calculate_decomposed(prev_st, act, next_st)
        self.assertAlmostEqual(r_exp, r_cpu + r_mem, places=6)

    def test_no_double_counting_shared_penalties(self):
        # Anomaly and reallocation are shared penalties; their sum across branches must equal the original single penalty
        prev_st = {"current_cpu_alloc": 0.5, "current_mem_alloc": 128, "sla_latency": 50.0}
        act = {"cpu": 2.0, "memory": 1024} # Reallocation on both
        next_st = {
            "current_cpu_util": 40.0, "current_mem_util": 128.0,
            "current_cpu_alloc": 2.0, "current_mem_alloc": 1024,
            "sla_latency": 50.0, "is_anomaly": True
        }
        r_cpu, r_mem, r_exp = self.decomposed.calculate_decomposed(prev_st, act, next_st)
        r_orig = self.original.calculate(prev_st, act, next_st)
        self.assertAlmostEqual(r_exp, r_orig, places=6)

    def test_cpu_heavy_case(self):
        # In a CPU spike with adequate memory, SLA penalty should fall entirely on r_cpu
        prev_st = {"current_cpu_alloc": 0.5, "current_mem_alloc": 128, "sla_latency": 50.0}
        act = {"cpu": 0.25, "memory": 128} # Under-allocated on CPU, adequate on memory
        target_cpu = 180.0
        target_mem = 128.0

        # Latency calculation: 50 + (180 - 25)*5 = 825 ms
        next_st = {
            "current_cpu_util": 25.0, "current_mem_util": 128.0,
            "current_cpu_alloc": 0.25, "current_mem_alloc": 128,
            "sla_latency": 825.0, "is_anomaly": True
        }
        r_cpu, r_mem, r_exp = self.decomposed.calculate_decomposed(
            prev_st, act, next_st, target_cpu_util=target_cpu, target_mem_util=target_mem
        )
        # Memory suffered zero deficit and zero waste, so r_mem only has base shared penalties (-1.0 anomaly)
        self.assertGreater(r_mem, -2.5)
        # CPU suffered massive deficit leading to full SLA penalty (-20), so r_cpu <= -20
        self.assertLessEqual(r_cpu, -20.0)

    def test_memory_heavy_case(self):
        # In a Memory spike with adequate CPU, SLA penalty should fall entirely on r_mem
        prev_st = {"current_cpu_alloc": 0.5, "current_mem_alloc": 128, "sla_latency": 50.0}
        act = {"cpu": 0.5, "memory": 128} # Adequate CPU (40%), under-allocated on memory (768MB)
        target_cpu = 40.0
        target_mem = 768.0

        # Latency calculation: 50 + (768 - 128)*20 = 12850 ms
        next_st = {
            "current_cpu_util": 40.0, "current_mem_util": 128.0,
            "current_cpu_alloc": 0.5, "current_mem_alloc": 128,
            "sla_latency": 12850.0, "is_anomaly": True
        }
        r_cpu, r_mem, r_exp = self.decomposed.calculate_decomposed(
            prev_st, act, next_st, target_cpu_util=target_cpu, target_mem_util=target_mem
        )
        # CPU suffered zero deficit and zero waste
        self.assertGreater(r_cpu, -2.5)
        # Memory suffered deficit leading to full SLA penalty (-20)
        self.assertLessEqual(r_mem, -20.0)

    def test_joint_stress_case(self):
        # Both resources stressed: SLA penalty split according to latency contribution
        prev_st = {"current_cpu_alloc": 0.5, "current_mem_alloc": 128, "sla_latency": 50.0}
        act = {"cpu": 0.25, "memory": 128}
        target_cpu = 150.0 # deficit = 125% -> 625 ms
        target_mem = 512.0 # deficit = 384 MB -> 7680 ms
        next_st = {
            "current_cpu_util": 25.0, "current_mem_util": 128.0,
            "current_cpu_alloc": 0.25, "current_mem_alloc": 128,
            "sla_latency": 8355.0, "is_anomaly": True
        }
        r_cpu, r_mem, r_exp = self.decomposed.calculate_decomposed(
            prev_st, act, next_st, target_cpu_util=target_cpu, target_mem_util=target_mem
        )
        # Both resources penalized significantly
        self.assertLess(r_cpu, -2.0)
        self.assertLess(r_mem, -15.0)
        self.assertAlmostEqual(r_exp, r_cpu + r_mem, places=6)

    def test_16_action_output_compatibility(self):
        # Verify 16-action agent works seamlessly with decomposed reward environment
        env = DecomposedRewardEnv(EnvConfig())
        trace = [{"cpu_util": 40.0, "mem_util": 128.0} for _ in range(5)]
        env.load_trace(trace)
        s, _ = env.reset()

        agent = DQNAgent(DQNAgentConfig())
        act_idx = agent.choose_action_index(s)
        self.assertTrue(0 <= act_idx < 16)

        next_s, r, term, trunc, info = env.step(act_idx)
        self.assertIn("r_cpu", info)
        self.assertIn("r_mem", info)
        self.assertIn("r_exp", info)
        self.assertAlmostEqual(r, info["r_exp"], places=6)

    def test_checkpoint_save_and_load(self):
        os.makedirs("test_checkpoints", exist_ok=True)
        path = "test_checkpoints/test_reward_aware.pt"
        agent = DQNAgent(DQNAgentConfig())
        agent.save(path)
        loaded = DQNAgent.load(path)
        self.assertEqual(agent.steps_done, loaded.steps_done)
        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")


if __name__ == '__main__':
    unittest.main()
