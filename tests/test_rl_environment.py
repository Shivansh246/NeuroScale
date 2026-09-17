"""Tests for the NeuroScale DRL Environment."""

import unittest
import numpy as np
from typing import Dict, Any

from rl.config import EnvConfig, ActionConfig, RewardConfig, StateConfig
from rl.state import StateBuilder, STATE_DIM, STATE_INDICES
from rl.actions import DiscreteActionSpace
from rl.reward import RewardCalculator
from rl.environment import NeuroScaleEnv

class TestStateRepresentation(unittest.TestCase):
    def setUp(self):
        self.config = StateConfig()
        self.builder = StateBuilder(self.config)

    def test_state_dimensionality(self):
        self.assertEqual(STATE_DIM, 11)
        raw = {}
        state = self.builder.build_state(raw)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(state.dtype, np.float32)

    def test_state_ordering_and_mapping(self):
        raw = {
            "current_cpu_util": 200.0,
            "sla_latency": 500.0,
            "is_anomaly": True
        }
        state = self.builder.build_state(raw)
        self.assertEqual(state[STATE_INDICES["current_cpu_util"]], 200.0 / self.config.max_cpu_util)
        self.assertEqual(state[STATE_INDICES["sla_latency"]], 500.0 / self.config.max_latency)
        self.assertEqual(state[STATE_INDICES["is_anomaly"]], 1.0)
        self.assertEqual(state[STATE_INDICES["current_mem_util"]], 0.0)

    def test_state_normalization_clipping(self):
        raw = {
            "current_cpu_util": 1000.0, # exceeds max 400.0
            "sla_latency": -50.0        # below 0
        }
        state = self.builder.build_state(raw)
        self.assertEqual(state[STATE_INDICES["current_cpu_util"]], 1.0)
        self.assertEqual(state[STATE_INDICES["sla_latency"]], 0.0)

    def test_deterministic_state_construction(self):
        raw = {"current_cpu_alloc": 2.0, "anomaly_score": 5.0}
        state1 = self.builder.build_state(raw)
        state2 = self.builder.build_state(raw)
        np.testing.assert_array_equal(state1, state2)


class TestActionSpace(unittest.TestCase):
    def setUp(self):
        self.config = ActionConfig(
            cpu_options=[0.5, 1.0, 2.0],
            memory_options=[256, 512]
        )
        self.actions = DiscreteActionSpace(self.config)

    def test_action_space_size(self):
        self.assertEqual(self.actions.n, 6) # 3 * 2

    def test_action_index_mapping(self):
        # Indices should be sorted lexicographically by cpu, then mem
        # 0: (0.5, 256), 1: (0.5, 512), 2: (1.0, 256), 3: (1.0, 512), etc.
        act0 = self.actions.get_action(0)
        self.assertEqual(act0["cpu"], 0.5)
        self.assertEqual(act0["memory"], 256)
        
        act5 = self.actions.get_action(5)
        self.assertEqual(act5["cpu"], 2.0)
        self.assertEqual(act5["memory"], 512)

    def test_invalid_action_handling(self):
        with self.assertRaises(ValueError):
            self.actions.get_action(-1)
        with self.assertRaises(ValueError):
            self.actions.get_action(6)

    def test_closest_index(self):
        idx = self.actions.get_closest_index(0.6, 300)
        act = self.actions.get_action(idx)
        self.assertEqual(act["cpu"], 0.5)
        self.assertEqual(act["memory"], 256)


class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(
            sla_penalty_weight=10.0,
            sla_latency_threshold_ms=200.0,
            resource_waste_weight=1.0,
            reallocation_penalty=0.5,
            under_provision_penalty=5.0,
            anomaly_penalty=2.0
        )
        self.calc = RewardCalculator(self.config)
        self.prev = {"current_cpu_alloc": 1.0, "current_mem_alloc": 256}

    def test_sla_violation_penalty(self):
        act = {"cpu": 1.0, "memory": 256}
        nxt = {"sla_latency": 300.0, "current_cpu_util": 50.0, "current_mem_util": 100}
        reward = self.calc.calculate(self.prev, act, nxt)
        # Latency=300 > 200. Ratio = (300-200)/200 = 0.5. Penalty = 10 * 1.5 = 15
        # Waste: CPU=(1.0 - 0.5)=0.5, Mem=(256-100)/1024=0.1523. Total waste = 0.6523 * 1 = 0.6523
        # Total reward = -15 - 0.6523 = -15.6523
        self.assertAlmostEqual(reward, -15.6523, places=3)

    def test_resource_waste_penalty(self):
        act = {"cpu": 2.0, "memory": 1024}
        nxt = {"sla_latency": 100.0, "current_cpu_util": 50.0, "current_mem_util": 100}
        reward = self.calc.calculate(self.prev, act, nxt)
        # CPU waste = 2.0 - 0.5 = 1.5
        # Mem waste = (1024 - 100) / 1024 = 0.9023
        # Reallocation penalty = 0.5
        # Total reward = - (1.5 + 0.9023) - 0.5 = -2.9023
        self.assertAlmostEqual(reward, -2.9023, places=3)

    def test_reallocation_penalty(self):
        # Change in allocation triggers penalty
        act = {"cpu": 2.0, "memory": 256}
        nxt = {"sla_latency": 50.0, "current_cpu_util": 200.0, "current_mem_util": 256}
        reward1 = self.calc.calculate(self.prev, act, nxt)
        
        # No change in allocation
        act2 = {"cpu": 1.0, "memory": 256}
        nxt2 = {"sla_latency": 50.0, "current_cpu_util": 100.0, "current_mem_util": 256}
        reward2 = self.calc.calculate(self.prev, act2, nxt2)
        
        self.assertLess(reward1, reward2) # Reallocation penalty applied to reward1


class TestEnvironmentSimulations(unittest.TestCase):
    def setUp(self):
        self.env = NeuroScaleEnv()
        
    def test_environment_reset(self):
        trace = [{"cpu_util": 50.0, "mem_util": 128}]
        self.env.load_trace(trace)
        state, raw = self.env.reset(seed=42)
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(raw["current_cpu_util"], 50.0)
        self.assertEqual(self.env.current_step, 0)
        
    def test_deterministic_repeated_runs(self):
        trace = [
            {"cpu_util": 50.0, "mem_util": 128},
            {"cpu_util": 150.0, "mem_util": 256},
            {"cpu_util": 300.0, "mem_util": 512}
        ]
        self.env.load_trace(trace)
        
        self.env.reset(seed=42)
        s1_1, r1_1, _, _, _ = self.env.step(5)
        s1_2, r1_2, _, _, _ = self.env.step(5)
        
        self.env.reset(seed=42)
        s2_1, r2_1, _, _, _ = self.env.step(5)
        s2_2, r2_2, _, _, _ = self.env.step(5)
        
        np.testing.assert_array_equal(s1_1, s2_1)
        np.testing.assert_array_equal(s1_2, s2_2)
        self.assertEqual(r1_1, r2_1)
        self.assertEqual(r1_2, r2_2)

    def test_cpu_spike_scenario_under_allocation(self):
        trace = [
            {"cpu_util": 50.0, "mem_util": 128},
            {"cpu_util": 300.0, "mem_util": 128} # Spike to 3 cores
        ]
        self.env.load_trace(trace)
        self.env.reset()
        
        # Action index for CPU=1.0, Mem=256
        action_idx = self.env.action_space.get_closest_index(1.0, 256)
        
        # Step into spike with 1 core allocated
        _, reward, _, _, info = self.env.step(action_idx)
        
        raw = info["raw_state"]
        # Actual utilization should be capped at 100.0 (1 core)
        self.assertEqual(raw["current_cpu_util"], 100.0)
        # Deficit is 300.0 - 100.0 = 200.0
        # Latency = 50.0 + 200.0 * 5.0 = 1050.0
        self.assertEqual(raw["sla_latency"], 1050.0)
        # SLA violated, large negative reward expected
        self.assertLess(reward, -50.0)

    def test_memory_spike_scenario(self):
        trace = [
            {"cpu_util": 50.0, "mem_util": 128},
            {"cpu_util": 50.0, "mem_util": 1024} # Mem spike to 1024MB
        ]
        self.env.load_trace(trace)
        self.env.reset()
        
        # Action index for CPU=1.0, Mem=256
        action_idx = self.env.action_space.get_closest_index(1.0, 256)
        
        _, reward, _, _, info = self.env.step(action_idx)
        raw = info["raw_state"]
        # Memory capped at 256
        self.assertEqual(raw["current_mem_util"], 256.0)
        # Deficit is 1024 - 256 = 768
        # Latency = 50.0 + 768 * 20.0 = 15410.0
        self.assertEqual(raw["sla_latency"], 15410.0)
        self.assertLess(reward, -100.0)
        
    def test_over_allocation(self):
        trace = [
            {"cpu_util": 25.0, "mem_util": 128},
            {"cpu_util": 25.0, "mem_util": 128}
        ]
        self.env.load_trace(trace)
        self.env.reset()
        
        # Action index for CPU=2.0, Mem=1024 (massive over allocation)
        action_idx = self.env.action_space.get_closest_index(2.0, 1024)
        _, reward, _, _, _ = self.env.step(action_idx)
        
        # Should be penalized for waste
        self.assertLess(reward, 0.0)
        self.assertGreater(reward, -10.0) # Not as bad as SLA violation

    def test_episode_termination(self):
        trace = [
            {"cpu_util": 50.0},
            {"cpu_util": 50.0},
            {"cpu_util": 50.0}
        ]
        self.env.load_trace(trace)
        self.env.reset()
        
        _, _, term1, _, _ = self.env.step(0)
        self.assertFalse(term1)
        _, _, term2, _, _ = self.env.step(0)
        self.assertTrue(term2)

if __name__ == '__main__':
    unittest.main()
