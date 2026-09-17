"""Tests for the Phase 7 Orchestration Loop."""

import unittest
import numpy as np
import os

from control.config import ControlLoopConfig
from control.loop import Orchestrator

# Mocks
class MockCollector:
    def collect(self, container_id):
        return {
            "timestamp": 123.0,
            "container_id": container_id,
            "cpu_percent": 50.0,
            "cpu_usage_ns": 1000,
            "memory_usage_mb": 128.0,
            "memory_percent": 10.0
        }

class MockPredictor:
    def predict(self, sequence):
        return {"cpu": 60.0, "memory": 200.0, "confidence": 0.9}

class MockDetector:
    def score(self, sequence):
        return {"score": 0.5, "is_anomaly": False}

class MockAgent:
    def choose_action_index(self, state, evaluate=True):
        return 5 # (0.5 CPU, 512 Mem based on standard sorting)

class MockController:
    def apply(self, container_id, cpu, memory):
        return {"success": True, "applied": {"cpu": cpu, "memory": memory}}

class MockActionSpace:
    def get_action(self, idx):
        return {"cpu": 0.5, "memory": 512}
        
class MockStateBuilder:
    def build_state(self, raw):
        return np.zeros(11, dtype=np.float32)

class MockRewardCalc:
    def calculate(self, prev_state, action, next_state):
        return -1.0

class TestOrchestration(unittest.TestCase):
    def setUp(self):
        self.config = ControlLoopConfig(window_size=3, warmup_cycles=3, log_file="tests/test_loop.jsonl", simulation_mode=False)
        self.orchestrator = Orchestrator(
            config=self.config,
            collector=MockCollector(),
            predictor=MockPredictor(),
            detector=MockDetector(),
            agent=MockAgent(),
            controller=MockController(),
            action_space=MockActionSpace(),
            state_builder=MockStateBuilder(),
            reward_calculator=MockRewardCalc()
        )

    def tearDown(self):
        if os.path.exists("tests/test_loop.jsonl"):
            os.remove("tests/test_loop.jsonl")

    def test_warmup(self):
        res1 = self.orchestrator.run_cycle("test-container")
        self.assertEqual(res1["status"], "warming_up")
        self.assertNotIn("predicted_cpu", res1)
        
        res2 = self.orchestrator.run_cycle("test-container")
        self.assertEqual(res2["status"], "warming_up")
        
    def test_full_cycle(self):
        # Warmup
        self.orchestrator.run_cycle("test-container")
        self.orchestrator.run_cycle("test-container")
        
        # Action cycle
        res3 = self.orchestrator.run_cycle("test-container")
        self.assertEqual(res3["status"], "active")
        self.assertEqual(res3["predicted_cpu"], 60.0)
        self.assertEqual(res3["anomaly_score"], 0.5)
        self.assertEqual(res3["selected_action_cpu"], 0.5)
        self.assertEqual(res3["selected_action_memory"], 512)
        self.assertTrue(res3["controller_success"])
        
        # Fourth cycle to check reward
        res4 = self.orchestrator.run_cycle("test-container")
        self.assertEqual(res4["reward"], -1.0)
        
    def test_simulation_mode(self):
        self.orchestrator.config.simulation_mode = True
        self.orchestrator.run_cycle("test-container")
        self.orchestrator.run_cycle("test-container")
        res = self.orchestrator.run_cycle("test-container")
        self.assertTrue(res["controller_success"])
        self.assertEqual(res["simulation_mode"], True)

if __name__ == '__main__':
    unittest.main()
