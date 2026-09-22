import unittest
from unittest.mock import MagicMock
import numpy as np

from control.loop import Orchestrator
from control.config import ControlLoopConfig

class DummyCollector:
    def __init__(self, sequence):
        self.sequence = sequence
        self.idx = 0
    def collect(self, container_id):
        if self.idx < len(self.sequence):
            res = self.sequence[self.idx]
            self.idx += 1
            return res
        return None

class TestFeatureContract(unittest.TestCase):
    def setUp(self):
        self.config = ControlLoopConfig()
        self.config.feature_columns = ("cpu_percent", "cpu_usage_ns_delta", "memory_usage_mb", "memory_percent")
        
        self.predictor = MagicMock()
        self.predictor.predict.return_value = {"cpu": 1.0, "memory": 256.0, "confidence": 0.9}
        self.detector = MagicMock()
        self.detector.score.return_value = {"score": 0.5, "is_anomaly": False}
        self.agent = MagicMock()
        self.agent.choose_action_index.return_value = 0
        self.action_space = MagicMock()
        self.action_space.get_action.return_value = {"cpu": 1.0, "memory": 256}
        self.state_builder = MagicMock()
        self.state_builder.build_state.return_value = np.zeros(11)
        self.reward_calc = MagicMock()
        self.controller = MagicMock()
        
    def test_cumulative_cpu_to_delta(self):
        seq = [
            {"cpu_usage_ns": 1000},
            {"cpu_usage_ns": 1500},
            {"cpu_usage_ns": 2500},
        ]
        collector = DummyCollector(seq)
        orchestrator = Orchestrator(self.config, collector, self.predictor, self.detector, self.agent, self.controller, self.action_space, self.state_builder, self.reward_calc)
        
        # 1st cycle: first sample -> delta is 0
        r1 = orchestrator.run_cycle("cont1")
        self.assertEqual(r1["cpu_usage_ns_delta"], 0)
        
        # 2nd cycle: 1500 - 1000
        r2 = orchestrator.run_cycle("cont1")
        self.assertEqual(r2["cpu_usage_ns_delta"], 500)
        
        # 3rd cycle: 2500 - 1500
        r3 = orchestrator.run_cycle("cont1")
        self.assertEqual(r3["cpu_usage_ns_delta"], 1000)

    def test_counter_reset(self):
        seq = [
            {"cpu_usage_ns": 5000},
            {"cpu_usage_ns": 1000}, # reset!
            {"cpu_usage_ns": 1500},
        ]
        collector = DummyCollector(seq)
        orchestrator = Orchestrator(self.config, collector, self.predictor, self.detector, self.agent, self.controller, self.action_space, self.state_builder, self.reward_calc)
        
        r1 = orchestrator.run_cycle("cont1")
        self.assertEqual(r1["cpu_usage_ns_delta"], 0)
        
        r2 = orchestrator.run_cycle("cont1")
        self.assertTrue(r2["counter_reset"])
        self.assertEqual(r2["cpu_usage_ns_delta"], 1000) # does not produce negative!
        
        r3 = orchestrator.run_cycle("cont1")
        self.assertFalse(r3["counter_reset"])
        self.assertEqual(r3["cpu_usage_ns_delta"], 500)

    def test_container_transition_no_reset(self):
        seq = [
            {"cpu_usage_ns": 1000},
            {"cpu_usage_ns": 2000},
            {"cpu_usage_ns": 1500}, # Different container!
            {"cpu_usage_ns": 1600},
        ]
        collector = DummyCollector(seq)
        orchestrator = Orchestrator(self.config, collector, self.predictor, self.detector, self.agent, self.controller, self.action_space, self.state_builder, self.reward_calc)
        
        r1 = orchestrator.run_cycle("cont1")
        self.assertEqual(r1["cpu_usage_ns_delta"], 0)
        
        r2 = orchestrator.run_cycle("cont1")
        self.assertEqual(r2["cpu_usage_ns_delta"], 1000)
        
        r3 = orchestrator.run_cycle("cont2")
        self.assertEqual(r3["cpu_usage_ns_delta"], 0) # reset for new container
        
        r4 = orchestrator.run_cycle("cont2")
        self.assertEqual(r4["cpu_usage_ns_delta"], 100)

    def test_workload_transition_no_reset(self):
        seq = [
            {"cpu_usage_ns": 1000},
            {"cpu_usage_ns": 2000},
        ]
        collector = DummyCollector(seq)
        orchestrator = Orchestrator(self.config, collector, self.predictor, self.detector, self.agent, self.controller, self.action_space, self.state_builder, self.reward_calc)
        
        r1 = orchestrator.run_cycle("cont1", workload_mode="idle")
        self.assertEqual(r1["cpu_usage_ns_delta"], 0)
        
        # Workload transition shouldn't reset delta if container is same
        r2 = orchestrator.run_cycle("cont1", workload_mode="cpu")
        self.assertEqual(r2["cpu_usage_ns_delta"], 1000)

    def test_live_feature_order_matches_config(self):
        seq = [{"cpu_percent": 10.0, "cpu_usage_ns": 1000, "memory_usage_mb": 50.0, "memory_percent": 10.0}]
        collector = DummyCollector(seq)
        orchestrator = Orchestrator(self.config, collector, self.predictor, self.detector, self.agent, self.controller, self.action_space, self.state_builder, self.reward_calc)
        
        orchestrator.run_cycle("cont1")
        
        # Inspect rolling window element 0
        w0 = orchestrator.rolling_window[0]
        feats = [w0[col] for col in self.config.feature_columns]
        self.assertEqual(feats, [10.0, 0, 50.0, 10.0]) # delta is 0 for first
        
        # And ensure the array matches
        arr = orchestrator._extract_sequence()
        self.assertEqual(arr.shape, (1, 4))
        self.assertTrue(np.isfinite(arr).all())
        
if __name__ == '__main__':
    unittest.main()
