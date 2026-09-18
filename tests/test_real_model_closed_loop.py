"""Tests for Real-Model Closed-Loop Integration with Resource-Factorized DQN."""

import os
import unittest
import numpy as np
import tempfile
import torch

from control.config import ControlLoopConfig
from control.loop import Orchestrator
from rl.resource_factorized_agent import ResourceFactorizedDQNAgent, ResourceFactorizedAgentConfig
from rl.state import StateBuilder
from rl.reward import RewardCalculator
from rl.config import StateConfig, RewardConfig, ActionConfig
from models.predictor import TransformerPredictor
from anomaly.detector import AutoencoderAnomalyDetector


class MockCollector:
    def __init__(self, cpu=20.0, mem=120.0):
        self.cpu = cpu
        self.mem = mem

    def collect(self, container_id):
        return {
            "timestamp": 123456.0,
            "container_id": container_id,
            "cpu_percent": self.cpu,
            "cpu_usage_ns": int(self.cpu * 1e7),
            "memory_usage_mb": self.mem,
            "memory_percent": round(self.mem / 10.24, 2),
            "memory_limit_mb": 1024.0,
        }


class MockController:
    def __init__(self):
        self.applied = []
        self.client = None

    def apply(self, container_id, cpu, memory):
        self.applied.append({"container_id": container_id, "cpu": cpu, "memory": memory})
        return {
            "success": True,
            "container_id": container_id,
            "applied": {"cpu": cpu, "memory": memory},
            "error": None,
        }


class TestRealModelClosedLoop(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = os.path.join(self.temp_dir.name, "test_closed_loop.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_factorized_agent_orchestrator_integration(self):
        """Test that ResourceFactorizedDQNAgent integrates seamlessly into Orchestrator."""
        config = ControlLoopConfig(
            window_size=3,
            warmup_cycles=3,
            log_file=self.log_file,
            simulation_mode=True,
        )
        agent = ResourceFactorizedDQNAgent(ResourceFactorizedAgentConfig(seed=42))

        # Use mock predictor and detector
        class DummyPredictor:
            def predict(self, seq):
                return {"cpu": 25.0, "memory": 200.0, "confidence": 0.85}

        class DummyDetector:
            def score(self, seq):
                return {"score": 0.02, "is_anomaly": False}

        collector = MockCollector(cpu=15.0, mem=100.0)
        controller = MockController()

        orchestrator = Orchestrator(
            config=config,
            collector=collector,
            predictor=DummyPredictor(),
            detector=DummyDetector(),
            agent=agent,
            controller=controller,
            action_space=agent.action_space,
            state_builder=StateBuilder(StateConfig()),
            reward_calculator=RewardCalculator(RewardConfig()),
        )

        # Run cycles 1 and 2 (warming up, len < 3)
        res1 = orchestrator.run_cycle("dummy-container")
        self.assertEqual(res1["status"], "warming_up")
        res2 = orchestrator.run_cycle("dummy-container")
        self.assertEqual(res2["status"], "warming_up")

        # Run cycle 3 (active, len == 3)
        res3 = orchestrator.run_cycle("dummy-container")
        self.assertEqual(res3["status"], "active")
        self.assertIn("action_index", res3)
        self.assertIn("cpu_action_index", res3)
        self.assertIn("mem_action_index", res3)

        # Verify joint index identity: joint_index == cpu_idx * 4 + mem_idx
        expected_joint = res3["cpu_action_index"] * 4 + res3["mem_action_index"]
        self.assertEqual(res3["action_index"], expected_joint)

        # Verify selected allocation matches action dict
        action_dict = agent.action_space.get_action(res3["action_index"])
        self.assertEqual(res3["selected_cpu_allocation"], action_dict["cpu"])
        self.assertEqual(res3["selected_memory_allocation"], action_dict["memory"])

    def test_real_model_checkpoints_exist_and_load(self):
        """Verify that all validated production checkpoints exist and load without stub fallback."""
        required = [
            ("Transformer", "checkpoints/best_transformer.pt", TransformerPredictor),
            ("Autoencoder", "checkpoints/best_autoencoder.pt", AutoencoderAnomalyDetector),
            ("ResourceFactorizedDQN", "checkpoints/best_resource_factorized_dqn.pt", ResourceFactorizedDQNAgent),
        ]
        for name, path, cls in required:
            self.assertTrue(os.path.exists(path), f"Checkpoint missing: {path}")
            if hasattr(cls, "from_checkpoint"):
                model = cls.from_checkpoint(path)
            else:
                model = cls.load(path)
            self.assertIsNotNone(model, f"Failed to load {name} from {path}")

    def test_real_inference_pipeline_execution(self):
        """Verify real Transformer + Autoencoder + Factorized DQN sequential inference."""
        transformer = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")
        autoencoder = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")
        agent = ResourceFactorizedDQNAgent.load("checkpoints/best_resource_factorized_dqn.pt")
        sb = StateBuilder(StateConfig())

        # Create a valid 12x4 sequence
        seq = np.zeros((12, 4), dtype=np.float32)
        seq[:, 0] = 30.0    # cpu_percent
        seq[:, 1] = 3e8     # cpu_usage_ns
        seq[:, 2] = 200.0   # memory_usage_mb
        seq[:, 3] = 20.0    # memory_percent

        pred = transformer.predict(seq)
        self.assertIn("cpu", pred)
        self.assertIn("memory", pred)
        self.assertIn("confidence", pred)
        self.assertFalse(np.isnan(pred["cpu"]))
        self.assertFalse(np.isnan(pred["memory"]))

        anom = autoencoder.score(seq)
        self.assertIn("score", anom)
        self.assertIn("is_anomaly", anom)

        raw_state = {
            "current_cpu_util": 30.0,
            "current_mem_util": 200.0,
            "predicted_cpu_demand": pred["cpu"],
            "predicted_mem_demand": pred["memory"],
            "current_cpu_alloc": 1.0,
            "current_mem_alloc": 256.0,
            "sla_latency": 50.0,
            "anomaly_score": anom["score"],
            "is_anomaly": anom["is_anomaly"],
            "host_cpu_avail": 4.0,
            "host_mem_avail": 4096.0,
        }
        state_vec = sb.build_state(raw_state)
        self.assertEqual(state_vec.shape, (11,))

        action_idx = agent.choose_action_index(state_vec, evaluate=True)
        self.assertIn(action_idx, range(16))
        action = agent.action_space.get_action(action_idx)
        self.assertIn(action["cpu"], [0.25, 0.5, 1.0, 2.0])
        self.assertIn(action["memory"], [128, 256, 512, 1024])


if __name__ == "__main__":
    unittest.main()
