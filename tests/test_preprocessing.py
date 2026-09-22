import unittest
import numpy as np

from control.config import ControlLoopConfig
from control.loop import Orchestrator

class DummyCollector:
    def collect(self, container_id):
        return None

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.config = ControlLoopConfig()
        self.config.feature_columns = ("cpu_percent", "cpu_usage_ns_delta", "memory_usage_mb", "memory_percent")
        
        # Instantiate with dummy dependencies since we only test sequence methods
        self.orchestrator = Orchestrator(self.config, DummyCollector(), None, None, None, None, None, None, None)

    def test_preprocessing_paths(self):
        # Create a canonical sequence representing one timestep (1, 4)
        # raw CPU delta is 5e9
        canonical_seq = np.array([[100.0, 5_000_000_000.0, 256.0, 50.0]], dtype=np.float32)
        
        # Make a deep copy to test immutability
        original_canonical = canonical_seq.copy()
        
        trans_seq = self.orchestrator._prepare_transformer_sequence(canonical_seq)
        anom_seq = self.orchestrator._prepare_anomaly_sequence(canonical_seq)
        
        # 1. Large raw CPU delta remains unchanged in canonical telemetry
        # 9. Canonical telemetry/history is not mutated
        np.testing.assert_array_equal(canonical_seq, original_canonical)
        
        # 2. Transformer preprocessing produces finite values
        self.assertTrue(np.all(np.isfinite(trans_seq)))
        
        # 3. Transformer receives exact representation (log1p applied exactly once)
        # 4. log1p applied exactly once
        expected_log1p = np.log1p(5_000_000_000.0)
        self.assertAlmostEqual(trans_seq[0, 1], expected_log1p, places=5)
        
        # 5. Autoencoder receives RAW CPU delta
        self.assertEqual(anom_seq[0, 1], 5_000_000_000.0)
        
        # 6. CPU percentage remains unchanged
        self.assertEqual(trans_seq[0, 0], 100.0)
        self.assertEqual(anom_seq[0, 0], 100.0)
        
        # 7. Memory usage remains unchanged
        self.assertEqual(trans_seq[0, 2], 256.0)
        self.assertEqual(anom_seq[0, 2], 256.0)
        
        # 8. Memory percentage remains unchanged
        self.assertEqual(trans_seq[0, 3], 50.0)
        self.assertEqual(anom_seq[0, 3], 50.0)
        
        # 10. Transformer and Autoencoder receive different CPU-delta representations
        self.assertNotEqual(trans_seq[0, 1], anom_seq[0, 1])

if __name__ == '__main__':
    unittest.main()
