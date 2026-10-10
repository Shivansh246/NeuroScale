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
        
        # 2. Transformer and Autoencoder preprocessing produce finite values
        self.assertTrue(np.all(np.isfinite(trans_seq)))
        self.assertTrue(np.all(np.isfinite(anom_seq)))
        
        # 3. Transformer receives exact representation (log1p applied exactly once)
        # 4. log1p applied exactly once
        expected_log1p = np.log1p(5_000_000_000.0)
        self.assertAlmostEqual(trans_seq[0, 1], expected_log1p, places=5)
        
        # 5. Autoencoder receives unified representation identical to Transformer
        self.assertAlmostEqual(anom_seq[0, 1], expected_log1p, places=5)
        
        # 6. CPU percentage remains unchanged
        self.assertEqual(trans_seq[0, 0], 100.0)
        self.assertEqual(anom_seq[0, 0], 100.0)
        
        # 7. Memory usage remains unchanged
        self.assertEqual(trans_seq[0, 2], 256.0)
        self.assertEqual(anom_seq[0, 2], 256.0)
        
        # 8. Memory percentage remains unchanged
        self.assertEqual(trans_seq[0, 3], 50.0)
        self.assertEqual(anom_seq[0, 3], 50.0)
        
        # 10. Transformer and Autoencoder receive IDENTICAL feature representations (unified pipeline)
        self.assertEqual(trans_seq[0, 1], anom_seq[0, 1])
        np.testing.assert_array_equal(trans_seq, anom_seq)

    def test_telemetry_feature_pipeline_standalone(self):
        from data.preprocessing import TelemetryFeaturePipeline
        pipe = TelemetryFeaturePipeline()
        
        # Test normal delta computation
        m1 = {"cpu_usage_ns": 1000, "cpu_percent": 10.0, "memory_usage_mb": 50.0, "memory_percent": 5.0}
        m2 = {"cpu_usage_ns": 1500, "cpu_percent": 15.0, "memory_usage_mb": 55.0, "memory_percent": 5.5}
        d1, r1 = pipe.compute_delta(m1)
        d2, r2 = pipe.compute_delta(m2)
        
        self.assertEqual(d1, 0)
        self.assertFalse(r1)
        self.assertEqual(d2, 500)
        self.assertFalse(r2)
        self.assertEqual(m1["cpu_usage_ns"], 1000)
        self.assertEqual(m2["cpu_usage_ns"], 1500)
        self.assertEqual(m1["cpu_usage_ns_delta"], 0)
        self.assertEqual(m2["cpu_usage_ns_delta"], 500)
        
        # Test counter reset handling
        m3 = {"cpu_usage_ns": 200, "cpu_percent": 20.0, "memory_usage_mb": 60.0, "memory_percent": 6.0}
        d3, r3 = pipe.compute_delta(m3)
        self.assertTrue(r3)
        self.assertEqual(d3, 200)
        self.assertEqual(m3["cpu_usage_ns_delta"], 200)
        
        # Test feature row log1p transformation
        row2 = pipe.extract_feature_row(m2)
        import math
        self.assertAlmostEqual(row2[1], math.log1p(500.0), places=5)
        self.assertEqual(row2[0], 15.0)
        self.assertEqual(row2[2], 55.0)
        self.assertEqual(row2[3], 5.5)

if __name__ == '__main__':
    unittest.main()
