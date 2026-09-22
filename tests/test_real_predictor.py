import unittest
import os
import torch
import numpy as np

from models.predictor import TransformerPredictor

class TestRealPredictor(unittest.TestCase):
    
    def test_production_predictor_remains_functional(self):
        # Assumes checkpoints/best_transformer.pt exists
        if os.path.exists("checkpoints/best_transformer.pt"):
            predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")
            seq = np.zeros((12, 4))
            res = predictor.predict(seq)
            self.assertTrue("cpu" in res)
            self.assertTrue("memory" in res)
            self.assertTrue("confidence" in res)
            
    def test_real_checkpoint_loading(self):
        # Assumes checkpoints/best_transformer_real.pt exists
        predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer_real.pt")
        self.assertIsNotNone(predictor)
        
    def test_correct_normalization_loading(self):
        predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer_real.pt")
        self.assertIsNotNone(predictor._feat_normalizer.mean_)
        self.assertIsNotNone(predictor._feat_normalizer.std_)
        self.assertIn("cpu_percent", predictor._target_norm_map)
        self.assertIn("memory_usage_mb", predictor._target_norm_map)
        
    def test_predictor_output_schema(self):
        predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer_real.pt")
        seq = np.zeros((12, 4))
        res = predictor.predict(seq)
        self.assertIsInstance(res, dict)
        self.assertIn("cpu", res)
        self.assertIn("memory", res)
        self.assertIn("confidence", res)
        self.assertIsInstance(res["cpu"], float)
        self.assertIsInstance(res["memory"], float)
        self.assertIsInstance(res["confidence"], float)

    def test_input_sequence_shape(self):
        predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer_real.pt")
        with self.assertRaises(ValueError):
            predictor.predict(np.zeros((12, 4, 1))) # 3D
        with self.assertRaises(ValueError):
            predictor.predict(np.zeros((12,))) # 1D
            
    def test_deterministic_prediction(self):
        predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer_real.pt")
        seq = np.random.rand(12, 4)
        res1 = predictor.predict(seq)
        res2 = predictor.predict(seq)
        self.assertEqual(res1["cpu"], res2["cpu"])
        self.assertEqual(res1["memory"], res2["memory"])
        
    def test_missing_checkpoint_failure(self):
        with self.assertRaises(FileNotFoundError):
            TransformerPredictor.from_checkpoint("checkpoints/non_existent.pt")

if __name__ == '__main__':
    unittest.main()
