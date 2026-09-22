import unittest
import numpy as np
import torch
import os

from experiments.prediction_models.data import process_features, create_windows, chronological_split, Normalizer
from experiments.prediction_models.gru_model import GRUModel
from experiments.prediction_models.xgboost_model import flatten_inputs
from experiments.prediction_models.metrics import compute_metrics
from experiments.prediction_models.transformer_model import train_transformer
from models.config import TransformerConfig
from models.transformer import NeuroScaleTransformer

class TestPredictionModels(unittest.TestCase):

    def setUp(self):
        # Create a dummy dataset
        self.records = [
            {"cpu_percent": float(i), "cpu_usage_ns_delta": float(i*1000), "memory_usage_mb": float(i*2), "memory_percent": float(i*0.1), "workload_mode": "idle", "timestamp": float(i)}
            for i in range(100)
        ]

    def test_process_features(self):
        features, targets, modes, timestamps = process_features(self.records)
        self.assertEqual(len(features), 100)
        self.assertEqual(features.shape[1], 4)
        self.assertEqual(targets.shape[1], 2)
        # log1p applied correctly
        self.assertAlmostEqual(features[1, 1], np.log1p(1000.0))

    def test_create_windows(self):
        features, targets, modes, timestamps = process_features(self.records)
        X, y, y_modes, y_timestamps, win_modes = create_windows(features, targets, modes, timestamps, seq_len=12, horizon=1)
        # 100 - 12 - 1 + 1 = 88 windows
        self.assertEqual(len(X), 88)
        self.assertEqual(X.shape[1], 12)
        self.assertEqual(y.shape[1], 2)

    def test_chronological_split_leakage(self):
        features, targets, modes, timestamps = process_features(self.records)
        X, y, y_modes, y_timestamps, win_modes = create_windows(features, targets, modes, timestamps, seq_len=12, horizon=1)
        
        splits = chronological_split(X, y, y_modes, y_timestamps, win_modes, train_ratio=0.7, val_ratio=0.15)
        train_X = splits["train"][0]
        val_X = splits["val"][0]
        
        # We need to ensure that the last target of train doesn't leak into the first input of val
        train_last_target_timestamp = splits["train"][3][-1]
        val_first_input_timestamp = val_X[0, 0, 0] # the feature 'cpu_percent' matches timestamp
        
        # Target of train is at index (end_idx - 1 + 12)
        # So val first input timestamp must be > train_last_target_timestamp
        self.assertGreater(val_first_input_timestamp, train_last_target_timestamp)

    def test_normalization(self):
        norm = Normalizer()
        X_train = np.array([[[1.0, 2.0, 3.0, 4.0]]])
        y_train = np.array([[10.0, 20.0]])
        norm.fit(X_train, y_train)
        
        # Test 0 division handling / scaling
        X_norm = norm.transform_x(X_train)
        y_norm = norm.transform_y(y_train)
        self.assertEqual(X_norm.shape, X_train.shape)
        self.assertEqual(y_norm.shape, y_train.shape)

    def test_transformer_forward_shape(self):
        cfg = TransformerConfig(feature_count=4, num_targets=2, prediction_horizon=1)
        model = NeuroScaleTransformer(cfg)
        dummy_input = torch.zeros(2, 12, 4)
        out = model(dummy_input)
        self.assertEqual(out.shape, (2, 1, 2))

    def test_gru_forward_shape(self):
        model = GRUModel(input_size=4, hidden_size=64, num_layers=2, output_size=2)
        dummy_input = torch.zeros(2, 12, 4)
        out = model(dummy_input)
        self.assertEqual(out.shape, (2, 2))

    def test_xgboost_prediction_shape(self):
        X = np.zeros((2, 12, 4))
        X_flat = flatten_inputs(X)
        self.assertEqual(X_flat.shape, (2, 48))

    def test_metric_calculation(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.0])
        metrics = compute_metrics(y_true, y_pred)
        self.assertTrue("mae" in metrics)
        self.assertTrue("rmse" in metrics)
        self.assertTrue("r2" in metrics)

    def test_deterministic_behavior(self):
        torch.manual_seed(42)
        model1 = GRUModel()
        torch.manual_seed(42)
        model2 = GRUModel()
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

if __name__ == '__main__':
    unittest.main()
