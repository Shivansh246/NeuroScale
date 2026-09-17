"""Tests for the NeuroScale data pipeline.

Covers loading, validation, resampling, normalization, and sliding window
generation. Uses in‑memory synthetic data; no Docker interaction.
"""

import unittest
import json
import os
import tempfile
import numpy as np

from data.config import PipelineConfig
from data.validation import validate_record, validate_records
from data.preprocessing import load_jsonl, resample_records
from data.normalization import Normalizer
from data.dataset import load_dataset

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic raw records spanning 60 seconds (12 samples at 5s interval)
        self.cfg = PipelineConfig()
        self.raw_records = []
        ts = 0
        for i in range(15):  # more than needed for windows
            rec = {
                "timestamp": ts,
                "container_id": "c1",
                "container_name": "test",
                "cpu_percent": 10 + i,
                "cpu_usage_ns": 1e9 * i,
                "memory_usage_bytes": 200 * 1024 * 1024 + i * 1024 * 1024,
                "memory_usage_mb": 200 + i,
                "memory_limit_bytes": 1024 * 1024 * 1024,
                "memory_limit_mb": 1024,
                "memory_percent": 20 + i,
                "workload_mode": "cpu"
            }
            self.raw_records.append(rec)
            ts += 3  # non‑aligned interval to test resampling
        # Write to a temporary JSONL file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.raw_path = os.path.join(self.tmp_dir.name, "raw.jsonl")
        with open(self.raw_path, "w", encoding="utf-8") as f:
            for r in self.raw_records:
                f.write(json.dumps(r) + "\n")
        # Override paths in config to point to temp files
        self.cfg.raw_data_path = self.raw_path
        self.cfg.processed_data_path = os.path.join(self.tmp_dir.name, "processed.jsonl")
        self.cfg.normalization_path = os.path.join(self.tmp_dir.name, "norm.json")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_validation_passes(self):
        # Should not raise
        validate_records(self.raw_records)
        # Corrupt a record and ensure ValueError
        bad = dict(self.raw_records[0])
        bad.pop("timestamp")
        with self.assertRaises(ValueError):
            validate_record(bad)

    def test_resampling(self):
        loaded = load_jsonl(self.raw_path)
        self.assertEqual(len(loaded), len(self.raw_records))
        resampled = resample_records(loaded, interval_seconds=self.cfg.sampling_interval_seconds)
        # Expected number of samples = floor((last_ts - first_ts)/interval)+1
        expected = ((self.raw_records[-1]["timestamp"] // self.cfg.sampling_interval_seconds) -
                    (self.raw_records[0]["timestamp"] // self.cfg.sampling_interval_seconds) + 1)
        self.assertEqual(len(resampled), expected)
        # Timestamps should be multiples of interval
        for r in resampled:
            self.assertEqual(r["timestamp"] % self.cfg.sampling_interval_seconds, 0)

    def test_resampling_float_timestamps(self):
        records = [
            dict(self.raw_records[0], timestamp=100.2),
            dict(self.raw_records[1], timestamp=103.7),
            dict(self.raw_records[2], timestamp=107.9),
        ]
        resampled = resample_records(records, interval_seconds=5)
        self.assertTrue(len(resampled) > 0)
        for r in resampled:
            self.assertIsInstance(r["timestamp"], int)
            self.assertEqual(r["timestamp"] % 5, 0)

    def test_normalizer_fit_transform(self):
        loaded = load_jsonl(self.raw_path)
        resampled = resample_records(loaded)
        # Extract feature array directly
        feature_cols = self.cfg.feature_columns
        feats = np.array([[r[col] for col in feature_cols] for r in resampled], dtype=float)
        norm = Normalizer().fit(feats)
        transformed = norm.transform(feats)
        # Mean of transformed should be approx 0, std approx 1
        self.assertTrue(np.allclose(transformed.mean(axis=0), 0, atol=1e-6))
        self.assertTrue(np.allclose(transformed.std(axis=0), 1, atol=1e-6))
        # Save and load round‑trip
        norm.save(self.cfg.normalization_path)
        new_norm = Normalizer().load(self.cfg.normalization_path)
        np.testing.assert_allclose(new_norm.mean_, norm.mean_)
        np.testing.assert_allclose(new_norm.std_, norm.std_)

    def test_full_dataset(self):
        ds = load_dataset(self.cfg)
        # Verify each split has windows
        for split_name in ["train", "val", "test"]:
            windows = ds[split_name]
            self.assertIsInstance(windows, list)
            if windows:  # may be empty if not enough data for split
                inp, tgt = windows[0]
                self.assertEqual(inp.shape[0], self.cfg.input_window)
                self.assertEqual(tgt.shape[0], self.cfg.prediction_horizon)
                self.assertEqual(inp.shape[1], len(self.cfg.feature_columns))
                self.assertEqual(tgt.shape[1], len(self.cfg.target_columns))

if __name__ == "__main__":
    unittest.main()
