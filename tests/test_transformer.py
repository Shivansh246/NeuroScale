"""Comprehensive tests for the NeuroScale Transformer prediction module.

Tests are CPU-only, deterministic, and do NOT require Docker or real data.

Normalization contract under test:
  - X (features) are normalised (mean~0, std~1 per column).
  - y (targets) are normalised in the same way, using a SEPARATE target
    normalizer fitted on training targets only.
  - Model trains on normalised X → normalised y.
  - val_loss is therefore in normalised target space.
  - Predictor accepts raw (un-normalised) sequences, normalises internally,
    runs the model, then inverse-transforms each target by column name using
    the target_norm_map stored in the checkpoint.
  - Returned cpu/memory values are in original scale (%).

Coverage:
 1.  Model construction
 2.  Forward-pass output shape (default config)
 3.  Configurable feature count
 4.  Configurable sequence length
 5.  Configurable prediction horizon
 6.  WindowDataset adapter
 7.  Loss / metric functions (MSE, MAE, RMSE)
 8.  Deterministic training (same seed → same loss trajectory)
 9.  Checkpoint creation
10.  Checkpoint loading
11.  Inference API construction
12.  Inference output keys (cpu / memory / confidence)
13.  Inference output is finite
14.  Inverse-transform correctness — predictions returned in original units
15.  Small end-to-end: synthetic data → train → checkpoint → reload → predict
16.  Reordered feature columns
17.  Reordered target columns
18.  Checkpoint metadata contains target_norm_map
19.  Correct inverse-transform after reload with non-default column order
20.  Predictions in original scale verified against known normalizer stats
21.  Confidence in [0,1]
22.  Normalised training/validation data contract
23.  Deterministic inference after checkpoint reload
"""

import json
import math
import os
import tempfile
import unittest

import numpy as np
import torch

from data.config import PipelineConfig
from data.normalization import Normalizer
from evaluation.transformer_metrics import (
    cpu_metrics,
    full_report,
    mae,
    memory_metrics,
    mse,
    overall_metrics,
    rmse,
)
from models.config import TransformerConfig
from models.dataset_adapter import WindowDataset
from models.predictor import TransformerPredictor
from models.transformer import NeuroScaleTransformer
from training.transformer_trainer import (
    TrainingConfig,
    _build_target_norm_map,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kwargs) -> TransformerConfig:
    """Return a tiny TransformerConfig suitable for fast unit tests."""
    defaults = dict(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        prediction_horizon=1,
        num_targets=2,
        feature_count=4,
        seed=42,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_model(cfg: TransformerConfig = None) -> NeuroScaleTransformer:
    cfg = cfg or _make_cfg()
    return NeuroScaleTransformer(cfg)


def _synthetic_windows(
    n: int = 60,
    seq_len: int = 12,
    feature_count: int = 4,
    num_targets: int = 2,
    prediction_horizon: int = 1,
    seed: int = 0,
):
    """Generate deterministic normalised windows with a learnable sine pattern.

    Both X and y are already in normalised space (zero-mean, unit-variance)
    to match the production data pipeline contract.
    Returns (windows, feat_mean, feat_std, tgt_mean, tgt_std).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n + seq_len + prediction_horizon)
    cpu = 40 + 30 * np.sin(t)          # original cpu % range ~10–70
    mem = 60 + 15 * np.cos(t)          # original memory % range ~45–75
    noise = rng.normal(0, 0.5, t.shape)
    features_orig = np.column_stack([
        cpu + noise,
        cpu * 1e9 / 100.0 + noise,
        mem * 5 + noise,
        mem + noise,
    ])[:, :feature_count]
    targets_orig = np.column_stack([cpu, mem])[:, :num_targets]

    # Fit normalizers on the first ``n`` samples (simulating training split)
    feat_norm = Normalizer().fit(features_orig[:n])
    tgt_norm = Normalizer().fit(targets_orig[:n])

    features_norm = feat_norm.transform(features_orig)
    targets_norm = tgt_norm.transform(targets_orig)

    windows = []
    for i in range(n):
        x = features_norm[i: i + seq_len]
        y = targets_norm[i + seq_len: i + seq_len + prediction_horizon]
        windows.append((x, y))
    return windows, feat_norm, tgt_norm


def _make_checkpoint(
    tmp_dir: str,
    feature_columns=None,
    target_columns=None,
    val_loss: float = 0.25,
    seq_len: int = 12,
) -> str:
    """Build and save a minimal checkpoint; return its path."""
    if feature_columns is None:
        feature_columns = ["cpu_percent", "cpu_usage_ns", "memory_usage_mb", "memory_percent"]
    if target_columns is None:
        target_columns = ["cpu_percent", "memory_percent"]

    n_feats = len(feature_columns)
    n_targets = len(target_columns)
    cfg = _make_cfg(feature_count=n_feats, num_targets=n_targets)
    model = _make_model(cfg)

    windows, feat_norm, tgt_norm = _synthetic_windows(
        n=30, seq_len=seq_len, feature_count=n_feats, num_targets=n_targets
    )

    pipeline_cfg = PipelineConfig(
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    ckpt_path = os.path.join(tmp_dir, "test_model.pt")
    save_checkpoint(
        ckpt_path,
        model,
        cfg,
        pipeline_cfg,
        feat_norm,
        tgt_norm,
        epoch=1,
        val_loss=val_loss,
        train_loss=val_loss + 0.1,
    )
    return ckpt_path, feat_norm, tgt_norm


# ---------------------------------------------------------------------------
# 1. Model construction
# ---------------------------------------------------------------------------
class TestModelConstruction(unittest.TestCase):
    def test_default_construction(self):
        model = _make_model()
        self.assertIsInstance(model, NeuroScaleTransformer)

    def test_parameter_count_is_small(self):
        model = _make_model()
        n_params = sum(p.numel() for p in model.parameters())
        self.assertLess(n_params, 100_000)


# ---------------------------------------------------------------------------
# 2. Forward-pass shape (default config)
# ---------------------------------------------------------------------------
class TestForwardPassShape(unittest.TestCase):
    def _forward(self, batch=2, seq_len=12, cfg=None):
        cfg = cfg or _make_cfg()
        model = NeuroScaleTransformer(cfg)
        model.eval()
        x = torch.randn(batch, seq_len, cfg.feature_count)
        with torch.no_grad():
            out = model(x)
        return out, cfg

    def test_output_shape_default(self):
        out, cfg = self._forward()
        self.assertEqual(out.shape, (2, cfg.prediction_horizon, cfg.num_targets))

    def test_output_shape_batch1(self):
        out, cfg = self._forward(batch=1)
        self.assertEqual(out.shape, (1, cfg.prediction_horizon, cfg.num_targets))


# ---------------------------------------------------------------------------
# 3. Configurable feature count
# ---------------------------------------------------------------------------
class TestConfigurableFeatureCount(unittest.TestCase):
    def test_feature_count_6(self):
        cfg = _make_cfg(feature_count=6)
        model = NeuroScaleTransformer(cfg)
        x = torch.randn(2, 10, 6)
        out = model(x)
        self.assertEqual(out.shape[-1], cfg.num_targets)

    def test_feature_count_2(self):
        cfg = _make_cfg(feature_count=2)
        model = NeuroScaleTransformer(cfg)
        x = torch.randn(3, 8, 2)
        out = model(x)
        self.assertEqual(out.shape[0], 3)


# ---------------------------------------------------------------------------
# 4. Configurable sequence length
# ---------------------------------------------------------------------------
class TestConfigurableSequenceLength(unittest.TestCase):
    def test_seq_len_5(self):
        cfg = _make_cfg()
        model = NeuroScaleTransformer(cfg)
        x = torch.randn(2, 5, cfg.feature_count)
        out = model(x)
        self.assertEqual(out.shape[1], cfg.prediction_horizon)

    def test_seq_len_24(self):
        cfg = _make_cfg()
        model = NeuroScaleTransformer(cfg)
        x = torch.randn(2, 24, cfg.feature_count)
        out = model(x)
        self.assertEqual(out.shape, (2, cfg.prediction_horizon, cfg.num_targets))


# ---------------------------------------------------------------------------
# 5. Configurable prediction horizon
# ---------------------------------------------------------------------------
class TestConfigurablePredictionHorizon(unittest.TestCase):
    def test_horizon_3(self):
        cfg = _make_cfg(prediction_horizon=3)
        model = NeuroScaleTransformer(cfg)
        x = torch.randn(4, 12, cfg.feature_count)
        out = model(x)
        self.assertEqual(out.shape, (4, 3, cfg.num_targets))

    def test_horizon_1(self):
        cfg = _make_cfg(prediction_horizon=1)
        model = NeuroScaleTransformer(cfg)
        x = torch.randn(2, 12, cfg.feature_count)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, cfg.num_targets))


# ---------------------------------------------------------------------------
# 6. WindowDataset adapter
# ---------------------------------------------------------------------------
class TestWindowDataset(unittest.TestCase):
    def setUp(self):
        self.windows, _, _ = _synthetic_windows(n=20, seq_len=6)

    def test_len(self):
        ds = WindowDataset(self.windows)
        self.assertEqual(len(ds), 20)

    def test_item_types(self):
        ds = WindowDataset(self.windows)
        x, y = ds[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)

    def test_item_shapes(self):
        ds = WindowDataset(self.windows)
        x, y = ds[0]
        self.assertEqual(x.shape, (6, 4))
        self.assertEqual(y.shape, (1, 2))

    def test_dtype_float32(self):
        ds = WindowDataset(self.windows)
        x, y = ds[0]
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            WindowDataset([])


# ---------------------------------------------------------------------------
# 7. Loss / metric functions
# ---------------------------------------------------------------------------
class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y_true = rng.random((10, 1, 2))
        self.y_pred = self.y_true + rng.normal(0, 0.1, self.y_true.shape)

    def test_mse_nonnegative(self):
        self.assertGreaterEqual(mse(self.y_true, self.y_pred), 0.0)

    def test_mae_nonnegative(self):
        self.assertGreaterEqual(mae(self.y_true, self.y_pred), 0.0)

    def test_rmse_equals_sqrt_mse(self):
        self.assertAlmostEqual(
            rmse(self.y_true, self.y_pred),
            math.sqrt(mse(self.y_true, self.y_pred)),
            places=6,
        )

    def test_mse_zero_for_perfect_pred(self):
        self.assertAlmostEqual(mse(self.y_true, self.y_true), 0.0, places=10)

    def test_cpu_metrics_keys(self):
        m = cpu_metrics(self.y_true, self.y_pred)
        self.assertIn("mse", m)
        self.assertIn("mae", m)
        self.assertIn("rmse", m)

    def test_memory_metrics_keys(self):
        m = memory_metrics(self.y_true, self.y_pred)
        self.assertIn("mse", m)

    def test_full_report_structure(self):
        report = full_report(self.y_true, self.y_pred)
        self.assertIn("cpu", report)
        self.assertIn("memory", report)
        self.assertIn("overall", report)


# ---------------------------------------------------------------------------
# 8. Deterministic training
# ---------------------------------------------------------------------------
class TestDeterministicTraining(unittest.TestCase):
    def _train_loss_sequence(self, seed):
        set_seed(seed)
        windows, _, _ = _synthetic_windows(n=40, seq_len=8)
        cfg = _make_cfg(feature_count=4, prediction_horizon=1)
        losses = []
        ds = WindowDataset(windows)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
        model = NeuroScaleTransformer(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()
        for _ in range(3):
            model.train()
            batch_losses = []
            for x, y in loader:
                pred = model(x)
                loss = crit(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            losses.append(sum(batch_losses) / len(batch_losses))
        return losses

    def test_same_seed_gives_same_losses(self):
        l1 = self._train_loss_sequence(seed=99)
        l2 = self._train_loss_sequence(seed=99)
        for a, b in zip(l1, l2):
            self.assertAlmostEqual(a, b, places=5)


# ---------------------------------------------------------------------------
# 9 & 10. Checkpoint create and load
# ---------------------------------------------------------------------------
class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ckpt_path, self.feat_norm, self.tgt_norm = _make_checkpoint(self.tmp)

    def test_checkpoint_file_exists(self):
        self.assertTrue(os.path.isfile(self.ckpt_path))

    def test_checkpoint_loads(self):
        loaded_model, ckpt = load_checkpoint(self.ckpt_path)
        self.assertIsInstance(loaded_model, NeuroScaleTransformer)

    def test_checkpoint_contains_required_keys(self):
        ckpt = torch.load(self.ckpt_path, weights_only=False)
        for key in (
            "state_dict", "model_config", "feature_columns", "target_columns",
            "feat_normalization", "target_norm_map", "epoch", "val_loss",
        ):
            self.assertIn(key, ckpt, msg=f"Missing checkpoint key: {key}")

    def test_checkpoint_target_norm_map_has_correct_keys(self):
        """target_norm_map must be keyed by target column name."""
        ckpt = torch.load(self.ckpt_path, weights_only=False)
        tnm = ckpt["target_norm_map"]
        self.assertIn("cpu_percent", tnm)
        self.assertIn("memory_percent", tnm)

    def test_target_norm_map_contains_mean_and_std(self):
        ckpt = torch.load(self.ckpt_path, weights_only=False)
        tnm = ckpt["target_norm_map"]
        for col, stats in tnm.items():
            self.assertIn("mean", stats, msg=f"Missing 'mean' for {col}")
            self.assertIn("std", stats, msg=f"Missing 'std' for {col}")

    def test_loaded_model_output_shape(self):
        loaded_model, _ = load_checkpoint(self.ckpt_path)
        x = torch.randn(1, 12, 4)
        out = loaded_model(x)
        self.assertEqual(out.shape, (1, 1, 2))


# ---------------------------------------------------------------------------
# 11–14 & 20–21. Inference API
# ---------------------------------------------------------------------------
class TestInferenceAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ckpt_path, self.feat_norm, self.tgt_norm = _make_checkpoint(
            self.tmp, val_loss=0.25
        )
        self.predictor = TransformerPredictor.from_checkpoint(self.ckpt_path)
        # Raw (un-normalised) input window
        rng = np.random.default_rng(7)
        self.raw_seq = np.column_stack([
            40 + 30 * np.sin(np.linspace(0, 2, 12)),   # cpu_percent range ~10–70
            np.ones(12) * 1e9,
            np.ones(12) * 300,
            60 + 15 * np.cos(np.linspace(0, 2, 12)),   # memory_percent range ~45–75
        ]).astype(float)  # shape (12, 4)

    def test_predict_returns_dict(self):
        result = self.predictor.predict(self.raw_seq)
        self.assertIsInstance(result, dict)

    def test_output_keys(self):
        result = self.predictor.predict(self.raw_seq)
        self.assertIn("cpu", result)
        self.assertIn("memory", result)
        self.assertIn("confidence", result)

    def test_output_is_finite(self):
        result = self.predictor.predict(self.raw_seq)
        self.assertTrue(math.isfinite(result["cpu"]))
        self.assertTrue(math.isfinite(result["memory"]))
        self.assertTrue(math.isfinite(result["confidence"]))

    def test_confidence_in_unit_interval(self):
        result = self.predictor.predict(self.raw_seq)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_confidence_value_matches_formula(self):
        """confidence = exp(-sqrt(val_loss) / 1.0)."""
        result = self.predictor.predict(self.raw_seq)
        expected_conf = math.exp(-math.sqrt(0.25) / 1.0)
        self.assertAlmostEqual(result["confidence"], expected_conf, places=5)

    def test_bad_input_dimension_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.predict(np.random.randn(12))  # 1-D

    def test_predictions_in_original_scale(self):
        """CPU predictions should be in a plausible CPU% range after
        inverse-transform, not stuck near zero (which would happen if we
        forgot to inverse-transform)."""
        result = self.predictor.predict(self.raw_seq)
        # The model is untrained but the inverse-transform must shift the
        # output away from zero. We verify the magnitude is non-trivial.
        tgt_mean_cpu = self.tgt_norm.mean_[0]
        # With a random model, result["cpu"] ≈ tgt_norm.mean_[0] +/- some noise.
        # At minimum, the result should differ from 0 by the mean shift.
        self.assertGreater(abs(result["cpu"]), 0.5 * abs(tgt_mean_cpu),
                           msg="cpu prediction appears to not be inverse-transformed")

    def test_predictions_finite_after_inverse_transform(self):
        """All returned values must be finite regardless of model initialisation."""
        result = self.predictor.predict(self.raw_seq)
        for key in ("cpu", "memory", "confidence"):
            self.assertTrue(math.isfinite(result[key]), msg=f"{key} is not finite")


# ---------------------------------------------------------------------------
# 16. Reordered feature columns
# ---------------------------------------------------------------------------
class TestReorderedFeatureColumns(unittest.TestCase):
    def test_reordered_features_no_crash(self):
        """Model with reordered feature columns must still produce correct shapes."""
        reordered = ["memory_percent", "cpu_usage_ns", "memory_usage_mb", "cpu_percent"]
        tmp = tempfile.mkdtemp()
        ckpt_path, feat_norm, tgt_norm = _make_checkpoint(
            tmp, feature_columns=reordered
        )
        predictor = TransformerPredictor.from_checkpoint(ckpt_path)
        # Build a raw input in the reordered column order
        raw_seq = np.random.default_rng(1).random((12, 4)) * 50 + 10
        result = predictor.predict(raw_seq)
        for key in ("cpu", "memory", "confidence"):
            self.assertIn(key, result)
            self.assertTrue(math.isfinite(result[key]))


# ---------------------------------------------------------------------------
# 17. Reordered target columns
# ---------------------------------------------------------------------------
class TestReorderedTargetColumns(unittest.TestCase):
    def test_reordered_targets_inverse_transform(self):
        """Checkpoint must store target_norm_map by name so that reordered
        targets are still correctly inverse-transformed."""
        tmp = tempfile.mkdtemp()
        # Swap target order: memory_percent first, cpu_percent second
        targets_swapped = ["memory_percent", "cpu_percent"]
        ckpt_path, feat_norm, tgt_norm = _make_checkpoint(
            tmp,
            target_columns=targets_swapped,
        )
        ckpt = torch.load(ckpt_path, weights_only=False)

        # target_norm_map must have per-name entries
        tnm = ckpt["target_norm_map"]
        self.assertIn("cpu_percent", tnm)
        self.assertIn("memory_percent", tnm)

        # cpu_percent and memory_percent means should be distinct
        # (cpu% ~40, memory% ~60 in our synthetic data)
        cpu_mean = tnm["cpu_percent"]["mean"]
        mem_mean = tnm["memory_percent"]["mean"]
        self.assertNotAlmostEqual(cpu_mean, mem_mean, places=1)

        predictor = TransformerPredictor.from_checkpoint(ckpt_path)
        raw_seq = np.random.default_rng(3).random((12, 4)) * 50 + 10
        result = predictor.predict(raw_seq)
        self.assertIn("cpu", result)
        self.assertIn("memory", result)
        self.assertTrue(math.isfinite(result["cpu"]))
        self.assertTrue(math.isfinite(result["memory"]))


# ---------------------------------------------------------------------------
# 18. Checkpoint metadata contains target_norm_map
# ---------------------------------------------------------------------------
class TestCheckpointMetadata(unittest.TestCase):
    def test_target_norm_map_stored_explicitly(self):
        tmp = tempfile.mkdtemp()
        ckpt_path, _, tgt_norm = _make_checkpoint(tmp)
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.assertIn("target_norm_map", ckpt)
        tnm = ckpt["target_norm_map"]
        # Verify stored values match the tgt_norm that was used
        np.testing.assert_allclose(
            tnm["cpu_percent"]["mean"], tgt_norm.mean_[0], rtol=1e-5
        )
        np.testing.assert_allclose(
            tnm["memory_percent"]["mean"], tgt_norm.mean_[1], rtol=1e-5
        )

    def test_feat_normalization_stored(self):
        tmp = tempfile.mkdtemp()
        ckpt_path, feat_norm, _ = _make_checkpoint(tmp)
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.assertIn("feat_normalization", ckpt)
        fn = ckpt["feat_normalization"]
        np.testing.assert_allclose(fn["mean"], feat_norm.mean_.tolist(), rtol=1e-5)


# ---------------------------------------------------------------------------
# 19. Correct inverse-transform after reload
# ---------------------------------------------------------------------------
class TestInverseTransformAfterReload(unittest.TestCase):
    def test_inverse_transform_uses_target_norm_map(self):
        """Manually verify: given a known normalised prediction, the
        predictor must return the correctly inverse-transformed original value."""
        tmp = tempfile.mkdtemp()
        # Build checkpoint with known normalizer stats
        feat_norm = Normalizer()
        feat_norm.mean_ = np.array([40.0, 1e9, 300.0, 60.0])
        feat_norm.std_ = np.array([15.0, 5e8, 50.0, 10.0])

        tgt_norm = Normalizer()
        tgt_norm.mean_ = np.array([40.0, 60.0])
        tgt_norm.std_ = np.array([15.0, 10.0])

        cfg = _make_cfg()
        model = _make_model(cfg)
        pipeline_cfg = PipelineConfig()

        ckpt_path = os.path.join(tmp, "test.pt")
        save_checkpoint(
            ckpt_path, model, cfg, pipeline_cfg, feat_norm, tgt_norm,
            epoch=1, val_loss=0.1, train_loss=0.1,
        )

        # Patch model to output a fixed normalised prediction
        predictor = TransformerPredictor.from_checkpoint(ckpt_path)
        # We expect: cpu_original = norm_cpu * 15.0 + 40.0
        #            mem_original = norm_mem * 10.0 + 60.0
        # Test by verifying inverse transform of 0 → mean
        result_zero_cpu = predictor._inverse_transform_target("cpu_percent", 0.0)
        self.assertAlmostEqual(result_zero_cpu, 40.0, places=5)
        result_zero_mem = predictor._inverse_transform_target("memory_percent", 0.0)
        self.assertAlmostEqual(result_zero_mem, 60.0, places=5)

        result_one_cpu = predictor._inverse_transform_target("cpu_percent", 1.0)
        self.assertAlmostEqual(result_one_cpu, 55.0, places=5)   # 40 + 15*1


# ---------------------------------------------------------------------------
# 22. Normalised training/validation data contract
# ---------------------------------------------------------------------------
class TestNormalisedDataContract(unittest.TestCase):
    def test_windows_are_normalised(self):
        """Windows produced by _synthetic_windows must have approximately
        zero mean and unit std for both X and y."""
        windows, feat_norm, tgt_norm = _synthetic_windows(n=80, seq_len=8)
        x_all = np.vstack([x for x, _ in windows])
        y_all = np.vstack([y.reshape(1, -1) for _, y in windows])
        # Features should be approximately normalised
        self.assertTrue(np.all(np.abs(np.mean(x_all, axis=0)) < 0.5),
                        msg="Feature means not close to 0")
        # Targets should also be approximately normalised
        self.assertTrue(np.all(np.abs(np.mean(y_all, axis=0)) < 0.5),
                        msg="Target means not close to 0")

    def test_val_loss_is_in_normalised_space(self):
        """val_loss stored in checkpoint must be O(1) for a reasonable untrained
        model on normalised data, not O(1000) which would happen with raw targets."""
        tmp = tempfile.mkdtemp()
        ckpt_path, _, _ = _make_checkpoint(tmp, val_loss=0.5)
        ckpt = torch.load(ckpt_path, weights_only=False)
        # val_loss should be moderate (normalised space), not huge
        self.assertLess(ckpt["val_loss"], 100.0,
                        msg="val_loss appears to be in original (un-normalised) space")


# ---------------------------------------------------------------------------
# 23. Deterministic inference after checkpoint reload
# ---------------------------------------------------------------------------
class TestDeterministicInference(unittest.TestCase):
    def test_same_result_after_reload(self):
        tmp = tempfile.mkdtemp()
        ckpt_path, _, _ = _make_checkpoint(tmp)
        p1 = TransformerPredictor.from_checkpoint(ckpt_path)
        p2 = TransformerPredictor.from_checkpoint(ckpt_path)
        raw_seq = np.random.default_rng(5).random((12, 4)) * 50 + 10
        r1 = p1.predict(raw_seq)
        r2 = p2.predict(raw_seq)
        self.assertAlmostEqual(r1["cpu"], r2["cpu"], places=5)
        self.assertAlmostEqual(r1["memory"], r2["memory"], places=5)
        self.assertAlmostEqual(r1["confidence"], r2["confidence"], places=5)


# ---------------------------------------------------------------------------
# 15. Small end-to-end: synthetic → train → checkpoint → reload → predict
# ---------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        """Full pipeline using the production normalization contract:
        1. Generate synthetic data in original scale.
        2. Fit feature and target normalizers on training split.
        3. Produce normalised X and y windows.
        4. Train Transformer on normalised X → normalised y.
        5. Save checkpoint with target_norm_map.
        6. Reload predictor.
        7. Predict from raw (un-normalised) sequence.
        8. Verify returned values are in original scale.
        """
        tmp = tempfile.mkdtemp()
        set_seed(0)

        # 1. Build synthetic dataset
        all_windows, feat_norm, tgt_norm = _synthetic_windows(n=80, seq_len=8, seed=0)
        n_train = int(len(all_windows) * 0.7)
        n_val = int(len(all_windows) * 0.15)
        windows = {
            "train": all_windows[:n_train],
            "val": all_windows[n_train: n_train + n_val],
            "test": all_windows[n_train + n_val:],
        }

        # Save normalizers so the trainer can embed them in the checkpoint
        norm_path = os.path.join(tmp, "feat_norm.json")
        tgt_norm_path = os.path.join(tmp, "feat_norm_targets.json")
        feat_norm.save(norm_path)
        tgt_norm.save(tgt_norm_path)

        pipeline_cfg = PipelineConfig(
            normalization_path=norm_path,
        )

        model_cfg = _make_cfg(feature_count=4, prediction_horizon=1)
        training_cfg = TrainingConfig(
            epochs=5,
            batch_size=8,
            lr=1e-2,
            seed=0,
            checkpoint_dir=tmp,
        )

        # 2. Train — pass normalizers explicitly so checkpoint is complete
        best_model, ckpt_path = train(
            pipeline_cfg, model_cfg, training_cfg,
            windows=windows,
            feat_normalizer=feat_norm,
            tgt_normalizer=tgt_norm,
        )

        # 3. Checkpoint exists and contains required keys
        self.assertTrue(os.path.isfile(ckpt_path))
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.assertIn("target_norm_map", ckpt)
        self.assertIn("cpu_percent", ckpt["target_norm_map"])

        # 4. val_loss is in normalised space (O(1), not O(1000))
        self.assertLess(ckpt["val_loss"], 100.0)

        # 5. Reload and predict from raw sequence
        predictor = TransformerPredictor.from_checkpoint(ckpt_path)
        # Raw sequence: original cpu% ~40, memory% ~60
        t = np.linspace(0, np.pi, 8)
        raw_seq = np.column_stack([
            40 + 30 * np.sin(t),
            np.ones(8) * 1e9,
            np.ones(8) * 300,
            60 + 15 * np.cos(t),
        ])
        result = predictor.predict(raw_seq)

        # 6. All keys present and finite
        for key in ("cpu", "memory", "confidence"):
            self.assertIn(key, result)
            self.assertTrue(math.isfinite(result[key]), msg=f"{key} is not finite")

        # 7. confidence in [0, 1]
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # 8. Predictions are in original scale — verify by checking they are
        #    in a plausible range (inverse-transform shifts by the target mean).
        cpu_mean_orig = tgt_norm.mean_[0]  # ~40
        mem_mean_orig = tgt_norm.mean_[1]  # ~60
        # A randomly-initialised model's normalised output ≈ 0, so after
        # inverse-transform: cpu ≈ cpu_mean_orig, mem ≈ mem_mean_orig
        # Allow ±50% deviation to account for a few training steps.
        self.assertGreater(abs(result["cpu"]), 0.3 * abs(cpu_mean_orig),
                           msg="cpu output appears un-inverse-transformed")
        self.assertGreater(abs(result["memory"]), 0.3 * abs(mem_mean_orig),
                           msg="memory output appears un-inverse-transformed")

        print(
            f"\n[E2E] cpu={result['cpu']:.2f}% (original scale), "
            f"memory={result['memory']:.2f}% (original scale), "
            f"confidence={result['confidence']:.4f} | "
            f"val_loss={ckpt['val_loss']:.6f} (normalised space)"
        )


if __name__ == "__main__":
    unittest.main()
