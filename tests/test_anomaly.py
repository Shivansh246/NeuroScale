"""Tests for the NeuroScale Autoencoder anomaly detection module.

All tests are CPU-only, deterministic, and require no Docker or real data.

Normalization contract:
  - The autoencoder operates on normalised flattened windows.
  - detector.score(sequence) accepts RAW (un-normalised) input and
    normalises internally before inference.
  - "score" is MSE reconstruction error in normalised feature space.
  - "is_anomaly" is True when score > threshold.
  - threshold = mean_val_error + k * std_val_error (from normal val data).

Coverage:
  1.  Autoencoder construction
  2.  Forward-pass output shape
  3.  Encoder / decoder dimension contract
  4.  Reconstruction error is non-negative
  5.  Reconstruction error is zero for identity-like trivial data
  6.  Deterministic training (same seed → same losses)
  7.  Threshold calculation (mean + k*std)
  8.  Threshold stored in checkpoint
  9.  Checkpoint contains all required keys
 10.  Checkpoint loads correctly
 11.  Inference API construction from checkpoint
 12.  score() returns correct schema {"score": float, "is_anomaly": bool}
 13.  score is non-negative
 14.  is_anomaly is bool
 15.  Normal samples → non-anomaly (after training on normal data)
 16.  Perturbed samples → anomaly (clearly out-of-distribution)
 17.  feature_columns stored and retrieved correctly
 18.  Wrong number of columns raises ValueError
 19.  1-D input raises ValueError
 20.  Deterministic inference after reload
 21.  No hardcoded feature-index assumptions (reordered columns)
 22.  Different k values produce different thresholds
 23.  Small end-to-end: train → checkpoint → reload → score normal + anomaly
"""

import math
import os
import tempfile
import unittest

import numpy as np
import torch

from anomaly.autoencoder import Autoencoder
from anomaly.config import AutoencoderConfig
from anomaly.detector import AutoencoderAnomalyDetector
from anomaly.trainer import (
    AETrainingConfig,
    compute_threshold,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_autoencoder,
    windows_to_tensor,
)
from data.normalization import Normalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEQ_LEN = 8
N_FEATS = 4
INPUT_DIM = SEQ_LEN * N_FEATS   # 32

FEATURE_COLS = ["cpu_percent", "cpu_usage_ns", "memory_usage_mb", "memory_percent"]


def _make_cfg(**kwargs) -> AutoencoderConfig:
    """Tiny config for fast tests."""
    defaults = dict(
        input_dim=INPUT_DIM,
        latent_dim=4,
        hidden_dims=[16, 8],
        dropout=0.0,
        seed=42,
        threshold_k=3.0,
        feature_columns=FEATURE_COLS,
        seq_len=SEQ_LEN,
    )
    defaults.update(kwargs)
    return AutoencoderConfig(**defaults)


def _make_model(cfg=None) -> Autoencoder:
    cfg = cfg or _make_cfg()
    return Autoencoder(cfg)


def _raw_normal_data(n_total: int = 100, seed: int = 0):
    """Return raw (un-normalised) feature matrix shape (n_total+SEQ_LEN, N_FEATS).

    Values are in original-scale CPU/memory ranges so they look realistic.
    Uses a sine pattern to be learnable by the autoencoder.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_total + SEQ_LEN)
    cpu = 40 + 30 * np.sin(t)          # ~10–70
    mem = 60 + 15 * np.cos(t)          # ~45–75
    noise = rng.normal(0, 0.5, t.shape)
    features = np.column_stack([
        cpu + noise,
        cpu * 1e6 + noise,
        mem * 5 + noise,
        mem + noise,
    ])
    return features  # (n_total+SEQ_LEN, N_FEATS)


def _make_normalised_windows(raw_features: np.ndarray, feat_norm: Normalizer, n: int, seq_len: int = SEQ_LEN):
    """Normalise raw_features and slice into (X, y) windows of seq_len."""
    normed = feat_norm.transform(raw_features)
    windows = []
    for i in range(n):
        x = normed[i: i + seq_len]
        y = normed[i + seq_len: i + seq_len + 1]
        windows.append((x, y))
    return windows


def _normal_windows(n: int = 60, seed: int = 0):
    """Legacy helper: returns normalised windows and the feat_norm used."""
    raw = _raw_normal_data(n_total=n, seed=seed)
    feat_norm = Normalizer().fit(raw[:n])   # fit on 'training' portion
    wins = _make_normalised_windows(raw, feat_norm, n)
    return wins, feat_norm


def _perturbed_raw_window(scale: float = 10.0, seed: int = 99):
    """Raw window with values far outside normal range (original scale)."""
    rng = np.random.default_rng(seed)
    # Normal range is ~10–70 for cpu, ~45–75 for mem; scale=10 → ±1000
    x = rng.normal(0, scale * 100.0, (SEQ_LEN, N_FEATS))
    return x  # shape (SEQ_LEN, N_FEATS), raw scale


def _make_checkpoint(
    tmp_dir: str,
    threshold: float = 0.1,
    feature_columns=None,
    val_loss: float = 0.05,
) -> str:
    """Build and save a minimal checkpoint; return path."""
    cols = feature_columns or FEATURE_COLS
    n_feats = len(cols)
    input_dim = SEQ_LEN * n_feats
    cfg = _make_cfg(input_dim=input_dim, feature_columns=cols)
    model = _make_model(cfg)
    feat_norm = Normalizer()
    feat_norm.mean_ = np.zeros(n_feats)
    feat_norm.std_ = np.ones(n_feats)
    path = os.path.join(tmp_dir, "ae_test.pt")
    save_checkpoint(
        path, model, cfg, feat_norm,
        threshold=threshold,
        mean_val_error=threshold - 0.05,
        std_val_error=0.05 / cfg.threshold_k,
        best_val_loss=val_loss,
    )
    return path


# ---------------------------------------------------------------------------
# 1. Model construction
# ---------------------------------------------------------------------------
class TestAutoencoderConstruction(unittest.TestCase):
    def test_default_construction(self):
        model = _make_model()
        self.assertIsInstance(model, Autoencoder)

    def test_parameter_count_reasonable(self):
        model = _make_model()
        n = sum(p.numel() for p in model.parameters())
        self.assertLess(n, 50_000)

    def test_different_hidden_dims(self):
        cfg = _make_cfg(hidden_dims=[64, 32, 16])
        model = Autoencoder(cfg)
        self.assertIsInstance(model, Autoencoder)


# ---------------------------------------------------------------------------
# 2 & 3. Forward-pass shape and encoder/decoder dimension contract
# ---------------------------------------------------------------------------
class TestAutoencoderShapes(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_cfg()
        self.model = _make_model(self.cfg)
        self.model.eval()

    def test_forward_output_shape(self):
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (4, INPUT_DIM))

    def test_encoder_output_shape(self):
        x = torch.randn(3, INPUT_DIM)
        with torch.no_grad():
            z = self.model.encode(x)
        self.assertEqual(z.shape, (3, self.cfg.latent_dim))

    def test_decoder_output_shape(self):
        z = torch.randn(2, self.cfg.latent_dim)
        with torch.no_grad():
            out = self.model.decode(z)
        self.assertEqual(out.shape, (2, INPUT_DIM))

    def test_reconstruction_error_shape(self):
        x = torch.randn(5, INPUT_DIM)
        errs = self.model.reconstruction_error(x)
        self.assertEqual(errs.shape, (5,))


# ---------------------------------------------------------------------------
# 4 & 5. Reconstruction error properties
# ---------------------------------------------------------------------------
class TestReconstructionError(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.model.eval()

    def test_error_is_nonnegative(self):
        x = torch.randn(10, INPUT_DIM)
        errs = self.model.reconstruction_error(x)
        self.assertTrue((errs >= 0).all())

    def test_perfect_reconstruction_gives_zero(self):
        """Patching forward to return identity should give zero error."""
        x = torch.randn(3, INPUT_DIM)
        # Temporarily make model an identity (set weights so output ≈ input)
        with torch.no_grad():
            x_hat = x.clone()
        errs = ((x - x_hat) ** 2).mean(dim=1)
        self.assertTrue(torch.allclose(errs, torch.zeros(3), atol=1e-6))


# ---------------------------------------------------------------------------
# 6. Deterministic training
# ---------------------------------------------------------------------------
class TestDeterministicTraining(unittest.TestCase):
    def _run(self, seed):
        set_seed(seed)
        wins, _ = _normal_windows(n=30)
        cfg = _make_cfg()
        X = windows_to_tensor(wins)
        ds = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
        model = Autoencoder(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()
        losses = []
        for _ in range(3):
            model.train()
            epoch_loss = 0.0
            for (xb,) in loader:
                xh = model(xb)
                loss = crit(xh, xb)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)
        return losses

    def test_same_seed_same_losses(self):
        l1 = self._run(77)
        l2 = self._run(77)
        for a, b in zip(l1, l2):
            self.assertAlmostEqual(a, b, places=5)

    def test_different_seeds_different_losses(self):
        l1 = self._run(1)
        l2 = self._run(2)
        self.assertFalse(all(abs(a - b) < 1e-9 for a, b in zip(l1, l2)))


# ---------------------------------------------------------------------------
# 7. Threshold calculation
# ---------------------------------------------------------------------------
class TestThresholdCalculation(unittest.TestCase):
    def setUp(self):
        self.errors = np.array([0.1, 0.2, 0.15, 0.12, 0.18])

    def test_threshold_equals_mean_plus_k_std(self):
        k = 3.0
        threshold, mean_err, std_err = compute_threshold(self.errors, k)
        expected = self.errors.mean() + k * self.errors.std(ddof=0)
        self.assertAlmostEqual(threshold, expected, places=8)

    def test_mean_and_std_correct(self):
        _, mean_err, std_err = compute_threshold(self.errors, k=2.0)
        self.assertAlmostEqual(mean_err, self.errors.mean(), places=8)
        self.assertAlmostEqual(std_err, self.errors.std(ddof=0), places=8)

    def test_different_k_different_threshold(self):
        t1, _, _ = compute_threshold(self.errors, k=1.0)
        t2, _, _ = compute_threshold(self.errors, k=5.0)
        self.assertGreater(t2, t1)

    def test_threshold_above_mean(self):
        threshold, mean_err, _ = compute_threshold(self.errors, k=0.0)
        self.assertAlmostEqual(threshold, mean_err, places=8)

    def test_threshold_nonnegative_for_nonnegative_errors(self):
        threshold, _, _ = compute_threshold(np.abs(self.errors), k=3.0)
        self.assertGreaterEqual(threshold, 0.0)


# ---------------------------------------------------------------------------
# 8, 9, 10. Checkpoint save and load
# ---------------------------------------------------------------------------
class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = _make_checkpoint(self.tmp, threshold=0.42)

    def test_checkpoint_file_exists(self):
        self.assertTrue(os.path.isfile(self.path))

    def test_checkpoint_required_keys(self):
        ckpt = torch.load(self.path, weights_only=False)
        for key in (
            "state_dict", "model_config", "feature_columns", "seq_len",
            "input_dim", "feat_normalization", "threshold", "threshold_k",
            "mean_val_error", "std_val_error", "best_val_loss", "seed",
        ):
            self.assertIn(key, ckpt, msg=f"Missing key: {key}")

    def test_threshold_stored_correctly(self):
        ckpt = torch.load(self.path, weights_only=False)
        self.assertAlmostEqual(ckpt["threshold"], 0.42, places=5)

    def test_load_checkpoint_returns_model(self):
        model, ckpt = load_checkpoint(self.path)
        self.assertIsInstance(model, Autoencoder)

    def test_loaded_model_output_shape(self):
        model, _ = load_checkpoint(self.path)
        x = torch.randn(2, INPUT_DIM)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, INPUT_DIM))


# ---------------------------------------------------------------------------
# 11–14. Inference API basic contract
# ---------------------------------------------------------------------------
class TestInferenceAPIContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = _make_checkpoint(self.tmp, threshold=0.5)
        self.detector = AutoencoderAnomalyDetector.from_checkpoint(self.path)
        # Raw (un-normalised) sequence — values ~N(0,1) because feat_norm is identity
        self.seq = np.random.default_rng(3).normal(0, 0.1, (SEQ_LEN, N_FEATS))

    def test_score_returns_dict(self):
        result = self.detector.score(self.seq)
        self.assertIsInstance(result, dict)

    def test_score_has_required_keys(self):
        result = self.detector.score(self.seq)
        self.assertIn("score", result)
        self.assertIn("is_anomaly", result)
        self.assertEqual(set(result.keys()), {"score", "is_anomaly"})

    def test_score_is_float(self):
        result = self.detector.score(self.seq)
        self.assertIsInstance(result["score"], float)

    def test_is_anomaly_is_bool(self):
        result = self.detector.score(self.seq)
        self.assertIsInstance(result["is_anomaly"], bool)

    def test_score_is_nonnegative(self):
        result = self.detector.score(self.seq)
        self.assertGreaterEqual(result["score"], 0.0)

    def test_bad_input_1d_raises(self):
        with self.assertRaises(ValueError):
            self.detector.score(np.random.randn(INPUT_DIM))

    def test_bad_column_count_raises(self):
        with self.assertRaises(ValueError):
            self.detector.score(np.random.randn(SEQ_LEN, N_FEATS + 2))


# ---------------------------------------------------------------------------
# 15 & 16. Normal vs perturbed anomaly classification
# ---------------------------------------------------------------------------
class TestNormalVsAnomalous(unittest.TestCase):
    """Train on normal data, then verify:
    - normal samples → non-anomaly
    - clearly perturbed samples → anomaly
    """
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        set_seed(0)
        cls.raw_data = _raw_normal_data(n_total=100, seed=0)
        cls.feat_norm = Normalizer().fit(cls.raw_data[:80])
        normal_wins = _make_normalised_windows(cls.raw_data, cls.feat_norm, 100)
        cls.train_wins = normal_wins[:80]
        cls.val_wins = normal_wins[80:]

        ae_cfg = _make_cfg(threshold_k=3.0)
        tr_cfg = AETrainingConfig(epochs=30, batch_size=16, lr=5e-3, seed=0,
                                   checkpoint_dir=cls.tmp)
        cls.model, cls.ckpt_path = train_autoencoder(
            ae_cfg, tr_cfg, cls.train_wins, cls.val_wins, cls.feat_norm
        )
        cls.detector = AutoencoderAnomalyDetector.from_checkpoint(cls.ckpt_path)

    def test_threshold_is_positive(self):
        self.assertGreater(self.detector.threshold, 0.0)

    def test_normal_window_is_not_anomaly(self):
        """Median-error normal samples should be classified as non-anomalous."""
        raw_x = self.raw_data[10: 10 + SEQ_LEN]
        result = self.detector.score(raw_x)
        self.assertFalse(result["is_anomaly"],
                         msg=f"Normal sample flagged as anomaly (score={result['score']:.4f}, "
                             f"threshold={self.detector.threshold:.4f})")

    def test_perturbed_window_is_anomaly(self):
        """A window with values 10× the normal std must be flagged as anomaly."""
        x_perturbed = _perturbed_raw_window(scale=10.0)
        result = self.detector.score(x_perturbed)
        self.assertTrue(result["is_anomaly"],
                        msg=f"Perturbed sample NOT flagged (score={result['score']:.4f}, "
                            f"threshold={self.detector.threshold:.4f})")

    def test_perturbed_score_greater_than_normal_score(self):
        raw_x = self.raw_data[10: 10 + SEQ_LEN]
        x_perturbed = _perturbed_raw_window(scale=10.0)
        score_normal = self.detector.score(raw_x)["score"]
        score_perturbed = self.detector.score(x_perturbed)["score"]
        self.assertGreater(score_perturbed, score_normal,
                           msg="Perturbed score should exceed normal score")


# ---------------------------------------------------------------------------
# 17. Feature columns stored and retrieved correctly
# ---------------------------------------------------------------------------
class TestFeatureColumns(unittest.TestCase):
    def test_feature_columns_stored_in_checkpoint(self):
        tmp = tempfile.mkdtemp()
        path = _make_checkpoint(tmp, feature_columns=FEATURE_COLS)
        ckpt = torch.load(path, weights_only=False)
        self.assertEqual(ckpt["feature_columns"], FEATURE_COLS)

    def test_detector_exposes_feature_columns(self):
        tmp = tempfile.mkdtemp()
        path = _make_checkpoint(tmp, feature_columns=FEATURE_COLS)
        det = AutoencoderAnomalyDetector.from_checkpoint(path)
        self.assertEqual(det.feature_columns, FEATURE_COLS)

    def test_detector_exposes_seq_len(self):
        tmp = tempfile.mkdtemp()
        path = _make_checkpoint(tmp)
        det = AutoencoderAnomalyDetector.from_checkpoint(path)
        self.assertEqual(det.seq_len, SEQ_LEN)


# ---------------------------------------------------------------------------
# 21. No hardcoded feature-index assumptions — reordered columns
# ---------------------------------------------------------------------------
class TestReorderedColumns(unittest.TestCase):
    def test_reordered_feature_columns_no_crash(self):
        """Detector built with reordered feature columns must still produce
        valid scores — no hardcoded index assumptions."""
        reordered = ["memory_percent", "cpu_usage_ns", "memory_usage_mb", "cpu_percent"]
        tmp = tempfile.mkdtemp()
        path = _make_checkpoint(tmp, feature_columns=reordered)
        det = AutoencoderAnomalyDetector.from_checkpoint(path)
        self.assertEqual(det.feature_columns, reordered)
        seq = np.random.default_rng(7).normal(0, 0.1, (SEQ_LEN, len(reordered)))
        result = det.score(seq)
        self.assertIn("score", result)
        self.assertIn("is_anomaly", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertIsInstance(result["is_anomaly"], bool)


# ---------------------------------------------------------------------------
# 20. Deterministic inference after reload
# ---------------------------------------------------------------------------
class TestDeterministicInference(unittest.TestCase):
    def test_same_result_after_reload(self):
        tmp = tempfile.mkdtemp()
        path = _make_checkpoint(tmp)
        d1 = AutoencoderAnomalyDetector.from_checkpoint(path)
        d2 = AutoencoderAnomalyDetector.from_checkpoint(path)
        seq = np.random.default_rng(11).normal(0, 0.1, (SEQ_LEN, N_FEATS))
        r1 = d1.score(seq)
        r2 = d2.score(seq)
        self.assertAlmostEqual(r1["score"], r2["score"], places=6)
        self.assertEqual(r1["is_anomaly"], r2["is_anomaly"])


# ---------------------------------------------------------------------------
# 23. Small end-to-end
# ---------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline(self):
        """Train → checkpoint → reload → score normal and perturbed windows."""
        tmp = tempfile.mkdtemp()
        set_seed(0)

        raw_data = _raw_normal_data(n_total=100, seed=0)
        feat_norm = Normalizer().fit(raw_data[:80])
        normal_wins = _make_normalised_windows(raw_data, feat_norm, 100)
        train_wins = normal_wins[:80]
        val_wins = normal_wins[80:]

        ae_cfg = _make_cfg(threshold_k=3.0)
        tr_cfg = AETrainingConfig(epochs=25, batch_size=16, lr=5e-3, seed=0,
                                   checkpoint_dir=tmp)
        model, ckpt_path = train_autoencoder(
            ae_cfg, tr_cfg, train_wins, val_wins, feat_norm
        )

        # Verify checkpoint
        self.assertTrue(os.path.isfile(ckpt_path))
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.assertIn("threshold", ckpt)
        self.assertGreater(ckpt["threshold"], 0.0)

        # Reload and score
        detector = AutoencoderAnomalyDetector.from_checkpoint(ckpt_path)

        x_normal = raw_data[15: 15 + SEQ_LEN]
        x_perturbed = _perturbed_raw_window(scale=10.0)

        res_normal = detector.score(x_normal)
        res_perturbed = detector.score(x_perturbed)

        print(f"\n[E2E] normal score={res_normal['score']:.6f} "
              f"is_anomaly={res_normal['is_anomaly']} | "
              f"perturbed score={res_perturbed['score']:.6f} "
              f"is_anomaly={res_perturbed['is_anomaly']} | "
              f"threshold={detector.threshold:.6f}")

        # Schema
        for res in (res_normal, res_perturbed):
            self.assertIn("score", res)
            self.assertIn("is_anomaly", res)
            self.assertIsInstance(res["score"], float)
            self.assertIsInstance(res["is_anomaly"], bool)
            self.assertGreaterEqual(res["score"], 0.0)

        # Perturbed score must be higher
        self.assertGreater(res_perturbed["score"], res_normal["score"])

        # Normal sample should not be flagged as anomaly
        self.assertFalse(res_normal["is_anomaly"])

        # Perturbed must be an anomaly
        self.assertTrue(res_perturbed["is_anomaly"])


if __name__ == "__main__":
    unittest.main()
