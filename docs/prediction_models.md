# Prediction Models in NeuroScale

This document details the predictive machine learning component of the NeuroScale closed-loop controller.

## 1. Why the Real Dataset Was Collected
Initially, NeuroScale's Transformer predictor was trained on purely synthetic traces. While this validated the end-to-end integration (Phase 1-9), synthetic data failed to generalize to the bursty, step-like dynamics of a real Docker workload. A 7-hour continuous dataset (`data/prediction_experiment/real_workload_long_5000.jsonl`) consisting of 5,000 real telemetry samples was collected from a persistent Docker container to train and benchmark models on realistic dynamics.

## 2. Transformer Architecture
The production architecture is a lightweight, sequence-length-agnostic Transformer Encoder optimized for CPU-only inference:
- **Input:** 4 features (CPU %, CPU ns delta, Memory MB, Memory %)
- **Projection:** Linear 4 → 64
- **Positional Encoding:** Sinusoidal
- **Transformer Encoder:** 2 layers (d_model=64, nhead=4, feedforward=128, dropout=0.1)
- **Output:** Linear representation mapped to 2 targets (CPU, Memory)

## 3. Training Dataset & Chronological Split
The 5,000-sample real dataset was split chronologically:
- **History length:** 12 samples
- **Prediction horizon:** 1 sample
- **Train/Val/Test Split:** 70% / 15% / 15%
- No random shuffling was used, explicitly preventing temporal and feature leakage across dataset boundaries.

## 4. Benchmark: Transformer vs GRU vs XGBoost
A controlled benchmark evaluated the Transformer against GRU and XGBoost regressors on the test set.

**CPU Prediction:**
- **Transformer:** MAE 3.87 | RMSE 12.46 | R² 0.92
- **GRU:** MAE 3.91 | RMSE 12.36 | R² 0.92
- **XGBoost:** MAE 3.99 | RMSE 11.98 | R² 0.92

**Memory Prediction:**
- **Transformer:** MAE 23.26 | RMSE 110.02 | R² 0.93
- **GRU:** MAE 22.38 | RMSE 106.75 | R² 0.94
- **XGBoost:** MAE 25.95 | RMSE 109.97 | R² 0.93

**Inference Latency (Mean):**
- **Transformer:** 1.35 ms
- **GRU:** 1.39 ms
- **XGBoost:** 0.46 ms

## 5. Persistence Baseline
A naive persistence baseline (predicting the last observed input value) was evaluated:
- **CPU:** MAE 7.49 | RMSE 22.39 | R² 0.75
- **Memory:** MAE 11.29 | RMSE 107.72 | R² 0.94

## 6. Memory Limitation
The real Docker workload exhibits an extremely polarized, step-like memory distribution (77% < 300MB, 22% > 800MB). Because memory changes are instantaneous jumps rather than smooth trends, the prediction models **did not outperform the persistence baseline on Memory MAE**. Memory predictions must be treated purely as a directional/state-change signal, not as a continuous curve forecast. CPU prediction, conversely, significantly outperformed persistence.

## 7. Leakage Audit
A rigorous methodological audit confirmed:
- No temporal leakage (train/val/test boundaries are strictly sequential).
- No feature leakage (`workload_mode` and future timestamps were completely excluded from inputs).
- Normalization integrity (scaling parameters derived exclusively from the training split).

## 8. Production Integration
The real-data-trained Transformer was repackaged to perfectly match the existing `predict(sequence)` API of `TransformerPredictor`. It is integrated into the control loop via the `--predictor real` flag in `scripts/run_closed_loop.py`. The downstream RL controller remains unchanged, seamlessly operating on the drastically improved predictive signals.
