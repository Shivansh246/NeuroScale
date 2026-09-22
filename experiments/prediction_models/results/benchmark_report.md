# Benchmark Report: Transformer vs GRU vs XGBoost

## 1. Overview
This experiment benchmarks three prediction models on the real continuous Docker telemetry dataset (`data/prediction_experiment/real_workload_long_5000.jsonl`).

**Models:**
1. Transformer (Existing architecture, parameters: 67394)
2. GRU (Hidden: 64, Layers: 2, parameters: 38530)
3. XGBoost (Trees: 300, Depth: 6)

**Data Info:**
- Input history: 12 samples
- Prediction horizon: 1 sample
- Split: Strict chronological 70/15/15
- Test set size: 737 windows
- Test transition windows: 192

## 2. Overall Test Metrics

| Model | CPU MAE | CPU RMSE | CPU R² | Memory MAE | Memory RMSE | Memory R² | Mean inference ms | P95 inference ms |
|---|---|---|---|---|---|---|---|---|
| Transformer | 3.8762 | 12.4632 | 0.9237 | 23.2603 | 110.0251 | 0.9377 | 1.3539 | 2.1801 |
| GRU | 3.9181 | 12.3665 | 0.9249 | 22.3888 | 106.7599 | 0.9414 | 1.3931 | 1.6773 |
| XGBoost | 3.9949 | 11.9887 | 0.9294 | 25.9503 | 109.9793 | 0.9378 | 0.4674 | 0.6068 |

## 3. Transition-Window Metrics (Windows with Workload Changes)

| Model | CPU MAE | CPU RMSE | CPU R² | Memory MAE | Memory RMSE | Memory R² |
|---|---|---|---|---|---|---|
| Transformer | 8.3082 | 22.5345 | 0.7535 | 49.1794 | 186.0976 | 0.8265 |
| GRU | 8.6566 | 22.2724 | 0.7592 | 40.9782 | 180.1703 | 0.8374 |
| XGBoost | 7.2867 | 20.9335 | 0.7873 | 42.3522 | 181.2465 | 0.8355 |

## 4. Non-Transition Metrics (Stable Workloads)

| Model | CPU MAE | CPU RMSE | CPU R² | Memory MAE | Memory RMSE | Memory R² |
|---|---|---|---|---|---|---|
| Transformer | 2.3148 | 5.5819 | 0.9846 | 14.1291 | 64.5714 | 0.9783 |
| GRU | 2.2487 | 5.6611 | 0.9842 | 15.8399 | 63.0642 | 0.9793 |
| XGBoost | 2.8352 | 6.3233 | 0.9803 | 20.1720 | 69.1638 | 0.9752 |

## 5. Training Details

| Model | Training Time (s) | Checkpoint |
|---|---|---|
| Transformer | 33.38 | `checkpoints/transformer_real.pt` |
| GRU | 24.99 | `checkpoints/gru_real.pt` |
| XGBoost | 0.50 | `checkpoints/xgboost_real_*.json` |

## 6. Limitations

The real workload's memory behavior is step-like (jumping from a few MB to ~1 GB and back). This benchmark does not prove the model's ability to forecast gradual memory-demand curves. This limitation must be explicitly considered when generalizing memory prediction performance.

