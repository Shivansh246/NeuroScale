# Benchmark Audit Report

## A. Dataset Verification
Source: `data/prediction_experiment/real_workload_long_5000.jsonl`
History: 12, Horizon: 1
Splits are strictly chronological. No random shuffling was used.

**TRAIN:**
First: 1790001558.45, Last: 1790019009.78, Windows: 3491

**VALIDATION:**
First: 1790019074.78, Last: 1790022749.88, Windows: 736

**TEST:**
First: 1790022814.88, Last: 1790026494.98, Windows: 737

Overlap leakage check (train_end < val_start < test_start): True

## B. Feature Verification
Features used: ['cpu_percent', 'cpu_usage_ns_delta (log1p)', 'memory_usage_mb', 'memory_percent']
Target leakage check: `workload_mode` is NOT supplied to any model as input. Future values and timestamps are not included.
Dimensionality:
- Transformer/GRU: 12x4
- XGBoost: 48 flattened features

## C. Target Transformation Verification
- Normalizer fits only on `X_train`, `y_train`.
- Validation and test are transformed using training parameters.
- Target predictions are inverse-transformed before metrics calculation.
Leakage: None detected in normalization.

## D. Transformer Architecture Comparison
The `experiments/prediction_models/transformer_model.py` directly imports and uses `models.transformer.NeuroScaleTransformer`. 
It provides `TransformerConfig` explicitly matching the production parameters (d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1). 
Result: Exactly the same architecture and implementation (Reused).

## E. Training Configuration
- Random seed: 42
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Maximum epochs: 50
- Early stopping patience: 5
- Validation metric: MSE Loss
- Test data untouched during training: Yes.

## F. XGBoost Configuration
- XGBoost trains two independent regressors (CPU, Memory).
- Parameters: 300 trees, depth 6, lr 0.05, early stopping 10 rounds on validation data.
- Test data untouched during tuning: Yes.

## G. Latency Methodology
- Warmup iterations: 10
- Measurement iterations: 100
- Execution: CPU only, predicting on a single slice (batch=1). 
- No disk I/O, dataset loading, or plotting is included in the timing loop.
- It appropriately measures just `model_func(sample_input)`.

## H. Transition-Window Methodology
Transition windows were defined using `test_win_modes`. The condition `len(set(modes)) > 1` checks if the 12 input steps + 1 target step contain more than 1 distinct workload label.
This uses future workload labels strictly for post-hoc stratification. The labels are never fed as input features.

## I. Memory-Distribution Analysis
- Min Memory: 0.47 MB
- Max Memory: 1038.43 MB
- Median Memory: 3.36 MB
- Samples near low memory (<300MB): 77.70%
- Samples near high memory (>800MB): 22.30%

The step-like memory behavior is extremely polarized. This means memory prediction is effectively a binary state change problem.

## J. Persistence Baseline
Persistence Model (predict last observed value in history) on Test Set:
**Overall:**
- CPU MAE: 7.4995, RMSE: 22.3933, R²: 0.7538
- Memory MAE: 11.2939, RMSE: 107.7234, R²: 0.9403

**Transition:**
- CPU MAE: 11.6823, RMSE: 28.9777, R²: 0.5924
- Memory MAE: 32.5241, RMSE: 182.7940, R²: 0.8326

**Non-Transition:**
- CPU MAE: 6.0259, RMSE: 19.5525, R²: 0.8114
- Memory MAE: 3.8147, RMSE: 62.6183, R²: 0.9796

## K. Discrepancies Discovered
No methodological discrepancies were discovered in the training, data splits, or feature extraction. The pipeline faithfully respects isolation principles.

## L. Methodological Validity
The benchmark is methodologically valid. Target leakage, temporal leakage, and test-set contamination have been successfully prevented. 

## M. Final Report Usability
All results are valid and can be used in the final report, with the explicit caveat that the Memory R2 and RMSE heavily reflect step-change forecasting rather than smooth continuous curve fitting, as confirmed by the distribution analysis.
