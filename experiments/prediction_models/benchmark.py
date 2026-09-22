import time
import json
import torch
import numpy as np
import os
from experiments.prediction_models.data import get_prepared_data
from experiments.prediction_models.transformer_model import train_transformer
from experiments.prediction_models.gru_model import train_gru
from experiments.prediction_models.xgboost_model import train_xgboost, predict_xgboost
from experiments.prediction_models.metrics import compute_metrics
from experiments.prediction_models.plots import generate_plots

def measure_latency(model_func, sample_input, iterations=100):
    # Warmup
    for _ in range(10):
        model_func(sample_input)
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model_func(sample_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # ms
        
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95))
    }

def get_params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("Loading and preparing data...")
    splits, norm, raw_records = get_prepared_data()
    
    X_train, y_train, train_modes, _, _ = splits["train"]
    X_val, y_val, val_modes, _, _ = splits["val"]
    X_test, y_test, test_modes, test_times, test_win_modes = splits["test"]
    
    # 1. Train Transformer
    print("Training Transformer...")
    t0 = time.time()
    trans_model, trans_val_loss = train_transformer(X_train, y_train, X_val, y_val)
    trans_train_time = time.time() - t0
    trans_params = get_params_count(trans_model)
    
    # 2. Train GRU
    print("Training GRU...")
    t0 = time.time()
    gru_model, gru_val_loss = train_gru(X_train, y_train, X_val, y_val)
    gru_train_time = time.time() - t0
    gru_params = get_params_count(gru_model)
    
    # 3. Train XGBoost
    # Using normalized data for XGBoost for fair comparison, though tree models don't strictly require it.
    print("Training XGBoost...")
    t0 = time.time()
    xgb_cpu, xgb_mem = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_train_time = time.time() - t0
    
    # Predict on test set
    print("Evaluating models...")
    X_test_t = torch.FloatTensor(X_test)
    trans_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        trans_pred_norm = trans_model(X_test_t).squeeze(1).numpy()
        gru_pred_norm = gru_model(X_test_t).numpy()
        
    xgb_pred_norm = predict_xgboost(xgb_cpu, xgb_mem, X_test)
    
    # Inverse transform
    y_test_inv = norm.inverse_transform_y(y_test)
    trans_pred = norm.inverse_transform_y(trans_pred_norm)
    gru_pred = norm.inverse_transform_y(gru_pred_norm)
    xgb_pred = norm.inverse_transform_y(xgb_pred_norm)
    
    # Identify transition windows
    is_transition = []
    for modes in test_win_modes:
        if len(set(modes)) > 1:
            is_transition.append(True)
        else:
            is_transition.append(False)
    is_transition = np.array(is_transition)
    
    def evaluate(y_true, y_pred, mask=None):
        if mask is not None:
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        cpu_metrics = compute_metrics(y_true[:, 0], y_pred[:, 0])
        mem_metrics = compute_metrics(y_true[:, 1], y_pred[:, 1])
        return {"cpu": cpu_metrics, "memory": mem_metrics}
        
    results = {
        "transformer": {
            "all": evaluate(y_test_inv, trans_pred),
            "transition": evaluate(y_test_inv, trans_pred, is_transition),
            "non_transition": evaluate(y_test_inv, trans_pred, ~is_transition)
        },
        "gru": {
            "all": evaluate(y_test_inv, gru_pred),
            "transition": evaluate(y_test_inv, gru_pred, is_transition),
            "non_transition": evaluate(y_test_inv, gru_pred, ~is_transition)
        },
        "xgboost": {
            "all": evaluate(y_test_inv, xgb_pred),
            "transition": evaluate(y_test_inv, xgb_pred, is_transition),
            "non_transition": evaluate(y_test_inv, xgb_pred, ~is_transition)
        }
    }
    
    # Latency measurement
    sample_input = X_test_t[0:1]
    sample_input_np = X_test[0:1]
    trans_lat = measure_latency(lambda x: trans_model(x), sample_input)
    gru_lat = measure_latency(lambda x: gru_model(x), sample_input)
    xgb_lat = measure_latency(lambda x: predict_xgboost(xgb_cpu, xgb_mem, x), sample_input_np)
    
    # Plots
    print("Generating plots...")
    generate_plots(y_test_inv, trans_pred, gru_pred, xgb_pred, test_times, "experiments/prediction_models/results/plots")
    
    # Save Report
    print("Generating report...")
    report = f"""# Benchmark Report: Transformer vs GRU vs XGBoost

## 1. Overview
This experiment benchmarks three prediction models on the real continuous Docker telemetry dataset (`data/prediction_experiment/real_workload_long_5000.jsonl`).

**Models:**
1. Transformer (Existing architecture, parameters: {trans_params})
2. GRU (Hidden: 64, Layers: 2, parameters: {gru_params})
3. XGBoost (Trees: 300, Depth: 6)

**Data Info:**
- Input history: 12 samples
- Prediction horizon: 1 sample
- Split: Strict chronological 70/15/15
- Test set size: {len(y_test_inv)} windows
- Test transition windows: {sum(is_transition)}

## 2. Overall Test Metrics

| Model | CPU MAE | CPU RMSE | CPU R² | Memory MAE | Memory RMSE | Memory R² | Mean inference ms | P95 inference ms |
|---|---|---|---|---|---|---|---|---|
| Transformer | {results['transformer']['all']['cpu']['mae']:.4f} | {results['transformer']['all']['cpu']['rmse']:.4f} | {results['transformer']['all']['cpu']['r2']:.4f} | {results['transformer']['all']['memory']['mae']:.4f} | {results['transformer']['all']['memory']['rmse']:.4f} | {results['transformer']['all']['memory']['r2']:.4f} | {trans_lat['mean_ms']:.4f} | {trans_lat['p95_ms']:.4f} |
| GRU | {results['gru']['all']['cpu']['mae']:.4f} | {results['gru']['all']['cpu']['rmse']:.4f} | {results['gru']['all']['cpu']['r2']:.4f} | {results['gru']['all']['memory']['mae']:.4f} | {results['gru']['all']['memory']['rmse']:.4f} | {results['gru']['all']['memory']['r2']:.4f} | {gru_lat['mean_ms']:.4f} | {gru_lat['p95_ms']:.4f} |
| XGBoost | {results['xgboost']['all']['cpu']['mae']:.4f} | {results['xgboost']['all']['cpu']['rmse']:.4f} | {results['xgboost']['all']['cpu']['r2']:.4f} | {results['xgboost']['all']['memory']['mae']:.4f} | {results['xgboost']['all']['memory']['rmse']:.4f} | {results['xgboost']['all']['memory']['r2']:.4f} | {xgb_lat['mean_ms']:.4f} | {xgb_lat['p95_ms']:.4f} |

## 3. Transition-Window Metrics (Windows with Workload Changes)

| Model | CPU MAE | CPU RMSE | CPU R² | Memory MAE | Memory RMSE | Memory R² |
|---|---|---|---|---|---|---|
| Transformer | {results['transformer']['transition']['cpu']['mae']:.4f} | {results['transformer']['transition']['cpu']['rmse']:.4f} | {results['transformer']['transition']['cpu']['r2']:.4f} | {results['transformer']['transition']['memory']['mae']:.4f} | {results['transformer']['transition']['memory']['rmse']:.4f} | {results['transformer']['transition']['memory']['r2']:.4f} |
| GRU | {results['gru']['transition']['cpu']['mae']:.4f} | {results['gru']['transition']['cpu']['rmse']:.4f} | {results['gru']['transition']['cpu']['r2']:.4f} | {results['gru']['transition']['memory']['mae']:.4f} | {results['gru']['transition']['memory']['rmse']:.4f} | {results['gru']['transition']['memory']['r2']:.4f} |
| XGBoost | {results['xgboost']['transition']['cpu']['mae']:.4f} | {results['xgboost']['transition']['cpu']['rmse']:.4f} | {results['xgboost']['transition']['cpu']['r2']:.4f} | {results['xgboost']['transition']['memory']['mae']:.4f} | {results['xgboost']['transition']['memory']['rmse']:.4f} | {results['xgboost']['transition']['memory']['r2']:.4f} |

## 4. Non-Transition Metrics (Stable Workloads)

| Model | CPU MAE | CPU RMSE | CPU R² | Memory MAE | Memory RMSE | Memory R² |
|---|---|---|---|---|---|---|
| Transformer | {results['transformer']['non_transition']['cpu']['mae']:.4f} | {results['transformer']['non_transition']['cpu']['rmse']:.4f} | {results['transformer']['non_transition']['cpu']['r2']:.4f} | {results['transformer']['non_transition']['memory']['mae']:.4f} | {results['transformer']['non_transition']['memory']['rmse']:.4f} | {results['transformer']['non_transition']['memory']['r2']:.4f} |
| GRU | {results['gru']['non_transition']['cpu']['mae']:.4f} | {results['gru']['non_transition']['cpu']['rmse']:.4f} | {results['gru']['non_transition']['cpu']['r2']:.4f} | {results['gru']['non_transition']['memory']['mae']:.4f} | {results['gru']['non_transition']['memory']['rmse']:.4f} | {results['gru']['non_transition']['memory']['r2']:.4f} |
| XGBoost | {results['xgboost']['non_transition']['cpu']['mae']:.4f} | {results['xgboost']['non_transition']['cpu']['rmse']:.4f} | {results['xgboost']['non_transition']['cpu']['r2']:.4f} | {results['xgboost']['non_transition']['memory']['mae']:.4f} | {results['xgboost']['non_transition']['memory']['rmse']:.4f} | {results['xgboost']['non_transition']['memory']['r2']:.4f} |

## 5. Training Details

| Model | Training Time (s) | Checkpoint |
|---|---|---|
| Transformer | {trans_train_time:.2f} | `checkpoints/transformer_real.pt` |
| GRU | {gru_train_time:.2f} | `checkpoints/gru_real.pt` |
| XGBoost | {xgb_train_time:.2f} | `checkpoints/xgboost_real_*.json` |

## 6. Limitations

The real workload's memory behavior is step-like (jumping from a few MB to ~1 GB and back). This benchmark does not prove the model's ability to forecast gradual memory-demand curves. This limitation must be explicitly considered when generalizing memory prediction performance.

"""
    with open("experiments/prediction_models/results/benchmark_report.md", "w") as f:
        f.write(report)
        
    print("Done!")

if __name__ == "__main__":
    main()
