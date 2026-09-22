# Predictor Comparison (Old vs Real-Data)

Evaluated on 500 real telemetry sequences.

## 1. CPU Prediction
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Old (Synthetic) | 31.7647 | 36.3208 | 0.3357 |
| New (Real-Data) | 3.1691 | 10.4096 | 0.9454 |

## 2. Memory Prediction
Note: Old model outputs `memory_percent` which was converted to MB assuming 1024MB limit for this comparison. New model outputs `memory_usage_mb` directly.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Old (Synthetic) | 250.1111 | 346.0063 | 0.3431 |
| New (Real-Data) | 22.5076 | 105.6320 | 0.9388 |

## 3. Confidence & Latency
| Model | Confidence (Fixed) | Mean Inference Latency (ms) |
|---|---|---|
| Old (Synthetic) | 0.8429 | 0.3889 |
| New (Real-Data) | 0.9368 | 0.3803 |

**Conclusion:** The new real-data model is substantially more accurate on the real telemetry dataset. The old synthetic-trained model failed to generalize to the real memory scales and CPU patterns.
