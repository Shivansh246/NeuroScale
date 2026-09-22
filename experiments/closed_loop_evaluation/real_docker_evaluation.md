# Phase 12: Real Docker Closed-Loop Evaluation

This document details the real-world validation of the NeuroScale RL pipeline controlling a live Docker container using the Real-Data Transformer checkpoint.

## 1. Experimental Setup
- **Duration**: ~10 minutes (12 warmup cycles + 105 active cycles).
- **Sampling Interval**: Exactly 5.0 seconds.
- **Phases**: Sequential transitions: `idle -> cpu -> idle -> memory -> idle -> mixed -> idle`.
- **Methodology**: The orchestrator continuously polled Docker stats, queried the predictor, detector, and agent, and applied `docker update` directly to the `neuroscale-smoke` container.

## 2. Real vs. Requested Allocations

The orchestrator effectively bridged the RL policy to the Docker Daemon API:

- **Total Active Cycles**: 105
- **Requested Reallocations**: 1
- **Successful Applications**: 1
- **Failed Applications**: 0
- **No-ops (No change needed)**: 104

*Note on Reallocations*: The RL agent scaled up to the maximum allowable limit (2.0 CPUs, 1024 MB) during the very first cycle and refused to scale down, resulting in zero subsequent requested changes.

## 3. Timing and Latency

| Component | Mean Latency (ms) |
|-----------|-------------------|
| Metrics Collection (`docker stats`) | ~1010 ms |
| Controller Operations (`docker update`) | ~16 ms |
| ML Inference (Predictor + Detector + DQN) | ~2.5 ms |
| **Total Control Cycle** | **~1030 ms** |

The orchestrator successfully maintained the 5.000s step interval by accounting for the ~1.03s execution time.

## 4. Pipeline Anomaly and Saturation Analysis

The experiment uncovered critical pipeline mismatches preventing normal operation on real data:

1. **Feature Mismatch (Anomaly Detector)**:
   - The original pipeline was trained using `cpu_usage_ns_delta` (often logged).
   - `ControlLoopConfig` explicitly extracts `cpu_usage_ns` (raw cumulative nanoseconds).
   - Consequently, the Autoencoder (trained on small deltas) received cumulative nanosecond values in the trillions. It flagged **100% of the cycles as extreme anomalies** (mean idle anomaly score: ~200,000).

2. **Defensive Policy Collapse**:
   - Because `is_anomaly = True` and `anomaly_score` was astronomically high for every step, the DQN policy predictably reacted by selecting Action 15 (2.0 CPUs, 1024 MB Mem) constantly as a defensive safeguard.

3. **Workload Saturation**:
   - The process of transitioning workloads via `docker exec -d python3 workload.py` failed to cleanly terminate the previous workload modes (PID 1 ignored `pkill`).
   - This led to multiple stacked background workloads, pushing the container's CPU utilization to a flatlined 200% (exactly hitting the 2.0 CPU hard quota set by the DQN).
   - Because the true utilization was synthetically clamped at the quota limit, the predictor (trained on normal bounds) severely underestimated the constrained CPU usage.

## 5. Conclusion

The pipeline correctly executes end-to-end against real Docker containers. The RL agent successfully influences live host configurations, and PyTorch ML inference runs under 3 milliseconds per cycle.

However, the real-world validation reveals that the existing DQN policy lacks true multi-dimensional independence and is highly vulnerable to feature extraction mismatches in the anomaly detector. To achieve autonomous dynamic scaling, the pipeline's feature extraction layer must be aligned, and the RL policy must be explicitly trained with decoupled action sub-networks.
