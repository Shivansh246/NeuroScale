# Phase 12: Real-Data Transformer + Resource-Factorized DQN Closed-Loop Evaluation

This directory contains the results and reports from the Phase 12 evaluation of the NeuroScale control loop.

## Overview

In this phase, we validated the integration of the real-data Transformer predictor and the Resource-Factorized DQN in a live closed-loop environment. The goals were to establish the exact system parameters, perform offline counterfactual evaluations, and execute a continuous real-world demonstration against a live Docker container.

## Deliverables

- **`offline_evaluation.md`**: Contains baseline comparisons (Static, Threshold, Prediction-only, DQN) over deterministic trace scenarios, policy behavior distributions, and a counterfactual analysis revealing the DQN's over-reliance on defensive scaling.
- **`real_docker_evaluation.md`**: Documents the live Docker demonstration, end-to-end component latencies, API success rates, and an analysis of pipeline feature mismatches and workload saturation effects.
- **`results/`**: 
  - `policy_analysis.json`: Raw Q-values from the counterfactual tests.
  - `offline_results.json`: Complete metrics for offline traces.
  - `real_docker_results.jsonl`: The raw telemetry and control log from the live 15-minute experiment.
  - `timing_results.json`: Averaged component latencies.
  - `prediction_errors.json`: Calculated error metrics from the live run.
- **`results/plots/`**: Visualizations of the real Docker evaluation (CPU, Memory, Anomaly, Confidence).

## Key Findings

1. **Inference Latency**: With correct PyTorch `no_grad()` scoping, the combined predictive + RL inference runs at ~2.5 ms, well within real-time SLA bounds.
2. **DQN Behavior**: The "Resource-Factorized" DQN does **not** demonstrate independent scaling. It heavily couples CPU and Memory decisions, often defaulting to maximum allocations when sensing any elevated signal.
3. **Pipeline Mismatches**: The live anomaly detector suffers from a feature extraction bug (`cpu_usage_ns` vs. `cpu_usage_ns_delta`) that artificially triggers constant anomaly flags, forcing the DQN into permanent defensive over-provisioning.
