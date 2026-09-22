# Phase 12: Offline Closed-Loop Evaluation

This document outlines the offline simulation results evaluating the Resource-Factorized DQN combined with the Real-Data Transformer Predictor.

## 1. System Under Test

- **Predictor**: `checkpoints/best_transformer_real.pt`
- **Anomaly Detector**: `checkpoints/best_autoencoder.pt`
- **DQN Policy**: `checkpoints/best_resource_factorized_dqn.pt`
- **Action Space**: 16 discrete joint actions (4 CPU options × 4 Memory options)
- **State Dimension**: 11
- **Controller Bounds**: CPU 0.25 to 2.0, Mem 128 to 1024 MB
- **SLA Threshold**: 200.0 ms
- **Reward Function**: `W_sla=10.0`, `W_res=1.0`
- **Sampling Interval**: 5.0 seconds

The evaluation forces strictly greedy deterministic action selection (`eval_mode = True`, effectively `epsilon = 0.0`).

## 2. Baseline Comparison

We evaluated four strategies across 200-step (50 steps per scenario) deterministic offline scenarios: `normal`, `cpu_spike`, `mem_spike`, and `sustained_load`.

| Scenario   | Strategy         | CPU Waste | Mem Waste | SLA Vio | Total Rwd | Reallocs |
|------------|------------------|-----------|-----------|---------|-----------|----------|
| normal     | Static           | 60.0      | 127.6     | 0       | -35.48    | 0        |
| normal     | Threshold        | 60.0      | 127.6     | 0       | -35.48    | 0        |
| normal     | Prediction-only  | 10.0      | 0.0       | 0       | -145.22   | 0        |
| normal     | NeuroScale (DQN) | 10.0      | 336.6     | 0       | -23.98    | 5        |
| cpu_spike  | Static           | 41.6      | 128.0     | 0       | -364.03   | 0        |
| cpu_spike  | Threshold        | 210.6     | 128.0     | 0       | -94.62    | 6        |
| cpu_spike  | Prediction-only  | 6.9       | 0.0       | 0       | -528.90   | 0        |
| cpu_spike  | NeuroScale (DQN) | 13.1      | 530.3     | 0       | -70.57    | 4        |
| mem_spike  | Static           | 60.0      | 88.8      | 0       | -7751.15  | 0        |
| mem_spike  | Threshold        | 60.0      | 167.2     | 0       | -812.90   | 4        |
| mem_spike  | Prediction-only  | 10.0      | 0.0       | 0       | -9642.90  | 0        |
| mem_spike  | NeuroScale (DQN) | 55.9      | 334.4     | 0       | -304.65   | 4        |
| sustained  | Static           | 18.4      | 39.2      | 0       | -9495.15  | 0        |
| sustained  | Threshold        | 176.5     | 384.0     | 0       | -330.15   | 6        |
| sustained  | Prediction-only  | 3.1       | 0.0       | 0       | -14404.40 | 0        |
| sustained  | NeuroScale (DQN) | 37.8      | 467.6     | 0       | -69.40    | 2        |

### Observations
- **NeuroScale DQN** successfully avoided all SLA violations and achieved the highest overall reward across dynamic scenarios.
- **CPU wastage** was tightly optimized, often outperforming the Static and Threshold baselines.
- **Memory wastage** remained notably high. The DQN heavily favors `A15` (max resources: 2.0C, 1024M) or `A6` (0.5C, 512M).

## 3. Policy Behavior Analysis

Across the 200 offline steps, the DQN action distribution was highly skewed:

- **Action 5** (0.5C, 256M): 14 selections
- **Action 6** (0.5C, 512M): 118 selections
- **Action 15** (2.0C, 1024M): 64 selections

**Independent Scaling Check**:
Out of 15 requested reallocations:
- CPU-only scaling: 0
- Mem-only scaling: 10
- Both CPU and Mem: 5

The policy **does not scale CPU independently**. When it upscales CPU, it *always* upscales memory simultaneously, indicating that independent resource scaling remains unsolved.

## 4. Counterfactual Policy Analysis

We queried the Q-network with deterministic, hand-crafted state combinations to observe the raw Q-values and action selections.

| State Name          | Q(CPU)_max | Q(Mem)_max | Selected CPU | Selected Mem |
|---------------------|------------|------------|--------------|--------------|
| A. normal           | -1.72      | -0.29      | 0.5          | 256          |
| B. high curr CPU    | -5.02      | -4.16      | 2.0          | 1024         |
| C. high curr mem    | -6.47      | -4.93      | 2.0          | 1024         |
| D. high pred CPU    | -5.23      | -4.25      | 2.0          | 1024         |
| E. high pred mem    | -5.63      | -4.43      | 2.0          | 1024         |
| F. high anomaly     | -10.38     | -7.28      | 2.0          | 1024         |
| G. high SLA latency | -6.07      | -4.71      | 2.0          | 1024         |

**Conclusion:** 
Except for the perfectly "normal" baseline state, **any** elevated signal (whether current utilization, predicted utilization, anomaly score, or SLA latency) causes the Q-network to defensively collapse into the maximum allowable allocation (Action 15). The DQN does not meaningfully differentiate between a CPU spike and a Memory spike.
