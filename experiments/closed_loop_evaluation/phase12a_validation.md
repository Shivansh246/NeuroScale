# Phase 12A Validation Report

## A. Structural Feature Discrepancy (Cumulative vs Delta)
The root cause of the constant anomalies and poor predictions in Phase 12 was a structural mismatch in the `cpu_usage_ns` feature. The synthetic data generation script (`generate_data.py`) originally synthesized this feature by computing `cpu_percent * 1e7`, creating an **interval-based magnitude (delta)** representing nanoseconds of CPU time spent per 5-second interval. However, the live control loop passed the **raw monotonically increasing cumulative nanoseconds** counter directly from Docker's cgroup to the models. This caused the Autoencoder to see values in the trillions (instead of the expected ~50M-5B range) and flagged every cycle as a severe anomaly.

## B. Synthesis of `cpu_usage_ns_delta`
To resolve this, we updated `control/loop.py` to compute `cpu_usage_ns_delta` live. The Orchestrator now maintains a session state (`self.prev_cpu_usage_ns` and `self.current_session_id`). During each cycle, it subtracts the previous cumulative value from the current cumulative value. If a new container session is detected, it resets the delta to `0`. If the Docker daemon counter resets (the current value is less than the previous value), it handles it gracefully by returning the current value without producing a negative delta.

## C. Proof of Safe Counter Handling
We added `tests/test_feature_contract.py` with 5 targeted unit tests:
1. `test_cumulative_cpu_to_delta`: Proves monotonically increasing cumulative metrics correctly translate to positive deltas.
2. `test_counter_reset`: Proves that a sudden drop in cumulative metrics (a Docker daemon reset) does not produce a negative delta, and sets a `counter_reset` flag.
3. `test_container_transition_no_reset`: Proves that transitioning to a new container ID resets the state cleanly.
4. `test_workload_transition_no_reset`: Proves that changing workloads within the same container does NOT reset the delta state.
5. `test_live_feature_order_matches_config`: Proves the sequence array structure exactly matches the feature definition.

## D. Current Live Feature Order
The `feature_columns` configuration in `control/config.py` was updated to:
`["cpu_percent", "cpu_usage_ns_delta", "memory_usage_mb", "memory_percent"]`

## E. Predictor vs Autoencoder Preprocessing Conflict
While both models require the exact same feature sequence order, they were unexpectedly trained on different preprocessing scales for this specific feature:
- The **Autoencoder** was trained on the raw delta (`cpu_percent * 1e7`).
- The **Transformer** predictor was trained on a log-scaled delta (`np.log1p(delta)`), as discovered in `experiments/prediction_models/data.py`. Its `Normalizer` checkpoint strictly expects inputs in the `log1p` space.

## F. Architectural Challenge (Unresolved)
Because the `Orchestrator` feeds the exact same numpy array sequence to both `detector.score(sequence)` and `predictor.predict(sequence)`, and because we are strictly instructed **not** to modify the model checkpoints or inference architectures, it is structurally impossible for the live loop to provide the raw delta to the Autoencoder and the `log1p` delta to the Predictor simultaneously under the same feature index. 
In the current setup, we feed the raw delta. This perfectly resolves the anomaly detector's behavior, but means the Transformer Predictor will continue to receive incorrectly scaled inputs.

## G. Phase 12 Workload Lifecycle Failure
In Phase 12, workloads "stacked up" because the transition script executed `pkill -f workload.py` inside the container. The `python:3.12-slim` Docker base image does not have `procps` installed, meaning `pkill` was silently failing with `executable file not found in $PATH`. Instead of replacing the workload, every transition launched a new concurrent background workload, eventually saturating the CPU at 200%+.

## H. Workload Lifecycle Resolution
The orchestration script was rewritten to properly manage process lifecycle:
1. It launches workloads using `docker exec ... sh -c 'nohup python3 workload.py ... > /dev/null 2>&1 & echo $!'` to capture the exact PID of the python process.
2. It kills the workload using the shell builtin `kill -9 <PID>` inside the container.
3. It verifies termination by checking if the `/proc/<PID>` directory still exists.

## I. Constant Container ID Evidence
```text
Container ID: f403440d12746cdf1f4375aedb9ecb9619884063237cfb71f139dbc53420cb81
--- Transitioned to idle. PIDs before: [], PIDs after: [13] ---
--- Transitioned to cpu. PIDs before: [], PIDs after: [37] ---
--- Transitioned to idle. PIDs before: [], PIDs after: [62] ---
```
The container ID never changes.

## J. Non-Stacking Workload Evidence
At every transition, the `PIDs before` is exactly `[]` and the `PIDs after` is exactly `[<new_pid>]`. Zero stacking occurs.

## K. Delta Baseline Normalization
```text
Cycle 16 | Mode: idle   | ns_delta:     588000
Cycle 24 | Mode: cpu    | ns_delta: 4989919000
Cycle 26 | Mode: idle   | ns_delta:     525000
```
The delta correctly spikes to ~5 billion nanoseconds (5 seconds) during the 100% CPU phase, and returns to baseline (~500k ns) during idle.

## L. Anomaly Score Normalization
During the initial idle warmup, anomaly scores are low and stable:
```text
Cycle 12 | Mode: idle   | Anomaly:    0.0410 (is_ano=False)
```
Upon entering the CPU phase, the anomaly correctly triggers:
```text
Cycle 17 | Mode: cpu    | Anomaly:    1.1726 (is_ano=True)
Cycle 24 | Mode: cpu    | Anomaly:   33.2047 (is_ano=True)
```
Upon returning to idle, the anomaly score begins to decay as the 12-cycle rolling window flushes out the CPU metrics:
```text
Cycle 26 | Mode: idle   | Anomaly:   39.7190 (is_ano=True)
Cycle 32 | Mode: idle   | Anomaly:   38.2520 (is_ano=True)
```

## M. Remaining Issues
The feature conflict detailed in section **E/F** remains. Because we are providing the raw delta to satisfy the Autoencoder and the structural sequence requirement, the Transformer's predictions will be heavily distorted as its loaded `Normalizer` expects `log1p(delta)`. If this prediction distortion affects the DQN's performance, the architecture may need to be revised to decouple the feature representations before prediction.
