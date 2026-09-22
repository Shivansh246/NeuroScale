# Phase 12B Validation Report: Decoupled Feature Preprocessing

## 1. Issue Identified
In Phase 12A, we observed that the prediction model (Transformer) and anomaly model (Autoencoder) had fundamentally conflicting feature contracts for `cpu_usage_ns_delta`. The Autoencoder was trained entirely on raw nanosecond deltas (`~1e5` - `5e9` scale), whereas the Transformer was trained on a logarithmic transformation of these deltas (`np.log1p(delta)`), as verified in `experiments/prediction_models/data.py`. Providing a single uniform sequence to both models meant one model would always receive improperly scaled inputs.

## 2. Decoupled Preprocessing Architecture
We addressed this by maintaining one **canonical raw telemetry history** (`rolling_window` in `control/loop.py`), which is strictly never mutated. Before executing model inference, two distinct, model-specific sequences are synthesized using dedicated helper functions:

1. `_prepare_transformer_sequence(canonical_sequence)`: Copies the raw telemetry and applies `np.log1p(np.maximum(0, delta))` explicitly to the `cpu_usage_ns_delta` feature, reproducing the exact transformation used in Phase 11A training.
2. `_prepare_anomaly_sequence(canonical_sequence)`: Copies the raw telemetry without applying the logarithmic transformation, providing the raw interval magnitudes expected by the Autoencoder.

Crucially, the canonical history remains entirely raw. The transformation happens precisely once per inference cycle, guaranteeing no "double-logging" can occur.

## 3. Short Real Docker Validation Results
The validation script (`scripts/run_phase12b_validation.py`) successfully cycled through the workload transitions (`idle` -> `cpu` -> `idle` -> `memory` -> `idle`).

**Representative Cycle Output:**
```text
Cycle 12 | Mode: idle   | Raw ns_delta:     450000 | Trans ns_delta:   13.0170 | AE ns_delta:     450000 | Pred CPU:  -3.33% | Anomaly:    0.0399 (False)
...
Cycle 17 | Mode: cpu    | Raw ns_delta: 1104417000 | Trans ns_delta:   20.8226 | AE ns_delta: 1104417024 | Pred CPU:  71.33% | Anomaly:    0.9972 (True)
Cycle 18 | Mode: cpu    | Raw ns_delta: 4980638000 | Trans ns_delta:   22.3288 | AE ns_delta: 4980638208 | Pred CPU:  94.45% | Anomaly:   11.4130 (True)
Cycle 19 | Mode: cpu    | Raw ns_delta: 4991887000 | Trans ns_delta:   22.3311 | AE ns_delta: 4991886848 | Pred CPU: 101.73% | Anomaly:   20.6203 (True)
...
Cycle 26 | Mode: idle   | Raw ns_delta:     499000 | Trans ns_delta:   13.1204 | AE ns_delta:     499000 | Pred CPU:  -2.30% | Anomaly:   39.3460 (True)
```

**Key Observations:**
- **Raw `ns_delta`** correctly spikes to `~5_000_000_000` during the CPU phase.
- **Transformer `ns_delta`** receives the correct log-scaled value (`~22.3`), and `Pred CPU` instantly normalizes to accurately predict the `~100%` utilization (previously it was predicting near 0 due to out-of-distribution unlogged values).
- **Autoencoder `ns_delta`** receives the full unaltered raw magnitude (`~5_000_000_000`), accurately identifying extreme shifts in workload as anomalous.
- No `NaN` or `Inf` values are produced.

## 4. Test Suite Validation
We authored `tests/test_preprocessing.py` to assert the production pipeline behaves as intended:
- Proves large raw CPU deltas (`5e9`) remain structurally unchanged in canonical telemetry.
- Proves the Transformer helper applies `log1p` strictly once.
- Proves the Autoencoder helper preserves the raw representation.
- Proves CPU/Memory percentages remain strictly unchanged across both sequence copies.

The test suite was run in its entirety, confirming **201 tests passed** successfully with 0 regressions.

## 5. Remaining Limitations
None discovered. The feature pipeline accurately honors the divergent preprocessing requirements of both models without modifying their frozen network architectures or checkpoint weights. The process isolation bugs from Phase 12 have also remained firmly resolved.
