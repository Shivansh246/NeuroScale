# NeuroScale

Autonomous Docker resource manager prototype combining workload generation, container metrics monitoring, and dynamic resource control.

## System Architecture

```
Workload Generator
       ↓
Docker Container
       ↓
Metrics Collector
       ↓
Structured Metrics
       ↓
Docker Controller
```

Future roadmap components (in upcoming milestones):
- Predictive forecasting (Lightweight Transformer)
- Anomaly detection (Autoencoder)
- Reinforcement learning policy (DRL Agent)
- SLA monitoring & live dashboard

---

## Foundation Modules

1. **Workload Generator (`workload/workload.py`, `Dockerfile`)**
   - Controlled workload modes:
     - `idle`: Low/zero utilization sleep cycle.
     - `cpu`: High-intensity continuous CPU arithmetic burning cycles.
     - `memory`: Memory buffer allocation touching RSS pages to guarantee physical cgroup memory accounting.
     - `burst`: Alternating active CPU spike cycles (`--burst-on`) and idle rest periods (`--burst-off`).
     - `mixed`: Simultaneous CPU execution while maintaining resident memory pressure.
   - Graceful termination on `SIGINT` / `SIGTERM`.

2. **Metrics Collector (`monitoring/collector.py`)**
   - Collects container statistics via Docker API and generates structured metrics:
     - `timestamp`: Float epoch time.
     - `container_id` & `container_name`: Target container identifiers.
     - `cpu_percent`: Normalized percentage based on container vs system CPU deltas and online cores.
     - `cpu_usage_ns`: Total container CPU time in nanoseconds.
     - `memory_usage_bytes` & `memory_usage_mb`: Current memory usage.
     - `memory_limit_bytes` & `memory_limit_mb`: Configured memory limits.
     - `memory_percent`: Ratio of memory usage to limit (%).
     - `workload_mode`: Automatically detected from container labels (`neuroscale.mode`), CLI command args (`--mode`), or environment variables (`WORKLOAD_MODE`).
     - `cpu`, `memory`, `network`, `disk_io`: Preserved detailed raw metrics.

3. **Docker Controller (`controller/controller.py`)**
   - Applies dynamic CPU quotas (`cpu_period=100000`, `cpu_quota=cpu * cpu_period`) and memory limits (`mem_limit`).
   - Strict input validation and sensible bounds checking (`min_cpu=0.1`, `max_cpu=16.0`, `min_memory_mb=64`, `max_memory_mb=32768`).
   - Granular Docker error handling (`NotFound`, `APIError`, `DockerException`) with fallback for systems without swap accounting.
   - Structured JSON response containing `success`, `container_id`, `applied`, and `error`.

---

## Quickstart & Usage

### 1. Run Tests
The test suite uses Python's standard `unittest` framework and requires no external testing dependencies:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Run Workload Directly (Host)
```bash
python workload/workload.py --mode burst --duration 30 --burst-on 2.0 --burst-off 1.0
python workload/workload.py --mode mixed --duration 30 --memory 128
```

### 3. Build & Run Workload in Docker
```bash
# Build the Docker image
docker build -t neuroscale-workload .

# Run a container with controlled workload
docker run -d --name neuroscale-test neuroscale-workload --mode cpu --duration 300

# Inspect stats
docker stats --no-stream
```

### 4. Collect Metrics
```python
from monitoring.collector import Collector

collector = Collector()
metrics = collector.collect("neuroscale-test")
print(metrics)
```

### 5. Control Resources Dynamically
```python
from controller.controller import Controller

controller = Controller()
result = controller.apply(
    container_id="neuroscale-test",
    cpu=1.5,      # 1.5 cores
    memory=512    # 512 MB
)
print(result)
```

---

## Phase 3: Transformer Prediction Module

### Architecture

```
Input (batch, seq_len, feature_count)
  → Linear projection → (batch, seq_len, d_model)
  → Sinusoidal Positional Encoding
  → Transformer Encoder (N layers)
  → Last prediction_horizon time-steps
  → Linear prediction head
  → Output (batch, prediction_horizon, 2)  ← [cpu_percent, memory_percent]
```

### Model Configuration (`models/config.py`)

| Parameter            | Default | Description                                     |
|----------------------|---------|-------------------------------------------------|
| `d_model`            | 64      | Embedding dimension                             |
| `nhead`              | 4       | Attention heads                                 |
| `num_encoder_layers` | 2       | Stacked encoder layers                          |
| `dim_feedforward`    | 128     | Feed-forward hidden dimension                   |
| `dropout`            | 0.1     | Dropout probability                             |
| `prediction_horizon` | 1       | Future steps to predict                         |
| `num_targets`        | 2       | CPU % and memory %                              |
| `feature_count`      | 4       | Must match `len(PipelineConfig.feature_columns)`|

### Training Command

```bash
# First collect data (Phase 2)
python -m data.collector --container <id> --out data/raw_metrics.jsonl --duration 600

# Then train the Transformer
python -m training.transformer_trainer \
    --raw-data data/raw_metrics.jsonl \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints \
    --seed 42
```

### Checkpoint

Saved to `checkpoints/best_transformer.pt` (not committed to Git).

Contains: `state_dict`, `model_config`, `feature_columns`, `target_columns`,
`normalization` (mean/std), `epoch`, `val_loss`, `train_loss`.

### Inference Example

```python
from models.predictor import TransformerPredictor
import numpy as np

predictor = TransformerPredictor.from_checkpoint("checkpoints/best_transformer.pt")

# sequence: (seq_len=12, feature_count=4) in original (un-normalised) scale
sequence = np.array(...)   # your 12-step window
result = predictor.predict(sequence)
# → {"cpu": 34.7, "memory": 12.1, "confidence": 0.61}
```

### Metrics

Evaluated per target (CPU, memory) and overall:

| Metric | Description                       |
|--------|-----------------------------------|
| MSE    | Mean Squared Error                |
| MAE    | Mean Absolute Error               |
| RMSE   | Root Mean Squared Error           |

```python
from evaluation.transformer_metrics import full_report
report = full_report(y_true, y_pred)
# → {"cpu": {"mse": ..., "mae": ..., "rmse": ...},
#    "memory": {...}, "overall": {...}}
```

### Confidence Score

Confidence is a **deterministic heuristic**:

```
confidence = exp(-sqrt(val_loss) / CONFIDENCE_SCALE)
```

where `val_loss` is the MSE on the validation set (in normalised units) saved
in the checkpoint, and `CONFIDENCE_SCALE = 1.0`.

- `val_loss = 0.0` → confidence = 1.0
- `val_loss = 1.0` (RMSE = 1 normalised unit) → confidence ≈ 0.37
- `val_loss = 4.0` → confidence ≈ 0.14

This is fixed at checkpoint load time and does not vary per prediction.

### Synthetic Demonstration

```bash
source .venv/bin/activate
python -m unittest tests.test_transformer -v
# Ran 34 tests in ~1 s — OK
```
