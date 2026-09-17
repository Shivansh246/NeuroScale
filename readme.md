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
# Ran 46 tests in ~1.2 s — OK
```

---

## Phase 4 — Autoencoder Anomaly Detection

Lightweight fully-connected PyTorch autoencoder that flags anomalous workload behavior by measuring metric-window reconstruction error.

### What It Detects

The autoencoder learns the typical correlation patterns across time steps and resource channels (CPU percentage, CPU usage ns, memory MB, memory percentage) during normal operation. It detects:
- Sudden uncharacteristic CPU spikes or drops
- Unexpected memory leakage or rapid consumption
- Out-of-distribution workload patterns that violate historical operating bounds

### Reconstruction Error to Anomaly Score

Input sequences of shape `(seq_len, feature_count)` are normalised using training statistics and flattened into a 1D vector of dimension `seq_len * feature_count`. The autoencoder compresses this into a low-dimensional bottleneck (latent representation) and attempts to reconstruct the input.

The **anomaly score** is the Mean Squared Error (MSE) between the normalised input $x$ and its reconstruction $\hat{x}$:

$$\text{score} = \frac{1}{D} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$

- Normal patterns: low reconstruction error (score < threshold)
- Anomalous/perturbed patterns: high reconstruction error (score > threshold)

### Threshold Calculation

The decision threshold is determined deterministically from reconstruction errors on the normal validation set after training:

$$\text{threshold} = \mu_{\text{val}} + k \cdot \sigma_{\text{val}}$$

where:
- $\mu_{\text{val}}$ is the mean validation reconstruction error
- $\sigma_{\text{val}}$ is the standard deviation of validation reconstruction errors
- $k$ is a configurable multiplier (`threshold_k`, default `3.0`)

The threshold is persisted inside the checkpoint so that inference never recomputes it.

### Inference Usage

```python
import numpy as np
from anomaly.detector import AutoencoderAnomalyDetector

detector = AutoencoderAnomalyDetector.from_checkpoint("checkpoints/best_autoencoder.pt")

# sequence: (seq_len=12, feature_count=4) in raw (un-normalised) scale
sequence = np.array(...)  # current 12-sample metric window
result = detector.score(sequence)
# → {"score": 0.0023, "is_anomaly": False}
```

### Limitations

1. **Static Threshold**: The threshold is derived from offline validation data and does not adapt dynamically to slow, benign workload shifts.
2. **Deterministic Distance, Not Probability**: The anomaly score represents geometric reconstruction error in normalised feature space, not a calibrated posterior probability.
3. **Training Data Sensitivity**: The detector assumes training data is purely normal. Contamination in training data inflates the threshold and reduces sensitivity.
4. **Point Window Evaluation**: Evaluates fixed-length sliding windows independently without cross-window temporal memory.

---

## Phase 5: DRL Environment + Control Policy Interface

### Architecture

NeuroScale models Docker resource allocation as a Markov Decision Process (MDP). Phase 5 introduces a lightweight, deterministic Gymnasium-compatible RL environment (`NeuroScaleEnv`) that isolates reinforcement learning from the live Docker runtime during offline training.

### State Space

The state is a 1D, 11-dimensional normalized `float32` vector:
1. `current_cpu_util`
2. `current_mem_util`
3. `predicted_cpu_demand` (from Transformer)
4. `predicted_mem_demand` (from Transformer)
5. `current_cpu_alloc`
6. `current_mem_alloc`
7. `sla_latency`
8. `anomaly_score` (from Autoencoder)
9. `is_anomaly` (boolean -> 0/1)
10. `host_cpu_avail`
11. `host_mem_avail`

### Action Space

The action space is a small discrete Cartesian product mapping an index to a specific CPU/Memory allocation pair, configured via `ActionConfig`.
- `cpu_options`: [0.25, 0.5, 1.0, 2.0] cores
- `memory_options`: [128, 256, 512, 1024] MB
Total Actions = 16.

### Reward Function

Rewards are modeled as penalties to be minimized:
```python
reward = - (sla_violation_penalty)
         - (resource_waste_penalty)
         - (under_provision_penalty)
         - (reallocation_penalty)
         - (anomaly_penalty)
```

### Simulation Model

`NeuroScaleEnv` operates purely on deterministic traces (e.g. `[{"cpu_util": ..., "mem_util": ...}, ...]`).
It accurately simulates:
- **Throttling**: If allocated CPU is less than the demand in the trace, utilization is capped.
- **SLA Violations**: Latency spikes proportionally to CPU/Memory deficits, simulating compute-starvation and OOM thrashing, generating the negative reward signal necessary for DQN training.

---

## Phase 6: DQN-based DRL Agent

### Architecture
The Deep Q-Network (DQN) agent maps the 11-dimensional normalized state vector from Phase 5 into 16 Q-values representing the expected future reward for each discrete CPU/Memory action.

The architecture is a lightweight multi-layer perceptron (MLP) in PyTorch:
```
Input(11) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(16)
```

Why DQN? The action space is a small discrete Cartesian product (4 CPU options × 4 Memory options = 16 actions). DQN is highly sample-efficient and stable for small discrete action spaces compared to continuous policy gradient methods like PPO or SAC.

### Components

- **Epsilon-Greedy Exploration**: During training, the agent explores random actions with probability `epsilon` (decaying from 1.0 to 0.05). During evaluation and inference, it acts greedily (`argmax Q(s, a)`).
- **Replay Buffer**: A fixed-capacity deterministic buffer (`capacity=50000`) stores MDP transitions `(state, action, reward, next_state, done)` to break temporal correlations during mini-batch training.
- **Target Network**: To stabilize learning, the Q-learning target is calculated using a separate Target Network `Q_target`, which is periodically synchronized with the online network.
- **Reward / Objective**: The agent minimizes the penalties defined in Phase 5 (SLA violations, resource waste, under-provisioning, reallocations). The loss function is the Huber Loss (`SmoothL1Loss`), optimizing the Bellman equation.

### Training & Evaluation

The DQN agent is trained offline against the deterministic simulation environment (`NeuroScaleEnv`) without interacting with Docker.

To run a sample training loop programmatically:
```python
from rl.agent import DQNAgentConfig
from rl.config import EnvConfig
from rl.trainer import DQNTrainer

trainer = DQNTrainer(DQNAgentConfig(), EnvConfig())
history = trainer.train(num_episodes=200, max_steps_per_episode=50, eval_freq=10)
```

During training, the agent is periodically evaluated on deterministic synthetic scenarios (e.g. CPU spikes, Memory spikes, under-provisioning cases). The best model is saved.

### Checkpoint and Public API

**Checkpoint Location**: `checkpoints/best_dqn.pt`

**Public Interface**:
```python
from rl.agent import DQNAgent

# Load pre-trained agent
agent = DQNAgent.load("checkpoints/best_dqn.pt")

# Predict optimal resource allocation
state_vector = np.array([...]) # 11-dim normalized state
action = agent.choose_action(state_vector)
# -> {"cpu": 1.0, "memory": 256}
```

### Limitations
- **Offline Training**: The agent is trained purely on synthetic simulation traces.
- **No Docker Feedback Loop**: Phase 6 implements the agent, but it is not yet wired up to the live metric collector and Docker controller (planned for Phase 7).
- **Static Action Space**: The agent cannot interpolate between the predefined discrete allocations.
