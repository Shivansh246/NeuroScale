# NeuroScale: Autonomous Vertical Pod Autoscaler

NeuroScale is an experimental research prototype of a dynamic, autonomous vertical container autoscaler that continuously perceives, predicts, detects, and optimizes compute resource allocations for Linux container workloads. By combining a Multi-Head Attention Transformer for predictive demand forecasting, a Deep Autoencoder for unsupervised workload anomaly detection, and a Resource-Factorized Deep Q-Network (DQN) with dual action heads, NeuroScale regulates CPU quotas and memory limits directly on live Docker containers via Linux cgroups in real time without human intervention.

> **Research Prototype Notice**: NeuroScale is designed and implemented as an academic and engineering prototype to investigate neural perception and factorized reinforcement learning for container resource control. It is not intended for production-critical Kubernetes or container orchestration environments.

---

## 1. The Autonomous Control Loop

NeuroScale executes an unbroken, closed-loop feedback cycle:

$$\text{observe} \longrightarrow \text{predict} \longrightarrow \text{detect} \longrightarrow \text{decide} \longrightarrow \text{control} \longrightarrow \text{measure} \longrightarrow \text{repeat}$$

1. **Observe**: The metric collector reads instantaneous cgroup resource metrics (`cpu_percent`, `cpu_usage_ns`, `memory_usage_mb`, `memory_percent`) from the target container.
2. **Predict**: A 12-step sequence is passed to a Transformer sequence model to forecast next-horizon CPU and memory demand along with an uncertainty confidence score.
3. **Detect**: An Autoencoder reconstructs the feature window, calculating reconstruction mean squared error (MSE) against a threshold to flag workload anomalies.
4. **Decide**: The 11-dimensional normalized state vector is fed to a Resource-Factorized Deep Q-Network, selecting discrete action branch indices for CPU and Memory.
5. **Control**: The Docker Controller applies the chosen allocation (`CpuQuota` and `Memory` limits) via the Docker Engine API.
6. **Measure**: The container's kernel `HostConfig` is immediately inspected to verify that the OS cgroups took effect.
7. **Repeat**: The cycle sleeps for the configured sampling interval (1.0s) and repeats.

---

## 2. Architecture & Module Overview

```
                      ┌─────────────────────────────────────────┐
                      │          Real Docker Container          │
                      │       (neuroscale-workload:latest)      │
                      └────────────────────┬────────────────────┘
                                           │ Linux cgroups metrics
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │       Monitoring Collector (cgroups)    │
                      │   cpu_percent, cpu_ns, mem_mb, mem_%    │
                      └────────────────────┬────────────────────┘
                                           │ 12-step Rolling Window
                                           ▼
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
          ┌───────────────────────────┐         ┌───────────────────────────┐
          │   Transformer Predictor   │         │    Autoencoder Detector   │
          │  Next CPU & Mem Demand %  │         │ Reconstruction Loss Score │
          └─────────────┬─────────────┘         └─────────────┬─────────────┘
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           │ Raw State Vector
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │         StateBuilder (11-dim)           │
                      │  Normalized utilization, demands, SLA,  │
                      │     anomaly flags, host headroom        │
                      └────────────────────┬────────────────────┘
                                           │ Normalized Tensor
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │     Resource-Factorized DQN Agent       │
                      │      Shared Trunk (11 -> 64 -> 64)      │
                      │    Q_cpu: 4 actions, Q_mem: 4 actions   │
                      │    joint_index = cpu_idx * 4 + mem_idx  │
                      └────────────────────┬────────────────────┘
                                           │ Discrete Action: {cpu, memory}
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │         Docker Controller API           │
                      │    CpuQuota (period 100k), mem_limit    │
                      └────────────────────┬────────────────────┘
                                           │ cgroups Update & HostConfig Verification
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Verified Docker State (Cpu & Mem)    │
                      └─────────────────────────────────────────┘
```

- **`control/`**: Real Docker closed loop, metric collection (`Collector`), API control (`Controller`), loop timing, and structured JSONL logging (`MetricsLogger`).
- **`models/`**: Transformer demand predictor (`TransformerPredictor`), rolling sequence buffer, and 11-dimensional state representation builder (`StateBuilder`).
- **`anomaly/`**: Bottleneck Autoencoder reconstruction anomaly detector (`AutoencoderAnomalyDetector`).
- **`rl/`**: Deep Reinforcement Learning agents and environments:
  - `rl/dqn.py`: Monolithic 16-action DQN baseline.
  - `rl/decoupled_dqn.py`: Fully independent dual-network DQN experiment.
  - `rl/resource_factorized_dqn.py`: Factorized shared-trunk dual-head DQN.
- **`dashboard/`**: Interactive Streamlit monitoring dashboard (`app.py`).
- **`scripts/`**: Automation scripts for training, evaluation, data generation, and live Docker demonstration.
- **`tests/`**: Automated unit and integration test suite (174 tests).
- **`docs/`**: Technical reports, evidence matrices, architecture specifications, and presentation materials.

---

## 3. Technology Stack

- **Core Runtime**: Python 3.12+ / 3.14
- **Deep Learning Framework**: PyTorch (CPU-optimized forward inference)
- **Container Virtualization**: Docker Engine API (`docker-py`), Linux cgroups (v1/v2 CFS quota and memory limits)
- **Monitoring & Visualization**: Streamlit, Plotly, Pandas, NumPy
- **Testing & Verification**: Python `unittest`, integration testing fixtures

---

## 4. Installation & Setup

### Prerequisites
- Linux Operating System (Ubuntu 22.04+ or similar)
- Docker Daemon installed and running (`systemctl status docker`)
- Python 3.12+ with virtual environment tools

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/Shivansh246/NeuroScale.git
cd NeuroScale

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build the workload container image
docker build -t neuroscale-workload:latest -f docker/Dockerfile.workload docker/
```

---

## 5. Execution & Usage

### Running the Test Suite
Execute the full test suite (172 passed, 2 skipped, 0 failures/errors):
```bash
python -m unittest discover tests -v
```

### Running the Real Docker Closed-Loop Demonstration
Run the 4-phase automated demonstration against a live Docker container:
```bash
python scripts/demonstrate_real_docker.py
```
This executes 42 cycles across Warmup, Normal, CPU Spike, Memory Spike, and Recovery regimes, recording telemetry to `data/real_docker_demo.jsonl`.

### Launching the Monitoring Dashboard
Launch the interactive web dashboard to visualize live metrics, model predictions, and cgroup limit changes:
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser and select **"Real Docker Demo (Live cgroups)"** from the sidebar.

### Running Comparative Baseline Evaluations
Evaluate the baseline heuristics and RL agents against synthetic workload scenarios:
```bash
python scripts/evaluate_comparison.py
```

---

## 6. Evaluation Summary

Evaluations conducted across synthetic test scenarios (`normal`, `cpu_spike`, `mem_spike`, `sustained`) and live Docker demonstrations established:

- **Heuristic Baselines**: Static allocation ($1.0\text{C} / 256\text{MB}$) incurs high SLA violations ($15$ to $35$) under stress; reactive thresholds scale memory but lag sudden spikes.
- **Monolithic DQN**: Completely eliminates SLA violations ($0$ violations across all scenarios) by learning defensive scaling, but collapses into rigid bimodal switching between Action 1 ($0.25\text{C}/256\text{MB}$, $56.5\%$) and Action 15 ($2.0\text{C}/1024\text{MB}$, $43.5\%$).
- **Resource-Factorized DQN**: Populates intermediate scaling actions ($A5$ at $0.5\text{C}/256\text{MB}$ and $A6$ at $0.5\text{C}/512\text{MB}$ represent $83.1\%$ of active steps). Under live Docker conditions, it operates in **Class B (co-scaling / partial specialization)** mode.
- **System Timing**: Pure Docker API control updates execute in **$11.4\text{ ms}$**; neural model inference executes in **$2.8\text{ ms}$** on CPU.

---

## 7. Major Empirical Limitations

NeuroScale's findings are grounded in audited empirical data with the following prototype limitations:

1. **Recovery & Downscaling**: Automated downscaling was not observed in live Docker runs (0 downscale events across 20 recovery cycles). Rolling window residual lag and cumulative counter saturation keep the policy at Action 15 throughout recovery.
2. **Memory Forecasting Accuracy**: The Transformer accurately captured CPU utilization surges ($71.2\%$ pred vs $74.3\%$ actual), but significantly underpredicted large memory spikes ($27.5\text{ MB}$ pred vs $316.1\text{ MB}$ actual mean, MAE: $296.9\text{ MB}$).
3. **Quiescent Anomaly False-Positives**: Idle containers ($0.01\%$ CPU, $10.9\text{ MB}$) lie $\sim 2\sigma$ below synthetic sinusoidal training data, producing an out-of-distribution reconstruction error of $0.0814$ (exceeding the $0.003814$ threshold) and asserting `is_anomaly=True`.
4. **Class B Co-Scaling**: Rather than independent CPU-only or memory-only autoscaling, the factorized heads co-scale to maximum branch ($2.0\text{C} / 1024\text{MB}$) under anomaly conditions.
5. **Headroom vs OOM Proof**: Demonstrating that a $417.8\text{ MB}$ workload fits safely within an applied $1024\text{ MB}$ limit proves adequate headroom was provided, but does not constitute proof that an otherwise inevitable OOM crash was prevented.

---

## 8. Repository Structure

```
NeuroScaler/
├── anomaly/                       # Autoencoder anomaly detector
│   ├── __init__.py
│   └── detector.py
├── checkpoints/                   # Trained neural model checkpoints
│   ├── best_autoencoder.pt        # Trained Autoencoder weights
│   ├── best_decoupled_dqn.pt      # Decoupled DQN weights
│   ├── best_dqn.pt                # Monolithic DQN baseline weights
│   ├── best_resource_factorized_dqn.pt # Resource-Factorized DQN weights
│   └── best_transformer.pt        # Transformer sequence predictor weights
├── control/                       # Real Docker closed-loop controller
│   ├── collector.py               # cgroups telemetry polling
│   ├── config.py                  # Loop configuration & parameters
│   ├── controller.py              # Docker Engine API actuation
│   ├── logger.py                  # JSONL telemetry logging
│   └── loop.py                    # Closed-loop orchestrator
├── dashboard/                     # Streamlit monitoring dashboard
│   └── app.py
├── docker/                        # Dockerfiles & container workload definitions
│   ├── Dockerfile.workload
│   └── workload.py                # CPU & RSS memory stress scripts
├── docs/                          # Technical reports, evidence, & presentation docs
│   ├── architecture.md            # System architecture & Mermaid flowcharts
│   ├── demo_guide.md              # Demonstration execution manual & talking points
│   ├── experiments.md             # Baseline & RL experimental progression
│   ├── final_evidence.md          # Audited master evidence matrix
│   └── presentation_metrics.md    # Compact PPT-ready summary tables
├── models/                        # Neural network architectures & state processing
│   ├── predictor.py               # Multi-head attention Transformer predictor
│   ├── state.py                   # 11-dimensional normalized state builder
│   └── transformer.py             # PyTorch Transformer model definition
├── rl/                            # Reinforcement learning algorithms & environments
│   ├── agent.py                   # Monolithic DQN agent
│   ├── config.py                  # Action space & reward weight parameters
│   ├── decoupled_agent.py         # Independent decoupled DQN agent
│   ├── decoupled_dqn.py           # Independent dual-network model
│   ├── environment.py             # Simulated gym environment
│   ├── resource_factorized_agent.py # Factorized dual-head DQN agent
│   ├── resource_factorized_dqn.py # Shared-trunk dual-head model
│   └── reward.py                  # Multi-objective reward function
├── scripts/                       # Experiment runners & live demonstration scripts
│   ├── demonstrate_isolated_stimuli.py # Isolated CPU/Memory stress runner
│   ├── demonstrate_real_docker.py # 4-phase live Docker demonstration runner
│   ├── demonstrate_recovery.py    # Dedicated 38-cycle recovery runner
│   ├── evaluate_comparison.py     # Multi-baseline evaluation runner
│   └── run_closed_loop.py         # CLI closed-loop driver
├── tests/                         # Unit and integration test suite
│   ├── test_real_model_closed_loop.py # End-to-end integration tests
│   └── ...
├── real_docker_demo_report.md     # Audited real Docker integration report
├── resource_factorized_report.md  # Audited Resource-Factorized DQN report
└── requirements.txt               # Python package dependencies
```
