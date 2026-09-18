# NeuroScale: Final Evidence Summary & Empirical Audit

**Project**: NeuroScale: Dynamic Autonomous Vertical Pod Autoscaler  
**Date**: September 18, 2026  
**Status**: Audited Ground Truth & Verified Claims  

---

## 1. Master Evidence Matrix

| # | Evidence Item | Experiment / Source | Empirical Result | Status | Specific Limitation |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Real Telemetry** | `control/collector.py`, `scripts/demonstrate_real_docker.py` | Extracted instantaneous CPU %, cumulative CPU ns, RSS memory MB, and memory % via Docker API across 42 cycles. | **Verified** | Metric polling blocked for $\sim 1004.8\text{ ms}$ due to Docker `stats()` internal 1s sleep for CPU delta computation. |
| **2** | **Workload Stimulus** | `scripts/demonstrate_real_docker.py`, `workload.py` | Triggered real guest OS processes inside container. Verified via `container.top()`: PID 31032 active with 99% CPU load. | **Verified** | Stimulus is an artificial stress script (`matrix multiplication` / `bytearray`), not a real user-facing web microservice. |
| **3** | **Transformer Inference** | `models/predictor.py`, `checkpoints/best_transformer.pt` | Evaluated 12-step sequence in $\sim 1.2\text{ ms}$; produced CPU & RAM predictions with confidence score ($0.8429$). | **Verified** | Predictions represent normalized future steps; high coupling observed between CPU and RAM outputs. |
| **4** | **Autoencoder Detection** | `anomaly/detector.py`, `checkpoints/best_autoencoder.pt` | Evaluated 48-dim feature vector in $\sim 0.4\text{ ms}$; generated dynamic reconstruction MSE scores. | **Verified** | Emitted false-positive ($0.0814 > 0.003814$) during quiescent idle state due to out-of-distribution training centering. |
| **5** | **Resource-Factorized DQN** | `rl/resource_factorized_agent.py`, `checkpoints/best_resource_factorized_dqn.pt` | Inferred branch actions $(cpu\_idx, mem\_idx)$ in $\sim 0.9\text{ ms}$; reconstructed joint action index ($cpu \times 4 + mem$). | **Verified** | Demonstrates Class B co-scaling under anomaly conditions rather than clean independent dimension scaling. |
| **6** | **Docker Control API** | `control/controller.py`, Docker Engine API | Successfully executed `POST /containers/{id}/update` in $\sim 11.4\text{ ms}$; applied quota and byte limits to cgroups. | **Verified** | Limited to Docker Linux cgroups v1/v2 update API; does not integrate with Kubernetes API server. |
| **7** | **Physical CPU Scaling** | `container.reload()`, `HostConfig.CpuQuota` | Quota transitioned from $100,000\text{ \mu s}$ ($1.0\text{C}$) to $200,000\text{ \mu s}$ ($2.0\text{C}$) in kernel cgroups. | **Verified** | Remained escalated at $2.0\text{C}$ throughout stress and recovery phases. |
| **8** | **Physical Memory Scaling** | `container.reload()`, `HostConfig.Memory` | Memory limit transitioned from $268,435,456\text{ B}$ ($256\text{ MB}$) to $1,073,741,824\text{ B}$ ($1024\text{ MB}$). | **Verified** | Workload peaked at $417.83\text{ MB}$; allocation provided headroom rather than tight proportional right-sizing. |
| **9** | **SLA Latency Tracking** | `control/loop.py`, `data/real_docker_demo.jsonl` | Monitored response latency remained at $50.0\text{ ms}$; zero SLA violations ($>200\text{ ms}$) across 42 cycles. | **Verified** | SLA metric is monitored synthetic telemetry; not measured from live external HTTP ingress. |
| **10** | **Controller Overhead** | System timing micro-benchmark | Net controller execution is $\sim 11.4\text{ ms}$; neural pipeline is $\sim 2.8\text{ ms}$. | **Verified** | Overall cycle wall time is $\sim 1021.5\text{ ms}$ due to Docker cgroup delta polling requirements. |
| **11** | **Prediction Accuracy** | `data/real_docker_demo.jsonl` (31 active cycles) | CPU MAE: $40.29\%$ in CPU spike; tracked surge trend. Memory MAE: $296.87\text{ MB}$ in memory spike. | **Partially Verified** | Memory prediction accuracy is poor; failed to anticipate large allocation surges ($27.55\text{ MB}$ pred vs $316.12\text{ MB}$ actual). |
| **12** | **Recovery & Downscaling** | `data/real_docker_recovery_demo.jsonl` (38 cycles) | 0 downscale events observed across 20 recovery cycles; policy maintained Action 15 ($2.0\text{C} / 1024\text{MB}$). | **Not Observed** | Downscaling prevented by 12-step rolling window history lag and persistent cumulative counter anomaly flags. |
| **13** | **Isolated CPU Test** | `data/real_docker_isolated_demo.jsonl` (Cycles 1..18) | Under CPU stress ($98\%$), agent allocated $2.0\text{C} / 1024\text{MB}$ (Action 15). | **Partially Verified** | Memory co-scaled with CPU to maximum limit instead of remaining at baseline. |
| **14** | **Isolated Memory Test** | `data/real_docker_isolated_demo.jsonl` (Cycles 19..36) | Under Memory stress ($412\text{ MB}$), agent allocated $2.0\text{C} / 1024\text{MB}$ (Action 15). | **Partially Verified** | CPU co-scaled with Memory to maximum limit instead of remaining at baseline. |

---

## 2. What NeuroScale Demonstrates

1. **Autonomous Closed-Loop Control**: Demonstrates a complete, functional closed loop (`observe → predict → detect → decide → control → measure → repeat`) driving physical Linux container cgroups in real time without human intervention.
2. **Multi-Model Neural Pipeline Integration**: Demonstrates that real Transformer sequence models, Autoencoder anomaly detectors, and Deep Q-Networks can be composed into a unified inference pipeline executing within **$2.8\text{ ms}$** on commodity CPU hardware.
3. **Low Latency Operating System Interaction**: Demonstrates that physical resource limit adjustments via the Docker Engine API execute in **$11.4\text{ ms}$**, satisfying real-time systems requirements.
4. **Physical Resource Provisioning**: Demonstrates verified kernel-level updates to `CpuQuota` and `Memory` limits matching discrete RL policy decisions.
5. **Stability Under Telemetry Perturbations**: Zero system crashes, NaN propagations, or SLA violations observed across all real Docker test runs.

---

## 3. What NeuroScale Does Not Demonstrate

1. **Independent Resource Dimension Scaling**: Does not demonstrate clean decoupled CPU-only or memory-only autoscaling in live Docker. The factorized DQN exhibited Class B co-scaling (Action 15 across both heads under stress).
2. **Dynamic Autonomous Down-Scaling**: Does not demonstrate automated capacity reclaiming upon workload cessation. Across 20 dedicated recovery cycles, zero downscale events occurred due to rolling window history retention and saturated anomaly scores.
3. **Accurate Memory Forecasting**: Does not demonstrate accurate future memory demand prediction (MAE: $296.87\text{ MB}$ during memory spikes).
4. **In-Distribution Anomaly Detection for Quiescent States**: Does not demonstrate robust anomaly classification on idle workloads; quiescent containers trigger out-of-distribution false positives ($0.0814 > 0.003814$).
5. **Absolute Proof of Preventing Inevitable OOM Crashes**: Demonstrates that the workload peaked at $417.83\text{ MB}$ and fit within the applied $1024\text{ MB}$ quota, but does not provide counterfactual proof that the container would have crashed without intervention.
6. **Production Container Orchestration**: Does not provide production-grade Kubernetes integration, multi-container pod scaling, horizontal autoscaling, or cross-node scheduling. It remains an experimental research prototype.
