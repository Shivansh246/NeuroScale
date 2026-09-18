# NeuroScale: Presentation Metrics & Slide-Ready Tables

**Project**: NeuroScale: Dynamic Autonomous Vertical Pod Autoscaler  
**Format**: Compact Markdown Tables Formatted for Presentation Slides & Reports  

---

## Slide Table A: System Timing Breakdown

| Operation Phase | Target Subsystem | Latency (ms) | Operational Details |
| :--- | :--- | :---: | :--- |
| **Telemetry Polling** | `Collector.collect()` | **$1004.8$** | Docker `stats()` internal 1.0s sleep for cgroups CPU delta |
| **Model Forward Pass** | Transformer + Autoencoder + DQN | **$2.8$** | PyTorch sequential CPU inference ($1.2\text{ms} + 0.4\text{ms} + 0.9\text{ms}$) |
| **Actuation API** | `Controller.apply()` | **$11.4$** | Docker Engine API call (`POST /containers/{id}/update`) |
| **State Verification** | `container.reload()` | **$2.5$** | Container `HostConfig` inspect JSON query |
| **Total Cycle Execution** | End-to-end active cycle | **$1021.5$** | Net execution time per control step |
| **Loop Sampling Sleep** | `time.sleep(1.0)` | **$1000.0$** | Configured loop interval between cycles |

---

## Slide Table B: Real Docker Demonstration Overview

| Demonstration Metric | Value | Empirical Status |
| :--- | :---: | :--- |
| **Total Executed Cycles** | **42** | Uninterrupted execution against live container |
| **Buffer Warmup Cycles** | **11** | FIFO buffer pre-filling (idle state) |
| **Active Autonomous Cycles** | **31** | Full neural inference & actuation active |
| **Initial Quota Allocation** | **1.0C / 256 MB** | Base cgroup limit at startup |
| **Escalated Allocation** | **2.0C / 1024 MB** | Quota applied at Cycle 12 (Action 15) |
| **Docker HostConfig Verified** | **100% (31/31)** | Confirmed via engine API inspection |
| **SLA Violations Observed** | **0 (0.0%)** | All latency measurements $< 200\text{ ms}$ threshold |
| **Container Crashes / OOMs** | **0** | Zero CFS throttling or kernel OOM kill events |

---

## Slide Table C: Live Transformer Forecasting Accuracy

| Workload Phase | Active Steps | Actual CPU | Pred CPU | CPU MAE | Actual RAM | Pred RAM | RAM MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Idle / Normal** | 3 | $0.01\%$ | $22.01\%$ | **$22.00\%$** | $10.9\text{ MB}$ | $12.7\text{ MB}$ | **$1.8\text{ MB}$** |
| **CPU Spike** | 8 | $74.34\%$ | $71.20\%$ | **$40.29\%$** | $15.1\text{ MB}$ | $20.4\text{ MB}$ | **$6.6\text{ MB}$** |
| **Memory Spike** | 8 | $2.05\%$ | $75.03\%$ | **$72.98\%$** | $316.1\text{ MB}$ | $27.5\text{ MB}$ | **$296.9\text{ MB}$** |
| **Recovery** | 12 | $0.01\%$ | $75.16\%$ | **$75.15\%$** | $11.0\text{ MB}$ | $27.4\text{ MB}$ | **$16.4\text{ MB}$** |

*Key Takeaway: Transformer effectively tracks CPU load surge direction, but underpredicts memory spikes and suffers from cross-metric correlation bias.*

---

## Slide Table D: Claim Status & Audit Verdicts

| # | System Capability Claim | Audit Status | Grounded Finding |
| :-: | :--- | :---: | :--- |
| **1** | Real Docker closed-loop control | **VERIFIED** | 42 cycles executed; cgroup limits dynamically updated |
| **2** | Real neural models integrated without mocks | **VERIFIED** | Transformer, Autoencoder, and Factorized DQN run live |
| **3** | Low-latency controller operation | **VERIFIED** | Docker API update takes $\sim 11.4\text{ ms}$; inference takes $\sim 2.8\text{ ms}$ |
| **4** | OS kernel cgroups physically updated | **VERIFIED** | Verified via `HostConfig.CpuQuota` & `HostConfig.Memory` |
| **5** | CPU workload trend forecasting | **VERIFIED** | Predicted mean $71.2\%$ vs actual mean $74.3\%$ in CPU spike |
| **6** | Zero SLA response violations | **VERIFIED** | Latency monitored at $50.0\text{ ms}$ ($< 200.0\text{ ms}$ SLA limit) |
| **7** | Dynamic autonomous downscaling | **NOT OBSERVED** | Zero downscales across 20 recovery cycles |
| **8** | Independent CPU/memory dimension scaling | **PARTIALLY VERIFIED** | Class B co-scaling under stress/anomaly regimes |
| **9** | Accurate memory demand forecasting | **POOR ACCURACY** | Underpredicted spikes ($27.5\text{ MB}$ pred vs $316.1\text{ MB}$ actual) |
| **10** | Guaranteed OOM crash prevention | **SCOPED HEADROOM** | $417.8\text{ MB}$ load fit in $1024\text{ MB}$ limit; crash not proved |

---

## Slide Table E: Core Prototype Limitations

| Limitation Area | Root Cause | Observable System Impact |
| :--- | :--- | :--- |
| **Quiescent Anomaly Alert** | Training data mean: $42.9\%$ CPU. Idle is $-1.98\sigma$ below training. | False-positive (`is_anomaly=True`, score $0.0814$) during idle. |
| **Persistent Escalation** | Real `cpu_usage_ns` is cumulative; saturated synthetic normalizers. | Anomaly score stays $>400$, preventing recovery downscaling. |
| **Rolling Window Lag** | 12-step FIFO sequence buffer retains historical spike frames. | Transformer predicts high demand for 12 cycles post-stress. |
| **Coupled Co-Scaling** | Shared trunk routes both heads to Branch 3 under anomaly flags. | System selects Action 15 ($2.0\text{C} / 1024\text{MB}$) across both heads. |
| **Research Prototype Scope** | Designed for single-container cgroups research evaluation. | Lacks Kubernetes pod orchestration and horizontal autoscaling. |
