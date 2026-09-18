# NeuroScale: Experimental Progression & Comparative Summary

**Project**: NeuroScale: Dynamic Autonomous Vertical Pod Autoscaler  
**Scope**: Progression from Heuristic Baselines to RL Architectures and Real Docker Integration  

---

## 1. Experimental Progression Overview

The NeuroScale research evaluated a sequence of autoscaling paradigms to investigate whether learned forecasting, anomaly perception, and reinforcement learning could balance SLA guarantees with resource efficiency:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ A. Static (1C)  │  ──>  │  B. Threshold   │  ──>  │ C. Pred-Only    │
│  Fixed 256 MB   │       │   Reactive Rule │       │  Transformer    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ G. Real Docker  │  <──  │ F. Factorized   │  <──  │ D. Monolithic   │
│   Closed Loop   │       │   DQN (Dual)    │       │   DQN (16 act)  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                   ▲
                                   │
                          ┌─────────────────┐
                          │  E. Decoupled   │
                          │   DQN (Indep.)  │
                          └─────────────────┘
```

---

## 2. Experimental Stages & Evaluation Results

All synthetic baseline and RL evaluations were executed across 4 standardized 50-step scenarios (`normal`, `cpu_spike`, `mem_spike`, `sustained`) recorded in `data/evaluation_results.jsonl`:

### A. Static Baseline (1.0 Core / 256 MB)
- **Policy**: Maintains constant allocation regardless of telemetry.
- **Normal**: Mean CPU $1.0\text{C}$, Mean Mem $256\text{MB}$, 0 SLA violations, Mean Reward $-0.848$.
- **CPU Spike**: 15 SLA violations, Mean Reward $-7.710$.
- **Memory Spike**: 15 SLA violations, Mean Reward $-7.815$.
- **Sustained**: 35 SLA violations, Mean Reward $-15.655$.
- **Limitation**: Inflexible; incurs heavy SLA penalties during spikes and wastes resources during low load.

### B. Reactive Threshold Baseline
- **Policy**: Scales memory to $512\text{MB}$ or $1024\text{MB}$ when utilization exceeds high watermarks ($80\%$).
- **Normal**: Mean CPU $1.0\text{C}$, Mean Mem $256\text{MB}$, 0 SLA violations, Mean Reward $-0.848$.
- **CPU Spike**: 15 SLA violations (fixed CPU limit at $1.0\text{C}$ cannot mitigate CPU spikes), Mean Reward $-7.710$.
- **Memory Spike**: Scaled memory to $481.3\text{MB}$, reducing SLA violations to 1, Mean Reward $-2.405$.
- **Sustained**: 35 SLA violations, Mean Reward $-12.855$.
- **Limitation**: Reactive rules trigger after resource exhaustion has already occurred; lack CPU elasticity in original baseline script.

### C. Prediction-Only Baseline (Transformer)
- **Policy**: Scales proportionally to next-horizon predictions from `checkpoints/best_transformer.pt`.
- **Normal**: Mean CPU $0.50\text{C}$, Mean Mem $128\text{MB}$, 13 SLA violations, Mean Reward $-3.159$.
- **CPU Spike**: 15 SLA violations, Mean Reward $-7.170$.
- **Memory Spike**: 15 SLA violations, Mean Reward $-7.235$.
- **Sustained**: 35 SLA violations, Mean Reward $-15.450$.
- **Limitation**: Highly aggressive downscaling at low load led to premature SLA breaches; lacks risk-aware reward objective.

### D. Monolithic Deep Q-Network (`checkpoints/best_dqn.pt`)
- **Architecture**: Single network ($11 \to 64 \to 64 \to 16$), mapping state to 16 joint actions ($4\text{ CPU} \times 4\text{ Mem}$).
- **Normal**: Mean CPU $0.25\text{C}$, Mean Mem $256\text{MB}$, 0 SLA violations, Mean Reward $-0.26$.
- **CPU Spike**: Mean CPU $1.16\text{C}$, Mean Mem $655\text{MB}$, 0 SLA violations, Mean Reward $-2.51$.
- **Memory Spike**: Mean CPU $1.16\text{C}$, Mean Mem $655\text{MB}$, 0 SLA violations, Mean Reward $-2.56$.
- **Sustained**: Mean CPU $1.48\text{C}$, Mean Mem $794\text{MB}$, 0 SLA violations, Mean Reward $-2.54$.
- **Action Profile**: Bimodal distribution across 200 evaluation steps: **$56.5\%$ Action 1** ($0.25\text{C}/256\text{MB}$) and **$43.5\%$ Action 15** ($2.0\text{C}/1024\text{MB}$).
- **Limitation**: Combinatorial coupling; cannot selectively scale one resource dimension without co-scaling the other.

### E. Decoupled DQN (`checkpoints/best_decoupled_dqn.pt`)
- **Architecture**: Two completely independent neural networks (one for CPU actions, one for Memory actions).
- **Normal**: 0 SLA violations, Mean Reward $-0.13$.
- **CPU Spike**: Mean CPU $0.33\text{C}$, Mean Mem $334\text{MB}$, 15 SLA violations, Mean Reward $-11.08$.
- **Memory Spike**: Mean CPU $0.79\text{C}$, Mean Mem $491\text{MB}$, 1 SLA violation, Mean Reward $-11.27$.
- **Sustained**: Mean CPU $1.46\text{C}$, Mean Mem $789\text{MB}$, 1 SLA violation, Mean Reward $-6.66$.
- **Limitation**: Complete decoupling removed cross-resource state correlation; CPU agent under-allocated during CPU spikes.

### F. Resource-Factorized DQN (`checkpoints/best_resource_factorized_dqn.pt`)
- **Architecture**: Shared feature trunk ($11 \to 64 \to 64$) with dual Q-heads ($64 \to 4\text{ CPU}$, $64 \to 4\text{ Memory}$), trained with factorized TD targets ($y_{cpu}, y_{mem}$) and separate Huber losses.
- **Normal**: Mean CPU $0.50\text{C}$, Mean Mem $256\text{MB}$, 0 SLA violations, Mean Reward $-0.23$.
- **CPU Spike**: Mean CPU $0.74\text{C}$, Mean Mem $418\text{MB}$, 8 SLA violations, Mean Reward $-6.30$.
- **Memory Spike**: Mean CPU $0.74\text{C}$, Mean Mem $418\text{MB}$, 8 SLA violations, Mean Reward $-48.08$.
- **Sustained**: Mean CPU $1.02\text{C}$, Mean Mem $611\text{MB}$, 18 SLA violations, Mean Reward $-16.09$.
- **Action Profile**: Shifted allocation density to intermediate configurations: **$67.3\%$ Action 5** ($0.5\text{C}/256\text{MB}$), **$15.8\%$ Action 6** ($0.5\text{C}/512\text{MB}$), and **$16.8\%$ Action 15** ($2.0\text{C}/1024\text{MB}$).

### G. Real Docker Integration (`data/real_docker_demo.jsonl`)
- **Environment**: Live Linux container (`neuroscale-workload:latest`) managed via Docker Engine API.
- **Components**: Real Transformer + Autoencoder + Factorized DQN.
- **Result**: 42 cycles executed; physical limits updated to $2.0\text{C} / 1024\text{MB}$ (Action 15); verified via Docker `HostConfig`.

---

## 3. Accepted Interpretation of RL Architecture Results

### Finding: Partial Specialization (Class B Behavior)
The comparative evaluations establish the following scientific conclusion:

> **Accepted Empirical Interpretation**:  
> The Resource-Factorized DQN architecture successfully breaks the rigid bimodal $\{A1, A15\}$ switching seen in the monolithic baseline by populating intermediate action branches ($A5$ and $A6$ account for $83.1\%$ of active steps). Under controlled single-metric counterfactual tests, the CPU head can scale ($0.5\text{C} \to 1.0\text{C}$) while keeping memory fixed ($512\text{MB}$).  
> 
> However, in dynamic multi-step workload scenarios and live Docker integration, **Resource-Factorized DQN provides partial specialization (Class B behavior) rather than clean independent dimension scaling**. When anomaly flags or multi-feature stressors trigger, the shared trunk routes both heads to escalate together (Action 15).

### Comparative Summary Table:

| Architecture | Action Space Formulation | Key Advantage | Principal Empirical Limitation | Accepted Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Monolithic DQN** | Joint Cartesian ($16$ actions) | Zero SLA violations across all test scenarios | Rigid bimodal switching ($A1$ vs $A15$); high resource over-provisioning | Coupled Baseline |
| **Decoupled DQN** | Independent Networks ($2 \times 4$) | Lowest wastage in calm periods | Loss of joint state context causes severe CPU under-allocation | Uncoordinated Decoupling |
| **Resource-Factorized DQN** | Shared Trunk + Dual Heads | Smoother intermediate provisioning ($A5, A6$); factorized TD loss | Co-scales both dimensions under stress; higher SLA violations during sudden spikes | Class B Partial Specialization |
