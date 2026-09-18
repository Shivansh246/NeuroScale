# NeuroScale: Live Demonstration Guide & Presenter Manual

**Project**: NeuroScale: Dynamic Autonomous Vertical Pod Autoscaler  
**Audience**: Evaluators, Instructors, and Technical Presenters  

---

## 1. Quick-Start Execution Commands

All commands assume execution from the root of the repository: `/home/shv/Documents/OS/OS_Project/NeuroScaler`.

### Step 1: Activate the Python Environment
```bash
source .venv/bin/activate
```

### Step 2: Verify Docker Daemon & Workload Image
Confirm that the Docker daemon is active and the workload test image is built:
```bash
docker --version
docker images | grep neuroscale-workload
```
*(If the image is not present, build it once via: `docker build -t neuroscale-workload:latest -f docker/Dockerfile.workload docker/`)*

### Step 3: Verify Checkpoint Integrity
Ensure that all neural checkpoints are present and non-empty:
```bash
ls -lh checkpoints/*.pt
```
Expected checkpoints:
- `checkpoints/best_transformer.pt` ($\sim 412\text{ KB}$)
- `checkpoints/best_autoencoder.pt` ($\sim 24\text{ KB}$)
- `checkpoints/best_resource_factorized_dqn.pt` ($\sim 101\text{ KB}$)
- `checkpoints/best_dqn.pt` ($\sim 106\text{ KB}$)
- `checkpoints/best_decoupled_dqn.pt` ($\sim 101\text{ KB}$)

### Step 4: Run the Real Docker Closed-Loop Demonstration
Execute the automated 4-phase demonstration:
```bash
python scripts/demonstrate_real_docker.py
```
This script runs 42 cycles:
- **Cycles 1..11**: Rolling buffer warmup (idle container).
- **Cycles 12..14**: Normal active regime ($1.0\text{C} / 256\text{MB} \to 2.0\text{C} / 1024\text{MB}$).
- **Cycles 15..22**: CPU spike injection (`matrix multiplication`).
- **Cycles 23..30**: Memory spike injection (`bytearray` $400\text{ MB}$).
- **Cycles 31..42**: Recovery phase (quiescent).
- Outputs saved to: `data/real_docker_demo.jsonl`.

### Step 5: Launch the Streamlit Monitoring Dashboard
Launch the interactive dashboard to visualize cycles, cgroup allocations, and SLA latency:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`. In the left sidebar, select **"Real Docker Demo (Live cgroups)"** as the data source.

### Step 6: Run the Full Test Suite
Run all unit and integration tests to verify system integrity:
```bash
python -m unittest discover tests -v
```
Expected output: 172 passed, 2 skipped, 0 failures/errors.

---

## 2. Presenter Talking Points: What to Highlight During Demo

During the presentation or live review, walk through the key components using the table below:

| UI / Log Element | Presenter Talking Point | Technical Clarification |
| :--- | :--- | :--- |
| **Docker Container** | "NeuroScale manages an actual Linux container running inside Docker Engine, not a simulated software model." | Container uses cgroups v1/v2 CFS quotas and memory limits. |
| **Real Telemetry** | "Every second, `Collector` polls instantaneous CPU % and RSS memory MB directly from the container cgroup counters." | Polling duration takes $\sim 1004.8\text{ ms}$ due to Docker `stats()` delta timing. |
| **Transformer Prediction** | "The Transformer evaluates a 12-step rolling window and projects next-step resource demand in $\sim 1.2\text{ ms}$." | Shows good trend tracking on CPU surges; note as a prototype limitation that memory prediction underpredicts sharp spikes. |
| **Autoencoder Anomaly Signal** | "The Autoencoder reconstructs the feature window. When reconstruction error spikes, `is_anomaly=True` is asserted." | Point out that during idle, quiescent zero-load triggers a false positive ($0.0814$) because training data was centered at $40\%$ CPU. |
| **Factorized DQN Decision** | "The Resource-Factorized DQN splits decisions into two heads: a 4-action CPU head and a 4-action Memory head." | Both heads select Branch 3 ($2.0\text{C} / 1024\text{MB}$, Action 15) during stress regimes to guarantee QoS. |
| **Actual Docker HostConfig (Before/After)** | "Notice the verified Docker HostConfig printout: `CpuQuota` changed from 100,000 to 200,000, and `Memory` changed from 256MB to 1024MB." | Demonstrates physical Linux kernel quota enforcement via Docker API `POST /update`. |
| **SLA Latency** | "Response latency was monitored throughout; zero SLA violations ($>200\text{ ms}$) occurred across all test cycles." | Container maintained performance without throttling. |
| **Controller Execution Overhead** | "Applying the cgroup update to Docker Engine took only **$\sim 11.4\text{ ms}$**, and model inference took **$\sim 2.8\text{ ms}$**." | Pure controller overhead is well within real-time requirements. |

---

## 3. Managing Audience Expectations: Honest Disclosures

To ensure rigorous scientific integrity, **do not make the following claims**:
1. **Do NOT claim that the live demo demonstrates dynamic downscaling**:  
   Across the 42-cycle demo and dedicated 38-cycle recovery test, **zero downscale events occurred**. Rolling-window history lag and persistent anomaly flags keep the system at Action 15 throughout recovery.
2. **Do NOT claim perfect memory forecasting**:  
   The Transformer achieved high accuracy on CPU trends, but underpredicted memory spikes ($27.55\text{ MB}$ predicted vs $316.12\text{ MB}$ actual).
3. **Do NOT claim independent dimension scaling in live Docker**:  
   The factorized policy operates in Class B co-scaling mode under anomaly conditions, escalating both heads together rather than adjusting CPU without memory.
4. **Do NOT claim guaranteed OOM prevention**:  
   The workload peaked at $417.83\text{ MB}$ and remained safely below the applied $1024\text{ MB}$ limit; this proves safe headroom was provided, but does not prove the container would have terminated on a different configuration.
