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
