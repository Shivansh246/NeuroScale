"""
Live Telemetry Engine for NeuroScale Dashboard (Phase 9.1 & 9.2).

Responsibilities:
- Automatically discovers running Docker containers via Docker SDK.
- Polls live telemetry every 1 second:
  - CPU %
  - CPU Usage (ns)
  - Memory %
  - Memory MB
  - Network RX (bytes)
  - Network TX (bytes)
  - Block Read (bytes)
  - Block Write (bytes)
  - PIDs
- Maintains a thread-safe, rolling in-memory ring buffer (default maxlen=300 samples per container).
- Computes cpu_usage_ns_delta via TelemetryFeaturePipeline.
- Integrates the trained AI pipeline (Phase 9.2):
  - TransformerPredictor: Demand prediction (CPU & Memory) and confidence
  - AutoencoderAnomalyDetector: Reconstruction score and adaptive fence anomaly decision
  - ResourceFactorizedDQNAgent: Dual-head resource allocation recommendation
  - Safety Guard: Enforces minimum allocations (CPU >= 0.25 cores, Memory >= 128 MB)
  - Produces recommendations ONLY (Never calls container.update() directly)
- Exposes clean APIs for dashboard consumption:
  - get_active_containers()
  - get_latest_metrics(container_id)
  - get_history_df(container_id=None, limit=120)
  - get_live_predictions()
  - get_live_container_states()
  - get_latest_ai_state(container_id)
  - start() / stop()
"""

import collections
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("neuroscale.live_engine")


class LiveTelemetryEngine:
    """Decoupled background engine that continuously polls live Docker containers and executes AI inference."""

    def __init__(
        self,
        poll_interval_seconds: float = 1.0,
        buffer_capacity: int = 300,
        filter_prefix: Optional[str] = None,
        checkpoints_dir: str = "checkpoints",
    ):
        self.poll_interval = float(poll_interval_seconds)
        self.buffer_capacity = int(buffer_capacity)
        self.filter_prefix = filter_prefix  # If set, filters container names (e.g. 'neuroscale')
        self.checkpoints_dir = checkpoints_dir

        # Thread-safe in-memory ring buffer: container_id -> deque of metric dicts
        self._buffers: Dict[str, collections.deque] = {}
        self._container_metadata: Dict[str, Dict[str, Any]] = {}
        
        # AI state cache: container_id -> latest AI state dict
        self._ai_states: Dict[str, Dict[str, Any]] = {}
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._docker_client = None
        self._is_running = False

        # AI Pipeline Components (Lazy loaded once)
        self._ai_loaded = False
        self._feature_columns = ["cpu_percent", "cpu_usage_ns_delta", "memory_usage_mb", "memory_percent"]
        self._seq_len = 12
        self._feature_pipelines: Dict[str, Any] = {}  # Per-container pipeline for delta calculation
        self._transformer = None
        self._autoencoder = None
        self._rl_agent = None
        self._state_builder = None

    # ----------------------------------------------------------------------
    # Docker Client Management
    # ----------------------------------------------------------------------

    def _get_client(self):
        """Lazily initialize and return the Docker client."""
        if self._docker_client is None:
            try:
                import docker
                self._docker_client = docker.from_env(timeout=3)
            except Exception as exc:
                logger.warning("Docker SDK client connection failed: %s", exc)
                return None
        return self._docker_client

    # ----------------------------------------------------------------------
    # AI Models Loading (Singleton pattern, loaded once)
    # ----------------------------------------------------------------------

    def _ensure_ai_pipeline(self) -> bool:
        """Load trained model checkpoints into memory once."""
        if self._ai_loaded:
            return True

        try:
            from models.predictor import TransformerPredictor
            from anomaly.detector import AutoencoderAnomalyDetector
            from rl.resource_factorized_agent import ResourceFactorizedDQNAgent
            from rl.state import StateBuilder
            from rl.config import StateConfig

            tf_path = os.path.join(self.checkpoints_dir, "best_transformer.pt")
            ae_path = os.path.join(self.checkpoints_dir, "best_autoencoder.pt")
            rl_path = os.path.join(self.checkpoints_dir, "best_resource_factorized_dqn.pt")

            if not (os.path.exists(tf_path) and os.path.exists(ae_path) and os.path.exists(rl_path)):
                logger.warning("One or more AI checkpoints missing in %s", self.checkpoints_dir)
                return False

            self._transformer = TransformerPredictor.from_checkpoint(tf_path)
            self._autoencoder = AutoencoderAnomalyDetector.from_checkpoint(ae_path)
            self._rl_agent = ResourceFactorizedDQNAgent.load(rl_path)
            self._state_builder = StateBuilder(StateConfig())

            self._ai_loaded = True
            logger.info("LiveTelemetryEngine AI pipeline loaded successfully.")
            return True
        except Exception as exc:
            logger.error("Failed to load AI pipeline: %s", exc)
            return False

    def _get_container_pipeline(self, container_id: str):
        """Get or create a TelemetryFeaturePipeline for delta computation."""
        from data.preprocessing import TelemetryFeaturePipeline
        if container_id not in self._feature_pipelines:
            self._feature_pipelines[container_id] = TelemetryFeaturePipeline(
                feature_columns=self._feature_columns
            )
        return self._feature_pipelines[container_id]

    # ----------------------------------------------------------------------
    # Container Discovery & Metric Extraction
    # ----------------------------------------------------------------------

    def discover_containers(self) -> List[Any]:
        """Discover currently running Docker containers."""
        client = self._get_client()
        if not client:
            return []
        try:
            all_containers = client.containers.list()
            if self.filter_prefix:
                return [c for c in all_containers if self.filter_prefix in c.name]
            return all_containers
        except Exception as exc:
            logger.debug("Container discovery failed: %s", exc)
            return []

    @staticmethod
    def _extract_cpu_percent(cpu_stats: Dict[str, Any], precpu_stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from container stats."""
        if not cpu_stats or not precpu_stats:
            return 0.0

        cpu_usage = cpu_stats.get("cpu_usage", {})
        precpu_usage = precpu_stats.get("cpu_usage", {})

        total_usage = cpu_usage.get("total_usage", 0)
        pre_total_usage = precpu_usage.get("total_usage", 0)
        cpu_delta = total_usage - pre_total_usage

        system_usage = cpu_stats.get("system_cpu_usage", 0)
        pre_system_usage = precpu_stats.get("system_cpu_usage", 0)
        system_delta = system_usage - pre_system_usage

        online_cpus = cpu_stats.get("online_cpus")
        if not online_cpus:
            percpu = cpu_usage.get("percpu_usage")
            online_cpus = len(percpu) if percpu else 1

        if cpu_delta > 0 and system_delta > 0:
            return round((cpu_delta / system_delta) * online_cpus * 100.0, 2)
        return 0.0

    @staticmethod
    def _extract_memory_stats(memory_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory usage, limit, and percentage."""
        if not memory_stats:
            return {
                "memory_usage_bytes": 0,
                "memory_usage_mb": 0.0,
                "memory_limit_bytes": 0,
                "memory_limit_mb": 0.0,
                "memory_percent": 0.0,
            }

        usage_bytes = memory_stats.get("usage", 0)
        limit_bytes = memory_stats.get("limit", 0)

        if limit_bytes > (1 << 62):  # Unlimited cgroup threshold
            limit_bytes = 0

        usage_mb = round(usage_bytes / (1024 * 1024), 2)
        limit_mb = round(limit_bytes / (1024 * 1024), 2) if limit_bytes > 0 else 0.0
        mem_percent = round((usage_bytes / limit_bytes) * 100.0, 2) if limit_bytes > 0 else 0.0

        return {
            "memory_usage_bytes": int(usage_bytes),
            "memory_usage_mb": float(usage_mb),
            "memory_limit_bytes": int(limit_bytes),
            "memory_limit_mb": float(limit_mb),
            "memory_percent": float(mem_percent),
        }

    @staticmethod
    def _aggregate_network_stats(networks: Dict[str, Any]) -> Dict[str, int]:
        """Aggregate rx_bytes and tx_bytes across network interfaces."""
        rx_total = 0
        tx_total = 0
        if isinstance(networks, dict):
            for iface in networks.values():
                if isinstance(iface, dict):
                    rx_total += iface.get("rx_bytes", 0)
                    tx_total += iface.get("tx_bytes", 0)
        return {"network_rx_bytes": int(rx_total), "network_tx_bytes": int(tx_total)}

    @staticmethod
    def _aggregate_blkio_stats(blkio_stats: Dict[str, Any]) -> Dict[str, int]:
        """Aggregate block I/O read and write bytes."""
        read_total = 0
        write_total = 0
        if isinstance(blkio_stats, dict):
            entries = blkio_stats.get("io_service_bytes_recursive", [])
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        op = str(entry.get("op", "")).lower()
                        val = int(entry.get("value", 0))
                        if op == "read":
                            read_total += val
                        elif op == "write":
                            write_total += val
        return {"block_read_bytes": int(read_total), "block_write_bytes": int(write_total)}

    def _poll_container(self, container: Any, timestamp: float) -> Optional[Dict[str, Any]]:
        """Fetch Docker stats for a single container and compute delta via pipeline."""
        try:
            stats = container.stats(stream=False)
        except Exception as exc:
            logger.debug("Failed to read stats for %s: %s", getattr(container, "name", str(container)), exc)
            return None

        if not isinstance(stats, dict) or not stats:
            return None

        cpu_stats = stats.get("cpu_stats", {}) or {}
        precpu_stats = stats.get("precpu_stats", {}) or {}
        memory_stats = stats.get("memory_stats", {}) or {}
        networks = stats.get("networks", {}) or {}
        blkio = stats.get("blkio_stats", {}) or {}
        pids_stats = stats.get("pids_stats", {}) or {}

        cpu_percent = self._extract_cpu_percent(cpu_stats, precpu_stats)
        raw_cpu_ns = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)
        mem_info = self._extract_memory_stats(memory_stats)
        net_info = self._aggregate_network_stats(networks)
        blk_info = self._aggregate_blkio_stats(blkio)
        pids = int(pids_stats.get("current", 1))

        cid = getattr(container, "id", "")
        cname = getattr(container, "name", str(cid)[:12])
        labels = getattr(container, "labels", {}) or {}
        mode = str(labels.get("mode", labels.get("neuroscale.mode", "unknown")))

        raw_record = {
            "timestamp": timestamp,
            "container_id": cid,
            "container_name": cname,
            "workload_mode": mode,
            "cpu_percent": float(cpu_percent),
            "cpu_usage_ns": int(raw_cpu_ns),
            "memory_usage_mb": float(mem_info["memory_usage_mb"]),
            "memory_usage_bytes": int(mem_info["memory_usage_bytes"]),
            "memory_limit_mb": float(mem_info["memory_limit_mb"]),
            "memory_limit_bytes": int(mem_info["memory_limit_bytes"]),
            "memory_percent": float(mem_info["memory_percent"]),
            "network_rx_bytes": int(net_info["network_rx_bytes"]),
            "network_tx_bytes": int(net_info["network_tx_bytes"]),
            "block_read_bytes": int(blk_info["block_read_bytes"]),
            "block_write_bytes": int(blk_info["block_write_bytes"]),
            "pids": pids,
        }

        # Compute delta via unified feature pipeline
        pipeline = self._get_container_pipeline(cid)
        delta, counter_reset = pipeline.compute_delta(raw_record, cid)
        raw_record["cpu_usage_ns_delta"] = float(delta)
        raw_record["counter_reset"] = counter_reset

        return raw_record

    # ----------------------------------------------------------------------
    # AI Inference & Safety Guard (Phase 9.2)
    # ----------------------------------------------------------------------

    def _run_ai_cycle(self, container_id: str, buffer: collections.deque) -> Optional[Dict[str, Any]]:
        """
        Run the complete inference pipeline on a container's rolling buffer:
        Raw Window -> FeaturePipeline -> Transformer -> Autoencoder -> Factorized DQN -> Safety Guard
        """
        if not self._ensure_ai_pipeline():
            return None

        if not buffer:
            return None

        pipeline = self._get_container_pipeline(container_id)
        # If buffer has fewer than seq_len samples, pad with available samples
        raw_list = list(buffer)
        if len(raw_list) < self._seq_len:
            pad_count = self._seq_len - len(raw_list)
            seq_records = [raw_list[0]] * pad_count + raw_list
        else:
            seq_records = raw_list[-self._seq_len:]

        feature_sequence = pipeline.extract_feature_sequence(seq_records, self._feature_columns)

        t_start = time.time()
        # 1. Transformer Predictor
        prediction = self._transformer.predict(feature_sequence)
        pred_cpu = float(prediction.get("cpu", 0.0))
        pred_mem = float(prediction.get("memory", 0.0))
        pred_confidence = float(prediction.get("confidence", 0.0))

        # 2. Autoencoder Anomaly Detector
        anomaly = self._autoencoder.score(feature_sequence)
        anomaly_score = float(anomaly.get("score", 0.0))
        is_anomaly = bool(anomaly.get("is_anomaly", False))

        # 3. RL State Vector Construction
        latest_rec = buffer[-1]
        cpu_util = float(latest_rec.get("cpu_percent", 0.0))
        mem_util = float(latest_rec.get("memory_usage_mb", 0.0))
        
        # Approximate SLA latency based on utilization thresholds
        sla_latency = 50.0
        if cpu_util > 90.0:
            sla_latency += 150.0
        if mem_util > 800.0:
            sla_latency += 300.0

        raw_state = {
            "current_cpu_util": cpu_util,
            "current_mem_util": mem_util,
            "predicted_cpu_demand": pred_cpu,
            "predicted_mem_demand": pred_mem,
            "current_cpu_alloc": 1.0,  # Default baseline allocation
            "current_mem_alloc": 256.0,
            "sla_latency": sla_latency,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "host_cpu_avail": 4.0,
            "host_mem_avail": 4096.0,
        }
        state_vec = self._state_builder.build_state(raw_state)

        # 4. Resource-Factorized DQN Action Selection
        action_dict = self._rl_agent.choose_action(state_vec)
        raw_rec_cpu = float(action_dict.get("cpu", 1.0))
        raw_rec_mem = int(action_dict.get("memory", 256))

        # 5. Safety Guard Layer
        # Rule: Never allow CPU < 0.25 cores, Memory < 128 MB
        safe_cpu = max(0.25, raw_rec_cpu)
        safe_mem = max(128, raw_rec_mem)

        inference_time_ms = round((time.time() - t_start) * 1000.0, 2)

        return {
            "container_id": container_id,
            "status": "ready",
            "samples_buffered": len(buffer),
            "inference_time_ms": inference_time_ms,
            "timestamp": time.time(),
            # Expose standard recommendation fields
            "recommended_cpu": safe_cpu,
            "recommended_memory": safe_mem,
            "prediction_confidence": pred_confidence,
            "predicted_cpu": pred_cpu,
            "predicted_memory": pred_mem,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "recommended_action": {
                "cpu": safe_cpu,
                "memory": safe_mem,
            },
        }

    # ----------------------------------------------------------------------
    # Background Polling Loop
    # ----------------------------------------------------------------------

    def _poll_and_update(self, c: Any, timestamp: float) -> Optional[str]:
        """Poll container metrics and run AI cycle in thread."""
        cid = getattr(c, "id", "")
        cname = getattr(c, "name", "")
        record = self._poll_container(c, timestamp=timestamp)
        if record:
            with self._lock:
                if cid not in self._buffers:
                    self._buffers[cid] = collections.deque(maxlen=self.buffer_capacity)
                self._buffers[cid].append(record)
                self._container_metadata[cid] = {
                    "name": cname,
                    "id": cid,
                    "status": getattr(c, "status", "running"),
                    "last_seen": timestamp,
                    "workload_mode": record["workload_mode"],
                }
                # Run AI inference on current window
                ai_state = self._run_ai_cycle(cid, self._buffers[cid])
                if ai_state:
                    self._ai_states[cid] = ai_state
            return cid
        return None

    def _loop(self) -> None:
        """Main 1-Hz background polling thread execution with concurrent container queries."""
        from concurrent.futures import ThreadPoolExecutor
        logger.info("LiveTelemetryEngine started background polling.")
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="NeuroScale-Poller") as pool:
            while not self._stop_event.is_set():
                t0 = time.time()
                active_containers = self.discover_containers()
                active_ids = {getattr(c, "id", "") for c in active_containers}

                if active_containers:
                    futures = [pool.submit(self._poll_and_update, c, t0) for c in active_containers]
                    for f in futures:
                        try:
                            f.result(timeout=2.5)
                        except Exception as exc:
                            logger.debug("Container poll task failed: %s", exc)

                # Clean up stale metadata for containers that have exited
                with self._lock:
                    stale_ids = [k for k in self._buffers if k not in active_ids]
                    for sid in stale_ids:
                        if sid in self._container_metadata:
                            self._container_metadata[sid]["status"] = "exited"

                # Drift compensation for steady 1-Hz cadence
                elapsed = time.time() - t0
                sleep_time = max(0.05, self.poll_interval - elapsed)
                time.sleep(sleep_time)

        logger.info("LiveTelemetryEngine stopped.")

    # ----------------------------------------------------------------------
    # Public Lifecycle Controls
    # ----------------------------------------------------------------------

    def start(self) -> None:
        """Start the background live polling engine."""
        if self._is_running:
            return
        self._stop_event.clear()
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._loop, name="NeuroScale-LiveTelemetryEngine", daemon=True
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the background polling engine cleanly."""
        if not self._is_running:
            return
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ----------------------------------------------------------------------
    # Public Query APIs for Dashboard Consumption
    # ----------------------------------------------------------------------

    def get_active_containers(self) -> List[Dict[str, Any]]:
        """Return list of discovered active containers with current metadata."""
        with self._lock:
            return [
                meta.copy()
                for meta in self._container_metadata.values()
                if meta.get("status") == "running"
            ]

    def get_latest_metrics(self, container_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the most recent telemetry sample."""
        with self._lock:
            if container_id and container_id in self._buffers and len(self._buffers[container_id]) > 0:
                return self._buffers[container_id][-1].copy()
            for buf in self._buffers.values():
                if buf:
                    return buf[-1].copy()
        return None

    def get_history_df(self, container_id: Optional[str] = None, limit: int = 120) -> pd.DataFrame:
        """Return recent telemetry history as a pandas DataFrame."""
        records = []
        with self._lock:
            if container_id and container_id in self._buffers:
                records = list(self._buffers[container_id])
            else:
                for buf in self._buffers.values():
                    records.extend(list(buf))

        if not records:
            return pd.DataFrame()

        records.sort(key=lambda r: r.get("timestamp", 0))
        if limit and len(records) > limit:
            records = records[-limit:]

        df = pd.DataFrame(records)
        if not df.empty and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def get_sample_count(self) -> int:
        """Return total samples buffered across all active containers."""
        with self._lock:
            return sum(len(buf) for buf in self._buffers.values())

    # ----------------------------------------------------------------------
    # Phase 9.2 New Exposed AI State APIs
    # ----------------------------------------------------------------------

    def get_live_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Return dictionary of container_id -> latest demand predictions."""
        with self._lock:
            out = {}
            for cid, state in self._ai_states.items():
                out[cid] = {
                    "predicted_cpu": state.get("predicted_cpu", 0.0),
                    "predicted_memory": state.get("predicted_memory", 0.0),
                    "confidence": state.get("prediction_confidence", 0.0),
                    "status": state.get("status", "unknown"),
                }
            return out

    def get_live_container_states(self) -> Dict[str, Dict[str, Any]]:
        """Return complete live states (metrics + AI inference + recommendations) for all containers."""
        with self._lock:
            out = {}
            for cid, meta in self._container_metadata.items():
                if meta.get("status") == "running":
                    metrics = self._buffers[cid][-1].copy() if cid in self._buffers and self._buffers[cid] else {}
                    ai_state = self._ai_states.get(cid, {}).copy()
                    out[cid] = {
                        "metadata": meta.copy(),
                        "metrics": metrics,
                        "ai": ai_state,
                    }
            return out

    def get_latest_ai_state(self, container_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the latest AI inference state and recommendations for a given container (or any active container)."""
        with self._lock:
            if container_id:
                state = self._ai_states.get(container_id)
                return state.copy() if state else None
            # Default to first available active container
            for state in self._ai_states.values():
                return state.copy()
        return None
