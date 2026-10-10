"""High-resolution concurrent Docker container telemetry collector for NeuroScale."""

from typing import Dict, Any, List, Optional
import concurrent.futures
import logging
import threading
import time

try:
    import docker
    from docker.errors import DockerException, NotFound, APIError
except ImportError:
    docker = None
    DockerException = Exception
    NotFound = Exception
    APIError = Exception


logger = logging.getLogger("telemetry.collector")


class DockerTelemetryCollector:
    """Collects raw, un-preprocessed Docker container metrics concurrently at 1Hz resolution.
    
    Adheres strictly to the NeuroScale contract:
    - Raw cumulative cpu_usage_ns is preserved without computing deltas.
    - No log1p or normalization preprocessing occurs here.
    - TelemetryFeaturePipeline remains solely responsible for downstream delta & feature transforms.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        timeout_seconds: float = 5.0,
        max_workers: int = 16,
    ) -> None:
        self._client = client
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="DockerCollectorWorker",
        )

    @property
    def client(self) -> Any:
        """Lazily initialize Docker SDK client in a thread-safe manner."""
        with self._lock:
            if self._client is None:
                if docker is None:
                    raise RuntimeError("The 'docker' Python package is not installed.")
                self._client = docker.from_env(timeout=int(self.timeout_seconds))
            return self._client

    # ---------------------------------------------------------------------------
    # Statistical Extraction & Aggregation Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def compute_cpu_percent(cpu_stats: Dict[str, Any], precpu_stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from container cpu_stats and precpu_stats.
        
        Formula:
            (cpu_delta / system_delta) * online_cpus * 100.0
        Returns 0.0 if precpu_stats is empty or system delta <= 0.
        """
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
    def extract_memory_stats(memory_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory usage, memory limit, and percentage.
        
        Handles missing fields, cgroup v1/v2 differences, and zero limits safely.
        """
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

        # In Linux cgroups, an unlimited container can report limit near 2^63 - 1
        if limit_bytes > (1 << 62):
            limit_bytes = 0

        usage_mb = round(usage_bytes / (1024 * 1024), 2)
        limit_mb = round(limit_bytes / (1024 * 1024), 2) if limit_bytes > 0 else 0.0
        mem_percent = round((usage_bytes / limit_bytes) * 100.0, 2) if limit_bytes > 0 else 0.0

        return {
            "memory_usage_bytes": usage_bytes,
            "memory_usage_mb": usage_mb,
            "memory_limit_bytes": limit_bytes,
            "memory_limit_mb": limit_mb,
            "memory_percent": mem_percent,
        }

    @staticmethod
    def aggregate_network_stats(networks: Dict[str, Any]) -> Dict[str, int]:
        """Aggregate rx_bytes and tx_bytes across all network interfaces (eth0, eth1, etc.)."""
        rx_total = 0
        tx_total = 0

        if isinstance(networks, dict):
            for iface_stats in networks.values():
                if isinstance(iface_stats, dict):
                    rx_total += iface_stats.get("rx_bytes", 0)
                    tx_total += iface_stats.get("tx_bytes", 0)

        return {
            "network_rx_bytes": int(rx_total),
            "network_tx_bytes": int(tx_total),
        }

    @staticmethod
    def aggregate_blkio_stats(blkio_stats: Dict[str, Any]) -> Dict[str, int]:
        """Aggregate block I/O read and write bytes across all block devices."""
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

        return {
            "block_read_bytes": int(read_total),
            "block_write_bytes": int(write_total),
        }

    @staticmethod
    def extract_pid_stats(pids_stats: Dict[str, Any]) -> int:
        """Extract active thread and process count from pids_stats."""
        if isinstance(pids_stats, dict):
            return int(pids_stats.get("current", 0))
        return 0

    @staticmethod
    def _extract_workload_mode(container: Any) -> str:
        """Extract workload mode label or metadata from container without blocking."""
        try:
            labels = getattr(container, "labels", None) or {}
            if "neuroscale.mode" in labels:
                return str(labels["neuroscale.mode"])
            if "mode" in labels:
                return str(labels["mode"])
        except Exception:
            pass
        return "unknown"

    # ---------------------------------------------------------------------------
    # Core Single-Container Polling (1Hz execution)
    # ---------------------------------------------------------------------------

    def collect_container(
        self,
        container_or_id: Any,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Docker stats API for a single container and extract raw telemetry record.
        
        Returns None if container is dead, removed, or Docker daemon fails.
        """
        sample_time = timestamp or time.time()
        container = container_or_id

        # Resolve container object if ID was passed
        if isinstance(container_or_id, str):
            try:
                container = self.client.containers.get(container_or_id)
            except Exception as e:
                logger.warning("Failed to resolve container %s: %s", container_or_id, str(e))
                return None

        # Fetch non-streaming Docker statistics
        try:
            stats = container.stats(stream=False)
        except Exception as e:
            logger.warning("Error reading stats for container %s: %s", getattr(container, "id", str(container)), str(e))
            return None

        if not isinstance(stats, dict) or not stats:
            return None

        # Extract raw sub-dictionaries
        cpu_stats = stats.get("cpu_stats", {}) or {}
        precpu_stats = stats.get("precpu_stats", {}) or {}
        memory_stats = stats.get("memory_stats", {}) or {}
        networks = stats.get("networks", {}) or {}
        blkio = stats.get("blkio_stats", {}) or {}
        pids_stats = stats.get("pids_stats", {}) or {}

        # 1. CPU Metrics (Preserve RAW cpu_usage_ns)
        cpu_percent = self.compute_cpu_percent(cpu_stats, precpu_stats)
        raw_cpu_ns = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)

        # 2. Memory Metrics
        mem_info = self.extract_memory_stats(memory_stats)

        # 3. Network Metrics
        net_info = self.aggregate_network_stats(networks)

        # 4. Block I/O Metrics
        blk_info = self.aggregate_blkio_stats(blkio)

        # 5. Process / Thread Count
        pids = self.extract_pid_stats(pids_stats)

        # 6. Container Identity & Workload Mode
        cid = getattr(container, "id", str(container_or_id))
        cname = getattr(container, "name", str(cid)[:12])
        mode = self._extract_workload_mode(container)

        return {
            "timestamp": sample_time,
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
            "pids": int(pids),
        }

    # ---------------------------------------------------------------------------
    # Concurrent Multi-Container Polling
    # ---------------------------------------------------------------------------

    def collect_all(
        self,
        containers: List[Any],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Concurrently poll all containers within a single 1-second collection cycle.
        
        Executes parallel calls using a thread pool to minimize polling drift across containers.
        """
        if not containers:
            return []

        cycle_time = timestamp or time.time()
        futures = {
            self._executor.submit(self.collect_container, c, cycle_time): c
            for c in containers
        }

        results: List[Dict[str, Any]] = []
        for future in concurrent.futures.as_completed(futures, timeout=self.timeout_seconds + 0.5):
            try:
                rec = future.result()
                if rec is not None:
                    results.append(rec)
            except Exception as e:
                c = futures[future]
                logger.error("Exception in container collection worker for %s: %s", getattr(c, "id", str(c)), str(e))

        return results

    def close(self) -> None:
        """Shutdown thread pool executor and close client connection."""
        self._executor.shutdown(wait=False)
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
