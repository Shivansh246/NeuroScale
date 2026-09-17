import time
import docker
from docker.errors import DockerException, NotFound, APIError


class Collector:
    """Collects and calculates structured resource and workload metrics from Docker containers."""

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    @staticmethod
    def _calculate_cpu_percent(cpu_stats, precpu_stats):
        """Calculate CPU usage percentage from container cpu_stats and precpu_stats."""
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
    def _extract_workload_mode(container):
        """Extract workload mode from container metadata (Cmd, Labels, or Env)."""
        # Check container labels
        labels = getattr(container, "labels", None) or {}
        if "neuroscale.mode" in labels:
            return labels["neuroscale.mode"]
        if "mode" in labels:
            return labels["mode"]

        # Check container Config
        attrs = getattr(container, "attrs", None) or {}
        config = attrs.get("Config", {}) or {}

        cmd = config.get("Cmd") or []
        if isinstance(cmd, list):
            for i, arg in enumerate(cmd):
                if arg == "--mode" and i + 1 < len(cmd):
                    return cmd[i + 1]
                if isinstance(arg, str) and arg.startswith("--mode="):
                    return arg.split("=", 1)[1]

        env_list = config.get("Env") or []
        if isinstance(env_list, list):
            for env in env_list:
                if env.startswith("WORKLOAD_MODE="):
                    return env.split("=", 1)[1]
                if env.startswith("MODE="):
                    return env.split("=", 1)[1]

        return None

    def collect(self, container_id):
        """Collect and return timestamped structured metrics for the given container."""
        container = self.client.containers.get(container_id)
        stats = container.stats(stream=False)

        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})
        memory_stats = stats.get("memory_stats", {})

        cpu_percent = self._calculate_cpu_percent(cpu_stats, precpu_stats)
        cpu_usage_ns = cpu_stats.get("cpu_usage", {}).get("total_usage", 0)

        mem_usage_bytes = memory_stats.get("usage", 0)
        mem_limit_bytes = memory_stats.get("limit", 0)

        mem_usage_mb = round(mem_usage_bytes / (1024 * 1024), 2)
        mem_limit_mb = round(mem_limit_bytes / (1024 * 1024), 2) if mem_limit_bytes else 0.0
        mem_percent = round((mem_usage_bytes / mem_limit_bytes) * 100.0, 2) if mem_limit_bytes > 0 else 0.0

        workload_mode = self._extract_workload_mode(container)

        timestamp = time.time()

        return {
            "timestamp": timestamp,
            "container_id": getattr(container, "id", container_id),
            "container_name": getattr(container, "name", str(container_id)),
            "cpu_percent": cpu_percent,
            "cpu_usage_ns": cpu_usage_ns,
            "memory_usage_bytes": mem_usage_bytes,
            "memory_usage_mb": mem_usage_mb,
            "memory_limit_bytes": mem_limit_bytes,
            "memory_limit_mb": mem_limit_mb,
            "memory_percent": mem_percent,
            "workload_mode": workload_mode,
            "cpu": cpu_stats,
            "memory": memory_stats,
            "network": stats.get("networks", {}),
            "disk_io": stats.get("blkio_stats", {}),
        }
