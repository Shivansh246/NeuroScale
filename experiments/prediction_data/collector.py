import time
import docker
import json
import os

from monitoring.collector import Collector

class PilotCollector:
    def __init__(self, client=None):
        self._collector = Collector(client=client)
        self._last_cpu_usage_ns = None

    def collect(self, container_id, workload_mode):
        """Collects metrics and computes cpu_usage_ns_delta."""
        # We can use the underlying collector to get raw stats, but it doesn't return block_read_bytes easily.
        # Actually, the base Collector returns raw `cpu`, `memory`, `network`, `disk_io` in its dict!
        base_metrics = self._collector.collect(container_id)
        
        if base_metrics is None:
            return None
            
        timestamp = base_metrics["timestamp"]
        cpu_usage_ns = base_metrics["cpu_usage_ns"]
        
        if self._last_cpu_usage_ns is None:
            cpu_usage_ns_delta = 0
        else:
            cpu_usage_ns_delta = max(0, cpu_usage_ns - self._last_cpu_usage_ns)
            
        self._last_cpu_usage_ns = cpu_usage_ns

        # Networks
        networks = base_metrics.get("network", {})
        rx_bytes = sum(net.get("rx_bytes", 0) for net in networks.values())
        tx_bytes = sum(net.get("tx_bytes", 0) for net in networks.values())
        
        # Blkio
        disk_io = base_metrics.get("disk_io", {})
        io_service_bytes = disk_io.get("io_service_bytes_recursive", [])
        read_bytes = 0
        write_bytes = 0
        if io_service_bytes:
            for entry in io_service_bytes:
                op = entry.get("op", "").lower()
                val = entry.get("value", 0)
                if op == "read":
                    read_bytes += val
                elif op == "write":
                    write_bytes += val

        return {
            "timestamp": timestamp,
            "container_id": base_metrics["container_id"],
            "container_name": base_metrics["container_name"],
            "cpu_percent": base_metrics["cpu_percent"],
            "cpu_usage_ns": cpu_usage_ns,
            "cpu_usage_ns_delta": cpu_usage_ns_delta,
            "memory_usage_mb": base_metrics["memory_usage_mb"],
            "memory_percent": base_metrics["memory_percent"],
            "cpu_limit": base_metrics["cpu"].get("cpu_usage", {}).get("online_cpus", 1),
            "memory_limit": base_metrics["memory_limit_mb"],
            "network_rx_bytes": rx_bytes,
            "network_tx_bytes": tx_bytes,
            "block_read_bytes": read_bytes,
            "block_write_bytes": write_bytes,
            "workload_mode": workload_mode
        }
