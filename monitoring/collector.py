import time
import docker


class Collector:
    def __init__(self):
        self.client = docker.from_env()

    def collect(self, container_id):
        container = self.client.containers.get(container_id)
        stats = container.stats(stream=False)

        return {
            "timestamp": time.time(),
            "container_id": container_id,
            "cpu": stats["cpu_stats"],
            "memory": stats["memory_stats"],
            "network": stats.get("networks", {}),
            "disk_io": stats.get("blkio_stats", {}),
        }
