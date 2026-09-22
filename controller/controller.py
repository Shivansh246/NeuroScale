import docker
from docker.errors import DockerException, NotFound, APIError


class Controller:
    """Safely applies dynamic CPU and memory resource allocations to Docker containers."""

    DEFAULT_MIN_CPU = 0.1
    DEFAULT_MAX_CPU = 16.0
    DEFAULT_MIN_MEMORY_MB = 64
    DEFAULT_MAX_MEMORY_MB = 32768  # 32 GB

    def __init__(
        self,
        client=None,
        min_cpu=DEFAULT_MIN_CPU,
        max_cpu=DEFAULT_MAX_CPU,
        min_memory_mb=DEFAULT_MIN_MEMORY_MB,
        max_memory_mb=DEFAULT_MAX_MEMORY_MB,
    ):
        self._client = client
        self.min_cpu = float(min_cpu)
        self.max_cpu = float(max_cpu)
        self.min_memory_mb = int(min_memory_mb)
        self.max_memory_mb = int(max_memory_mb)

    @property
    def client(self):
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def validate_inputs(self, container_id, cpu, memory):
        """Validate input parameters and enforce bounds."""
        if not container_id or not isinstance(container_id, str):
            return False, "container_id must be a non-empty string"

        if not isinstance(cpu, (int, float)) or isinstance(cpu, bool):
            return False, f"cpu must be a numeric value, got {type(cpu).__name__}"

        if not (self.min_cpu <= cpu <= self.max_cpu):
            return False, (
                f"cpu {cpu} out of bounds: must be between "
                f"{self.min_cpu} and {self.max_cpu} cores"
            )

        if not isinstance(memory, (int, float)) or isinstance(memory, bool):
            return False, f"memory must be a numeric value, got {type(memory).__name__}"

        if not (self.min_memory_mb <= memory <= self.max_memory_mb):
            return False, (
                f"memory {memory}MB out of bounds: must be between "
                f"{self.min_memory_mb}MB and {self.max_memory_mb}MB"
            )

        return True, ""

    def apply(self, container_id, cpu, memory):
        """Safely update container CPU quota and memory limits."""
        # 1. Input validation & sensible bounds check
        is_valid, err_msg = self.validate_inputs(container_id, cpu, memory)
        if not is_valid:
            return {
                "success": False,
                "container_id": container_id,
                "applied": {},
                "error": f"Validation error: {err_msg}",
            }

        cpu_val = float(cpu)
        mem_val = int(memory)

        cpu_period = 100000
        cpu_quota = int(cpu_val * cpu_period)

        # 2. Docker container lookup and resource update
        try:
            container = self.client.containers.get(container_id)

            update_kwargs = {}
            if cpu is not None:
                update_kwargs["cpu_period"] = 100000
                update_kwargs["cpu_quota"] = int(float(cpu) * 100000)
            if memory is not None:
                update_kwargs["mem_limit"] = f"{int(memory)}m"

            try:
                self.client.api.update_container(
                    container.id,
                    **update_kwargs,
                    memswap_limit=f"{int(memory)}m" if memory is not None else None,
                )
            except APIError as api_err:
                # Fallback without memswap_limit if swap accounting is not enabled on host
                if "swap" in str(api_err).lower():
                    self.client.api.update_container(container.id, **update_kwargs)
                else:
                    raise

            return {
                "success": True,
                "container_id": container.id,
                "applied": {
                    "cpu": cpu_val,
                    "memory": mem_val,
                    "cpu_period": cpu_period,
                    "cpu_quota": cpu_quota,
                    "mem_limit": f"{mem_val}m",
                },
                "error": None,
            }

        except NotFound:
            return {
                "success": False,
                "container_id": container_id,
                "applied": {},
                "error": f"Container not found: {container_id}",
            }
        except APIError as e:
            return {
                "success": False,
                "container_id": container_id,
                "applied": {},
                "error": f"Docker API error: {getattr(e, 'explanation', str(e))}",
            }
        except DockerException as e:
            return {
                "success": False,
                "container_id": container_id,
                "applied": {},
                "error": f"Docker daemon error: {str(e)}",
            }
        except Exception as e:
            return {
                "success": False,
                "container_id": container_id,
                "applied": {},
                "error": f"Unexpected error: {str(e)}",
            }
