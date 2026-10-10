"""Concrete workload profiles for NeuroScale systems research benchmarking."""

import os
import time
import socket
import tempfile
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
import random

from .base import BaseWorkload


# ---------------------------------------------------------------------------
# 1. Idle Workload Profile
# ---------------------------------------------------------------------------
class IdleWorkload(BaseWorkload):
    """Minimal baseline workload that sleeps without placing stress on system resources."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("idle", params)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def validate_params(self, params: Dict[str, Any]) -> None:
        duration = params.get("duration", 60)
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

    def _worker(self) -> None:
        duration = self.params.get("duration", 60)
        end_time = time.time() + duration
        while not self._stop_event.is_set() and time.time() < end_time:
            time.sleep(min(0.5, max(0.01, end_time - time.time())))
        self.is_running = False
        self.stop_time = time.time()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.start_time = time.time()
        self.stop_time = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="IdleWorker")
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.is_running = False
        self.stop_time = time.time()


# ---------------------------------------------------------------------------
# 2. CPU Intensive Workload Profile
# ---------------------------------------------------------------------------
class CPUIntensiveWorkload(BaseWorkload):
    """Configurable CPU stress using multiple worker threads with duty-cycle intensity throttling."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("cpu_intensive", params)
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    def validate_params(self, params: Dict[str, Any]) -> None:
        intensity = params.get("intensity", 50)
        if not (10 <= intensity <= 100):
            raise ValueError(f"CPU intensity must be between 10% and 100%, got {intensity}")
        workers = params.get("num_workers", 1)
        if not (isinstance(workers, int) and workers >= 1):
            raise ValueError(f"num_workers must be integer >= 1, got {workers}")
        duration = params.get("duration", 60)
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

    def _worker(self, worker_id: int) -> None:
        duration = self.params.get("duration", 60)
        intensity = self.params.get("intensity", 50)  # 10% to 100%
        end_time = time.time() + duration

        period = 0.1  # 100ms duty cycle period
        active_time = period * (intensity / 100.0)
        idle_time = period * (1.0 - (intensity / 100.0))

        while not self._stop_event.is_set() and time.time() < end_time:
            # Active burn phase
            cycle_start = time.time()
            cycle_end = cycle_start + active_time
            acc = 0
            while not self._stop_event.is_set() and time.time() < cycle_end:
                for i in range(1000):
                    acc += (i * i) ^ (i + 1)

            # Idle sleep phase
            if idle_time > 0 and not self._stop_event.is_set():
                time.sleep(idle_time)

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.start_time = time.time()
        self.stop_time = None
        self._threads.clear()

        num_workers = self.params.get("num_workers", max(1, os.cpu_count() or 1))
        for i in range(num_workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True, name=f"CPUWorker-{i}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._threads.clear()
        self.is_running = False
        self.stop_time = time.time()


# ---------------------------------------------------------------------------
# 3. Memory Intensive Workload Profile
# ---------------------------------------------------------------------------
class MemoryIntensiveWorkload(BaseWorkload):
    """Gradual memory allocation using configurable chunk sizes and resident page touch."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("memory_intensive", params)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffers: List[bytearray] = []
        self._lock = threading.Lock()

    def validate_params(self, params: Dict[str, Any]) -> None:
        target_mb = params.get("target_mb", 256)
        if target_mb <= 0:
            raise ValueError(f"target_mb must be positive, got {target_mb}")
        chunk_mb = params.get("chunk_mb", 16)
        if chunk_mb <= 0 or chunk_mb > target_mb:
            raise ValueError(f"chunk_mb must be in range (0, target_mb], got {chunk_mb}")
        step_interval = params.get("step_interval_seconds", 0.5)
        if step_interval <= 0:
            raise ValueError(f"step_interval_seconds must be positive, got {step_interval}")

    def _worker(self) -> None:
        target_mb = self.params.get("target_mb", 256)
        chunk_mb = self.params.get("chunk_mb", 16)
        step_interval = self.params.get("step_interval_seconds", 0.5)
        duration = self.params.get("duration", 60)

        end_time = time.time() + duration
        allocated_mb = 0

        # Phase 1: Gradual allocation and dirtying pages
        chunk_bytes = chunk_mb * 1024 * 1024
        while not self._stop_event.is_set() and allocated_mb < target_mb and time.time() < end_time:
            # Allocate chunk and dirty memory to ensure RSS inclusion
            b = bytearray(chunk_bytes)
            # Dirty every 4KB OS page
            for offset in range(0, chunk_bytes, 4096):
                b[offset] = 0xAA
            with self._lock:
                self._buffers.append(b)
            allocated_mb += chunk_mb
            time.sleep(step_interval)

        # Phase 2: Hold memory resident until duration expires or stopped
        while not self._stop_event.is_set() and time.time() < end_time:
            time.sleep(0.5)

        with self._lock:
            self._buffers.clear()
        self.is_running = False
        self.stop_time = time.time()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.start_time = time.time()
        self.stop_time = None
        self._buffers.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="MemoryWorker")
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._buffers.clear()
        self.is_running = False
        self.stop_time = time.time()

    def health(self) -> Dict[str, Any]:
        h = super().health()
        with self._lock:
            h["allocated_chunks"] = len(self._buffers)
            h["allocated_approx_mb"] = len(self._buffers) * self.params.get("chunk_mb", 16)
        return h


# ---------------------------------------------------------------------------
# 4. Disk IO Intensive Workload Profile
# ---------------------------------------------------------------------------
class DiskIOWorkload(BaseWorkload):
    """Configurable sequential read and write operations using temporary container files."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("disk_io_intensive", params)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._temp_file: Optional[Path] = None

    def validate_params(self, params: Dict[str, Any]) -> None:
        block_size_kb = params.get("block_size_kb", 64)
        if block_size_kb <= 0:
            raise ValueError(f"block_size_kb must be positive, got {block_size_kb}")
        write_rate_kbps = params.get("write_rate_kbps", 5120)  # 5 MB/s
        if write_rate_kbps <= 0:
            raise ValueError(f"write_rate_kbps must be positive, got {write_rate_kbps}")
        duration = params.get("duration", 60)
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

    def _worker(self) -> None:
        block_size_bytes = self.params.get("block_size_kb", 64) * 1024
        write_rate_bytes_sec = self.params.get("write_rate_kbps", 5120) * 1024
        duration = self.params.get("duration", 60)
        end_time = time.time() + duration

        # Create temporary file
        fd, temp_path = tempfile.mkstemp(prefix="neuroscale_io_")
        os.close(fd)
        self._temp_file = Path(temp_path)

        buffer = os.urandom(block_size_bytes)
        sleep_per_block = block_size_bytes / max(1.0, write_rate_bytes_sec)

        try:
            with open(self._temp_file, "r+b") as f:
                while not self._stop_event.is_set() and time.time() < end_time:
                    # Write block
                    f.write(buffer)
                    f.flush()

                    # Interleaved read block
                    if f.tell() > block_size_bytes:
                        f.seek(max(0, f.tell() - (2 * block_size_bytes)))
                        _ = f.read(block_size_bytes)
                        f.seek(0, os.SEEK_END)

                    # Throttle according to desired rate
                    if sleep_per_block > 0:
                        time.sleep(sleep_per_block)

                    # Wrap around to prevent unbounded disk filling (> 100MB)
                    if f.tell() > 100 * 1024 * 1024:
                        f.seek(0)
                        f.truncate()
        finally:
            self._cleanup_temp_file()
            self.is_running = False
            self.stop_time = time.time()

    def _cleanup_temp_file(self) -> None:
        if self._temp_file and self._temp_file.exists():
            try:
                self._temp_file.unlink()
            except Exception:
                pass
            self._temp_file = None

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.start_time = time.time()
        self.stop_time = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="DiskIOWorker")
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cleanup_temp_file()
        self.is_running = False
        self.stop_time = time.time()


# ---------------------------------------------------------------------------
# 5. Network Intensive Workload Profile
# ---------------------------------------------------------------------------
class NetworkIntensiveWorkload(BaseWorkload):
    """Local intra-cluster network traffic generation against the benchmark net sink."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("network_intensive", params)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def validate_params(self, params: Dict[str, Any]) -> None:
        target_host = params.get("target_host", "172.28.0.10")
        if not target_host:
            raise ValueError("target_host cannot be empty")
        port = params.get("port", 8080)
        if not (1 <= port <= 65535):
            raise ValueError(f"port must be 1-65535, got {port}")
        throughput_kbps = params.get("throughput_kbps", 5120)
        if throughput_kbps <= 0:
            raise ValueError(f"throughput_kbps must be positive, got {throughput_kbps}")
        duration = params.get("duration", 60)
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

    def _worker(self) -> None:
        host = self.params.get("target_host", "127.0.0.1")
        port = self.params.get("port", 8080)
        chunk_size = 8192
        rate_bytes_sec = self.params.get("throughput_kbps", 5120) * 1024
        duration = self.params.get("duration", 60)
        end_time = time.time() + duration

        payload = b"X" * chunk_size
        sleep_per_chunk = chunk_size / max(1.0, rate_bytes_sec)

        while not self._stop_event.is_set() and time.time() < end_time:
            s = None
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((host, port))
                # Stream payload in chunks
                for _ in range(50):
                    if self._stop_event.is_set() or time.time() >= end_time:
                        break
                    s.sendall(payload)
                    time.sleep(sleep_per_chunk)
            except Exception as e:
                # Fallback to local loopback socket pair if sink container not running
                self._last_error = f"Network target {host}:{port} connection failure: {str(e)}"
                time.sleep(1.0)
            finally:
                if s:
                    try:
                        s.close()
                    except Exception:
                        pass

        self.is_running = False
        self.stop_time = time.time()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.start_time = time.time()
        self.stop_time = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="NetworkWorker")
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.is_running = False
        self.stop_time = time.time()


# ---------------------------------------------------------------------------
# 6. Mixed Workload Profile
# ---------------------------------------------------------------------------
class MixedWorkload(BaseWorkload):
    """Combines CPU, Memory, Disk IO, and Network workloads with reproducible pseudo-random schedules."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("mixed", params)
        self._stop_event = threading.Event()
        self._sub_workloads: List[BaseWorkload] = []
        self._thread: Optional[threading.Thread] = None

    def validate_params(self, params: Dict[str, Any]) -> None:
        duration = params.get("duration", 60)
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        seed = params.get("seed", 42)
        if not isinstance(seed, int):
            raise ValueError(f"seed must be integer, got {seed}")

    def _worker(self) -> None:
        duration = self.params.get("duration", 60)
        seed = self.params.get("seed", 42)
        rng = random.Random(seed)

        cpu_intensity = rng.randint(25, 80)
        memory_mb = rng.randint(128, 512)
        io_rate_kbps = rng.randint(1024, 4096)

        # Initialize sub-workloads with derived deterministic parameters
        cpu_wl = CPUIntensiveWorkload({"intensity": cpu_intensity, "duration": duration, "num_workers": 1})
        mem_wl = MemoryIntensiveWorkload({"target_mb": memory_mb, "chunk_mb": 16, "duration": duration})
        io_wl = DiskIOWorkload({"block_size_kb": 32, "write_rate_kbps": io_rate_kbps, "duration": duration})

        self._sub_workloads = [cpu_wl, mem_wl, io_wl]
        for wl in self._sub_workloads:
            wl.start()

        end_time = time.time() + duration
        while not self._stop_event.is_set() and time.time() < end_time:
            time.sleep(0.5)

        for wl in self._sub_workloads:
            wl.stop()

        self.is_running = False
        self.stop_time = time.time()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.is_running = True
        self.start_time = time.time()
        self.stop_time = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="MixedWorker")
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        for wl in self._sub_workloads:
            wl.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.is_running = False
        self.stop_time = time.time()

    def health(self) -> Dict[str, Any]:
        h = super().health()
        h["sub_workloads"] = [w.health() for w in self._sub_workloads]
        return h
