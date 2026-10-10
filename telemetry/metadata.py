"""System metadata and manifest generation utilities for experiment auditing."""

from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import hashlib
import json
import os
import platform
import sys


def get_docker_version() -> str:
    """Safely query local Docker server and client version without collecting telemetry."""
    try:
        import docker
        client = docker.from_env()
        version_info = client.version()
        return version_info.get("Version", "unknown")
    except Exception:
        # Fall back to CLI if docker package fails or daemon unreachable
        try:
            import subprocess
            res = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
    return "docker_unavailable"


def get_total_ram_bytes() -> int:
    """Retrieve total host memory in bytes across macOS, Linux, and Windows."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass

    # macOS / BSD sysctl
    if sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        except Exception:
            pass

    # Linux /proc/meminfo
    if Path("/proc/meminfo").is_file():
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            pass

    return 0


class MetadataCollector:
    """Captures host environment, hardware, timestamps, and configuration snapshots."""

    @staticmethod
    def collect(
        experiment_id: str,
        config: Dict[str, Any],
        start_time_utc: Optional[str] = None,
        end_time_utc: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Gather complete metadata dictionary."""
        total_ram = get_total_ram_bytes()
        total_ram_gb = round(total_ram / (1024 ** 3), 2) if total_ram > 0 else "unknown"

        return {
            "experiment_id": experiment_id,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "start_time_utc": start_time_utc,
            "end_time_utc": end_time_utc,
            "random_seed": random_seed,
            "environment": {
                "os": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": sys.version.split()[0],
                "python_compiler": platform.python_compiler(),
                "docker_version": get_docker_version(),
                "cpu_count": os.cpu_count() or 1,
                "total_ram_bytes": total_ram,
                "total_ram_gb": total_ram_gb,
            },
            "configuration": config,
        }

    @staticmethod
    def save(metadata: Dict[str, Any], destination_path: Path | str) -> Path:
        """Write metadata JSON formatted to disk."""
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return dest


class ManifestBuilder:
    """Builds SHA256 cryptographic manifests for all generated experiment artifacts."""

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Compute SHA256 checksum for a file."""
        p = Path(file_path)
        if not p.is_file():
            return "FILE_NOT_FOUND"

        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def build(
        cls,
        experiment_id: str,
        files_to_track: Dict[str, Path | str],
        validation_status: str = "PENDING",
    ) -> Dict[str, Any]:
        """Construct the manifest dictionary tracking file sizes, records count, and checksums."""
        manifest: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "validation_status": validation_status,
            "files": {},
        }

        for label, file_path in files_to_track.items():
            p = Path(file_path)
            if p.is_file():
                size = p.stat().st_size
                sha256 = cls.compute_sha256(p)
                entry: Dict[str, Any] = {
                    "path": str(p.name),
                    "size_bytes": size,
                    "sha256": sha256,
                }
                # Count lines if JSONL or CSV
                if p.suffix in (".jsonl", ".csv"):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            entry["records_count"] = sum(1 for _ in f)
                    except Exception:
                        entry["records_count"] = "unknown"
                manifest["files"][label] = entry
            else:
                manifest["files"][label] = {
                    "path": str(p.name),
                    "status": "NOT_GENERATED",
                    "sha256": "placeholder",
                }

        return manifest

    @staticmethod
    def save(manifest: Dict[str, Any], destination_path: Path | str) -> Path:
        """Write manifest JSON to disk."""
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return dest
