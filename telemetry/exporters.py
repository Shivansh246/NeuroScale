"""Dual-format stream exporters for JSONL and CSV with atomic export syncing."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import json
import threading


# Canonical field ordering for CSV export
CSV_FIELDNAMES = [
    "timestamp",
    "container_id",
    "container_name",
    "workload_mode",
    "cpu_percent",
    "cpu_usage_ns",
    "memory_usage_mb",
    "memory_usage_bytes",
    "memory_limit_mb",
    "memory_limit_bytes",
    "memory_percent",
    "network_rx_bytes",
    "network_tx_bytes",
    "block_read_bytes",
    "block_write_bytes",
    "pids",
]


class DatasetExporter:
    """Synchronously exports validated telemetry records to both JSONL and CSV."""

    def __init__(
        self,
        jsonl_path: Optional[Path | str] = None,
        csv_path: Optional[Path | str] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).resolve() if jsonl_path else None
        self.csv_path = Path(csv_path).resolve() if csv_path else None
        self._lock = threading.Lock()

        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer = None
        self._records_written = 0

        self._open_files()

    def _open_files(self) -> None:
        """Initialize target export files and headers."""
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")

        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.csv_path, "w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=CSV_FIELDNAMES,
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()

    def write_record(self, record: Dict[str, Any]) -> None:
        """Write a single record to both JSONL and CSV streams in a thread-safe manner."""
        with self._lock:
            # Write to JSONL
            if self._jsonl_file:
                self._jsonl_file.write(json.dumps(record) + "\n")
                self._jsonl_file.flush()

            # Write to CSV
            if self._csv_writer:
                self._csv_writer.writerow(record)
                self._csv_file.flush()

            self._records_written += 1

    @property
    def records_written(self) -> int:
        return self._records_written

    def close(self) -> None:
        """Flush and close all open export file handles."""
        with self._lock:
            if self._jsonl_file and not self._jsonl_file.closed:
                self._jsonl_file.flush()
                self._jsonl_file.close()
            if self._csv_file and not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()

    def __enter__(self) -> "DatasetExporter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
