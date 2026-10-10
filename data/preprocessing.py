"""Data preprocessing utilities for NeuroScale.

Provides functions to load raw JSONL metric records, validate them using
:data:, and resample them to a deterministic interval.
"""

import json
import math
from typing import List, Dict, Optional, Sequence, Tuple, Any

try:
    import numpy as np
except ImportError:
    np = None

from data.validation import validate_records
from data.config import PipelineConfig


class TelemetryFeaturePipeline:
    """Unified telemetry preprocessing and feature extraction component.

    Provides identical feature processing for both offline training and online inference:
    1. Preserves raw cumulative cpu_usage_ns in telemetry records.
    2. Computes cpu_usage_ns_delta from raw cpu_usage_ns.
    3. Handles container transitions and counter resets gracefully (no negative deltas).
    4. Transforms cpu_usage_ns_delta using log1p(max(0, delta)) for model features.
    5. Exposes identical processed features for both Transformer and Autoencoder.
    """

    DEFAULT_FEATURE_COLUMNS = (
        "cpu_percent",
        "cpu_usage_ns_delta",
        "memory_usage_mb",
        "memory_percent",
    )

    def __init__(self, feature_columns: Optional[Sequence[str]] = None) -> None:
        self.feature_columns = list(feature_columns or self.DEFAULT_FEATURE_COLUMNS)
        self.prev_cpu_usage_ns: Optional[int] = None
        self.current_session_id: Optional[str] = None

    def reset(self) -> None:
        """Reset internal session and counter state."""
        self.prev_cpu_usage_ns = None
        self.current_session_id = None

    def compute_delta(
        self,
        metrics: Dict[str, Any],
        container_id: Optional[str] = None,
    ) -> Tuple[int, bool]:
        """Compute cpu_usage_ns_delta from raw cpu_usage_ns and update metrics.

        Parameters
        ----------
        metrics : dict
            Metric dictionary containing 'cpu_usage_ns'.
        container_id : str, optional
            Container identifier to track container restarts/transitions.

        Returns
        -------
        tuple of (delta: int, counter_reset: bool)
        """
        current_cpu_ns = metrics.get("cpu_usage_ns", 0)
        if current_cpu_ns is None:
            current_cpu_ns = 0
        current_cpu_ns = int(current_cpu_ns)

        counter_reset = False
        target_container = container_id if container_id is not None else metrics.get("container_id")

        if target_container is not None and self.current_session_id != target_container:
            self.current_session_id = target_container
            delta = 0
            self.prev_cpu_usage_ns = current_cpu_ns
        elif self.prev_cpu_usage_ns is None:
            delta = 0
            self.prev_cpu_usage_ns = current_cpu_ns
        else:
            if current_cpu_ns < self.prev_cpu_usage_ns:
                # Docker daemon or container counter reset
                counter_reset = True
                delta = current_cpu_ns
            else:
                delta = current_cpu_ns - self.prev_cpu_usage_ns
            self.prev_cpu_usage_ns = current_cpu_ns

        delta = max(0, int(delta))
        metrics["cpu_usage_ns_delta"] = delta
        return delta, counter_reset

    def process_records(
        self,
        records: List[Dict[str, Any]],
        reset_state: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process an ordered batch of records, ensuring cpu_usage_ns_delta is computed.

        Operates in-place on records and returns the list.
        """
        if reset_state:
            self.reset()
        for r in records:
            if "cpu_usage_ns_delta" in r and "cpu_usage_ns" not in r:
                continue
            self.compute_delta(r, r.get("container_id"))
        return records

    def extract_feature_row(
        self,
        record: Dict[str, Any],
        feature_columns: Optional[Sequence[str]] = None,
    ) -> List[float]:
        """Extract a single feature vector applying model-ready transforms."""
        cols = feature_columns or self.feature_columns
        row: List[float] = []
        for col in cols:
            val = record.get(col, 0.0)
            if col == "cpu_usage_ns_delta":
                val = math.log1p(max(0.0, float(val)))
            row.append(float(val))
        return row

    def extract_feature_sequence(
        self,
        records: Sequence[Dict[str, Any]],
        feature_columns: Optional[Sequence[str]] = None,
    ) -> Any:
        """Extract a 2D array of shape (N, feature_count) with log1p delta."""
        rows = [self.extract_feature_row(r, feature_columns) for r in records]
        if np is not None:
            return np.array(rows, dtype=np.float32)
        return rows

    def transform_sequence(
        self,
        sequence: Any,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> Any:
        """Apply pipeline transforms (log1p delta) to a pre-extracted raw feature matrix."""
        cols = list(feature_columns or self.feature_columns)
        if np is not None and isinstance(sequence, np.ndarray):
            seq = sequence.copy()
            if "cpu_usage_ns_delta" in cols:
                idx = cols.index("cpu_usage_ns_delta")
                seq[:, idx] = np.log1p(np.maximum(0.0, seq[:, idx]))
            return seq
        return sequence


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dict records.

    Parameters
    ----------
    path: str
        Path to the JSONL file.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _nearest_resample(sorted_records: List[Dict], interval_seconds: int) -> List[Dict]:
    """Resample a time‑ordered list of records using nearest‑sample.

    The first record defines the start timestamp (rounded down to the nearest
    interval). Subsequent timestamps are mapped to the nearest grid point.
    """
    if not sorted_records:
        return []
    start_ts = int((sorted_records[0]["timestamp"] // interval_seconds) * interval_seconds)
    end_ts = int(sorted_records[-1]["timestamp"])
    grid = list(range(start_ts, end_ts + 1, interval_seconds))
    resampled = []
    idx = 0
    for t in grid:
        while idx < len(sorted_records) - 1 and sorted_records[idx]["timestamp"] < t:
            idx += 1
        if idx == 0:
            chosen = sorted_records[0]
        else:
            prev = sorted_records[idx - 1]
            cur = sorted_records[idx]
            chosen = cur if abs(cur["timestamp"] - t) < abs(t - prev["timestamp"]) else prev
        cloned = dict(chosen)
        cloned["timestamp"] = t
        resampled.append(cloned)
    return resampled


def resample_records(records: List[Dict], interval_seconds: int = None) -> List[Dict]:
    """Validate and resample raw records.

    Returns a list of records ordered by timestamp and sampled at the given
    interval (default from :class:).
    """
    cfg = PipelineConfig()
    interval = interval_seconds if interval_seconds is not None else cfg.sampling_interval_seconds
    validate_records(records)
    sorted_records = sorted(records, key=lambda r: r["timestamp"])
    return _nearest_resample(sorted_records, interval)
