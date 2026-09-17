"""Data preprocessing utilities for NeuroScale.

Provides functions to load raw JSONL metric records, validate them using
:data:, and resample them to a deterministic interval.
"""

import json
from typing import List, Dict

from data.validation import validate_records
from data.config import PipelineConfig


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
