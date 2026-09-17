import datetime
from typing import List, Dict, Any

REQUIRED_FIELDS = [
    "timestamp",
    "container_id",
    "container_name",
    "cpu_percent",
    "cpu_usage_ns",
    "memory_usage_bytes",
    "memory_usage_mb",
    "memory_limit_bytes",
    "memory_limit_mb",
    "memory_percent",
    "workload_mode",
]

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single metric record.

    Returns the record unchanged if it passes validation.
    Raises ``ValueError`` with a descriptive message otherwise.
    """
    # Check required fields presence
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Timestamp must be numeric (epoch seconds) and non‑negative
    ts = record["timestamp"]
    if not _is_number(ts) or ts < 0:
        raise ValueError(f"Invalid timestamp: {ts}")

    # Numeric resource fields must be non‑negative numbers
    numeric_fields = [
        "cpu_percent",
        "cpu_usage_ns",
        "memory_usage_bytes",
        "memory_usage_mb",
        "memory_limit_bytes",
        "memory_limit_mb",
        "memory_percent",
    ]
    for f in numeric_fields:
        val = record[f]
        if not _is_number(val) or val < 0:
            raise ValueError(f"Invalid value for {f}: {val}")

    # workload_mode should be a non‑empty string
    mode = record["workload_mode"]
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"Invalid workload_mode: {mode!r}")

    return record

def validate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate a list of records, preserving order.

    Returns the list of validated records. Raises ``ValueError`` on the first
    malformed record.
    """
    return [validate_record(r) for r in records]
