"""Data collection utility for NeuroScale.

Provides a command‑line entry point that repeatedly invokes the existing
:class:`monitoring.collector.Collector` to collect metrics from a running
Docker container and writes each record as a JSON line to the specified
output file.

The collector runs for a configurable ``duration`` (seconds) or until
interrupted (Ctrl‑C).  The collection interval defaults to the value from
:class:`data.config.PipelineConfig`.
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import Optional

from monitoring.collector import Collector
from data.config import PipelineConfig

_stop_requested = False

def _signal_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\nStop signal received, terminating collection…", file=sys.stderr)

def collect_to_file(
    container_id: str,
    output_path: str,
    client: Optional[object] = None,
    interval: Optional[int] = None,
    duration: Optional[int] = None,
) -> None:
    """Collect metrics from *container_id* and append them as JSONL.

    Parameters
    ----------
    container_id: str
        Docker container identifier.
    output_path: str
        Path to the JSONL file that will receive metric records.
    client: optional
        Optional Docker client to pass to :class:`Collector`. If ``None`` the
        default client is created inside the collector.
    interval: int, optional
        Collection interval in seconds. If ``None`` the interval from
        :class:`PipelineConfig` is used.
    duration: int, optional
        Total runtime in seconds. If ``None`` collection continues until the
        process is interrupted (SIGINT/SIGTERM).
    """
    cfg = PipelineConfig()
    interval_seconds = interval if interval is not None else cfg.sampling_interval_seconds
    collector = Collector(client=client)
    start_time = time.time()
    records_written = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        while True:
            if _stop_requested:
                break
            if duration is not None and (time.time() - start_time) >= duration:
                break
            record = collector.collect(container_id)
            f.write(json.dumps(record) + "\n")
            f.flush()
            records_written += 1
            time.sleep(interval_seconds)
    print(f"Collected {records_written} records to {output_path}")

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Docker container metrics to JSONL.")
    parser.add_argument("--container", required=True, help="Container ID or name to monitor")
    parser.add_argument("--out", required=True, help="Output JSONL file path (will be created if missing)")
    parser.add_argument("--interval", type=int, help="Override collection interval in seconds (default from config)")
    parser.add_argument("--duration", type=int, help="Run for this many seconds; omit for indefinite until interrupted")
    return parser.parse_args(argv)

def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    collect_to_file(
        container_id=args.container,
        output_path=args.out,
        interval=args.interval,
        duration=args.duration,
    )

if __name__ == "__main__":
    main()
