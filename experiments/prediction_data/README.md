# Prediction Data Collection Pilot

This directory contains the Phase 10A prediction data pilot code. It orchestrates a deterministic workload schedule using the `neuroscale-workload` container and collects high-fidelity telemetry, enforcing delta-CPU measurements and explicit workload labels.

## Contents
- `collector.py`: A wrapper around `monitoring.collector.Collector` that adds `cpu_usage_ns_delta` and other needed fields.
- `workload_schedule.py`: Orchestrator that cycles through a 45-minute sequence of idle, CPU, memory, and mixed stresses.
- `analysis.py`: Validates the output JSONL file and performs distribution analysis on CPU/Memory per workload phase.
- `tests/test_prediction_data_pilot.py`: Unit tests (run via `python -m unittest`).

## Output
Data is saved to `data/prediction_experiment/real_workload_pilot.jsonl`.
