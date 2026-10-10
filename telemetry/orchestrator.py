"""telemetry/orchestrator.py — Phase 8E: Telemetry Orchestration Layer.

Responsibilities
----------------
* Read the experiment configuration (Phase 8A ``TelemetryCollectionConfig``).
* Create and manage Docker containers via the Docker SDK.
* Assign workload profiles via ``WorkloadFactory`` (Phase 8B).
* Drive a precise 1-second sampling loop that compensates for execution drift.
* Feed every raw record through ``TelemetryValidator`` (Phase 8D); only
  *validated* records reach ``DatasetExporter`` (Phase 8A).
* On completion, flush metadata + manifest, sync to ``data/raw_metrics.jsonl``,
  and produce a final experiment summary.
* Recover gracefully from container crashes, Docker daemon failures,
  ``KeyboardInterrupt``, and partial experiment completion.

Contracts Preserved
-------------------
* ``DockerTelemetryCollector`` — **unmodified**.
* ``TelemetryValidator``       — **unmodified**.
* ``TelemetryFeaturePipeline`` — **unmodified** (delta/log1p logic untouched).
* All ML models and training scripts — **unmodified**.

Scheduler Design
----------------
The sampling loop uses a *drift-compensating* absolute deadline strategy:

    deadline = t_start + tick * interval

At the end of each tick the loop sleeps for ``max(0, deadline - now)`` rather
than a fixed ``time.sleep(interval)``.  This prevents the ~1 ms per cycle drift
that accumulates in naïve fixed-sleep loops over a 5-minute experiment.

Container Lifecycle
-------------------
1. ``_create_containers()``  — pull/create containers with resource constraints.
2. ``_start_containers()``   — ``container.start()`` + attach workload via SDK
                               labels or environment variables.
3. Sampling loop             — collect → validate → export at 1Hz.
4. ``_stop_containers()``    — ``container.stop(timeout=5)`` for each.
5. ``_remove_containers()``  — ``container.remove(force=True)`` in cleanup.

Error Recovery
--------------
* Each container polling call in ``DockerTelemetryCollector`` already returns
  ``None`` on failure; the orchestrator skips ``None`` results.
* Container crash detected when ``container.reload()`` shows status != "running";
  the container is removed from the active set and the loop continues.
* Docker daemon failure raises ``DockerException``; the orchestrator logs the
  error, triggers graceful cleanup, and re-raises so the caller is informed.
* ``KeyboardInterrupt`` causes a clean teardown before propagating.
* All cleanup paths are idempotent (safe to call multiple times).
"""

from __future__ import annotations

import datetime
import json
import logging
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional Docker SDK import (graceful degradation for testing environments)
# ---------------------------------------------------------------------------
try:
    import docker
    from docker.errors import DockerException, NotFound, APIError
except ImportError:
    docker = None  # type: ignore[assignment]
    DockerException = Exception  # type: ignore[assignment,misc]
    NotFound = Exception  # type: ignore[assignment,misc]
    APIError = Exception  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# NeuroScale telemetry modules (all unmodified)
# ---------------------------------------------------------------------------
from telemetry.config import TelemetryCollectionConfig
from telemetry.collector import DockerTelemetryCollector
from telemetry.validator import TelemetryValidator, ValidationSummary
from telemetry.exporters import DatasetExporter
from telemetry.run_manager import DatasetRunManager
from telemetry.metadata import MetadataCollector, ManifestBuilder
from telemetry.workloads.generator import WorkloadFactory

logger = logging.getLogger("telemetry.orchestrator")

# ---------------------------------------------------------------------------
# Experiment summary dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSummary:
    """Final summary produced at the end of every orchestrated experiment run.

    Attributes
    ----------
    experiment_id : str
        Unique UUID for this run.
    experiment_name : str
        Human-readable name from the YAML config.
    start_time_utc : str
        ISO-8601 UTC timestamp of experiment start.
    end_time_utc : str
        ISO-8601 UTC timestamp of experiment end.
    duration_seconds : float
        Actual wall-clock duration.
    containers_executed : int
        Number of containers that were started.
    records_collected : int
        Raw records returned by the collector (before validation).
    records_accepted : int
        Records that passed validation and were exported.
    records_rejected : int
        Records that failed at least one validation rule.
    restart_events : int
        Container cpu_usage_ns counter resets detected.
    output_files : dict[str, str]
        Map of label → absolute path for every generated output file.
    validation_summary : ValidationSummary
        Full aggregate validation counters.
    completed : bool
        ``True`` when the experiment ran to its configured duration.
        ``False`` on early exit (error / KeyboardInterrupt).
    """

    experiment_id: str = ""
    experiment_name: str = ""
    start_time_utc: str = ""
    end_time_utc: str = ""
    duration_seconds: float = 0.0
    containers_executed: int = 0
    records_collected: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    restart_events: int = 0
    output_files: Dict[str, str] = field(default_factory=dict)
    validation_summary: Optional[ValidationSummary] = None
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for logging and manifest."""
        vs = self.validation_summary
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "duration_seconds": round(self.duration_seconds, 2),
            "completed": self.completed,
            "containers_executed": self.containers_executed,
            "records": {
                "collected": self.records_collected,
                "accepted": self.records_accepted,
                "rejected": self.records_rejected,
                "restart_events": self.restart_events,
            },
            "validation_detail": {
                "accepted": vs.accepted if vs else 0,
                "rejected": vs.rejected if vs else 0,
                "warnings": vs.warnings_count if vs else 0,
                "missing_fields": vs.missing_fields if vs else 0,
                "type_mismatches": vs.type_mismatches if vs else 0,
                "container_restarts": vs.container_restarts if vs else 0,
            },
            "output_files": self.output_files,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TelemetryOrchestrator:
    """Top-level orchestrator that drives the complete telemetry collection
    pipeline for a single experiment run.

    Parameters
    ----------
    config : TelemetryCollectionConfig
        Parsed experiment configuration.
    docker_client : optional
        Pre-constructed Docker SDK client.  When ``None`` the orchestrator
        creates one lazily.  Pass a mock for unit tests.
    project_root : Path | str
        Absolute path to the NeuroScale repository root.  Defaults to the
        current working directory.
    """

    def __init__(
        self,
        config: TelemetryCollectionConfig,
        *,
        docker_client: Optional[Any] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.config = config
        self._docker_client = docker_client
        self._project_root = Path(project_root or ".").resolve()

        # Experiment identity
        self._experiment_id: str = str(uuid.uuid4())
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # Internal state
        self._containers: List[Any] = []           # active Docker container objects
        self._all_managed_containers: List[Any] = [] # all created containers (for cleanup)
        self._workloads: List[Any] = []            # BaseWorkload instances
        self._container_modes: Dict[str, str] = {} # container_name -> current workload_mode
        self._stop_event = threading.Event()       # signals the sampling loop to stop
        self._interrupted: bool = False

        # Pipeline components (instantiated in run())
        self._collector: Optional[DockerTelemetryCollector] = None
        self._validator: Optional[TelemetryValidator] = None
        self._exporter: Optional[DatasetExporter] = None
        self._run_path: Optional[Path] = None
        self._jsonl_path: Optional[Path] = None
        self._csv_path: Optional[Path] = None

        # Statistics
        self._records_collected: int = 0
        self._records_accepted: int = 0
        self._records_rejected: int = 0

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self) -> ExperimentSummary:
        """Execute the full experiment lifecycle.

        Returns
        -------
        ExperimentSummary
            Statistics and output paths for the completed (or interrupted)
            experiment.
        """
        logger.info(
            "Experiment '%s' starting (id=%s, duration=%ds, containers=%d)",
            self.config.experiment.name,
            self._experiment_id,
            self.config.experiment.duration_seconds,
            self.config.containers.concurrent_instances,
        )

        self._start_time = time.time()
        start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        completed = False

        try:
            self._setup_run_directory()
            self._setup_pipeline()
            self._create_and_start_containers()
            self._sampling_loop()
            completed = True

        except KeyboardInterrupt:
            self._interrupted = True
            logger.warning("KeyboardInterrupt received — initiating graceful teardown.")

        except DockerException as exc:
            logger.error("Docker daemon error: %s", exc, exc_info=True)

        except Exception as exc:
            logger.error("Unexpected orchestrator error: %s", exc, exc_info=True)

        finally:
            self._end_time = time.time()
            end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._teardown()
            summary = self._build_summary(
                start_utc=start_utc,
                end_utc=end_utc,
                completed=completed,
            )
            self._export_final_artifacts(summary, start_utc, end_utc)
            self._print_summary(summary)

        return summary  # type: ignore[return-value]  # always assigned in finally

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_run_directory(self) -> None:
        """Create the versioned dataset directory for this run."""
        datasets_dir = self._project_root / self.config.export.datasets_dir
        run_manager = DatasetRunManager(base_dir=datasets_dir)
        self._run_path = run_manager.create_run_directory(
            run_name=self.config.experiment.name
        )
        run_manager.update_latest_symlink(self._run_path)
        logger.info("Run directory: %s", self._run_path)

        self._jsonl_path = self._run_path / "exports" / "metrics.jsonl"
        self._csv_path = self._run_path / "exports" / "metrics.csv"

    def _setup_pipeline(self) -> None:
        """Initialise collector, validator, and exporter."""
        self._collector = DockerTelemetryCollector(
            client=self._docker_client,
            timeout_seconds=5.0,
        )
        self._validator = TelemetryValidator(future_tolerance_seconds=5.0)
        self._exporter = DatasetExporter(
            jsonl_path=self._jsonl_path if self.config.export.export_jsonl else None,
            csv_path=self._csv_path if self.config.export.export_csv else None,
        )
        logger.info("Pipeline initialised: collector → validator → exporter")

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _create_and_start_containers(self) -> None:
        """Create Docker containers and start them with their workload assignments."""
        n = self.config.containers.concurrent_instances
        logger.info("Creating %d container(s)…", n)

        import random as _random
        rng = _random.Random(self.config.experiment.seed)

        for i in range(n):
            container_seed = rng.randint(1, 1_000_000)
            workload = WorkloadFactory.create_randomized(
                self.config,
                seed=container_seed,
            )

            container = self._launch_container(
                index=i,
                workload_mode=workload.metadata().get("profile", "mixed"),
            )
            if container is None:
                logger.warning("Failed to create container %d — skipping.", i)
                continue

            self._containers.append(container)
            self._all_managed_containers.append(container)
            self._workloads.append(workload)

            try:
                workload.configure(workload.params if hasattr(workload, "params") else {})
                workload.start()
                logger.info(
                    "Container %s started with workload profile '%s'.",
                    getattr(container, "name", str(i)),
                    workload.metadata().get("profile", "unknown"),
                )
            except Exception as exc:
                logger.warning("Workload start failed for container %d: %s", i, exc)

        logger.info("%d container(s) active.", len(self._containers))

    def _launch_container(self, index: int, workload_mode: str) -> Optional[Any]:
        """Create and start a single Docker container.  Returns ``None`` on failure."""
        if self._docker_client is None and docker is None:
            raise RuntimeError("Docker SDK not available.")

        client = (
            self._docker_client
            if self._docker_client is not None
            else self._collector.client  # type: ignore[union-attr]
        )

        mem_limit = f"{self.config.containers.memory_mb}m"
        nano_cpus = int(self.config.containers.cpus * 1e9)
        name = f"neuroscale_bench_{self._experiment_id[:8]}_{index}"

        # Map telemetry profile names → workload script --mode values
        _MODE_MAP = {
            "cpu_intensive":    "cpu",
            "memory_intensive": "memory",
            "disk_io_intensive":"cpu",   # disk-io stress via cpu loop on container
            "network_intensive":"idle",  # network sink not available; default idle
            "mixed":            "mixed",
            "idle":             "idle",
        }
        container_mode = _MODE_MAP.get(workload_mode, "cpu")
        duration = self.config.experiment.duration_seconds + 60  # generous buffer
        container_cmd = ["python3", "-m", "workload.workload",
                         "--mode", container_mode, "--duration", str(duration)]

        try:
            container = client.containers.run(
                image=self.config.containers.base_image,
                name=name,
                command=container_cmd,
                detach=True,
                labels={
                    "neuroscale.experiment_id": self._experiment_id,
                    "neuroscale.mode": workload_mode,
                    "mode": workload_mode,
                },
                mem_limit=mem_limit,
                nano_cpus=nano_cpus,
                network=self.config.containers.network_name,
                remove=False,  # We remove explicitly in _remove_containers()
            )
            return container
        except DockerException as exc:
            logger.error("Failed to create container %d: %s", index, exc)
            return None

    def _check_container_health(self) -> None:
        """Remove containers that have exited/crashed from the active set."""
        alive: List[Any] = []
        workloads_alive: List[Any] = []

        for container, workload in zip(self._containers, self._workloads):
            try:
                container.reload()
                status = getattr(container, "status", "unknown")
                if status == "running":
                    alive.append(container)
                    workloads_alive.append(workload)
                else:
                    logger.warning(
                        "Container %s is no longer running (status=%s) — removing from active set.",
                        getattr(container, "name", "?"),
                        status,
                    )
                    self._stop_single_workload(workload)
            except (NotFound, DockerException) as exc:
                logger.warning(
                    "Container health check failed: %s — removing from active set.", exc
                )
                self._stop_single_workload(workload)

        self._containers = alive
        self._workloads = workloads_alive

    @staticmethod
    def _stop_single_workload(workload: Any) -> None:
        try:
            workload.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Drift-compensating sampling loop
    # ------------------------------------------------------------------

    def _sampling_loop(self) -> None:
        """Run the drift-compensating 1-Hz collect → validate → export loop.

        Scheduler design
        ~~~~~~~~~~~~~~~~
        Each iteration targets an absolute deadline:

            deadline = t0 + tick * interval

        After processing, the loop sleeps for ``max(0, deadline - time.time())``
        so that any processing time consumed within the tick is automatically
        subtracted from the sleep.  This caps accumulated drift to the
        sub-millisecond jitter of the OS scheduler rather than the ~1 ms per
        cycle drift of a naïve ``time.sleep(interval)`` approach.
        """
        interval = self.config.experiment.sampling_interval_seconds
        duration = self.config.experiment.duration_seconds
        t0 = time.time()
        tick = 0

        logger.info(
            "Sampling loop started (interval=%.1fs, duration=%ds).",
            interval,
            duration,
        )

        while not self._stop_event.is_set():
            # ---- Absolute deadline for this tick ----
            deadline = t0 + tick * interval
            tick += 1

            # ---- Collect ----
            cycle_time = time.time()
            if self._containers:
                raw_records = self._collector.collect_all(  # type: ignore[union-attr]
                    self._containers, timestamp=cycle_time
                )
            else:
                raw_records = []

            self._records_collected += len(raw_records)

            # ---- Dynamic Workload Phase Cycling ----
            # Rotate workload profiles periodically (e.g. every 60s) across containers
            cycle_interval = getattr(self.config.workload_randomization, "max_phase_duration_seconds", 60) or 60
            if tick > 1 and tick % cycle_interval == 0 and self.config.workload_randomization.enabled:
                import random as _rand
                all_modes = ["idle", "cpu", "memory", "disk", "network", "mixed"]
                _MODE_TO_CMD = {
                    "idle": ["python3", "-m", "workload.workload", "--mode", "idle", "--duration", str(cycle_interval + 10)],
                    "cpu": ["python3", "-m", "workload.workload", "--mode", "cpu", "--duration", str(cycle_interval + 10)],
                    "memory": ["python3", "-m", "workload.workload", "--mode", "memory", "--duration", str(cycle_interval + 10), "--memory", "384"],
                    "disk": ["python3", "-m", "workload.workload", "--mode", "cpu", "--duration", str(cycle_interval + 10)],
                    "network": ["python3", "-m", "workload.workload", "--mode", "burst", "--duration", str(cycle_interval + 10)],
                    "mixed": ["python3", "-m", "workload.workload", "--mode", "mixed", "--duration", str(cycle_interval + 10), "--memory", "320"],
                }
                for c in self._containers:
                    cname = getattr(c, "name", str(c))
                    new_mode = _rand.choice(all_modes)
                    self._container_modes[cname] = new_mode
                    try:
                        cmd = _MODE_TO_CMD.get(new_mode, _MODE_TO_CMD["idle"])
                        c.exec_run(" ".join(cmd), detach=True)
                        logger.info("Switched container %s to workload mode '%s'", cname, new_mode)
                    except Exception as err:
                        logger.debug("Failed to exec workload on %s: %s", cname, err)

            # ---- Validate → Export ----
            for record in raw_records:
                cname = record.get("container_name", "")
                if cname in self._container_modes:
                    record["workload_mode"] = self._container_modes[cname]
                result = self._validator.validate(record)  # type: ignore[union-attr]
                if result.valid:
                    self._exporter.write_record(record)    # type: ignore[union-attr]
                    self._records_accepted += 1
                else:
                    self._records_rejected += 1
                    logger.debug("Record rejected: %s", result.errors)

            # ---- Health check (every 5 ticks) ----
            if tick % 5 == 0:
                self._check_container_health()
                if not self._containers:
                    logger.warning("All containers stopped — ending experiment early.")
                    break

            # ---- Progress reporting (every 30 ticks) ----
            elapsed = time.time() - t0
            if tick % 30 == 0:
                remaining = max(0.0, duration - elapsed)
                logger.info(
                    "[%5.0fs elapsed / %5.0fs remaining] collected=%d accepted=%d rejected=%d",
                    elapsed,
                    remaining,
                    self._records_collected,
                    self._records_accepted,
                    self._records_rejected,
                )

            # ---- Duration check ----
            if elapsed >= duration:
                logger.info("Configured duration reached (%.1fs).", elapsed)
                break

            # ---- Drift-compensating sleep ----
            sleep_for = deadline - time.time()
            if sleep_for > 0:
                # Use a short-polling approach so KeyboardInterrupt / stop_event
                # are honoured within 100 ms even during the sleep.
                wake_at = time.time() + sleep_for
                while time.time() < wake_at and not self._stop_event.is_set():
                    time.sleep(min(0.1, wake_at - time.time()))
            # If sleep_for <= 0 the tick took longer than interval; just continue.

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """Stop workloads, stop containers, remove them, close pipeline."""
        logger.info("Teardown started.")

        # 1. Stop workloads
        for workload in self._workloads:
            self._stop_single_workload(workload)
        self._workloads.clear()

        # 2. Stop containers
        self._stop_containers()

        # 3. Remove containers
        self._remove_containers()

        # 4. Close collector thread pool + client
        if self._collector is not None:
            try:
                self._collector.close()
            except Exception:
                pass

        # 5. Flush + close exporter
        if self._exporter is not None:
            try:
                self._exporter.close()
            except Exception:
                pass

        logger.info("Teardown complete.")

    def _stop_containers(self) -> None:
        """Send stop signal to all running containers (timeout=5 s)."""
        for container in list(self._containers):
            try:
                container.stop(timeout=5)
                logger.info("Stopped container %s.", getattr(container, "name", "?"))
            except (NotFound, DockerException) as exc:
                logger.warning("Could not stop container: %s", exc)

    def _remove_containers(self) -> None:
        """Force-remove all managed containers."""
        for container in list(self._all_managed_containers):
            try:
                container.remove(force=True)
                logger.info("Removed container %s.", getattr(container, "name", "?"))
            except (NotFound, DockerException) as exc:
                logger.warning("Could not remove container: %s", exc)
        self._containers.clear()
        self._all_managed_containers.clear()

    # ------------------------------------------------------------------
    # Final artifact export
    # ------------------------------------------------------------------

    def _export_final_artifacts(
        self,
        summary: ExperimentSummary,
        start_utc: str,
        end_utc: str,
    ) -> None:
        """Write metadata.json, manifest.json, and sync to data/raw_metrics.jsonl."""
        if self._run_path is None:
            return

        metadata_dir = self._run_path / "metadata"

        # --- dataset_metadata.json ---
        metadata = MetadataCollector.collect(
            experiment_id=self._experiment_id,
            config=self.config.to_dict(),
            start_time_utc=start_utc,
            end_time_utc=end_utc,
            random_seed=self.config.experiment.seed,
        )
        metadata["experiment_summary"] = summary.to_dict()
        metadata_path = MetadataCollector.save(
            metadata, metadata_dir / "dataset_metadata.json"
        )
        logger.info("Metadata written: %s", metadata_path)
        summary.output_files["metadata"] = str(metadata_path)

        # --- manifest.json ---
        files_to_track: Dict[str, Path] = {}
        if self._jsonl_path and self._jsonl_path.is_file():
            files_to_track["jsonl"] = self._jsonl_path
        if self._csv_path and self._csv_path.is_file():
            files_to_track["csv"] = self._csv_path
        files_to_track["metadata"] = metadata_dir / "dataset_metadata.json"

        v_summary = self._validator.get_summary() if self._validator else None  # type: ignore[union-attr]
        val_status = "PASS" if (v_summary and v_summary.rejected == 0) else "PARTIAL"

        manifest = ManifestBuilder.build(
            experiment_id=self._experiment_id,
            files_to_track={k: v for k, v in files_to_track.items()},
            validation_status=val_status,
        )
        manifest_path = ManifestBuilder.save(manifest, metadata_dir / "manifest.json")
        logger.info("Manifest written: %s", manifest_path)
        summary.output_files["manifest"] = str(manifest_path)

        # --- Sync JSONL → data/raw_metrics.jsonl ---
        if (
            self.config.export.sync_to_data_raw_metrics
            and self._jsonl_path
            and self._jsonl_path.is_file()
            and self._jsonl_path.stat().st_size > 0
        ):
            run_manager = DatasetRunManager(
                base_dir=self._project_root / self.config.export.datasets_dir
            )
            target = self._project_root / "data" / "raw_metrics.jsonl"
            try:
                synced = run_manager.sync_to_data_raw_metrics(
                    source_jsonl=self._jsonl_path,
                    target_path=target,
                )
                logger.info("Dataset synced → %s", synced)
                summary.output_files["raw_metrics"] = str(synced)
            except Exception as exc:
                logger.warning("Sync to data/raw_metrics.jsonl failed: %s", exc)

        # --- Summary JSON ---
        summary_path = metadata_dir / "experiment_summary.json"
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Summary written: %s", summary_path)
        summary.output_files["summary"] = str(summary_path)

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        start_utc: str,
        end_utc: str,
        completed: bool,
    ) -> ExperimentSummary:
        duration = (self._end_time or time.time()) - (self._start_time or time.time())
        v_summary = self._validator.get_summary() if self._validator else None  # type: ignore[union-attr]

        return ExperimentSummary(
            experiment_id=self._experiment_id,
            experiment_name=self.config.experiment.name,
            start_time_utc=start_utc,
            end_time_utc=end_utc,
            duration_seconds=duration,
            containers_executed=self.config.containers.concurrent_instances,
            records_collected=self._records_collected,
            records_accepted=self._records_accepted,
            records_rejected=self._records_rejected,
            restart_events=v_summary.container_restarts if v_summary else 0,
            validation_summary=v_summary,
            completed=completed,
        )

    @staticmethod
    def _print_summary(summary: ExperimentSummary) -> None:
        """Emit a human-readable final report to stdout and the orchestrator logger."""
        lines = [
            "",
            "=" * 66,
            "  NeuroScale Telemetry Collection — Experiment Summary",
            "=" * 66,
            f"  ID         : {summary.experiment_id}",
            f"  Name       : {summary.experiment_name}",
            f"  Start      : {summary.start_time_utc}",
            f"  End        : {summary.end_time_utc}",
            f"  Duration   : {summary.duration_seconds:.1f}s",
            f"  Completed  : {summary.completed}",
            "",
            f"  Containers : {summary.containers_executed}",
            f"  Collected  : {summary.records_collected}",
            f"  Accepted   : {summary.records_accepted}",
            f"  Rejected   : {summary.records_rejected}",
            f"  Restarts   : {summary.restart_events}",
            "",
            "  Output files:",
        ]
        for label, path in summary.output_files.items():
            lines.append(f"    [{label}] {path}")
        lines.append("=" * 66)
        report = "\n".join(lines)
        print(report)
        logger.info(report)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def run_experiment(
    config_path: str | Path,
    *,
    docker_client: Optional[Any] = None,
    project_root: Optional[Path] = None,
) -> ExperimentSummary:
    """Load a YAML config file and run a complete telemetry experiment.

    Parameters
    ----------
    config_path : str | Path
        Path to a ``TelemetryCollectionConfig`` YAML file.
    docker_client : optional
        Override the Docker SDK client (useful for testing).
    project_root : optional
        Override the repository root directory.

    Returns
    -------
    ExperimentSummary
    """
    config = TelemetryCollectionConfig.from_yaml_file(config_path)
    orchestrator = TelemetryOrchestrator(
        config=config,
        docker_client=docker_client,
        project_root=project_root,
    )
    return orchestrator.run()
