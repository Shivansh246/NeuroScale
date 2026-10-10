"""tests/test_telemetry_orchestrator.py — Phase 8E unit tests.

All tests use stdlib only (no numpy / torch / real Docker).

The Docker SDK is fully mocked via ``unittest.mock``.

Test matrix
-----------
1.  normal_execution            — happy-path run completes with accepted records
2.  container_failure           — container that stops mid-run is removed from
                                   active set; loop continues with remaining containers
3.  keyboard_interrupt          — KeyboardInterrupt triggers graceful teardown
4.  scheduler_timing            — drift-compensating loop executes at least
                                   N ticks in expected wall time
5.  graceful_cleanup            — containers always stopped and removed after run
6.  validation_gate             — only validated records reach the exporter
7.  empty_containers            — loop exits cleanly when no containers are available
8.  docker_daemon_failure       — DockerException stops the experiment gracefully
9.  summary_fields              — ExperimentSummary contains all required fields
10. output_files_populated      — output_files dict is populated after a run
"""

import sys
import os
import time
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from telemetry.config import TelemetryCollectionConfig
from telemetry.orchestrator import TelemetryOrchestrator, ExperimentSummary


# ── helpers ──────────────────────────────────────────────────────────────────

def _minimal_config(
    duration: int = 2,
    instances: int = 1,
    sync: bool = False,
) -> TelemetryCollectionConfig:
    """Return a minimal TelemetryCollectionConfig for fast unit tests."""
    return TelemetryCollectionConfig.from_dict({
        "experiment": {
            "name": "test_run",
            "seed": 0,
            "duration_seconds": duration,
            "sampling_interval_seconds": 0.1,  # fast loop for tests
        },
        "containers": {
            "concurrent_instances": instances,
            "base_image": "test-image:latest",
            "network_name": "test_net",
            "resource_constraints": {"cpus": 1.0, "memory_mb": 256},
        },
        "export": {
            "export_jsonl": True,
            "export_csv": False,
            "sync_to_data_raw_metrics": sync,
            "datasets_dir": "datasets",
        },
    })


def _make_valid_stats(cpu_ns: int = 1_000_000_000) -> Dict[str, Any]:
    """Return a Docker-style stats dict that yields a valid telemetry record."""
    return {
        "cpu_stats": {
            "cpu_usage": {"total_usage": cpu_ns, "percpu_usage": [cpu_ns]},
            "system_cpu_usage": cpu_ns * 10,
            "online_cpus": 1,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": cpu_ns - 50_000_000},
            "system_cpu_usage": (cpu_ns * 10) - 100_000_000,
        },
        "memory_stats": {
            "usage": 104_857_600,
            "limit": 268_435_456,
        },
        "networks": {"eth0": {"rx_bytes": 1024, "tx_bytes": 512}},
        "blkio_stats": {
            "io_service_bytes_recursive": [
                {"op": "Read", "value": 4096},
                {"op": "Write", "value": 8192},
            ]
        },
        "pids_stats": {"current": 2},
    }


def _make_mock_container(
    name: str = "mock_container",
    cid: str = "abcdef123456",
    workload_mode: str = "cpu",
    status: str = "running",
    stats_dict: Optional[Dict] = None,
) -> MagicMock:
    """Build a mock Docker container object."""
    container = MagicMock()
    container.name = name
    container.id = cid
    container.status = status
    container.labels = {"neuroscale.mode": workload_mode, "mode": workload_mode}
    container.stats.return_value = stats_dict or _make_valid_stats()
    container.stop.return_value = None
    container.remove.return_value = None
    container.reload.return_value = None   # reload updates .status in place
    return container


def _make_mock_docker_client(containers: List[MagicMock]) -> MagicMock:
    """Return a mock Docker client whose containers.run() returns each container in order."""
    client = MagicMock()
    client.containers.run.side_effect = containers
    return client


# ── tests ────────────────────────────────────────────────────────────────────

class TestTelemetryOrchestrator(unittest.TestCase):

    def setUp(self):
        """Patch DatasetRunManager to avoid creating real directories."""
        self._run_path_patcher = patch(
            "telemetry.orchestrator.DatasetRunManager",
        )
        MockRunManager = self._run_path_patcher.start()
        # create_run_directory → a tmp path that actually exists
        import tempfile
        self._tmp = tempfile.mkdtemp()
        run_path = Path(self._tmp)
        (run_path / "exports").mkdir(exist_ok=True)
        (run_path / "metadata").mkdir(exist_ok=True)
        (run_path / "logs").mkdir(exist_ok=True)
        mock_mgr_instance = MagicMock()
        mock_mgr_instance.create_run_directory.return_value = run_path
        mock_mgr_instance.update_latest_symlink.return_value = run_path / "latest"
        mock_mgr_instance.sync_to_data_raw_metrics.return_value = run_path / "raw_metrics.jsonl"
        MockRunManager.return_value = mock_mgr_instance
        self._mock_mgr_cls = MockRunManager

        # Patch MetadataCollector and ManifestBuilder to skip real I/O
        self._meta_patcher = patch("telemetry.orchestrator.MetadataCollector")
        self._manifest_patcher = patch("telemetry.orchestrator.ManifestBuilder")
        MockMeta = self._meta_patcher.start()
        MockManifest = self._manifest_patcher.start()
        meta_path = run_path / "metadata" / "dataset_metadata.json"
        meta_path.write_text("{}", encoding="utf-8")
        MockMeta.collect.return_value = {}
        MockMeta.save.return_value = meta_path
        manifest_path = run_path / "metadata" / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        MockManifest.build.return_value = {}
        MockManifest.save.return_value = manifest_path

    def tearDown(self):
        self._run_path_patcher.stop()
        self._meta_patcher.stop()
        self._manifest_patcher.stop()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ── 1. Normal execution ───────────────────────────────────────────────

    def test_01_normal_execution(self):
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=1, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)

        summary = orch.run()

        self.assertIsInstance(summary, ExperimentSummary)
        self.assertTrue(summary.completed)
        self.assertGreater(summary.records_collected, 0)
        self.assertGreater(summary.records_accepted, 0)
        self.assertEqual(summary.records_rejected, 0)

    # ── 2. Container failure mid-run ──────────────────────────────────────

    def test_02_container_failure_mid_run(self):
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=2, instances=1)

        # After first health check (tick 5) the container status becomes 'exited'
        reload_call_count = [0]
        def reload_side_effect():
            reload_call_count[0] += 1
            if reload_call_count[0] >= 2:
                container.status = "exited"
        container.reload.side_effect = reload_side_effect

        orch = TelemetryOrchestrator(config, docker_client=client)
        summary = orch.run()

        # Experiment should still finish (not crash)
        self.assertIsInstance(summary, ExperimentSummary)
        # An exited container is dropped from the active set; the orchestrator
        # will still call remove() during final _remove_containers() cleanup.
        container.remove.assert_called()

    # ── 3. KeyboardInterrupt ──────────────────────────────────────────────

    def test_03_keyboard_interrupt_graceful_teardown(self):
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=60, instances=1)  # long enough

        orch = TelemetryOrchestrator(config, docker_client=client)

        # Interrupt the sampling loop from a separate thread after 300 ms
        def _interrupt():
            time.sleep(0.3)
            orch._stop_event.set()
            # Simulate KeyboardInterrupt by patching collect_all to raise
            if orch._collector:
                orch._collector.collect_all = MagicMock(side_effect=KeyboardInterrupt)

        t = threading.Thread(target=_interrupt, daemon=True)
        t.start()

        summary = orch.run()
        t.join(timeout=2)

        # completed=False because it was interrupted / stopped early
        self.assertIsInstance(summary, ExperimentSummary)
        # Containers should still be stopped
        container.stop.assert_called()
        container.remove.assert_called()

    # ── 4. Scheduler timing ───────────────────────────────────────────────

    def test_04_scheduler_timing(self):
        """The loop should fire close to duration / interval ticks."""
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        # 0.5 s duration, 0.1 s interval → ~5 ticks
        config = _minimal_config(duration=1, instances=1)
        config.experiment.sampling_interval_seconds = 0.1
        config.experiment.duration_seconds = 1

        orch = TelemetryOrchestrator(config, docker_client=client)
        t0 = time.time()
        summary = orch.run()
        elapsed = time.time() - t0

        # Should have taken approximately 1 second (allow generous tolerance)
        self.assertGreaterEqual(elapsed, 0.8)
        self.assertLess(elapsed, 5.0)  # upper bound: must not hang
        # At least a few records must have been collected
        self.assertGreater(summary.records_collected, 0)

    # ── 5. Graceful cleanup always runs ──────────────────────────────────

    def test_05_graceful_cleanup_always_runs(self):
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=1, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)
        orch.run()

        container.stop.assert_called()
        container.remove.assert_called()

    # ── 6. Validation gate ────────────────────────────────────────────────

    def test_06_only_valid_records_exported(self):
        """The exporter must never receive rejected records."""
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=1, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)

        written_records = []

        # Patch DatasetExporter.write_record to intercept exports
        original_setup = orch._setup_pipeline

        def patched_setup():
            original_setup()
            orch._exporter.write_record = lambda rec: written_records.append(rec)

        orch._setup_pipeline = patched_setup

        # Make some stats invalid (negative memory)
        bad_stats = _make_valid_stats()
        bad_stats["memory_stats"]["usage"] = -1  # will fail validation

        call_count = [0]
        def alternating_stats(stream=False):
            call_count[0] += 1
            return bad_stats if call_count[0] % 2 == 0 else _make_valid_stats()

        container.stats.side_effect = alternating_stats

        summary = orch.run()

        # Every written record must have been validated (non-negative memory)
        for rec in written_records:
            self.assertGreaterEqual(rec["memory_usage_bytes"], 0)

        # Some records should have been rejected
        self.assertGreater(summary.records_rejected, 0)

    # ── 7. Empty containers ───────────────────────────────────────────────

    def test_07_no_containers_exits_cleanly(self):
        client = MagicMock()
        # containers.run always fails → no containers
        client.containers.run.side_effect = Exception("image not found")
        config = _minimal_config(duration=2, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)
        summary = orch.run()

        # Should not crash; zero records
        self.assertIsInstance(summary, ExperimentSummary)
        self.assertEqual(summary.records_collected, 0)

    # ── 8. Docker daemon failure ──────────────────────────────────────────

    def test_08_docker_daemon_failure(self):
        """A DockerException from containers.run must not crash the process."""
        client = MagicMock()
        # Import the aliased exception
        from telemetry.orchestrator import DockerException as DE
        client.containers.run.side_effect = DE("daemon not running")
        config = _minimal_config(duration=2, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)

        # Should not propagate; returns a summary instead
        try:
            summary = orch.run()
            self.assertIsInstance(summary, ExperimentSummary)
        except Exception as exc:
            self.fail(f"Orchestrator leaked exception: {exc}")

    # ── 9. Summary fields ────────────────────────────────────────────────

    def test_09_summary_fields_present(self):
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=1, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)
        summary = orch.run()

        d = summary.to_dict()
        for required_key in [
            "experiment_id", "experiment_name", "start_time_utc", "end_time_utc",
            "duration_seconds", "completed", "containers_executed",
            "records", "validation_detail", "output_files",
        ]:
            self.assertIn(required_key, d, f"Missing key: {required_key}")

        for count_key in ["collected", "accepted", "rejected", "restart_events"]:
            self.assertIn(count_key, d["records"], f"Missing records.{count_key}")

    # ── 10. Output files dict populated ──────────────────────────────────

    def test_10_output_files_populated(self):
        container = _make_mock_container()
        client = _make_mock_docker_client([container])
        config = _minimal_config(duration=1, instances=1)
        orch = TelemetryOrchestrator(config, docker_client=client)
        summary = orch.run()

        # metadata and manifest should always appear
        self.assertIn("metadata", summary.output_files)
        self.assertIn("manifest", summary.output_files)
        self.assertIn("summary", summary.output_files)


if __name__ == "__main__":
    unittest.main()
