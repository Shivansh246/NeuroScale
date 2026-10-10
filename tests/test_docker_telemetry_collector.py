"""Unit tests for DockerTelemetryCollector using mocked Docker SDK responses."""

import unittest
from unittest.mock import MagicMock
from telemetry.collector import DockerTelemetryCollector


class TestDockerTelemetryCollector(unittest.TestCase):

    def setUp(self):
        self.collector = DockerTelemetryCollector()

    def test_compute_cpu_percent(self):
        cpu_stats = {
            "cpu_usage": {"total_usage": 1500000000},
            "system_cpu_usage": 10000000000,
            "online_cpus": 2,
        }
        precpu_stats = {
            "cpu_usage": {"total_usage": 1000000000},
            "system_cpu_usage": 9000000000,
        }
        # delta cpu = 500M, delta sys = 1000M -> (500M / 1000M) * 2 * 100 = 100.0%
        pct = self.collector.compute_cpu_percent(cpu_stats, precpu_stats)
        self.assertEqual(pct, 100.0)

        # Missing or empty precpu_stats should return 0.0
        self.assertEqual(self.collector.compute_cpu_percent(cpu_stats, {}), 0.0)
        self.assertEqual(self.collector.compute_cpu_percent({}, precpu_stats), 0.0)

    def test_extract_memory_stats(self):
        mem_stats = {
            "usage": 268435456,        # 256 MB
            "limit": 1073741824,       # 1024 MB
        }
        res = self.collector.extract_memory_stats(mem_stats)
        self.assertEqual(res["memory_usage_bytes"], 268435456)
        self.assertEqual(res["memory_usage_mb"], 256.0)
        self.assertEqual(res["memory_limit_bytes"], 1073741824)
        self.assertEqual(res["memory_limit_mb"], 1024.0)
        self.assertEqual(res["memory_percent"], 25.0)

        # Empty memory stats returns zeros
        res_empty = self.collector.extract_memory_stats({})
        self.assertEqual(res_empty["memory_usage_mb"], 0.0)
        self.assertEqual(res_empty["memory_percent"], 0.0)

    def test_aggregate_network_stats(self):
        networks = {
            "eth0": {"rx_bytes": 1000, "tx_bytes": 2000},
            "eth1": {"rx_bytes": 500, "tx_bytes": 300},
        }
        res = self.collector.aggregate_network_stats(networks)
        self.assertEqual(res["network_rx_bytes"], 1500)
        self.assertEqual(res["network_tx_bytes"], 2300)

        # Disappeared or empty interfaces
        self.assertEqual(self.collector.aggregate_network_stats({}), {"network_rx_bytes": 0, "network_tx_bytes": 0})

    def test_aggregate_blkio_stats(self):
        blkio = {
            "io_service_bytes_recursive": [
                {"major": 8, "minor": 0, "op": "Read", "value": 4096},
                {"major": 8, "minor": 0, "op": "Write", "value": 8192},
                {"major": 8, "minor": 16, "op": "Read", "value": 2048},
                {"major": 8, "minor": 16, "op": "Write", "value": 4096},
            ]
        }
        res = self.collector.aggregate_blkio_stats(blkio)
        self.assertEqual(res["block_read_bytes"], 6144)
        self.assertEqual(res["block_write_bytes"], 12288)

        # Empty blkio
        self.assertEqual(self.collector.aggregate_blkio_stats({}), {"block_read_bytes": 0, "block_write_bytes": 0})

    def test_extract_pid_stats(self):
        self.assertEqual(self.collector.extract_pid_stats({"current": 12}), 12)
        self.assertEqual(self.collector.extract_pid_stats({}), 0)

    def test_collect_container_normal_flow(self):
        mock_container = MagicMock()
        mock_container.id = "cont_abc123"
        mock_container.name = "worker-1"
        mock_container.labels = {"neuroscale.mode": "mixed"}
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 1234567890},
                "system_cpu_usage": 5000000000,
                "online_cpus": 1,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1000000000},
                "system_cpu_usage": 4500000000,
            },
            "memory_stats": {
                "usage": 134217728,  # 128 MB
                "limit": 536870912,  # 512 MB
            },
            "networks": {
                "eth0": {"rx_bytes": 512, "tx_bytes": 1024}
            },
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "Read", "value": 100},
                    {"op": "Write", "value": 200},
                ]
            },
            "pids_stats": {"current": 4},
        }

        rec = self.collector.collect_container(mock_container, timestamp=1700000000.0)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["container_id"], "cont_abc123")
        self.assertEqual(rec["container_name"], "worker-1")
        self.assertEqual(rec["workload_mode"], "mixed")
        # CPU delta = 234567890, sys delta = 500000000 -> 46.91%
        self.assertEqual(rec["cpu_percent"], 46.91)
        # CRITICAL: cpu_usage_ns must remain the raw total_usage from Docker
        self.assertEqual(rec["cpu_usage_ns"], 1234567890)
        self.assertEqual(rec["memory_usage_mb"], 128.0)
        self.assertEqual(rec["memory_percent"], 25.0)
        self.assertEqual(rec["network_rx_bytes"], 512)
        self.assertEqual(rec["network_tx_bytes"], 1024)
        self.assertEqual(rec["block_read_bytes"], 100)
        self.assertEqual(rec["block_write_bytes"], 200)
        self.assertEqual(rec["pids"], 4)

    def test_collect_container_api_failure_and_missing_stats(self):
        # 1. Stats call raises Exception
        mock_failing_container = MagicMock()
        mock_failing_container.stats.side_effect = RuntimeError("Docker API timeout")
        self.assertIsNone(self.collector.collect_container(mock_failing_container))

        # 2. Stats call returns empty dictionary
        mock_empty_container = MagicMock()
        mock_empty_container.stats.return_value = {}
        self.assertIsNone(self.collector.collect_container(mock_empty_container))

    def test_collect_all_multi_containers(self):
        c1 = MagicMock()
        c1.id = "c1"
        c1.name = "worker-1"
        c1.labels = {}
        c1.stats.return_value = {
            "cpu_stats": {"cpu_usage": {"total_usage": 100}},
            "precpu_stats": {},
            "memory_stats": {"usage": 1024, "limit": 2048},
        }

        c2 = MagicMock()
        c2.id = "c2"
        c2.name = "worker-2"
        c2.labels = {}
        c2.stats.return_value = {
            "cpu_stats": {"cpu_usage": {"total_usage": 200}},
            "precpu_stats": {},
            "memory_stats": {"usage": 2048, "limit": 4096},
        }

        records = self.collector.collect_all([c1, c2], timestamp=1700000001.0)
        self.assertEqual(len(records), 2)
        ids = {r["container_id"] for r in records}
        self.assertEqual(ids, {"c1", "c2"})

    def tearDown(self):
        self.collector.close()


if __name__ == "__main__":
    unittest.main()
