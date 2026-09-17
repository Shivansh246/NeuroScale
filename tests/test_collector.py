import unittest
from unittest.mock import MagicMock
from monitoring.collector import Collector
from docker.errors import NotFound


class TestCollector(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_container = MagicMock()
        self.mock_container.id = "c1234567890abcdef"
        self.mock_container.name = "neuroscale-test"
        self.mock_container.labels = {"neuroscale.mode": "burst"}
        self.mock_container.attrs = {
            "Config": {
                "Cmd": ["--mode", "burst", "--duration", "60"],
                "Env": ["WORKLOAD_MODE=burst"],
            }
        }
        self.mock_client.containers.get.return_value = self.mock_container
        self.collector = Collector(client=self.mock_client)

    def test_calculate_cpu_percent(self):
        # 1 core fully loaded (cpu_delta == system_delta, online_cpus = 1) -> 100.0%
        cpu_stats = {
            "cpu_usage": {"total_usage": 200_000_000},
            "system_cpu_usage": 2_000_000_000,
            "online_cpus": 1,
        }
        precpu_stats = {
            "cpu_usage": {"total_usage": 100_000_000},
            "system_cpu_usage": 1_900_000_000,
        }
        pct = Collector._calculate_cpu_percent(cpu_stats, precpu_stats)
        self.assertEqual(pct, 100.0)

        # 4 cores, 50% loaded
        cpu_stats_4 = {
            "cpu_usage": {"total_usage": 500_000_000},
            "system_cpu_usage": 2_000_000_000,
            "online_cpus": 4,
        }
        precpu_stats_4 = {
            "cpu_usage": {"total_usage": 400_000_000},
            "system_cpu_usage": 1_200_000_000,
        }
        # (100M / 800M) * 4 * 100 = 50.0%
        pct_4 = Collector._calculate_cpu_percent(cpu_stats_4, precpu_stats_4)
        self.assertEqual(pct_4, 50.0)

    def test_calculate_cpu_percent_zero_or_invalid_deltas(self):
        self.assertEqual(Collector._calculate_cpu_percent({}, {}), 0.0)
        self.assertEqual(Collector._calculate_cpu_percent(None, None), 0.0)

        # Zero system delta
        cpu_stats = {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 500}
        precpu_stats = {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 500}
        self.assertEqual(Collector._calculate_cpu_percent(cpu_stats, precpu_stats), 0.0)

    def test_extract_workload_mode_from_labels(self):
        mock_c = MagicMock()
        mock_c.labels = {"neuroscale.mode": "memory"}
        mock_c.attrs = {}
        self.assertEqual(Collector._extract_workload_mode(mock_c), "memory")

    def test_extract_workload_mode_from_cmd(self):
        mock_c = MagicMock()
        mock_c.labels = {}
        mock_c.attrs = {"Config": {"Cmd": ["python", "workload.py", "--mode", "mixed"]}}
        self.assertEqual(Collector._extract_workload_mode(mock_c), "mixed")

        mock_c.attrs = {"Config": {"Cmd": ["--mode=cpu"]}}
        self.assertEqual(Collector._extract_workload_mode(mock_c), "cpu")

    def test_extract_workload_mode_from_env(self):
        mock_c = MagicMock()
        mock_c.labels = {}
        mock_c.attrs = {"Config": {"Env": ["FOO=BAR", "WORKLOAD_MODE=idle"]}}
        self.assertEqual(Collector._extract_workload_mode(mock_c), "idle")

    def test_extract_workload_mode_none(self):
        mock_c = MagicMock()
        mock_c.labels = {}
        mock_c.attrs = {"Config": {}}
        self.assertIsNone(Collector._extract_workload_mode(mock_c))

    def test_collect_structured_metrics(self):
        self.mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200_000_000},
                "system_cpu_usage": 2_000_000_000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 150_000_000},
                "system_cpu_usage": 1_900_000_000,
            },
            "memory_stats": {
                "usage": 268435456,  # 256 MB
                "limit": 1073741824,  # 1024 MB
            },
            "networks": {"eth0": {"rx_bytes": 1024, "tx_bytes": 2048}},
            "blkio_stats": {},
        }

        metrics = self.collector.collect("c1234567890abcdef")

        self.assertIn("timestamp", metrics)
        self.assertEqual(metrics["container_id"], "c1234567890abcdef")
        self.assertEqual(metrics["container_name"], "neuroscale-test")
        self.assertEqual(metrics["workload_mode"], "burst")
        self.assertEqual(metrics["cpu_usage_ns"], 200_000_000)
        self.assertEqual(metrics["cpu_percent"], 100.0)
        self.assertEqual(metrics["memory_usage_bytes"], 268435456)
        self.assertEqual(metrics["memory_usage_mb"], 256.0)
        self.assertEqual(metrics["memory_limit_bytes"], 1073741824)
        self.assertEqual(metrics["memory_limit_mb"], 1024.0)
        self.assertEqual(metrics["memory_percent"], 25.0)

    def test_collect_not_found(self):
        self.mock_client.containers.get.side_effect = NotFound("Container not found")
        with self.assertRaises(NotFound):
            self.collector.collect("nonexistent_container")


if __name__ == "__main__":
    unittest.main()
