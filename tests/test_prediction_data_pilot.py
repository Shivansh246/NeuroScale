import unittest
from unittest.mock import MagicMock
import time

from experiments.prediction_data.collector import PilotCollector

class TestPilotCollector(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_container = MagicMock()
        self.mock_container.id = "c123456"
        self.mock_container.name = "neuroscale-workload"
        self.mock_container.labels = {}
        self.mock_container.attrs = {}
        self.mock_client.containers.get.return_value = self.mock_container
        self.collector = PilotCollector(client=self.mock_client)

    def test_collect_first_sample(self):
        self.mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 100_000_000},
                "system_cpu_usage": 2_000_000_000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 50_000_000},
                "system_cpu_usage": 1_900_000_000,
            },
            "memory_stats": {
                "usage": 268435456,
                "limit": 1073741824,
            },
            "networks": {"eth0": {"rx_bytes": 1024, "tx_bytes": 2048}},
            "blkio_stats": {"io_service_bytes_recursive": [{"op": "Read", "value": 500}, {"op": "Write", "value": 600}]},
        }

        # First sample should have cpu_usage_ns_delta as 0 or None.
        metrics = self.collector.collect("c123456", workload_mode="idle")
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["workload_mode"], "idle")
        self.assertEqual(metrics["cpu_usage_ns"], 100_000_000)
        self.assertEqual(metrics["cpu_usage_ns_delta"], 0)
        self.assertEqual(metrics["network_rx_bytes"], 1024)
        self.assertEqual(metrics["network_tx_bytes"], 2048)
        self.assertEqual(metrics["block_read_bytes"], 500)
        self.assertEqual(metrics["block_write_bytes"], 600)

    def test_collect_second_sample_delta(self):
        # Sample 1
        self.mock_container.stats.return_value = {
            "cpu_stats": {"cpu_usage": {"total_usage": 100_000_000}, "system_cpu_usage": 2_000_000_000, "online_cpus": 2},
            "precpu_stats": {"cpu_usage": {"total_usage": 50_000_000}, "system_cpu_usage": 1_900_000_000},
            "memory_stats": {"usage": 268435456, "limit": 1073741824},
            "networks": {},
            "blkio_stats": {},
        }
        self.collector.collect("c123456", workload_mode="idle")

        # Sample 2
        self.mock_container.stats.return_value = {
            "cpu_stats": {"cpu_usage": {"total_usage": 250_000_000}, "system_cpu_usage": 2_100_000_000, "online_cpus": 2},
            "precpu_stats": {"cpu_usage": {"total_usage": 100_000_000}, "system_cpu_usage": 2_000_000_000},
            "memory_stats": {"usage": 268435456, "limit": 1073741824},
            "networks": {},
            "blkio_stats": {},
        }
        metrics2 = self.collector.collect("c123456", workload_mode="normal")
        
        self.assertEqual(metrics2["cpu_usage_ns_delta"], 150_000_000)
        self.assertEqual(metrics2["cpu_usage_ns"], 250_000_000)

if __name__ == "__main__":
    unittest.main()
