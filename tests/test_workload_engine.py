"""Unit tests for NeuroScale workload profiles, factory, and validation bounds."""

import unittest
import time
from telemetry.workloads.base import BaseWorkload
from telemetry.workloads.profiles import (
    IdleWorkload,
    CPUIntensiveWorkload,
    MemoryIntensiveWorkload,
    DiskIOWorkload,
    NetworkIntensiveWorkload,
    MixedWorkload,
)
from telemetry.workloads.generator import WorkloadFactory
from telemetry.config import TelemetryCollectionConfig


class TestWorkloadProfiles(unittest.TestCase):

    def test_idle_workload_lifecycle_and_validation(self):
        wl = IdleWorkload({"duration": 1})
        self.assertEqual(wl.name, "idle")
        self.assertFalse(wl.is_running)
        
        # Validation error on negative duration
        with self.assertRaises(ValueError):
            wl.validate_params({"duration": -10})

        wl.start()
        self.assertTrue(wl.is_running)
        time.sleep(0.1)
        meta = wl.metadata()
        self.assertEqual(meta["name"], "idle")
        self.assertTrue(meta["health"]["is_running"])
        wl.stop()
        self.assertFalse(wl.is_running)

    def test_cpu_workload_bounds_and_lifecycle(self):
        # Invalid intensity bounds (<10% or >100%)
        with self.assertRaises(ValueError):
            CPUIntensiveWorkload({"intensity": 5})
        with self.assertRaises(ValueError):
            CPUIntensiveWorkload({"intensity": 105})
        with self.assertRaises(ValueError):
            CPUIntensiveWorkload({"intensity": 50, "num_workers": 0})

        wl = CPUIntensiveWorkload({"intensity": 40, "num_workers": 1, "duration": 2})
        wl.start()
        self.assertTrue(wl.is_running)
        time.sleep(0.1)
        wl.stop()
        self.assertFalse(wl.is_running)

    def test_memory_workload_bounds_and_lifecycle(self):
        with self.assertRaises(ValueError):
            MemoryIntensiveWorkload({"target_mb": -10})
        with self.assertRaises(ValueError):
            MemoryIntensiveWorkload({"target_mb": 64, "chunk_mb": 128})  # chunk > target

        wl = MemoryIntensiveWorkload({"target_mb": 32, "chunk_mb": 16, "step_interval_seconds": 0.05, "duration": 2})
        wl.start()
        self.assertTrue(wl.is_running)
        time.sleep(0.15)
        h = wl.health()
        self.assertIn("allocated_chunks", h)
        wl.stop()
        self.assertFalse(wl.is_running)
        self.assertEqual(wl.health()["allocated_chunks"], 0)  # Cleaned up

    def test_disk_io_workload_bounds_and_lifecycle(self):
        with self.assertRaises(ValueError):
            DiskIOWorkload({"block_size_kb": 0})
        with self.assertRaises(ValueError):
            DiskIOWorkload({"write_rate_kbps": -50})

        wl = DiskIOWorkload({"block_size_kb": 16, "write_rate_kbps": 1024, "duration": 1})
        wl.start()
        self.assertTrue(wl.is_running)
        time.sleep(0.1)
        wl.stop()
        self.assertFalse(wl.is_running)
        self.assertIsNone(wl._temp_file)  # Cleaned up temp file

    def test_network_workload_bounds(self):
        with self.assertRaises(ValueError):
            NetworkIntensiveWorkload({"port": 70000})
        with self.assertRaises(ValueError):
            NetworkIntensiveWorkload({"throughput_kbps": -1})

        wl = NetworkIntensiveWorkload({"target_host": "127.0.0.1", "port": 65500, "duration": 1})
        wl.start()
        self.assertTrue(wl.is_running)
        time.sleep(0.1)
        wl.stop()
        self.assertFalse(wl.is_running)

    def test_mixed_workload_lifecycle(self):
        wl = MixedWorkload({"duration": 2, "seed": 42})
        wl.start()
        self.assertTrue(wl.is_running)
        time.sleep(0.1)
        h = wl.health()
        self.assertIn("sub_workloads", h)
        self.assertEqual(len(h["sub_workloads"]), 3)
        wl.stop()
        self.assertFalse(wl.is_running)

    def test_workload_factory_creation(self):
        self.assertIsInstance(WorkloadFactory.create("idle", {"duration": 10}), IdleWorkload)
        self.assertIsInstance(WorkloadFactory.create("cpu", {"intensity": 50, "duration": 10}), CPUIntensiveWorkload)
        self.assertIsInstance(WorkloadFactory.create("memory", {"target_mb": 64, "chunk_mb": 16, "duration": 10}), MemoryIntensiveWorkload)
        self.assertIsInstance(WorkloadFactory.create("disk", {"block_size_kb": 16, "write_rate_kbps": 1024, "duration": 10}), DiskIOWorkload)
        self.assertIsInstance(WorkloadFactory.create("network", {"duration": 10}), NetworkIntensiveWorkload)
        self.assertIsInstance(WorkloadFactory.create("mixed", {"duration": 10}), MixedWorkload)

        with self.assertRaises(ValueError):
            WorkloadFactory.create("unknown_profile")

    def test_reproducible_randomized_generation(self):
        cfg = TelemetryCollectionConfig()
        
        # Fixed seed should generate identical workloads
        w1 = WorkloadFactory.create_randomized(cfg, seed=12345, duration=30)
        w2 = WorkloadFactory.create_randomized(cfg, seed=12345, duration=30)
        self.assertEqual(w1.name, w2.name)
        self.assertEqual(w1.params, w2.params)


if __name__ == "__main__":
    unittest.main()
