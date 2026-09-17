import unittest
from workload.workload import (
    build_parser,
    run_workload,
    allocate_memory,
    idle_work,
    cpu_work,
    memory_work,
    burst_work,
    mixed_work,
)


class TestWorkload(unittest.TestCase):

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.mode, "cpu")
        self.assertEqual(args.duration, 60)
        self.assertEqual(args.memory, 256)
        self.assertEqual(args.burst_active, 2.0)
        self.assertEqual(args.burst_idle, 1.0)

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--mode", "burst",
            "--duration", "120",
            "--memory", "512",
            "--burst-on", "3.5",
            "--burst-off", "1.5",
        ])
        self.assertEqual(args.mode, "burst")
        self.assertEqual(args.duration, 120)
        self.assertEqual(args.memory, 512)
        self.assertEqual(args.burst_active, 3.5)
        self.assertEqual(args.burst_idle, 1.5)

    def test_allocate_memory(self):
        buf = allocate_memory(2)  # 2MB
        self.assertEqual(len(buf), 2 * 1024 * 1024)

    def test_run_workload_modes_execution(self):
        # Micro durations for quick test runs
        run_workload(mode="idle", duration=0.05)
        run_workload(mode="cpu", duration=0.05)
        run_workload(mode="memory", duration=0.05, memory_mb=1)
        run_workload(mode="burst", duration=0.05, burst_active=0.02, burst_idle=0.01)
        run_workload(mode="mixed", duration=0.05, memory_mb=1)

    def test_run_workload_invalid_mode(self):
        with self.assertRaises(ValueError):
            run_workload(mode="quantum_supercompute", duration=0.01)


if __name__ == "__main__":
    unittest.main()
