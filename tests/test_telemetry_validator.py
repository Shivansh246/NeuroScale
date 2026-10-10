"""tests/test_telemetry_validator.py — Phase 8D unit tests.

All tests use stdlib only (no numpy / torch / docker SDK).

Test matrix
-----------
1.  valid_record_accepted          — happy-path full record passes every rule
2.  missing_required_field         — record lacking a required field is rejected
3.  invalid_type_non_numeric       — non-numeric cpu_percent is rejected
4.  negative_cpu_percent           — cpu_percent < 0 is rejected
5.  negative_optional_counter      — network_rx_bytes < 0 is rejected
6.  invalid_workload_mode          — unknown mode string is rejected
7.  future_timestamp               — timestamp far in the future is rejected
8.  non_monotonic_timestamp        — decreasing timestamp for same container rejected
9.  restart_detection              — cpu_usage_ns decrease flagged (warning, not error)
10. restart_does_not_modify_value  — raw cpu_usage_ns unchanged after restart
11. multiple_containers_independent— non-monotonic ts for one container does not
                                      affect the other
12. summary_statistics             — accepted/rejected/restart/missing counters correct
"""

import sys
import os
import time
import unittest

# Ensure project root is on sys.path so that 'data' and 'telemetry' are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from telemetry.validator import TelemetryValidator, ValidationResult, ValidationSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> dict:
    """Build a minimal valid telemetry record, applying any keyword overrides."""
    base = {
        "timestamp": time.time() - 1.0,   # 1 s in the past — always valid
        "container_id": "abc123",
        "container_name": "test_container",
        "workload_mode": "cpu",
        "cpu_percent": 25.0,
        "cpu_usage_ns": 1_000_000_000,
        "memory_usage_bytes": 104_857_600,
        "memory_usage_mb": 100.0,
        "memory_limit_bytes": 2_147_483_648,
        "memory_limit_mb": 2048.0,
        "memory_percent": 4.88,
        "network_rx_bytes": 0,
        "network_tx_bytes": 0,
        "block_read_bytes": 0,
        "block_write_bytes": 0,
        "pids": 1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestTelemetryValidator(unittest.TestCase):

    def setUp(self):
        """Create a fresh validator before each test."""
        self.v = TelemetryValidator(future_tolerance_seconds=5.0)

    # ------------------------------------------------------------------
    # 1. Happy path
    # ------------------------------------------------------------------

    def test_01_valid_record_accepted(self):
        rec = _make_record()
        result = self.v.validate(rec)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.errors, [])
        self.assertFalse(result.restart_detected)

    # ------------------------------------------------------------------
    # 2. Missing required field
    # ------------------------------------------------------------------

    def test_02_missing_required_field(self):
        rec = _make_record()
        del rec["cpu_percent"]
        result = self.v.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(any("Missing required fields" in e for e in result.errors),
                        result.errors)
        self.assertEqual(self.v.summary.missing_fields, 1)

    # ------------------------------------------------------------------
    # 3. Invalid type (non-numeric cpu_percent)
    # ------------------------------------------------------------------

    def test_03_invalid_type_non_numeric(self):
        rec = _make_record(cpu_percent="high")
        result = self.v.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(result.errors)

    # ------------------------------------------------------------------
    # 4. Negative required numeric field
    # ------------------------------------------------------------------

    def test_04_negative_cpu_percent(self):
        rec = _make_record(cpu_percent=-1.0)
        result = self.v.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(any("cpu_percent" in e for e in result.errors),
                        result.errors)

    # ------------------------------------------------------------------
    # 5. Negative optional counter
    # ------------------------------------------------------------------

    def test_05_negative_optional_counter(self):
        rec = _make_record(network_rx_bytes=-500)
        result = self.v.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("network_rx_bytes" in e for e in result.errors),
            result.errors,
        )

    # ------------------------------------------------------------------
    # 6. Invalid workload_mode
    # ------------------------------------------------------------------

    def test_06_invalid_workload_mode(self):
        rec = _make_record(workload_mode="turbo")
        result = self.v.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("workload_mode" in e for e in result.errors),
            result.errors,
        )

    # ------------------------------------------------------------------
    # 7. Future timestamp
    # ------------------------------------------------------------------

    def test_07_future_timestamp(self):
        rec = _make_record(timestamp=time.time() + 60.0)  # 60 s ahead
        result = self.v.validate(rec)
        self.assertFalse(result.valid)
        self.assertTrue(
            any("future" in e.lower() or "in the future" in e for e in result.errors),
            result.errors,
        )

    # ------------------------------------------------------------------
    # 8. Non-monotonic timestamp for same container
    # ------------------------------------------------------------------

    def test_08_non_monotonic_timestamp(self):
        now = time.time() - 10.0
        rec1 = _make_record(timestamp=now, container_id="ctr-A")
        rec2 = _make_record(timestamp=now - 2.0, container_id="ctr-A")  # earlier
        r1 = self.v.validate(rec1)
        r2 = self.v.validate(rec2)
        self.assertTrue(r1.valid, r1.errors)
        self.assertFalse(r2.valid)
        self.assertTrue(
            any("Non-monotonic" in e or "non-monotonic" in e.lower() for e in r2.errors),
            r2.errors,
        )

    # ------------------------------------------------------------------
    # 9. Restart detection — warning, record still valid
    # ------------------------------------------------------------------

    def test_09_restart_detection(self):
        cid = "ctr-restart"
        now = time.time() - 10.0
        rec1 = _make_record(timestamp=now,       container_id=cid, cpu_usage_ns=5_000_000_000)
        rec2 = _make_record(timestamp=now + 1.0, container_id=cid, cpu_usage_ns=1_000)  # reset
        r1 = self.v.validate(rec1)
        r2 = self.v.validate(rec2)
        self.assertTrue(r1.valid, r1.errors)
        self.assertTrue(r2.valid, r2.errors)         # still valid
        self.assertTrue(r2.restart_detected)         # restart flagged
        self.assertTrue(r2.warnings)                 # warning emitted
        self.assertEqual(self.v.summary.container_restarts, 1)

    # ------------------------------------------------------------------
    # 10. Restart does NOT modify the raw value
    # ------------------------------------------------------------------

    def test_10_restart_does_not_modify_value(self):
        cid = "ctr-val"
        now = time.time() - 10.0
        original_cpu_ns = 999
        rec1 = _make_record(timestamp=now,       container_id=cid, cpu_usage_ns=5_000_000_000)
        rec2 = _make_record(timestamp=now + 1.0, container_id=cid, cpu_usage_ns=original_cpu_ns)
        self.v.validate(rec1)
        r2 = self.v.validate(rec2)
        # The value in the result record must be the original, unmodified value.
        self.assertEqual(r2.record["cpu_usage_ns"], original_cpu_ns)

    # ------------------------------------------------------------------
    # 11. Multiple containers — per-container independence
    # ------------------------------------------------------------------

    def test_11_multiple_containers_independent(self):
        """Non-monotonic timestamp for container A must not affect container B."""
        now = time.time() - 10.0
        # Container A: forward in time
        recA1 = _make_record(timestamp=now,       container_id="A", cpu_usage_ns=1_000)
        recA2 = _make_record(timestamp=now + 1.0, container_id="A", cpu_usage_ns=2_000)
        # Container B: starts at a completely different time (but monotonically increasing)
        recB1 = _make_record(timestamp=now + 5.0, container_id="B", cpu_usage_ns=1_000)
        recB2 = _make_record(timestamp=now + 6.0, container_id="B", cpu_usage_ns=2_000)

        for rec in [recA1, recA2, recB1, recB2]:
            r = self.v.validate(rec)
            self.assertTrue(r.valid, f"Unexpected failure: {r.errors}")

    # ------------------------------------------------------------------
    # 12. Summary statistics correctness
    # ------------------------------------------------------------------

    def test_12_summary_statistics(self):
        now = time.time() - 10.0
        cid = "ctr-sum"

        # Record 1: valid
        self.v.validate(_make_record(timestamp=now,       container_id=cid, cpu_usage_ns=1_000))
        # Record 2: valid + restart warning
        self.v.validate(_make_record(timestamp=now + 1.0, container_id=cid, cpu_usage_ns=500))
        # Record 3: missing field
        rec_missing = _make_record(timestamp=now + 2.0, container_id=cid, cpu_usage_ns=600)
        del rec_missing["memory_percent"]
        self.v.validate(rec_missing)
        # Record 4: valid again (cpu_usage_ns continues from 500)
        self.v.validate(_make_record(timestamp=now + 3.0, container_id=cid, cpu_usage_ns=700))

        s = self.v.get_summary()
        self.assertEqual(s.accepted, 3)      # records 1, 2, 4
        self.assertEqual(s.rejected, 1)      # record 3
        self.assertEqual(s.missing_fields, 1)
        self.assertEqual(s.container_restarts, 1)
        self.assertGreater(s.warnings_count, 0)


if __name__ == "__main__":
    unittest.main()
