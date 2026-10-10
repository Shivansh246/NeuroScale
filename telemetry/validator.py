"""telemetry/validator.py — Phase 8D: Telemetry Validation Layer.

Responsibilities
----------------
* Call ``data.validation.validate_record()`` (the existing ML-pipeline
  validator) as the first validation gate — **without modifying that module**.
* Add extended range checks for optional fields not covered by
  ``data/validation.py`` (network, blkio, pids).
* Enforce that ``workload_mode`` belongs to the controlled vocabulary.
* Enforce per-container timestamp monotonicity and reject future timestamps.
* Detect container restarts (``cpu_usage_ns`` decrease) and flag them in the
  result **without altering the raw value**.  Delta computation remains
  exclusively the responsibility of ``TelemetryFeaturePipeline``.

Public API
----------
``TelemetryValidator.validate(record)`` → ``ValidationResult``
``TelemetryValidator.get_summary()``    → ``ValidationSummary``
``TelemetryValidator.reset()``          → clears per-container state + summary
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import the existing ML-pipeline validator (must not be modified).
# ---------------------------------------------------------------------------
from data.validation import validate_record, REQUIRED_FIELDS  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Controlled vocabulary for workload_mode.
VALID_WORKLOAD_MODES: frozenset[str] = frozenset(
    {"idle", "cpu", "memory", "disk", "network", "mixed", "unknown"}
)

#: Optional numeric fields that may appear in collector output.
_OPTIONAL_NON_NEGATIVE_FIELDS: tuple[str, ...] = (
    "network_rx_bytes",
    "network_tx_bytes",
    "block_read_bytes",
    "block_write_bytes",
    "pids",
)

# ---------------------------------------------------------------------------
# Result / summary data-classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Per-record validation outcome.

    Attributes
    ----------
    valid : bool
        ``True`` when the record passes every validation rule and is safe to
        export.
    warnings : list[str]
        Non-fatal observations (e.g. a restart was detected but the value is
        preserved and the record is still exported).
    errors : list[str]
        Fatal problems that caused ``valid`` to be ``False``.
    restart_detected : bool
        ``True`` when ``cpu_usage_ns`` is smaller than the previous sample for
        the same container — indicating a cgroup counter reset / container
        restart.  The raw value is **never modified**.
    record : dict
        The original, unmodified record that was submitted for validation.
    """

    valid: bool
    warnings: List[str]
    errors: List[str]
    restart_detected: bool
    record: Dict[str, Any]


@dataclass
class ValidationSummary:
    """Aggregate validation statistics across all records processed so far.

    Attributes
    ----------
    accepted : int
        Records that passed all validation rules.
    rejected : int
        Records that failed at least one fatal rule.
    warnings_count : int
        Total warnings emitted (a single record may contribute many).
    container_restarts : int
        Total restart events detected across all containers.
    missing_fields : int
        Records rejected because one or more required fields were absent.
    type_mismatches : int
        Records rejected because a field had the wrong type.
    """

    accepted: int = 0
    rejected: int = 0
    warnings_count: int = 0
    container_restarts: int = 0
    missing_fields: int = 0
    type_mismatches: int = 0


# ---------------------------------------------------------------------------
# Main validator class
# ---------------------------------------------------------------------------


class TelemetryValidator:
    """Stateful validator for Docker telemetry records.

    State is kept **per container** so that timestamp monotonicity and restart
    detection work correctly when records from multiple containers are
    interleaved.

    Parameters
    ----------
    future_tolerance_seconds : float
        Timestamps farther ahead of ``time.time()`` than this value are
        rejected.  Defaults to 5.0 s to tolerate minor clock skew.
    """

    def __init__(self, future_tolerance_seconds: float = 5.0) -> None:
        self._future_tol: float = future_tolerance_seconds

        # Per-container state: keyed by container_id.
        self._prev_cpu_ns: Dict[str, int] = {}
        self._prev_timestamps: Dict[str, float] = {}

        self.summary: ValidationSummary = ValidationSummary()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, record: Dict[str, Any]) -> ValidationResult:
        """Validate a single telemetry record.

        Validation pipeline (in order):
        1. ``data.validation.validate_record()`` — required fields + basic
           numeric ranges as defined by the ML pipeline.
        2. Extended optional-field range checks (network, blkio, pids).
        3. ``workload_mode`` controlled-vocabulary check.
        4. Timestamp: NaN / infinity / future / per-container monotonicity.
        5. ``cpu_usage_ns`` decrease → restart detection (warning, not error).
        6. Summary statistics update.

        The *record* object is **never mutated**.  ``ValidationResult.record``
        always contains the original dict reference.

        Returns
        -------
        ValidationResult
            ``valid=True`` when the record should be exported; ``False``
            otherwise.
        """
        errors: List[str] = []
        warnings: List[str] = []
        restart_detected: bool = False

        # ----------------------------------------------------------------
        # Step 1: ML-pipeline gate — data.validation.validate_record()
        # ----------------------------------------------------------------
        base_valid, missing_field_error, type_error = self._run_base_validation(
            record, errors
        )

        # ----------------------------------------------------------------
        # Step 2: Extended optional-field numeric ranges
        # ----------------------------------------------------------------
        if base_valid:
            self._check_optional_numeric_fields(record, errors)

        # ----------------------------------------------------------------
        # Step 3: workload_mode controlled vocabulary
        # ----------------------------------------------------------------
        if base_valid:
            self._check_workload_mode(record, errors)

        # ----------------------------------------------------------------
        # Step 4: Timestamp validation
        # ----------------------------------------------------------------
        container_id: Optional[str] = record.get("container_id")
        if base_valid:
            self._check_timestamp(record, container_id, errors)

        # ----------------------------------------------------------------
        # Step 5: cpu_usage_ns restart detection
        # ----------------------------------------------------------------
        if base_valid and container_id is not None:
            restart_detected = self._check_restart(record, container_id, warnings)

        # ----------------------------------------------------------------
        # Step 6: Update per-container state (only on valid records)
        # ----------------------------------------------------------------
        valid = len(errors) == 0
        if valid and container_id is not None:
            self._update_container_state(record, container_id)

        # ----------------------------------------------------------------
        # Step 7: Update aggregate summary
        # ----------------------------------------------------------------
        self._update_summary(
            valid=valid,
            warnings=warnings,
            restart_detected=restart_detected,
            missing_field_error=missing_field_error,
            type_error=type_error,
        )

        return ValidationResult(
            valid=valid,
            warnings=warnings,
            errors=errors,
            restart_detected=restart_detected,
            record=record,
        )

    def get_summary(self) -> ValidationSummary:
        """Return a copy of the current aggregate validation summary."""
        return ValidationSummary(
            accepted=self.summary.accepted,
            rejected=self.summary.rejected,
            warnings_count=self.summary.warnings_count,
            container_restarts=self.summary.container_restarts,
            missing_fields=self.summary.missing_fields,
            type_mismatches=self.summary.type_mismatches,
        )

    def reset(self) -> None:
        """Clear all per-container state and reset aggregate statistics."""
        self._prev_cpu_ns.clear()
        self._prev_timestamps.clear()
        self.summary = ValidationSummary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_base_validation(
        self,
        record: Dict[str, Any],
        errors: List[str],
    ) -> tuple[bool, bool, bool]:
        """Invoke ``data.validation.validate_record()`` and classify errors.

        Returns
        -------
        (base_valid, missing_field_error, type_error)
        """
        missing_field_error = False
        type_error = False
        try:
            validate_record(record)
            return True, False, False
        except ValueError as exc:
            msg = str(exc)
            errors.append(f"[base] {msg}")
            # Classify the root cause for summary counters.
            if "Missing required fields" in msg:
                missing_field_error = True
            elif "Invalid value" in msg or "Invalid workload_mode" in msg:
                # data/validation.py raises "Invalid value for <field>" for
                # type mismatches (non-numeric values pass the _is_number check
                # as False → same error path as negative values).  We treat
                # "Invalid value" as a type/range mismatch for counter purposes.
                type_error = True
            return False, missing_field_error, type_error

    @staticmethod
    def _is_number(value: Any) -> bool:
        """Return True for int/float but not bool."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _check_optional_numeric_fields(
        self,
        record: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """Validate optional collector fields not covered by data/validation.py."""
        for fname in _OPTIONAL_NON_NEGATIVE_FIELDS:
            if fname not in record:
                continue  # Optional — absence is allowed.
            val = record[fname]
            if not self._is_number(val):
                errors.append(
                    f"[range] '{fname}' must be numeric, got {type(val).__name__}: {val!r}"
                )
            elif val < 0:
                errors.append(
                    f"[range] '{fname}' must be >= 0, got {val}"
                )

    @staticmethod
    def _check_workload_mode(
        record: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """Enforce controlled vocabulary for workload_mode."""
        mode = record.get("workload_mode", "")
        # data/validation.py already confirmed it is a non-empty string.
        if mode not in VALID_WORKLOAD_MODES:
            errors.append(
                f"[workload_mode] '{mode}' is not in the approved set "
                f"{sorted(VALID_WORKLOAD_MODES)}"
            )

    def _check_timestamp(
        self,
        record: Dict[str, Any],
        container_id: Optional[str],
        errors: List[str],
    ) -> None:
        """Validate timestamp for NaN, Inf, future, and per-container monotonicity."""
        import math

        ts = record.get("timestamp")

        # data/validation.py already rejected non-numeric / negative timestamps
        # so we only need NaN / Inf / future / monotonicity here.
        if ts is None:
            return  # Already caught by base validation.

        if math.isnan(ts) or math.isinf(ts):
            errors.append(f"[timestamp] NaN or Inf timestamp: {ts}")
            return

        now = time.time()
        if ts > now + self._future_tol:
            errors.append(
                f"[timestamp] Timestamp {ts:.3f} is {ts - now:.1f}s in the "
                f"future (tolerance={self._future_tol}s)"
            )
            return

        if container_id is not None:
            prev_ts = self._prev_timestamps.get(container_id)
            if prev_ts is not None and ts < prev_ts:
                errors.append(
                    f"[timestamp] Non-monotonic timestamp for container "
                    f"'{container_id}': {ts:.3f} < previous {prev_ts:.3f}"
                )

    def _check_restart(
        self,
        record: Dict[str, Any],
        container_id: str,
        warnings: List[str],
    ) -> bool:
        """Detect a cpu_usage_ns counter reset (container restart).

        The raw value is NEVER modified.  A warning is appended and
        ``True`` is returned to signal the restart event.
        """
        cpu_ns = record.get("cpu_usage_ns")
        if cpu_ns is None or not self._is_number(cpu_ns):
            return False  # Already rejected by base validation.

        prev = self._prev_cpu_ns.get(container_id)
        if prev is not None and cpu_ns < prev:
            warnings.append(
                f"[restart] Container '{container_id}': cpu_usage_ns decreased "
                f"from {prev} to {cpu_ns} — counter reset / restart detected. "
                f"Raw value preserved; TelemetryFeaturePipeline handles delta."
            )
            return True
        return False

    def _update_container_state(
        self,
        record: Dict[str, Any],
        container_id: str,
    ) -> None:
        """Persist per-container state for the next sample."""
        ts = record.get("timestamp")
        if ts is not None and self._is_number(ts):
            self._prev_timestamps[container_id] = float(ts)

        cpu_ns = record.get("cpu_usage_ns")
        if cpu_ns is not None and self._is_number(cpu_ns):
            self._prev_cpu_ns[container_id] = int(cpu_ns)

    def _update_summary(
        self,
        valid: bool,
        warnings: List[str],
        restart_detected: bool,
        missing_field_error: bool,
        type_error: bool,
    ) -> None:
        """Atomically update aggregate counters."""
        if valid:
            self.summary.accepted += 1
        else:
            self.summary.rejected += 1
            if missing_field_error:
                self.summary.missing_fields += 1
            if type_error:
                self.summary.type_mismatches += 1

        self.summary.warnings_count += len(warnings)

        if restart_detected:
            self.summary.container_restarts += 1
