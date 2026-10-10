"""Base abstract interface for NeuroScale workload profiles."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time


class BaseWorkload(ABC):
    """Abstract base class that all workload profiles must implement."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.params: Dict[str, Any] = {}
        self.is_running: bool = False
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self._last_error: Optional[str] = None

        if params:
            self.configure(params)

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validate input parameters before configuring. Raises ValueError if invalid."""
        pass

    def configure(self, params: Dict[str, Any]) -> None:
        """Validate and apply workload parameters."""
        self.validate_params(params)
        self.params.update(params)

    @abstractmethod
    def start(self) -> None:
        """Start workload execution in background worker thread(s)/process(es)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop workload execution and clean up allocated resources."""
        pass

    def health(self) -> Dict[str, Any]:
        """Return runtime health status of the workload."""
        elapsed = 0.0
        if self.start_time:
            end = self.stop_time or time.time()
            elapsed = max(0.0, end - self.start_time)

        return {
            "name": self.name,
            "is_running": self.is_running,
            "elapsed_seconds": round(elapsed, 2),
            "last_error": self._last_error,
            "healthy": self._last_error is None,
        }

    def metadata(self) -> Dict[str, Any]:
        """Return runtime configuration and diagnostic metadata."""
        return {
            "name": self.name,
            "params": dict(self.params),
            "health": self.health(),
        }
