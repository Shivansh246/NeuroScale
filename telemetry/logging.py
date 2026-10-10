"""Structured logging infrastructure for the NeuroScale telemetry collection framework."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional


class StructuredLoggerFactory:
    """Manages dedicated timestamped loggers for collection, orchestration, and validation."""

    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: Path | str,
        filename: Optional[str] = None,
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ) -> logging.Logger:
        """Create or return a configured logger with safe file rotation and stdout streaming.
        
        Parameters
        ----------
        name : str
            Logger identifier (e.g. 'collector', 'orchestrator', 'validator').
        log_dir : Path or str
            Destination directory where logs will be stored.
        filename : str, optional
            Filename for the log file; defaults to f"{name}.log".
        level : int
            Logging level (e.g. logging.INFO, logging.DEBUG).
        max_bytes : int
            Threshold in bytes for rotating files.
        backup_count : int
            Number of rotated backup log files to retain.
        """
        key = f"{name}:{str(log_dir)}"
        if key in cls._loggers:
            return cls._loggers[key]

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        target_file = log_path / (filename or f"{name}.log")

        logger = logging.getLogger(f"telemetry.{name}")
        logger.setLevel(level)
        logger.propagate = False

        # Clear any preexisting handlers on this logger
        logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Rotating file handler
        file_handler = RotatingFileHandler(
            filename=str(target_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        cls._loggers[key] = logger
        return logger

    @classmethod
    def setup_run_loggers(cls, log_dir: Path | str) -> Dict[str, logging.Logger]:
        """Convenience helper to initialize collector, orchestrator, and validator loggers."""
        d = Path(log_dir)
        return {
            "collector": cls.get_logger("collector", d),
            "orchestrator": cls.get_logger("orchestrator", d),
            "validator": cls.get_logger("validator", d),
        }
