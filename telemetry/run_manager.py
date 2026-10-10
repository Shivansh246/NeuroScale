"""Dataset run manager for experiment versioning, directory layout, and dataset syncing."""

from pathlib import Path
from typing import Optional
import datetime
import os
import shutil
import re


class DatasetRunManager:
    """Manages versioned experiment runs (datasets/run_001, run_002, ...) and symlinks."""

    def __init__(self, base_dir: Path | str = "datasets") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_run_id(self) -> str:
        """Determine the next sequential run ID (e.g. run_001, run_002)."""
        existing_indices = []
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    match = re.match(r"^run_(\d+)", item.name)
                    if match:
                        existing_indices.append(int(match.group(1)))

        next_idx = max(existing_indices, default=0) + 1
        return f"run_{next_idx:03d}"

    def create_run_directory(self, run_name: Optional[str] = None) -> Path:
        """Create a new versioned experiment run directory with subfolders.
        
        Directory layout:
            datasets/run_001_YYYYMMDD_HHMMSS/
                logs/
                metadata/
                exports/
        """
        run_id = self.get_next_run_id()
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_name}" if run_name else ""
        folder_name = f"{run_id}_{timestamp}{suffix}"
        
        run_path = self.base_dir / folder_name
        (run_path / "logs").mkdir(parents=True, exist_ok=True)
        (run_path / "metadata").mkdir(parents=True, exist_ok=True)
        (run_path / "exports").mkdir(parents=True, exist_ok=True)

        return run_path

    def update_latest_symlink(self, run_path: Path) -> Path:
        """Update or create the 'datasets/latest' pointer pointing to the current run path."""
        latest_link = self.base_dir / "latest"
        
        # On POSIX systems, create a relative symlink; fall back safely if Windows or error
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            # Relative target makes repo relocatable
            target_rel = run_path.name
            latest_link.symlink_to(target_rel)
        except (OSError, NotImplementedError):
            # Fallback text pointer if symlinks not supported by OS/filesystem
            pointer_file = self.base_dir / "LATEST_RUN.txt"
            pointer_file.write_text(str(run_path.resolve()), encoding="utf-8")

        return latest_link

    def sync_to_data_raw_metrics(
        self,
        source_jsonl: Path | str,
        target_path: Path | str = "data/raw_metrics.jsonl",
    ) -> Path:
        """Copy the validated JSONL dataset to data/raw_metrics.jsonl for seamless ML training."""
        src = Path(source_jsonl).resolve()
        dst = Path(target_path).resolve()

        if not src.is_file():
            raise FileNotFoundError(f"Source JSONL file does not exist: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst
