"""JSONL logger for control cycles."""

import json
import os
from typing import Dict, Any

class CycleLogger:
    """Appends structured cycle records to a JSONL file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def log(self, record: Dict[str, Any]):
        """Write a single cycle record as a JSON line."""
        try:
            with open(self.filepath, "a") as f:
                json.dump(record, f)
                f.write("\n")
        except Exception as e:
            # Do not crash the loop if logging fails
            print(f"[CycleLogger] Error writing to {self.filepath}: {e}")
