"""Baseline control strategies for evaluation."""

from typing import Dict, Any, Tuple

class StaticController:
    """Fixed CPU and memory allocation."""
    def __init__(self, cpu: float = 1.0, memory: int = 256):
        self.cpu = cpu
        self.memory = memory
        
    def decide(self, state: Dict[str, Any]) -> Tuple[float, int]:
        return self.cpu, self.memory

class ThresholdController:
    """Increase/decrease resources based on utilization thresholds."""
    def __init__(self, high_cpu=80.0, low_cpu=30.0, high_mem=80.0, low_mem=30.0):
        self.high_cpu = high_cpu
        self.low_cpu = low_cpu
        self.high_mem = high_mem
        self.low_mem = low_mem
        
    def decide(self, state: Dict[str, Any], current_cpu: float, current_mem: int) -> Tuple[float, int]:
        cpu_util = state.get("current_cpu", 0.0)
        mem_util_pct = (state.get("current_memory", 0.0) / current_mem * 100.0) if current_mem > 0 else 0.0
        
        target_cpu = current_cpu
        target_mem = current_mem
        
        if cpu_util > self.high_cpu:
            target_cpu = min(target_cpu + 0.5, 4.0)
        elif cpu_util < self.low_cpu:
            target_cpu = max(target_cpu - 0.5, 0.5)
            
        if mem_util_pct > self.high_mem:
            target_mem = min(target_mem * 2, 1024)
        elif mem_util_pct < self.low_mem:
            target_mem = max(target_mem // 2, 64)
            
        return target_cpu, target_mem

class PredictionOnlyController:
    """Use Transformer predictions with a safety margin, bypassing DQN."""
    def __init__(self, margin_multiplier: float = 1.2):
        self.margin = margin_multiplier
        
    def decide(self, state: Dict[str, Any]) -> Tuple[float, int]:
        pred_cpu = state.get("predicted_cpu", 0.0)
        pred_mem = state.get("predicted_memory", 0.0)
        
        target_cpu = max(0.5, min(pred_cpu * self.margin / 100.0, 4.0))
        target_mem = max(64, min(int(pred_mem * self.margin), 1024))
        
        # Round CPU to nearest 0.5 and mem to nearest power of 2 (or 64 multiple)
        target_cpu = round(target_cpu * 2) / 2
        target_mem = max(64, (target_mem // 64) * 64)
        
        return target_cpu, target_mem
