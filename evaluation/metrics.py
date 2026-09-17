"""Evaluation metric calculation from trace logs."""

import numpy as np
from typing import List, Dict, Any

def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
        
    cpu_allocs = [r.get("current_cpu_alloc", 0.0) for r in records if "current_cpu_alloc" in r]
    mem_allocs = [r.get("current_mem_alloc", 0.0) for r in records if "current_mem_alloc" in r]
    
    cpu_utils = [r.get("current_cpu", 0.0) for r in records if "current_cpu" in r]
    mem_utils = [r.get("current_memory", 0.0) for r in records if "current_memory" in r]
    
    # Calculate wastage: Allocation - Utilization
    # CPU util is in %, allocation is in cores. 1 core = 100%
    cpu_wastage = []
    for alloc, util in zip(cpu_allocs, cpu_utils):
        wastage = max(0.0, (alloc * 100.0) - util)
        cpu_wastage.append(wastage)
        
    mem_wastage = []
    for alloc, util in zip(mem_allocs, mem_utils):
        wastage = max(0.0, alloc - util)
        mem_wastage.append(wastage)
        
    sla_violations = [r.get("sla_violation", False) for r in records if "sla_violation" in r]
    
    # Count reallocations
    reallocations = 0
    for i in range(1, len(records)):
        prev = records[i-1]
        curr = records[i]
        if (prev.get("current_cpu_alloc") != curr.get("current_cpu_alloc") or 
            prev.get("current_mem_alloc") != curr.get("current_mem_alloc")):
            reallocations += 1
            
    avg_control_latency = np.mean([r.get("control_latency_ms", 0.0) for r in records if "control_latency_ms" in r])
    
    # ML metrics (optional)
    cpu_errors = []
    mem_errors = []
    for r in records:
        if "predicted_cpu" in r and "current_cpu" in r:
            cpu_errors.append(abs(r["predicted_cpu"] - r["current_cpu"]))
        if "predicted_memory" in r and "current_memory" in r:
            mem_errors.append(abs(r["predicted_memory"] - r["current_memory"]))
            
    metrics = {
        "duration": len(records),
        "avg_cpu_allocation": float(np.mean(cpu_allocs)) if cpu_allocs else 0.0,
        "avg_memory_allocation": float(np.mean(mem_allocs)) if mem_allocs else 0.0,
        "avg_cpu_utilization": float(np.mean(cpu_utils)) if cpu_utils else 0.0,
        "avg_memory_utilization": float(np.mean(mem_utils)) if mem_utils else 0.0,
        "cpu_wastage": float(np.mean(cpu_wastage)) if cpu_wastage else 0.0,
        "memory_wastage": float(np.mean(mem_wastage)) if mem_wastage else 0.0,
        "sla_violations": sum(sla_violations),
        "sla_violation_rate": sum(sla_violations) / len(sla_violations) if sla_violations else 0.0,
        "reallocations": reallocations,
        "successful_reallocations": reallocations, # simplified
        "failed_reallocations": 0,
        "avg_control_latency_ms": float(avg_control_latency) if not np.isnan(avg_control_latency) else 0.0,
        "reward": float(np.mean([r.get("reward", 0.0) for r in records if "reward" in r]))
    }
    
    if cpu_errors:
        metrics["transformer_cpu_mae"] = float(np.mean(cpu_errors))
        metrics["transformer_cpu_rmse"] = float(np.sqrt(np.mean(np.square(cpu_errors))))
    if mem_errors:
        metrics["transformer_memory_mae"] = float(np.mean(mem_errors))
        metrics["transformer_memory_rmse"] = float(np.sqrt(np.mean(np.square(mem_errors))))
        
    return metrics
