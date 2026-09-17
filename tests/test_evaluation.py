import unittest
from evaluation.metrics import compute_metrics
from evaluation.runner import run_evaluation_suite
from evaluation.baselines import StaticController, ThresholdController, PredictionOnlyController

class TestEvaluation(unittest.TestCase):
    def test_metrics_empty(self):
        m = compute_metrics([])
        self.assertEqual(m, {})
        
    def test_metrics_calculation(self):
        records = [
            {"current_cpu_alloc": 2.0, "current_cpu": 100.0, "sla_violation": False},
            {"current_cpu_alloc": 2.0, "current_cpu": 50.0, "sla_violation": True}
        ]
        m = compute_metrics(records)
        self.assertEqual(m["avg_cpu_allocation"], 2.0)
        self.assertEqual(m["avg_cpu_utilization"], 75.0)
        self.assertEqual(m["cpu_wastage"], 125.0)
        self.assertEqual(m["sla_violations"], 1)
        
    def test_baselines(self):
        sc = StaticController(2.0, 512)
        c, m = sc.decide({})
        self.assertEqual(c, 2.0)
        
        tc = ThresholdController()
        c, m = tc.decide({"current_cpu": 90.0, "current_memory": 900.0}, 1.0, 1024)
        self.assertEqual(c, 1.5)
        
    def test_evaluation_suite(self):
        trace = [{"cpu_util": 50.0, "mem_util": 256, "pred_cpu": 50.0, "pred_mem": 256}]
        res = run_evaluation_suite("test", trace)
        self.assertEqual(len(res), 4) # 4 strategies

