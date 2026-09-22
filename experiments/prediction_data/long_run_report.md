# Long Run Report

A. Exact sample count: 5000
B. Exact duration: 24996.53 seconds
C. Unique container IDs: 1
D. Sampling interval statistics:
   - mean: 5.0003
   - std: 0.0126
   - min: 4.9920
   - max: 5.8841
E. CPU statistics:
   - mean: 29.86%
F. Memory statistics:
   - mean: 234.80 MB
G. cpu_usage_ns_delta statistics:
   - mean: 1485754097.60
   - negative count: 0
H. Workload distribution:
{
  "idle": 2762,
  "cpu": 560,
  "memory": 561,
  "burst": 562,
  "mixed": 555
}
I. Transition counts: 111
J. Transition examples: (idle -> cpu, etc.)
K. 12->1 window counts: 4988
L. 12->3 window counts: 4986
M. 12->6 window counts: 4983
N. Transition-containing window counts (12->1): 1332
O. Chronological train/validation/test split proposal:
   Train: 0 to 3500
   Val: 3500 to 4250
   Test: 4250 to 5000
P. Usable window counts after split (12->6):
   Train: 3483
   Val: 733
   Test: 733
Q. Data-quality assessment: Timestamps strictly increasing = True. Zero negative deltas = True.
R. Remaining limitations: The memory step behavior limits conclusions about gradual memory forecasting.
S. Recommendation on whether the dataset is ready for model benchmarking:
   Yes, 5000 samples provide a substantial basis for comparing Transformer, XGBoost, and GRU models. 
   More data might slightly improve generalization, but 5000 is sufficient for an initial robust benchmark.
