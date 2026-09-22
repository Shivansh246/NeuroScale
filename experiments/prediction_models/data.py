import json
import numpy as np
import os
import pickle

def load_dataset(file_path):
    records = []
    with open(file_path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def process_features(records):
    # Features: cpu_percent, cpu_usage_ns_delta, memory_usage_mb, memory_percent
    features = []
    targets = []
    modes = []
    timestamps = []
    
    for r in records:
        c_pct = r.get("cpu_percent", 0.0)
        c_delta = r.get("cpu_usage_ns_delta", 0.0)
        # Apply log1p to cpu_usage_ns_delta as it is highly skewed
        c_delta_log = np.log1p(max(0, c_delta))
        m_mb = r.get("memory_usage_mb", 0.0)
        m_pct = r.get("memory_percent", 0.0)
        
        features.append([c_pct, c_delta_log, m_mb, m_pct])
        targets.append([c_pct, m_mb])
        modes.append(r.get("workload_mode", "unknown"))
        timestamps.append(r.get("timestamp", 0.0))
        
    return np.array(features), np.array(targets), modes, timestamps

def create_windows(features, targets, modes, timestamps, seq_len=12, horizon=1):
    X, y = [], []
    y_modes = []
    y_timestamps = []
    window_modes = []
    
    num_samples = len(features)
    for i in range(num_samples - seq_len - horizon + 1):
        X.append(features[i:i+seq_len])
        y.append(targets[i+seq_len+horizon-1])
        y_modes.append(modes[i+seq_len+horizon-1])
        y_timestamps.append(timestamps[i+seq_len+horizon-1])
        window_modes.append(modes[i:i+seq_len+horizon])
        
    return np.array(X), np.array(y), y_modes, y_timestamps, window_modes

def chronological_split(X, y, y_modes, y_timestamps, window_modes, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # We must avoid temporal leakage. By iterating `i` up to `n - seq_len - horizon + 1`, 
    # the target of window `train_end - 1` is at index `train_end - 1 + seq_len + horizon - 1`.
    # The next window `train_end` starts at index `train_end`. 
    # This means its input overlaps with the target of the previous window? 
    # Yes, input[0] of window `train_end` is `train_end`.
    # Target of `train_end-1` is `train_end-1 + 12 = train_end + 11`.
    # To truly avoid leakage, the first input of val should be AFTER the last target of train.
    # Last target of train is at index `train_end - 1 + 12 = train_end + 11`.
    # So first input of val should be `train_end + 12`.
    # This means we drop seq_len+horizon-1 windows at the boundary.
    
    seq_len = X.shape[1]
    horizon = 1 # We can make this dynamic if needed
    gap = seq_len + horizon - 1
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end + gap, val_end))
    test_indices = list(range(val_end + gap, n))
    
    return {
        "train": (X[train_indices], y[train_indices], [y_modes[i] for i in train_indices], [y_timestamps[i] for i in train_indices], [window_modes[i] for i in train_indices]),
        "val": (X[val_indices], y[val_indices], [y_modes[i] for i in val_indices], [y_timestamps[i] for i in val_indices], [window_modes[i] for i in val_indices]),
        "test": (X[test_indices], y[test_indices], [y_modes[i] for i in test_indices], [y_timestamps[i] for i in test_indices], [window_modes[i] for i in test_indices])
    }

class Normalizer:
    def __init__(self):
        self.feat_mean = None
        self.feat_std = None
        self.target_mean = None
        self.target_std = None

    def fit(self, X_train, y_train):
        self.feat_mean = np.mean(X_train, axis=(0, 1))
        self.feat_std = np.std(X_train, axis=(0, 1))
        self.feat_std[self.feat_std == 0] = 1e-6
        
        self.target_mean = np.mean(y_train, axis=0)
        self.target_std = np.std(y_train, axis=0)
        self.target_std[self.target_std == 0] = 1e-6

    def transform_x(self, X):
        return (X - self.feat_mean) / self.feat_std

    def transform_y(self, y):
        return (y - self.target_mean) / self.target_std

    def inverse_transform_y(self, y_norm):
        return y_norm * self.target_std + self.target_mean
        
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

def get_prepared_data(file_path="data/prediction_experiment/real_workload_long_5000.jsonl", seq_len=12, horizon=1):
    records = load_dataset(file_path)
    features, targets, modes, timestamps = process_features(records)
    
    X, y, y_modes, y_timestamps, window_modes = create_windows(features, targets, modes, timestamps, seq_len, horizon)
    
    splits = chronological_split(X, y, y_modes, y_timestamps, window_modes)
    
    norm = Normalizer()
    norm.fit(splits["train"][0], splits["train"][1])
    
    norm.save("experiments/prediction_models/checkpoints/normalizer.pkl")
    
    # Apply normalization
    out = {}
    for split_name in ["train", "val", "test"]:
        X_s, y_s, modes_s, times_s, win_modes_s = splits[split_name]
        X_norm = norm.transform_x(X_s)
        y_norm = norm.transform_y(y_s)
        out[split_name] = (X_norm, y_norm, modes_s, times_s, win_modes_s)
        
    return out, norm, records
