import xgboost as xgb
import numpy as np
import os
import json

def flatten_inputs(X):
    # X shape: (batch, seq_len, features) -> (batch, seq_len * features)
    return X.reshape(X.shape[0], -1)

def train_xgboost(X_train, y_train, X_val, y_val, seed=42):
    X_train_flat = flatten_inputs(X_train)
    X_val_flat = flatten_inputs(X_val)
    
    # y_train has 2 targets. XGBoost regression handles single targets.
    # We will train 2 models.
    
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "early_stopping_rounds": 10,
        "objective": "reg:squarederror"
    }
    
    # Train CPU model (index 0)
    model_cpu = xgb.XGBRegressor(**params)
    model_cpu.fit(
        X_train_flat, y_train[:, 0],
        eval_set=[(X_val_flat, y_val[:, 0])],
        verbose=False
    )
    
    # Train Memory model (index 1)
    model_mem = xgb.XGBRegressor(**params)
    model_mem.fit(
        X_train_flat, y_train[:, 1],
        eval_set=[(X_val_flat, y_val[:, 1])],
        verbose=False
    )
    
    os.makedirs("experiments/prediction_models/checkpoints", exist_ok=True)
    model_cpu.save_model("experiments/prediction_models/checkpoints/xgboost_real_cpu.json")
    model_mem.save_model("experiments/prediction_models/checkpoints/xgboost_real_memory.json")
    
    return model_cpu, model_mem

def predict_xgboost(model_cpu, model_mem, X):
    X_flat = flatten_inputs(X)
    pred_cpu = model_cpu.predict(X_flat)
    pred_mem = model_mem.predict(X_flat)
    return np.column_stack((pred_cpu, pred_mem))
