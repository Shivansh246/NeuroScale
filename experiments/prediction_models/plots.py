import matplotlib.pyplot as plt
import os

def generate_plots(y_true, y_pred_trans, y_pred_gru, y_pred_xgb, timestamps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Actual vs predicted CPU for a representative test segment
    # Let's take the first 100 samples
    idx = slice(0, 150)
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[idx, 0], label='Actual CPU', color='black', linewidth=2)
    plt.plot(y_pred_trans[idx, 0], label='Transformer', alpha=0.7)
    plt.plot(y_pred_gru[idx, 0], label='GRU', alpha=0.7)
    plt.plot(y_pred_xgb[idx, 0], label='XGBoost', alpha=0.7)
    plt.title("Actual vs Predicted CPU% (Representative Test Segment)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "cpu_predictions.png"))
    plt.close()
    
    # 2. Actual vs predicted Memory for the same segment
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[idx, 1], label='Actual Memory', color='black', linewidth=2)
    plt.plot(y_pred_trans[idx, 1], label='Transformer', alpha=0.7)
    plt.plot(y_pred_gru[idx, 1], label='GRU', alpha=0.7)
    plt.plot(y_pred_xgb[idx, 1], label='XGBoost', alpha=0.7)
    plt.title("Actual vs Predicted Memory MB (Representative Test Segment)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "memory_predictions.png"))
    plt.close()
    
    # 5. Error distribution by model
    # CPU Error
    err_trans = y_true[:, 0] - y_pred_trans[:, 0]
    err_gru = y_true[:, 0] - y_pred_gru[:, 0]
    err_xgb = y_true[:, 0] - y_pred_xgb[:, 0]
    
    plt.figure(figsize=(10, 4))
    plt.hist(err_trans, bins=50, alpha=0.5, label='Transformer')
    plt.hist(err_gru, bins=50, alpha=0.5, label='GRU')
    plt.hist(err_xgb, bins=50, alpha=0.5, label='XGBoost')
    plt.title("CPU Error Distribution (Actual - Predicted)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "cpu_error_dist.png"))
    plt.close()
