import torch
import torch.nn as nn
import os
from models.config import TransformerConfig
from models.transformer import NeuroScaleTransformer

def train_transformer(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=1e-3, patience=5, seed=42):
    torch.manual_seed(seed)
    
    cfg = TransformerConfig(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        prediction_horizon=1,
        num_targets=2,
        feature_count=4,
        seed=seed
    )
    
    model = NeuroScaleTransformer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_X_t = torch.FloatTensor(X_val)
    val_y_t = torch.FloatTensor(y_val).unsqueeze(1) # shape matches output (batch, ph, targets)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            by = by.unsqueeze(1)
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = criterion(val_pred, val_y_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    model.load_state_dict(best_model_state)
    
    os.makedirs("experiments/prediction_models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "experiments/prediction_models/checkpoints/transformer_real.pt")
    
    return model, best_val_loss
