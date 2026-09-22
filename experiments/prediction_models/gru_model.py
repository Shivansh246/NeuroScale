import torch
import torch.nn as nn
import os

class GRUModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.gru(x)
        # Take the final hidden representation
        out = out[:, -1, :]
        return self.linear(out)

def train_gru(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=1e-3, patience=5, seed=42):
    torch.manual_seed(seed)
    model = GRUModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_X_t = torch.FloatTensor(X_val)
    val_y_t = torch.FloatTensor(y_val)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
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
    torch.save(model.state_dict(), "experiments/prediction_models/checkpoints/gru_real.pt")
    
    return model, best_val_loss
