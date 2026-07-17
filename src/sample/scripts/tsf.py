import time
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Check for CUDA (Mamba requires a CUDA device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != "cuda":
    print("⚠️ WARNING: Mamba-SSM requires a CUDA GPU. If run on CPU, Mamba may fail to import or execute.")

# Attempt to import Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ Error: Could not import 'mamba_ssm'. Install it using 'pip install mamba-ssm triton causal-conv1d'.")
    raise

from src.sample.utils.plotting_utils import *

# =====================================================================
# DATA GENERATION
# =====================================================================
def generate_synthetic_data(n_samples=5000, seq_len=100, pred_len=1):
    t = np.linspace(0, 100, n_samples)
    signal = np.sin(t) + 0.5 * np.cos(2 * t) + 0.1 * t + np.random.normal(0, 0.1, n_samples)
    
    scaler = StandardScaler()
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    
    X, Y = [], []
    for i in range(len(signal_scaled) - seq_len - pred_len + 1):
        X.append(signal_scaled[i : i + seq_len])
        Y.append(signal_scaled[i + seq_len : i + seq_len + pred_len])
        
    X = np.array(X)[..., np.newaxis] 
    Y = np.array(Y)                  
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), scaler


# =====================================================================
# MODEL ARCHITECTURES
# =====================================================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class MambaForecaster(nn.Module):
    def __init__(self, input_dim=1, d_model=64, d_state=16, d_conv=4, expand=2, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x_proj = self.input_proj(x)                  
        mamba_out = self.mamba(x_proj)               
        return self.output_proj(mamba_out[:, -1, :])


# =====================================================================
# TRAINING & EVALUATION HOOKS
# =====================================================================
def train_model(model, dataloader, criterion, optimizer, epochs=15):
    model.train()
    epoch_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.6f}")
        
    total_time = time.time() - start_time
    return epoch_losses, total_time


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(predictions.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)


# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 50
    PRED_LEN = 1
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 1e-3
    
    # 1. Prepare Datasets
    print("Preparing synthetic time series datasets...")
    X, Y, scaler = generate_synthetic_data(n_samples=4000, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    split_idx = int(0.8 * len(X))
    
    train_dataset = TensorDataset(X[:split_idx], Y[:split_idx])
    val_dataset = TensorDataset(X[split_idx:], Y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Instantiate networks
    lstm_model = LSTMForecaster(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    mamba_model = MambaForecaster(input_dim=1, d_model=64, d_state=16, expand=2, output_dim=1).to(device)
    
    criterion = nn.MSELoss()
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)
    mamba_optimizer = torch.optim.Adam(mamba_model.parameters(), lr=LR)
    
    # 3. Training passes
    print("\n🤖 Training LSTM Baseline...")
    lstm_losses, lstm_time = train_model(lstm_model, train_loader, criterion, lstm_optimizer, epochs=EPOCHS)
    lstm_val_loss, lstm_preds, true_vals = evaluate_model(lstm_model, val_loader, criterion)
    
    print("\n🌀 Training Mamba SSM...")
    mamba_losses, mamba_time = train_model(mamba_model, train_loader, criterion, mamba_optimizer, epochs=EPOCHS)
    mamba_val_loss, mamba_preds, _ = evaluate_model(mamba_model, val_loader, criterion)
    
    # 4. Format outputs for custom plot_signals integration
    epochs_axis = np.arange(1, EPOCHS + 1)
    
    # Unscale predictions to real metric values
    true_unscaled = scaler.inverse_transform(true_vals).flatten()
    lstm_unscaled = scaler.inverse_transform(lstm_preds).flatten()
    mamba_unscaled = scaler.inverse_transform(mamba_preds).flatten()
    
    # Isolate a smaller validation sample segment for clean visual tracking
    plot_points = 150
    t_val_axis = np.arange(plot_points)
    
    # =====================================================================
    # 5. GENERATE PLOTS USING YOUR CUSTOM PLOT_SIGNALS IMPLEMENTATION
    # =====================================================================
    print("\n📈 Plotting convergence and predictions using your custom 'plot_signals' function...")
    
    # Plot 1: Training convergence loss comparison
    plot_signals(
        t=epochs_axis,
        signals=[np.array(lstm_losses), np.array(mamba_losses)],
        labels=["LSTM Train Loss", "Mamba Train Loss"],
        title="Training Convergence Comparison",
        xlabel="Epochs",
        ylabel="MSE Loss",
        figsize=(6, 6),
        filename="training_loss_curves",
        dirname="benchmark_plots",
        asp=1.0,
        show=False
    )
    
    # Plot 2: Time Series forecasting validation output comparison
    plot_signals(
        t=t_val_axis,
        signals=[
            true_unscaled[:plot_points],
            lstm_unscaled[:plot_points],
            mamba_unscaled[:plot_points]
        ],
        labels=["True Signal", "LSTM Prediction", "Mamba Prediction"],
        title="Validation Set Prediction Alignment",
        xlabel="Time Steps (dt)",
        ylabel="Amplitude",
        figsize=(8, 6),
        filename="validation_prediction_comparison",
        dirname="benchmark_plots",
        asp=0.5, # Slightly flatter rectangular profile
        show=False
    )
    
    # 6. Performance overview console summary
    performance_summary = pd.DataFrame({
        "Model": ["LSTM", "Mamba-SSM"],
        "Training Time (s)": [f"{lstm_time:.2f}s", f"{mamba_time:.2f}s"],
        "Validation MSE": [f"{lstm_val_loss:.6f}", f"{mamba_val_loss:.6f}"],
        "Throughput Speed": [
            f"{len(train_dataset)/lstm_time:.1f} seqs/s", 
            f"{len(train_dataset)/mamba_time:.1f} seqs/s"
        ]
    })
    
    print("\n" + "="*60)
    print("📊 BENCHMARK COMPARISON SUMMARY")
    print("="*60)
    print(performance_summary.to_string(index=False))
    print("="*60)