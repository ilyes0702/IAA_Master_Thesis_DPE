import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler 
from src.sample.utils.plotting_utils import plot_signals
from matplotlib import pyplot as plt
from src.sample.utils.general_utils import seed_everything
# Import the official Mamba block from the Tri Dao / Albert Gu repository
from mamba_ssm import Mamba

try:
    plt.style.use("src/sample/style.mplstyle")
except:
    pass

# Check if CUDA is available since the official mamba_ssm requires a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("⚠️ WARNING: official mamba_ssm requires a CUDA GPU. Running on CPU will crash.")

# =========================================================
# 🔧 SIMPLE PLANT
# =========================================================
class SimplePlant:
    def __init__(self, a=0.9, b=0.1):
        self.a = a
        self.b = b

    def step(self, y, u):
        return self.a * y* y * 0.2 / (1+y) + self.b * u

# =========================================================
# 📊 DATA GENERATION (CANADAY'S METHOD)
# =========================================================
def generate_data(plant, seq_len=100, num_sequences=5000):
    X, Y = [], []
    dt = 0.1
    lambd = 2.0  
    p = 0.2      

    for _ in range(num_sequences):
        raw = torch.rand((1, seq_len)) * 2 - 1
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, d=dt)
        
        cutoff = 1.0 / lambd
        fft_sig[:, freqs > cutoff] = 0
        
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
        
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min ) - 1
        
        u = (v_norm * p).squeeze(0)  

        y = torch.zeros(seq_len + 1)
        for t in range(seq_len):
            y[t+1] = plant.step(y[t], u[t])
            
        y_t = y[:-1]
        y_next = y[1:]
        
        x = torch.stack([y_t, y_next], dim=-1)  
        y_target = u.unsqueeze(-1)               
        
        X.append(x)
        Y.append(y_target)
        
    return torch.stack(X), torch.stack(Y)

# =========================================================
# 🧠 OFFICIAL MAMBA CONTROLLER
# =========================================================
class SimpleMambaController(nn.Module):
    def __init__(self):
        super().__init__()
        # Project our 2 input features [y(t), y(t+1)] up to the Mamba model dimension
        self.input_proj = nn.Linear(2, 32)
        
        # Instantiate the official selective structural state space model block
        self.core = Mamba(
            d_model=32,    # Model dimension 
            d_state=16,    # SSM state expansion factor
            d_conv=4,      # Local convolution width 
            expand=2       # Block expansion factor
        )
        
        # Project back down to 1 dimension for our predicted control action u(t)
        self.output_proj = nn.Linear(32, 1)

    def forward(self, y_t, y_next):
        # Concatenate features into [Batch, Seq_len, 2]
        x = torch.cat([y_t, y_next], dim=-1)
        
        x = self.input_proj(x)
        x = self.core(x)  # Dispatches sequence context using official custom CUDA kernels
        return self.output_proj(x)

# =========================================================
# 📉 TRAINING FUNCTION
# =========================================================
def train(model, X, Y, dt, dirname="plots", epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_losses = []

    # Using standard mini-batches because official Mamba thrives on paralleled batch matrices
    batch_size = 64
    num_samples = X.shape[0]

    for epoch in range(epochs):
        total_loss = 0
        permutation = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i : i + batch_size]
            # Move memory slices dynamically to the GPU device 
            batch_x = X[indices].to(device)
            batch_y = Y[indices].to(device)

            y_t = batch_x[:, :, 0:1]
            y_next = batch_x[:, :, 1:2]
            
            pred = model(y_t, y_next)
            loss = loss_fn(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(indices)
            
        avg_loss = total_loss / num_samples
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss={avg_loss:.6f}")

    t_axis = np.arange(len(train_losses))
    plot_signals(
        t=t_axis, signals=[np.array(train_losses)], labels=["Train Loss"],
        xlabel="Epoch", ylabel="Loss", title="Training Loss Curve",
        filename="loss_curve", dirname=dirname
    )
    return train_losses

# =========================================================
# 🔍 PREDICTION PLOT 
# =========================================================
def plot_prediction(model, X, Y, dt, dirname="plots", y_scaler=None):
    model.eval() 
    # Grab validation slice and cast to GPU
    x = X[0:1].to(device) 
    y_true = Y[0:1]   

    y_t = x[:, :, 0:1]   
    y_next = x[:, :, 1:2]

    with torch.no_grad():
        y_pred = model(y_t, y_next)

    # Bring results back to CPU for numpy plotting
    y_true_np = y_true.squeeze().numpy()
    y_pred_np = y_pred.cpu().squeeze().numpy()

    t_axis = np.arange(len(y_true_np)) * dt
    plot_signals(
        t=t_axis, signals=[y_true_np, y_pred_np], labels=["True u(t)", "Predicted u(t)"],
        xlabel="Time (s)", ylabel="Control", title="Prediction vs Ground Truth",
        filename="prediction", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (WITH INVERSE TRANSFORM)
# =========================================================
def simulate_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=1.0, ref_amplitude=0.18):
    model.eval()
    y = torch.tensor([0.0])
    y_log, u_log, ref_log = [], [], []

    for t in range(steps):
        model.eval()
    y = torch.tensor([0.0])
    y_log, u_log, ref_log = [], [], []
    
    # Keep track of the historical pairs over time
    history_pairs = []

    for t in range(steps):
        y_ref = torch.tensor([ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)])

        input_pair = np.array([[y.item(), y_ref.item()]])  
        input_normalized = x_scaler.transform(input_pair) if x_scaler else input_pair
        history_pairs.append(input_normalized[0]) # Shape: [2]

        # Convert the entire history up to now to a tensor
        # Shape: [Batch=1, Seq_len=t+1, Feature=2]
        history_tensor = torch.tensor(history_pairs, dtype=torch.float32).unsqueeze(0).to(device)
        
        y_t_norm = history_tensor[:, :, 0:1]
        y_next_norm = history_tensor[:, :, 1:2]

        with torch.no_grad():
            u_seq_norm = model(y_t_norm, y_next_norm)
        
        # We only want the control signal for the CURRENT time step (the last item in the sequence)
        u_norm_np = u_seq_norm[:, -1, :].cpu().squeeze().numpy()
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np
        u = torch.tensor(u, dtype=torch.float32)

        y = plant.step(y, u)

        y_log.append(y.item())
        u_log.append(u.item())
        ref_log.append(y_ref.item())

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t)", "reference"],
        xlabel="Time (s)", ylabel="Output", title="Closed-loop Tracking",
        filename="tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t)"],
        xlabel="Time (s)", ylabel="Control", title="Control Signal",
        filename="control", dirname=dirname
    )

def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_value=0.18):
    model.eval()
    y = torch.tensor([0.0])
    y_log, u_log, ref_log = [], [], []
    
    # Keep track of the historical pairs over time for Mamba's memory
    history_pairs = []

    for t in range(steps):
        # Set the reference to a fixed constant value instead of a sine wave
        y_ref = torch.tensor([ref_value])

        input_pair = np.array([[y.item(), y_ref.item()]])  
        input_normalized = x_scaler.transform(input_pair) if x_scaler else input_pair
        history_pairs.append(input_normalized[0]) 

        # Build the sequence context tensor: [Batch=1, Seq_len=t+1, Feature=2]
        history_tensor = torch.tensor(history_pairs, dtype=torch.float32).unsqueeze(0).to(device)
        
        y_t_norm = history_tensor[:, :, 0:1]
        y_next_norm = history_tensor[:, :, 1:2]

        with torch.no_grad():
            u_seq_norm = model(y_t_norm, y_next_norm)
        
        # Extract the control signal for the CURRENT step (the last element in the sequence)
        u_norm_np = u_seq_norm[:, -1, :].cpu().squeeze().numpy()
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np
        u = torch.tensor(u, dtype=torch.float32)

        # Step the plant forward
        y = plant.step(y, u)

        # Log metrics
        y_log.append(y.item())
        u_log.append(u.item())
        ref_log.append(y_ref.item())

    # --- Plotting Results ---
    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t)", "reference"],
        xlabel="Time (s)", ylabel="Output", title="Closed-loop Constant Tracking",
        filename="constant_tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t)"],
        xlabel="Time (s)", ylabel="Control", title="Constant Control Signal",
        filename="constant_control", dirname=dirname
    )

# =========================================================
# 📊 DATA VISUALIZATION
# =========================================================
def plot_dataset(X, Y, dt, dirname="plots"):
    x = X[0].numpy()
    y = Y[0].numpy()
    y_t = x[:, 0]
    y_next = x[:, 1]
    u = y[:, 0]
    t_axis = np.arange(len(y_t)) * dt
    plot_signals(
        t=t_axis, signals=[y_t, y_next, u], labels=["y(t)", "y(t+Δ)", "u(t)"],
        xlabel="Time (s)", ylabel="Value", title="Dataset Example",
        filename="dataset", dirname=dirname
    )

# =========================================================
# 🔥 MAIN PIPELINE (WITH NORMALIZATION)
# =========================================================
def main():
    seed_everything(seed=2)
    dt = 0.1
    plant = SimplePlant()
    
    # Instantiate the controller and migrate its parameters directly into GPU SRAM memory
    model = SimpleMambaController().to(device)

    # --- Generate data ---
    X, Y = generate_data(plant)
    plot_dataset(X, Y, dt)

    # --- Normalize data ---
    X_np = X.numpy().reshape(-1, 2)  
    Y_np = Y.numpy().reshape(-1, 1)  

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_normalized = x_scaler.fit_transform(X_np).reshape(X.shape)
    Y_normalized = y_scaler.fit_transform(Y_np).reshape(Y.shape)

    X_normalized = torch.tensor(X_normalized, dtype=torch.float32)
    Y_normalized = torch.tensor(Y_normalized, dtype=torch.float32)

    # --- Train (Mini-batch loop safely handles the .to(device) step) ---
    train(model, X_normalized, Y_normalized, dt)

    # --- Prediction ---
    plot_prediction(model, X_normalized, Y_normalized, dt)

    # --- Closed-loop simulation ---
    simulate_controller(
        model, plant, dt,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_freq=0.6, ref_amplitude=0.01  
    )

    simulate_constant_controller(
        model, plant, dt,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=0.019  
    )

if __name__ == "__main__":
    main()