import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler 
from seqControl.sample.utils.plotting_utils import plot_signals
from matplotlib import pyplot as plt
from seqControl.sample.utils.general_utils import seed_everything
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
# 🔧 NONLINEAR 2-STATE PLANT (BATCHED RK4 INTEGRATION)
# =========================================================
class TwoStateNonlinearPlant:
    """
    A 2-state nonlinear plant solved using 4th-order Runge-Kutta (RK4).
    Supports parallel batch execution.
    State vector: x shape [Batch, 2]
    Equations of motion:
        dx1/dt = x2
        dx2/dt = -0.5 * x2 - x1^3 + u
    Output:
        y = x1
    """
    def __init__(self, dt=0.1):
        self.dt = dt

    def _dynamics(self, x, u):
        """Continuous-time ODE equations for batches: dx/dt = f(x, u)"""
        x1 = x[..., 0]
        x2 = x[..., 1]
        
        dx1 = x2
        dx2 = -0.5 * x2 - (x1 ** 3) + u
        
        return torch.stack([dx1, dx2], dim=-1)

    def step(self, x_current, u):
        """Discretized batched step using 4th-order Runge-Kutta (RK4)"""
        dt = self.dt
        
        # RK4 Coefficients computed in parallel across the batch axis
        k1 = self._dynamics(x_current, u)
        k2 = self._dynamics(x_current + 0.5 * dt * k1, u)
        k3 = self._dynamics(x_current + 0.5 * dt * k2, u)
        k4 = self._dynamics(x_current + dt * k3, u)
        
        x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

# =========================================================
# 📊 PARALLEL DATA GENERATION (BATCHED CANADAY'S METHOD)
# =========================================================
def generate_data_parallel(plant, seq_len=1000, num_sequences=5000):
    dt = plant.dt
    lambd = 2.0  
    p = 0.5  

    # 1. Generate all random signals simultaneously: [num_sequences, seq_len]
    raw = torch.rand((num_sequences, seq_len)) * 2 - 1
    
    # 2. Batched 1D Real FFT along the sequence dimension
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=dt)
    
    # Mask frequencies higher than cutoff across the entire batch
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff] = 0
    
    # 3. Batched Inverse FFT back to the time domain
    v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
    
    # Normalize along sequence dimension (dim=1) for every sequence in parallel
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = 2 * (v_train - v_min) / (v_max - v_min) - 1
    
    # Control signals matrix: [num_sequences, seq_len]
    u_all = v_norm * p  

    # 4. Batched State Evolution
    # Initialize state matrix [num_sequences, 2] and output array [num_sequences, seq_len + 1]
    x_state = torch.zeros((num_sequences, 2))
    y_all = torch.zeros((num_sequences, seq_len + 1))
    y_all[:, 0] = x_state[:, 0] # y = x1

    # Loop through time steps, executing updates for all sequences concurrently
    for t in range(seq_len):
        u_t = u_all[:, t] # Shape: [num_sequences]
        x_state = plant.step(x_state, u_t)
        y_all[:, t+1] = x_state[:, 0]
        
    y_t = y_all[:, :-1]
    y_next = y_all[:, 1:]
    
    # Stack features to match expected format: [num_sequences, seq_len, 2]
    X = torch.stack([y_t, y_next], dim=-1)  
    Y = u_all.unsqueeze(-1) # Shape: [num_sequences, seq_len, 1]
    
    return X, Y

# =========================================================
# 🧠 OFFICIAL MAMBA CONTROLLER
# =========================================================
class StatefulMambaController(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 32
        self.input_proj = nn.Linear(2, self.d_model)
        
        self.core = Mamba(
            d_model=self.d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.output_proj = nn.Linear(32, 1)

    def forward(self, y_t, y_next):
        """Used ONLY for parallel training with the full sequence."""
        x = torch.cat([y_t, y_next], dim=-1)
        x = self.input_proj(x)
        x = self.core(x)
        return self.output_proj(x)

    def allocate_inference_states(self, batch_size=1, device="cuda"):
        """Allocates empty state caches for the convolution and SSM layers."""
        conv_state = torch.zeros(batch_size, self.d_model * 2, self.core.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, self.d_model * 2, self.core.d_state, device=device)
        return conv_state, ssm_state

    def step(self, y_t_single, y_next_single, conv_state, ssm_state):
        """Processes a SINGLE time step and modifies state caches in-place."""
        y_t_flat = y_t_single.reshape(-1)
        y_next_flat = y_next_single.reshape(-1)

        x = torch.stack([y_t_flat, y_next_flat], dim=-1)
        x = self.input_proj(x) 
        
        x_3d = x.unsqueeze(1)
        x_out_3d, conv_state, ssm_state = self.core.step(x_3d, conv_state, ssm_state)
        
        x_out = x_out_3d.squeeze(1)
        u_out = self.output_proj(x_out)
        
        return u_out, conv_state, ssm_state

# =========================================================
# 📉 TRAINING FUNCTION
# =========================================================
def train(model, X, Y, dt, dirname="plots", epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_losses = []

    batch_size = 64
    num_samples = X.shape[0]

    for epoch in range(epochs):
        total_loss = 0
        permutation = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i : i + batch_size]
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
    x = X[0:1].to(device) 
    y_true = Y[0:1]   

    y_t = x[:, :, 0:1]   
    y_next = x[:, :, 1:2]

    with torch.no_grad():
        y_pred = model(y_t, y_next)

    y_true_np = y_true.squeeze().numpy()
    y_pred_np = y_pred.cpu().squeeze().numpy()

    t_axis = np.arange(len(y_true_np)) * dt
    plot_signals(
        t=t_axis, signals=[y_true_np, y_pred_np], labels=["True u(t)", "Predicted u(t)"],
        xlabel="Time (s)", ylabel="Control", title="Prediction vs Ground Truth",
        filename="prediction", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (WITH STATEFUL CACHE)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=1.0, ref_amplitude=0.18):
    model.eval()
    device = next(model.parameters()).device
    
    # Validation deployment uses single-sequence inputs
    x_state = torch.zeros(2, device=device) 
    y_log, u_log, ref_log = [], [], []

    conv_state, ssm_state = model.allocate_inference_states(batch_size=1, device=device)

    for t in range(steps):
        y_ref = torch.tensor([[ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)]], device=device)
        y_current = x_state[0]

        input_pair = np.array([[y_current.item(), y_ref.item()]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)
            
        y_t_norm = torch.tensor([[input_pair[0, 0]]], dtype=torch.float32, device=device)
        y_next_norm = torch.tensor([[input_pair[0, 1]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_norm_tensor, conv_state, ssm_state = model.step(
                y_t_norm, y_next_norm, conv_state, ssm_state
            )
        
        u_norm_np = u_norm_tensor.cpu().numpy() 
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np)[0, 0]
        else:
            u = u_norm_np[0, 0]
            
        u = torch.tensor(u, dtype=torch.float32, device=device)

        # Step the single validation sequence forward
        x_state = plant.step(x_state.unsqueeze(0), u).squeeze(0)

        y_log.append(y_current.item())
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
    x_state = torch.zeros(2, device=device) 
    y_log, u_log, ref_log = [], [], []
    history_pairs = []

    for t in range(steps):
        y_ref = torch.tensor([ref_value])
        y_current = x_state[0]

        input_pair = np.array([[y_current.item(), y_ref.item()]])  
        input_normalized = x_scaler.transform(input_pair) if x_scaler else input_pair
        history_pairs.append(input_normalized[0]) 

        history_tensor = torch.tensor(history_pairs, dtype=torch.float32).unsqueeze(0).to(device)
        
        y_t_norm = history_tensor[:, :, 0:1]
        y_next_norm = history_tensor[:, :, 1:2]

        with torch.no_grad():
            u_seq_norm = model(y_t_norm, y_next_norm)
        
        u_norm_np = u_seq_norm[:, -1, :].cpu().squeeze().numpy()
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np
        u = torch.tensor(u, dtype=torch.float32, device=device)

        x_state = plant.step(x_state.unsqueeze(0), u).squeeze(0)

        y_log.append(y_current.item())
        u_log.append(u.item())
        ref_log.append(y_ref.item())

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
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=2)
    dt = 0.1
    plant = TwoStateNonlinearPlant(dt=dt)
    
    model = StatefulMambaController().to(device)

    # --- Generate data in Parallel ---
    X, Y = generate_data_parallel(plant)
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

    # --- Train ---
    train(model, X_normalized, Y_normalized, dt)

    # --- Prediction ---
    plot_prediction(model, X_normalized, Y_normalized, dt)

    # --- Closed-loop simulation ---
    simulate_controller_stateful(
        model, plant, dt,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_freq=0.4, ref_amplitude=0.15  
    )

    simulate_constant_controller(
        model, plant, dt,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=0.12  
    )

if __name__ == "__main__":
    main()