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
# 🔧 NON-BIOLOGICAL MIMO PLANT (TWO-TANK SYSTEM VIA RK4)
# =========================================================
class TwoTankNonlinearPlant:
    """
    A non-linear coupled two-tank system (MIMO: 2 Inputs, 2 Outputs).
    States / Outputs:
        x1 (y1): Liquid level in Tank 1 (m)
        x2 (y2): Liquid level in Tank 2 (m)
    Inputs:
        u1: Flow rate from Pump 1 into Tank 1 (m^3/s)
        u2: Flow rate from Pump 2 into Tank 2 (m^3/s)
    Dynamics:
        dx1/dt = (u1 - c1 * sqrt(x1) - c12 * sign(x1 - x2) * sqrt(|x1 - x2|)) / Area1
        dx2/dt = (u2 - c2 * sqrt(x2) + c12 * sign(x1 - x2) * sqrt(|x1 - x2|)) / Area2
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        # Physical Parameters
        self.A1 = 1.0   # Cross-sectional area Tank 1
        self.A2 = 1.0   # Cross-sectional area Tank 2
        self.c1 = 0.2   # Valve coefficient Outflow 1
        self.c2 = 0.2   # Valve coefficient Outflow 2
        self.c12 = 0.1  # Valve coefficient Inter-tank flow

    def _dynamics(self, x, u):
        """Continuous-time non-linear fluid dynamics for batches: dx/dt = f(x, u)"""
        x1 = x[..., 0]
        x2 = x[..., 1]
        u1 = u[..., 0]
        u2 = u[..., 1]
        
        # Enforce physical floor of 0 for heights inside square roots
        x1_safe = torch.clamp(x1, min=1e-5)
        x2_safe = torch.clamp(x2, min=1e-5)
        diff = x1 - x2
        abs_diff_safe = torch.clamp(torch.abs(diff), min=1e-5)
        
        # Inter-tank flow rate calculation
        flow_12 = self.c12 * torch.sign(diff) * torch.sqrt(abs_diff_safe)
        
        dx1 = (u1 - self.c1 * torch.sqrt(x1_safe) - flow_12) / self.A1
        dx2 = (u2 - self.c2 * torch.sqrt(x2_safe) + flow_12) / self.A2
        
        return torch.stack([dx1, dx2], dim=-1)

    def step(self, x_current, u):
        """Discretized batched step using 4th-order Runge-Kutta (RK4)"""
        dt = self.dt
        
        k1 = self._dynamics(x_current, u)
        k2 = self._dynamics(x_current + 0.5 * dt * k1, u)
        k3 = self._dynamics(x_current + 0.5 * dt * k2, u)
        k4 = self._dynamics(x_current + dt * k3, u)
        
        x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Prevent levels from dipping below absolute zero due to discretization margins
        return torch.clamp(x_next, min=0.0)

# =========================================================
# 📊 PARALLEL DATA GENERATION (BATCHED MIMO METHOD)
# =========================================================
def generate_data_parallel(plant, seq_len=100, num_sequences=5000):
    dt = plant.dt
    lambd = 2.0  
    p = 0.4  

    # 1. Generate 2 separate random signals concurrently per sequence
    raw = torch.rand((num_sequences, seq_len, 2)) * 2 - 1
    
    # Process FFT along the time axis (dim=1)
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=dt)
    
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff, :] = 0
    
    v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
    
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = (v_train - v_min) / (v_max - v_min + 1e-6) 
    
    # Scale inputs to a meaningful positive operating range [0.05, 0.45]
    u_all = 0.05 + v_norm * p  

    # 2. Batched State Evolution
    x_state = torch.tensor([[0.2, 0.1]], dtype=torch.float32).repeat(num_sequences, 1)
    y_all = torch.zeros((num_sequences, seq_len + 1, 2))
    y_all[:, 0, :] = x_state # y = x

    for t in range(seq_len):
        u_t = u_all[:, t, :] # Shape: [num_sequences, 2]
        x_state = plant.step(x_state, u_t)
        y_all[:, t+1, :] = x_state
        
    y_t = y_all[:, :-1, :]       # [num_sequences, seq_len, 2]
    y_next = y_all[:, 1:, :]     # [num_sequences, seq_len, 2]
    
    # Combine outputs: total feature dim = 4 (y1_t, y2_t, y1_next, y2_next)
    X = torch.cat([y_t, y_next], dim=-1)  
    Y = u_all                     # Target is the 2-dimensional control vector [u1, u2]
    
    return X, Y

# =========================================================
# 🧠 OFFICIAL MIMO MAMBA CONTROLLER
# =========================================================
class StatefulMambaController(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 32
        # Input features: 4 (2 outputs at t, 2 outputs at t+1)
        self.input_proj = nn.Linear(4, self.d_model)
        
        self.core = Mamba(
            d_model=self.d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        # Output projections: 2 control signals (u1, u2)
        self.output_proj = nn.Linear(32, 2)

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

        # Concatenate into flat feature vector of length 4
        x = torch.cat([y_t_flat, y_next_flat], dim=-1).unsqueeze(0) # [1, 4]
        x = self.input_proj(x) 
        
        x_3d = x.unsqueeze(1) # [1, 1, 32]
        x_out_3d, conv_state, ssm_state = self.core.step(x_3d, conv_state, ssm_state)
        
        x_out = x_out_3d.squeeze(1)
        u_out = self.output_proj(x_out)
        
        return u_out, conv_state, ssm_state

# =========================================================
# 📉 TRAINING FUNCTION
# =========================================================
def train(model, X, Y, dt, dirname="plots", epochs=10):
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

            y_t = batch_x[:, :, 0:2]
            y_next = batch_x[:, :, 2:4]
            
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
def plot_prediction(model, X, Y, dt, dirname="plots"):
    model.eval() 
    x = X[0:1].to(device) 
    y_true = Y[0:1]   

    y_t = x[:, :, 0:2]   
    y_next = x[:, :, 2:4]

    with torch.no_grad():
        y_pred = model(y_t, y_next)

    y_true_np = y_true.squeeze().numpy()         # [seq_len, 2]
    y_pred_np = y_pred.cpu().squeeze().numpy()   # [seq_len, 2]

    t_axis = np.arange(len(y_true_np)) * dt
    plot_signals(
        t=t_axis, signals=[y_true_np[:, 0], y_pred_np[:, 0]], labels=["True u1(t)", "Pred u1(t)"],
        xlabel="Time (s)", ylabel="Control 1", title="Prediction vs Ground Truth Pump 1",
        filename="prediction_u1", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATIONS (MIMO)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None):
    model.eval()
    device = next(model.parameters()).device
    
    x_state = torch.tensor([[0.2, 0.1]], device=device) 
    y1_log, y2_log, u1_log, u2_log, ref1_log, ref2_log = [], [], [], [], [], []

    conv_state, ssm_state = model.allocate_inference_states(batch_size=1, device=device)

    for t in range(steps):
        # Set two separate reference tracking trajectories (sine waves)
        ref1 = 0.25 + 0.05 * np.sin(2 * np.pi * 0.1 * t * dt)
        ref2 = 0.15 + 0.03 * np.cos(2 * np.pi * 0.08 * t * dt)
        
        y_current = x_state.squeeze(0) # [y1, y2]

        input_quad = np.array([[y_current[0].item(), y_current[1].item(), ref1, ref2]])  
        if x_scaler:
            input_quad = x_scaler.transform(input_quad)
            
        y_t_norm = torch.tensor([[input_quad[0, 0], input_quad[0, 1]]], dtype=torch.float32, device=device)
        y_next_norm = torch.tensor([[input_quad[0, 2], input_quad[0, 3]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_norm_tensor, conv_state, ssm_state = model.step(
                y_t_norm, y_next_norm, conv_state, ssm_state
            )
        
        u_norm_np = u_norm_tensor.cpu().numpy() 
        
        if y_scaler:
            u_actual = y_scaler.inverse_transform(u_norm_np)[0]
        else:
            u_actual = u_norm_np[0]
            
        u = torch.tensor(u_actual, dtype=torch.float32, device=device).unsqueeze(0)

        x_state = plant.step(x_state, u)

        y1_log.append(y_current[0].item())
        y2_log.append(y_current[1].item())
        u1_log.append(u[0, 0].item())
        u2_log.append(u[0, 1].item())
        ref1_log.append(ref1)
        ref2_log.append(ref2)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y1_log, ref1_log, y2_log, ref2_log], 
        labels=["y1 (Tank 1)", "ref1", "y2 (Tank 2)", "ref2"],
        xlabel="Time (s)", ylabel="Levels (m)", title="MIMO Closed-loop Tracking",
        filename="tracking_mimo", dirname=dirname
    )

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=2)
    dt = 0.1
    plant = TwoTankNonlinearPlant(dt=dt)
    model = StatefulMambaController().to(device)

    # --- Generate data in Parallel ---
    X, Y = generate_data_parallel(plant)

    # --- Normalize data ---
    X_np = X.numpy().reshape(-1, 4)  
    Y_np = Y.numpy().reshape(-1, 2)  

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
        model, plant, dt, steps=100,
        x_scaler=x_scaler, y_scaler=y_scaler
    )

if __name__ == "__main__":
    main()