import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler 
from matplotlib import pyplot as plt

# =========================================================
# 📦 OFFICIAL TRI DAO / ALBERT GU MAMBA IMPORT
# =========================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install mamba_ssm via pip (requires a CUDA environment).")

# Setup hardware device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("⚠️ WARNING: official mamba_ssm requires a CUDA GPU. Running on CPU will crash.")

# Mock plotting function to replace external dependencies safely
def plot_signals(t, signals, labels, xlabel, ylabel, title, filename, dirname="plots"):
    os.makedirs(dirname, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for sig, lbl in zip(signals, labels):
        plt.plot(t, sig, label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dirname, f"{filename}.png"), bbox_inches='tight')
    plt.close()

def seed_everything(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# 🔧 NON-BIOLOGICAL PLANT: NONLINEAR HYDRAULIC TWO-TANK SYSTEM
# =========================================================
class NonlinearHydraulicPlant:
    """
    Nonlinear Two-Tank Hydraulic System.
    States: 
        x1: Liquid level in Tank 1 (m)
        x2: Liquid level in Tank 2 (m)
    Input:
        u: Pump inflow stream into Tank 1 (m^3/h)
    Output:
        y: Nonlinear sensor discharge rate curve (m^3/h)
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        # Physical Parameters (Flow coefficients and pipe geometry constants)
        self.alpha = 2.5      # Maximum theoretical discharge rate
        self.beta = 1.2       # Valve saturation constant (similar to K_S)
        self.c1 = 0.8         # Cross-flow coefficient from Tank 1 to Tank 2
        self.c2 = 0.5         # Gravity discharge constant from Tank 2
        self.b1 = 1.5         # Pump actuator gain
        
        # Internal clock to track time-varying pipeline pressure restrictions
        self.current_time = 0.0

    def reset_time(self):
        """Resets the internal plant clock for a new simulation sequence."""
        self.current_time = 0.0

    def _get_pressure_modifier(self, t):
        """Computes a time-dependent dynamic valve restriction coefficient."""
        term1 = 0.2 * (t - 5.0) if t > 5.0 else 0.0
        term2 = 0.2 * (t - 15.0) if t > 15.0 else 0.0
        return 1.0 + term1 - term2

    def output_characteristic(self, x2, t):
        """Nonlinear sensor output calculation (Rational saturation curve)"""
        mod = self._get_pressure_modifier(t)
        # Bounded saturation characteristic equivalent to the structure of Monod kinetics
        return (self.alpha * x2) / (self.beta * mod + x2)

    def _dynamics(self, x, u, t):
        """Continuous-time physical fluid balance ODEs: dx/dt = f(x, u, t)"""
        x1 = x[..., 0]
        x2 = x[..., 1]
        
        # Interacting tank hydraulics wrapped with safety clamping to avoid negative values
        flow_1_to_2 = self.c1 * torch.sqrt(torch.clamp(x1, min=0.0))
        flow_out_2  = self.c2 * torch.sqrt(torch.clamp(x2, min=0.0))
        
        # Conservation of volume state derivative equations
        dx1 = -flow_1_to_2 + self.b1 * u
        dx2 = flow_1_to_2 - flow_out_2
        
        return torch.stack([dx1, dx2], dim=-1)

    def step(self, x_current, u):
        """Discretized batched step using 4th-order Runge-Kutta (RK4)"""
        dt = self.dt
        t = self.current_time
        
        k1 = self._dynamics(x_current, u, t)
        k2 = self._dynamics(x_current + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self._dynamics(x_current + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self._dynamics(x_current + dt * k3, u, t + dt)
        
        x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        self.current_time += dt
        return x_next

# =========================================================
# 📊 PARALLEL DATA GENERATION (THREE-STEP LOOKBACK WINDOW)
# =========================================================
def generate_data_parallel(plant, seq_len=50, num_sequences=5000):
    dt = plant.dt
    lambd = 2.0  
    p = 2.5  

    # 1. Generate trajectories (extended to provide sliding lookback context)
    total_steps = seq_len + 1 
    raw = torch.rand((num_sequences, total_steps)) * 2 - 1
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(total_steps, d=dt)
    
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff] = 0
    
    v_train = torch.fft.irfft(fft_sig, n=total_steps, dim=1)
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = (v_train - v_min) / (v_max - v_min)  
    
    u_all = v_norm * p  

    # 2. Batched State Evolution
    plant.reset_time()
    x_state = torch.tensor([[2.0, 1.5]], dtype=torch.float32).repeat(num_sequences, 1)
    
    y_all = torch.zeros((num_sequences, total_steps + 1))
    y_all[:, 0] = plant.output_characteristic(x_state[:, 1], plant.current_time) 

    for t in range(total_steps):
        u_t = u_all[:, t] 
        x_state = plant.step(x_state, u_t)
        y_all[:, t+1] = plant.output_characteristic(x_state[:, 1], plant.current_time)
        
    # Slicing out window frames for triplet representation: [y_prev, y_t, y_next]
    y_prev = y_all[:, 0:-2]
    y_t    = y_all[:, 1:-1]
    y_next = y_all[:, 2:]
    
    X = torch.stack([y_prev, y_t, y_next], dim=-1) # Shape: [num_sequences, seq_len, 3] 
    Y = u_all[:, 1:].unsqueeze(-1)                 # Shape: [num_sequences, seq_len, 1]
    
    return X, Y

# =========================================================
# 🧠 OFFICIAL MAMBA CONTROLLER (THREE-CHANNEL INPUTS)
# =========================================================
class StatefulMambaController(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 32
        # Projection changed to dimension 3 to fit: [y_prev, y_t, y_next]
        self.input_proj = nn.Linear(3, self.d_model)
        
        self.core = Mamba(
            d_model=self.d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.output_proj = nn.Linear(32, 1)

    def forward(self, y_prev, y_t, y_next):
        x = torch.cat([y_prev, y_t, y_next], dim=-1)
        x = self.input_proj(x)
        x = self.core(x)
        return self.output_proj(x)

    def allocate_inference_states(self, batch_size=1, device="cuda"):
        conv_state = torch.zeros(batch_size, self.d_model * 2, self.core.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, self.d_model * 2, self.core.d_state, device=device)
        return conv_state, ssm_state

    def step(self, y_prev_single, y_t_single, y_next_single, conv_state, ssm_state):
        y_prev_flat = y_prev_single.reshape(-1)
        y_t_flat = y_t_single.reshape(-1)
        y_next_flat = y_next_single.reshape(-1)

        x = torch.stack([y_prev_flat, y_t_flat, y_next_flat], dim=-1)
        x = self.input_proj(x) 
        
        x_3d = x.unsqueeze(1)
        x_out_3d, conv_state, ssm_state = self.core.step(x_3d, conv_state, ssm_state)
        
        x_out = x_out_3d.squeeze(1)
        u_out = self.output_proj(x_out)
        
        return u_out, conv_state, ssm_state

# =========================================================
# 📉 TRAINING FUNCTION
# =========================================================
def train(model, X, Y, dt, dirname="plots", epochs=20):
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

            y_prev = batch_x[:, :, 0:1]
            y_t    = batch_x[:, :, 1:2]
            y_next = batch_x[:, :, 2:3]
            
            pred = model(y_prev, y_t, y_next)
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

    y_prev = x[:, :, 0:1]
    y_t    = x[:, :, 1:2]   
    y_next = x[:, :, 2:3]

    with torch.no_grad():
        y_pred = model(y_prev, y_t, y_next)

    y_true_np = y_true.squeeze().numpy()
    y_pred_np = y_pred.cpu().squeeze().numpy()

    t_axis = np.arange(len(y_true_np)) * dt
    plot_signals(
        t=t_axis, signals=[y_true_np, y_pred_np], labels=["True u(t)", "Predicted u(t)"],
        xlabel="Time (s)", ylabel="Control", title="Prediction vs Ground Truth",
        filename="prediction", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP DATASET SEQUENTIAL SIMULATION
# =========================================================
def simulate_controller_stateful(model, plant, dt, ref_sequence, dirname="plots", x_scaler=None, y_scaler=None):
    model.eval()
    device = next(model.parameters()).device
    
    plant.reset_time()
    x_state = torch.tensor([[2.0, 1.5]], device=device) 
    y_log, u_log, ref_log = [], [], []

    conv_state, ssm_state = model.allocate_inference_states(batch_size=1, device=device)
    steps = len(ref_sequence)

    # Prime historical measurement tracker at t=0
    y_prev_val = plant.output_characteristic(x_state[:, 1], plant.current_time).item()

    for t in range(steps):
        y_ref_val = ref_sequence[t]
        y_current = plant.output_characteristic(x_state[:, 1], plant.current_time).item()

        input_triplet = np.array([[y_prev_val, y_current, y_ref_val]])  
        if x_scaler:
            input_triplet = x_scaler.transform(input_triplet)
            
        y_prev_norm = torch.tensor([[input_triplet[0, 0]]], dtype=torch.float32, device=device)
        y_t_norm    = torch.tensor([[input_triplet[0, 1]]], dtype=torch.float32, device=device)
        y_next_norm = torch.tensor([[input_triplet[0, 2]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_norm_tensor, conv_state, ssm_state = model.step(
                y_prev_norm, y_t_norm, y_next_norm, conv_state, ssm_state
            )
        
        u_norm_np = u_norm_tensor.cpu().numpy() 
        u = y_scaler.inverse_transform(u_norm_np)[0, 0] if y_scaler else u_norm_np[0, 0]
        u = np.clip(u, 0.0, 5.0)
        
        u = torch.tensor(u, dtype=torch.float32, device=device)
        
        y_prev_val = y_current
        x_state = plant.step(x_state, u)

        y_log.append(y_current)
        u_log.append(u.item())
        ref_log.append(y_ref_val)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) (discharge)", "reference"],
        xlabel="Time (h)", ylabel="Flow Rate (m^3/h)", title="Closed-loop Dataset Tracking",
        filename="tracking_dataset", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t) (Pump Flow)"],
        xlabel="Time (h)", ylabel="Pump Rate (m^3/h)", title="Control Signal",
        filename="control_dataset", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP CONSTANT STEP SIMULATION
# =========================================================
def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_value=1.5):
    model.eval()
    device = next(model.parameters()).device
    
    plant.reset_time()
    x_state = torch.tensor([[2.0, 1.5]], device=device) 
    y_log, u_log, ref_log = [], [], []

    conv_state, ssm_state = model.allocate_inference_states(batch_size=1, device=device)

    # Prime historical measurement tracker at t=0
    y_prev_val = plant.output_characteristic(x_state[:, 1], plant.current_time).item()

    for t in range(steps):
        y_current = plant.output_characteristic(x_state[:, 1], plant.current_time).item()
        y_ref_val = ref_value

        input_triplet = np.array([[y_prev_val, y_current, y_ref_val]])  
        if x_scaler:
            input_triplet = x_scaler.transform(input_triplet)
            
        y_prev_norm = torch.tensor([[input_triplet[0, 0]]], dtype=torch.float32, device=device)
        y_t_norm    = torch.tensor([[input_triplet[0, 1]]], dtype=torch.float32, device=device)
        y_next_norm = torch.tensor([[input_triplet[0, 2]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_norm_tensor, conv_state, ssm_state = model.step(
                y_prev_norm, y_t_norm, y_next_norm, conv_state, ssm_state
            )
        
        u_norm_np = u_norm_tensor.cpu().numpy() 
        u = y_scaler.inverse_transform(u_norm_np)[0, 0] if y_scaler else u_norm_np[0, 0]
        u = np.clip(u, 0.0, 5.0)
            
        u = torch.tensor(u, dtype=torch.float32, device=device)
        
        y_prev_val = y_current 
        x_state = plant.step(x_state, u)

        y_log.append(y_current)
        u_log.append(u.item())
        ref_log.append(y_ref_val)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) (discharge)", "reference (y*)"],
        xlabel="Time (h)", ylabel="Flow Rate (m^3/h)", title=f"Closed-loop Step Response (y*={ref_value})",
        filename="tracking_constant", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t) (Pump Flow)"],
        xlabel="Time (h)", ylabel="Pump Rate (m^3/h)", title="Step Response Control Effort",
        filename="control_constant", dirname=dirname
    )

# =========================================================
# 📊 DATA VISUALIZATION
# =========================================================
def plot_dataset(X, Y, dt, dirname="plots"):
    x = X[0].numpy()
    y = Y[0].numpy()
    y_prev = x[:, 0]
    y_t = x[:, 1]
    y_next = x[:, 2]
    u = y[:, 0]
    t_axis = np.arange(len(y_t)) * dt
    plot_signals(
        t=t_axis, signals=[y_prev, y_t, y_next, u], labels=["y(t-Δ)", "y(t)", "y(t+Δ)", "u(t)"],
        xlabel="Time (h)", ylabel="Value", title="Dataset Example with History Context",
        filename="dataset", dirname=dirname
    )

# =========================================================
# 🔥 MAIN EXECUTION PIPELINE
# =========================================================
def main():
    seed_everything(seed=2)
    dt = 0.1
    plant = NonlinearHydraulicPlant(dt=dt)
    
    model = StatefulMambaController().to(device)

    # --- Data Generation Pipeline ---
    X, Y = generate_data_parallel(plant)
    plot_dataset(X, Y, dt)

    # Grab the dataset trajectory reference tracking sequence (index 2 corresponds to y_next)
    ref_sequence = X[0, :, 2].numpy()

    # --- Data Standardization and Shaping ---
    X_np = X.numpy().reshape(-1, 3)  
    Y_np = Y.numpy().reshape(-1, 1)  

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_normalized = x_scaler.fit_transform(X_np).reshape(X.shape)
    Y_normalized = y_scaler.fit_transform(Y_np).reshape(Y.shape)

    X_normalized = torch.tensor(X_normalized, dtype=torch.float32)
    Y_normalized = torch.tensor(Y_normalized, dtype=torch.float32)

    # --- Train System ---
    train(model, X_normalized, Y_normalized, dt)

    # --- Run Single Validation Plot ---
    plot_prediction(model, X_normalized, Y_normalized, dt)

    # --- Closed-loop Simulation 1: Sequence Tracking ---
    simulate_controller_stateful(
        model, plant, dt, ref_sequence=ref_sequence,
        x_scaler=x_scaler, y_scaler=y_scaler
    )

    # --- Closed-loop Simulation 2: Constant Step Response Tracking ---
    simulate_constant_controller(
        model, plant, dt, steps=50,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=7  
    )

if __name__ == "__main__":
    main()