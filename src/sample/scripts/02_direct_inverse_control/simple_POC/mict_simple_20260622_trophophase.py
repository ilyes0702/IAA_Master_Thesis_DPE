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
# 🔧 TROPHOPHASE FERMENTATION PLANT (BATCHED RK4)
# =========================================================
class TrophophaseFermentationPlant:
    """
    Penicillin Fermentation Process - Trophophase Plant Model.
    States: 
        x1: Biomass mass (g TS)
        x2: Substrate mass (mg S)
    Input:
        u: Glucose feed stream u1 (l/h)
    Output:
        y: Growth rate mu(x2) (1/h)
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        # Model Parameters from Table 1
        self.mu_max = 0.12
        self.K_S = 50.0
        self.m_S = 23.0
        self.p1 = 0.00047
        self.p2 = 200000.0
        
        # Internal clock to track time-dependent volume V(t)
        self.current_time = 0.0

    def reset_time(self):
        """Resets the internal plant clock for a new simulation sequence."""
        self.current_time = 0.0

    def _get_V(self, t):
        """Computes time-dependent reactor volume V(t) using Heaviside steps."""
        # V(t) = 150 + 2*(t-5)*sigma(t-5) - 2*(t-15)*sigma(t-15)
        term1 = 2.0 * (t - 5.0) if t > 5.0 else 0.0
        term2 = 2.0 * (t - 15.0) if t > 15.0 else 0.0
        return 150.0 + term1 - term2

    
    def mu(self, x2, t):
        """Monod growth kinetics calculation: mu(x2)"""
        V = self._get_V(t)
        return (self.mu_max * x2) / (self.K_S * V + x2)

    def _dynamics(self, x, u, t):
        """Continuous-time balance ODE equations: dx/dt = f(x, u, t)"""
        x1 = x[..., 0]
        x2 = x[..., 1]
        
        mu_val = self.mu(x2, t)
        
        # Balance equations (1) & (2)
        dx1 = mu_val * x1
        dx2 = -(1.0 / self.p1) * mu_val * x1 - self.m_S * x1 + self.p2 * u
        
        return torch.stack([dx1, dx2], dim=-1)

    def step(self, x_current, u):
        """Discretized batched step using 4th-order Runge-Kutta (RK4)"""
        dt = self.dt
        t = self.current_time
        
        # RK4 Coefficients
        k1 = self._dynamics(x_current, u, t)
        k2 = self._dynamics(x_current + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self._dynamics(x_current + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self._dynamics(x_current + dt * k3, u, t + dt)
        
        x_next = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Advance clock forward by dt
        self.current_time += dt
        return x_next

# =========================================================
# 📊 PARALLEL DATA GENERATION (BATCHED CANADAY'S METHOD)
# =========================================================
def generate_data_parallel(plant, seq_len=200, num_sequences=5000):
    dt = plant.dt
    lambd = 2.0  
    p = 0.5  # Input scaling factor within the limits [0, 1]

    # 1. Generate random control trajectories in parallel
    raw = torch.rand((num_sequences, seq_len)) * 2 - 1
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=dt)
    
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff] = 0
    
    v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = (v_train - v_min) / (v_max - v_min)  # Scale to [0, 1] to respect real physical limits
    
    u_all = v_norm * p  

    # 2. Batched State Evolution
    plant.reset_time()
    # Initialize state matrix with typical starting fermentation conditions
    # x1_0 = 1.0 g TS, x2_0 = 1000.0 mg S
    x_state = torch.tensor([[1.0, 1000.0]], dtype=torch.float32).repeat(num_sequences, 1)
    
    y_all = torch.zeros((num_sequences, seq_len + 1))
    y_all[:, 0] = plant.mu(x_state[:, 1], plant.current_time) # Initial growth rate y = mu(x2)

    for t in range(seq_len):
        u_t = u_all[:, t] 
        x_state = plant.step(x_state, u_t)
        y_all[:, t+1] = plant.mu(x_state[:, 1], plant.current_time)
        
    y_t = y_all[:, :-1]
    y_next = y_all[:, 1:]
    
    X = torch.stack([y_t, y_next], dim=-1)  
    Y = u_all.unsqueeze(-1) 
    
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
        x = torch.cat([y_t, y_next], dim=-1)
        x = self.input_proj(x)
        x = self.core(x)
        return self.output_proj(x)

    def allocate_inference_states(self, batch_size=1, device="cuda"):
        conv_state = torch.zeros(batch_size, self.d_model * 2, self.core.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, self.d_model * 2, self.core.d_state, device=device)
        return conv_state, ssm_state

    def step(self, y_t_single, y_next_single, conv_state, ssm_state):
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
def train(model, X, Y, dt, dirname="plots", epochs=30):
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
# 🤖 CLOSED-LOOP SIMULATIONS
# =========================================================
def simulate_controller_stateful(model, plant, dt, ref_sequence, dirname="plots", x_scaler=None, y_scaler=None):
    model.eval()
    device = next(model.parameters()).device
    
    plant.reset_time()
    x_state = torch.tensor([[1.0, 1000.0]], device=device) 
    y_log, u_log, ref_log = [], [], []

    conv_state, ssm_state = model.allocate_inference_states(batch_size=1, device=device)
    steps = len(ref_sequence)

    for t in range(steps):
        # Use the true target value from the training data sequence
        y_ref_val = ref_sequence[t]
        y_ref = torch.tensor([[y_ref_val]], device=device)
        y_current = plant.mu(x_state[:, 1], plant.current_time)

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
        x_state = plant.step(x_state, u)

        y_log.append(y_current.item())
        u_log.append(u.item())
        ref_log.append(y_ref.item())

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) (mu)", "reference"],
        xlabel="Time (h)", ylabel="Growth Rate (1/h)", title="Closed-loop Dataset Tracking",
        filename="tracking_dataset", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t) (Glucose)"],
        xlabel="Time (h)", ylabel="Feed Rate (l/h)", title="Control Signal",
        filename="control_dataset", dirname=dirname
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
        xlabel="Time (h)", ylabel="Value", title="Dataset Example",
        filename="dataset", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP CONSTANT STEP SIMULATION
# =========================================================
def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_value=0.015):
    """
    Runs a closed-loop step response simulation tracking a fixed constant value.
    """
    model.eval()
    device = next(model.parameters()).device
    
    plant.reset_time()
    # Initialize state matrix with typical starting conditions (x1_0 = 1.0 g TS, x2_0 = 1000.0 mg S)
    x_state = torch.tensor([[1.0, 1000.0]], device=device) 
    y_log, u_log, ref_log = [], [], []

    # Allocate clean tracking hidden states for Mamba core sequence step
    conv_state, ssm_state = model.allocate_inference_states(batch_size=1, device=device)

    for t in range(steps):
        # Current true measured plant growth rate y_t = mu(x2)
        y_current = plant.mu(x_state[:, 1], plant.current_time)
        
        # The target value for the next step is our fixed constant reference
        y_ref_val = ref_value

        # Pack into numpy array to safely pipe through StandardScaler
        input_pair = np.array([[y_current.item(), y_ref_val]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)
            
        y_t_norm = torch.tensor([[input_pair[0, 0]]], dtype=torch.float32, device=device)
        y_next_norm = torch.tensor([[input_pair[0, 1]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            # Step the Mamba state space engine forward 1 step in sequence time
            u_norm_tensor, conv_state, ssm_state = model.step(
                y_t_norm, y_next_norm, conv_state, ssm_state
            )
        
        u_norm_np = u_norm_tensor.cpu().numpy() 
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np)[0, 0]
        else:
            u = u_norm_np[0, 0]
            
        # ⚠️ Clamp physical constraints: 0 <= u1(t) <= 1.0 l/h
        u = np.clip(u, 0.0, 1.0)
            
        u = torch.tensor(u, dtype=torch.float32, device=device)
        
        # Propagate the real physical environment forward using RK4
        x_state = plant.step(x_state, u)

        # Log real values for metrics visualization
        y_log.append(y_current.item())
        u_log.append(u.item())
        ref_log.append(y_ref_val)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) (mu)", "reference (mu*)"],
        xlabel="Time (h)", ylabel="Growth Rate (1/h)", title=f"Closed-loop Step Response (\u03bc*={ref_value})",
        filename="tracking_constant", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t) (Glucose)"],
        xlabel="Time (h)", ylabel="Feed Rate (l/h)", title="Step Response Control Effort",
        filename="control_constant", dirname=dirname
    )

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=2)
    dt = 0.1
    plant = TrophophaseFermentationPlant(dt=dt)
    
    model = StatefulMambaController().to(device)

    # --- Generate data in Parallel ---
    X, Y = generate_data_parallel(plant)
    plot_dataset(X, Y, dt)

    # 🎯 EXTRACT REFERENCE TRAJECTORY FROM TRAINING DATA
    # We grab the first sequence (index 0) and look at index 1 of the last dim, which is y_next
    ref_sequence = X[0, :, 1].numpy()

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

    # --- Closed-loop simulation using the dataset trajectory ---
    simulate_controller_stateful(
        model, plant, dt, ref_sequence=ref_sequence,
        x_scaler=x_scaler, y_scaler=y_scaler
    )

    # (Optional) You can still run your constant tracking if desired
    simulate_constant_controller(
        model, plant, dt, steps=50,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=0.03  
    )

if __name__ == "__main__":
    main()