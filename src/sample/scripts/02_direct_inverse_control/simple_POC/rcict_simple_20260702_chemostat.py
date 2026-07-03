import numpy as np
import torch
from sklearn.preprocessing import StandardScaler 
from src.sample.utils.plotting_utils import plot_signals
from matplotlib import pyplot as plt
from src.sample.utils.general_utils import seed_everything

# --- ReservoirPy Imports ---
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

try:
    plt.style.use("src/sample/style.mplstyle")
except:
    pass

# =========================================================
# 🔧 CHEMOSTAT PLANT
# =========================================================
class ChemostatPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Biological Parameters from Config
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu-max"], device=self.device)
        self.Ks = torch.tensor(hyperparam_config["plant"]["Ks"], device=self.device)
        self.Y = torch.tensor(hyperparam_config["plant"]["Y"], device=self.device)
        self.sR = torch.tensor(hyperparam_config["plant"]["sR"], device=self.device)

    def get_initial_state(self, batch_size):
        """Returns [batch_size, 2] tensor of [Biomass (x), Substrate (s)]."""
        x_init = torch.rand((batch_size, 1), device=self.device) * 0.2 + 0.2 
        s_init = torch.rand((batch_size, 1), device=self.device) * 0.2 + 0.1 
        return torch.cat([x_init, s_init], dim=1)

    def get_y(self, state, t=None):
        """Calculates and returns Growth Rate (mu) as the observable output."""
        s = state[:, 1:2]
        mu = (self.mu_max * s) / (self.Ks + s)
        return mu 

    def dynamics(self, x, s, u, t=None):
        mu = (self.mu_max * s) / (self.Ks + s)
        dxdt = mu * x - u * x
        dsdt = u * (self.sR - s) - (mu * x / self.Y)
        return dxdt, dsdt

    def step(self, state, u, t, dt):
        """Standard Runge-Kutta 4th Order numerical integration."""
        x, s = state[:, 0:1], state[:, 1:2]
        
        dx1, ds1 = self.dynamics(x, s, u)
        dx2, ds2 = self.dynamics(x + 0.5*dt*dx1, s + 0.5*dt*ds1, u)
        dx3, ds3 = self.dynamics(x + 0.5*dt*dx2, s + 0.5*dt*ds2, u)
        dx4, ds4 = self.dynamics(x + dt*dx3, s + dt*ds3, u)
        
        x_next = x + (dt/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        s_next = s + (dt/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
        
        state_next = torch.cat([x_next, s_next], dim=1)
        return state_next, self.get_y(state_next)

# =========================================================
# 📊 DATA GENERATION (CANADAY'S METHOD)
# =========================================================
def generate_data(plant, seq_len=100, num_sequences=10000):
    X, Y = [], []
    dt = plant.dt
    lambd = 2.0  
    p = 0.2      

    for _ in range(num_sequences):
        raw = torch.rand((1, seq_len), device=plant.device) * 2 - 1
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, d=dt, device=plant.device)
        
        cutoff = 1.0 / lambd
        fft_sig[:, freqs > cutoff] = 0
        
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
        
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min) - 1
        
        # Ensure dilution rate (u) stays positive for chemostat safety
        u = ((v_norm * p) + 0.25).squeeze(0)  

        # Initialize tracking states
        state = plant.get_initial_state(batch_size=1)
        y_signals = torch.zeros(seq_len + 1, device=plant.device)
        y_signals[0] = plant.get_y(state).item()

        for t in range(seq_len):
            u_t = u[t].unsqueeze(0).unsqueeze(1) # Batch shape [1, 1]
            state, y_next_t = plant.step(state, u_t, t, dt)
            y_signals[t+1] = y_next_t.item()
            
        y_t = y_signals[:-1].cpu()
        y_next = y_signals[1:].cpu()
        
        x = np.stack([y_t.numpy(), y_next.numpy()], axis=-1)  
        y_target = u.cpu().unsqueeze(-1).numpy()                    
        
        X.append(x)
        Y.append(y_target)
        
    return X, Y

# =========================================================
# 🧠 RESERVOIR COMPUTING CONTROLLER
# =========================================================
def build_reservoir_controller(units=300, lr=0.3, sr=0.9, ridge_reg=1e-6):
    reservoir = Reservoir(units=units, lr=lr, sr=sr, rc_connectivity=0.1)
    readout = Ridge(ridge=ridge_reg)
    return reservoir >> readout

def train_reservoir(model, X, Y):
    print("--- Training Reservoir (Analytical Fit) ---")
    model.fit(X, Y, warmup=10)
    print("Training complete!")
    return [0.0]

def plot_prediction(model, X, Y, dt, dirname="plots"):
    x_sample = X[0]
    y_true_np = Y[0].squeeze()

    model.reset()
    y_pred = model.run(x_sample)
    y_pred_np = y_pred.squeeze()

    t_axis = np.arange(len(y_true_np)) * dt
    plot_signals(
        t=t_axis, signals=[y_true_np, y_pred_np], labels=["True u(t)", "Predicted u(t)"],
        xlabel="Time (s)", ylabel="Control", title="Prediction vs Ground Truth",
        filename="prediction", dirname=dirname
    )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (STATEFUL)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=1.0, ref_amplitude=0.05):
    # Setup initial biological condition
    state = plant.get_initial_state(batch_size=1)
    y = plant.get_y(state).item()
    
    y_log, u_log, ref_log = [], [], []
    model.reset()

    for t in range(steps):
        # Sine tracking target for output growth rate
        y_ref = 0.1 + ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)

        input_pair = np.array([[y, y_ref]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)
            
        x_step = np.array([input_pair[0, 0], input_pair[0, 1]])
        u_norm_np = model(x_step) 
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np[0]

        # Evolve torch biological plant
        u_tensor = torch.tensor([[u]], device=plant.device, dtype=torch.float32)
        state, y_tensor = plant.step(state, u_tensor, t, dt)
        y = y_tensor.item()

        y_log.append(y)
        u_log.append(u)
        ref_log.append(y_ref)

    t_axis = np.arange(steps) * dt
    plot_signals(t=t_axis, signals=[y_log, ref_log], labels=["μ(t)", "μ_ref"], xlabel="Time (s)", ylabel="Growth Rate", title="Closed-loop Tracking", filename="tracking", dirname=dirname)
    plot_signals(t=t_axis, signals=[u_log], labels=["D(t)"], xlabel="Time (s)", ylabel="Dilution Rate", title="Control Signal", filename="control", dirname=dirname)

def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_value=0.1):
    state = plant.get_initial_state(batch_size=1)
    y = plant.get_y(state).item()
    
    y_log, u_log, ref_log = [], [], []
    model.reset()

    for t in range(steps):
        y_ref = ref_value

        input_pair = np.array([[y, y_ref]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)

        x_step = np.array([input_pair[0, 0], input_pair[0, 1]])
        u_norm_np = model(x_step)
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np[0]

        u_tensor = torch.tensor([[u]], device=plant.device, dtype=torch.float32)
        state, y_tensor = plant.step(state, u_tensor, t, dt)
        y = y_tensor.item()

        y_log.append(y)
        u_log.append(u)
        ref_log.append(y_ref)

    t_axis = np.arange(steps) * dt
    plot_signals(t=t_axis, signals=[y_log, ref_log], labels=["μ(t)", "μ_ref"], xlabel="Time (s)", ylabel="Growth Rate", title="Constant Tracking", filename="constant_tracking", dirname=dirname)

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=2)
    rpy.set_seed(2) 
    
    # ⚙️ Mocking config structure expected by ChemostatPlant
    config = {
        "train": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        "signal": {"dt": 0.1},
        "plant": {
            "mu-max": 0.5, # Max specific growth rate
            "Ks": 0.2,     # Half-velocity constant
            "Y": 0.6,      # Yield coefficient 
            "sR": 1.0      # Influent substrate concentration
        }
    }
    
    dt = config["signal"]["dt"]
    plant = ChemostatPlant(config)
    model = build_reservoir_controller(units=400, lr=0.2, sr=0.8) # Adjusted for biology temporal scales

    # --- Pipeline Flow ---
    X, Y = generate_data(plant, seq_len=150, num_sequences=15)
    
    X_flat, Y_flat = np.vstack(X), np.vstack(Y)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaler.fit(X_flat)
    y_scaler.fit(Y_flat)

    X_normalized = [x_scaler.transform(seq) for seq in X]
    Y_normalized = [y_scaler.transform(seq) for seq in Y]

    train_reservoir(model, X_normalized, Y_normalized)
    plot_prediction(model, X_normalized, Y_normalized, dt)

    simulate_controller_stateful(model, plant, dt, x_scaler=x_scaler, y_scaler=y_scaler, ref_freq=0.2, ref_amplitude=0.04)
    simulate_constant_controller(model, plant, dt, x_scaler=x_scaler, y_scaler=y_scaler, ref_value=0.1)
    print("Pipeline execution complete!")
if __name__ == "__main__":
    main()
    