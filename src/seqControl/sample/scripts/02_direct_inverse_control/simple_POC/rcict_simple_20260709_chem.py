import numpy as np
import torch  
from sklearn.preprocessing import StandardScaler 
from seqControl.sample.utils.plotting_utils import plot_signals
from matplotlib import pyplot as plt
from seqControl.sample.utils.general_utils import seed_everything

# --- ReservoirPy Imports ---
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

try:
    plt.style.use("src/sample/style.mplstyle")
except:
    pass

# =========================================================
# 🛑 GLOBAL CONFIGURATION & ACTUATOR BOUNDS
# =========================================================
# Control Input u(t) is the dilution rate (1/hour)
U_MIN = 0.05    # Minimum flow rate to prevent cell starvation
U_MAX = 0.45    # Maximum flow rate capped to prevent complete cell washout

# =========================================================
# 🧫 CONTINUOUS CHEMOSTAT PLANT (MONOD KINETICS)
# =========================================================
class ContinuousChemostatPlant:
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # Kinetic and Physical parameters
        self.mu_max = 0.5      # Maximum specific growth rate (1/h)
        self.K_s = 0.2         # Half-saturation constant (g/L)
        self.Y_xs = 0.6        # Yield coefficient (biomass produced per gram of nutrient)
        self.S_in = 5.0        # Inflow nutrient concentration (g/L)
        
        # Internal states: [Biomass concentration X (g/L), Nutrient concentration S (g/L)]
        self.state = np.array([1.5, 2.0]) 

    def dynamics(self, state, u):
        """Defines the continuous-time nonlinear bio-kinetics: dx/dt = f(x, u)"""
        X_biomass = state[0]
        S_nutrient = state[1]
        dilution_rate = float(u) 
        
        # Specific growth rate via Monod equation
        mu = (self.mu_max * S_nutrient) / (self.K_s + S_nutrient)
        
        # Differential equations
        dX_dt = (mu - dilution_rate) * X_biomass
        dS_dt = dilution_rate * (self.S_in - S_nutrient) - (mu * X_biomass) / self.Y_xs
        
        return np.array([dX_dt, dS_dt])

    def output_function(self, state):
        """
        Defines y = g(x), mapping hidden states to a measured output.
        Simulates an optical density (OD) sensor measuring Biomass and outputting 0-5V.
        """
        X_biomass = state[0]
        y_measured = 1.2 * X_biomass + 0.1
        return y_measured

    def step(self, u):
        """Advances the continuous states using RK4 and returns the function-mapped output."""
        dt = self.dt
        x_n = self.state
        
        # RK4 state integration steps
        k1 = self.dynamics(x_n, u)
        k2 = self.dynamics(x_n + 0.5 * dt * k1, u)
        k3 = self.dynamics(x_n + 0.5 * dt * k2, u)
        k4 = self.dynamics(x_n + dt * k3, u)
        
        # Update internal hidden state vector safely (clipping at 0 to prevent unphysical negative values)
        self.state = np.clip(x_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0, None)
        
        return self.output_function(self.state)

    def get_full_states(self):
        """Extracts the underlying hidden state array for logging/monitoring."""
        return self.state
    
# =========================================================
# 📊 DATA GENERATION (CANADAY'S METHOD WITH BOUNDS)
# =========================================================
def generate_data(plant, seq_len, num_sequences=10):
    X, Y = [], []
    STATES = []  
    
    dt = plant.dt 
    lambd = 4.0        # Smoothness factor adjusted for biological inertia
    p = 0.25           # Amplitude scale around the nominal value
    u_nominal = 0.35   # Stable operational nominal dilution rate

    for _ in range(num_sequences):
        plant.state = np.array([1.5, 2.0]) # Reset to stable starting operating point
        
        # Sample raw signal
        raw = torch.rand((1, seq_len)) * 2 - 1

        # FFT processing for low-pass filtering
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, d=dt)
        cutoff = 1.0 / lambd
        fft_sig[:, freqs > cutoff] = 0
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)

        # Normalize 
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min) - 1
        
        u = (u_nominal + v_norm * p).squeeze(0)  
        u = torch.clamp(u, min=U_MIN, max=U_MAX)

        y_meas = torch.zeros(seq_len + 1)
        state_history = []
        
        # Initial tracking measurements
        y_meas[0] = plant.output_function(plant.get_full_states())
        state_history.append(plant.get_full_states().copy())

        for t in range(seq_len):
            y_meas[t+1] = plant.step(u[t].item())
            state_history.append(plant.get_full_states().copy())
            
        y_t = y_meas[:-1]
        y_next = y_meas[1:]
        
        x = np.stack([y_t.numpy(), y_next.numpy()], axis=-1)   
        y_target = u.unsqueeze(-1).numpy()                    
        seq_states = np.array(state_history[:-1])              
        
        X.append(x)
        Y.append(y_target)
        STATES.append(seq_states)
        
    return X, Y, STATES

# =========================================================
# 🧠 RESERVOIR COMPUTING CONTROLLER
# =========================================================
def build_reservoir_controller(units=300, lr=0.3, sr=0.9, ridge_reg=1e-6):
    reservoir = Reservoir(units=units, lr=lr, sr=sr, rc_connectivity=0.1)
    readout = Ridge(ridge=ridge_reg)
    return reservoir >> readout

# =========================================================
# 📉 TRAINING FUNCTION
# =========================================================
def train_reservoir(model, X, Y):
    print("--- Training Reservoir (Analytical Fit) ---")
    model.fit(X, Y, warmup=10)
    print("Training complete!")
    return [0.0]

# =========================================================
# 🔍 PREDICTION PLOT 
# =========================================================
def plot_prediction(model, X, Y, dt, dirname="plots"):
    for idx, (x_seq, y_seq) in enumerate(zip(X, Y)):
        y_true_np = y_seq.squeeze()

        model.reset()
        y_pred = model.run(x_seq)
        y_pred_np = y_pred.squeeze()

        t_axis = np.arange(len(y_true_np)) * dt
        plot_signals(
            t=t_axis, signals=[y_true_np, y_pred_np], labels=["True u(t)", "Predicted u(t)"],
            xlabel="Time (h)", ylabel="Dilution Rate (1/h)", title=f"Prediction vs Ground Truth (seq {idx})",
            filename=f"prediction_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (STATEFUL & SATURATED)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=0.02, ref_amplitude=0.3):
    plant.state = np.array([2.0, 1.5]) 
    y = plant.output_function(plant.get_full_states()) 
    y_log, u_log, ref_log = [], [], []

    model.reset()

    for t in range(steps):
        # Tracking trajectory for the OD Sensor reading space
        # Base sensor target around 2.5V, fluctuating due to reference dynamics
        y_ref = 2.5 + ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)

        input_pair = np.array([[y, y_ref]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)
            
        x_step = np.array([input_pair[0, 0], input_pair[0, 1]])
        u_norm_np = model(x_step) 
        
        if y_scaler:
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np[0]

        u = np.clip(u, U_MIN, U_MAX)
        y = plant.step(u)

        y_log.append(y)
        u_log.append(u)
        ref_log.append(y_ref)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) OD Sensor", "reference"],
        xlabel="Time (h)", ylabel="Sensor Output (V)", title="Closed-loop Bioreactor Tracking",
        filename="tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t)"],
        xlabel="Time (h)", ylabel="Dilution Rate (1/h)", title="Control Signal (Flow)",
        filename="control", dirname=dirname
    )

def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_value=2.8):
    plant.state = np.array([2.0, 1.5])
    y = plant.output_function(plant.get_full_states())
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

        u = np.clip(u, U_MIN, U_MAX)
        y = plant.step(u)

        y_log.append(y)
        u_log.append(u)
        ref_log.append(y_ref)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) OD Sensor", "reference"],
        xlabel="Time (h)", ylabel="Sensor Output (V)", title="Closed-loop Constant Biomass Tracking",
        filename="constant_tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t)"],
        xlabel="Time (h)", ylabel="Dilution Rate (1/h)", title="Constant Dilution Control",
        filename="constant_control", dirname=dirname
    )

# =========================================================
# 📊 DATA VISUALIZATION
# =========================================================
def plot_dataset(X, Y, dt, dirname="plots"):
    for idx, (x, y) in enumerate(zip(X, Y)):
        y_t = x[:, 0]
        y_next = x[:, 1]
        u = y[:, 0]
        t_axis = np.arange(len(y_t)) * dt
        plot_signals(
            t=t_axis, signals=[y_t, y_next, u], labels=["y(t)", "y(t+Δ)", "u(t)"],
            xlabel="Time (h)", ylabel="Value", title=f"Dataset Example Sequence {idx}",
            filename=f"dataset_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=42)
    rpy.set_seed(42) 
    dt = 0.1 # Bioreactors evolve slower than thermal models, higher dt works better here
    plant = ContinuousChemostatPlant(dt=dt)
    
    model = build_reservoir_controller(units=400, lr=0.2, sr=0.95) # Tweaked reservoir hyperparameters for nonlinear system

    X, Y, _ = generate_data(plant, seq_len=200, num_sequences=12)
    plot_dataset(X, Y, dt)

    X_flat = np.vstack(X)
    Y_flat = np.vstack(Y)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_flat)
    y_scaler.fit(Y_flat)

    X_normalized = [x_scaler.transform(seq) for seq in X]
    Y_normalized = [y_scaler.transform(seq) for seq in Y]

    train_reservoir(model, X_normalized, Y_normalized)

    plot_prediction(model, X_normalized, Y_normalized, dt)

    # Tracking simulation with slower biological system frequencies
    simulate_controller_stateful(
        model, plant, dt, steps=100,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_freq=0.01,        
        ref_amplitude=0.5     
    )

    simulate_constant_controller(
        model, plant, dt, steps=100,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=3.0        
    )

if __name__ == "__main__":
    main()