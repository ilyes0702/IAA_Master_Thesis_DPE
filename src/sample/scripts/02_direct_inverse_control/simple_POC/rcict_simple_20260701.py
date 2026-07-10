import numpy as np
import torch  # Kept purely to maintain your plant execution data shapes if desired
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
# 🔧 SIMPLE PLANT
# =========================================================
class SimplePlant:
    def __init__(self, a=0.9, b=0.1):
        self.a = a
        self.b = b

    def step(self, y, u):
        # Keeps compatibility with both scalar floats and numpy/torch primitives
        y_val = float(y)
        u_val = float(u)
        return self.a * y_val * y_val * 0.2 / (1 + y_val) + self.b * u_val

import numpy as np

class ContinuousChemostatPlant:
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # Physical/Biological parameters (Pseudomonas putida degrading phenol)
        self.mu_max = 0.45    # Max growth rate (h^-1)
        self.K_s = 20.0       # Half-saturation constant (mg/L)
        self.K_i = 250.0      # Inhibition constant (mg/L)
        self.Y = 0.60         # Yield coefficient (mg/mg)
        self.s_in = 500.0     # Influent concentration (mg/L)
        
        # Internal states: [Biomass (x), Substrate (s)]
        # Initialized at a standard operational steady-state
        self.state = np.array([297.0, 5.0]) 

    def _biomass_growth_rate(self, s):
        """Haldane kinetics model for growth rate with substrate inhibition."""
        return (self.mu_max * s) / (self.K_s + s + (s**2 / self.K_i))

    def dynamics(self, state, u):
        """
        Defines the continuous-time ODE equations: dx/dt = f(x, u)
        State vector:  state[0] = x (Biomass), state[1] = s (Substrate)
        Control Input: u = D (Dilution rate)
        """
        x = state[0]
        s = state[1]
        D = float(u) # Control input is the dilution rate
        
        mu = self._biomass_growth_rate(s)
        
        # dx/dt = (mu - D)*x
        d_biomass = (mu - D) * x
        
        # ds/dt = D*(s_in - s) - (mu*x)/Y
        d_substrate = D * (self.s_in - s) - (mu * x) / self.Y
        
        return np.array([d_biomass, d_substrate])

    def step(self, u):
        """
        Advances the continuous system by dt using 4th-Order Runge-Kutta (RK4).
        Returns the new measured system output (y), which is the substrate concentration.
        """
        dt = self.dt
        x_n = self.state
        
        # RK4 Coefficients
        k1 = self.dynamics(x_n, u)
        k2 = self.dynamics(x_n + 0.5 * dt * k1, u)
        k3 = self.dynamics(x_n + 0.5 * dt * k2, u)
        k4 = self.dynamics(x_n + dt * k3, u)
        
        # Update internal state vector
        self.state = x_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # In wastewater control, our measured tracking output 'y' is 
        # usually the effluent pollutant concentration (substrate 's')
        y_measured = self.state[1] 
        
        return y_measured

    def get_full_states(self):
        """Helper to extract hidden internal states for your logging utility."""
        return self.state
    
# =========================================================
# 📊 DATA GENERATION (CANADAY'S METHOD)
# =========================================================
def generate_data(plant, seq_len=100, num_sequences=10):
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
        
        # Convert directly to numpy for ReservoirPy compatibility
        x = np.stack([y_t.numpy(), y_next.numpy()], axis=-1)  # [seq_len, 2]
        y_target = u.unsqueeze(-1).numpy()                    # [seq_len, 1]
        
        X.append(x)
        Y.append(y_target)
        
    return X, Y  # Returns a list of 2D arrays

# =========================================================
# 🧠 RESERVOIR COMPUTING CONTROLLER
# =========================================================
def build_reservoir_controller(units=300, lr=0.3, sr=0.9, ridge_reg=1e-6):
    """
    Creates an Echo State Network (ESN) using ReservoirPy nodes.
    - units: dimension of the reservoir (replaces d_model)
    - lr: leaking rate (controls timescale dynamics)
    - sr: spectral radius (controls memory echo duration)
    """
    reservoir = Reservoir(units=units, lr=lr, sr=sr, rc_connectivity=0.1)
    readout = Ridge(ridge=ridge_reg)
    
    # Connect them into a single pipeline node
    esn_model = reservoir >> readout
    return esn_model

# =========================================================
# 📉 TRAINING FUNCTION
# =========================================================
def train_reservoir(model, X, Y):
    """
    Trains an ESN analytically via ridge regression.
    ReservoirPy handles lists of sequences out-of-the-box, resetting
    internal states automatically between distinct sequences.
    """
    print("--- Training Reservoir (Analytical Fit) ---")
    # Warmup clears transient dynamics from initial zero-states
    model.fit(X, Y, warmup=10)
    print("Training complete!")
    return [0.0]  # Returns dummy list since training is instant (one-shot calculation)

# =========================================================
# 🔍 PREDICTION PLOT 
# =========================================================
def plot_prediction(model, X, Y, dt, dirname="plots"):
    # Grab the first sequence from validation pool
    x_sample = X[0]
    y_true_np = Y[0].squeeze()

    # Reset internal memory state before verification run
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
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=1.0, ref_amplitude=0.18):
    y = 0.0
    y_log, u_log, ref_log = [], [], []

    # 🟢 1. Reset reservoir state before real-time simulation tracking loop
    model.reset()

    for t in range(steps):
        y_ref = ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)

        # 2. Scale current structural values
        input_pair = np.array([[y, y_ref]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)
            
        # 🟢 3. Convert current step to standard 1D shape [features,] (Fixes the ValueError)
        x_step = np.array([input_pair[0, 0], input_pair[0, 1]])

        # 🟢 4. Forward single step using internal memory state persistence
        u_norm_np = model(x_step) # Returns a 1D array of shape (1,)
        
        if y_scaler:
            # Reshape back to 2D for sklearn scaler's requirement
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np[0]

        # 5. Plant evolution update
        y = plant.step(y, u)

        # Log metrics
        y_log.append(y)
        u_log.append(u)
        ref_log.append(y_ref)

    # --- Plotting ---
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
    y = 0.0
    y_log, u_log, ref_log = [], [], []
    
    # 🟢 Reset reservoir state
    model.reset()

    for t in range(steps):
        y_ref = ref_value

        input_pair = np.array([[y, y_ref]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)

        # 🟢 Convert current step to standard 1D shape [features,] (Fixes the ValueError)
        x_step = np.array([input_pair[0, 0], input_pair[0, 1]])

        # Execute single step update
        u_norm_np = model(x_step)
        
        if y_scaler:
            # Reshape back to 2D for sklearn scaler's requirement
            u = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0, 0]
        else:
            u = u_norm_np[0]

        # Step the plant forward
        y = plant.step(y, u)

        # Log metrics
        y_log.append(y)
        u_log.append(u)
        ref_log.append(y_ref)

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
    x = X[0]
    y = Y[0]
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
    rpy.set_seed(2) # Ensure ReservoirPy weights are initialized deterministically
    dt = 0.1
    plant = ContinuousChemostatPlant(dt=dt)
    
    # Instantiate the Echo State Network model
    model = build_reservoir_controller(units=300, lr=0.3, sr=0.9)

    # --- Generate data ---
    X, Y = generate_data(plant)
    plot_dataset(X, Y, dt)

    # --- Normalize data ---
    # Concatenate all sequences together to fit standard scalers cleanly
    X_flat = np.vstack(X)
    Y_flat = np.vstack(Y)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_flat)
    y_scaler.fit(Y_flat)

    # Transform data keeping list structures isolated
    X_normalized = [x_scaler.transform(seq) for seq in X]
    Y_normalized = [y_scaler.transform(seq) for seq in Y]

    # --- Train (Analytical Least-Squares Engine) ---
    train_reservoir(model, X_normalized, Y_normalized)

    # --- Prediction ---
    plot_prediction(model, X_normalized, Y_normalized, dt)

    # --- Closed-loop simulation ---
    simulate_controller_stateful(
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