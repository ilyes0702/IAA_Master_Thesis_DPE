import numpy as np
import torch  
from sklearn.preprocessing import StandardScaler 
from src.sample.utils.plotting_utils import plot_signals
from matplotlib import pyplot as plt
from src.sample.utils.general_utils import seed_everything
import math
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
# Control Input u1(t) is the glucose feed rate (l/h)
U_MIN = 0.0    
U_MAX = 1.0    # Capped at u1,max = 1 l/h per problem statement

# =========================================================
# 🧫 TROPHOPHASE FERMENTATION PLANT
# =========================================================
class TrophophasePlant:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.t = 0.0  # Internal clock to compute time-varying volume V(t)
        
        # System Parameters from Table 1
        self.mu_max = 0.12       # Maximum specific growth rate (1/h)
        self.K_s = 50.0          # Half-saturation constant (mg S/l)
        self.m_s = 23.0          # Maintenance coefficient (mg S / (g TS h))
        self.p1 = 0.0047         # Yield-related coefficient (g TS / mg S)
        self.p2 = 200000.0       # Substrate feed concentration factor (mg S / l)
        
        # Initial states: [Biomass x1 (g), Substrate x2 (mg)]
        self.initial_state = np.array([1500.0, 2000.0]) 
        self.state = self.initial_state.copy()

    def get_V(self, t):
        """Computes the time-varying reactor volume V(t) using step functions."""
        # V(t) = 150 + 2*(t-5)*sigma(t-5) - 2*(t-15)*sigma(t-15)
        V = 150.0
        if t > 5.0:
            V += 2.0 * (t - 5.0)
        if t > 15.0:
            V -= 2.0 * (t - 15.0)
        return V

    def dynamics(self, state, u, t):
        """Defines the trophophase continuous-time nonlinear bio-kinetics."""
        x1_biomass = state[0]
        x2_substrate = state[1]
        u1_feed = float(u)
        
        V = self.get_V(t)
        
        # Monod growth kinetics equation
        mu = (self.mu_max * x2_substrate) / (self.K_s * V + x2_substrate)
        
        # Differential equations
        dx1_dt = mu * x1_biomass
        dx2_dt = -(1.0 / self.p1) * mu * x1_biomass - self.m_s * x1_biomass + self.p2 * u1_feed
        
        return np.array([dx1_dt, dx2_dt])

    def output_function(self, state, t):
        """The controlled variable during the trophophase is the growth rate y1 = mu(x2)."""
        x1_biomass = state[0]
        x2_substrate = state[1]
        V = self.get_V(t)
        mu = (self.mu_max * x2_substrate) / (self.K_s * V + x2_substrate)
        return mu

    def step(self, u):
        """Advances the continuous states using RK4 tracking explicit time."""
        dt = self.dt
        x_n = self.state
        t = self.t
        
        # RK4 state integration steps taking time-varying V(t) into account
        k1 = self.dynamics(x_n, u, t)
        k2 = self.dynamics(x_n + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self.dynamics(x_n + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self.dynamics(x_n + dt * k3, u, t + dt)
        
        # Update internal state vector safely
        self.state = np.clip(x_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0, None)
        self.t += dt
        
        return self.output_function(self.state, self.t)

    def reset(self):
        """Resets the plant states and internal clock back to initial conditions."""
        self.state = self.initial_state.copy()
        self.t = 0.0

    def get_full_states(self):
        """Extracts the underlying state array for logging."""
        return self.state
    
# =========================================================
# 📊 DATA GENERATION (CANADAY'S METHOD WITH BOUNDS)
# =========================================================
def generate_data(plant, seq_len, num_sequences=100):
    X, Y = [], []
    STATES = []  
    
    dt = plant.dt 
    lambd = 4        # Smoothness factor adjusted for biological system inertia
    p = 0.5           # Amplitude scale around the nominal value

    for _ in range(num_sequences):
        plant.reset()
        u_nominal = 0.6 + 0.3 * torch.rand(1)  # Center of control bounds [0.8, 0.9]
        
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
        y_meas[0] = plant.output_function(plant.get_full_states(), plant.t)
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


        # Extract just y_t (shape: seq_len) and u (shape: seq_len)
        y_t_series = x[:, 0]  
        u_series = y_target.squeeze(-1) 

        # Calculate DTW for this specific sequence pair
        from tslearn.metrics import dtw
        dtw_score = dtw(y_t_series, u_series)
        #print(f"Sequence DTW score: {dtw_score}")
        
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
            xlabel="Time (h)", ylabel="Glucose Feed Rate (l/h)", title=f"Prediction vs Ground Truth (seq {idx})",
            filename=f"prediction_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (STATEFUL & SATURATED)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=2001, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=0.02, ref_amplitude=0.005):
    plant.reset()
    y = plant.output_function(plant.get_full_states(), plant.t) 
    y_log, u_log, ref_log = [], [], []

    model.reset()

    for t in range(steps):
        # Dynamic tracking target around the prescribed growth rate 0.015
        #y_ref = 0.015 + ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)

        y_ref = plant.initial_state[0] * math.exp(0.015*t*dt)  # Scale reference to biomass level
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
        t=t_axis, signals=[y_log, ref_log], labels=["y1(t) Growth Rate", "reference"],
        xlabel="Time (h)", ylabel="Growth Rate mu (1/h)", title="Closed-loop Dynamic Tracking",
        filename="tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u1(t)"],
        xlabel="Time (h)", ylabel="Glucose Feed Rate (l/h)", title="Control Signal (Glucose Feed)",
        filename="control", dirname=dirname
    )

def simulate_constant_controller(model, plant, dt, steps=2001, dirname="plots", x_scaler=None, y_scaler=None, ref_value=0.015):
    plant.reset()
    y = plant.output_function(plant.get_full_states(), plant.t)
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
        t=t_axis, signals=[y_log, ref_log], labels=["y1(t) Growth Rate", "reference"],
        xlabel="Time (h)", ylabel="Growth Rate mu (1/h)", title="Closed-loop Constant Growth Rate Tracking",
        filename="constant_tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u1(t)"],
        xlabel="Time (h)", ylabel="Glucose Feed Rate (l/h)", title="Constant Feed Control",
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
    dt = 0.01 
    plant = TrophophasePlant(dt=dt)
    
    model = build_reservoir_controller(units=400, lr=0.2, sr=0.95) 

    X, Y, _ = generate_data(plant, seq_len=2001, num_sequences=12)
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


    # Tracking simulation with slow biological tracking variations
    simulate_controller_stateful(
        model, plant, dt, steps=2001,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_freq=0.005,        
        ref_amplitude=0.003     
    )

    # Constant target simulation matching task description (mu* = 0.015)
    simulate_constant_controller(
        model, plant, dt, steps=2001,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=0.015        
    )

if __name__ == "__main__":
    main()