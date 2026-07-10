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
# 🛑 GLOBAL CONFIGURATION & ACTUATOR BOUNDS
# =========================================================
U_MIN = 0.0    # Heater completely turned off
U_MAX = 10.0   # Maximum power capacity of the heating element

# =========================================================
# 🔧 CONTINUOUS THERMAL PLANT (WITH OUTPUT FUNCTION)
# =========================================================
class ContinuousThermalPlant:
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # Physical system parameters
        self.tau = 5.0          # Thermal time constant (seconds)
        self.T_ambient = 20.0   # Base ambient fluid temperature (°C)
        self.beta = 1.5         # Heating gain coefficient (°C / kW)
        
        # Internal state: Current fluid temperature (°C)
        self.state = np.array([20.0]) 

    def dynamics(self, state, u):
        """Defines the continuous-time linear ODE: dx/dt = f(x, u)"""
        temp = state[0]
        power = float(u) 
        
        # Temperature rate of change (State derivative)
        dtemp_dt = (self.T_ambient - temp) / self.tau + self.beta * power
        return np.array([dtemp_dt])

    def output_function(self, state):
        """
        Defines y = g(x), mapping the hidden state to a measured output.
        Example: Simulating a transducer converting temperature to a 0-5V signal.
        """
        temp = state[0]
        y_measured = 0.1 * temp + 0.5
        return y_measured

    def step(self, u):
        """Advances the continuous state using RK4 and returns the function-mapped output."""
        dt = self.dt
        x_n = self.state
        
        # RK4 state integration steps
        k1 = self.dynamics(x_n, u)
        k2 = self.dynamics(x_n + 0.5 * dt * k1, u)
        k3 = self.dynamics(x_n + 0.5 * dt * k2, u)
        k4 = self.dynamics(x_n + dt * k3, u)
        
        # Update hidden internal state vector
        self.state = x_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Return the transformed measurement function instead of the state
        return self.output_function(self.state)

    def get_full_states(self):
        """Extracts the underlying hidden state array for logging/monitoring."""
        return self.state
    
# =========================================================
# 📊 DATA GENERATION (CANADAY'S METHOD WITH BOUNDS)
# =========================================================
def generate_data(plant, seq_len=100, num_sequences=10):
    X, Y = [], []
    STATES = []  
    
    dt = plant.dt 
    lambd = 2.0  
    p = 3.0            
    u_nominal = 4.0    

    for _ in range(num_sequences):
        plant.state = np.array([20.0]) 
        
        # Sample values at discrete intervals from a uniform distribution
        raw = torch.rand((1, seq_len)) * 2 - 1

        # Fourier-transform the raw signal to frequency domain
        fft_sig = torch.fft.rfft(raw, dim=1)

        # Extract the frequences corresponding to the FFT components
        freqs = torch.fft.rfftfreq(seq_len, d=dt)
        
        # Define the cutoff frequency for low-pass filtering
        cutoff = 1.0 / lambd

        # Zero out high-frequency components to enforce smoothness
        fft_sig[:, freqs > cutoff] = 0
        
        # Construct training data by inverse Fourier-transforming the filtered signal
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)

        # Normalize the signal to [-1, 1] range for consistent scaling        
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2* (v_train - v_min) / (v_max - v_min) -1
        
        u = (u_nominal + v_norm * p).squeeze(0)  
        u = torch.clamp(u, min=U_MIN, max=U_MAX)

        y_meas = torch.zeros(seq_len + 1)
        state_history = []
        
        # Initial output measurement from the initial state mapping
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
            xlabel="Time (s)", ylabel="Control", title=f"Prediction vs Ground Truth (seq {idx})",
            filename=f"prediction_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (STATEFUL & SATURATED)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_freq=1.0, ref_amplitude=0.5):
    plant.state = np.array([44.0]) 
    y = plant.output_function(plant.get_full_states()) # Measure initial functional output
    y_log, u_log, ref_log = [], [], []

    model.reset()

    for t in range(steps):
        # The reference tracking signal is now specified in terms of our function output space
        # Base target around 4.9V, fluctuating by ref_amplitude
        y_ref = 4.9 + ref_amplitude * np.sin(2 * np.pi * ref_freq * t * dt)

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
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) Function", "reference"],
        xlabel="Time (s)", ylabel="Output Function Value", title="Closed-loop Tracking",
        filename="tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t)"],
        xlabel="Time (s)", ylabel="Control (kW)", title="Control Signal",
        filename="control", dirname=dirname
    )

def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_value=5.3):
    plant.state = np.array([44.0])
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
        t=t_axis, signals=[y_log, ref_log], labels=["y(t) Function", "reference"],
        xlabel="Time (s)", ylabel="Output Function Value", title="Closed-loop Constant Tracking",
        filename="constant_tracking", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u_log], labels=["u(t)"],
        xlabel="Time (s)", ylabel="Control (kW)", title="Constant Control Signal",
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
            xlabel="Time (s)", ylabel="Value", title=f"Dataset Example Sequence {idx}",
            filename=f"dataset_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=2)
    rpy.set_seed(2) 
    dt = 0.1
    plant = ContinuousThermalPlant(dt=dt)
    
    model = build_reservoir_controller(units=300, lr=0.3, sr=0.9)

    X, Y, _ = generate_data(plant)
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

    # Note: Closed-loop references are now scaled to match the output sensor function bounds
    simulate_controller_stateful(
        model, plant, dt,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_freq=0.05,        
        ref_amplitude=0.4     
    )

    simulate_constant_controller(
        model, plant, dt,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_value=5.1         
    )

if __name__ == "__main__":
    main()