import numpy as np
import torch  
from sklearn.preprocessing import StandardScaler 
from src.sample.utils.plotting_utils import plot_signals
from matplotlib import pyplot as plt
from src.sample.utils.general_utils import seed_everything
import math
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

try:
    plt.style.use("src/sample/style.mplstyle")
except:
    pass

# =========================================================
# 🛑 GLOBAL CONFIGURATION & ACTUATOR BOUNDS
# =========================================================
# The paper utilizes continuous optogenetic inputs (e.g., light intensities) 
# to control consortium compositions. We establish bounds for 2 light inputs.
U_MIN = 0.0    
U_MAX = None   

# =========================================================
# 🧫 CO-CULTIVATION CYBERGENETIC PLANT
# =========================================================
class CoCultivationPlant:
    """
    Implements a two-strain microbial consortium in a chemostat with 
    light-mediated (optogenetic) growth control as outlined in the paper's case study.
    """
    def __init__(self, dt=1):
        self.dt = dt
        self.t = 0.0  

        # Bioprocess Operating Parameters
        
        # Kinetic Parameters for Strain 1 & Strain 2
        self.mu_max1 = 0.982       # Max growth rate strain 1 (1/h)
        self.mu_max2 = 0.982       # Max growth rate strain 2 (1/h)
        self.k_g_1 = 2.964e-4
        self.k_g_2 = 2.964e-4
        self.f_c = 1100
        self.k_a_1 = 1.7
        self.k_a_2 = 0.182
        self.Y_g_b1 = 10.18
        self.Y_g_b2 = 10.18
        self.q_a_max_1 = 0.337
        self.q_a_max_2 = 0.036
        self.n_1 = 2
        self.k_I_1 = 1.052
        self.n_2 = 4.865
        self.k_I_2 = 1.34
        self.d_l = 0.15
        self.S_in = 200
        self.d_a_1 = 0
        self.d_a_2 = 0
        

    
        # Initial states: [Biomass X1 (g/L), Biomass X2 (g/L), Substrate S (g/L)]
        self.initial_state = np.array([0.005, 0.005, 1, 1.545e-2, 1.655e-3]) 
        self.state = self.initial_state.copy()

    def dynamics(self, state, u, t):
        """Defines the continuous-time co-culture differential equations."""
        X1 = state[0]
        X2 = state[1]
        S  = state[2]
        A1 = state[3]
        A2 = state[4]
        
        # u is now an array-like configuration of two light intensities [u1, u2]
        I_1 = float(u[0])
        I_2 = float(u[1])
        
        # Optogenetically-modulated Monod Kinetics 
        # Light inputs attenuate the specific growth rates (cybergenetic inhibition)
        mu1 = ((self.mu_max1 * S) / (self.k_g_1 + S)) * (self.f_c * A1 / (self.f_c*A1 + self.k_a_1))
        mu2 = ((self.mu_max2 * S) / (self.k_g_2 + S)) * (self.f_c * A2 / (self.f_c*A2 + self.k_a_2))
        
        q_g_1 = self.Y_g_b1 * mu1
        q_g_2 = self.Y_g_b2 * mu2

        q_a_1 = self.q_a_max_1 * (I_1**(self.n_1)/ (I_1**(self.n_1) + self.k_I_1**(self.n_1)))
        q_a_2 = self.q_a_max_2 * (I_2**(self.n_2)/ (I_2**(self.n_2) + self.k_I_2**(self.n_2)))
        # Chemostat Mass Balance Differential Equations
        dX1_dt = (mu1 - self.d_l) * X1
        dX2_dt = (mu2 - self.d_l) * X2
        dS_dt  = self.d_l * (self.S_in - S) - q_g_1 * X1 - q_g_2 * X2
        dA1_dt = q_a_1 - (self.d_a_1 + mu1) * A1
        dA2_dt = q_a_2 - (self.d_a_2 + mu2) * A2

        return np.array([dX1_dt, dX2_dt, dS_dt, dA1_dt, dA2_dt])

    def output_function(self, state, t):
        """
        The controlled variables are the individual biomass values 
        of the two strains to manage population balancing.
        """
        return state[0:2] # Returns [X1, X2]

    def step(self, u):
        """Advances the continuous states using RK4 integration."""
        dt = self.dt
        x_n = self.state
        t = self.t
        
        k1 = self.dynamics(x_n, u, t)
        k2 = self.dynamics(x_n + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self.dynamics(x_n + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self.dynamics(x_n + dt * k3, u, t + dt)
        
        self.state = np.clip(x_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0, None)
        self.t += dt
        
        return self.output_function(self.state, self.t)

    def reset(self):
        """Resets the plant states and internal clock back to initial conditions."""
        self.state = self.initial_state.copy()
        self.t = 0.0

    def get_full_states(self):
        return self.state
    
# =========================================================
# 📊 DATA GENERATION (MULTIVARIABLE INPUTS & OUTPUTS)
# =========================================================
def generate_data(plant, seq_len, num_sequences=10):
    X, Y = [], []
    STATES = []  
    
    dt = plant.dt 
    lambd = 10 
    p = 0.25 
    u_nominal = 2 # Shifted nominal value for dynamic operating window

    for _ in range(num_sequences):
        plant.reset()
        
        # Generate raw signal for BOTH light inputs (dim=2)
        raw = torch.rand((2, seq_len)) * 2 - 1

        # FFT Low-pass processing for both channels
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, d=dt)
        cutoff = 1.0 / lambd
        fft_sig[:, freqs > cutoff] = 0
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)

        # Normalize across the sequence length axis
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min) - 1
        
        u = (u_nominal + v_norm * p) 
        u = torch.clamp(u, min=U_MIN, max=U_MAX) # u shape: (2, seq_len)

        # Array tracking for multi-outputs
        y_meas = torch.zeros((seq_len + 1, 2))
        state_history = []
        
        y_meas[0] = torch.tensor(plant.output_function(plant.get_full_states(), plant.t))
        state_history.append(plant.get_full_states().copy())

        for t in range(seq_len):
            u_t = np.array([u[0, t].item(), u[1, t].item()])
            y_meas[t+1] = torch.tensor(plant.step(u_t))
            state_history.append(plant.get_full_states().copy())
            
        y_t = y_meas[:-1]
        y_next = y_meas[1:]
        
        # Feature pair includes state tracking paths for both strains
        x = np.hstack([y_t.numpy(), y_next.numpy()])   # Shape: (seq_len, 4)
        y_target = u.T.numpy()                         # Shape: (seq_len, 2)
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
    model.fit(X, Y, warmup=50)
    print("Training complete!")
    return [0.0]

# =========================================================
# 🔍 PREDICTION PLOT 
# =========================================================
def plot_prediction(model, X, Y, dt, dirname="plots"):
    for idx, (x_seq, y_seq) in enumerate(zip(X, Y)):
        model.reset()
        y_pred = model.run(x_seq)

        t_axis = np.arange(len(y_seq)) * dt
        
        # Plot for Actuator Input 1 (Light 1)
        plot_signals(
            t=t_axis, signals=[y_seq[:, 0], y_pred[:, 0]], labels=["True u1(t)", "Predicted u1(t)"],
            xlabel="Time (h)", ylabel="Light Intensity 1", title=f"U1 Prediction (seq {idx})",
            filename=f"prediction_u1_seq_{idx}", dirname=dirname
        )
        # Plot for Actuator Input 2 (Light 2)
        plot_signals(
            t=t_axis, signals=[y_seq[:, 1], y_pred[:, 1]], labels=["True u2(t)", "Predicted u2(t)"],
            xlabel="Time (h)", ylabel="Light Intensity 2", title=f"U2 Prediction (seq {idx})",
            filename=f"prediction_u2_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🤖 CLOSED-LOOP SIMULATION (STATEFUL & SATURATED)
# =========================================================
def simulate_controller_stateful(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None):
    plant.reset()
    y = plant.output_function(plant.get_full_states(), plant.t) 
    y1_log, y2_log, u1_log, u2_log = [], [], [], []
    ref1_log, ref2_log = [], []

    model.reset()

    for t in range(steps):
        # Multi-trajectory tracking references for the 2 strains
        y_ref1 = plant.initial_state[0] + 0.5 * np.sin(2 * np.pi * 0.005 * t * dt)
        y_ref2 = plant.initial_state[1] + 0.3 * np.cos(2 * np.pi * 0.005 * t * dt)
        
        input_pair = np.array([[y[0], y[1], y_ref1, y_ref2]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)
            
        u_norm_np = model(input_pair[0]) 
        
        if y_scaler:
            u_actual = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0]
        else:
            u_actual = u_norm_np[0]

        # Clamp multi-inputs
        u_actual = np.clip(u_actual, U_MIN, U_MAX)
        y = plant.step(u_actual)

        y1_log.append(y[0])
        y2_log.append(y[1])
        u1_log.append(u_actual[0])
        u2_log.append(u_actual[1])
        ref1_log.append(y_ref1)
        ref2_log.append(y_ref2)

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y1_log, ref1_log], labels=["X1 Biomass", "Ref 1"],
        xlabel="Time (h)", ylabel="Concentration (g/L)", title="Strain 1 Tracking",
        filename="tracking_strain1", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[y2_log, ref2_log], labels=["X2 Biomass", "Ref 2"],
        xlabel="Time (h)", ylabel="Concentration (g/L)", title="Strain 2 Tracking",
        filename="tracking_strain2", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[u1_log, u2_log], labels=["u1 Light", "u2 Light"],
        xlabel="Time (h)", ylabel="Light Input", title="Optogenetic Control Actions",
        filename="control_signals", dirname=dirname
    )

def simulate_constant_controller(model, plant, dt, steps=50, dirname="plots", x_scaler=None, y_scaler=None, ref_values=[1, 6]):
    plant.reset()
    y = plant.output_function(plant.get_full_states(), plant.t)
    y1_log, y2_log, u1_log, u2_log = [], [], [], []
    
    model.reset()

    for t in range(steps):
        input_pair = np.array([[y[0], y[1], ref_values[0], ref_values[1]]])  
        if x_scaler:
            input_pair = x_scaler.transform(input_pair)

        u_norm_np = model(input_pair[0])
        
        if y_scaler:
            u_actual = y_scaler.inverse_transform(u_norm_np.reshape(1, -1))[0]
        else:
            u_actual = u_norm_np[0]

        u_actual = np.clip(u_actual, U_MIN, U_MAX)
        y = plant.step(u_actual)

        y1_log.append(y[0])
        y2_log.append(y[1])
        u1_log.append(u_actual[0])
        u2_log.append(u_actual[1])

    t_axis = np.arange(steps) * dt
    plot_signals(
        t=t_axis, signals=[y1_log, [ref_values[0]]*steps], labels=["X1 Biomass", "Ref 1"],
        xlabel="Time (h)", ylabel="Biomass (g/L)", title="Strain 1 Constant Setpoint",
        filename="constant_tracking_strain1", dirname=dirname
    )
    plot_signals(
        t=t_axis, signals=[y2_log, [ref_values[1]]*steps], labels=["X2 Biomass", "Ref 2"],
        xlabel="Time (h)", ylabel="Biomass (g/L)", title="Strain 2 Constant Setpoint",
        filename="constant_tracking_strain2", dirname=dirname
    )

# =========================================================
# 📊 DATA VISUALIZATION
# =========================================================
def plot_dataset(X, Y, dt, dirname="plots"):
    for idx, (x, y) in enumerate(zip(X, Y)):
        t_axis = np.arange(len(x)) * dt
        plot_signals(
            t=t_axis, signals=[x[:, 0], x[:, 1], y[:, 0], y[:, 1]], labels=["X1(t)", "X2(t)", "u1(t)", "u2(t)"],
            xlabel="Time (h)", ylabel="Value", title=f"Dataset Profile Sequence {idx}",
            filename=f"dataset_seq_{idx}", dirname=dirname
        )

# =========================================================
# 🔥 MAIN PIPELINE
# =========================================================
def main():
    seed_everything(seed=42)
    rpy.set_seed(42) 
    dt = 0.01 
    plant = CoCultivationPlant(dt=dt)
    
    # Scale network to accommodate multivariable processing arrays
    model = build_reservoir_controller(units=500, lr=0.15, sr=0.95) 

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

    # Multi-trajectory tracking simulation
    simulate_controller_stateful(
        model, plant, dt, steps=200,
        x_scaler=x_scaler, y_scaler=y_scaler
    )

    # Multi-setpoint tracking simulation (Constant references)
    simulate_constant_controller(
        model, plant, dt, steps=200,
        x_scaler=x_scaler, y_scaler=y_scaler,
        ref_values=[1, 6]        
    )

if __name__ == "__main__":
    main()