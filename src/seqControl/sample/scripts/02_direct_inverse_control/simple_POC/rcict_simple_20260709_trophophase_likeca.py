import numpy as np
import torch  
from sklearn.preprocessing import StandardScaler 
import math
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from seqControl.sample.utils.general_utils import seed_everything

# Assuming save_plot_image exists in your environment or utils
try:
    from seqControl.sample.utils.plotting_utils import save_plot_image, plot_signals
except ImportError:
    # Fallback to definition if imported differently
    pass

try:
    plt.style.use("src/sample/style.mplstyle")
except:
    pass

# =========================================================
# 🛑 GLOBAL CONFIGURATION & ACTUATOR BOUNDS
# =========================================================
U_MIN = 0.0    
U_MAX = 1.0    

# =========================================================
# 🧫 TROPHOPHASE FERMENTATION PLANT
# =========================================================
class TrophophasePlant:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.t = 0.0  
        self.mu_max = 0.12       
        self.K_s = 50.0          
        self.m_s = 23.0          
        self.p1 = 0.0047         
        self.p2 = 200000.0       
        self.initial_state = np.array([1500.0, 2000.0]) 
        self.state = self.initial_state.copy()

    def get_V(self, t):
        V = 150.0
        if t > 5.0:
            V += 2.0 * (t - 5.0)
        if t > 15.0:
            V -= 2.0 * (t - 15.0)
        return V

    def dynamics(self, state, u, t):
        x1_biomass = state[0]
        x2_substrate = state[1]
        u1_feed = float(u)
        V = self.get_V(t)
        mu = (self.mu_max * x2_substrate) / (self.K_s * V + x2_substrate)
        dx1_dt = mu * x1_biomass
        dx2_dt = -(1.0 / self.p1) * mu * x1_biomass - self.m_s * x1_biomass + self.p2 * u1_feed
        return np.array([dx1_dt, dx2_dt])

    def output_function(self, state, t):
        return state[0]

    def step(self, u):
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
        self.state = self.initial_state.copy()
        self.t = 0.0

    def get_full_states(self):
        return self.state

# =========================================================
# 🧠 RESERVOIR COMPUTING CONTROLLER
# =========================================================
def build_reservoir_controller(units=200, lr=0.2, sr=0.95, ridge_reg=1e-6):
    reservoir = Reservoir(units=units, lr=lr, sr=sr, rc_connectivity=0.1)
    readout = Ridge(ridge=ridge_reg)
    return reservoir >> readout

# =========================================================
# 📈 SIGNAL GENERATION UTILITY
# =========================================================
def generate_excitation_signal(length, dt, lambd=2.0):
    raw = torch.rand((1, length)) * 2 - 1
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(length, d=dt)
    cutoff = 1.0 / lambd
    fft_sig[:, freqs > cutoff] = 0
    v_train = torch.fft.irfft(fft_sig, n=length, dim=1)
    v_min, v_max = v_train.min(), v_train.max()
    v_norm = 2 * (v_train - v_min) / (v_max - v_min) - 1
    u = 0.8 + v_norm.squeeze(0) * 0.15  
    return torch.clamp(u, min=U_MIN, max=U_MAX).numpy()

# =========================================================
# 🔥 MAIN TIMELINE PIPELINE
# =========================================================
def main():
    seed_everything(seed=42)
    rpy.set_seed(42) 
    
    dt = 0.01 
    T_train = 40.0   
    T_test = 100.0   

    train_steps = int((T_train + dt) / dt)
    test_steps = int((T_test - T_train) / dt)
    
    plant = TrophophasePlant(dt=dt)
    plant.reset()

    # -----------------------------------------------------
    # PHASE 1: TRAINING DATA COLLECTION (Forced Input u)
    # -----------------------------------------------------
    print(f"--- Step 1: Imposing Excitation Input Signal (t = 0 to {T_train + dt}h) ---")
    u_train_profile = generate_excitation_signal(train_steps, dt)
    
    y_history = []
    u_history = []
    t_history = []
    
    y_history.append(plant.output_function(plant.get_full_states(), plant.t))
    t_history.append(plant.t)

    for step in range(train_steps):
        u_t = u_train_profile[step]
        y_next = plant.step(u_t)
        
        u_history.append(u_t)
        y_history.append(y_next)
        t_history.append(plant.t)

    y_history = np.array(y_history)
    X_train = np.stack([y_history[:-1], y_history[1:]], axis=-1)  
    Y_train = np.array(u_history).reshape(-1, 1)                  

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_norm = x_scaler.transform(X_train)
    Y_train_norm = y_scaler.transform(Y_train)

    model = build_reservoir_controller(units=400, lr=0.2, sr=0.95)
    print("Training Reservoir Model...")
    model.fit(X_train_norm, Y_train_norm, warmup=10)
    print("Training Complete.")

    # -----------------------------------------------------
    # PHASE 2: CLOSED-LOOP TESTING (Replacing y_next with r_next)
    # -----------------------------------------------------
    print(f"--- Step 2: Transitioning to Closed-Loop RC Tracking (t > {T_train + dt}h to {T_test}h) ---")
    
    model.reset()
    _ = model.run(X_train_norm)

    current_y = y_history[-1]
    
    global_t = list(t_history[:-1])
    global_u = list(u_history)
    global_y = list(y_history[:-1])
    global_ref = [0.0] * len(global_t)  # Use 0 baseline or fallback during training profile

    for step in range(test_steps):
        t_curr = plant.t
        global_t.append(t_curr)
        
        # Define Reference Signal (r_next at t + dt)
        r_next = plant.initial_state[0] * math.exp(0.012 * (t_curr + dt))
        global_ref.append(r_next)

        input_pair = np.array([[current_y, r_next]])
        input_pair_norm = x_scaler.transform(input_pair)

        u_norm = model(input_pair_norm[0])
        u_pred = y_scaler.inverse_transform(u_norm.reshape(1, -1))[0, 0]
        u_applied = np.clip(u_pred, U_MIN, U_MAX)

        current_y = plant.step(u_applied)

        global_u.append(u_applied)
        global_y.append(current_y)

    global_t = np.array(global_t)
    global_u = np.array(global_u)
    global_y = np.array(global_y)
    global_ref = np.array(global_ref)

    # -----------------------------------------------------
    # 📊 PLOTTING RESULTS USING USER'S PLOT_SIGNALS
    # -----------------------------------------------------
    print("--- Step 3: Generating Visualizations via custom plot_signals ---")

    # Plot 1: Control Input Over Time
    # Note: Since your function sets strict aspect ratios, a square layout (6, 6) is clean
    plot_signals(
        t=global_t,
        signals=[global_u],
        labels=["Glucose Feed Rate u(t)"],
        title="Control Input Profile Across Unified Timeline",
        xlabel="Time (h)",
        ylabel="Glucose Feed Rate (l/h)",
        figsize=(6, 6),
        filename="unified_control_input",
        dirname="plots"
    )

    # Plot 2: Output Variable & Reference Tracking Over Time
    plot_signals(
        t=global_t,
        signals=[global_y, global_ref],
        labels=["Plant Biomass Output y(t)", "Target Reference r(t)"],
        title="Output Variable vs Reference Across Unified Timeline",
        xlabel="Time (h)",
        ylabel="Biomass (g)",
        figsize=(6, 6),
        filename="unified_tracking_output",
        dirname="plots"
    )

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    main()