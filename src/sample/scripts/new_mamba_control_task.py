import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from src.sample.utils.plotting_utils import plot_signals
plt.style.use('src/sample/style.mplstyle')

# Import your custom utility module
from src.sample.utils.saving_utils import save_plot_image

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. THE NONLINEAR PLANT SIMULATOR
# =====================================================================
class NonlinearPlant:
    """
    Simulates a non-linear liquid level tank.
    dh/dt = u(t) - c * sqrt(h(t))
    y(t) = h(t) (height of liquid)
    """
    def __init__(self, dt=0.1, c=0.2):
        self.dt = dt
        self.c = c
        self.h = 1.0  # Initial state
        
    def reset(self, initial_h=1.0):
        self.h = initial_h
        return self.h
        
    def step(self, u):
        u = np.clip(u, 0.0, 2.0)
        dh = (u - self.c * np.sqrt(max(self.h, 0.0))) * self.dt
        self.h += dh
        return self.h

# =====================================================================
# 2. GENERATING TRAINING DATA
# =====================================================================
def generate_canaday_training_data(plant, num_steps=5000, dt=0.1, lambda_cutoff=10.0, p_scale=6.0, u_min=0.1, u_max=20):
    """
    Generates training data following Canaday's frequency-domain filtering method.
    
    Steps:
    1. Uniform sampling (White noise in time-domain)
    2. Fourier Transform (Move to frequency domain)
    3. Frequency cutoff (Zero out frequencies above 1/lambda)
    4. Inverse Fourier Transform (Return to time domain as a smooth signal)
    5. Scaling to [-p, p] and shifting to physical actuator bounds [u_min, u_max]
    """
    # Step 1: Uniform sampling of white noise in time-domain
    raw_noise = np.random.uniform(-1.0, 1.0, num_steps)
    
    # Step 2: Fourier Transform
    fft_signal = np.fft.rfft(raw_noise)
    frequencies = np.fft.rfftfreq(num_steps, d=dt)
    
    # Step 3: Frequency cutoff (1 / lambda)
    cutoff_freq = 1.0 / lambda_cutoff
    fft_signal[frequencies > cutoff_freq] = 0.0
    
    # Step 4: Inverse Fourier Transform (take real part to fix tiny numerical residuals)
    smooth_signal = np.fft.irfft(fft_signal, n=num_steps)
    
    # Step 5: Scaling to range [-p, p] 
    # First normalize to [-1, 1]
    smooth_signal = smooth_signal / np.max(np.abs(smooth_signal))
    # Scale to target amplitude boundary p
    smooth_signal = smooth_signal * p_scale
    
    # Finally, shift and map explicitly into physical actuator bounds [u_min, u_max]
    # This translates the symmetric zero-centered signal into your real pump limits.
    u_data = 10 + (smooth_signal - (-p_scale)) * (u_max - u_min) / (2 * p_scale)
    #u_data = np.clip(u_data, u_min, u_max)
    
    # Run the smooth, band-limited signal through the nonlinear plant to gather states
    y_data = []
    plant.reset(initial_h=1.0)
    for u_step in u_data:
        y = plant.step(u_step)
        y_data.append(y)
        
    return u_data, np.array(y_data)

# =====================================================================
# 3. MAMBA-INSPIRED STRUCTURAL CONTROLLER
# =====================================================================
class MambaControlBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.x_proj  = nn.Linear(hidden_dim, hidden_dim) 
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            
        x_mapped = torch.relu(self.in_proj(x))
        gate = torch.sigmoid(self.x_proj(x_mapped))
        h_next = (1 - gate) * h_prev + gate * x_mapped
        u_out = torch.tanh(self.out_proj(h_next)) + 1.0 
        return u_out, h_next

# =====================================================================
# 4. PREPARING WINDOWED SEQUENCE DATASET
# =====================================================================
def create_dataset(u_data, y_data, lookback=5, lookahead=10):
    X, Y = [], []
    for i in range(lookback, len(u_data) - lookahead):
        past_y = y_data[i-lookback:i+1]          
        past_u = u_data[i-lookback:i]            
        future_r = y_data[i:i+lookahead+1]       
        
        features = np.concatenate([past_y, past_u, future_r])
        X.append(features)
        Y.append(u_data[i])                      
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)




#=====================================================================
# MAIN RUNTIME: EXECUTION LOOP WITH CUSTOM PLOT CALL
# =====================================================================
if __name__ == "__main__":
    dt = 0.1
    lookback = 1
    lookahead = 1
    
    sim_plant = NonlinearPlant(dt=dt)
    u_raw, y_raw = generate_canaday_training_data(sim_plant, num_steps=10000)
    plot_steps = 1000
    training_time_axis = np.arange(plot_steps) * dt
    training_signals_to_plot = [
        u_raw[:plot_steps], 
        y_raw[:plot_steps]
    ]
    training_labels = [
        "Band-Limited Input Force u(t)", 
        "Plant System Response y(t)"
    ]
    
    print("Generating Canaday-structured training signal diagnostic plot...")
    plot_signals(
        t=training_time_axis,
        signals=training_signals_to_plot,
        labels=training_labels,
        title="Canaday Method: Band-Limited System Training Signal",
        xlabel="Time (seconds)",
        ylabel="Amplitude",
        figsize=(7, 7),
        show=True,
        filename="canaday_training_signals.png",
        dirname="results_plots"
    )

    X, Y = create_dataset(u_raw, y_raw, lookback, lookahead)
    
    train_size = int(0.8 * len(X))
    X_train, Y_train = X[:train_size], Y[:train_size]
    
    input_dim = X.shape[1]
    model = MambaControlBlock(input_dim=input_dim, hidden_dim=32, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("Training Mamba Controller...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions, _ = model(X_train)
        loss = criterion(predictions, Y_train)
        loss.backward()
        optimizer.step()

    # =============================================================# =====================================================================
    # 4. CLOSED-LOOP EVALUATION (SINE + LINEAR TREND TRACKING TEST)
    # =====================================================================
    print("\nRunning Closed-Loop Tracking Test (Sine + Linear)...")
    model.eval()
    test_plant = NonlinearPlant(dt=dt)
    current_y = test_plant.reset(initial_h=0.4) # Start at a low initial height
    
    test_steps = 400
    extended_steps = test_steps + lookahead + 2
    time_steps_array = np.arange(extended_steps)
    
    # --- NEW: Reference Trajectory = Sinusoid + Linear Trend ---
    # Baseline linear ramp moving upwards, mixed with an oscillating sine wave
    linear_trend = 0.4 + (0.0015 * time_steps_array)
    sinusoid = 0.3 * np.sin(2 * np.pi * 0.01 * time_steps_array)
    reference_trajectory = linear_trend + sinusoid
    
    history_y = [current_y] * (lookback + 1)
    history_u = [0.0] * lookback
    
    tracked_y, applied_u = [], []
    h_eval = None 
    
    for t in range(test_steps):
        past_y_feat = np.array(history_y[-lookback-1:])
        past_u_feat = np.array(history_u[-lookback:])
        future_r_feat = reference_trajectory[t:t+lookahead+1]
        
        x_step = np.concatenate([past_y_feat, past_u_feat, future_r_feat])
        x_tensor = torch.tensor(x_step, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_u, h_eval = model(x_tensor, h_eval)
            u_action = pred_u.item()
            
        current_y = test_plant.step(u_action)
        
        tracked_y.append(current_y)
        applied_u.append(u_action)
        history_y.append(current_y)
        history_u.append(u_action)

    # =====================================================================
    # 5. OUTPUT EVALUATION RESULTS
    # =====================================================================
    time_axis = np.arange(test_steps) * dt
    signals_to_plot = [
        reference_trajectory[:test_steps], 
        np.array(tracked_y), 
        np.array(applied_u)
    ]
    signal_labels = [
        "Reference r(t) [Sine + Linear]", 
        "Mamba Output y(t)", 
        "Control Input u(t)"
    ]
    
    print("Generating final tracking performance plot...")
    plot_signals(
        t=time_axis,
        signals=signals_to_plot,
        labels=signal_labels,
        title="Mamba Architecture Tracking: Sinusoid + Linear Trend",
        xlabel="Time (seconds)",
        ylabel="Amplitude",
        figsize=(8, 8),
        show=True,
        filename="mamba_sine_linear_tracking.png",
        dirname="results_plots"
    )