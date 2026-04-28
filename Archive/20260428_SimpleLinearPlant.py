import numpy as np
from src.sample.classes.BasePlant import BasePlant
import torch


class SimpleLinearPlant(BasePlant):
    def __init__(self, seed = "none"):
        # Gain=0.8 means if u=1, y will eventually reach 0.8
        super().__init__(y_max=1.0, u_max=1.0)
        self.rng = np.random.default_rng(seed)
        self.tau = 2.0  # Time constant (seconds/hours)
        self.gain = 0.8 
        self.ref_value = 0.5 
        
        # Initialize trajectory using the local RNG
        self.reset_trajectory()

    def reset_trajectory(self):
        """Randomizes sine-wave properties using local RNG"""
        # Replace np.random with self.rng
        self.curr_amp = self.rng.uniform(0.2, 0.45)
        self.curr_freq = self.rng.uniform(0.05, 0.15)
        self.curr_phase = self.rng.uniform(0, 2 * np.pi)

    def get_initial_state(self):
        # State is simply the current value of the output y
        return np.array([0.2])

    def get_y(self, state, t=None):
        # The output is just the state itself
        return state[0]

    def step(self, state, u, t, dt=0.1):
        y = state[0]
        
        # First-order ODE: dy/dt = (Gain * u - y) / tau
        dy = (self.gain * u - y) / self.tau
        
        new_y = y + dy * dt
        return np.array([new_y]), new_y

    def reset_trajectory(self):
        """Randomizes the sine-wave properties for each training epoch"""
        self.curr_amp = np.random.uniform(0.2, 0.45)
        self.curr_freq = np.random.uniform(0.05, 0.15)
        self.curr_phase = np.random.uniform(0, 2 * np.pi)

    def generate_random_u(self, t):
        """Sine-wave approximation with rectangles"""
        # 1. Base sine wave
        sine_val = self.curr_amp * np.sin(2 * np.pi * self.curr_freq * t + self.curr_phase) + 0.5
        
        # 2. Rectangle logic: Hold the value for 2-second blocks
        # This creates the 'staircase' effect you wanted
        hold_interval = 2.0
        t_snapped = (t // hold_interval) * hold_interval
        
        # Sample the sine wave at the start of the block
        u_rect = self.curr_amp * np.sin(2 * np.pi * self.curr_freq * t_snapped + self.curr_phase) + 0.5
        
        return np.clip(u_rect, 0, self.U_MAX)

    def get_plot_config(self):
        return [
            {
                "cols": ["y", "r"], # The column names in your CSV/DataFrame
                "labels": ["Growth", "Target"], # What to show in the legend
                "title": "Growth Rate", # Header of the subplot
                "ylabel": "1/h" # Unit for the Y-axis
            },
            {
                "cols": ["u"], 
                "labels": ["Feed"], 
                "title": "Control Action", 
                "ylabel": "L/h"
            }
        ]

    def parse_state(self, state):
        return {"process_value": state[0]}
    



    import torch

class GPUSimpleLinearPlant:
    def __init__(self, batch_size, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        
        # Physical constants
        self.tau = torch.tensor(2.0, device=device) # Time constant
        self.gain = torch.tensor(0.8, device=device) # System gain
        
        # Normalization and constraints
        self.U_MAX = 1.0
        self.Y_MAX = 1.0
        
        # Buffer for Canaday signal
        self.u_buffer = None
        self.ref_value = torch.tensor(0.5, device=device)

    def get_initial_state(self):
        """Returns [batch_size, 1] tensor of [y]"""
        return torch.zeros((self.batch_size, 1), device=self.device)

    def get_y(self, state, t=None):
        """The output is just the state (y)"""
        return state[:, 0:1]

    def step(self, state, u, t, dt=0.01):
        """Vectorized first-order dynamics step"""
        y = state[:, 0:1]
        
        # dy/dt = (Gain * u - y) / tau
        dy = (self.gain * u - y) / self.tau
        
        state_next = y + dy * dt
        return state_next, y

    def reset_trajectory(self, seq_len, dt, lambd=5.0, p=0.4):
        """
        Canaday's FFT-based signal generation.
        Matches the logic used in your Fermentation plant.
        """
        # 1. Sample uniform noise [-1, 1]
        raw = torch.rand((self.batch_size, seq_len), device=self.device) * 2 - 1
        
        # 2. FFT to Frequency Domain
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, d=dt)
        
        # 3. Low-pass Filter (cutoff = 1/lambda)
        fft_sig[:, freqs > (1.0 / lambd)] = 0
        
        # 4. Inverse FFT back to Time Domain
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
        
        # 5. Rescale to [-1, 1] then center and scale by p
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1
        
        # Final control signal shifted to physical range [0.5 - p, 0.5 + p]
        self.u_buffer = torch.clamp(0.5 + (v_norm * p), 0.0, self.U_MAX)

    def get_u_at_step(self, t_idx):
        """Extracts the control value for the current time step"""
        return self.u_buffer[:, t_idx].unsqueeze(1)
    
    def get_plot_config(self):
        return [
            {"cols": ["y", "r"], "labels": ["Output (y)", "Target (r)"], "title": "System Tracking", "ylabel": "Value"},
            {"cols": ["u"], "labels": ["Control Action (u)"], "title": "Pump Input", "ylabel": "Signal"}
        ]

    def parse_state(self, state):
        return {"process_value": state[0].item() if torch.is_tensor(state) else state[0]}