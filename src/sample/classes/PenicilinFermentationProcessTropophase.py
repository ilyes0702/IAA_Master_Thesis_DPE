import numpy as np
from src.sample.classes.BasePlant import BasePlant
import torch.fft

import torch

class GPUFermentationProcessFFT:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]

        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]
        self.dt = hyperparam_config["signal"]["dt"]
        
        # Physical constants (fixed, no noise)
        self.mu_max = torch.tensor(0.12, device=self.device)
        self.Ks = torch.tensor(0.05, device=self.device)
        self.p1 = torch.tensor(0.47, device=self.device)
        self.p2 = torch.tensor(200.0, device=self.device)
        self.ms = torch.tensor(0.023, device=self.device)
        self.ref_value = torch.tensor(0.015, device=self.device)  # Target growth rate for simulation tests
        
        # Normalization and constraints
        self.U_MAX = hyperparam_config["plant"]["u_max"]
        self.Y_MAX = hyperparam_config["plant"]["y_max"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        # Control buffer
        self.u_buffer = None

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor:
        [Biomass, Substrate]
        """
        base_state = torch.tensor(
            [1.0, 5e-3],
            device=self.device,
            dtype=torch.float32
        )
        return base_state.unsqueeze(0).repeat(batch_size, 1)

    # ------------------------------------------------------------------
    # Plant outputs
    # ------------------------------------------------------------------

    def get_V(self, t):
        """Vectorized reactor volume"""
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)

        return torch.where(
            t < 5.0,
            torch.tensor(150.0, device=self.device),
            torch.where(
                t < 15.0,
                150.0 + 2.0 * (t - 5.0),
                torch.tensor(170.0, device=self.device),
            )
        )

    def get_y(self, state, t):
        """
        Growth rate μ (this is the plant output)
        """
        x2 = state[:, 1:2]
        V = self.get_V(t)
        mu = (self.mu_max * x2) / (self.Ks * V + x2)
        return mu

    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------

    def step(self, state, u, t, dt=None):
        """
        Vectorized plant dynamics
        """
        if dt is None:
            dt = self.dt

        x1 = state[:, 0:1]  # Biomass

        mu = self.get_y(state, t)

        dx1 = mu * x1
        dx2 = -(1.0 / self.p1) * mu * x1 - self.ms * x1 + self.p2 * u

        derivative = torch.cat([dx1, dx2], dim=1)
        state_next = state + derivative * dt

        # Physical constraint
        state_next = torch.clamp(state_next, min=1e-6)

        return state_next, mu

    # ------------------------------------------------------------------
    # Control trajectory generation (FFT)
    # ------------------------------------------------------------------

    def reset_trajectory(self):
        """
        Generates smooth control trajectories using FFT.
        """
        seq_len = self.seq_len
        dt = self.dt
        lambd = self.lambd
        p = self.p

        # 1. Sample uniform noise
        raw = torch.rand((self.batch_size, seq_len), device=self.device) * 2 - 1

        # 2. FFT
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, d=dt)

        # 3. Low-pass filter
        fft_sig[:, freqs > (1.0 / lambd)] = 0

        # 4. Inverse FFT
        v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)

        # 5. Normalize to control range
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1

        self.u_buffer = torch.clamp(
            0.5 + (v_norm * p),
            0.0,
            self.U_MAX
        )

    def get_u_at_step(self, t_idx):
        """
        Returns [batch_size, 1] control input
        """
        return self.u_buffer[:, t_idx].unsqueeze(1)

    

class FermentationProcess(BasePlant):
    def __init__(self, seed="none"):
        # MU_MAX is the physical limit; Y_MAX is the normalization scale
        self.mu_max = 0.12 
        super().__init__(y_max=0.12, u_max=1.0)
        self.rng = np.random.default_rng(seed)
        self.Ks = 0.05    
        self.p1 = 0.47    
        self.p2 = 200.0   
        self.ms = 0.023   
        self.ref_value = 0.08  # Target growth rate for simulation tests

        # Training signal parameters (randomized via reset_trajectory)
        self.reset_trajectory()

    def get_initial_state(self):
        # Starting concentrations: [Biomass, Substrate]
        return np.array([1.0, 5e-3])

    def get_y(self, state, t):
        """Calculates growth rate (mu) accurately using current Volume (V)"""
        x1, x2 = state
        V = self.get_V(t)
        # Monod kinetics equation
        return (self.mu_max * x2) / (self.Ks * V + x2)

    def step(self, state, u1, t, dt):
        x1, x2 = state
        # Source of truth: use the get_y method
        mu = self.get_y(state, t)

        # Differential equations
        dx1 = mu * x1
        dx2 = -(1/self.p1) * mu * x1 - self.ms * x1 + self.p2 * u1

        state_next = state + np.array([float(dx1), float(dx2)]) * dt
        # Physical constraint: concentrations cannot be negative
        state_next = np.maximum(state_next, 1e-6)

        return state_next, mu

    def get_V(self, t):
        """Volume dynamics: simulating a fed-batch or changing reactor level"""
        if t < 5: return 150.0
        if 5 <= t < 15: return 150.0 + 2 * (t - 5)
        return 170.0

    def reset_trajectory(self):
        """Called by train_controller at start of each epoch to diversify data"""
        self.curr_amp = self.rng.uniform(0.2, 0.45)
        self.curr_freq = self.rng.uniform(0.02, 0.1) # Fermentation is slower than linear
        self.curr_phase = self.rng.uniform(0, 2 * np.pi)

    def generate_random_u(self, t):
        """Staircase/Rectangular sine-wave approximation for robust training"""
        # 1. Sample the target sine wave at fixed intervals (e.g., every 5s)
        hold_interval = 5.0 
        t_snapped = (t // hold_interval) * hold_interval
        
        # 2. Calculate the 'held' control value
        u_val = self.curr_amp * np.sin(2 * np.pi * self.curr_freq * t_snapped + self.curr_phase) + 0.5
        
        return np.clip(u_val, 0, self.U_MAX)

    def parse_state(self, state):
        return {
            "biomass_x1": state[0],
            "substrate_x2": state[1]
        }

    def get_plot_config(self):
        return [
            {"cols": ["y", "r"], "labels": ["Growth rate", "Target"], "title": "Growth Rate", "ylabel": "1/h"},
            {"cols": ["u"], "labels": ["Glucose Feed"], "title": "Control Action", "ylabel": "L/h"},
            {"cols": ["biomass_x1"], "labels": ["X1"], "title": "Biomass", "ylabel": "g/L"},
            {"cols": ["substrate_x2"], "labels": ["X2"], "title": "Substrate", "ylabel": "g/L"}
        ]
    


import torch

import torch

import torch

class GPUFermentationProcess:
    def __init__(self, batch_size=256, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        
        # Physical constants (Fixed/No noise)
        self.mu_max = torch.tensor(0.12, device=device)
        self.Ks = torch.tensor(0.05, device=device)
        self.p1 = torch.tensor(0.47, device=device)
        self.p2 = torch.tensor(200.0, device=device)
        self.ms = torch.tensor(0.023, device=device)
        
        # Normalization and constraints
        self.U_MAX = 1.0
        self.Y_MAX = 0.12

        # Trajectory parameters for data diversity during training
        self.curr_amp = torch.tensor(0.3, device=device)
        self.curr_freq = torch.tensor(0.05, device=device)
        self.curr_phase = torch.tensor(0.0, device=device)
        self.ref_value = torch.tensor(0.015, device=device)

    def get_initial_state(self):
        """Returns a [batch_size, 2] tensor of [Biomass, Substrate]"""
        base_state = torch.tensor([1.0, 5e-3], device=self.device, dtype=torch.float32)
        return base_state.repeat(self.batch_size, 1)

    def get_V(self, t):
        """Vectorized Volume dynamics: handles scalars or tensors of t"""
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
            
        # Efficient GPU switching logic
        v = torch.where(t < 5.0, 
                        torch.tensor(150.0, device=self.device), 
                        torch.where(t < 15.0, 150.0 + 2.0 * (t - 5.0), torch.tensor(170.0, device=self.device)))
        return v

    def get_y(self, state, t):
        """Calculates growth rate (mu) based on Monod kinetics"""
        x2 = state[:, 1:2] 
        V = self.get_V(t)
        mu = (self.mu_max * x2) / (self.Ks * V + x2)
        return mu

    def step(self, state, u, t, dt=0.01):
        """Vectorized physics step for the entire batch using RK4"""
        # RK4 Integration
        # k1
        mu_k1 = self.get_y(state, t)
        x1_k1 = state[:, 0:1]
        dx1_k1 = mu_k1 * x1_k1
        dx2_k1 = -(1.0/self.p1) * mu_k1 * x1_k1 - self.ms * x1_k1 + self.p2 * u
        k1 = torch.cat([dx1_k1, dx2_k1], dim=1)
        
        # k2
        state_k2 = state + k1 * dt / 2
        mu_k2 = self.get_y(state_k2, t + dt / 2)
        x1_k2 = state_k2[:, 0:1]
        dx1_k2 = mu_k2 * x1_k2
        dx2_k2 = -(1.0/self.p1) * mu_k2 * x1_k2 - self.ms * x1_k2 + self.p2 * u
        k2 = torch.cat([dx1_k2, dx2_k2], dim=1)
        
        # k3
        state_k3 = state + k2 * dt / 2
        mu_k3 = self.get_y(state_k3, t + dt / 2)
        x1_k3 = state_k3[:, 0:1]
        dx1_k3 = mu_k3 * x1_k3
        dx2_k3 = -(1.0/self.p1) * mu_k3 * x1_k3 - self.ms * x1_k3 + self.p2 * u
        k3 = torch.cat([dx1_k3, dx2_k3], dim=1)
        
        # k4
        state_k4 = state + k3 * dt
        mu_k4 = self.get_y(state_k4, t + dt)
        x1_k4 = state_k4[:, 0:1]
        dx1_k4 = mu_k4 * x1_k4
        dx2_k4 = -(1.0/self.p1) * mu_k4 * x1_k4 - self.ms * x1_k4 + self.p2 * u
        k4 = torch.cat([dx1_k4, dx2_k4], dim=1)
        
        # Combine steps
        state_next = state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
        # Physical boundary: Concentrations must be positive
        state_next = torch.clamp(state_next, min=1e-6)

        return state_next, mu_k1

    def reset_trajectory(self):
        """Randomizes signal parameters for the next training epoch"""
        self.curr_amp = torch.rand(1, device=self.device) * 0.25 + 0.2
        self.curr_freq = torch.rand(1, device=self.device) * 0.08 + 0.02
        self.curr_phase = torch.rand(1, device=self.device) * 2 * torch.pi

    def generate_random_u(self, t):
        """Generates a batch-wide control signal based on current trajectory parameters"""
        # Periodic 'held' signal to simulate realistic pump behavior
        hold_interval = 5.0 
        t_snapped = (t // hold_interval) * hold_interval
        
        u_val = self.curr_amp * torch.sin(2 * torch.pi * self.curr_freq * t_snapped + self.curr_phase) + 0.5
        u_clamped = torch.clamp(u_val, 0, self.U_MAX)
        
        # Expand to [batch_size, 1]
        return u_clamped.repeat(self.batch_size, 1)
    def get_plot_config(self):
        """Specifies which columns from the simulation DataFrame to plot"""
        return [
            {
                "cols": ["y", "r"], 
                "labels": ["Growth Rate (μ)", "Target"], 
                "title": "Fermentation Growth Rate", 
                "ylabel": "Growth Rate (1/h)"
            },
            {
                "cols": ["u"], 
                "labels": ["Glucose Feed"], 
                "title": "Control Action", 
                "ylabel": "Glucose Feed (L/h)"
            },
            {
                "cols": ["biomass", "substrate"], 
                "labels": ["Biomass (x1)", "Substrate (x2)"], 
                "title": "Reactor Concentrations", 
                "ylabel": "Concentration (g/L)"
            }
        ]

    def parse_state(self, state):
        """Converts the state tensor/array into named components for the CSV"""
        return {
            "biomass": state[0],
            "substrate": state[1]
        }