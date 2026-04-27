import numpy as np
from src.sample.classes.BasePlant import BasePlant

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
        """Vectorized physics step for the entire batch"""
        x1 = state[:, 0:1] # Biomass
        
        mu = self.get_y(state, t)

        # ODE Equations
        dx1 = mu * x1
        dx2 = -(1.0/self.p1) * mu * x1 - self.ms * x1 + self.p2 * u

        # Euler Integration
        # Concatenating dx along the state dimension
        derivative = torch.cat([dx1, dx2], dim=1)
        state_next = state + derivative * dt
        
        # Physical boundary: Concentrations must be positive
        state_next = torch.clamp(state_next, min=1e-6)

        return state_next, mu

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