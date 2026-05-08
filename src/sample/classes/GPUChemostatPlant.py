import torch
import numpy as np

class GPUChemostatPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Biological Parameters from Config
        p_cfg = hyperparam_config["plant"]
        self.mu_max = torch.tensor(p_cfg.get("mu_max", 0.5), device=self.device)
        self.Ks = torch.tensor(p_cfg.get("Ks", 0.2), device=self.device)
        self.Y = torch.tensor(p_cfg.get("Y", 0.6), device=self.device)
        self.sR = torch.tensor(p_cfg.get("sR", 1.0), device=self.device)
        
        self.U_MAX = p_cfg.get("u_max", 0.6) # Max Dilution Rate
        # Buffer for control signals
        self.u_buffer = None

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor of [Biomass (x), Substrate (s)].
        Initializes with random biological values to ensure robust learning.
        """
        x_init = torch.rand((batch_size, 1), device=self.device) * 0.5 + 0.1 # 0.1 to 0.6
        s_init = torch.rand((batch_size, 1), device=self.device) * 0.5       # 0.0 to 0.5
        return torch.cat([x_init, s_init], dim=1)

    def get_y(self, state, t=None):
        """
        The inverse learner usually wants to track Growth Rate (mu) 
        or Biomass (x). Here we return Growth Rate as the 'observable' y.
        """
        x = state[:, 0:1]
        s = state[:, 1:2]
        mu = (self.mu_max * s) / (self.Ks + s)
        return mu # Growth rate is our output to track

    def dynamics(self, x, s, u):
        mu = (self.mu_max * s) / (self.Ks + s)
        dxdt = mu * x - u * x
        dsdt = u * (self.sR - s) - (mu * x / self.Y)
        return dxdt, dsdt

    def step(self, state, u, t, dt):
        x, s = state[:, 0:1], state[:, 1:2]
        
        # k1
        dx1, ds1 = self.dynamics(x, s, u)
        # k2
        dx2, ds2 = self.dynamics(x + 0.5*dt*dx1, s + 0.5*dt*ds1, u)
        # k3
        dx3, ds3 = self.dynamics(x + 0.5*dt*dx2, s + 0.5*dt*ds2, u)
        # k4
        dx4, ds4 = self.dynamics(x + dt*dx3, s + dt*ds3, u)
        
        x_next = x + (dt/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        s_next = s + (dt/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
        
        state_next = torch.cat([x_next, s_next], dim=1)
        return state_next, self.get_y(state_next)

    def reset_trajectory(self):
        """
        Implements the Canaday/RC logic:
        1. Uniform sampling
        2. Fourier Transform
        3. Frequency cutoff (1/lambda)
        4. Inverse Fourier Transform
        5. Scaling to range [-p, p] and shifting to physical bounds.
        """
        # Step 1: Sample values from a uniform distribution [-1, 1]
        raw = torch.rand((self.batch_size, self.seq_len), device=self.device) * 2 - 1
        
        # Step 2: Fourier-transform to frequency domain
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(self.seq_len, d=self.dt)
        
        # Step 3: Drop frequencies above 1/lambda
        # Note: lambda must be > delta (the plant's characteristic time)
        cutoff = 1.0 / self.lambd
        fft_sig[:, freqs > cutoff] = 0
        
        # Step 4: Inverse-Fourier-transform
        v_train = torch.fft.irfft(fft_sig, n=self.seq_len, dim=1)
        
        # Step 5: Normalize and Scale to [-p, p]
        # First, ensure the signal is centered and normalized to [-1, 1]
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1
        
        # Apply scaling p and shift to a safe operating point for the Chemostat
        # D_center should be roughly 0.5 * mu_max to keep the plant 'alive'
        D_center = 0.25
        self.u_buffer = D_center + (v_norm * self.p)
        
        # Clamp to ensure we don't hit negative dilution or extreme washout
        self.u_buffer = torch.clamp(self.u_buffer, 0.01, self.U_MAX)

    def get_u_at_step(self, t_idx):
        return self.u_buffer[:, t_idx].unsqueeze(1)
    
    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2"],
                "labels": ["Biomass (X)", "Substrate (S)"],
                "title": "Chemostat State Evolution",
                "ylabel": "Concentration [g/L]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual μ", "Target μ_ref"],
                "title": "Growth Rate Inverse Learning",
                "ylabel": "Growth Rate [1/h]"
            },
            {
                "cols": ["u"],
                "labels": ["Dilution Rate (D)"],
                "title": "Control Input (D)",
                "ylabel": "Dilution Rate [1/h]"
            }
        ]

    def parse_state(self, state):
        # state is [x, s]
        return {
            "biomass": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate": state[1].item() if torch.is_tensor(state) else state[1]
        }