import torch
import numpy as np
import random

class SecondOrderLinearPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]
        self.dt = hyperparam_config["signal"]["dt"]

        # System parameters (Extracting omega from config, defaulting to 1.0 if not present)
        # Note: Added an absolute value to ensure omega handles squaring/damping correctly
        self.omega = torch.tensor(hyperparam_config["signal"].get("omega", 1.0), device=self.device)
        
        # Buffer for synthesized control signals
        self.u_buffer = None
      
    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor of states [x1, x2].
        Initializes with randomized initial conditions for robust learning.
        """
        x1_init = torch.rand((batch_size, 1), device=self.device) * 2.0 - 1.0  # -1.0 to 1.0
        x2_init = torch.rand((batch_size, 1), device=self.device) * 2.0 - 1.0  # -1.0 to 1.0
        return torch.cat([x1_init, x2_init], dim=1)

    def get_y(self, state, t=None):
        """
        The system output observable is explicitly defined as y = x1.
        """
        x1 = state[:, 0:1]
        return x1 

    def dynamics(self, x1, x2, u):
        """
        Implements your specified system equations:
        dx1/dt = x2
        dx2/dt = -23 * omega * x2 + (omega^2) * x1 + u
        """
        dx1dt = x2
        dx2dt = -23.0 * self.omega * x2 + (self.omega ** 2) * x1 + u
        return dx1dt, dx2dt

    def step(self, state, u, t, dt):
        """
        RK4 Integration engine to step the second-order system forward in time.
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        
        # k1
        dx1_1, dx2_1 = self.dynamics(x1, x2, u)
        # k2
        dx1_2, dx2_2 = self.dynamics(x1 + 0.5 * dt * dx1_1, x2 + 0.5 * dt * dx2_1, u)
        # k3
        dx1_3, dx2_3 = self.dynamics(x1 + 0.5 * dt * dx1_2, x2 + 0.5 * dt * dx2_2, u)
        # k4
        dx1_4, dx2_4 = self.dynamics(x1 + dt * dx1_3, x2 + dt * dx2_3, u)
        
        x1_next = x1 + (dt / 6.0) * (dx1_1 + 2 * dx1_2 + 2 * dx1_3 + dx1_4)
        x2_next = x2 + (dt / 6.0) * (dx2_1 + 2 * dx2_2 + 2 * dx2_3 + dx2_4)
        
        state_next = torch.cat([x1_next, x2_next], dim=1)
        return state_next, self.get_y(state_next)

    def reset_trajectory(self):
        """
        Implements the Canaday/RC Fourier band-limited signal synthesis logic.
        Generates a smooth random-walk excitation profile for the input u.
        """
        # Step 1: Sample values from a uniform distribution [-1, 1]
        raw = torch.rand((self.batch_size, self.seq_len), device=self.device) * 2 - 1
        
        # Step 2: Fourier-transform to frequency domain
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(self.seq_len, d=self.dt)
        
        # Step 3: Drop frequencies above 1/lambda
        cutoff = 1.0 / self.lambd
        fft_sig[:, freqs > cutoff] = 0
        
        # Step 4: Inverse-Fourier-transform
        v_train = torch.fft.irfft(fft_sig, n=self.seq_len, dim=1)
        
        # Step 5: Normalize and Scale to [-p, p]
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1
        
        # Center point behavior for a linear mechanical system 
        # Since it is symmetric around 0, we center u_buffer around 0.0 or a small offset
        u_center = torch.rand((self.batch_size, 1), device=self.device) * 0.2 - 0.1 # -0.1 to 0.1
        self.current_u_center = u_center
        self.u_buffer = u_center + (v_norm * self.p)
        
        return u_center

    def get_u_at_step(self, t_idx):
        return self.u_buffer[:, t_idx].unsqueeze(1)
    
    def get_plot_config(self):
        """
        Updated plotting definitions to match your new state-space configuration.
        """
        return [
            {
                "cols": ["x1", "x2"],
                "labels": ["Position (x1)", "Velocity (x2)"],
                "title": "System State Space Evolution",
                "ylabel": "States Magnitude"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual System Output (y)", "Target Reference (r)"],
                "title": "Output Tracking Performance",
                "ylabel": "Position (x1)"
            },
            {
                "cols": ["u"],
                "labels": ["Control Force (u)"],
                "title": "Synthesized Actuator Input",
                "ylabel": "Force/Voltage Magnitude"
            }
        ]

    def parse_state(self, state):
        return {
            "x1": state[0].item() if torch.is_tensor(state) else state[0],
            "x2": state[1].item() if torch.is_tensor(state) else state[1]
        }