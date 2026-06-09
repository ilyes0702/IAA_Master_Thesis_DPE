import torch
import numpy as np
import random

class LorenzPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Classic chaotic Lorenz parameters (Prandtl number, Rayleigh number, geometric dimension)
        self.sigma = torch.tensor(10.0, device=self.device)
        self.rho = torch.tensor(28.0, device=self.device)
        self.beta = torch.tensor(8.0 / 3.0, device=self.device)
        
        # Buffer for external control signals
        self.u_buffer = None

    def get_initial_state(self, batch_size):
        """
        Returns a [batch_size, 3] tensor representing the coordinates [x, y, z].
        Initializes near the active attractor region with a small amount of random noise.
        """
        # Lorenz states scale much larger than the Chemostat (typically between -20 and 50)
        x_init = torch.randn((batch_size, 1), device=self.device) * 5.0
        y_init = torch.randn((batch_size, 1), device=self.device) * 5.0
        z_init = torch.rand((batch_size, 1), device=self.device) * 10.0 + 15.0 # Centered higher around 20
        
        return torch.cat([x_init, y_init, z_init], dim=1)

    def get_y(self, state, t=None):
        """
        The tracking coordinate for the inverse learner.
        Typically, we track the 'x' position of the attractor.
        """
        return state[:, 0:1] # Tracking the x coordinate

    def dynamics(self, x, y, z, u):
        """
        The continuous 3D Lorenz ODEs with an added external control input 'u' 
        acting as a force on the x-axis acceleration.
        """
        dxdt = self.sigma * (y - x) + u  # Control force applied to the x dimension
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return dxdt, dydt, dzdt

    def step(self, state, u, t, dt):
        """
        Standard RK4 integration step across the 3 dimensions.
        """
        x, y, z = state[:, 0:1], state[:, 1:2], state[:, 2:3]
        
        # k1
        dx1, dy1, dz1 = self.dynamics(x, y, z, u)
        # k2
        dx2, dy2, dz2 = self.dynamics(x + 0.5*dt*dx1, y + 0.5*dt*dy1, z + 0.5*dt*dz1, u)
        # k3
        dx3, dy3, dz3 = self.dynamics(x + 0.5*dt*dx2, y + 0.5*dt*dy2, z + 0.5*dt*dz2, u)
        # k4
        dx4, dy4, dz4 = self.dynamics(x + dt*dx3, y + dt*dy3, z + dt*dz3, u)
        
        x_next = x + (dt/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y_next = y + (dt/6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        z_next = z + (dt/6.0) * (dz1 + 2*dz2 + 2*dz3 + dz4)
        
        state_next = torch.cat([x_next, y_next, z_next], dim=1)
        return state_next, self.get_y(state_next)

    def reset_trajectory(self):
        """
        Implements your exact Canaday/RC logic to generate band-limited
        smooth random control signals u(t).
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
        
        # Scale to p. For Lorenz, the signal is centered around 0.0 because 
        # it acts as a balanced vector perturbation pulling the wings left or right.
        u_center = torch.zeros((self.batch_size, 1), device=self.device)
        self.u_buffer = u_center + (v_norm * self.p)
        
        return u_center

    def get_u_at_step(self, t_idx):
        return self.u_buffer[:, t_idx].unsqueeze(1)
    
    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2", "x3"],
                "labels": ["x (Convection)", "y (Temp Diff)", "z (Distortion)"],
                "title": "Lorenz Attractor 3D State Evolution",
                "ylabel": "State Values"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual x(t)", "Target x_ref(t)"],
                "title": "Position Tracking Inverse Learning",
                "ylabel": "X Coordinate Value"
            },
            {
                "cols": ["u"],
                "labels": ["Perturbation Force (u)"],
                "title": "Control Input (u)",
                "ylabel": "Forcing Magnitude"
            }
        ]

    def parse_state(self, state):
        return {
            "x": state[0].item() if torch.is_tensor(state) else state[0],
            "y": state[1].item() if torch.is_tensor(state) else state[1],
            "z": state[2].item() if torch.is_tensor(state) else state[2]
        }