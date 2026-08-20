import torch
import numpy as np
import random

class MackeyGlassPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Mackey-Glass System Parameters
        # Standard chaotic parameters: a=0.2, b=0.1, n=10
        self.a = torch.tensor(0.2, device=self.device)
        self.b = torch.tensor(0.1, device=self.device)
        self.n = torch.tensor(10.0, device=self.device)
        
        # Time delay parameter (tau). 
        # Typically, tau > 16.8 causes chaotic behavior.
        self.tau = 17.0 
        # Calculate how many history steps we need to store for the delay
        self.delay_steps = int(self.tau / self.dt)
        
        # History buffer to hold past states for the DDE calculation
        self.history_buffer = None
        
        # Buffer for external control signals
        self.u_buffer = None

    def get_initial_state(self, batch_size):
        """
        Returns a [batch_size, 1] tensor representing the current state x(t).
        Also initializes the history buffer required for the delay differential equation.
        """
        # Initialize a historical log of states [batch_size, delay_steps]
        # We fill it with values around a standard steady state (e.g., 1.2) with some noise
        self.history_buffer = torch.rand((batch_size, self.delay_steps), device=self.device) * 0.4 + 1.0
        
        # The current initial state is the last item in our history buffer
        return self.history_buffer[:, -1:].clone()

    def get_y(self, state, t=None):
        """
        Returns the observable output. For Mackey-Glass, the state itself 
        is typically the tracked output.
        """
        return state # Shape: [batch_size, 1]

    def dynamics(self, x, x_delayed, u):
        """
        The classic Mackey-Glass equation with an added control input 'u':
        dx/dt = (a * x_delayed) / (1 + x_delayed^n) - b * x + u
        """
        numerator = self.a * x_delayed
        denominator = 1.0 + torch.pow(x_delayed, self.n)
        dxdt = (numerator / denominator) - (self.b * x) + u
        return dxdt

    def step(self, state, u, t, dt):
        """
        RK4 Integration step updated to handle the history delay.
        """
        x = state[:, 0:1]
        
        # Extract the delayed state x(t - tau) from the front of the history buffer
        x_delayed = self.history_buffer[:, 0:1]
        
        # RK4 Integration steps
        # Note: For simple DDEs, assuming x_delayed is constant across the small dt step
        # is standard unless using complex adaptive solvers.
        k1 = self.dynamics(x, x_delayed, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, x_delayed, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, x_delayed, u)
        k4 = self.dynamics(x + dt * k3, x_delayed, u)
        
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Slide the history buffer window: drop oldest step, append the new state
        self.history_buffer = torch.cat([self.history_buffer[:, 1:], x_next], dim=1)
        
        return x_next, self.get_y(x_next)

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
        
        # Scale to p. For Mackey-Glass, we center the control input around 0.0
        # to let it perturb the natural chaotic rhythm, or a small random bias.
        u_center = torch.rand((self.batch_size, 1), device=self.device) * 0.05 - 0.025 # -0.025 to 0.025
        self.u_buffer = u_center + (v_norm * self.p)
        
        return u_center

    def get_u_at_step(self, t_idx):
        return self.u_buffer[:, t_idx].unsqueeze(1)
    
    def get_plot_config(self):
        return [
            {
                "cols": ["x1"],
                "labels": ["State x(t)"],
                "title": "Mackey-Glass State Evolution",
                "ylabel": "Value"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual x(t)", "Target x_ref(t)"],
                "title": "State Tracking Inverse Learning",
                "ylabel": "Value"
            },
            {
                "cols": ["u"],
                "labels": ["Control Forcing (u)"],
                "title": "Control Input (u)",
                "ylabel": "Forcing Amplitude"
            }
        ]

    def parse_state(self, state):
        return {
            "x": state[0].item() if torch.is_tensor(state) else state[0]
        }