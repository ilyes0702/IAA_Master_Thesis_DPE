import torch

class GPUSimpleLinearPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]

        p_cfg = hyperparam_config["plant"]
        self.tau = torch.tensor(p_cfg["tau"], device=self.device)
        self.gain = torch.tensor(p_cfg["gain"], device=self.device)
        self.U_MAX = p_cfg["u_max"]
        self.Y_MAX = p_cfg["y_max"]
        
        # Buffer for Canaday signal
        self.u_buffer = None
        self.ref_value = torch.tensor(0.2, device=self.device)
        self.dt = hyperparam_config["signal"]["dt"]

    def get_initial_state(self, batch_size):
        """Returns [batch_size, 1] tensor of [y]"""
        #return torch.full((self.batch_size, 1), 0.2, device=self.device)
        # Each batch item gets a random start between -1 and 1
        return (torch.rand((batch_size, 1), device=self.device) * 2) - 1

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

    def reset_trajectory(self):
        """
        Canaday's FFT-based signal generation.
        Matches the logic used in your Fermentation plant.
        """
        # 1. Sample uniform noise [-1, 1]
        raw = torch.rand((self.batch_size, self.seq_len), device=self.device) * 2 - 1
        
        # 2. FFT to Frequency Domain
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(self.seq_len, d=self.dt)
        
        # 3. Low-pass Filter (cutoff = 1/lambda)
        fft_sig[:, freqs > (1.0 / self.lambd)] = 0
        
        # 4. Inverse FFT back to Time Domain
        v_train = torch.fft.irfft(fft_sig, n=self.seq_len, dim=1)
        
        # 5. Rescale to [-1, 1] then center and scale by p
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1
        
        # Final control signal shifted to physical range [0.5 - p, 0.5 + p]
        self.u_buffer = torch.clamp(0.5 + (v_norm * self.p), 0.0, self.U_MAX)

    def get_u_at_step(self, t_idx):
        """Extracts the control value for the current time step"""
        return self.u_buffer[:, t_idx].unsqueeze(1)
    
    def get_plot_config(self):
        return [
            # --- State evolution ---
            {
                "cols": ["x1", "x2"],
                "labels": ["Biomass (X)", "Substrate (S)"],
                "title": "Fermentation State Evolution",
                "ylabel": "Concentration"
            },

            # --- System tracking ---
            {
                "cols": ["y", "r"],
                "labels": ["Growth Rate (μ)", "Target (μ_ref)"],
                "title": "Growth Rate Tracking",
                "ylabel": "1 / h"
            },

            # --- Control input ---
            {
                "cols": ["u"],
                "labels": ["Feed Rate (u)"],
                "title": "Feed Input Signal",
                "ylabel": "Input"
            }
        ]

    def parse_state(self, state):
        return {"process_value": state[0].item() if torch.is_tensor(state) else state[0]}