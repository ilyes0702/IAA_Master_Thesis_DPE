import torch

class GPUSimpleLinearPlant:
    def __init__(self, batch_size, hyperparam_config, device="cuda"):
        self.device = device
        self.batch_size = batch_size

        p_cfg = hyperparam_config["plant"]
        self.tau = torch.tensor(p_cfg["tau"], device=device)
        self.gain = torch.tensor(p_cfg["gain"], device=device)
        self.U_MAX = p_cfg["u_max"]
        self.Y_MAX = p_cfg["y_max"]
        
        # Buffer for Canaday signal
        self.u_buffer = None
        self.ref_value = torch.tensor(0.5, device=device)

    def get_initial_state(self):
        """Returns [batch_size, 1] tensor of [y]"""
        #return torch.full((self.batch_size, 1), 0.2, device=self.device)
        # Each batch item gets a random start between -1 and 1
        return (torch.rand((self.batch_size, 1), device=self.device) * 2) - 1

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