import torch
import numpy as np

class GPUTrophophasePlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.batch_size = hyperparam_config["train"]["batch_size"]
        self.seq_len = hyperparam_config["signal"]["seq_len"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Parameters from Table 1 
        self.mu_max = torch.tensor(0.12, device=self.device)
        self.Ks = torch.tensor(50.0, device=self.device)
        self.mS = torch.tensor(23, device=self.device)
        self.p1 = torch.tensor(0.00047, device=self.device)
        self.p2 = torch.tensor(0.1, device=self.device)
        
        # Buffer for control signals u1
        self.u_buffer = None

        self.lambd = hyperparam_config["signal"]["lambd"]
        self.p = hyperparam_config["signal"]["p"]

    def get_volume(self, t):
        """ 
        Implements V(t) from Table 1.
        V(t) = 150 + 2(t-5)sigma(t-5) - 2(t-15)sigma(t-15)
        """
        # Ensure t is a tensor for logical operations
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)

        tau1, tau2 = 5.0, 15.0
        v0 = 150.0
        
        # Using torch.clamp or logical masking to simulate the step function sigma(t)
        term1 = 2.0 * torch.clamp(t - tau1, min=0.0)
        term2 = 2.0 * torch.clamp(t - tau2, min=0.0)
        
        return v0 + term1 - term2

    def get_initial_state(self, batch_size):
        """ Returns [batch_size, 2] for [Biomass x1, Substrate x2] """
        # Initial biomass and substrate mass 
        x1_init = torch.rand((batch_size, 1), device=self.device) * 100 + 1000
        x2_init = torch.rand((batch_size, 1), device=self.device) * 50 + 7500
        return torch.cat([x1_init, x2_init], dim=1)

    def get_y(self, state, t):
        """ Controlled variable y1 = mu(x2) [cite: 18] """
        x2 = state[:, 1:2]
        V = self.get_volume(t)
        # Monod growth kinetics: mu(x2) = (mu_max * x2) / (Ks * V + x2) [cite: 13]
        mu = (self.mu_max * x2) / (self.Ks * V + x2)
        return mu

    def dynamics(self, x1, x2, u1, t):
        """ Implementation of equations (1) and (2)  """
        V = self.get_volume(t)
        mu = (self.mu_max * x2) / (self.Ks * V + x2)
        
        # x1_dot = mu * x1 [cite: 10]
        dx1dt = mu * x1
        
        # x2_dot = -(1/p1)*mu*x1 - mS*x1 + p2*u1 
        dx2dt = -(1.0 / self.p1) * mu * x1 - self.mS * x1 + self.p2 * u1
        
        return dx1dt, dx2dt

    def step(self, state, u, t, dt):
        """ Runge-Kutta 4th Order Integration """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        
        # RK4 Steps
        k1_x1, k1_x2 = self.dynamics(x1, x2, u, t)
        
        k2_x1, k2_x2 = self.dynamics(x1 + 0.5*dt*k1_x1, x2 + 0.5*dt*k1_x2, u, t + 0.5*dt)
        
        k3_x1, k3_x2 = self.dynamics(x1 + 0.5*dt*k2_x1, x2 + 0.5*dt*k2_x2, u, t + 0.5*dt)
        
        k4_x1, k4_x2 = self.dynamics(x1 + dt*k3_x1, x2 + dt*k3_x2, u, t + dt)
        
        x1_next = x1 + (dt/6.0) * (k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1)
        x2_next = x2 + (dt/6.0) * (k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2)
        
        state_next = torch.cat([x1_next, x2_next], dim=1)
        return state_next, self.get_y(state_next, t + dt)


    def reset_trajectory(self):
        """
        Generates a band-limited random trajectory for the glucose feed u1.
        Logic based on the provided Chemostat example:
        1. Uniform sampling [-1, 1]
        2. FFT and Frequency cutoff (1/lambda)
        3. Inverse FFT
        4. Scaling to the real-world process bounds [0, 1]
        """
        # Step 1: Sample values from a uniform distribution [-1, 1]
        raw = torch.rand((self.batch_size, self.seq_len), device=self.device) * 2 - 1
        
        # Step 2: Fourier-transform to frequency domain
        fft_sig = torch.fft.rfft(raw, dim=1)
        freqs = torch.fft.rfftfreq(self.seq_len, d=self.dt)
        
        # Step 3: Drop frequencies above 1/lambda 
        # (lambda is the characteristic time provided in your config)
        cutoff = 1.0 / self.lambd
        fft_sig[:, freqs > cutoff] = 0
        
        # Step 4: Inverse-Fourier-transform
        v_train = torch.fft.irfft(fft_sig, n=self.seq_len, dim=1)
        
        # Step 5: Normalize to [0, 1] to respect process constraints
        # Trophophase u1 constraint: 0 <= u1 <= 1.0 
        v_min = v_train.min(dim=1, keepdim=True)[0]
        v_max = v_train.max(dim=1, keepdim=True)[0]
        
        # Scale the filtered signal into the valid actuator range [0, 1]
        # We use a slight safety margin or randomization for the center point
        #self.u_buffer = (v_train - v_min) / (v_max - v_min + 1e-8)

        D_center = torch.rand((self.batch_size, 1), device=self.device) * 0.6 + 0.2 # 0.2 to 0.5
        self.current_D_center = D_center
        self.u_buffer = D_center + (v_train * self.p)
        
        # Return a starting point or a parameter if needed by your trainer
        return D_center

    def get_u_at_step(self, t_idx):
        """Returns the pre-calculated glucose feed u1 for the current time step [cite: 8, 11]"""
        return self.u_buffer[:, t_idx].unsqueeze(1)
    

    def get_plot_config(self):
        return [
            {
                "cols": ["x1"],
                "labels": ["Biomass Mass (x1)"],
                "title": "Biomass Growth",
                "ylabel": "Mass [g]"
            },
            {
                "cols": ["x2"],
                "labels": ["Substrate Mass (x2)"],
                "title": "Substrate Consumption",
                "ylabel": "Mass [mg]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual Growth Rate", "Target mu*"],
                "title": "Trophophase Control Tracking",
                "ylabel": "Growth Rate [1/h]"
            }
        ]
    
    def parse_state(self, state, t):
        """
        Parses the state vector for the trophophase.
        state[0]: Biomass mass (x1) [cite: 10]
        state[1]: Substrate mass (x2) [cite: 11]
        """
        x1 = state[0].item() if torch.is_tensor(state) else state[0]
        x2 = state[1].item() if torch.is_tensor(state) else state[1]
        
        # Calculate current volume to provide concentration context 
        V_t = self.get_volume(t)
        
        return {
            "biomass_mass": x1,         # [g TS] 
            "substrate_mass": x2,       # [mg S] 
            "volume": V_t,              # [l] [cite: 43]
            "substrate_conc": x2 / V_t  # [mg S / l] - useful for Monod kinetics 
        }
    

# Problem: Kinetic saturation. If initialization of x2 (substrate) is too high or feed u1 is adding substrate much faster than the biomass can consume it, mu will stay at mu_max and the input cannot influence the growth rate, making it impossible for the controller to learn.