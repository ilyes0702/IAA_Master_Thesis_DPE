import torch
import torchode

class TrophophasePlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["training_data_cfg"]["dt"]
        self.plant_cfg = hyperparam_config["plant"]

        # Biological and physical parameters from Table 1
        self.mu_max = self.plant_cfg["mu_max"]      # [1/h]
        self.Ks = self.plant_cfg["Ks"]         # [mg S/l]
        self.m_S = self.plant_cfg["m_S"]        # [mg S/(g TS h)]
        self.p1 = self.plant_cfg["p1"]      # [g TS/(mg S)]
        self.p2 = self.plant_cfg["p2"]     # [mg S/l]

        self.hyperparam_config = hyperparam_config

    def get_volume(self, t):
        """
        Calculates the time-dependent reactor volume V(t) based on Table 1:
        V(t) = 150 + 2*(t - 5)*sigma(t - 5) - 2*(t - 15)*sigma(t - 15)
        """
        # Ensure t is a tensor for element-wise operations
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
            
        ramp1 = torch.clamp(t - 5.0, min=0.0)
        ramp2 = torch.clamp(t - 15.0, min=0.0)
        
        return 150.0 + 2.0 * ramp1 - 2.0 * ramp2

    def get_initial_state(self, batch_size, randomize=True):
        """
        Returns [batch_size, 2] tensor of [Biomass Mass (x1), Substrate Mass (x2)].
        If randomize is True, initial values are randomized within ±5% of nominal values.
        If randomize is False, nominal configuration values are used uniformly.
        """
        # Fetch nominal values from config
        x1_nominal = self.hyperparam_config["plant"]["x10"]
        x2_nominal = self.hyperparam_config["plant"]["x20"]
        
        if randomize:
            # Randomization formula: nominal * (0.95 + 0.1 * rand) -> [0.95*nominal, 1.05*nominal)
            x1_values = x1_nominal * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
            x2_values = x2_nominal * (0.95 + 0.1 * torch.rand((batch_size, 1), device=self.device))
        else:
            # Create tensors filled entirely with the nominal values
            x1_values = torch.full((batch_size, 1), x1_nominal, device=self.device, dtype=torch.float32)
            x2_values = torch.full((batch_size, 1), x2_nominal, device=self.device, dtype=torch.float32)
            
        return torch.cat([x1_values, x2_values], dim=1)

    def get_y(self, state, t):
        """
        Calculates and returns the Growth Rate y1 = mu(x2) as the observable output tracker.
        Note: x2 is mass, so substrate concentration is c2 = x2 / V(t).
        """
        x2 = state[:, 1:2]
        V = self.get_volume(t)
        
        # Substrate concentration c2
        c2 = x2 / V
        
        # Monod growth kinetics: mu(x2) = (mu_max * x2) / (Ks * V + x2)
        # Separated into concentration components: (mu_max * c2) / (Ks + c2)
        mu = (self.mu_max * c2) / (self.Ks + c2)
        return mu
    
    def get_v_dot(self, t):
        """Calculates dV/dt using Heaviside step functions."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
        
        # Heaviside step function: 1.0 if t >= threshold else 0.0
        sigma1 = (t >= 5.0).to(torch.float32)
        sigma2 = (t >= 15.0).to(torch.float32)
        
        return 2.0 * sigma1 - 2.0 * sigma2

    def get_y_dot(self, state, u1, t):
        """
        Calculates the first derivative of the growth rate (dy/dt).
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        V = self.get_volume(t)
        V_dot = self.get_v_dot(t)
        
        c2 = x2 / V
        y = (self.mu_max * c2) / (self.Ks + c2)
        
        # System dynamics: dx2/dt
        _, dx2dt = self.dynamics(x1, x2, u1, t)
        
        # dc2/dt
        c2_dot = (dx2dt - c2 * V_dot) / V
        
        # dy/dc2
        dy_dc2 = (self.mu_max * self.Ks) / ((self.Ks + c2) ** 2)
        
        y_dot = dy_dc2 * c2_dot
        return y_dot

    def get_y_ddot(self, state, u1, t, u1_dot=0.0):
        """
        Calculates the second derivative of the growth rate (d^2y/dt^2).
        Assumes u1_dot (du1/dt) is 0.0 unless provided.
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        V = self.get_volume(t)
        V_dot = self.get_v_dot(t)
        # V_ddot is 0 almost everywhere
        V_ddot = 0.0 
        
        c2 = x2 / V
        y = (self.mu_max * c2) / (self.Ks + c2)
        
        # First derivatives
        dx1dt, dx2dt = self.dynamics(x1, x2, u1, t)
        c2_dot = (dx2dt - c2 * V_dot) / V
        
        dy_dc2 = (self.mu_max * self.Ks) / ((self.Ks + c2) ** 2)
        y_dot = dy_dc2 * c2_dot
        
        # Secondary derivatives for state equations
        # d^2x2/dt^2 = -(1/p1)*(y_dot*x1 + y*dx1dt) - m_S*dx1dt + p2*u1_dot
        dx2dt_dot = -(1.0 / self.p1) * (y_dot * x1 + y * dx1dt) - self.m_S * dx1dt + self.p2 * u1_dot
        
        # d^2c2/dt^2
        c2_ddot = (dx2dt_dot - 2.0 * c2_dot * V_dot - c2 * V_ddot) / V
        
        # d^2y/dc2^2
        d2y_dc22 = -2.0 * (self.mu_max * self.Ks) / ((self.Ks + c2) ** 3)
        
        # d^2y/dt^2 = (dy/dc2)*c2_ddot + (d2y/dc22)*(c2_dot^2)
        y_ddot = dy_dc2 * c2_ddot + d2y_dc22 * (c2_dot ** 2)
        
        return y_ddot

    def dynamics(self, x1, x2, u1, t):
        """
        Calculates continuous derivative transformations for the trophophase system.
        Equations (1):
          dx1/dt = mu(x2) * x1
          dx2/dt = - (1/p1) * mu(x2) * x1 - m_S * x1 + p2 * u1
        """
        V = self.get_volume(t)
        c2 = x2 / V
        mu = (self.mu_max * c2) / (self.Ks + c2)
        
        # dx1/dt
        dx1dt = mu * x1
        
        # dx2/dt
        dx2dt = -(1.0 / self.p1) * mu * x1 - self.m_S * x1 + self.p2 * u1
        return dx1dt, dx2dt

    def step(self, state, u, t, dt):
        """
        Dormand-Prince (RK45) integration accounting for explicitly time-dependent dynamics.
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        
        # k1 at t
        dx1_1, ds1_1 = self.dynamics(x1, x2, u, t)
        
        # k2 at t + (1/5)*dt
        x1_2 = x1 + dt * (1/5 * dx1_1)
        x2_2 = x2 + dt * (1/5 * ds1_1)
        dx1_2, ds1_2 = self.dynamics(x1_2, x2_2, u, t + 0.2 * dt)
        
        # k3 at t + (3/10)*dt
        x1_3 = x1 + dt * (3/40 * dx1_1 + 9/40 * dx1_2)
        x2_3 = x2 + dt * (3/40 * ds1_1 + 9/40 * ds1_2)
        dx1_3, ds1_3 = self.dynamics(x1_3, x2_3, u, t + 0.3 * dt)
        
        # k4 at t + (4/5)*dt
        x1_4 = x1 + dt * (44/45 * dx1_1 - 56/15 * dx1_2 + 32/9 * dx1_3)
        x2_4 = x2 + dt * (44/45 * ds1_1 - 56/15 * ds1_2 + 32/9 * ds1_3)
        dx1_4, ds1_4 = self.dynamics(x1_4, x2_4, u, t + 0.8 * dt)
        
        # k5 at t + (8/9)*dt
        x1_5 = x1 + dt * (19372/6561 * dx1_1 - 25360/2187 * dx1_2 + 64448/6561 * dx1_3 - 212/729 * dx1_4)
        x2_5 = x2 + dt * (19372/6561 * ds1_1 - 25360/2187 * ds1_2 + 64448/6561 * ds1_3 - 212/729 * ds1_4)
        dx1_5, ds1_5 = self.dynamics(x1_5, x2_5, u, t + (8/9) * dt)
        
        # k6 at t + dt
        x1_6 = x1 + dt * (9017/3168 * dx1_1 - 355/33 * dx1_2 + 46732/5247 * dx1_3 + 49/176 * dx1_4 - 5103/18656 * dx1_5)
        x2_6 = x2 + dt * (9017/3168 * ds1_1 - 355/33 * ds1_2 + 46732/5247 * ds1_3 + 49/176 * ds1_4 - 5103/18656 * ds1_5)
        dx1_6, ds1_6 = self.dynamics(x1_6, x2_6, u, t + dt)

        # 5th-order accurate state update
        x1_next = x1 + dt * (35/384 * dx1_1 + 500/1113 * dx1_3 + 125/192 * dx1_4 - 2187/6784 * dx1_5 + 11/84 * dx1_6)
        x2_next = x2 + dt * (35/384 * ds1_1 + 500/1113 * ds1_3 + 125/192 * ds1_4 - 2187/6784 * ds1_5 + 11/84 * ds1_6)

        state_next = torch.cat([x1_next, x2_next], dim=1)
        
        # Return next state and tracking output evaluated at t + dt
        return state_next, self.get_y(state_next, t + dt)


    def get_plot_config(self):
        return [
            {
                            "cols": ["t"],
                            "labels": [r"$t$ [$\mathrm{h}$]"],
                            "xlabel": [r"$t$ [$\mathrm{h}$]"]
                        },
            {
                "cols": ["x_1", "x_2"],
                "labels": [r"$x_1$ [$\mathrm{g}$]", r"$x_2$ [$\mathrm{mg}$]"],
                "ylabel": [r"$x_1$ [$\mathrm{g}$]", r"$x_2$ [$\mathrm{mg}$]"]
            },
            {
                "cols": ["y"],
                "labels": [r"$y_1$ [$\mathrm{h}^{-1}$]"],
                "ylabel": r"$y_1$ [$\mathrm{h}^{-1}$]"
            },
            {
                "cols": ["u"],
                "labels": [r"$u_1$ [$\mathrm{h}^{-1}$]"],
                "ylabel": r"$u_1$ [$\mathrm{h}^{-1}$]"
            }
        ]

    def parse_state(self, state):
        return {
            "biomass_concentration": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate_concentration": state[1].item() if torch.is_tensor(state) else state[1]
        }
    

hyperparam_config_TrophophasePlant = {
    "plant" :{
        "mu_max": 0.12,
        "Ks": 50,
        "m_S": 23.0, 
        "p1": 0.00047,
        "p2": 200000.0,
        
        "u_1_D_center_min": 0.6,
        "u_1_D_center_max": 0.9,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min": 0,
        "x_1_hard_max": None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.12,

        "x10": 1500.0,   #wenn trainiert mit 1500 aber getesttet mit 1600, gute performnce
        "x20": 2000.0,

        "input_dim": 1,  # y
        "output_dim": 1  # u

        
    },
    "training_data_cfg" : {
        "u_1_D_center_min": 0.6,
        "u_1_D_center_max": 0.9,

        "u_1_hard_min": 0.0,
        "u_1_hard_max": 1,

        "x_1_hard_min": 0,
        "x_1_hard_max": None,

        "y_1_hard_min": 0,
        "y_1_hard_max": 0.12,

        "input_dim": 1,  # y
        "output_dim": 1,  # u

        "u_1_p" : 0.5,
        "u_1_lambd" : 4,

        "dt" : 0.01,

        "batch_size": 100,
        "seq_len":    2001,

        "min_correlation_threshold": -1.1

    },

    "train": {
        # Hyperparameters to be constant
        "device": "cuda", 
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate":1,
        "k_folds": 2,
        "lr": 1e-3,

        "mini_batch_size": 1,

        # Hyperparameters for tuning
        "epochs": 100,
        "min_correlation_threshold": -1.1,
        "constant_signal_probability": 0.0,
        "n_u": 2,
        "n_y": 2,
        "val_patience_epochs": 3,
        "val_min_delta": 0.0001,
        "lookback_offset": 20
    },

    "mamba": {
        "expand": 9,
        "d_state": 31,
        "d_conv": 9,
        "n_layer": 5
    },

    "esn": {
            "units": 200,   
            "lr": 0.5,
            "sr": 0.9,
            "ridge": 1e-7,    # Regularization coefficient  
        },

    "mamba_param_space" : {
    "mamba.d_conv":  {"type": "int", "low": 1, "high": 10},
    "mamba.d_state": {"type": "int", "low": 1, "high": 64},
    "mamba.expand":  {"type": "int", "low": 1, "high": 10},
    },

    "transformer": {
        "nhead" : 2,
        "num_layers" : 6,
        "dim_feedforward" : 256,
        "max_seq_len" : 2000
    },
    "transformer_param_space":  {
    "transformer.nhead":           {"type": "categorical", "choices": [1, 2, 3]}, # Must divide d_model
    "transformer.num_layers":      {"type": "int", "low": 1, "high": 4},
    "transformer.dim_feedforward": {"type": "categorical", "choices": [64, 128, 256]},
    },

    "simulate": {
        "batch_size": 10,
        "seq_len": 2001,
    }
}