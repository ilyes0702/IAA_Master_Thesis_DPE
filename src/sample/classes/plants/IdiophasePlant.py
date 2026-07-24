import torch

class IdiophasePlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Biological and Plant Parameters from Context
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu_max"], device=self.device) # e.g., 0.12
        self.Ks = torch.tensor(hyperparam_config["plant"]["Ks"], device=self.device)         # e.g., 50
        self.p1 = torch.tensor(hyperparam_config["plant"]["p1"], device=self.device)         # e.g., 0.00047
        self.p2 = torch.tensor(hyperparam_config["plant"]["p2"], device=self.device)         # e.g., 200000
        self.p5 = torch.tensor(hyperparam_config["plant"]["p5"], device=self.device)         # e.g., 0.9
        self.p6 = torch.tensor(hyperparam_config["plant"]["p6"], device=self.device)         # e.g., 100
        self.p7 = torch.tensor(hyperparam_config["plant"]["p7"], device=self.device)         # e.g., 0.04
        self.q = torch.tensor(hyperparam_config["plant"]["q"], device=self.device)           # e.g., 2000
        self.mu_Pen = torch.tensor(hyperparam_config["plant"]["mu_Pen"], device=self.device) # e.g., 3
        self.m_S = torch.tensor(hyperparam_config["plant"]["m_S"], device = self.device)

        self.hyperparam_config = hyperparam_config

        # Fixed volume for Idiophase if specified as constant, or base initialization
        # Table 2 specifies V(t) = 170 for the idiophase phase
        self.V_const = torch.tensor(hyperparam_config["plant"].get("V_idiophase", 170.0), device=self.device)

    def get_V(self, t):
        """
        Returns the reactor volume V(t). 
        Can handle either tensor time horizons or scalar step environments.
        """
        if torch.is_tensor(t):
            return torch.full_like(t, self.V_const, device=self.device)
        return self.V_const

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 4] tensor initialized to identical constant values across the batch.
        """
        # torch.full creates a 2D tensor of shape [batch_size, 1] filled with the scalar value
        x1_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x10"], device=self.device)
        x2_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x20"], device=self.device)
        x3_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x30"], device=self.device)
        x4_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x40"], device=self.device)

        return torch.cat([x1_init, x2_init, x3_init, x4_init], dim=1)

    def get_y(self, state, t):
        """
        Calculates and returns the 2-dimensional MIMO tracker vector:
        y = [y1, y2] = [Growth Rate (mu), Precursor Concentration (x3/V)]
        """
        x2 = state[:, 1:2]
        x3 = state[:, 2:3]
        V = self.get_V(t)

        # Monod growth kinetics equation
        mu = (self.mu_max * x2) / (self.Ks * V + x2)
        
        # Precursor Concentration
        c3 = x3 / V
        
        # Concatenate into MIMO tracking vector [Batch, 2]
        return torch.cat([mu, c3], dim=1)

    def dynamics(self, x1, x2, x3, x4, u, t):
        """
        Calculates continuous derivative state transformations for the Idiophase.
        u parameter is a [Batch, 2] tensor containing [u1, u2].
        """
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        V = self.get_V(t)

        # Monod growth kinetics
        mu = (self.mu_max * x2) / (self.Ks * V + x2)

        # System differential state equations
        dx1dt = mu * x1
        dx2dt = - (1.0 / self.p1) * mu * x1 - (1.0 / self.p5) * self.mu_Pen * x1 - self.m_S * x1 + self.p2 * u1
        dx3dt = - (1.0 / self.q) * self.mu_Pen * x1 + self.p6 * u2
        dx4dt = self.mu_Pen * x1 - self.p7 * x4

        return dx1dt, dx2dt, dx3dt, dx4dt

    

    def step(self, state, u, t, dt):
        """
        MIMO Classic 4th-order Runge-Kutta (RK4) numerical integration block.
        Designed for constant step size execution.
        """
        # Unpack state dimensions
        x1, x2, x3, x4 = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]

        # Precompute half time step
        dt_half = dt * 0.5

        # --- STAGE 1 (k1) ---
        dx1_1, dx2_1, dx3_1, dx4_1 = self.dynamics(
            x1, x2, x3, x4, u, t
        )

        # --- STAGE 2 (k2) ---
        dx1_2, dx2_2, dx3_2, dx4_2 = self.dynamics(
            x1 + dt_half * dx1_1,
            x2 + dt_half * dx2_1,
            x3 + dt_half * dx3_1,
            x4 + dt_half * dx4_1,
            u, t + dt_half
        )

        # --- STAGE 3 (k3) ---
        dx1_3, dx2_3, dx3_3, dx4_3 = self.dynamics(
            x1 + dt_half * dx1_2,
            x2 + dt_half * dx2_2,
            x3 + dt_half * dx3_2,
            x4 + dt_half * dx4_2,
            u, t + dt_half
        )

        # --- STAGE 4 (k4) ---
        dx1_4, dx2_4, dx3_4, dx4_4 = self.dynamics(
            x1 + dt * dx1_3,
            x2 + dt * dx2_3,
            x3 + dt * dx3_3,
            x4 + dt * dx4_3,
            u, t + dt
        )

        # --- COMBINE STAGES ---
        dt_6 = dt / 6.0
        x1_next = x1 + dt_6 * (dx1_1 + 2.0 * dx1_2 + 2.0 * dx1_3 + dx1_4)
        x2_next = x2 + dt_6 * (dx2_1 + 2.0 * dx2_2 + 2.0 * dx2_3 + dx2_4)
        x3_next = x3 + dt_6 * (dx3_1 + 2.0 * dx3_2 + 2.0 * dx3_3 + dx3_4)
        x4_next = x4 + dt_6 * (dx4_1 + 2.0 * dx4_2 + 2.0 * dx4_3 + dx4_4)
        
        state_next = torch.cat([x1_next, x2_next, x3_next, x4_next], dim=1)
        
        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2", "x3", "x4"],
                "labels": [
                    r"$X$ / $\mathrm{mg}$", 
                    r"$S$ / $\mathrm{g}$", 
                    r"$M_{\mathrm{pre}}$ / $\mathrm{g}$", 
                    r"$M_{\mathrm{pen}}$ / $\mathrm{g}$"
                ],
                "ylabel": [
                    r"$X$ / $\mathrm{mg}$", 
                    r"$S$ / $\mathrm{g}$", 
                    r"$M_{\mathrm{pre}}$ / $\mathrm{g}$", 
                    r"$M_{\mathrm{pen}}$ / $\mathrm{g}$"
                ]
            },
            {
                "cols": ["y1", "y2"],
                "labels": [
                    r"$\mu$ / $\mathrm{h}^{-1}$", 
                    r"$c_{\mathrm{pre}}$ / $\mathrm{g}\,\mathrm{L}^{-1}$"
                ],
                "ylabel": [
                    r"$\mu$ / $\mathrm{h}^{-1}$", 
                    r"$c_{\mathrm{pre}}$ / $\mathrm{g}\,\mathrm{L}^{-1}$"
                ]
            },
            {
                "cols": ["u1", "u2"],
                "labels": [
                    r"$F_{\mathrm{glu}}$ / $\mathrm{L}\,\mathrm{h}^{-1}$", 
                    r"$F_{\mathrm{pre}}$ / $\mathrm{L}\,\mathrm{h}^{-1}$"
                ],
                "ylabel": [
                    r"$F_{\mathrm{glu}}$ / $\mathrm{L}\,\mathrm{h}^{-1}$", 
                    r"$F_{\mathrm{pre}}$ / $\mathrm{L}\,\mathrm{h}^{-1}$"
                ]
            }
        ]

    def parse_state(self, state):
        return {
            "biomass_mass": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate_mass": state[1].item() if torch.is_tensor(state) else state[1],
            "precursor_mass": state[2].item() if torch.is_tensor(state) else state[2],
            "penicillin_mass": state[3].item() if torch.is_tensor(state) else state[3]
        }
    



hyperparam_config_IdiophasePlant = {
        "signal": {
            "seq_len": 2001,
            "dt": 0.01,
            #/ 🕹️ Channel 1 Signal Parameters (e.g., highly dynamic)
            "u_1_lambd": 4,        
            "u_1_p": 0.5,            
            
            #// 🕹️ Channel 2 Signal Parameters (e.g., highly filtered, slow moving)
            "u_2_lambd": 4,        
            "u_2_p": 0.5,
        
        },
        "train": {
            "batch_size": 1000,
            "device": "cuda",
            "delay_steps": 1,
            "epochs": 50,
            "lr": 1e-3,
            "loss_function": "MSELoss()",
            "k_folds": 5,
            "lr_decay_rate":1,
            "min_correlation_threshold": -1.1,
            "n_y": 4,
            "n_u": 2,
            "lookback_offset": 100,
            "val_patience_epochs": 3,
            "val_min_delta": 0.00001
        },
        "plant": {
            "mu_max": 0.12,
            "Ks": 50.0,
            "p1": 0.00047,
            "p2": 200000.0,
            "p5": 0.9,
            "p6": 100.0,
            "p7": 0.04,
            "q": 2000.0,
            "mu_Pen": 3.0,
            "V_idiophase": 170.0,
            "m_S": 23,

            "x10": 1500,
            "x20": 2000,
            "x30": 25,
            "x40": 1600,

            "x_1_hard_min": 0.0,
            "x_1_hard_max": None,

            "x_2_hard_min": 0.0,
            "x_2_hard_max": None,

            "x_3_hard_min": 0.0,
            "x_3_hard_max": None,

            "x_4_hard_min": 0.0,
            "x_4_hard_max": None,
            
            "u_1_hard_min": 0.0,
            "u_1_hard_max": 1.0,

            "u_2_hard_min": 0.0,
            "u_2_hard_max": 1.0,

            "y_1_hard_min": 0.0,
            "y_1_hard_max": 0.12,

            "y_2_hard_min": 0.0,
            "y_2_hard_max": None,

            "u_1_D_center_min": 0.6,
            "u_1_D_center_max": 0.9,

            "u_2_D_center_min": 0.0,
            "u_2_D_center_max": 0.5,

            
            "input_dim": 2,  # y1, y2
            "output_dim": 2  # u1, u2
        },
        "mamba": {
            "expand": 1,
            "d_state": 1,
        },
        "simulate": {
            "batch_size": 10,
            "seq_len": 2001,
        }
    }