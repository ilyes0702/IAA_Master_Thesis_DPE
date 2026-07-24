import torch

class IndForProteinProductionPlant:
    def __init__(self, hyperparam_config):
        self.hyperparam_config = hyperparam_config
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Constants and parameters from image description
        # (Ensure these keys are updated in your hyperparam dict as needed)
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu_max"], device=self.device)
        self.K_CI = torch.tensor(hyperparam_config["plant"]["K_CI"], device=self.device)
        self.k_22 = torch.tensor(hyperparam_config["plant"]["k_22"], device=self.device)
        
        self.K_s = torch.tensor(hyperparam_config["plant"]["K_s"], device=self.device)
        self.f_I_0 = torch.tensor(hyperparam_config["plant"]["f_I_0"], device=self.device)
        self.C_n_f = torch.tensor(hyperparam_config["plant"]["C_n_f"], device=self.device)
        self.Y = torch.tensor(hyperparam_config["plant"]["Y"], device=self.device)
        self.K_CN = torch.tensor(hyperparam_config["plant"]["K_CN"], device=self.device)
        self.k_11 = torch.tensor(hyperparam_config["plant"]["k_11"], device=self.device)
        self.K_IX = torch.tensor(hyperparam_config["plant"]["K_IX"], device=self.device)
        self.f_max = torch.tensor(hyperparam_config["plant"]["f_max"], device=self.device)
        
        self.K_I = torch.tensor(hyperparam_config["plant"]["K_I"], device=self.device)
        self.C_i_f = torch.tensor(hyperparam_config["plant"]["C_i_f"], device=self.device)
        

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 7] tensor corresponding to the 7 system states:
        [x1 (Vol), x2 (X), x3 (N), x4 (P), x5 (Ind), x6 (Shock), x7 (Recovery)]
        """
        # Distribute state values randomly within realistic boundaries around nominal points
        
        x1_init = self.hyperparam_config["plant"]["x10"] * torch.ones((batch_size, 1), device=self.device)
        x2_init = self.hyperparam_config["plant"]["x20"] * torch.ones((batch_size, 1), device=self.device)
        x3_init = self.hyperparam_config["plant"]["x30"] * torch.ones((batch_size, 1), device=self.device)
        x4_init = self.hyperparam_config["plant"]["x40"] * torch.ones((batch_size, 1), device=self.device)
        x5_init = self.hyperparam_config["plant"]["x50"] * torch.ones((batch_size, 1), device=self.device)
        x6_init = self.hyperparam_config["plant"]["x60"] * torch.ones((batch_size, 1), device=self.device)
        x7_init = self.hyperparam_config["plant"]["x70"] * torch.ones((batch_size, 1), device=self.device)

        return torch.cat([x1_init, x2_init, x3_init, x4_init, x5_init, x6_init, x7_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates and returns the 3 specified desired (controlled) output variables:
        y = [x1 (Volume), x2 (Cell Density), x4 (Protein Concentration)]
        """
        x1 = state[:, 0:1]
        x2 = state[:, 1:2]
        x4 = state[:, 3:4]
        return torch.cat([x1, x2, x4], dim=1)

    def dynamics(self, x1, x2, x3, x4, x5, x6, x7, u, t=None):
        """
        Calculates differential equations derived from Equation (1)-(4) in the plant document.
        u is a 2D tensor where:
          u[:, 0:1] = u1 (Glucose feed rate)
          u[:, 1:2] = u2 (Inducer feed rate)
        """
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        
        R_R = self.K_CI/(self.K_CI+x5)
        # Intermediate kinetics (Eq 2, 3, 4)
        mu = (self.mu_max * x3 / (self.K_CN + x3 * (1.0 + x3 / self.K_s))) * \
             (x6 + x7 * R_R)
             
        R_f_p = (self.f_max * x3 / (self.K_CN + x3 * (1.0 + x3 / self.K_s))) * \
            ((self.f_I_0 + x5) / (self.K_I + x5))
            
        k_1 = self.k_11 * x5 / (self.K_IX + x5)
        k_2 = self.k_22 * x5 / (self.K_IX + x5)

        # Main Differential State Vector Mapping (Eq 1)
        dx1dt = u1 + u2
        dx2dt = x2 * mu - ((u1 + u2) / x1) * x2
        dx3dt = (u1 * self.C_n_f / x1) - ((u1 + u2) / x1) * x3 - (mu / self.Y) * x2
        dx4dt = x2 * R_f_p - ((u1 + u2) / x1) * x4
        dx5dt = (u2 * self.C_i_f / x1) - ((u1 + u2) / x1) * x5
        dx6dt = -k_1 * x6
        dx7dt = k_2 * (1.0 - x7)

        return dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt

    def step(self, state, u, t, dt):
        """
        Vectorized adaptive-step Dormand-Prince (RK45) parallel tracking engine.
        """
        batch_size = state.shape[0]
        device = state.device
        
        t_start = t.item() if torch.is_tensor(t) else t
        t_end = t_start + dt
        
        # Dormand-Prince Tableau Matrix constants
        c2, a21 = 1/5, 1/5
        c3, a31, a32 = 3/10, 3/40, 9/40
        c4, a41, a42, a43 = 4/5, 44/45, -56/15, 32/9
        c5, a51, a52, a53, a54 = 8/9, 19372/6561, -25360/2187, 64448/6561, -212/729
        c6, a61, a62, a63, a64, a65 = 1.0, 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        b1, b2, b3, b4, b5, b6, b7 = 35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0
        b1_star, b2_star, b3_star, b4_star, b5_star, b6_star, b7_star = (
            5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
        )

        rtol, atol = 1e-4, 1e-6
        current_t = torch.full((batch_size, 1), t_start, device=device)
        y = state.clone()
        h = torch.full((batch_size, 1), dt / 10.0, device=device)
        
        max_steps, step_count = 100, 0
        
        while torch.any(current_t < t_end) and step_count < max_steps:
            h = torch.clamp(h, max=t_end - current_t)
            
            def unpack(s):
                return (s[:, 0:1], s[:, 1:2], s[:, 2:3], s[:, 3:4], 
                        s[:, 4:5], s[:, 5:6], s[:, 6:7])
            
            k1 = torch.cat(self.dynamics(*unpack(y), u, current_t), dim=1)
            y2 = y + h * (a21 * k1)
            k2 = torch.cat(self.dynamics(*unpack(y2), u, current_t + c2 * h), dim=1)
            y3 = y + h * (a31 * k1 + a32 * k2)
            k3 = torch.cat(self.dynamics(*unpack(y3), u, current_t + c3 * h), dim=1)
            y4 = y + h * (a41 * k1 + a42 * k2 + a43 * k3)
            k4 = torch.cat(self.dynamics(*unpack(y4), u, current_t + c4 * h), dim=1)
            y5 = y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
            k5 = torch.cat(self.dynamics(*unpack(y5), u, current_t + c5 * h), dim=1)
            y6 = y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
            k6 = torch.cat(self.dynamics(*unpack(y6), u, current_t + c6 * h), dim=1)
            
            y_next = y + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
            k7 = torch.cat(self.dynamics(*unpack(y_next), u, current_t + h), dim=1)
            y_next_star = y + h * (b1_star*k1 + b2_star*k2 + b3_star*k3 + b4_star*k4 + b5_star*k5 + b6_star*k6 + b7_star*k7)
            
            error = torch.abs(y_next - y_next_star)
            scale = atol + rtol * torch.max(torch.abs(y), torch.abs(y_next))
            norm_error = torch.sqrt(torch.mean((error / scale) ** 2, dim=1, keepdim=True))
            
            step_accepted = norm_error <= 1.0
            y = torch.where(step_accepted, y_next, y)
            current_t = torch.where(step_accepted, current_t + h, current_t)
            
            scale_factor = torch.clamp(0.9 * (norm_error ** -0.2), min=0.2, max=5.0)
            h = h * scale_factor
            h = torch.clamp(h, min=1e-5)
            step_count += 1

        return y, self.get_y(y, t_end)

    def get_plot_config(self):
        """
        Dynamically configures labels following publication constraints.
        Using math-mode strings with double braces inside raw f-strings.
        """
        return [
            {
                "cols": ["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"],
                "labels": [r"$x_1 \; / \; \mathrm{{L}}$", r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$"],
                "ylabel": [r"$x_1 \; / \; \mathrm{{L}}$", r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$",r"$x_1 \; / \; \mathrm{{L}}$"]
            },
            {
                "cols": ["u_1", "u_2"],
                "labels": [
                    rf"$u_1 \; / \; \mathrm{{L \cdot h^{{-1}}}}$", 
                    rf"$u_2 \; / \; \mathrm{{L \cdot h^{{-1}}}}$"
                ],
                "ylabels": [
                    rf"$u_1 \; / \; \mathrm{{L \cdot h^{{-1}}}}$", 
                    rf"$u_2 \; / \; \mathrm{{L \cdot h^{{-1}}}}$"
                ]
            },
            {
                "cols": ["y_1", "y_2", "y_3"],
                "labels": [
                    rf"$x_1 \; / \; \mathrm{{L}}$", 
                    rf"$x_2 \; / \; \mathrm{{g \cdot L^{{-1}}}}$", 
                    rf"$x_4 \; / \; \mathrm{{g \cdot L^{{-1}}}}$"
                ],
                "ylabels": [
                    rf"$x_1 \; / \; \mathrm{{L}}$", 
                    rf"$x_2 \; / \; \mathrm{{g \cdot L^{{-1}}}}$", 
                    rf"$x_4 \; / \; \mathrm{{g \cdot L^{{-1}}}}$"
                ]
            }
        ]

    def parse_state(self, state):
        return {
            "volume": state[0].item() if torch.is_tensor(state) else state[0],
            "cell_density": state[1].item() if torch.is_tensor(state) else state[1],
            "nutrient": state[2].item() if torch.is_tensor(state) else state[2],
            "protein": state[3].item() if torch.is_tensor(state) else state[3],
            "inducer": state[4].item() if torch.is_tensor(state) else state[4],
            "shock_factor": state[5].item() if torch.is_tensor(state) else state[5],
            "recovery_factor": state[6].item() if torch.is_tensor(state) else state[6]
        }
    
hyperparam_config_IndForProteinProductionPlant = {
    "plant": {
        # --- Kinematic & Yield Parameters (Lee & Ramirez Model) ---
        "mu_max": 0.407,       # Maximum specific growth rate [1/h]
        "K_CI": 0.22,         # Inducer inhibition/shock structural constant [g/L]
        "k_22": 0.09,          # Deactivation rate coefficient for protein shock [1/h]
        "K_s": 14814.8,           # Substrate inhibition constant multiplier [g/L]
        "f_I_0": 0.0005,
        "C_n_f": 100,
        "Y": 0.51,



        "K_CN": 0.108,          # Nitrogen/Nutrient saturation constant [g/L]
        "k_11": 0.09,          # Deactivation rate coefficient for growth shock [1/h]
        "K_IX": 0.034,          # Cell density impact factor on deactivation [g/L]
        
        "f_max": 0.095,     # Max specific foreign protein production rate [1/h]
        "K_IX": 0.034,     # Inducer activation affinity constant [g/L]   
        
        "f_max": 0.095,
        "K_I": 0.022,
        "C_i_f": 4,        
        
        
        #"N": 100.0,           # Nutrient concentration in glucose feed stream [g/L]
        #"I": 4.0,             # Inducer concentration in activator feed stream [g/L]
        #"Y": 0.5,             # Biomass growth yield coefficient [g dry cells / g nutrient]

        # --- Actuator Flow Rate Bounds (2 Inputs: u_1 = Glucose Feed, u_2 = Inducer Feed) ---
        "u_1_hard_min": 0.0,       # Glucose pump fully off [L/h]
        "u_1_hard_max": 1.5,       # Max glucose volumetric flow capacity [L/h]
        "u_2_hard_min": 0.0,       # Inducer pump fully off [L/h]
        "u_2_hard_max": 0.5,       # Max inducer volumetric flow capacity [L/h]

        "u_1_D_center_min": 0.05,  # Operational envelope floors
        "u_1_D_center_max": 0.80,
        "u_2_D_center_min": 0.00,
        "u_2_D_center_max": 0.25,

        # --- State Trajectory Bounds (7 Dimensions) ---
        "x_1_hard_min": 0.0,       # Minimum reactor heel volume to cover sensors [L]
        "x_1_hard_max": None,       # Total structural capacity of the tank vessel [L]
        "x_2_hard_min": 0.0,       # Biomass density floor [g/L]
        "x_3_hard_min": 0.0,       # Nutrient limitation floor [g/L]
        "x_4_hard_min": 0.0,       # Protein concentration floor [g/L]
        "x_5_hard_min": 0.0,       # Inducer concentration floor [g/L]
        "x_6_hard_min": 0.0,       # Shock factor boundary bounds
        "x_6_hard_max": None,
        "x_7_hard_min": 0.0,       # Recovery factor bounds
        "x_7_hard_max": None,

        "x10": 1,
        "x20": 0.1,
        "x30": 40,
        "x40": 0,
        "x50": 0,
        "x60": 1,
        "x70": 0,

        # --- System Order Configurations ---
        "input_dim": 2,            # Dim(u) = [u1, u2]
        "output_dim": 3,           # Dim(y) = [x1, x2, x4]
    },
    "signal": {
        "lambd": 10,               # Signal filtering/noise properties
        "p": 0.15,                 # Discontinuity probability factor
        "seq_len": 1501,           # Length of sequential time-series paths
        "dt": 0.01                 # Timestep integration window [h] (50 ms steps)
    },
    "train": {
        "k_folds": 5,              # Cross-validation splits
        "epochs": 150,             # Total training iterations
        "batch_size": 1000,         # Number of batch elements
        "lr": 1e-3,                # Base optimization learning rate
        "device": "cuda",          # Core processing target execution context
        "delay_steps": 5,          # Latency control parameter markers
        "loss_function": "MSELoss()",
        "lr_decay_rate": 1, # 0.98,     # Multiplicative factor per epoch decay
        "min_correlation_threshold": -1.1,
        "n_y": 2,
        "n_u": 2,
        "val_patience_epochs": 3,
        "val_min_delta": 0.0001,
        "lookback_offset": 10,
    },
    "mamba": {
        "d_state": 32,             # State expansion dimension space
        "expand": 1               # Core internal block expansion coefficient width
    },
    "simulate": {
        "batch_size": 16,          # Validation trajectory evaluation batch scale
        "seq_len": 3001            # Length of a full multi-day production run
    }
}

