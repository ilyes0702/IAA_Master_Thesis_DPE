import torch

class BajpaiReussPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Kinematic & Yield Parameters from Config
        self.mu_x = torch.tensor(hyperparam_config["plant"]["mu_x"], device=self.device)
        self.K_x = torch.tensor(hyperparam_config["plant"]["K_x"], device=self.device)
        self.Y_xs = torch.tensor(hyperparam_config["plant"]["Y_xs"], device=self.device)
        
        self.mu_p = torch.tensor(hyperparam_config["plant"]["mu_p"], device=self.device)
        self.S_ps = torch.tensor(hyperparam_config["plant"]["Y_ps"], device=self.device)
        self.K_I = torch.tensor(hyperparam_config["plant"]["K_I"], device=self.device)
        self.Y_ps = torch.tensor(hyperparam_config["plant"]["Y_ps"], device=self.device)
        
        self.m_x = torch.tensor(hyperparam_config["plant"]["m_x"], device=self.device)
        self.K = torch.tensor(hyperparam_config["plant"]["K"], device=self.device)
        self.S_0 = torch.tensor(hyperparam_config["plant"]["S_0"], device=self.device) # Feed substrate conc.
        self.S_F = torch.tensor(hyperparam_config["plant"]["S_F"], device=self.device) 

        self.K_ox = torch.tensor(hyperparam_config["plant"]["K_ox"], device=self.device)
        self.K_p = torch.tensor(hyperparam_config["plant"]["K_p"], device=self.device)
        self.K_op = torch.tensor(hyperparam_config["plant"]["K_op"], device=self.device)
        self.p = torch.tensor(hyperparam_config["plant"]["p"], device=self.device)
        self.m_o = torch.tensor(hyperparam_config["plant"]["m_o"], device=self.device)
        self.Y_xo = torch.tensor(hyperparam_config["plant"]["Y_xo"], device=self.device)
        self.Y_po = torch.tensor(hyperparam_config["plant"]["Y_po"], device=self.device)

        self.K_La = torch.tensor(hyperparam_config["plant"]["K_La"], device=self.device)

        self.C_L_star = 1


    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 4] tensor of [Biomass (X), Substrate (S), Penicillin (P), Volume (V)].
        """
        X_init = 0.1 * (0.95 + 0.10 * torch.rand((batch_size, 1), device=self.device))   # 0.1 to 2.1 g/L
        S_init = 15 * (0.95 + 0.10 * torch.rand((batch_size, 1), device=self.device))   # 0.1 to 1.1 g/L
        P_init = torch.zeros((batch_size, 1), device=self.device)              # 0.0 g/L initial product
        V_init = 1.0 * torch.ones((batch_size, 1), device=self.device)        # 100.0 L initial volume
        
        C_L_init = 1.16 * (0.95 + 0.10 * torch.rand((batch_size, 1), device=self.device))  # ±5% random variation
        return torch.cat([X_init, S_init, P_init, C_L_init, V_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates and returns Penicillin Production Rate (pi) as the observable plant output tracker.
        Alternately, you can swap this out for Biomass Growth Rate (mu) depending on target configurations.
        """
        S = state[:, 1:2]
        
        return S 

    def dynamics(self, X, S, P,C_L, V, u, t=None):
        """
        Calculates continuous derivative transformations for the state vectors.
        Control action input 'u' is mapped directly as Volumetric Flow Rate 'F' [L/h].
        """
        F = u # Volumetric substrate feeding rate
        dVdt = F*V/self.S_F
        dXdt = self.mu_x*S*C_L*X/((self.K_x*X+S)*(self.K_ox*X+C_L)) - (X/V)*dVdt
        dPdt = self.mu_p * S * (C_L**self.p) * X / ((self.K_p + S * (1 + S / self.K_I)) * (self.K_op * X + C_L**self.p)) - self.K * P - (P / V) * dVdt

        # For substrate (Equation 18)
        dSdt = - (self.mu_x / self.Y_xs) * S * C_L * X / ((self.K_x * X + S) * (self.K_ox * X + C_L)) \
            - (self.mu_p / self.Y_ps) * S * (C_L**self.p) * X / ((self.K_p + S * (1 + S / self.K_I)) * (self.K_op * X + C_L**self.p)) \
            - self.m_x * X + F - (S / V) * dVdt

        # For dissolved oxygen (Equation 19)
        dC_Ldt = - (self.mu_x / self.Y_xo) * S * C_L * X / ((self.K_x * X + S) * (self.K_ox * X + C_L)) \
                - (self.mu_p / self.Y_po) * S * (C_L**self.p) * X / ((self.K_p + S * (1 + S / self.K_I)) * (self.K_op * X + C_L**self.p)) \
                - self.m_o * X + self.K_La * (self.C_L_star - C_L) - (C_L / V) * dVdt
        
        return dXdt, dSdt, dPdt, dC_Ldt, dVdt

    def step(self, state, u, t, dt):
        """
        A native, batch-vectorized Dormand-Prince (RK45) adaptive-step integrator.
        Runs entirely on the GPU in parallel across all batch elements.
        Splits the interval [t, t + dt] into dynamic adaptive sub-steps.
        """
        batch_size = state.shape[0]
        device = state.device
        
        # We want to integrate from t_start to t_end
        t_start = t.item() if torch.is_tensor(t) else t
        t_end = t_start + dt
        
        # Dormand-Prince 5(4) Butcher Tableau Coefficients
        c2, a21 = 1/5, 1/5
        c3, a31, a32 = 3/10, 3/40, 9/40
        c4, a41, a42, a43 = 4/5, 44/45, -56/15, 32/9
        c5, a51, a52, a53, a54 = 8/9, 19372/6561, -25360/2187, 64448/6561, -212/729
        c6, a61, a62, a63, a64, a65 = 1.0, 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        
        # 5th-order coefficients (for the update)
        b1, b2, b3, b4, b5, b6, b7 = 35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0
        # 4th-order coefficients (for error estimation)
        b1_star, b2_star, b3_star, b4_star, b5_star, b6_star, b7_star = (
            5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
        )

        # Solver Tolerances
        rtol, atol = 1e-4, 1e-6
        
        # Track time and state for each batch element individually
        current_t = torch.full((batch_size, 1), t_start, device=device)
        y = state.clone()
        
        # Initialize step size h for each batch element
        h = torch.full((batch_size, 1), dt / 10.0, device=device)
        
        # Integrate until all batch elements have reached t_end
        # (This handles different trajectories adapting their step sizes independently)
        max_steps = 100
        step_count = 0
        
        while torch.any(current_t < t_end) and step_count < max_steps:
            # Ensure we do not step past t_end
            h = torch.clamp(h, max=t_end - current_t)
            
            # Helper to split state variables
            def unpack(s):
                return s[:, 0:1], s[:, 1:2], s[:, 2:3], s[:, 3:4], s[:, 4:5]
            
            # Stage 1
            k1 = torch.cat(self.dynamics(*unpack(y), u, current_t), dim=1)
            
            # Stage 2
            y2 = y + h * (a21 * k1)
            k2 = torch.cat(self.dynamics(*unpack(y2), u, current_t + c2 * h), dim=1)
            
            # Stage 3
            y3 = y + h * (a31 * k1 + a32 * k2)
            k3 = torch.cat(self.dynamics(*unpack(y3), u, current_t + c3 * h), dim=1)
            
            # Stage 4
            y4 = y + h * (a41 * k1 + a42 * k2 + a43 * k3)
            k4 = torch.cat(self.dynamics(*unpack(y4), u, current_t + c4 * h), dim=1)
            
            # Stage 5
            y5 = y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
            k5 = torch.cat(self.dynamics(*unpack(y5), u, current_t + c5 * h), dim=1)
            
            # Stage 6
            y6 = y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
            k6 = torch.cat(self.dynamics(*unpack(y6), u, current_t + c6 * h), dim=1)
            
            # Proposed next state (5th-order)
            y_next = y + h * (b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
            
            # Stage 7 (First Same As Last property - FSAL)
            k7 = torch.cat(self.dynamics(*unpack(y_next), u, current_t + h), dim=1)
            
            # Evaluate alternative 4th-order state to calculate error
            y_next_star = y + h * (b1_star*k1 + b2_star*k2 + b3_star*k3 + b4_star*k4 + b5_star*k5 + b6_star*k6 + b7_star*k7)
            
            # Truncation error calculation
            error = torch.abs(y_next - y_next_star)
            scale = atol + rtol * torch.max(torch.abs(y), torch.abs(y_next))
            
            # Normalize error across variables for each batch element
            # (Root Mean Square norm across the 5 state dimensions)
            norm_error = torch.sqrt(torch.mean((error / scale) ** 2, dim=1, keepdim=True))
            
            # Condition: If error is acceptable, accept step. Otherwise, reject and shrink h.
            # norm_error <= 1.0 means error is within tolerance limits
            step_accepted = norm_error <= 1.0
            
            # Update state for elements whose step was accepted
            y = torch.where(step_accepted, y_next, y)
            current_t = torch.where(step_accepted, current_t + h, current_t)
            
            # Step size controller (PI control logic or standard scaling)
            # Prevent h from scaling up/down too rapidly (safety factors 0.2 and 5.0)
            safety_factor = 0.9
            scale_factor = safety_factor * (norm_error ** -0.2)
            scale_factor = torch.clamp(scale_factor, min=0.2, max=5.0)
            
            # Apply scaling factor to adjust next step size
            h = h * scale_factor
            
            # Small floor to prevent infinite loops on severe integration failure
            h = torch.clamp(h, min=1e-5)
            step_count += 1

        return y, self.get_y(y, t_end)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2", "x3", "x4"],
                "labels": ["Biomass concentration", "Substrate (S)", "Penicillin (P)", "Volume (V)"],
                "title": "Fed-Batch Penicillin State Evolution",
                "ylabel": "Concentration [g/L] / Vol [L]"
            },
            {
                "cols": ["y"],
                "labels": [rf"$S \; / \; \mathrm{{g \cdot L^{{-1}}}}$"],
                "title": "Penicillin Synthesis Output Tracking",
                "ylabel": "Specific Production Rate [1/h]"
            },
            {
                "cols": ["u"],
                "labels": [rf"$F \; / \; \mathrm{{L \cdot h^{{-1}}}}$"],
                "title": "Control Input Flow Profile",
                "ylabel": "Flow Rate [L/h]"
            }
        ]

    def parse_state(self, state):
        return {
            "biomass": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate": state[1].item() if torch.is_tensor(state) else state[1],
            "penicillin": state[2].item() if torch.is_tensor(state) else state[2],
            "dissolved_oxygen": state[3].item() if torch.is_tensor(state) else state[3],
            "volume": state[4].item() if torch.is_tensor(state) else state[4]
        }
    

hyperparam_config_BajpaiReussPlant = {
    "plant": {
        # --- Kinematic & Yield Parameters ---
        "mu_x": 0.092,     # Maximum specific growth rate [1/h] (Contois)
        "K_x": 0.15,       # Contois saturation constant [g substrate / g biomass]
        "mu_p": 0.005,       # Maximum specific rate of product formation ( g product / (g dry wt cells)* h)
        

        
        "Y_xs": 0.45,       # Yield of biomass on substrate [g dry wt cells/g substrate]
        "Y_ps": 0.9,        # Xield of product on substrate [g product/ g substrate]

        "m_x" : 0.014,      # Maintenance requirement of substrate [g substrate/ (g dry wet cell)*h]  
        "F": 0.33,          # [g glucose/(dm^3)*h]

        "S_0": 0.1,         # [g/dm^3]                
        
        "K_p": 0.0002,      # Monod saturation constants for substrate limitation of product formation [g/dm^3]
        "K_I": 0.10,        # Substrate inhibition constant for product [g/dm^3]

        "S_F": 400,       #Substrate concentration in feed stream [g/dm^3] ##NOT SURE
        
        "K": 0.04,          # First-order decay rate constant for product [1/h]

        "K_La": 60,         #[1/h]

        "C_L_star": 0.27,   # solubility of oxygen in broth [mmol/dm^3]


        "K_ox": 0.00111,    # Contois saturation constant for oxygen limitation of product formation [(mmol/ g dry wt cells)]
        "K_op": 3e-5,       # Contois saturation constant for oxygen limitation of product formation [(mmol/g dry wt cells)^1/p]
        "p": 2.74,          # Exponent of CL in oxygen limitation of product formation
        "m_o": 0.467,       # Maintenance requirement of oxygen [mmol O2/(g dry wt cells)*h]
        "Y_xo": 0.04,       # Yield of biomass on oxygen [g dry wt cells/ mmol O2]
        "Y_po": 0.2,        # Yield of product on oxygen [g product/mmol O2]

        # --- Volumetric Flow Rate Control Bounds (u_1 = F) ---
        "u_1_D_center_min": 0.0,  # Typical optimal feeding profile floor [L/h]
        "u_1_D_center_max": 1,  # Typical optimal feeding profile ceiling [L/h]

        "u_1_hard_min": 0.0,       # Pump fully off [L/h]
        "u_1_hard_max": 2.0,       # Maximum physical actuator saturation pump limit [L/h]

        # --- Hard State Lower/Upper Bounds ---
        "x_1_hard_min": 0.0,       # Biomass (X) cannot be negative
        "x_2_hard_min": 0.0,       # Substrate (S) cannot be negative
        "x_3_hard_min": 0.0,       # Penicillin (P) cannot be negative
        "x_4_hard_min": 0.0,       # C_L cannot be negative
        "x_5_hard_min": 0.1,       # Volume (V) must maintain a physical minimum heel (e.g. 0.1L)
        "x_5_hard_max": 15.0,      # Maximum structural capacity limit of the vessel tank [L]

        "input_dim": 1,
        "output_dim": 1,
    },
    "signal": {
        "lambd": 10,
        "p": 0.15,
        "seq_len": 1001,
        "dt": 0.01                 # Fermentations evolve slower than chemostats; a slightly higher dt is normal
    },
    "train": {
        "k_folds": 5,
        "epochs": 100,
        "batch_size": 200,
        "lr": 1e-3,
        "device": "cuda",
        "delay_steps": 10,
        "loss_function": "MSELoss()", 
        "lr_decay_rate": 1,
        "min_correlation_threshold": -1
    },
    "mamba": {
        "d_state": 16,
        "input_dim": 1,            # Tracking 1 observable output (e.g., pi or mu)
        "output_dim": 1,           # Regulating 1 physical control output (Feed Rate F)
        "expand": 32               # Expansion factor maps core dimension (input_dim * 2) -> 64 internal tracking lines
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 2001,            # Fed-batch runs span much longer horizons (e.g., 200 hours total at dt=0.25)
    }
}