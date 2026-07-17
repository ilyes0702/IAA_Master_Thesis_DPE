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
        X_init = 0.1 #* (0.95 + 0.10 * torch.rand((batch_size, 1), device=self.device))   # 0.1 to 2.1 g/L
        S_init = 15 #* (0.95 + 0.10 * torch.rand((batch_size, 1), device=self.device))   # 0.1 to 1.1 g/L
        P_init = torch.zeros((batch_size, 1), device=self.device)              # 0.0 g/L initial product
        V_init = 100.0 * torch.ones((batch_size, 1), device=self.device)        # 100.0 L initial volume
        
        C_L_init = 1.16 #* (0.95 + 0.10 * torch.rand((batch_size, 1), device=self.device))  # ±5% random variation
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
        Standard Runge-Kutta 4th Order numerical integration execution block.
        Updates all 5 state variables: X, S, P, C_L, V without clamping modifications.
        """
        # Unpack current batch states (Shape: [batch_size, 1])
        X   = state[:, 0:1]
        S   = state[:, 1:2]
        P   = state[:, 2:3]
        C_L = state[:, 3:4]
        V   = state[:, 4:5]
        
        # --- k1 ---
        dX1, dS1, dP1, dC_L1, dV1 = self.dynamics(X, S, P, C_L, V, u, t)
        
        # --- k2 ---
        X2   = X   + 0.5 * dt * dX1
        S2   = S   + 0.5 * dt * dS1
        P2   = P   + 0.5 * dt * dP1
        C_L2 = C_L + 0.5 * dt * dC_L1
        V2   = V   + 0.5 * dt * dV1
        dX2, dS2, dP2, dC_L2, dV2 = self.dynamics(X2, S2, P2, C_L2, V2, u, t + 0.5 * dt)
        
        # --- k3 ---
        X3   = X   + 0.5 * dt * dX2
        S3   = S   + 0.5 * dt * dS2
        P3   = P   + 0.5 * dt * dP2
        C_L3 = C_L + 0.5 * dt * dC_L2
        V3   = V   + 0.5 * dt * dV2
        dX3, dS3, dP3, dC_L3, dV3 = self.dynamics(X3, S3, P3, C_L3, V3, u, t + 0.5 * dt)
        
        # --- k4 ---
        X4   = X   + dt * dX3
        S4   = S   + dt * dS3
        P4   = P   + dt * dP3
        C_L4 = C_L + dt * dC_L3
        V4   = V   + dt * dV3
        dX4, dS4, dP4, dC_L4, dV4 = self.dynamics(X4, S4, P4, C_L4, V4, u, t + dt)
        
        # --- RK4 Assembly Weighting ---
        X_next   = X   + (dt / 6.0) * (dX1   + 2 * dX2   + 2 * dX3   + dX4)
        S_next   = S   + (dt / 6.0) * (dS1   + 2 * dS2   + 2 * dS3   + dS4)
        P_next   = P   + (dt / 6.0) * (dP1   + 2 * dP2   + 2 * dP3   + dP4)
        C_L_next = C_L + (dt / 6.0) * (dC_L1 + 2 * dC_L2 + 2 * dC_L3 + dC_L4)
        V_next   = V   + (dt / 6.0) * (dV1   + 2 * dV2   + 2 * dV3   + dV4)
        
        state_next = torch.cat([X_next, S_next, P_next, C_L_next, V_next], dim=1)
        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2", "x3", "x4"],
                "labels": ["Biomass (X)", "Substrate (S)", "Penicillin (P)", "Volume (V)"],
                "title": "Fed-Batch Penicillin State Evolution",
                "ylabel": "Concentration [g/L] / Vol [L]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual Production Rate (π)", "Target π_ref"],
                "title": "Penicillin Synthesis Output Tracking",
                "ylabel": "Specific Production Rate [1/h]"
            },
            {
                "cols": ["u"],
                "labels": ["Substrate Feed Rate (F)"],
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
        "delay_steps": 1,
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