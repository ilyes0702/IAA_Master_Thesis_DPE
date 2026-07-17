import torch

class CoCultivationPlant:
    """
    Implements a two-strain microbial consortium in a chemostat with 
    light-mediated (optogenetic) growth control using PyTorch for batch simulations.
    """
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]
        self.plant_cfg = hyperparam_config["plant"]

        # Kinetic and Bioprocess Parameters
        self.mu_max1 = self.plant_cfg["mu_max1"]
        self.mu_max2 = self.plant_cfg["mu_max2"]
        self.k_g_1 = self.plant_cfg["k_g_1"]
        self.k_g_2 = self.plant_cfg["k_g_2"]
        self.f_c = self.plant_cfg["f_c"]
        self.k_a_1 = self.plant_cfg["k_a_1"]
        self.k_a_2 = self.plant_cfg["k_a_2"]
        self.Y_g_b1 = self.plant_cfg["Y_g_b1"]
        self.Y_g_b2 = self.plant_cfg["Y_g_b2"]
        self.q_a_max_1 = self.plant_cfg["q_a_max_1"]
        self.q_a_max_2 = self.plant_cfg["q_a_max_2"]
        self.n_1 = self.plant_cfg["n_1"]
        self.k_I_1 = self.plant_cfg["k_I_1"]
        self.n_2 = self.plant_cfg["n_2"]
        self.k_I_2 = self.plant_cfg["k_I_2"]
        self.d_l = self.plant_cfg["d_l"]
        self.S_in = self.plant_cfg["S_in"]
        self.d_a_1 = 150
        self.d_a_2 = 150

        self.hyperparam_config = hyperparam_config

    def get_initial_state(self, batch_size, randomize=True):
        """
        Returns [batch_size, 5] tensor of:
        [Biomass X1, Biomass X2, Substrate S, Enzyme A1, Enzyme A2]
        """
        # Fetch nominal values from config or defaults
        x1_nom = self.plant_cfg["x10"]
        x2_nom = self.plant_cfg["x20"]
        s_nom  = self.plant_cfg["s0"]
        a1_nom = self.plant_cfg["a10"]
        a2_nom = self.plant_cfg["a20"]

        nominal = torch.tensor([x1_nom, x2_nom, s_nom, a1_nom, a2_nom], 
                               device=self.device, dtype=torch.float32)
        
        # Broadcast across batch dimension
        states = nominal.repeat(batch_size, 1)

        if randomize:
            # Randomization within ±5% boundaries
            rand_scale = 0.95 + 0.10 * torch.rand((batch_size, 5), device=self.device)
            states = states * rand_scale
            
        return states

    def get_y(self, state, t=None):
        """
        The tracking output variables are the individual biomass values 
        of the two strains to monitor community balancing: returns [batch_size, 2]
        """
        return state[:, 0:2] # Returns [X1, X2]

    def dynamics(self, X1, X2, S, A1, A2, u, t):
        """
        Calculates continuous derivative transformations for the co-cultivation system.
        Expects states split or extracted to maintain shape [batch_size, 1].
        """
        # u expected shape: [batch_size, 2] -> [I_1, I_2]
        I_1 = u[:, 0:1]
        I_2 = u[:, 1:2]

        # Optogenetically-modulated Monod Kinetics
        mu1 = ((self.mu_max1 * S) / (self.k_g_1 + S)) * ((self.f_c * A1) / (self.f_c * A1 + self.k_a_1))
        mu2 = ((self.mu_max2 * S) / (self.k_g_2 + S)) * ((self.f_c * A2) / (self.f_c * A2 + self.k_a_2))
        
        q_g_1 = self.Y_g_b1 * mu1
        q_g_2 = self.Y_g_b2 * mu2

        # Enzyme synthesis rates via light inputs
        q_a_1 = self.q_a_max_1 * (I_1**self.n_1 / (I_1**self.n_1 + self.k_I_1**self.n_1))
        q_a_2 = self.q_a_max_2 * (I_2**self.n_2 / (I_2**self.n_2 + self.k_I_2**self.n_2))

        # Chemostat system differential math
        dX1_dt = (mu1 - self.d_l) * X1
        dX2_dt = (mu2 - self.d_l) * X2
        dS_dt  = self.d_l * (self.S_in - S) - q_g_1 * X1 - q_g_2 * X2
        dA1_dt = q_a_1 - (self.d_a_1 + mu1) * A1
        dA2_dt = q_a_2 - (self.d_a_2 + mu2) * A2

        return dX1_dt, dX2_dt, dS_dt, dA1_dt, dA2_dt

    def step(self, state, u, t, dt=None):
        """
        Advances continuous batch states using explicit 4th-order Runge Kutta integration.
        """
        if dt is None:
            dt = self.dt

        # Unpack tensor along features dimension
        X1, X2 = state[:, 0:1], state[:, 1:2]
        S      = state[:, 2:3]
        A1, A2 = state[:, 3:4], state[:, 4:5]

        # k1
        dX1_1, dX2_1, dS_1, dA1_1, dA2_1 = self.dynamics(X1, X2, S, A1, A2, u, t)

        # k2
        dX1_2, dX2_2, dS_2, dA1_2, dA2_2 = self.dynamics(
            X1 + 0.5 * dt * dX1_1, X2 + 0.5 * dt * dX2_1, 
            S  + 0.5 * dt * dS_1,  A1 + 0.5 * dt * dA1_1, A2 + 0.5 * dt * dA2_1, 
            u, t + 0.5 * dt
        )

        # k3
        dX1_3, dX2_3, dS_3, dA1_3, dA2_3 = self.dynamics(
            X1 + 0.5 * dt * dX1_2, X2 + 0.5 * dt * dX2_2, 
            S  + 0.5 * dt * dS_2,  A1 + 0.5 * dt * dA1_2, A2 + 0.5 * dt * dA2_2, 
            u, t + 0.5 * dt
        )

        # k4
        dX1_4, dX2_4, dS_4, dA1_4, dA2_4 = self.dynamics(
            X1 + dt * dX1_3, 
            X2 + dt * dX2_3, 
            S  + dt * dS_3,  
            A1 + dt * dA1_3, 
            A2 + dt * dA2_3,  # <-- Change this from dA2_4 to dA2_3
            u, 
            t + dt
        )

        # Compute integrated states step
        X1_next = X1 + (dt / 6.0) * (dX1_1 + 2.0 * dX1_2 + 2.0 * dX1_3 + dX1_4)
        X2_next = X2 + (dt / 6.0) * (dX2_1 + 2.0 * dX2_2 + 2.0 * dX2_3 + dX2_4)
        S_next  = S  + (dt / 6.0) * (dS_1  + 2.0 * dS_2  + 2.0 * dS_3  + dS_4)
        A1_next = A1 + (dt / 6.0) * (dA1_1 + 2.0 * dA1_2 + 2.0 * dA1_3 + dA1_4)
        A2_next = A2 + (dt / 6.0) * (dA2_1 + 2.0 * dA2_2 + 2.0 * dA2_3 + dA2_4)

        # Re-pack and force non-negativity constraints via clamp
        state_next = torch.cat([X1_next, X2_next, S_next, A1_next, A2_next], dim=1)
        state_next = torch.clamp(state_next, min=0.0)

        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2", "s", "a1", "a2"],
                "labels": [r"$b_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$b_2 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$s / \mathrm{g}\,\mathrm{L}^{-1}$",r"$a_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$a_2 / \mathrm{g}\,\mathrm{L}^{-1}$"],
                "ylabel": [r"$b_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$b_2 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$s / \mathrm{g}\,\mathrm{L}^{-1}$",r"$a_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$a_2 / \mathrm{g}\,\mathrm{L}^{-1}$"]
            },
            {
                "cols": ["y1", "y2"],
                "labels": [r"$b_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$b_2 / \mathrm{g}\,\mathrm{L}^{-1}$"],
                "ylabel": [r"$b_1 / \mathrm{g}\,\mathrm{L}^{-1}$",r"$b_2 / \mathrm{g}\,\mathrm{L}^{-1}$"]
            },
            {
                "cols": ["u1", "u2"],
                "labels": [r"$I_1 / \mathrm{W}\,\mathrm{m}^{-2}$", r"$I_2 / \mathrm{W}\,\mathrm{m}^{-2}$"],
                "ylabel": [r"$I_1 / \mathrm{W}\,\mathrm{m}^{-2}$", r"$I_2 / \mathrm{W}\,\mathrm{m}^{-2}$"]
            }
        ]    

    def parse_state(self, state):
        def _val(x): return x.item() if torch.is_tensor(x) else x
        return {
            "biomass_strain_1": _val(state[0]),
            "biomass_strain_2": _val(state[1]),
            "substrate": _val(state[2]),
            "enzyme_1": _val(state[3]),
            "enzyme_2": _val(state[4])
        }
        
hyperparam_config_CoCultivationPlant = {
    "plant": {
        # Kinetic Parameters for Strain 1 & Strain 2
        "mu_max1": 0.982,       # Max growth rate strain 1 (1/h)
        "mu_max2": 0.982,       # Max growth rate strain 2 (1/h)
        "k_g_1": 2.964e-4,      # Substrate affinity constant strain 1
        "k_g_2": 2.964e-4,      # Substrate affinity constant strain 2
        "f_c": 1100.0,          # Conversion factor scaling enzyme concentration
        "k_a_1": 1.7,           # Activation constant for strain 1 growth
        "k_a_2": 0.182,         # Activation constant for strain 2 growth
        "Y_g_b1": 10.18,        # Yield coefficient factor for strain 1
        "Y_g_b2": 10.18,        # Yield coefficient factor for strain 2
        "q_a_max_1": 0.337,     # Max enzyme expression rate via light 1
        "q_a_max_2": 0.036,     # Max enzyme expression rate via light 2
        "n_1": 2.0,             # Hill coefficient for light input 1
        "k_I_1": 1.052,         # Light intensity constant for induction 1
        "n_2": 4.865,           # Hill coefficient for light input 2
        "k_I_2": 1.34,          # Light intensity constant for induction 2
        "d_l": 0.15,            # Dilution rate of the chemostat (1/h)
        "S_in": 200.0,          # Substrate concentration in the feed (g/L)
        "d_a_1": 0.15,           # Enzyme degradation rate 1 # unkown
        "d_a_2": 0.15,           # Enzyme degradation rate 2 #unknown
        
        # Signal generation center boundaries for Strain 1 (Channel 1)
        "u_1_D_center_min": 0.5,   # Adjust these values based on your light intensity needs
        "u_1_D_center_max": 2.0,   
        
        # Signal generation center boundaries for Strain 2 (Channel 2)
        "u_2_D_center_min": 0.5,   # Adjust these values based on your light intensity needs
        "u_2_D_center_max": 2.0,
        
        # Operational limits / bounds (Adjust boundaries based on your light units/caps)
        "u_1_hard_min": 0.0,
        "u_1_hard_max": 5.0,      # Max expected light intensity cap 
        
        "u_2_hard_min": 0.0,
        "u_2_hard_max": 5.0,      # Max expected light
        # Initial conditions (nominal state values)
        "x10": 0.005,           # Biomass X1 (g/L)
        "x20": 0.005,           # Biomass X2 (g/L)
        "s0": 1.0,              # Substrate S (g/L)
        "a10": 1.545e-2,        # Enzyme concentration A1
        "a20": 1.655e-3,        # Enzyme concentration A2

        # IO Dimensions: 2 controlled tracker variables (X1, X2), 2 actuators (I1, I2)
        "input_dim": 2,         # Tracker dimension (y)
        "output_dim": 2         # Actuator control dimension (u)
    },
    "signal": {
        "lambd": 4,
        "p": 0.5,
        "seq_len": 2001,
        "dt": 0.01               # Matching the dt=1 step time from your original code
    },
    "train": {
        "k_folds": 5,
        "epochs": 100,
        "batch_size": 1000,
        "lr": 1e-3,
        "device": "cuda",       # Automatically falls back to device selection patterns
        "delay_steps": 1,
        "loss_function": "MSELoss()", 
        "lr_decay_rate": 1,
        "min_correlation_threshold": -1.1
    },
    "mamba": {
        "expand": 32,
        "d_state": 16,
        "input_dim": 2,         # Matches tracking output [X1, X2]
        "output_dim": 2         # Matches structural light input [I1, I2]
    },
    "simulate": {
        "batch_size": 10,
        "seq_len": 2001,
    }
}