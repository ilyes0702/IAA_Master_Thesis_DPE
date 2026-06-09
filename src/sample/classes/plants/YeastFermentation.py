import torch

import json
import os
import torch

class FedBatchYeastPlant:
    def __init__(self, hyperparam_config):
        """
        Initializes the plant by dynamically extracting parameters from the configuration structure.
        
        Args:
            hyperparam_config (dict or str): Configuration dictionary, or path to its JSON representation.
        """
        # Automatically load from path string if user passes a file pointer instead of a dict
        if isinstance(hyperparam_config, str):
            if not os.path.exists(hyperparam_config):
                raise FileNotFoundError(f"Configuration file not found at: {hyperparam_config}")
            with open(hyperparam_config, "r") as f:
                config = json.load(f)
        else:
            config = hyperparam_config

        # Set up processing device allocation safety string mapping
        device_str = config["train"]["device"]
        if device_str == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # Time integration parameters
        self.dt = config["signal"]["dt"]

        # Parse Stoichiometric Yield Coefficients (Table 1 values mapped from dict)
        self.a1 = torch.tensor(config["plant"]["a1"], device=self.device)
        self.b1 = torch.tensor(config["plant"]["b1"], device=self.device)
        self.c1 = torch.tensor(config["plant"]["c1"], device=self.device)
        self.b2 = torch.tensor(config["plant"]["b2"], device=self.device)
        self.c2 = torch.tensor(config["plant"]["c2"], device=self.device)
        self.d2 = torch.tensor(config["plant"]["d2"], device=self.device)
        self.a3 = torch.tensor(config["plant"]["a3"], device=self.device)
        self.b3 = torch.tensor(config["plant"]["b3"], device=self.device)
        self.c3 = torch.tensor(config["plant"]["c3"], device=self.device)

        # Parse Kinetic Parameters (Table 2 values mapped from dict)
        self.ks = torch.tensor(config["plant"]["ks"], device=self.device)
        self.ko = torch.tensor(config["plant"]["ko"], device=self.device)
        self.kp = torch.tensor(config["plant"]["kp"], device=self.device)
        self.Ks_val = torch.tensor(config["plant"]["Ks_val"], device=self.device)
        self.Ko = torch.tensor(config["plant"]["Ko"], device=self.device)
        self.Kp = torch.tensor(config["plant"]["Kp"], device=self.device)

        # Parse Process Operating Conditions (Table 3 values mapped from dict)
        self.Sin = torch.tensor(config["plant"]["Sin"], device=self.device)
        self.Pin = torch.tensor(config["plant"]["Pin"], device=self.device)
        self.O2_star = torch.tensor(config["plant"]["O2_star"], device=self.device)
        self.kla = torch.tensor(config["plant"]["kla"], device=self.device)
        
        # Save operational configurations for volume references
        self.V_init_val = config["plant"]["V_init"]
        self.V_max_val = config["plant"]["V_max"]

    def get_initial_state(self, batch_size):
        """
        Returns a [batch_size, 5] tensor of initial states:
        [Biomass (X), Substrate (S), Ethanol (P), Oxygen (O2), Volume (V)]
        Initial values use operating baseline benchmarks defined in config file.
        """
        X_init = torch.ones((batch_size, 1), device=self.device) * 1.5
        S_init = torch.ones((batch_size, 1), device=self.device) * 0.023
        P_init = torch.ones((batch_size, 1), device=self.device) * 10.0
        O2_init = torch.ones((batch_size, 1), device=self.device) * 0.039
        V_init = torch.ones((batch_size, 1), device=self.device) * self.V_init_val
        
        return torch.cat([X_init, S_init, P_init, O2_init, V_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates and returns the primary observable output: Ethanol concentration (P).
        """
        P = state[:, 2:3]
        return P 

    def dynamics(self, X, S, P, O2, V, F):
        # Numerical protection step to ensure sub-integrations maintain absolute non-negativity
        S = torch.clamp(S, min=0.0)
        O2 = torch.clamp(O2, min=0.0)
        P = torch.clamp(P, min=0.0)
        X = torch.clamp(X, min=0.0)

        # Epsilon to guarantee complete prevention of divide-by-zero operations
        eps = 1e-8

        # Kinetic Uptake Rates using parameter fields populated by your configuration
        rs = self.ks * (S / (S + self.Ks_val + eps))
        ro = self.ko * (O2 / (O2 + self.Ko + eps))
        rp = self.kp * (P / (P + self.Kp + eps))

        # Metabolic Switch Mechanics (Sonnleitner Overflow Bottleneck model)
        r1 = torch.min(rs, ro / (self.a1 + eps))
        r2 = torch.max(torch.zeros_like(rs), rs - (ro / (self.a1 + eps)))
        
        remaining_oxygen = torch.max(torch.zeros_like(ro), ro - (self.a1 * rs))
        r3 = torch.max(torch.zeros_like(rs), torch.min(rp, remaining_oxygen / (self.a3 + eps)))

        # Combined biomass production rate
        biomass_growth = (self.b1 * r1 + self.b2 * r2 + self.b3 * r3) * X

        # Fed-batch volumetric mass balances 
        V_safe = torch.clamp(V, min=1e-5)
        
        dXdt = biomass_growth - (F / V_safe) * X
        dSdt = -(r1 + r2) * X + (F / V_safe) * (self.Sin - S)
        dPdt = (self.d2 * r2 - r3) * X + (F / V_safe) * (self.Pin - P)
        dO2dt = self.kla * (self.O2_star - O2) - (self.a1 * r1 + self.a3 * r3) * X - (F / V_safe) * O2
        dVdt = F

        return dXdt, dSdt, dPdt, dO2dt, dVdt

    def step(self, state, u, t, dt):
        X, S, P, O2, V = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4], state[:, 4:5]
        F = u

        # Balanced Runge-Kutta 4th Order continuous space integration steps
        dX1, dS1, dP1, dO21, dV1 = self.dynamics(X, S, P, O2, V, F)
        dX2, dS2, dP2, dO22, dV2 = self.dynamics(X + 0.5*dt*dX1, S + 0.5*dt*dS1, P + 0.5*dt*dP1, O2 + 0.5*dt*dO21, V + 0.5*dt*dV1, F)
        dX3, dS3, dP3, dO23, dV3 = self.dynamics(X + 0.5*dt*dX2, S + 0.5*dt*dS2, P + 0.5*dt*dP2, O2 + 0.5*dt*dO22, V + 0.5*dt*dV2, F)
        dX4, dS4, dP4, dO24, dV4 = self.dynamics(X + dt*dX3, S + dt*dS3, P + dt*dP3, O2 + dt*dO23, V + dt*dV3, F)

        # Recompile intermediate vector calculations
        X_next = X + (dt/6.0) * (dX1 + 2*dX2 + 2*dX3 + dX4)
        S_next = S + (dt/6.0) * (dS1 + 2*dS2 + 2*dS3 + dS4)
        P_next = P + (dt/6.0) * (dP1 + 2*dP2 + 2*dP3 + dP4)
        O2_next = O2 + (dt/6.0) * (dO21 + 2*dO22 + 2*dO23 + dO24)
        V_next = V + (dt/6.0) * (dV1 + 2*dV2 + 2*dV3 + dV4)

        # Apply robust physical reality bounds directly to state tracking vectors
        X_next = torch.clamp(X_next, min=1e-4)
        S_next = torch.clamp(S_next, min=0.0)
        P_next = torch.clamp(P_next, min=0.0)
        O2_next = torch.clamp(O2_next, min=1e-6, max=self.O2_star.item() * 2)

        state_next = torch.cat([X_next, S_next, P_next, O2_next, V_next], dim=1)
        return state_next, self.get_y(state_next)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2", "x3"],
                "labels": ["Biomass (X)", "Substrate (S)", "Ethanol (P)"],
                "title": "Bioreactor Species Concentrations",
                "ylabel": "Concentration [g/L]"
            },
            {
                "cols": ["x4"],
                "labels": ["Dissolved Oxygen (O2)"],
                "title": "Dissolved Oxygen Profile",
                "ylabel": "DO Concentration [g/L]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual Ethanol (P)", "Target P_ref"],
                "title": "Overflow Metabolite Feedback Tracking",
                "ylabel": "Ethanol [g/L]"
            },
            {
                "cols": ["u"],
                "labels": ["Feed Rate (F)"],
                "title": "Substrate Feed Manipulation Profile",
                "ylabel": "Flow Rate [L/h]"
            }
        ]

    def parse_state(self, state):
        return {
            "biomass": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate": state[1].item() if torch.is_tensor(state) else state[1],
            "ethanol": state[2].item() if torch.is_tensor(state) else state[2],
            "oxygen": state[3].item() if torch.is_tensor(state) else state[3],
            "volume": state[4].item() if torch.is_tensor(state) else state[4]
        }