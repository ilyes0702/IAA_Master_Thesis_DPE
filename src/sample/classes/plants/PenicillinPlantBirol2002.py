import torch
import math

class PenicillinPlantBirol2002:
    def __init__(self, hyperparam_config):
        """
        Custom Penicillin Fermentation Plant based directly on:
        Birol, G., Ündey, C., & Çinar, A. (2002). "A modular simulation package 
        for fed-batch fermentation: penicillin production."
        """
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # --- Kinetic Parameters (From Table 2 of the paper) ---
        self.mu_x = torch.tensor(hyperparam_config["plant"].get("mu_x", 0.092), device=self.device)     # Max specific growth rate (1/h)
        self.K_x = torch.tensor(hyperparam_config["plant"].get("K_x", 0.15), device=self.device)        # Contois saturation constant (g/l)
        
        self.mu_p = torch.tensor(hyperparam_config["plant"].get("mu_p", 0.005), device=self.device)     # Specific rate of pen production (1/h)
        self.K_p = torch.tensor(hyperparam_config["plant"].get("K_p", 0.0002), device=self.device)      # Substrate inhibition constant (g/l)
        self.K_I = torch.tensor(hyperparam_config["plant"].get("K_I", 0.10), device=self.device)        # Inhibition constant for product formation (g/l)
        self.p_pow = torch.tensor(hyperparam_config["plant"].get("p_pow", 3.0), device=self.device)     # Constant 'p' power exponent
        self.K_h = torch.tensor(hyperparam_config["plant"].get("K_h", 0.04), device=self.device)        # Penicillin hydrolysis rate constant (1/h)
        
        # --- Yield Constants (From Table 2) ---
        self.Y_xs = torch.tensor(hyperparam_config["plant"].get("Y_xs", 0.45), device=self.device)      # Yield: g biomass / g glucose
        self.Y_ps = torch.tensor(hyperparam_config["plant"].get("Y_ps", 0.90), device=self.device)      # Yield: g penicillin / g glucose
        self.m_x = torch.tensor(hyperparam_config["plant"].get("m_x", 0.014), device=self.device)       # Maintenance coefficient on substrate (1/h)
        
        # --- Feed Parameters (From Table 2) ---
        self.s_f = torch.tensor(hyperparam_config["plant"].get("s_f", 600.0), device=self.device)       # Feed substrate concentration (g/l)
        
        # Heuristic Constant Evaporative Loss at nominal 25°C (Section 2.8)
        self.F_loss_const = torch.tensor(hyperparam_config["plant"].get("F_loss", 2.5e-4), device=self.device) # (l/h)

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 4] tensor of:
        [X (Biomass conc g/L), S (Substrate conc g/L), P (Penicillin conc g/L), V (Volume L)]
        Initial conditions adapted from Table 2 nominal operational starts.
        """
        # Incorporates slight uniform variance for robust inverse tracking verification
        X_init = torch.rand((batch_size, 1), device=self.device) * 2.0 + 4.0      # Nominal X around 4-6 g/L entering production phase
        S_init = torch.rand((batch_size, 1), device=self.device) * 5.0 + 10.0     # Nominal S around 10-15 g/L
        P_init = torch.zeros((batch_size, 1), device=self.device)                  # Yield begins at 0 g/L
        V_init = torch.rand((batch_size, 1), device=self.device) * 10.0 + 100.0   # Nominal initial volume 100L
        
        return torch.cat([X_init, S_init, P_init, V_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates and returns the 2-dimensional MIMO tracking target output vector:
        y = [y1, y2] = [Specific Growth Rate (mu), Penicillin Concentration (P)]
        """
        X = state[:, 0:1]
        S = state[:, 1:2]
        P = state[:, 2:3]

        # Contois Kinetics (Eq. 2) simplified assuming oxygen saturation is maintained (C_L limitation term -> 1.0)
        mu = (self.mu_x * S) / (self.K_x * X + S)
        
        return torch.cat([mu, P], dim=1)

    def dynamics(self, state, u, t=None):
        """
        Calculates continuous derivative state transformations matching the paper's differential equations.
        state: [Batch, 4] -> [X, S, P, V]
        u: [Batch, 1]     -> [F (Glucose Feed Flow Rate in L/h)]
        """
        X = state[:, 0:1]
        S = state[:, 1:2]
        P = state[:, 2:3]
        V = state[:, 3:4]
        
        F = u[:, 0:1] # Control input: Feed rate
        
        # 1. Volumetric loss and net volume change (Eq. 14)
        # For simplicity in specific tracking control loops, ignoring transient acid/base flows
        dVdt = F - self.F_loss_const 

        # 2. Specific growth rate (μ) via Contois kinetics (Eq. 2)
        mu = (self.mu_x * S) / (self.K_x * X + S)

        # 3. Specific penicillin production rate (μ_pp) via Substrate Inhibition kinetics (Eq. 10)
        # Term involving C_L drops out under constant saturation design 
        mu_pp = self.mu_p * (S / (self.K_p + S + (S**2 / self.K_I)))

        # 4. State Differential Equations (Eq. 1, 11, 9)
        # Note: The paper writes concentration derivatives accounting for dilution (- (State/V)*dVdt)
        dXdt = mu * X - (X / V) * dVdt
        dSdt = - (mu / self.Y_xs) * X - (mu_pp / self.Y_ps) * X - self.m_x * X + (F * self.s_f) / V - (S / V) * dVdt
        dPdt = mu_pp * X - self.K_h * P - (P / V) * dVdt

        return torch.cat([dXdt, dSdt, dPdt, dVdt], dim=1)

    def step(self, state, u, t=None, dt=None):
        """
        MIMO Runge-Kutta 4th Order numerical integration execution block.
        Ensures strict concentrations stay above zero for numeric stability.
        """
        # If dt is passed explicitly by the utility script, use it. 
        # Otherwise, fall back to the internal self.dt configuration.
        integration_dt = dt if dt is not None else self.dt
        
        # RK4 Integration steps
        k1 = self.dynamics(state, u, t)
        k2 = self.dynamics(state + 0.5 * integration_dt * k1, u, t)
        k3 = self.dynamics(state + 0.5 * integration_dt * k2, u, t)
        k4 = self.dynamics(state + integration_dt * k3, u, t)
        
        state_next = state + (integration_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Physical boundary clamp (Concentrations & volumes cannot be negative)
        state_next = torch.clamp(state_next, min=1e-6)
        
        return state_next, self.get_y(state_next)

    def parse_state(self, state):
        return {
            "biomass_conc": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate_conc": state[1].item() if torch.is_tensor(state) else state[1],
            "penicillin_conc": state[2].item() if torch.is_tensor(state) else state[2],
            "volume": state[3].item() if torch.is_tensor(state) else state[3]
        }

    def get_plot_config(self):
        return [
            {
                "cols": ["X", "S"],
                "labels": ["Biomass Conc (X)", "Substrate Conc (S)"],
                "title": "Biomass & Substrate Evolution Profile",
                "ylabel": "Concentration [g/L]"
            },
            {
                "cols": ["P"],
                "labels": ["Penicillin Conc (P)"],
                "title": "Penicillin Product Yield Profile",
                "ylabel": "Concentration [g/L]"
            },
            {
                "cols": ["V"],
                "labels": ["Reactor Volume (V)"],
                "title": "Culture Volume Dilution Profile",
                "ylabel": "Volume [L]"
            },
            {
                "cols": ["y1", "r1"],
                "labels": ["Actual Specific Growth μ", "Target μ*"],
                "title": "Tracking Output 1: Kinetic Growth Control",
                "ylabel": "Specific Growth [1/h]"
            }
        ]