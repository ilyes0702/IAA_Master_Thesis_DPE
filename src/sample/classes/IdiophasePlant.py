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
        Returns [batch_size, 4] tensor of [x1 (Biomass), x2 (Substrate), x3 (Precursor), x4 (Penicillin)].
        Initialized with positive uniform distributions for robust inverse control convergence.
        """
        x1_init = torch.rand((batch_size, 1), device=self.device) * 5.0 + 10.0   # Biomass mass
        x2_init = torch.rand((batch_size, 1), device=self.device) * 100.0 + 50.0 # Substrate mass
        x3_init = torch.rand((batch_size, 1), device=self.device) * 1000.0 + 500.0 # Precursor mass
        x4_init = torch.rand((batch_size, 1), device=self.device) * 5.0 + 1.0    # Penicillin mass
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
        dx2dt = - (1.0 / self.p1) * mu * x1 - (1.0 / self.p5) * self.mu_Pen * x1 - self.m_S_functional(x1) + self.p2 * u1
        dx3dt = - (1.0 / self.q) * self.mu_Pen * x1 + self.p6 * u2
        dx4dt = self.mu_Pen * x1 - self.p7 * x4

        return dx1dt, dx2dt, dx3dt, dx4dt

    def m_S_functional(self, x1):
        """Maintenance energy loss functional term (m_S * x1)."""
        # ms parameter value from Tab 1 is 23
        return 23.0 * x1

    def step(self, state, u, t, dt):
        """
        MIMO Runge-Kutta 4th Order numerical integration execution block.
        """
        x1, x2, x3, x4 = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
        
        # k1
        dx1_1, dx2_1, dx3_1, dx4_1 = self.dynamics(x1, x2, x3, x4, u, t)
        
        # k2
        dx1_2, dx2_2, dx3_2, dx4_2 = self.dynamics(
            x1 + 0.5 * dt * dx1_1, x2 + 0.5 * dt * dx2_1, 
            x3 + 0.5 * dt * dx3_1, x4 + 0.5 * dt * dx4_1, u, t + 0.5 * dt
        )
        
        # k3
        dx1_3, dx2_3, dx3_3, dx4_3 = self.dynamics(
            x1 + 0.5 * dt * dx1_2, x2 + 0.5 * dt * dx2_2, 
            x3 + 0.5 * dt * dx3_2, x4 + 0.5 * dt * dx4_2, u, t + 0.5 * dt
        )
        
        # k4
        dx1_4, dx2_4, dx3_4, dx4_4 = self.dynamics(
            x1 + dt * dx1_3, x2 + dt * dx2_3, 
            x3 + dt * dx3_3, x4 + dt * dx4_3, u, t + dt
        )
        
        x1_next = x1 + (dt / 6.0) * (dx1_1 + 2 * dx1_2 + 2 * dx1_3 + dx1_4)
        x2_next = x2 + (dt / 6.0) * (dx2_1 + 2 * dx2_2 + 2 * dx2_3 + dx2_4)
        x3_next = x3 + (dt / 6.0) * (dx3_1 + 2 * dx3_2 + 2 * dx3_3 + dx3_4)
        x4_next = x4 + (dt / 6.0) * (dx4_1 + 2 * dx4_2 + 2 * dx4_3 + dx4_4)
        
        state_next = torch.cat([x1_next, x2_next, x3_next, x4_next], dim=1)
        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2"],
                "labels": ["Biomass Mass (x1)", "Substrate Mass (x2)"],
                "title": "Biomass & Substrate Evolution",
                "ylabel": "Mass [g or mg]"
            },
            {
                "cols": ["x3", "x4"],
                "labels": ["Precursor Mass (x3)", "Penicillin Mass (x4)"],
                "title": "Precursor & Product Yield",
                "ylabel": "Mass [g or mg]"
            },
            {
                "cols": ["y1", "r1"],
                "labels": ["Actual μ", "Target μ*"],
                "title": "MIMO Tracking Output 1: Growth Rate",
                "ylabel": "Growth Rate [1/h]"
            },
            {
                "cols": ["y2", "r2"],
                "labels": ["Actual c3", "Target c3*"],
                "title": "MIMO Tracking Output 2: Precursor Conc.",
                "ylabel": "Concentration [g Pre/L]"
            },
            {
                "cols": ["u1", "u2"],
                "labels": ["Glucose Feed (u1)", "Precursor Feed (u2)"],
                "title": "MIMO Control Inputs Space",
                "ylabel": "Flow Rates [l/h]"
            }
        ]

    def parse_state(self, state):
        return {
            "biomass_mass": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate_mass": state[1].item() if torch.is_tensor(state) else state[1],
            "precursor_mass": state[2].item() if torch.is_tensor(state) else state[2],
            "penicillin_mass": state[3].item() if torch.is_tensor(state) else state[3]
        }