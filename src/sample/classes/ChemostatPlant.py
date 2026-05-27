import torch

class ChemostatPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Biological Parameters from Config
        self.mu_max = torch.tensor(hyperparam_config["plant"]["mu-max"], device=self.device)
        self.Ks = torch.tensor(hyperparam_config["plant"]["Ks"], device=self.device)
        self.Y = torch.tensor(hyperparam_config["plant"]["Y"], device=self.device)
        self.sR = torch.tensor(hyperparam_config["plant"]["sR"], device=self.device)
        self.Ki = torch.tensor(hyperparam_config["plant"]["Ki"], device=self.device)

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor of [Biomass (x), Substrate (s)].
        Initializes with random biological values to ensure robust learning.
        """
        x_init = torch.rand((batch_size, 1), device=self.device) * 0.5 + 0.1 # 0.1 to 0.6
        s_init = torch.rand((batch_size, 1), device=self.device) * 0.5 + 0.1 # 0.1 to 0.6
        return torch.cat([x_init, s_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates and returns Growth Rate (mu) as the observable plant output tracker.
        """
        s = state[:, 1:2]
        mu = (self.mu_max * s) / (self.Ks + s)
        return mu 

    def dynamics(self, x, s, u):
        """
        Calculates continuous derivative transformations for standard Chemostat vessels.
        """
        mu = (self.mu_max * s) / (self.Ks + s)
        dxdt = mu * x - u * x
        dsdt = u * (self.sR - s) - (mu * x / self.Y)
        return dxdt, dsdt

    def step(self, state, u, t, dt):
        """
        Standard Runge-Kutta 4th Order numerical integration execution block.
        """
        x, s = state[:, 0:1], state[:, 1:2]
        
        # k1
        dx1, ds1 = self.dynamics(x, s, u)
        # k2
        dx2, ds2 = self.dynamics(x + 0.5*dt*dx1, s + 0.5*dt*ds1, u)
        # k3
        dx3, ds3 = self.dynamics(x + 0.5*dt*dx2, s + 0.5*dt*ds2, u)
        # k4
        dx4, ds4 = self.dynamics(x + dt*dx3, s + dt*ds3, u)
        
        x_next = x + (dt/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        s_next = s + (dt/6.0) * (ds1 + 2*ds2 + 2*ds3 + ds4)
        
        state_next = torch.cat([x_next, s_next], dim=1)
        return state_next, self.get_y(state_next)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2"],
                "labels": ["Biomass (X)", "Substrate (S)"],
                "title": "Chemostat State Evolution",
                "ylabel": "Concentration [g/L]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual μ", "Target μ_ref"],
                "title": "Growth Rate Inverse Learning",
                "ylabel": "Growth Rate [1/h]"
            },
            {
                "cols": ["u"],
                "labels": ["Dilution Rate (D)"],
                "title": "Control Input (D)",
                "ylabel": "Dilution Rate [1/h]"
            }
        ]

    def parse_state(self, state):
        return {
            "biomass": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate": state[1].item() if torch.is_tensor(state) else state[1]
        }