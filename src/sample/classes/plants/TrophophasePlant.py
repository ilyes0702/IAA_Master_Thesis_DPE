import torch
import torchode

class TrophophasePlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]
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
        
        #return 150.0 + 2.0 * ramp1 - 2.0 * ramp2
        return 150

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor of [Biomass Mass (x1), Substrate Mass (x2)].
        Initial values should be customized based on your initial culture mass.
        """
        # Example initialization for masses (in grams and milligrams respectively)
        x1_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x10"], device=self.device)  
        x2_init = torch.full((batch_size, 1), self.hyperparam_config["plant"]["x20"], device=self.device)
        return torch.cat([x1_init, x2_init], dim=1)

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

    def step(self, state, u1, t, dt):
        """
        4th Order Runge-Kutta integration accounting for explicitly time-dependent dynamics.
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        
        # k1 at t
        dx1_1, ds1_1 = self.dynamics(x1, x2, u1, t)
        
        # k2 at t + 0.5*dt
        dx1_2, ds1_2 = self.dynamics(x1 + 0.5*dt*dx1_1, x2 + 0.5*dt*ds1_1, u1, t + 0.5*dt)
        
        # k3 at t + 0.5*dt
        dx1_3, ds1_3 = self.dynamics(x1 + 0.5*dt*dx1_2, x2 + 0.5*dt*ds1_2, u1, t + 0.5*dt)
        
        # k4 at t + dt
        dx1_4, ds1_4 = self.dynamics(x1 + dt*dx1_3, x2 + dt*ds1_3, u1, t + dt)
        
        x1_next = x1 + (dt / 6.0) * (dx1_1 + 2.0*dx1_2 + 2.0*dx1_3 + dx1_4)
        x2_next = x2 + (dt / 6.0) * (ds1_1 + 2.0*ds1_2 + 2.0*ds1_3 + ds1_4)
        
        state_next = torch.cat([x1_next, x2_next], dim=1)
        
        # Return next state and the corresponding tracking output at time t + dt
        return state_next, self.get_y(state_next, t + dt)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1"],
                "labels": ["Biomass concentration (x1)"],
                "title": "Biomass Accumulation",
                "ylabel": "Concentration [g/l]"
            },
            {
                "cols": ["x2"],
                "labels": ["Substrate concentration (x2)"],
                "title": "Substrate Available",
                "ylabel": "Concentration [mg/l]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual μ", "Target μ*"],
                "title": "Growth Rate Trophophase Control",
                "ylabel": "Growth Rate [1/h]"
            },
            {
                "cols": ["u1"],
                "labels": ["Glucose Feed Rate (u1)"],
                "title": "Control Input Profile",
                "ylabel": "Feed Rate [l/h]"
            }
        ]

    def parse_state(self, state):
        return {
            "biomass_concentration": state[0].item() if torch.is_tensor(state) else state[0],
            "substrate_concentration": state[1].item() if torch.is_tensor(state) else state[1]
        }