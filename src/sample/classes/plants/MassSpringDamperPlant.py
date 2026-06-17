import torch

class MassSpringDamperPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]

        # Physical Parameters from Config
        self.m = torch.tensor(hyperparam_config["plant"]["m"], device=self.device)   # Mass
        self.d = torch.tensor(hyperparam_config["plant"]["d"], device=self.device)   # Damping constant
        self.k = torch.tensor(hyperparam_config["plant"]["k"], device=self.device)   # Spring constant

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor of [Position (x), Velocity (v)].
        Initializes with random physical states to ensure robust learning.
        """
        # Example ranges: position between -1.0 and 1.0, velocity between -0.5 and 0.5
        x_init = (torch.rand((batch_size, 1), device=self.device) * 2.0) - 1.0 
        v_init = (torch.rand((batch_size, 1), device=self.device) * 1.0) - 0.5 
        return torch.cat([x_init, v_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates and returns the observable plant output.
        Typically, position x is observed/tracked.
        """
        x = state[:, 0:1]
        return x 

    def dynamics(self, x, v, u):
        """
        Calculates continuous derivative transformations based on the matrix:
        dxdt = v
        dvdt = -(k/m)*x - (d/m)*v + (1/m)*F
        where u represents the external force F(t).
        """
        dxdt = v
        dvdt = -(self.k / self.m) * x - (self.d / self.m) * v + (1.0 / self.m) * u
        return dxdt, dvdt

    def step(self, state, u, t, dt):
        """
        Standard Runge-Kutta 4th Order numerical integration execution block.
        """
        x, v = state[:, 0:1], state[:, 1:2]
        
        # k1
        dx1, dv1 = self.dynamics(x, v, u)
        # k2
        dx2, dv2 = self.dynamics(x + 0.5*dt*dx1, v + 0.5*dt*dv1, u)
        # k3
        dx3, dv3 = self.dynamics(x + 0.5*dt*dx2, v + 0.5*dt*dv2, u)
        # k4
        dx4, dv4 = self.dynamics(x + dt*dx3, v + dt*dv3, u)
        
        x_next = x + (dt/6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        v_next = v + (dt/6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4) # Fixed variable tracking mapping
        
        state_next = torch.cat([x_next, v_next], dim=1)
        return state_next, self.get_y(state_next)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2"],
                "labels": ["Position (x)", "Velocity (v)"],
                "title": "Mechanical System State Evolution",
                "ylabel": "States [m, m/s]"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual Position x", "Target x_ref"],
                "title": "Position Tracking Inverse Learning",
                "ylabel": "Position [m]"
            },
            {
                "cols": ["u"],
                "labels": ["Input Force (F)"],
                "title": "Control Input (Force)",
                "ylabel": "Force [N]"
            }
        ]

    def parse_state(self, state):
        return {
            "position": state[0].item() if torch.is_tensor(state) else state[0],
            "velocity": state[1].item() if torch.is_tensor(state) else state[1]
        }