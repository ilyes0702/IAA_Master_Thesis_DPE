import torch

class SimpleLinearPlant:
    def __init__(self, hyperparam_config):
        self.device = hyperparam_config["train"]["device"]
        self.dt = hyperparam_config["signal"]["dt"]
        
        cfg = hyperparam_config["plant"]
        # System Matrices: dx/dt = A*x + B*u
        self.a11 = torch.tensor(cfg["a11"], device=self.device)
        self.a12 = torch.tensor(cfg["a12"], device=self.device)
        self.a21 = torch.tensor(cfg["a21"], device=self.device)
        self.a22 = torch.tensor(cfg["a22"], device=self.device)
        
        self.b1 = torch.tensor(cfg["b1"], device=self.device)
        self.b2 = torch.tensor(cfg["b2"], device=self.device)
        
        # Output Matrix: y = C*x
        self.c1 = torch.tensor(cfg["c1"], device=self.device)
        self.c2 = torch.tensor(cfg["c2"], device=self.device)

    def get_initial_state(self, batch_size):
        """
        Returns [batch_size, 2] tensor of [State_1, State_2].
        Initializes values across a stable continuous window.
        """
        x1_init = torch.rand((batch_size, 1), device=self.device) * 4.0 - 2.0  # -2.0 to +2.0
        x2_init = torch.rand((batch_size, 1), device=self.device) * 4.0 - 2.0  # -2.0 to +2.0
        return torch.cat([x1_init, x2_init], dim=1)

    def get_y(self, state, t=None):
        """
        Calculates output metric matrix equation: y = C * x
        """
        x1 = state[:, 0:1]
        x2 = state[:, 1:2]
        y = self.c1 * x1 + self.c2 * x2
        return y 

    def dynamics(self, x1, x2, u, t=None):
        """
        Calculates continuous derivative transformations for a standard linear system.
        """
        # dx1/dt = a11*x1 + a12*x2 + b1*u
        dx1dt = self.a11 * x1 + self.a12 * x2 + self.b1 * u
        # dx2/dt = a21*x1 + a22*x2 + b2*u
        dx2dt = self.a21 * x1 + self.a22 * x2 + self.b2 * u
        return dx1dt, dx2dt

    def step(self, state, u, t, dt):
        """
        Standard Runge-Kutta 4th Order numerical integration execution block.
        """
        x1, x2 = state[:, 0:1], state[:, 1:2]
        
        # k1
        dx1_1, dx2_1 = self.dynamics(x1, x2, u)
        # k2
        dx1_2, dx2_2 = self.dynamics(x1 + 0.5 * dt * dx1_1, x2 + 0.5 * dt * dx2_1, u)
        # k3
        dx1_3, dx2_3 = self.dynamics(x1 + 0.5 * dt * dx1_2, x2 + 0.5 * dt * dx2_2, u)
        # k4
        dx1_4, dx2_4 = self.dynamics(x1 + dt * dx1_3, x2 + dt * dx2_3, u)
        
        x1_next = x1 + (dt / 6.0) * (dx1_1 + 2 * dx1_2 + 2 * dx1_3 + dx1_4)
        x2_next = x2 + (dt / 6.0) * (dx2_1 + 2 * dx2_2 + 2 * dx2_3 + dx2_4)
        
        state_next = torch.cat([x1_next, x2_next], dim=1)
        return state_next, self.get_y(state_next)

    def get_plot_config(self):
        return [
            {
                "cols": ["x1", "x2"],
                "labels": ["State 1 (X1)", "State 2 (X2)"],
                "title": "Linear Plant State Evolution",
                "ylabel": "State Values"
            },
            {
                "cols": ["y", "r"],
                "labels": ["Actual Output y", "Target y_ref"],
                "title": "System Tracking Analysis",
                "ylabel": "Output Magnitude"
            },
            {
                "cols": ["u"],
                "labels": ["Control Input (u)"],
                "title": "Control Input Vector Profile",
                "ylabel": "Input Drive Force"
            }
        ]

    def parse_state(self, state):
        return {
            "state_1": state[0].item() if torch.is_tensor(state) else state[0],
            "state_2": state[1].item() if torch.is_tensor(state) else state[1]
        }