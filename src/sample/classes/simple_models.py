import numpy as np
from src.sample.classes.BasePlant import BasePlant

class SimpleLinearPlant(BasePlant):
    def __init__(self):
        # We'll use 1.0 as max for simplicity
        super().__init__(y_max=1.0, u_max=1.0)
        self.tau = 2.0  # Time constant (seconds/hours). Higher = slower response.
        self.gain = 0.8 # The output y will reach 80% of the input u
        self.ref_value = 0.5 # Default test target

    def get_initial_state(self):
        # State is just the current value of y
        return np.array([0.0])

    def get_y(self, state):
        return state[0]

    def step(self, state, u, t, dt=0.1):
        y = state[0]
        
        # Simple First-Order ODE: dy/dt = (Gain * u - y) / tau
        dy = (self.gain * u - y) / self.tau
        
        new_y = y + dy * dt
        return np.array([new_y]), new_y

    def parse_state(self, state):
        return {"process_value": state[0]}

    def get_plot_config(self):
        return [
            {"cols": ["y", "r"], "labels": ["Output", "Target"], "title": "Linear Tracking", "ylabel": "Value"},
            {"cols": ["u"], "labels": ["Control Signal"], "title": "Control Action", "ylabel": "U"}
        ]

    def generate_random_u(self, t):
        # Use a simple square wave to see if Mamba learns the "lag"
        return 0.7 if (t // 5) % 2 == 0 else 0.2