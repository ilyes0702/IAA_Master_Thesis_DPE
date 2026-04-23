import numpy as np
from src.sample.classes.BasePlant import BasePlant

# Move these into the class or keep them as constants
U_MAX = 1.0 
MU_MAX = 0.1

class FermentationProcess(BasePlant):
    def __init__(self):
        # Initialize the BasePlant with normalization constants
        super().__init__(y_max=MU_MAX, u_max=U_MAX)
        
        self.mu_max = MU_MAX
        self.Ks = 0.05    
        self.p1 = 0.47    
        self.p2 = 200.0   
        self.ms = 0.023   
        self.ref_value = 0.08  # Target growth rate for control reference

    def get_initial_state(self):
        # Generic function to tell the simulator where to start
        return np.array([1.0, 5e-3])

    def get_y(self, state):
        # Extract 'mu' from the current state for the controller
        # Note: In your current step, mu is calculated dynamically. 
        # For the generic method, we re-calculate it or store it.
        x1, x2 = state
        # We need a time 't' for V, or assume a default for the measurement
        V = 150.0 # Simplified for measurement extraction
        return (self.mu_max * x2) / (self.Ks * V + x2)

    def step(self, state, u1, t, dt=0.1):
        x1, x2 = state
        V = self.get_V(t)
        mu = (self.mu_max * x2) / (self.Ks * V + x2)

        dx1 = mu * x1
        dx2 = -(1/self.p1) * mu * x1 - self.ms * x1 + self.p2 * u1

        state_new = state + np.array([float(dx1), float(dx2)]) * dt
        state_new = np.maximum(state_new, 1e-6)

        return state_new, mu

    def get_V(self, t):
        if t < 5: return 150.0
        if 5 <= t < 15: return 150.0 + 2 * (t - 5)
        return 170.0

    def parse_state(self, state):
        # Tells the CSV saver how to label the columns
        return {
            "biomass_x1": state[0],
            "substrate_x2": state[1]
        }

    def get_plot_config(self):
        # Tells the generic plotter what signals to group together
        return [
            {"cols": ["y", "r"], "labels": ["Growth rate", "Target growth rate"], "title": "Growth Rate", "ylabel": "Growth rate (1/h)"},
            {"cols": ["u"], "labels": ["Glucose feed rate"],  "title": "Glucose Feed Rate", "ylabel": "Glucose feed rate (1/h)"},
            {"cols": ["biomass_x1"], "labels": ["Biomass concentration"], "title": "Biomass", "ylabel": "Biomass concentration (g/L)"},
            {"cols": ["substrate_x2"], "labels": ["Substrate concentration"], "title": "Substrate", "ylabel": "Substrate concentration (g/L)"}
        ]
    
    def generate_random_u(self, t):
        """Standardized way for the plant to provide a training signal"""
        # Move your sine/rect logic here
        u_sine = 0.5 * np.sin(0.5 * t) + 0.5
        return np.clip(u_sine, 0, self.U_MAX)