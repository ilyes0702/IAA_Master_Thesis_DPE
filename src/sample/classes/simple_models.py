import numpy as np
from src.sample.classes.BasePlant import BasePlant

class SimpleLinearPlant(BasePlant):
    def __init__(self, seed = "none"):
        # Gain=0.8 means if u=1, y will eventually reach 0.8
        super().__init__(y_max=1.0, u_max=1.0)
        self.rng = np.random.default_rng(seed)
        self.tau = 2.0  # Time constant (seconds/hours)
        self.gain = 0.8 
        self.ref_value = 0.5 
        
        # Initialize trajectory using the local RNG
        self.reset_trajectory()

    def reset_trajectory(self):
        """Randomizes sine-wave properties using local RNG"""
        # Replace np.random with self.rng
        self.curr_amp = self.rng.uniform(0.2, 0.45)
        self.curr_freq = self.rng.uniform(0.05, 0.15)
        self.curr_phase = self.rng.uniform(0, 2 * np.pi)

    def get_initial_state(self):
        # State is simply the current value of the output y
        return np.array([0.0])

    def get_y(self, state, t=None):
        # The output is just the state itself
        return state[0]

    def step(self, state, u, t, dt=0.1):
        y = state[0]
        
        # First-order ODE: dy/dt = (Gain * u - y) / tau
        dy = (self.gain * u - y) / self.tau
        
        new_y = y + dy * dt
        return np.array([new_y]), new_y

    def reset_trajectory(self):
        """Randomizes the sine-wave properties for each training epoch"""
        self.curr_amp = np.random.uniform(0.2, 0.45)
        self.curr_freq = np.random.uniform(0.05, 0.15)
        self.curr_phase = np.random.uniform(0, 2 * np.pi)

    def generate_random_u(self, t):
        """Sine-wave approximation with rectangles"""
        # 1. Base sine wave
        sine_val = self.curr_amp * np.sin(2 * np.pi * self.curr_freq * t + self.curr_phase) + 0.5
        
        # 2. Rectangle logic: Hold the value for 2-second blocks
        # This creates the 'staircase' effect you wanted
        hold_interval = 2.0
        t_snapped = (t // hold_interval) * hold_interval
        
        # Sample the sine wave at the start of the block
        u_rect = self.curr_amp * np.sin(2 * np.pi * self.curr_freq * t_snapped + self.curr_phase) + 0.5
        
        return np.clip(u_rect, 0, self.U_MAX)

    def get_plot_config(self):
        return [
            {
                "cols": ["y", "r"], # The column names in your CSV/DataFrame
                "labels": ["Growth", "Target"], # What to show in the legend
                "title": "Growth Rate", # Header of the subplot
                "ylabel": "1/h" # Unit for the Y-axis
            },
            {
                "cols": ["u"], 
                "labels": ["Feed"], 
                "title": "Control Action", 
                "ylabel": "L/h"
            }
        ]

    def parse_state(self, state):
        return {"process_value": state[0]}