class BasePlant:
    def __init__(self, y_max, u_max):
        self.Y_MAX = y_max  # The normalization factor for the output
        self.U_MAX = u_max  # The normalization factor for the control
        
    def step(self, state, u, t, dt):
        """Must return (next_state, y_measured)"""
        raise NotImplementedError

    def get_initial_state(self):
        raise NotImplementedError

    def get_y(self, state):
        """Extracts the specific variable we want to control (e.g., mu)"""
        raise NotImplementedError