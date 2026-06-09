class BasePlant:
    def __init__(self, y_max, u_max):
        self.Y_MAX = y_max 
        self.U_MAX = u_max 
        
    def step(self, state, u, t, dt):
        """Must return (next_state, y_measured)"""
        raise NotImplementedError

    def get_initial_state(self):
        raise NotImplementedError

    def get_y(self, state, t): # Added t
        """Extracts the specific variable we want to control (e.g., mu)"""
        raise NotImplementedError

    def parse_state(self, state):
        """Returns a dict for CSV logging"""
        return {}

    def generate_random_u(self, t):
        """Returns a control signal for training"""
        raise NotImplementedError
        
    def reset_trajectory(self):
        """Randomizes internal training parameters (Amp, Freq)"""
        pass