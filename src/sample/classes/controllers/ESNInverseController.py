import numpy as np
from reservoirpy.nodes import Reservoir, Ridge

class ESNInverseController:
    def __init__(self, hyperparam_config):
        """
        MIMO Echo State Network Inverse Controller using ReservoirPy.
        """
        # Read MIMO dimensions from configuration
        self.input_dim = hyperparam_config["plant"]["input_dim"]     # e.g., number of plant outputs
        self.output_dim = hyperparam_config["plant"]["output_dim"]   # e.g., number of plant control inputs
        
        # Hyperparameters specific to ESN
        self.units = 200
        self.lr = 0.3
        self.sr = 1.0
        self.ridge = 1e-6    # Regularization coefficient
        
        # Initialize ReservoirPy Nodes
        # The input dimension to the reservoir will automatically adapt to (input_dim * 2) 
        # when data is first passed or during connection.
        self.reservoir = Reservoir(units=self.units, lr=self.lr, sr=self.sr)
        self.readout = Ridge(ridge=self.ridge)
        
        # Link them to build the ESN
        self.model = self.reservoir >> self.readout
    
    def state_dict(self):
        """Mock method to prevent PyTorch wrapper crashes during K-Fold initialization."""
        return {}

    def load_state_dict(self, state_dict):
        """Mock method to handle K-Fold resets seamlessly."""
        # Only reset if the nodes have been initialized with data
        if hasattr(self.reservoir, "state"):
            self.reservoir.reset()
        if hasattr(self.readout, "state"):
            self.readout.reset()
        return self

    def fit(self, X_train_list, Y_train_list):
        """
        Train the linear readout layer via Ridge regression using lists of sequences.
        
        Args:
            X_train_list (list of np.ndarray): Pre-concatenated tracking data arrays.
            Y_train_list (list of np.ndarray): Target control inputs.
        """
        # Clear residual states safely; skip if model is uninitialized
        try:
            self.model.reset()
        except AttributeError:
            pass  # Node states do not exist yet; no reset needed
        
        # ReservoirPy fits the readout instantly via offline linear regression
        self.model = self.model.fit(X_train_list, Y_train_list)

    def forward(self, x):
        """
        Run the ESN forward pass on a single sequence sample.
        
        Args:
            x (np.ndarray): Pre-concatenated state-target vector, shape [Seq_len, input_dim * 2]
            
        Returns:
            np.ndarray: Predicted control inputs, shape [Seq_len, output_dim]
        """
        try:
            self.model.reset()
        except AttributeError:
            pass
            
        # Run the sequence through the reservoir and readout
        return self.model.run(x)