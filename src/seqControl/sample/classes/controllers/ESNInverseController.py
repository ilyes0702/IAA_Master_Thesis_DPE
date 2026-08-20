import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
import torch
class ESNInverseController:
    def __init__(self, hyperparam_config):
        """
        MIMO Echo State Network Inverse Controller using ReservoirPy.
        """
        # Read MIMO dimensions from configuration
        self.input_dim = hyperparam_config["plant"]["input_dim"]     # e.g., number of plant outputs
        self.output_dim = hyperparam_config["plant"]["output_dim"]   # e.g., number of plant control inputs
        
        # Hyperparameters specific to ESN
        self.units = hyperparam_config["esn"]["units"]  # Number of reservoir units
        self.lr = hyperparam_config["esn"]["lr"]
        self.sr = hyperparam_config["esn"]["sr"]
        self.ridge = hyperparam_config["esn"]["ridge"]    # Regularization coefficient
        
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
    
    def save_parameters(self, path):
        """Save the trained ESN model to disk."""

        print(self.reservoir.bias)





class ESNInverseController_torch:
    def __init__(self, hyperparam_config):
        """
        MIMO Echo State Network Inverse Controller using ReservoirPy.
        Compatible with PyTorch step-by-step simulation loops.
        """
        self.input_dim = hyperparam_config["plant"]["input_dim"]
        self.output_dim = hyperparam_config["plant"]["output_dim"]
        
        self.units = hyperparam_config["esn"]["units"]
        self.lr = hyperparam_config["esn"]["lr"]
        self.sr = hyperparam_config["esn"]["sr"]
        self.ridge = hyperparam_config["esn"]["ridge"]
        
        self.reservoir = Reservoir(units=self.units, lr=self.lr, sr=self.sr)
        self.readout = Ridge(ridge=self.ridge)
        self.model = self.reservoir >> self.readout

    # -------------------------------------------------------------------------
    # PyTorch Interface Compatibility Methods
    # -------------------------------------------------------------------------

    def eval(self):
        """Mock PyTorch eval() method to prevent AttributeError."""
        return self

    def reset_memory(self, batch_size=1, device=None):
        """
        Resets internal reservoir state between simulation runs.
        Matches the simulation loop check: hasattr(model, 'reset_memory')
        """
        try:
            self.model.reset()
        except AttributeError:
            pass  # Node states do not exist yet

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a single stateful time-step inference.
        
        Args:
            x (torch.Tensor): Input feature tensor [batch_size, feature_dim]
        Returns:
            torch.Tensor: Predicted output actions tensor [batch_size, output_dim]
        """
        # 1. Convert PyTorch tensor (GPU/CPU) to NumPy array
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        
        # 2. Step forward in ReservoirPy (uses .call for step-by-step evaluation)
        u_np = self.model.call(x_np)
        
        # 3. Convert result back to PyTorch Tensor on original device
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
        return torch.tensor(u_np, dtype=torch.float32, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback caller matching PyTorch forward pass syntax."""
        return self.step(x)
    def save_parameters(self, path):
            """Save the trained ESN model to disk."""
    
            print(self.reservoir.bias)
    # -------------------------------------------------------------------------
    # Training & Offline Batch Execution
    # -------------------------------------------------------------------------

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        self.reset_memory()
        return self

    def fit(self, X_train_list, Y_train_list):
        """Train the linear readout layer via Ridge regression."""
        self.reset_memory()
        self.model = self.model.fit(X_train_list, Y_train_list)

    def forward(self, x):
        """Run batch sequence inference over an entire trajectory array."""
        self.reset_memory()
        return self.model.run(x)