# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.sample.classes.BaseInverseController import BaseInverseController

# Define a Mamba-based controller class that inherits from the base controller interface
class MambaInverseController(BaseInverseController):
    """
    A neural network controller using the Mamba state-space model architecture.
    This controller processes sequential input data and produces control signals.
    """
    
    def __init__(self, d_model=32, d_state=16):
        """
        Initialize the Mamba controller with specified model and state dimensions.
        
        Args:
            d_model (int): Dimension of the model hidden state (default: 32)
            d_state (int): Dimension of the state-space state (default: 16)
        """
        super().__init__(input_dim=2, output_dim=1)
        
        # Project input from 2 dimensions to d_model dimensions
        # Input is expected to be 2-dimensional (e.g., reference and current values)
        self.input_proj = nn.Linear(2, d_model)
        
        # Mamba SSM block: efficient state-space model for sequence processing
        # d_conv=4: convolution dimension, expand=2: expansion factor for hidden layers
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        
        # Project output from d_model dimensions back to 1 dimension (scalar control output)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, y_seq):
        """
        Forward pass through the controller.
        
        Args:
            y_seq: Input sequence tensor of shape (..., 2)
            
        Returns:
            Control output tensor of shape (..., 1) with values between 0 and 1
        """
        # Project input to model dimension
        x = self.input_proj(y_seq)
        
        # Process through Mamba state-space model for sequence understanding
        x = self.mamba(x)
        
        # Project to output dimension and apply sigmoid activation for bounded output [0, 1]
        return torch.sigmoid(self.output_proj(x))