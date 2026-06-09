# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from sample.classes.controllers.BaseInverseController import BaseInverseController

# Define an LSTM-based controller class that inherits from the base controller interface
class LSTMInverseController(BaseInverseController):
    """
    A neural network controller using LSTM (Long Short-Term Memory) architecture.
    This controller processes sequential input data and produces control signals.
    """
    
    def __init__(self, hidden_dim=32, num_layers=2):
        """
        Initialize the LSTM controller with specified hidden and layer dimensions.
        
        Args:
            hidden_dim (int): Dimension of the LSTM hidden state (default: 32)
            num_layers (int): Number of stacked LSTM layers (default: 2)
        """
        super().__init__(input_dim=2, output_dim=1)
        
        # Project input from 2 dimensions to hidden_dim dimensions
        # Input is expected to be 2-dimensional (e.g., reference and current values)
        self.input_proj = nn.Linear(2, hidden_dim)
        
        # LSTM block: captures temporal dependencies in sequences
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                           num_layers=num_layers, batch_first=True)
        
        # Project output from hidden_dim dimensions back to 1 dimension (scalar control output)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, y_seq):
        """
        Forward pass through the controller.
        
        Args:
            y_seq: Input sequence tensor of shape (..., 2)
            
        Returns:
            Control output tensor of shape (..., 1) with values between 0 and 1
        """
        # Project input to hidden dimension
        x = self.input_proj(y_seq)
        
        # Process through LSTM for sequence understanding
        x, (h_n, c_n) = self.lstm(x)
        
        # Project to output dimension and apply sigmoid activation for bounded output [0, 1]
        return torch.sigmoid(self.output_proj(x))
