# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.sample.classes.BaseInverseController import BaseInverseController
from mamba_ssm.utils.generation import InferenceParams

# Define a Mamba-based controller class that inherits from the base controller interface
class MambaInverseController(BaseInverseController):
    def __init__(self, hyperparam_config):
        super().__init__(input_dim=2, output_dim=1)
        self.d_model = hyperparam_config["mamba"]["d_model"]
        self.d_state = hyperparam_config["mamba"]["d_state"]

        self.input_proj = nn.Linear(2, self.d_model)
        self.mamba = Mamba(d_model=self.d_model, 
                           d_state=self.d_state, 
                           d_conv=4, 
                           expand=2,
                           layer_idx=0)
        self.output_proj = nn.Linear(self.d_model, 1)
        
        # New: Placeholder for inference memory
        self.memory = None

    def reset_memory(self, batch_size, device):
        """Initializes a fresh memory object for a new simulation."""
        self.memory = InferenceParams(
            max_seqlen=2000, # Set this higher than your max simulation steps
            max_batch_size=batch_size
        )

    def forward(self, y_seq, use_memory=False):
        # 1. Project input
        x = self.input_proj(y_seq)
        
        # 2. Process through Mamba
        if use_memory and self.memory is not None:
            # This uses the efficient 'step' kernel that updates 
            # the state instead of re-running the whole convolution
            x = self.mamba(x, inference_params=self.memory)
        else:
            # Standard training mode (parallel)
            x = self.mamba(x)
        
        # 3. Project to output
        x = self.output_proj(x)
        return x
