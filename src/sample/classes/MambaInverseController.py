# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.sample.classes.BaseInverseController import BaseInverseController
from mamba_ssm.utils.generation import InferenceParams

# Define a Mamba-based controller class that inherits from the base controller interface
class MambaInverseController_old(BaseInverseController):
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





class MambaInverseController_seq2seq(BaseInverseController):
    def __init__(self, hyperparam_config):  # Fixed typo: __init__
        super().__init__(input_dim=2, output_dim=1)
        # Standardize variable naming
        self.d_model = hyperparam_config["mamba"]["d_model"]
        self.d_state = hyperparam_config["mamba"]["d_state"]

        self.input_proj = nn.Linear(2, self.d_model)
        self.mamba = Mamba(
            d_model=self.d_model, 
            d_state=self.d_state, 
            d_conv=4, 
            expand=2,
            layer_idx=0
        )
        self.output_proj = nn.Linear(self.d_model, 1)
        self.memory = None

    def reset_memory(self, batch_size, device):
        self.memory = InferenceParams(
            max_seqlen=2000,
            max_batch_size=batch_size
        )

    def forward(self, y_seq, use_memory=False):
        # y_seq expected shape: (batch_size, seq_len, 2)
        x = self.input_proj(y_seq)

        if use_memory and self.memory is not None:
            x = self.mamba(x, inference_params=self.memory)
        else:
            x = self.mamba(x)

        x = self.output_proj(x)
        return x


import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaInverseController(nn.Module):
    def __init__(self, hyperparam_config):
        super().__init__()
        
        self.d_model = hyperparam_config["mamba"]["d_model"]
        self.d_state = hyperparam_config["mamba"]["d_state"]

        # The linear layer still takes a feature size of 2 
        # because we will combine the 2 inputs inside the forward pass
        self.input_proj = nn.Linear(2, self.d_model)
        
        self.core = Mamba(
            d_model=self.d_model, 
            d_state=self.d_state, 
            d_conv=4, 
            expand=2
        )
        self.output_proj = nn.Linear(self.d_model, 1)

    def forward(self, y_t, y_next):
        # Concatenate features into [Batch, Seq_len, 2]
        x = torch.cat([y_t, y_next], dim=-1)
        
        x = self.input_proj(x)
        x = self.core(x)  # Dispatches sequence context using official custom CUDA kernels
        return self.output_proj(x)