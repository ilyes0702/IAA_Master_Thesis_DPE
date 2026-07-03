# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.sample.classes.controllers.BaseInverseController import BaseInverseController
from mamba_ssm.utils.generation import InferenceParams

# Define a Mamba-based controller class that inherits from the base controller interface
class MambaInverseController_old(BaseInverseController):
    def __init__(self, hyperparam_config):
        super().__init__(input_dim=2, output_dim=1)
        self.d_model = hyperparam_config["mamba"]["d_model"]
        self.d_state = hyperparam_config["mamba"]["d_state"]

        self.mamba = Mamba(d_model=self.d_model, 
                           d_state=self.d_state, 
                           d_conv=4, 
                           expand=2,
                           layer_idx=0)
        
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

class MambaInverseControllerSISO(nn.Module):
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




class MambaInverseController(nn.Module):
    def __init__(self, hyperparam_config):
        super().__init__()
        
        # Read MIMO dimensions from configuration dynamically
        self.input_dim = hyperparam_config["mamba"]["input_dim"]   # Number of plant outputs to track (e.g., y1, y2)
        self.output_dim = hyperparam_config["mamba"]["output_dim"] # Number of plant control inputs (e.g., u1, u2)
        
        #self.d_model = hyperparam_config["mamba"]["d_model"]
        self.d_state = hyperparam_config["mamba"]["d_state"]

        # Because we concatenate y_t and y_next, the total feature dimension 
        # entering the projection layer is always 2 * input_dim.
        #self.input_proj = nn.Linear(self.input_dim * 2, self.d_model)
        self.expand = hyperparam_config["mamba"]["expand"]  # Default expansion factor if not specified
        
        self.core = Mamba(
            d_model=self.input_dim * 2,  # The Mamba core will receive the concatenated features directly
            d_state=self.d_state, 
            d_conv=4, 
            expand=self.expand
            )
                
        # Maps the latent space back to the multi-variable control inputs space
        self.output_proj = nn.Linear(self.core.d_model, self.output_dim)

    def forward(self, y_t, y_next):
        """
        Args:
            y_t (Tensor): Current plant outputs, shape [Batch, Seq_len, input_dim]
            y_next (Tensor): Target/next plant outputs, shape [Batch, Seq_len, input_dim]

        Returns:
            Tensor: Control input vectors, shape [Batch, Seq_len, output_dim]
        """
        # Concatenate y_t and y_next along the feature dimension
        x = torch.cat([y_t, y_next], dim=-1)  # Shape: [Batch, Seq_len, input_dim * 2]

        #x = self.input_proj(x)  # Shape: [Batch, Seq_len, d_model]
        x = self.core(x)  # Shape: [Batch, Seq_len, d_model]
        return self.output_proj(x)  # Shape: [Batch, Seq_len, output_dim]
    
    @property
    def A_bar(self):
        """Access discretized A_bar from the Mamba core. Shape: [B, L, d_inner, d_state]"""
        return getattr(self.core, "A_bar", None)

    @property
    def B_bar(self):
        """Access discretized B_bar from the Mamba core. Shape: [B, L, d_inner, d_state]"""
        return getattr(self.core, "B_bar", None)
    
    @property
    def B(self):
        """Access B from the Mamba core."""
        return self.core.B

    @property
    def C(self):
        """Access C from the Mamba core."""
        return self.core.C
    
    @property
    def A(self):
        """Access A from the Mamba core."""
        return self.core.A
    
    @property
    def D(self):
        """Access D from the Mamba core."""
        return self.core.extracted_D

    @property
    def mamba_dt(self):
        """Access dt from the Mamba core."""
        return self.core.extracted_dt
    



import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaInverseController_stateful(nn.Module):
    def __init__(self, hyperparam_config):
        super().__init__()
        
        # Extract MIMO configuration dimensions dynamically
        self.input_dim = hyperparam_config["mamba"]["input_dim"]   # e.g., 2 for MIMO
        self.output_dim = hyperparam_config["mamba"]["output_dim"] # e.g., 2
        self.d_state = hyperparam_config["mamba"]["d_state"]       # e.g., 16
        self.expand = hyperparam_config["mamba"]["expand"]         # e.g., 2
        
        # d_model represents the raw concatenated features (y_t and y_next)
        self.d_model = self.input_dim * 2  
        
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=4,
            expand=self.expand
        )
        
        # Maps from Mamba's core dimension back to your multi-variable control inputs
        self.output_proj = nn.Linear(self.d_model, self.output_dim)

    def forward(self, y_t, y_next):
        """
        Sequence/Batch training forward pass.
        y_t/y_next shapes: [Batch, Seq_len, input_dim]
        """
        x = torch.cat([y_t, y_next], dim=-1)  # Shape: [Batch, Seq_len, d_model]
        x = self.core(x)                      # Shape: [Batch, Seq_len, d_model]
        return self.output_proj(x)            # Shape: [Batch, Seq_len, output_dim]

    def allocate_inference_states(self, batch_size=1, device="cuda"):
        """
        Allocates zero-filled tracking tensors matching mamba_ssm dimensions.
        Internal state size tracks (d_model * expand).
        """
        d_inner = self.d_model * self.expand
        conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)
        return conv_state, ssm_state

    def step(self, y_t_single, y_next_single, conv_state, ssm_state):
        """
        Stateful decoding step supporting any arbitrary input dimensions (MIMO safe).
        y_t_single/y_next_single shapes: [Batch, input_dim]
        """
        # ─── MIMO FIX ────────────────────────────────────────────────────────
        # Concatenate along the feature dimension to preserve the batch structure.
        # Shape: [Batch, input_dim] + [Batch, input_dim] -> [Batch, d_model]
        x = torch.cat([y_t_single, y_next_single], dim=-1) 
        
        # 3D Sequence layout adapter required by mamba_ssm.step: [Batch, 1, d_model]
        x_3d = x.unsqueeze(1)
        
        # Feed the 3D raw feature token step to Mamba core
        x_out_3d, conv_state, ssm_state = self.core.step(x_3d, conv_state, ssm_state)
        
        # Remove sequence dimension: [Batch, 1, d_model] -> [Batch, d_model]
        x_out = x_out_3d.squeeze(1)
        
        # Linearly map back to plant actuator dimensions
        u_out = self.output_proj(x_out) # Shape: [Batch, output_dim]
        
        return u_out, conv_state, ssm_state

    def reset_hooks_storage(self):
        self.captured_V_sigma = []

    # --- PROPERTIES FOR METADATA LOGGING ---
    @property
    def A_bar(self):
        """Access discretized A_bar from the Mamba core. Shape: [B, L, d_inner, d_state]"""
        return getattr(self.core, "A_bar", None)

    @property
    def B_bar(self):
        """Access discretized B_bar from the Mamba core. Shape: [B, L, d_inner, d_state]"""
        return getattr(self.core, "B_bar", None)
    
    @property
    def B(self):
        return self.core.B

    @property
    def C(self):
        return self.core.C
    
    @property
    def A(self):
        return self.core.A
    
    @property
    def D(self):
        return getattr(self.core, "extracted_D", getattr(self.core, "D", None))
    
    @property
    def mamba_dt(self):
        return getattr(self.core, "extracted_dt", getattr(self.core, "dt", None))
import torch
import torch.nn as nn

class MambaInverseController_stateful_SISO(nn.Module):
    def __init__(self, hyperparam_config):
        super().__init__()
        
        # Extract MIMO configuration dimensions dynamically
        self.input_dim = hyperparam_config["mamba"]["input_dim"]   # e.g., 1 for your working example
        self.output_dim = hyperparam_config["mamba"]["output_dim"] # e.g., 1
        self.d_state = hyperparam_config["mamba"]["d_state"]       # e.g., 16
        self.expand = hyperparam_config["mamba"]["expand"]         # e.g., 2
        
        # 🌟 CRITICAL: Without input_proj, d_model IS exactly the raw concatenated features
        self.d_model = self.input_dim * 2  # e.g., 1 * 2 = 2
        
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=4,
            expand=self.expand
        )
        
        # Maps from Mamba's core dimension back to your multi-variable control inputs
        self.output_proj = nn.Linear(self.d_model, self.output_dim)

    def forward(self, y_t, y_next):
        """
        Sequence/Batch training forward pass.
        y_t/y_next shapes: [Batch, Seq_len, input_dim]
        """
        x = torch.cat([y_t, y_next], dim=-1)  # Shape: [Batch, Seq_len, d_model]
        x = self.core(x)                      # Shape: [Batch, Seq_len, d_model]
        return self.output_proj(x)            # Shape: [Batch, Seq_len, output_dim]

    def allocate_inference_states(self, batch_size=1, device="cuda"):
        """
        Allocates zero-filled tracking tensors matching mamba_ssm dimensions.
        Internal state size tracks (d_model * expand).
        """
        d_inner = self.d_model * self.expand
        conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)
        return conv_state, ssm_state

    def step(self, y_t_single, y_next_single, conv_state, ssm_state):
        """
        Stateful decoding step using structural layout logic from your reference code.
        """
        # Maintain batch structure when flattening multi-channel slices
        # Shapes transform from [Batch, input_dim] -> [Batch * input_dim]
        y_t_flat = y_t_single.reshape(-1)
        y_next_flat = y_next_single.reshape(-1)

        # Re-stack into a single batch structure [Batch, d_model]
        # For a batch of 10 and 1-channel inputs: stacks two [10] vectors into [10, 2]
        x = torch.stack([y_t_flat, y_next_flat], dim=-1) 
        
        # 3D Sequence layout adapter required by mamba_ssm.step: [Batch, 1, d_model]
        x_3d = x.unsqueeze(1)
        
        # Feed the 3D raw feature token step to Mamba
        x_out_3d, conv_state, ssm_state = self.core.step(x_3d, conv_state, ssm_state)
        
        # Remove sequence dimension: [Batch, 1, d_model] -> [Batch, d_model]
        x_out = x_out_3d.squeeze(1)
        
        # Linearly map back to plant actuator dimensions
        u_out = self.output_proj(x_out) # Shape: [Batch, output_dim]
        
        return u_out, conv_state, ssm_state

    # Add this inside your MambaInverseController_stateful class
    def reset_hooks_storage(self):
        self.captured_V_sigma = []

    # --- PROPERTIES FOR METADATA LOGGING ---
    @property
    def A_bar(self):
        """Access discretized A_bar from the Mamba core. Shape: [B, L, d_inner, d_state]"""
        return getattr(self.core, "A_bar", None)

    @property
    def B_bar(self):
        """Access discretized B_bar from the Mamba core. Shape: [B, L, d_inner, d_state]"""
        return getattr(self.core, "B_bar", None)
    
    @property
    def B(self):
        return self.core.B

    @property
    def C(self):
        return self.core.C
    
    @property
    def A(self):
        return self.core.A
    
    @property
    def D(self):
        return getattr(self.core, "extracted_D", getattr(self.core, "D", None))
    
    @property
    def mamba_dt(self):
        return getattr(self.core, "extracted_dt", getattr(self.core, "dt", None))
    


import torch
import torch.nn as nn

class MambaInverseController_stateful_w_der(nn.Module):
    def __init__(self, hyperparam_config):
        super().__init__()
        
        # Extract MIMO configuration dimensions dynamically
        self.input_dim = hyperparam_config["mamba"]["input_dim"]   # e.g., 1
        self.output_dim = hyperparam_config["mamba"]["output_dim"] # e.g., 1
        self.d_state = hyperparam_config["mamba"]["d_state"]       # e.g., 16
        self.expand = hyperparam_config["mamba"]["expand"]         # e.g., 2
        
        # 🌟 UPDATED: d_model now holds 4 components per input channel:
        # [y_t, y_next, y_dot, y_ddot] -> input_dim * 4
        self.d_model = self.input_dim * 2  # e.g., 1 * 4 = 4
        
        # Using standard Mamba core configuration from mamba_ssm
        from mamba_ssm import Mamba
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=4,
            expand=self.expand
        )
        
        # Maps from Mamba's core dimension back to your multi-variable control inputs
        self.output_proj = nn.Linear(self.d_model, self.output_dim)

    def forward(self, y_t, y_next):
        x = torch.cat([y_t, y_next], dim=-1)  # Shape: [Batch, Seq_len, d_model]
        x = self.core(x)                      
        return self.output_proj(x)

    def allocate_inference_states(self, batch_size=1, device="cuda"):
        """Allocates zero-filled tracking tensors matching mamba_ssm dimensions."""
        d_inner = self.d_model * self.expand
        conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)
        return conv_state, ssm_state

    def step(self, y_t_single, y_next_single, conv_state, ssm_state):
        y_t_flat = y_t_single.reshape(-1)
        y_next_flat = y_next_single.reshape(-1)

        x = torch.stack([y_t_flat, y_next_flat], dim=-1) 
        x_3d = x.unsqueeze(1)
        x_out_3d, conv_state, ssm_state = self.core.step(x_3d, conv_state, ssm_state)
        x_out = x_out_3d.squeeze(1)
        
        # 🌟 UPDATED INFERENCE STEP: Splitting predictions
        preds_out = self.output_proj(x_out) # Shape: [Batch, total_predictions_dim]
        
        u_out = preds_out[:, :self.output_dim]
        y_dot_pred = preds_out[:, self.output_dim : self.output_dim + self.input_dim]
        y_ddot_pred = preds_out[:, self.output_dim + self.input_dim:]
        
        return u_out, y_dot_pred, y_ddot_pred, conv_state, ssm_state

    def reset_hooks_storage(self):
        self.captured_V_sigma = []

    # --- PROPERTIES FOR METADATA LOGGING ---
    @property
    def A_bar(self): return getattr(self.core, "A_bar", None)
    @property
    def B_bar(self): return getattr(self.core, "B_bar", None)
    @property
    def B(self): return self.core.B
    @property
    def C(self): return self.core.C
    @property
    def A(self): return self.core.A
    @property
    def D(self): return getattr(self.core, "extracted_D", getattr(self.core, "D", None))
    @property
    def mamba_dt(self): return getattr(self.core, "extracted_dt", getattr(self.core, "dt", None))