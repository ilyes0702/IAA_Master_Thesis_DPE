# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaInverseController(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        Redesigned Mamba Inverse Controller supporting arbitrary sliding window inputs.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector v_k. 
                       If not provided, it will be calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["mamba"]["input_dim"]   # Dimension of plant output y (e.g. 2)
        self.output_dim = hyperparam_config["mamba"]["output_dim"] # Dimension of plant control u (e.g. 2)
        self.d_state = hyperparam_config["mamba"]["d_state"]
        self.expand = hyperparam_config["mamba"]["expand"]
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["mamba"]["n_y"]
            n_u = hyperparam_config["mamba"]["n_u"]
            # v_k = [y_{k+1}, y_k ... y_{k-n_y}, u_{k-1} ... u_{k-n_u}]
            self.d_model = self.input_dim + (n_y + 1) * self.input_dim + n_u * self.output_dim
        print("d_model: ", self.d_model)
        print(f"🛠️ Initializing Mamba core with d_model (feature_dim) = {self.d_model}")
        
        # 3. Instantiate Mamba Core
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=4,
            expand=self.expand
        )
        
        # 4. Map latent features back to actuator control dimensions
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # Inference memory state buffers
        self.conv_state = None
        self.ssm_state = None

    def forward(self, v_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - v_seq: Tensor of shape [Batch, Seq_Len, d_model] (Contains stacked v_k sequences)
        
        Returns:
        - predicted_u: Tensor of shape [Batch, Seq_Len, output_dim] (Target control actions u_k)
        """
        # Pass sequence through Mamba S6 engine
        x = self.core(v_seq)  # Shape: [Batch, Seq_Len, d_model]
        return self.output_proj(x)  # Shape: [Batch, Seq_Len, output_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Allocates or resets zero-filled tracking memory tensors.
        Essential for stateful step-by-step rolling simulation.
        """
        d_inner = self.d_model * self.expand
        self.conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        self.ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)

    def step(self, v_k_single):
        """
        Closed-loop evaluation step. Seamlessly consumes current lookback vector 
        to output the next control instruction while updating recurring states.
        
        Parameters:
        - v_k_single: Tensor of shape [Batch, d_model] (or [Batch, 1, d_model])
        
        Returns:
        - u_out: Tensor of shape [Batch, output_dim] (Unprojected control action)
        """
        if self.conv_state is None or self.ssm_state is None:
            raise RuntimeError("Inference states are uninitialized. Please call reset_memory() first.")
            
        # Standardize input to 3D tensor layout required by mamba_ssm.step: [Batch, 1, d_model]
        if v_k_single.dim() == 2:
            x_3d = v_k_single.unsqueeze(1)
        else:
            x_3d = v_k_single
            
        # Recurrent state computation
        x_out_3d, self.conv_state, self.ssm_state = self.core.step(
            x_3d, self.conv_state, self.ssm_state
        )
        
        # Flatten back sequence step to 2D
        x_out = x_out_3d.squeeze(1)
        
        # Project to physical actuator outputs
        return self.output_proj(x_out) # Shape: [Batch, output_dim]

    # --- PROPERTIES FOR SYSTEM MODEL ANALYSIS ---
    @property
    def A(self):
        return self.core.A
        
    @property
    def B(self):
        return self.core.B

    @property
    def C(self):
        return self.core.C
    
    @property
    def D(self):
        return getattr(self.core, "extracted_D", getattr(self.core, "D", None))
    
    @property
    def mamba_dt(self):
        return getattr(self.core, "extracted_dt", getattr(self.core, "dt", None))


class MambaInverseController_marcia(nn.Module):
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