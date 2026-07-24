# Import necessary libraries for PyTorch neural network functionality
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams


import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaSysIDModel(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        Mamba-based forward system identification model (NARX-style).
        
        Given a regressor window of past outputs and past+current inputs,
        predicts the plant output at the current time step.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector w_k.
                       If not provided, it will be calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["plant"]["input_dim"]   # Dimension of plant output y (e.g. 2)
        self.output_dim = hyperparam_config["plant"]["output_dim"] # Dimension of plant control u (e.g. 2)
        self.d_state = hyperparam_config["mamba"]["d_state"]
        self.expand = hyperparam_config["mamba"]["expand"]
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["train"]["n_y"]
            n_u = hyperparam_config["train"]["n_u"]
            # w_k = [u_k, u_{k-1} ... u_{k-n_u}, y_{k-1} ... y_{k-n_y}]
            print(self.input_dim)
            print(self.output_dim)
            self.d_model = (n_u + 1) * self.output_dim + n_y * self.input_dim
        print("d_model: ", self.d_model)
        print(f"🛠️ Initializing Mamba core with d_model (feature_dim) = {self.d_model}")
        
        # 3. Instantiate Mamba Core
        self.core = Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=4,
            expand=self.expand
        )
        
        # 4. Map latent features to predicted plant output dimension
        self.output_proj = nn.Linear(self.d_model, self.input_dim)
        
        # Inference memory state buffers
        self.conv_state = None
        self.ssm_state = None

    def forward(self, w_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - w_seq: Tensor of shape [Batch, Seq_Len, d_model] (Contains stacked w_k regressor sequences)
        
        Returns:
        - predicted_y: Tensor of shape [Batch, Seq_Len, input_dim] (Target plant outputs y_k)
        """
        x = self.core(w_seq)  # Shape: [Batch, Seq_Len, d_model]
        return self.output_proj(x)  # Shape: [Batch, Seq_Len, input_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Allocates or resets zero-filled tracking memory tensors.
        Essential for stateful step-by-step rolling simulation.
        """
        d_inner = self.d_model * self.expand
        self.conv_state = torch.zeros(batch_size, d_inner, self.core.d_conv, device=device)
        self.ssm_state = torch.zeros(batch_size, d_inner, self.core.d_state, device=device)

    def step(self, w_k_single):
        """
        Closed-loop / rolling-simulation evaluation step. Consumes the current
        regressor vector (past y's + current & past u's) to predict y_k while
        updating recurrent states.
        
        Parameters:
        - w_k_single: Tensor of shape [Batch, d_model] (or [Batch, 1, d_model])
        
        Returns:
        - y_out: Tensor of shape [Batch, input_dim] (Predicted plant output)
        """
        if self.conv_state is None or self.ssm_state is None:
            raise RuntimeError("Inference states are uninitialized. Please call reset_memory() first.")
            
        if w_k_single.dim() == 2:
            x_3d = w_k_single.unsqueeze(1)
        else:
            x_3d = w_k_single
            
        x_out_3d, self.conv_state, self.ssm_state = self.core.step(
            x_3d, self.conv_state, self.ssm_state
        )
        
        x_out = x_out_3d.squeeze(1)
        return self.output_proj(x_out)  # Shape: [Batch, input_dim]

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