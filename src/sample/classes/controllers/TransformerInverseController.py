import torch
import torch.nn as nn

class TransformerInverseController(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        Transformer Inverse Controller supporting sequential inputs and step-by-step inference.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector v_k. 
                       If not provided, calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["plant"]["input_dim"]   # Dimension of plant output y
        self.output_dim = hyperparam_config["plant"]["output_dim"] # Dimension of plant control u
        
        # Transformer-specific hyperparams with sensible fallbacks
        trans_cfg = hyperparam_config["transformer"]
        self.nhead = trans_cfg["nhead"]
        self.num_layers = trans_cfg["num_layers"]
        self.dim_feedforward = trans_cfg["dim_feedforward"]
        self.max_seq_len = trans_cfg["max_seq_len"] # Maximum horizon for positional embedding
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["train"]["n_y"]
            n_u = hyperparam_config["train"]["n_u"]
            self.d_model = n_u * self.input_dim + (n_y + 2) * self.output_dim
            
        # Ensure d_model is divisible by nhead for MultiheadAttention
        if self.d_model % self.nhead != 0:
            # Adjust nhead dynamically or throw an informative error
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead}). "
                f"Consider setting nhead={self._find_valid_nhead(self.d_model)} in your config."
            )
            
        print(f"🛠️ Initializing Transformer core with d_model = {self.d_model}, nhead = {self.nhead}")
        
        # 3. Instantiate Learnable Positional Encodings
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        nn.init.normal_(self.pos_encoder, std=0.02)
        
        # 4. Instantiate Transformer Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True
        )
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 5. Map latent features back to actuator control dimensions
        self.output_proj = nn.Linear(self.d_model, self.input_dim)
        
        # Inference rolling sequence buffer
        self.seq_buffer = None

    def _generate_causal_mask(self, sz, device):
        """Generates an upper-triangular matrix of -inf to enforce causal self-attention."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, v_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - v_seq: Tensor of shape [Batch, Seq_Len, d_model]
        
        Returns:
        - predicted_u: Tensor of shape [Batch, Seq_Len, input_dim]
        """
        batch_size, seq_len, _ = v_seq.shape
        
        # Add learnable positional embeddings (truncated to current seq_len)
        x = v_seq + self.pos_encoder[:, :seq_len, :]
        
        # Enforce causal masking so timestep t cannot attend to future timesteps t+1
        mask = self._generate_causal_mask(seq_len, v_seq.device)
        
        # Pass sequence through Transformer core
        x_out = self.core(x, mask=mask, is_causal=True) # Shape: [Batch, Seq_Len, d_model]
        
        return self.output_proj(x_out) # Shape: [Batch, Seq_Len, input_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Resets the sliding history buffer used during closed-loop evaluation.
        """
        # Initialize an empty buffer for storing input history during step-by-step rollout
        self.seq_buffer = torch.empty(batch_size, 0, self.d_model, device=device)

    def step(self, v_k_single):
        """
        Closed-loop evaluation step. Appends current step vector to history, 
        evaluates attention over the active window, and predicts the next control action.
        
        Parameters:
        - v_k_single: Tensor of shape [Batch, d_model] or [Batch, 1, d_model]
        
        Returns:
        - u_out: Tensor of shape [Batch, input_dim]
        """
        if self.seq_buffer is None:
            raise RuntimeError("Inference memory is uninitialized. Please call reset_memory() first.")
            
        # Standardize input shape to [Batch, 1, d_model]
        if v_k_single.dim() == 2:
            x_3d = v_k_single.unsqueeze(1)
        else:
            x_3d = v_k_single
            
        # Append latest step to sliding history buffer
        self.seq_buffer = torch.cat([self.seq_buffer, x_3d], dim=1)
        
        # Maintain window size within maximum positional encoding length
        if self.seq_buffer.size(1) > self.max_seq_len:
            self.seq_buffer = self.seq_buffer[:, -self.max_seq_len:, :]
            
        # Process full sequence history through forward pass
        full_out = self.forward(self.seq_buffer) # Shape: [Batch, Current_Seq_Len, input_dim]
        
        # Extract prediction corresponding strictly to the latest timestep
        return full_out[:, -1, :] # Shape: [Batch, input_dim]

    @staticmethod
    def _find_valid_nhead(d_model):
        """Utility to suggest a valid head count if the default fails."""
        for h in [8, 4, 2, 1]:
            if d_model % h == 0:
                return h
        return 1