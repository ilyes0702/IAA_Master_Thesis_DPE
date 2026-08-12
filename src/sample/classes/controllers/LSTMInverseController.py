import torch
import torch.nn as nn

class LSTMInverseController(nn.Module):
    def __init__(self, hyperparam_config, feature_dim=None):
        """
        LSTM-based Inverse Controller supporting arbitrary sliding window inputs 
        and stateful step-by-step rolling evaluation.
        
        Parameters:
        - hyperparam_config: Dictionary containing model architecture settings.
        - feature_dim: (Optional) Explicit dimension of vector v_k. 
                       If not provided, calculated from n_y, n_u, and plant dimensions.
        """
        super().__init__()
        
        # 1. Extract dynamic MIMO dimensions
        self.input_dim = hyperparam_config["plant"]["input_dim"]   # Dimension of plant output y
        self.output_dim = hyperparam_config["plant"]["output_dim"] # Dimension of plant control u
        
        # LSTM-specific hyperparams with sensible fallbacks
        lstm_cfg = hyperparam_config["lstm"]
        self.hidden_dim = lstm_cfg["hidden_size"]
        self.num_layers = lstm_cfg["num_layers"]
        self.dropout = lstm_cfg["dropout"] if self.num_layers > 1 else 0.0
        
        # 2. Compute dynamic input dimension based on sliding window sizes
        if feature_dim is not None:
            self.d_model = feature_dim
        else:
            n_y = hyperparam_config["train"]["n_y"]
            n_u = hyperparam_config["train"]["n_u"]
            self.d_model = n_u * self.input_dim + (n_y + 2) * self.output_dim
            
        print(f"🛠️ Initializing LSTM core with d_model = {self.d_model}, hidden_dim = {self.hidden_dim}, num_layers = {self.num_layers}")
        
        # 3. Instantiate LSTM Core
        self.core = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        
        # 4. Map latent features back to actuator control dimensions
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)
        
        # Inference recurrent hidden and cell states
        self.lstm_state = None

    def forward(self, v_seq):
        """
        Standard 3D Batch sequence training forward pass.
        
        Parameters:
        - v_seq: Tensor of shape [Batch, Seq_Len, d_model]
        
        Returns:
        - predicted_u: Tensor of shape [Batch, Seq_Len, input_dim]
        """
        # Pass full sequence through LSTM core
        # x shape: [Batch, Seq_Len, hidden_dim]
        x, _ = self.core(v_seq)
        
        return self.output_proj(x) # Shape: [Batch, Seq_Len, input_dim]

    def reset_memory(self, batch_size=1, device="cuda"):
        """
        Allocates or resets zero-filled (h_0, c_0) state tuple.
        Essential for stateful step-by-step rolling simulation.
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        self.lstm_state = (h_0, c_0)

    def step(self, v_k_single):
        """
        Closed-loop evaluation step. Consumes current lookback vector, updates 
        hidden/cell states, and outputs the control instruction.
        
        Parameters:
        - v_k_single: Tensor of shape [Batch, d_model] or [Batch, 1, d_model]
        
        Returns:
        - u_out: Tensor of shape [Batch, input_dim]
        """
        if self.lstm_state is None:
            raise RuntimeError("Inference states are uninitialized. Please call reset_memory() first.")
            
        # Standardize input shape to 3D: [Batch, 1, d_model]
        if v_k_single.dim() == 2:
            x_3d = v_k_single.unsqueeze(1)
        else:
            x_3d = v_k_single
            
        # Compute single timestep forward pass and update hidden states (h, c)
        x_out_3d, self.lstm_state = self.core(x_3d, self.lstm_state)
        
        # Squeeze sequence dimension back to 2D: [Batch, hidden_dim]
        x_out = x_out_3d.squeeze(1)
        
        # Project to physical actuator outputs
        return self.output_proj(x_out) # Shape: [Batch, input_dim]