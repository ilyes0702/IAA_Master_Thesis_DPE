import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from seqControl.sample.utils.plotting_utils import plot_signals

# ---------------------------------------------------------------------
# MANDATORY: import core hardware-accelerated Mamba blocks
# Ensure you have run: pip install mamba-ssm causal-conv1d
# ---------------------------------------------------------------------
from mamba_ssm import Mamba

# Mock tracking utility for standard system executions
def save_plot_image(image, filename, dirname):
    pass 

# ===========================================================

# =====================================================================
# 1. SIMULATE NONLINEAR DYNAMICAL PLANT
# =====================================================================
print("Simulating non-linear dynamical system trajectories...")
dt = 0.01
seq_len = 1000  # length of each trajectory timeline sequence
num_sequences = 3000  # Batch size equivalent

# Construct arrays tracking state values across multiple sequence environments
t = np.arange(seq_len) * dt
u_batches = []
y_batches = []

for s in range(num_sequences):
    # Dynamic varied multi-sine excitation signal per trial run
    ph1, ph2 = np.random.uniform(0, 2*np.pi, 2)
    u = 1.2 * np.sin(2 * np.pi * 0.6 * t + ph1) + 0.8 * np.cos(2 * np.pi * 1.4 * t + ph2)
    
    y = np.zeros(seq_len)
    y[0], y[1] = 0.0, 0.0
    for k in range(1, seq_len - 1):
        # Continuous plant logic equation: y_ddot + (y^2 - 1)*y_dot + y^3 = u
        y_dot = (y[k] - y[k-1]) / dt
        y_ddot = u[k] - (y[k]**2 - 1)*y_dot - y[k]**3
        y[k+1] = 2*y[k] - y[k-1] + (dt**2) * y_ddot
        
    u_batches.append(u)
    y_batches.append(y)

# Shape layout adjustments for modern Mamba layers: (Batch, Sequence, Dim)
X_data = torch.tensor(np.array(u_batches), dtype=torch.float32).unsqueeze(-1) # (30, 1000, 1)
Y_data = torch.tensor(np.array(y_batches), dtype=torch.float32).unsqueeze(-1) # (30, 1000, 1)

# Divide into standard sequence test segments
X_train, X_val = X_data[:24], X_data[24:]
Y_train, Y_val = Y_data[:24], Y_data[24:]

# =====================================================================
# 2. DESIGN MAMBA SYSTEM IDENTIFICATION BACKBONE
# =====================================================================
class MambaSystemIdentifier(nn.Module):
    def __init__(self, d_model=32, d_state=16):
        super().__init__()
        # Map raw control input feature dimension up to state tracking size
        self.input_projection = nn.Linear(1, d_model)
        
        # Core Selective Structured State Space layer block configuration
        self.mamba_block = Mamba(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=4, 
            expand=2
        )
        
        # Outgoing projection layer reducing signals to physical target states
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, u_seq):
        # Input shape: (Batch, Seq_Len, 1)
        x = self.input_projection(u_seq)       # -> (Batch, Seq_Len, d_model)
        x = self.mamba_block(x)                # -> (Batch, Seq_Len, d_model)
        y_hat = self.output_projection(x)      # -> (Batch, Seq_Len, 1)
        return y_hat

# Mamba layers strictly require CUDA architecture execution optimizations
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ WARNING: Native mamba-ssm operations require GPU acceleration to execute successfully.")

model = MambaSystemIdentifier(d_model=32, d_state=16).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
criterion = nn.MSELoss()

# =====================================================================
# 3. SEQUENCE LEARNING LOOP
# =====================================================================
print(f"\nTraining Mamba SSM Identifier on: {device}...")
X_train, Y_train = X_train.to(device), Y_train.to(device)

model.train()
for epoch in range(150):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, Y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Batch Trajectory Tracking MSE: {loss.item():.6f}")

# =====================================================================
# 4. SYSTEM VERIFICATION VIA PLOT_SIGNALS
# =====================================================================
print("\nEvaluating Mamba on unseen validation trajectory records...")
model.eval()
X_val, Y_val = X_val.to(device), Y_val.to(device)

with torch.no_grad():
    val_predictions = model(X_val)

# Pick an arbitrary out-of-sample sequence sequence index (e.g. index 0 of validation subset)
true_trajectory = Y_val[0].cpu().numpy().ravel()
mamba_simulated = val_predictions[0].cpu().numpy().ravel()

signals_comparison = [true_trajectory, mamba_simulated]
labels_comparison = ["True Plant Profile", "Mamba Recurrent Tracking"]

print(signals_comparison[0][:10])  # Print first 10 values of true trajectory
print("dattebayo") 
plot_signals(
    t=t,
    signals=signals_comparison,
    labels=labels_comparison,
    title="Mamba SSM State Identification Performance",
    xlabel="Time [s]",
    ylabel="Plant Output Status ($y$)",
    figsize=(7, 7),
    filename="mamba_ssm_identification_results",
    dirname="mamba_ssm_results",
)