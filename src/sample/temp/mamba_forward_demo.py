import torch
import torch.nn as nn
from mamba_ssm import Mamba

# ============================================================
# 1. Device setup (CRITICAL for Mamba)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. Reproducibility
# ============================================================
torch.manual_seed(0)

# ============================================================
# 3. Simulated controlled dynamical system
#    y_{k+1} = a y_k + b u_k
# ============================================================
def simulate(T=1000):
    a, b = 0.9, 0.5
    y = torch.zeros(T + 1, 1, device=device)
    u = torch.randn(T, 1, device=device)

    for k in range(T):
        y[k + 1] = a * y[k] + b * u[k]

    return y[:-1], u, y[1:]

# Generate data
y, u, y_next = simulate()

# Stack inputs as [y_k, u_k]
x = torch.cat([y, u], dim=1)

# ============================================================
# 4. Sliding window dataset
# ============================================================
L = 20  # sequence length

def make_sequences(x, target):
    Xs, Ys = [], []
    for i in range(len(x) - L):
        Xs.append(x[i : i + L])
        Ys.append(target[i + L])
    return torch.stack(Xs), torch.stack(Ys)

X, Y = make_sequences(x, y_next)

# Ensure tensors are on GPU
X = X.to(device)
Y = Y.to(device)

print("Data on CUDA:", X.is_cuda)

# ============================================================
# 5. Mamba forward model
# ============================================================
class MambaModel(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()

        # Input projection: [y, u] → latent
        self.input_proj = nn.Linear(2, d_model)

        # Mamba SSM block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Output projection: latent → y_{k+1}
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, sequence_length, input_dim)
        x = self.input_proj(x)
        x = self.mamba(x)
        return self.output_proj(x[:, -1])

# ============================================================
# 6. Training setup
# ============================================================
model = MambaModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("Model on CUDA:", next(model.parameters()).is_cuda)

# ============================================================
# 7. Training loop
# ============================================================
epochs = 20

for epoch in range(epochs):
    optimizer.zero_grad()

    y_hat = model(X)
    loss = loss_fn(y_hat, Y)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss = {loss.item():.6f}")

# ============================================================
# 8. Final sanity check
# ============================================================
with torch.no_grad():
    test_pred = model(X[:1])
    print("Sample prediction:", test_pred.item())