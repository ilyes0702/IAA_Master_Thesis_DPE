import torch
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate a simple sine wave time series
T = 10
t = torch.linspace(0, 50, T)
data = torch.sin(t)

# Create sliding windows
def create_dataset(series, seq_len=20):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return torch.stack(X), torch.stack(y)

X, y = create_dataset(data)
X = X.unsqueeze(-1)  # shape: (batch, seq_len, 1)
print(X)
X = X.to(device)
y = y.to(device)

import torch.nn as nn
from mamba_ssm import Mamba

class MambaTimeSeries(nn.Module):
    def __init__(self, d_model=32, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # use last time step
        x = x[:, -1, :]
        return self.output_proj(x)

import torch
import math

model = MambaTimeSeries().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    
    preds = model(X)
    loss = loss_fn(preds.squeeze(), y)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


    # Take last window
test_input = X[-1:].clone()

model.eval()
with torch.no_grad():
    pred = model(test_input)

print("True value:", y[-1].item())
print("Predicted:", pred.item())