import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
from src.sample.utils.plotting_utils import plot_signals
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MIMO Fermentation Process (Idiophase) ---
class IdiophaseMIMO:
    def __init__(self):
        # Parameters from Table 1 & 2
        self.mu_max = 0.12
        self.Ks = 50.0
        self.p1 = 0.00047
        self.p2 = 200000.0
        self.ms = 23.0
        self.mu_pen = 3.0  # [mg Pen/(gTSh)] [cite: 117]
        self.p5 = 0.9      # [cite: 117]
        self.p6 = 100.0    # [cite: 117]
        self.p7 = 0.04     # [cite: 117]
        self.q = 2000.0    # [cite: 117]
        self.V = 170.0     # Volume is constant 170l in idiophase [cite: 117]

    def step(self, state, u, dt=0.1):
        # state: [x1, x2, x3, x4] | u: [u1, u2]
        x1, x2, x3, x4 = state
        u1, u2 = u
        
        mu = (self.mu_max * x2) / (self.Ks * self.V + x2)
        
        # ODEs (4) [cite: 99-102]
        dx1 = mu * x1
        dx2 = -(1/self.p1)*mu*x1 - (1/self.p5)*self.mu_pen*x1 - self.ms*x1 + self.p2*u1
        dx3 = -(1/self.q)*self.mu_pen*x1 + self.p6*u2
        dx4 = self.mu_pen*x1 - self.p7*x4
        
        state_new = state + np.array([dx1, dx2, dx3, dx4]) * dt
        state_new = np.maximum(state_new, 0)
        
        # Outputs: growth rate and precursor concentration [cite: 107, 108]
        y = np.array([mu, x3/self.V])
        return state_new, y

# --- 2. MIMO Mamba Controller ---
class MambaMIMOController(nn.Module):
    def __init__(self, d_model=64, d_state=16):
        super().__init__()
        # Input: [mu_target, c3_target] | Output: [u1, u2]
        self.input_proj = nn.Linear(2, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, y_target_seq):
        x = self.input_proj(y_target_seq.to(device))
        x = self.mamba(x)
        # Apply sigmoid to ensure 0 <= u_j <= 1 [cite: 114]
        return torch.sigmoid(self.output_proj(x))

# --- 3. Execution and Stabilization ---
def run_mimo_control():
    process = IdiophaseMIMO()
    model = MambaMIMOController().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3a. Training on Random Trajectories
    print("Training MIMO Mamba Controller...")
    for epoch in range(200):
        state = np.array([50.0, 500.0, 8500.0, 10.0]) # Idiophase start context
        u_hist, y_hist = [], []
        for _ in range(50):
            u_rand = np.random.uniform(0, 1, 2)
            state, y = process.step(state, u_rand)
            u_hist.append(u_rand)
            y_hist.append(y)
            
        y_tensor = torch.tensor(y_hist, dtype=torch.float32).unsqueeze(0).to(device)
        u_target = torch.tensor(u_hist, dtype=torch.float32).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(y_tensor), u_target)
        loss.backward()
        optimizer.step()

    # 3b. Stabilization Simulation
    mu_star, c3_star = 0.015, 50.0 # Target values [cite: 113]
    state = np.array([50.0, 500.0, 8500.0, 10.0])
    results = {"mu": [], "c3": [], "u1": [], "u2": []}
    
    # Generate targets sequence
    y_ref = torch.tensor([[mu_star, c3_star]] * 100, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        u_preds = model(y_ref).squeeze(0).cpu().numpy()

    for i in range(100):
        # Apply Mamba feedforward control
        u_applied = u_preds[i]
        state, y_actual = process.step(state, u_applied)
        
        results["mu"].append(y_actual[0])
        results["c3"].append(y_actual[1])
        results["u1"].append(u_applied[0])
        results["u2"].append(u_applied[1])

    # Plotting
    plot_signals(list(range(100)), [results["mu"]],
                 labels=["Growth Rate (mu)"],
                 xlabel="Time Steps", ylabel="Value",
                 title="Growth Rate (mu)",
                 dirname="ex06", filename="mu")
    plot_signals(list(range(100)), [results["c3"]],
                 labels=["Precursor Conc. (c3)"],
                 xlabel="Time Steps", ylabel="Value",
                 title="Precursor Conc. (c3)",
                 dirname="ex06", filename="c3")
    plot_signals(list(range(100)), [results["u1"]],
                 labels=["Dilution Rate u1"],
                 xlabel="Time Steps", ylabel="Value",
                 title="Dilution Rate u1",
                 dirname="ex06", filename="u1")
    plot_signals(list(range(100)), [results["u2"]],
                 labels=["Dilution Rate u2"],
                 xlabel="Time Steps", ylabel="Value",
                 title="Dilution Rate u2",
                 dirname="ex06", filename="u2")

if __name__ == "__main__":
    run_mimo_control()