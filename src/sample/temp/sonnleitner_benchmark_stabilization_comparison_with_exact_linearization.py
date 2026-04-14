"""
Sonnleitner Yeast Fermentation
Equilibrium Stabilization Benchmark

Controllers:
- Exact feedback linearization (FL)
- ESN (learned stabilizing policy)
- Mamba (learned stabilizing policy)

Objective:
Stabilize biomass X at X_star and compare control deviation.
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import time
import matplotlib.pyplot as plt
from src.sample.utils.plotting_utils import plot_signals

from reservoirpy.nodes import Reservoir, Ridge

import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

plt.style.use("src/sample/style.mplstyle")

# ============================================================
# Model parameters
# ============================================================
mu_max  = 0.4
K_S     = 0.1
Y_XS    = 0.5
q_s_max = 1.0
alpha  = 0.8
S_in   = 20.0
dt     = 0.01

# Equilibrium
X_star = 2.0
mu_est = mu_max * S_in / (K_S + S_in)
D_star = mu_est

# Feedback linearization gain
k_fl = 0.8

# ============================================================
# Sonnleitner model
# ============================================================
def sonnleitner_step(x, s, p, D):
    mu = mu_max * s / (K_S + s)
    q_s = mu / Y_XS

    if q_s <= q_s_max:
        q_eth = 0.0
    else:
        q_eth = alpha * (q_s - q_s_max)

    dx = (mu - D) * x
    ds = D * (S_in - s) - q_s * x
    dp = q_eth * x - D * p

    return dx, ds, dp


def simulate_step(x, s, p, D):
    dx, ds, dp = sonnleitner_step(x, s, p, D)
    return x + dt * dx, s + dt * ds, p + dt * dp

# ============================================================
# ✅ Correct training data generator (stabilizing!)
# ============================================================
def build_dataset_stabilization(
    N=20000,
    X_min=0.3,
    X_max=10.0,
):
    """
    Train on the stabilizing feedback-linearizing policy
    over a wide off-equilibrium state range.
    """
    X = np.random.uniform(X_min, X_max, size=(N, 1))
    S = S_in * np.ones_like(X)

    mu = mu_max * S / (K_S + S)
    D = mu + k_fl * (X - X_star) / X
    D = np.clip(D, 0.01, 0.6)

    return X, D

# ============================================================
# ESN
# ============================================================
def train_esn(X, D):
    res = Reservoir(
        units=500,
        sr=0.9,
        lr=1.0,
        input_scaling=1.0,
        seed=42,
    )
    readout = Ridge(ridge=1e-6)
    esn = res >> readout

    start = time.perf_counter()
    esn.fit(X, D, warmup=200)
    return esn, time.perf_counter() - start

# ============================================================
# Mamba
# ============================================================
class MambaController(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.mamba(x)
        return self.out_proj(x)


def train_mamba(X, D, device="cpu", epochs=60):
    model = MambaController().to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    D_t = torch.tensor(D, dtype=torch.float32, device=device).unsqueeze(0)

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), D_t)
        loss.backward()
        opt.step()

    if device == "cuda":
        torch.cuda.synchronize()

    return model, time.perf_counter() - start

# ============================================================
# Controllers
# ============================================================
def controller_fl(x, s):
    mu = mu_max * s / (K_S + s)
    return mu + k_fl * (x - X_star) / x


def controller_esn(esn, x):
    return esn.run(np.array([[x]]))[0, 0]


def controller_mamba(model, x, device):
    with torch.no_grad():
        x_t = torch.tensor([[x]], dtype=torch.float32, device=device)
        return model(x_t.unsqueeze(0)).item()

# ============================================================
# Closed-loop simulation
# ============================================================
def simulate_closed_loop(controller, controller_type, T=3000, device="cpu"):
    t = np.arange(0, T, dt)
    x, s, p = X_star + 0.3, S_in, 0.0

    xs, Ds = [], []

    for _ in t:
        if controller_type == "fl":
            D = controller_fl(x, s)
        elif controller_type == "esn":
            D = controller_esn(controller, x)
        else:
            D = controller_mamba(controller, x, device)

        D = np.clip(D, 0.01, 0.6)
        x, s, p = simulate_step(x, s, p, D)

        xs.append(x)
        Ds.append(D)

    return t, np.array(xs), np.array(Ds)

# ============================================================
# Metrics
# ============================================================
def control_metrics(D):
    DeltaD = D - D_star
    return {
        "RMS ΔD": np.sqrt(np.mean(DeltaD**2)),
        "Mean |ΔD|": np.mean(np.abs(DeltaD)),
        "Max |ΔD|": np.max(np.abs(DeltaD)),
    }

# ============================================================
# Experiment
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Training data
    X_train, D_train = build_dataset_stabilization()

    # Train controllers
    esn, t_esn = train_esn(X_train, D_train)
    mamba, t_mamba = train_mamba(X_train, D_train, device=device)

    # Closed-loop simulations
    t, X_fl, D_fl = simulate_closed_loop(None, "fl")
    _, X_esn, D_esn = simulate_closed_loop(esn, "esn")
    _, X_mamba, D_mamba = simulate_closed_loop(mamba, "mamba", device=device)

    print("\nTraining time:")
    print("ESN:", t_esn)
    print("Mamba:", t_mamba)

    print("\nControl deviation metrics:")
    print("FL:", control_metrics(D_fl))
    print("ESN:", control_metrics(D_esn))
    print("Mamba:", control_metrics(D_mamba))

    # Biomass plot
    plot_signals(
        t,
        [X_fl, X_esn, X_mamba],
        labels=["Feedback linearization", "ESN", "Mamba"],
        title="Sonnleitner Fermentation – Biomass Stabilization",
        xlabel="Time [h]",
        ylabel="Biomass X [g/L]",
        filename="biomass_stabilization_all",
        dirname="sonnleitner_final",
    )

    # Control deviation plot
    plot_signals(
        t,
        [D_fl - D_star, D_esn - D_star, D_mamba - D_star],
        labels=["FL ΔD", "ESN ΔD", "Mamba ΔD"],
        title="Control Deviation Comparison",
        xlabel="Time [h]",
        ylabel="ΔD [1/h]",
        filename="control_deviation_all",
        dirname="sonnleitner_final",
    )
