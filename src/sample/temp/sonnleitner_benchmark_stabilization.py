"""
Sonnleitner Yeast Fermentation
Equilibrium Stabilization Benchmark
ESN vs REAL Mamba (Selective SSM)

Control input: dilution rate D(t)
Control objective: stabilize biomass at X*
Analysis includes control deviation ΔD(t)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import time
from src.sample.utils.plotting_utils import plot_signals
import matplotlib.pyplot as plt
# Reservoir Computing
from reservoirpy.nodes import Reservoir, Ridge

# PyTorch + Mamba
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

plt.style.use("src/sample/style.mplstyle")
# ============================================================
# Sonnleitner model parameters
# ============================================================
mu_max  = 0.4        # 1/h
K_S     = 0.1        # g/L
Y_XS    = 0.5        # gX / gS
q_s_max = 1.0        # gS / (gX h)
alpha  = 0.8        # ethanol formation coefficient
S_in   = 20.0       # g/L
dt     = 0.01       # h

# Desired equilibrium
X_star = 2.0          # g/L
mu_est = mu_max * 0.8
D_star = mu_est       # equilibrium dilution rate

# ============================================================
# Sonnleitner model dynamics
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


def simulate_sonnleitner(D_traj, x0=0.5, s0=20.0, p0=0.0):
    x, s, p = x0, s0, p0
    xs = []

    for D in D_traj:
        dx, ds, dp = sonnleitner_step(x, s, p, D)
        x += dt * dx
        s += dt * ds
        p += dt * dp
        xs.append(x)

    return np.array(xs)

# ============================================================
# Stabilization training data
# ============================================================
def build_dataset(t):
    # small perturbations around operating point
    X_ref = X_star + 0.05 * np.random.randn(len(t))

    D_ref = D_star * np.ones_like(t)

    X_in = X_ref.reshape(-1, 1)
    Y_out = D_ref.reshape(-1, 1)

    return X_in, Y_out

# ============================================================
# ESN
# ============================================================
def train_esn(X, Y):
    res = Reservoir(
        units=400,
        sr=0.9,
        lr=1.0,
        input_scaling=1.0,
        seed=42,
    )
    readout = Ridge(ridge=1e-6)
    esn = res >> readout

    start = time.perf_counter()
    esn.fit(X, Y, warmup=200)
    train_time = time.perf_counter() - start

    return esn, train_time

# ============================================================
# Mamba inverse model
# ============================================================
class MambaInverseModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.mamba(x)
        return self.out_proj(x)


def train_mamba(X, Y, device="cpu", epochs=80):
    model = MambaInverseModel().to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), Y_t)
        loss.backward()
        opt.step()

    if device == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    return model, train_time

# ============================================================
# Closed-loop stabilization
# ============================================================
def evaluate(model, model_type, X_ctrl, device="cpu"):
    if model_type == "esn":
        D = model.run(X_ctrl).flatten()
    else:
        with torch.no_grad():
            X_t = torch.tensor(X_ctrl, dtype=torch.float32, device=device).unsqueeze(0)
            D = model(X_t).squeeze().cpu().numpy()

    D = np.clip(D, 0.01, 0.6)
    X = simulate_sonnleitner(D)

    return X, D

# ============================================================
# Metrics
# ============================================================
def control_metrics(DeltaD):
    rel_dev = np.abs(DeltaD) / np.abs(D_star)
    return {
        "RMS ΔD": np.sqrt(np.mean(DeltaD**2)),
        "Mean |ΔD|": np.mean(np.abs(DeltaD)),
        "Max |ΔD|": np.max(np.abs(DeltaD)),
        "RMS rel. dev. [%]": 100 * np.sqrt(np.mean(rel_dev**2)),
        "Mean rel. dev. [%]": 100 * np.mean(rel_dev),
        "Max rel. dev. [%]": 100 * np.max(rel_dev),
    }

# ============================================================
# Experiment
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    T = 3000
    t = np.arange(0, T, dt)

    # Training data
    X_train, Y_train = build_dataset(t)

    # Train controllers
    esn, t_esn = train_esn(X_train, Y_train)
    mamba, t_mamba = train_mamba(X_train, Y_train, device=device)

    # Initial deviation from equilibrium
    X_ctrl = (X_star + 0.3 * np.ones_like(t)).reshape(-1, 1)

    # Closed-loop simulations
    X_esn, D_esn = evaluate(esn, "esn", X_ctrl)
    X_mamba, D_mamba = evaluate(mamba, "mamba", X_ctrl, device)

    # Control deviation
    DeltaD_esn = D_esn - D_star
    DeltaD_mamba = D_mamba - D_star

    print("\nTraining time:")
    print("ESN:", t_esn)
    print("Mamba:", t_mamba)

    print("\nControl deviation metrics:")
    print("ESN:", control_metrics(DeltaD_esn))
    print("Mamba:", control_metrics(DeltaD_mamba))

    # Biomass stabilization plot
    plot_signals(
        t,
        [X_esn, X_mamba],
        labels=["ESN", "Mamba"],
        title="Sonnleitner Yeast Fermentation – Biomass Stabilization",
        xlabel="Time [h]",
        ylabel="Biomass X [g/L]",
        filename="sonnleitner_stabilization_biomass",
        dirname="sonnleitner_stabilization",
    )

    # Control deviation plot
    plot_signals(
        t,
        [DeltaD_esn, DeltaD_mamba],
        labels=["ESN ΔD", "Mamba ΔD"],
        title="Control Deviation from Equilibrium Dilution Rate",
        xlabel="Time [h]",
        ylabel="ΔD [1/h]",
        filename="sonnleitner_control_deviation",
        dirname="sonnleitner_stabilization",
    )