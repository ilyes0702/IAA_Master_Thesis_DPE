"""
Sonnleitner Yeast Fermentation
Inverse + Feedback Tracking Control
ESN vs REAL Mamba (Selective SSM)

Control input: dilution rate D(t)
Tracking objective: biomass X(t)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import time
from src.sample.utils.plotting_utils import plot_signals
import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")

# Reservoir Computing
from reservoirpy.nodes import Reservoir, Ridge

# PyTorch + Mamba
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

# ============================================================
# Model parameters
# ============================================================
mu_max = 0.4
K_S    = 0.1
Y_XS   = 0.5
q_s_max = 1.0
alpha = 0.8
S_in  = 20.0
dt    = 0.01

# Feedback gain (tune 0.05–0.3)
k_fb = 0.15

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


# ============================================================
# Reference trajectory + inverse target
# ============================================================
def build_dataset(t, kind="train"):
    if kind == "train":
        X_ref = 2.0 + 0.5 * np.sin(0.2 * t)
    else:
        X_ref = 2.0 + 0.4 * np.sin(0.5 * t + 1.0)

    dX_ref = np.gradient(X_ref, dt)

    # Approximate inverse from biomass balance
    mu_est = mu_max * 0.8
    D_ref = mu_est - dX_ref / X_ref
    D_ref = np.clip(D_ref, 0.01, 0.6)

    X_in = np.column_stack([X_ref, dX_ref])
    Y_out = D_ref.reshape(-1, 1)

    return X_in, Y_out, X_ref, dX_ref


# ============================================================
# ESN
# ============================================================
def train_esn(X, Y):
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
    esn.fit(X, Y, warmup=200)
    train_time = time.perf_counter() - start

    return esn, train_time


# ============================================================
# Mamba inverse model
# ============================================================
class MambaInverseModel(nn.Module):
    def __init__(self, input_dim=2, d_model=64):
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
# Inverse + feedback tracking simulation
# ============================================================
def evaluate(model, model_type, X_ref, dX_ref, device="cpu"):
    x, s, p = X_ref[0], S_in, 0.0

    xs = []
    Ds = []

    for k in range(len(X_ref)):
        # --- inverse term ---
        if model_type == "esn":
            D_inv = model.run(
                np.array([[X_ref[k], dX_ref[k]]])
            )[0, 0]
        else:
            with torch.no_grad():
                inp = torch.tensor(
                    [[X_ref[k], dX_ref[k]]],
                    dtype=torch.float32,
                    device=device
                )
                D_inv = model(inp.unsqueeze(0)).item()

        # --- feedback correction ---
        D = D_inv + k_fb * (x - X_ref[k])
        D = np.clip(D, 0.01, 0.6)

        dx, ds, dp = sonnleitner_step(x, s, p, D)
        x += dt * dx
        s += dt * ds
        p += dt * dp

        xs.append(x)
        Ds.append(D)

    return np.array(xs), np.array(Ds)


# ============================================================
# Experiment
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    T = 150
    t = np.arange(0, T, dt)

    # Training
    X_train, Y_train, _, _ = build_dataset(t, "train")
    esn, t_esn = train_esn(X_train, Y_train)
    mamba, t_mamba = train_mamba(X_train, Y_train, device=device)

    # Test / tracking
    _, _, Xref_test, dX_test = build_dataset(t, "test")

    X_esn, D_esn = evaluate(esn, "esn", Xref_test, dX_test, device)
    X_mamba, D_mamba = evaluate(mamba, "mamba", Xref_test, dX_test, device)

    print("\nTraining time:")
    print("ESN:", t_esn)
    print("Mamba:", t_mamba)

    plot_signals(
        t,
        [Xref_test, X_esn, X_mamba],
        labels=["Reference", "ESN", "Mamba"],
        title="Sonnleitner Fermentation – Inverse + Feedback Tracking",
        xlabel="Time [h]",
        ylabel="Biomass X [g/L]",
        filename="sonnleitner_inverse_feedback_tracking",
        dirname="sonnleitner_benchmark",
    )