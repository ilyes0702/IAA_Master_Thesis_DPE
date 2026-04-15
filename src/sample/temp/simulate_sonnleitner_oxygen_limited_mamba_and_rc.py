"""
Oxygen-limited Sonnleitner fermentation
Inverse + feedback tracking control

Comparison:
- Reservoir Computing (ESN) inverse
- Mamba inverse

Both learned via teacher forcing from the true plant
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

from reservoirpy.nodes import Reservoir, Ridge

import matplotlib.pyplot as plt
from src.sample.utils.plotting_utils import plot_signals
plt.style.use("src/sample/style.mplstyle")

# ============================================================
# Model parameters
# ============================================================
mu_max = 0.6
K_S    = 0.1
K_O2   = 0.002
Y_XS   = 0.5

S_in   = 20.0
O2_sat = 0.008
kLa    = 200.0
q_O2_yield = 1.2e-3

dt = 0.01
k_fb = 0.1

WINDOW = 50    # temporal context length

# ============================================================
# Growth and plant dynamics
# ============================================================
def mu_growth(S, O2):
    return mu_max * S / (K_S + S) * O2 / (K_O2 + O2)

def sonnleitner_step(x, s, o2, D):
    mu = mu_growth(s, o2)
    q_s  = mu / Y_XS
    q_O2 = q_O2_yield * mu

    dx  = (mu - D) * x
    ds  = D * (S_in - s) - q_s * x
    dO2 = kLa * (O2_sat - o2) - q_O2 * x
    return dx, ds, dO2

# ============================================================
# Reference trajectory
# ============================================================
def reference_trajectory(t):
    X_ref = 2.0 + 0.4 * np.sin(0.2 * t)
    dX_ref = np.gradient(X_ref, dt)
    return X_ref, dX_ref

# ============================================================
# True inverse (teacher)
# ============================================================
def true_inverse(X_ref, dX_ref, S, O2):
    return mu_growth(S, O2) - dX_ref / X_ref

# ============================================================
# Generate teacher-forced inverse data
# ============================================================
def generate_teacher_data(t):
    X_ref, dX_ref = reference_trajectory(t)

    x, s, o2 = X_ref[0], S_in, O2_sat
    X_seq, Y = [], []

    for k in range(WINDOW, len(t)):
        D_true = true_inverse(X_ref[k], dX_ref[k], s, o2)
        D_true += k_fb * (x - X_ref[k])
        D_true = np.clip(D_true, 0.01, 0.6)

        seq = np.stack(
            [X_ref[k-WINDOW:k], dX_ref[k-WINDOW:k]],
            axis=1
        )
        X_seq.append(seq)
        Y.append(D_true)

        dx, ds, dO2 = sonnleitner_step(x, s, o2, D_true)
        x  += dt * dx
        s  += dt * ds
        o2 += dt * dO2

    return np.array(X_seq), np.array(Y).reshape(-1, 1)

# ============================================================
# Mamba inverse model
# ============================================================
class MambaInverse(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.in_proj = nn.Linear(2, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.mamba(x)
        return self.out_proj(x[:, -1])

def train_mamba(X, Y, epochs=80):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MambaInverse().to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), Y_t)
        loss.backward()
        opt.step()

    return model, device

# ============================================================
# Reservoir Computing inverse (ESN)
# ============================================================
def train_esn(X, Y):
    X_flat = X.reshape(X.shape[0], -1)

    res = Reservoir(
        units=500,
        sr=0.9,
        lr=1.0,
        input_scaling=1.0,
        seed=42
    )
    readout = Ridge(ridge=1e-6)
    esn = res >> readout
    esn.fit(X_flat, Y)
    return esn

# ============================================================
# Closed-loop tracking simulation
# ============================================================
def simulate(controller, t, mode, device=None):
    X_ref, dX_ref = reference_trajectory(t)
    x, s, o2 = X_ref[0], S_in, O2_sat
    X_cl = []

    for k in range(WINDOW, len(t)):
        seq = np.stack(
            [X_ref[k-WINDOW:k], dX_ref[k-WINDOW:k]],
            axis=1
        )

        if mode == "mamba":
            inp = torch.tensor(seq, dtype=torch.float32, device=device)
            with torch.no_grad():
                D_inv = controller(inp.unsqueeze(0)).item()
        else:  # ESN
            D_inv = controller.run(seq.reshape(1, -1))[0, 0]

        D = D_inv + k_fb * (x - X_ref[k])
        D = np.clip(D, 0.01, 0.6)

        dx, ds, dO2 = sonnleitner_step(x, s, o2, D)
        x  += dt * dx
        s  += dt * ds
        o2 += dt * dO2

        X_cl.append(x)

    return X_ref[WINDOW:], np.array(X_cl)

# ============================================================
# Main experiment
# ============================================================
if __name__ == "__main__":

    T = 8
    t = np.arange(0, T, dt)

    print("Generating teacher data...")
    X_train, Y_train = generate_teacher_data(t)

    print("Training ESN inverse...")
    esn = train_esn(X_train, Y_train)

    print("Training Mamba inverse...")
    mamba, device = train_mamba(X_train, Y_train)

    print("Running closed-loop simulations...")
    X_ref, X_esn = simulate(esn, t, mode="esn")
    _,    X_mamba = simulate(mamba, t, mode="mamba", device=device)

    plot_signals(
        t[WINDOW:], [X_ref, X_esn, X_mamba],
        labels=["Reference", "RC + FB", "Mamba + FB"],
        title="Oxygen-Limited Sonnleitner Tracking",
        dirname="sonnleitner_oxygen_limited",
        filename="sonnleitner_rc_vs_mamba_tracking.png"
    )