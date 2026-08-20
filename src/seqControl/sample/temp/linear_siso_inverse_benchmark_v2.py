"""
Linear SISO Inverse-Control Benchmark
Reservoir Computing (ESN) vs Mamba (Selective SSM)

System:
    x_dot = -a x + b u
    y = x

Task:
    Learn inverse mapping (x, x_dot) -> u
    and evaluate closed-loop tracking.
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from seqControl.sample.utils.plotting_utils import plot_signals
import time

# Reservoir Computing
from reservoirpy.nodes import Reservoir, Ridge

# PyTorch + Mamba
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

# ============================================================
# System definition
# ============================================================
a = 1.0
b = 1.0
dt = 0.01


def simulate_linear_system(u, x0=0.0):
    x = x0
    xs = []
    for ui in u:
        dx = -a * x + b * ui
        x += dt * dx
        xs.append(x)
    return np.array(xs)


def inverse_control(x, dx):
    return (dx + a * x) / b


# ============================================================
# Reference trajectories
# ============================================================
def generate_reference(t, kind="train"):
    if kind == "train":
        return 0.5 * np.sin(0.5 * t) + 0.2 * np.sin(1.5 * t)
    elif kind == "test":
        return 0.4 * np.sin(1.0 * t + 0.5) + 0.3 * np.sin(2.0 * t)
    else:
        raise ValueError("Unknown trajectory type")


def build_dataset(t, kind="train"):
    x_ref = generate_reference(t, kind)
    dx_ref = np.gradient(x_ref, dt)
    u_ref = inverse_control(x_ref, dx_ref)

    X = np.column_stack([x_ref, dx_ref])
    Y = u_ref.reshape(-1, 1)
    return X, Y, x_ref


# ============================================================
# Reservoir Computing (ESN)
# ============================================================
def train_esn(X, Y):
    res = Reservoir(
        units=300,
        sr=0.9,
        lr=1.0,
        input_scaling=1.0,
        seed=42,
    )
    readout = Ridge(ridge=1e-6)
    esn = res >> readout

    start = time.perf_counter()
    esn.fit(X, Y, warmup=100)
    train_time = time.perf_counter() - start

    return esn, train_time


# ============================================================
# Mamba Model (REAL Selective SSM)
# ============================================================
class MambaInverseModel(nn.Module):
    def __init__(
        self,
        input_dim=2,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, time, input_dim)
        """
        x = self.input_proj(x)
        x = self.mamba(x)
        u = self.output_proj(x)
        return u


def train_mamba(X, Y, epochs=50, lr=1e-3, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device).unsqueeze(0)

    model = MambaInverseModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = loss_fn(y_pred, Y_t)
        loss.backward()
        optimizer.step()
    train_time = time.perf_counter() - start

    return model, train_time


# ============================================================
# Closed-loop evaluation
# ============================================================
def evaluate_controller(model, model_type, X, device="cpu"):
    if model_type == "esn":
        u = model.run(X).flatten()

    elif model_type == "mamba":
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
            u = model(X_t).squeeze().cpu().numpy()

    else:
        raise ValueError("Unknown model type")

    return simulate_linear_system(u)


# ============================================================
# Metrics
# ============================================================
def tracking_metrics(x, x_ref):
    e = x - x_ref
    return {
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "IAE": np.sum(np.abs(e)) * dt,
        "Max error": np.max(np.abs(e)),
    }


# ============================================================
# Main Experiment
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    T = 200
    t = np.arange(0, T, dt)

    # Training data
    X_train, Y_train, _ = build_dataset(t, kind="train")

    # Train ESN
    esn, t_esn = train_esn(X_train, Y_train)

    # Train Mamba
    mamba_model, t_mamba = train_mamba(
        X_train, Y_train, epochs=60, lr=1e-3, device=device
    )

    # Test data
    X_test, _, x_ref_test = build_dataset(t, kind="test")

    # Closed-loop evaluation
    x_esn = evaluate_controller(esn, "esn", X_test)
    x_mamba = evaluate_controller(mamba_model, "mamba", X_test, device=device)

    # Metrics
    print("\n=== Performance Metrics ===")
    print("ESN:", tracking_metrics(x_esn, x_ref_test))
    print("Mamba:", tracking_metrics(x_mamba, x_ref_test))

    print("\n=== Training Time ===")
    print(f"ESN:    {t_esn:.3f} s")
    print(f"Mamba:  {t_mamba:.3f} s")

    # Plot
    
    plot_signals(
        t,
        [x_ref_test],
        labels=["Reference"],
        title="Linear SISO Inverse Control – Closed-loop Tracking",
        xlabel="Time [s]",
        ylabel="x",
        filename="linear_siso_inverse_closedloop_tracking",
        dirname="linear_siso_inverse_benchmark",
    )