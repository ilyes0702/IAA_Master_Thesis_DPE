"""
Linear SISO Inverse-Control Benchmark
RC (ESN) vs SSM (Mamba-like)

System:
    x_dot = -a x + b u
    y = x

Task:
    Learn inverse mapping (x, x_dot) -> u
    and evaluate closed-loop tracking.
"""

# ===============================
# Imports
# ===============================
import numpy as np
from seqControl.sample.utils.plotting_utils import plot_signals
import time

# Reservoir Computing
from reservoirpy.nodes import Reservoir, Ridge

# Mamba-like SSM (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim

# ===============================
# System definition
# ===============================
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


# ===============================
# Reference trajectories
# ===============================
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


# ===============================
# Reservoir Computing (ESN)
# ===============================
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


# ===============================
# Mamba-like SSM (Causal)
# ===============================
class SimpleSSM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.C = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, time, input_dim)
        batch, T, _ = x.shape
        h = torch.zeros(batch, self.A.shape[0])
        ys = []
        for t in range(T):
            h = torch.tanh(h @ self.A + x[:, t] @ self.B.T)
            ys.append(self.C(h))
        return torch.stack(ys, dim=1)


def train_ssm(X, Y, epochs=50, lr=1e-3):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(0)

    model = SimpleSSM()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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


# ===============================
# Closed-loop evaluation
# ===============================
def evaluate_controller(model, model_type, X):
    if model_type == "esn":
        u = model.run(X).flatten()
    elif model_type == "ssm":
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
            u = model(X_t).squeeze().numpy()
    else:
        raise ValueError("Unknown model type")

    return simulate_linear_system(u)


# ===============================
# Metrics
# ===============================
def tracking_metrics(x, x_ref):
    e = x - x_ref
    return {
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "IAE": np.sum(np.abs(e)) * dt,
        "Max error": np.max(np.abs(e)),
    }


# ===============================
# Main experiment
# ===============================
if __name__ == "__main__":
    T = 20
    t = np.arange(0, T, dt)

    # Training data
    X_train, Y_train, _ = build_dataset(t, kind="train")

    # Train models
    esn, t_esn = train_esn(X_train, Y_train)
    ssm, t_ssm = train_ssm(X_train, Y_train)

    # Test data
    X_test, _, x_ref_test = build_dataset(t, kind="test")

    # Closed-loop simulations
    x_esn = evaluate_controller(esn, "esn", X_test)
    x_ssm = evaluate_controller(ssm, "ssm", X_test)

    # Metrics
    print("\n=== Performance Metrics ===")
    print("ESN:", tracking_metrics(x_esn, x_ref_test))
    print("SSM:", tracking_metrics(x_ssm, x_ref_test))

    print("\n=== Training Time ===")
    print(f"ESN: {t_esn:.3f} s")
    print(f"SSM: {t_ssm:.3f} s")

    plot_signals(
        t,
        [x_ref_test, x_esn, x_ssm],
        labels=["Reference", "ESN", "SSM"],
        title="Linear SISO Inverse Control – Closed-loop Tracking",
        xlabel="Time [s]",
        ylabel="x",
        figsize=(10, 4),
        show=True,
        filename="linear_siso_inverse_closedloop_tracking",
        dirname="linear_siso_inverse_benchmark",
    )

