"""
Sonnleitner Yeast Fermentation
Inverse + Feedback Trajectory Tracking

Comparison:
- Exact feedback linearization
- Reservoir Computing (ESN)

Control input: dilution rate D(t)
Tracking objective: biomass X(t)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import time
import matplotlib.pyplot as plt
from seqControl.sample.utils.plotting_utils import plot_signals

from reservoirpy.nodes import Reservoir, Ridge

plt.style.use("src/sample/style.mplstyle")

# ============================================================
# Model parameters
# ============================================================
mu_max = 0.4        # 1/h
K_S    = 0.1        # g/L
Y_XS   = 0.5
q_s_max = 1.0
alpha = 0.8
S_in  = 20.0
dt    = 0.01

# Feedback gain (shared by both controllers!)
k_fb = 0.5

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
# Reference trajectory
# ============================================================
def generate_reference(t, kind="train"):
    if kind == "train":
        #X_ref = 2.0 + 0.5 * np.sin(0.2 * t)
        X_ref = np.exp(-0.05 * t)  # more challenging
    else:
        #X_ref = 2.0 + 0.4 * np.sin(0.5 * t + 1.0)
        X_ref = np.exp(-0.05 * t)  # more challenging

    dX_ref = np.gradient(X_ref, dt)
    return X_ref, dX_ref

# ============================================================
# Training data for ESN inverse
# ============================================================
def build_dataset(t):
    X_ref, dX_ref = generate_reference(t, "train")

    mu_est = mu_max * 0.8
    D_ref = mu_est - dX_ref / X_ref
    D_ref = np.clip(D_ref, 0.01, 0.6)

    X_in = np.column_stack([X_ref, dX_ref])
    Y_out = D_ref.reshape(-1, 1)
    return X_in, Y_out

# ============================================================
# ESN inverse model
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
# Exact feedback linearization inverse
# ============================================================
def inverse_feedback_linearization(X_ref, dX_ref, S):
    mu = mu_max * S / (K_S + S)
    D_inv = mu - dX_ref / X_ref
    return np.clip(D_inv, 0.01, 0.6)

# ============================================================
# Closed-loop inverse + feedback tracking
# ============================================================
def simulate_tracking(controller_type, controller, X_ref, dX_ref):
    x, s, p = X_ref[0], S_in, 0.0
    xs = []

    for k in range(len(X_ref)):
        # inverse term
        if controller_type == "fl":
            D_inv = inverse_feedback_linearization(X_ref[k], dX_ref[k], s)
        else:
            D_inv = controller.run(
                np.array([[X_ref[k], dX_ref[k]]])
            )[0, 0]

        # shared feedback correction
        D = D_inv + k_fb * (x - X_ref[k])
        D = np.clip(D, 0.01, 0.6)

        dx, ds, dp = sonnleitner_step(x, s, p, D)
        x += dt * dx
        s += dt * ds
        p += dt * dp

        xs.append(x)

    return np.array(xs)

# ============================================================
# Experiment
# ============================================================
if __name__ == "__main__":
    T = 150
    t = np.arange(0, T, dt)

    # Training ESN
    X_train, Y_train = build_dataset(t)
    esn, t_esn = train_esn(X_train, Y_train)

    # Test reference
    X_ref, dX_ref = generate_reference(t, "test")

    # Closed-loop simulations
    X_fl = simulate_tracking("fl", None, X_ref, dX_ref)
    X_esn = simulate_tracking("esn", esn, X_ref, dX_ref)

    print("Training time ESN:", t_esn)

    plot_signals(
        t,
        [X_ref, X_fl, X_esn],
        labels=["Reference", "Feedback linearization", "ESN"],
        title="Sonnleitner Fermentation – Inverse + Feedback Tracking",
        xlabel="Time [h]",
        ylabel="Biomass X [g/L]",
        filename="sonnleitner_tracking_esn_vs_fl",
        dirname="sonnleitner_rc_vs_fl",
    )
