"""
Sonnleitner Yeast Fermentation
Inverse + Feedback Tracking Control

Comparison:
- Exact feedback linearization (model-based inverse)
- Learned inverse (e.g. ESN or Mamba)

Control input: dilution rate D(t)
Output: biomass X(t)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# Optional: replace with your plot_signals utility
from seqControl.sample.utils.plotting_utils import plot_signals
plt.style.use("src/sample/style.mplstyle")

# ============================================================
# Model parameters
# ============================================================
mu_max  = 0.4        # 1/h
K_S     = 0.1        # g/L
Y_XS    = 0.5
q_s_max = 1.0
alpha  = 0.8
S_in   = 20.0        # g/L

dt = 0.01            # h
k_fb = 0.1           # shared feedback gain

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
def reference_trajectory(t):
    X_ref = 2.0 + 0.4 * np.sin(0.3 * t)
    dX_ref = np.gradient(X_ref, dt)
    return X_ref, dX_ref

# ============================================================
# Exact inverse: feedback linearization
# ============================================================
def inverse_feedback_linearization(X_ref, dX_ref, S):
    """
    Analytical inverse of biomass dynamics
    """
    mu = mu_max * S / (K_S + S)
    D_inv = mu - dX_ref / X_ref
    return np.clip(D_inv, 0.01, 0.6)

# ============================================================
# Example placeholder for learned inverse
# Replace this with ESN / Mamba output in practice
# ============================================================
def learned_inverse_placeholder(X_ref, dX_ref):
    """
    Placeholder inverse (acts like a slightly biased inverse).
    In your real experiments, this is esn.run(...) or mamba(...)
    """
    mu_est = mu_max * S_in / (K_S + S_in)
    D_inv = mu_est - dX_ref / X_ref
    return np.clip(D_inv, 0.01, 0.6)

# ============================================================
# Closed-loop inverse + feedback tracking
# ============================================================
def simulate_tracking(controller_type, t):
    X_ref, dX_ref = reference_trajectory(t)

    x, s, p = X_ref[0], S_in, 0.0
    X_cl, D_cl = [], []

    for k in range(len(t)):

        # --- inverse term ---
        if controller_type == "fl":
            D_inv = inverse_feedback_linearization(
                X_ref[k], dX_ref[k], s
            )
        elif controller_type == "learned":
            D_inv = learned_inverse_placeholder(
                X_ref[k], dX_ref[k]
            )
        else:
            raise ValueError("Unknown controller type.")

        # --- shared feedback ---
        D = D_inv + k_fb * (x - X_ref[k])
        D = np.clip(D, 0.01, 0.6)

        dx, ds, dp = sonnleitner_step(x, s, p, D)
        x += dt * dx
        s += dt * ds
        p += dt * dp

        X_cl.append(x)
        D_cl.append(D)

    return X_ref, np.array(X_cl), np.array(D_cl)

# ============================================================
# Performance metrics
# ============================================================
def performance_metrics(X, X_ref, D):
    e = X - X_ref
    return {
        "RMSE": np.sqrt(np.mean(e**2)),
        "IAE": np.sum(np.abs(e)) * dt,
        "Max error": np.max(np.abs(e)),
        "Control RMS": np.sqrt(np.mean((D - np.mean(D))**2)),
    }

# ============================================================
# Main experiment
# ============================================================
if __name__ == "__main__":

    T = 60
    t = np.arange(0, T, dt)

    # --- simulate controllers ---
    X_ref, X_fl, D_fl = simulate_tracking("fl", t)
    _,     X_learned, D_learned = simulate_tracking("learned", t)

    # --- metrics ---
    metrics_fl = performance_metrics(X_fl, X_ref, D_fl)
    metrics_learned = performance_metrics(X_learned, X_ref, D_learned)

    print("\nExact feedback linearization:")
    for k, v in metrics_fl.items():
        print(f"  {k}: {v:.4f}")

    print("\nLearned inverse + feedback:")
    for k, v in metrics_learned.items():
        print(f"  {k}: {v:.4f}")

    # --- plots ---
    plot_signals(
        t, [X_ref, X_fl, X_learned],
        labels=["Reference", "FL Tracking", "Learned Tracking"],
        xlabel="Time (h)", ylabel="Biomass X (g/L)",
        title="Biomass Tracking",
        dirname="sonnleitner_model_tests", filename="biomass_tracking.png"
    )