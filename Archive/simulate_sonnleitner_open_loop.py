import numpy as np
import matplotlib.pyplot as plt
from seqControl.sample.utils.plotting_utils import plot_signals
plt.style.use("src/sample/style.mplstyle")

# ============================================================
# Sonnleitner model parameters
# ============================================================
mu_max = 0.6        # 1/h
K_S    = 0.1        # g/L
Y_XS   = 0.5        # gX / gS
q_s_max = 1.0       # gS / (gX h)
alpha  = 0.8
S_in   = 20.0       # g/L

dt = 0.01           # h
T  = 150            # h
t  = np.arange(0, T, dt)

# Constant dilution rate (open-loop!)
D = 0.2 * np.ones_like(t)

# ============================================================
# Initial conditions
# ============================================================
X0 = 0.1   # biomass [g/L]
S0 = 20.0  # substrate [g/L]
P0 = 0.0   # ethanol [g/L]

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

# ============================================================
# Simulation loop
# ============================================================
X, S, P = X0, S0, P0

Xs, Ss, Ps = [], [], []

for k in range(len(t)):
    dx, ds, dp = sonnleitner_step(X, S, P, D[k])

    X += dt * dx
    S += dt * ds
    P += dt * dp

    Xs.append(X)
    Ss.append(S)
    Ps.append(P)

Xs = np.array(Xs)
Ss = np.array(Ss)
Ps = np.array(Ps)

# ============================================================
# Plot results
# ============================================================
plot_signals(
    t,
    [Xs, Ss, Ps],
    labels=["Biomass X [g/L]", "Substrate S [g/L]", "Ethanol P [g/L]"],
    title="Sonnleitner Yeast Fermentation (Open-Loop Simulation)",
    dirname="sonnleitner_open_loop",
    filename="sonnleitner_open_loop_state_plot.png",
)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, Xs)
plt.ylabel("Biomass X [g/L]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, Ss)
plt.ylabel("Substrate S [g/L]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, Ps)
plt.ylabel("Ethanol P [g/L]")
plt.xlabel("Time [h]")
plt.grid(True)

plt.tight_layout()
plt.show()