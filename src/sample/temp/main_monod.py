import os
import sys
from utils.plotting_utils import *
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import numpy as np
import pandas as pd

# ============================
#  BIOPROCESS PARAMETERS
# ============================
mu_max = 0.8     # 1/h
Ks     = 0.1     # g/L
kd     = 0.05    # 1/h

def simulate_monod(u, dt=0.01):
    """
    Nonlinear biological system:
        dx/dt = mu_max * u/(Ks+u) * x - kd * x
    """
    x = 0.1   # initial biomass
    xs = []

    for ui in u:
        mu = mu_max * ui / (Ks + ui)
        dx = mu * x - kd * x
        x = x + dt * dx
        xs.append(x)

    return np.array(xs)


# ============================
#  TIME + DESIRED TRAJECTORY
# ============================
T = 10
dt = 0.01
t = np.arange(0, T, dt)

# Desired biomass trajectory
x_desired = 0.5 + 0.2 * np.sin(2 * np.pi * 0.2 * t)
dx_desired = np.gradient(x_desired, dt)

plot_signals(t, [x_desired], filename="desired_biomass", dirname="test")


# ============================
#  ANALYTICAL INVERSE CONTROL
# ============================
# u = Ks*(dx + kd*x) / (mu_max*x - (dx + kd*x))

num = Ks * (dx_desired + kd * x_desired)
den = mu_max * x_desired - (dx_desired + kd * x_desired)

# Avoid division by zero
den = np.where(np.abs(den) < 1e-6, 1e-6, den)

u_inverse = num / den

# Clip negative feed rates
u_inverse = np.clip(u_inverse, 0, None)

x_sim = simulate_monod(u_inverse, dt=dt)

plot_signals(t, [x_sim], filename="analytical_controlled_bio", dirname="test")


# ============================
#  PREPARE TRAINING DATA
# ============================
X = np.column_stack([x_desired, dx_desired])   # (N, 2)
Y = u_inverse.reshape(-1, 1)                   # (N, 1)

print("Any NaNs in X?", np.isnan(X).any())
print("Any NaNs in Y?", np.isnan(Y).any())


# ============================
#  RESERVOIR COMPUTING (ESN)
# ============================
print("Start RC")

from reservoirpy.nodes import Reservoir, Ridge

res = Reservoir(
    units=300,
    sr=0.9,
    lr=1.0,
    input_scaling=1.0,
    seed=42
)

readout = Ridge(ridge=1e-6)
esn = res >> readout

# Train ESN
esn = esn.fit(X, Y, warmup=100)

# Predict inverse control
u_esn = esn.run(X)

# Simulate biological system
x_esn = simulate_monod(u_esn, dt=dt)

if x_esn.ndim == 1:
    x_esn = x_esn.reshape(-1, 1)


# ============================
#  PLOT TRACKING PERFORMANCE
# ============================
plot_signals(
    t,
    [x_desired, x_esn],
    labels=["Desired", "ESN tracking"],
    title="Monod Bioprocess Tracking with ESN",
    filename="esn_monod_tracking",
    dirname="test"
)
