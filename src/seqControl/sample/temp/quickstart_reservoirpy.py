import os
import sys
from seqControl.sample.utils.plotting_utils import *
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import numpy as np
import pandas as pd



from reservoirpy.nodes import Reservoir

# Create a reservoir with 100 units, spectral radius of 0.9, and leaking rate of 0.5
# lr (leaking rate) controls the speed of the reservoir's state update, with 1.0 meaning no leaking (full update) and values < 1.0 introducing a memory effect.
# sr (spectral radius) controls the echo state property, with values < 1.0 ensuring stability and values close to 1.0 allowing for richer dynamics.
reservoir = Reservoir(100, 
                      lr=0.5, 
                      sr=0.9)

X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
plot_signals(np.arange(100), [X.flatten()], filename="sinewave", dirname="quickstart_reservoirpy")

# Calliong on a single timestep

s = reservoir(X[0])
print("Reservoir state at first timestep:", s)
print("New state vector shape:", s.shape)

# the node internal state can be accessed in the state dictionary
s = reservoir.state["out"]

states = np.empty((len(X), reservoir.output_dim))
for i in range(len(X)):
    states[i] = reservoir(X[i])

plot_signals(
    np.arange(states.shape[0]),
    [states[:, :4]],
    title="Activation of 20 reservoir neurons.",
    xlabel="$t$",
    ylabel="$reservoir(sin(t))$",
    figsize=(10, 3),
    filename="activation_of_20_reservoir_neurons", 
    dirname="quickstart_reservoirpy"
)

# Gathering the activations of a node over a timeseries can be done without using a for-loop with the run() method. This method takes arrays of shape (timesteps, features) as input and returns the corresponding reservoir states for each timestep. The run() method is optimized for performance and is the recommended way to process time series data through a reservoir node.
states = reservoir.run(X)

# A node state can then be reset to a null vector to wash out its internal memory using the reset() method.
_ = reservoir.reset()




from reservoirpy.nodes import Ridge

# Creating a Readout
readout = Ridge(ridge=1e-7)

X_train = X[:50]
Y_train = X[1:51]
plot_signals(
np.arange(len(X_train)),
[X_train, Y_train],
labels=["sin(t)", "sin(t+1)"],
title="A sine wave and its future.",
xlabel="t",
filename="a_sine_wave_and_its_future", 
dirname="quickstart_reservoirpy"
)
# we train the readout node as a standalone node

# compute the train states using the run method of the reservoir node
train_states = reservoir.run(X_train)

# We can then fit the readout to learn the mapping from the reservoir states to the target output Y_train. The warmup parameter specifies how many initial timesteps to discard from the training data to allow the reservoir to wash out its initial state and stabilize its dynamics.
readout = readout.fit(train_states, Y_train, warmup=10)

test_states = reservoir.run(X[50:])
Y_pred = readout.run(test_states)

min_len = min(len(Y_pred), len(X[51:]))
t = np.arange(min_len)
pred = Y_pred.ravel()[:min_len]
real = X[51:51 + min_len].ravel()

plot_signals(
    t,
    [pred, real],
    labels=["Predicted sin(t)", "Real sin(t+1)"],
    title="A sine wave and its future.",
    xlabel="$t$",
    figsize=(10, 3),
    filename="a_sine_wave_and_its_future_prediction", 
    dirname="quickstart_reservoirpy",
)

from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)

# Connect the reservoir to the readout to create an ESN model
esn_model = reservoir >> readout

# Train the ESN model on the training data
esn_model = esn_model.fit(X_train, Y_train, warmup=10)

print(reservoir.initialized, readout.initialized)



Y_pred = esn_model.run(X[50:])

min_len = min(len(Y_pred.ravel()), len(X[51:]))
t = np.arange(min_len)
pred = Y_pred.ravel()[:min_len]
real = X[51:51 + min_len].ravel()

plot_signals(
    t,
    [pred, real],
    labels=["Predicted sin(t)", "Real sin(t+1)"],
    title="A sine wave and its future.",
    xlabel="$t$",
    figsize=(10, 3),
    filename="a_sine_wave_and_its_future_prediction_esn",
    dirname="quickstart_reservoirpy",
)


exit()


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
