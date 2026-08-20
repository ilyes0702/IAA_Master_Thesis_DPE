import os
import sys
from seqControl.sample.utils.plotting_utils import *
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

import numpy as np
import pandas as pd
import numpy as np

# 1. Parameters
mu_max = 0.4      # 1/h
K_s    = 0.1
Y_xs   = 0.5
S_in   = 10.0
dt     = 0.1      # h

params = (mu_max, K_s, Y_xs, S_in)

D_min, D_max = 0.05, 0.5

T = 100000          # time steps per trajectory
N_traj = 100      # number of trajectories


def chemostat_step(X, S, D, params, dt):
    mu_max, K_s, Y_xs, S_in = params
    mu = mu_max * S / (K_s + S + 1e-8)
    dX = (mu - D) * X
    dS = D * (S_in - S) - (1.0 / Y_xs) * mu * X
    X_next = X + dt * dX
    S_next = S + dt * dS
    # avoid negative concentrations
    X_next = max(X_next, 0.0)
    S_next = max(S_next, 0.0)
    return X_next, S_next


def generate_prbs(T, D_min, D_max, min_hold=5, max_hold=20, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    D_seq = np.zeros(T)
    k = 0
    while k < T:
        D_val = rng.uniform(D_min, D_max)
        hold = rng.integers(min_hold, max_hold + 1)
        end = min(T, k + hold)
        D_seq[k:end] = D_val
        k = end
    return D_seq

def random_piecewise_constant(T, D_min, D_max, min_hold=5, max_hold=30, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    D_seq = np.zeros(T)
    k = 0
    while k < T:
        D_val = rng.uniform(D_min, D_max)
        hold = rng.integers(min_hold, max_hold + 1)
        end = min(T, k + hold)
        D_seq[k:end] = D_val
        k = end
    return D_seq


def step_sequence(T, D_low, D_high, step_times):
    """
    step_times: list of k where you switch between low/high
    """
    D_seq = np.full(T, D_low)
    high = False
    for t in step_times:
        high = not high
        D_seq[t:] = D_high if high else D_low
    return D_seq


def ramp_sequence(T, D_start, D_end):
    return np.linspace(D_start, D_end, T)


def noisy_baseline(T, D_mean, D_amp, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, D_amp, size=T)
    D_seq = D_mean + noise
    return np.clip(D_seq, 0.0, None)


def simulate_trajectory(T, params, dt, D_seq, X0, S0):
    X = np.zeros(T + 1)
    S = np.zeros(T + 1)
    X[0], S[0] = X0, S0
    for k in range(T):
        X[k+1], S[k+1] = chemostat_step(X[k], S[k], D_seq[k], params, dt)
    return X, S


# 5. Generate dataset
rng = np.random.default_rng(42)

all_D = []
all_X = []
all_S = []

for n in range(N_traj):
    # sample initial conditions
    X0 = rng.uniform(0.1, 2.0)   # biomass
    S0 = rng.uniform(0.1, 10.0)  # substrate

    # generate input sequence
    #D_seq = generate_prbs(T, D_min, D_max, rng=rng)
    #D_seq = random_piecewise_constant(T, D_min, D_max, rng=rng)
    #D_seq = step_sequence(T, D_low=0.1, D_high=0.4, step_times=[100, 300, 600, 800])
    #D_seq = ramp_sequence(T, D_start=0.05, D_end=0.5)
    D_seq = noisy_baseline(T, D_mean=0.25, D_amp=0.1, rng=rng)
    # simulate
    X_seq, S_seq = simulate_trajectory(T, params, dt, D_seq, X0, S0)

    all_D.append(D_seq)
    all_X.append(X_seq)
    all_S.append(S_seq)

all_D = np.array(all_D)  # shape: (N_traj, T)
all_X = np.array(all_X)  # shape: (N_traj, T+1)
all_S = np.array(all_S)  # shape: (N_traj, T+1)

plot_signals(np.arange(T+1)*dt, [all_X[0], all_S[0]], filename="example_trajectory", dirname="test")

print("D shape:", all_D.shape)
print("X shape:", all_X.shape)
print("S shape:", all_S.shape)
