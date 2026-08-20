import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 1. Define Model Parameters
mu_max = 0.5   # Max growth rate
Ks = 0.2       # Half-saturation constant
Y = 0.6        # Yield coefficient
sR = 1.0       # Input substrate concentration

# Monod growth function
def mu(s):
    return (mu_max * s) / (Ks + s)

# 2. Define the System of ODEs (at steady state, these equal 0)
def chemostat(vars, D):
    x, s = vars
    dxdt = mu(s) * x - D * x
    dsdt = D * (sR - s) - (mu(s) * x) / Y
    return [dxdt, dsdt]

# 3. Perform Bifurcation Analysis
# We vary D from 0 to slightly above mu_max
D_values = np.linspace(0.01, 0.6, 100)
biomass_results = []
substrate_results = []

# Initial guesses for the two possible states:
# 1. Stable growth state
# 2. Washout state (x=0)
guess_growth = [0.5, 0.1] 

for D in D_values:
    # Solve for steady state
    sol = fsolve(chemostat, guess_growth, args=(D))
    biomass_results.append(sol[0])
    substrate_results.append(sol[1])
    # Update guess to follow the branch (continuation)
    guess_growth = sol

# 4. Plot the Bifurcation Diagram
from seqControl.sample.utils.plotting_utils import plot_signals

plot_signals(D_values, 
             [biomass_results], 
             labels=['Biomass (x)'], 
             xlabel='Dilution Rate (D) [1/h]', 
             ylabel='Steady State Concentration', 
             filename='bifurcation_diagram.png', 
             dirname='results')
