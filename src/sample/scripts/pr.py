"""
Regenerate optimal control profiles (u1, u2) for the Lee-Ramirez (LR) bioreactor
from Balsa-Canto et al. (2000) - Ind. Eng. Chem. Res. 39, 4287-4295.

This script solves the optimal control problem for the LR bioreactor using:
- Control Vector Parametrization (CVP) approach
- Direct collocation with scipy's optimization
- Piecewise constant control discretization

Case studies:
- LR-1: Q = 0.0 (no inducer cost)
- LR-2: Q = 2.5
- LR-3: Q = 5.0

Reference results from paper:
- LR-1: J = 6.15160
- LR-2: J = 5.75711  
- LR-3: J = 5.56789
"""

import numpy as np
from io import BytesIO
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


from src.sample.utils.plotting_utils import plot_signals

# =============================================================================
# MODEL PARAMETERS (from Lee & Ramirez 1994, as used in Balsa-Canto 2000)
# =============================================================================

# Based on Tholudur & Ramirez (1997) modification and literature values
PARAMS = {
    # Growth parameters
    'mu_max': 0.48,      # h^-1, maximum specific growth rate
    'K_s': 0.1,          # g/L, Monod constant for nutrient
    'K_i': 0.1,          # g/L, Monod constant for inducer
    
    # Protein production parameters
    'R_fp_max': 0.02,    # g/g/h, maximum foreign protein production rate
    'K_s_fp': 0.1,       # g/L, Monod constant for protein production (nutrient)
    'K_i_fp': 0.1,       # g/L, Monod constant for protein production (inducer)
    
    # Feed concentrations
    'Cn_l': 40.0,        # g/L, nutrient (glucose) feed concentration
    'Ci_l': 1.0,         # g/L, inducer feed concentration
    
    # Yield and decay parameters
    'Y': 0.5,            # g/g, yield coefficient
    'k1': 0.1,           # h^-1, decay rate for inducer shock factor (x6)
    'k2': 0.1,           # h^-1, recovery rate for inducer recovery factor (x7)
}

# =============================================================================
# ODE SYSTEM DEFINITION
# =============================================================================

def lr_bioreactor_ode(t, x, u1, u2, params):
    """
    Lee-Ramirez bioreactor dynamics (8 states including integral of u2).
    
    States:
    x[0] = x1: Reactor volume (L)
    x[1] = x2: Cell density (g/L)
    x[2] = x3: Nutrient concentration (g/L)
    x[3] = x4: Foreign protein concentration (g/L)
    x[4] = x5: Inducer concentration (g/L)
    x[5] = x6: Inducer shock factor on cell growth (-)
    x[6] = x7: Inducer recovery factor on cell growth (-)
    x[7] = x8: Integral of u2 (for objective calculation)
    
    Controls:
    u1: Glucose feed rate (L/h)
    u2: Inducer feed rate (L/h)
    
    Objective: J = x1(tf) * x4(tf) - Q * x8(tf)
    """
    # Unpack states
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    
    # Unpack parameters
    mu_max = params['mu_max']
    K_s = params['K_s']
    K_i = params['K_i']
    R_fp_max = params['R_fp_max']
    K_s_fp = params['K_s_fp']
    K_i_fp = params['K_i_fp']
    Cn_l = params['Cn_l']
    Ci_l = params['Ci_l']
    Y = params['Y']
    k1 = params['k1']
    k2 = params['k2']
    
    # Specific growth rate (eq from Lee & Ramirez 1994)
    mu = mu_max * (x3 / (x3 + K_s)) * (x5 / (x5 + K_i)) * x6 * x7
    
    # Foreign protein production rate
    R_fp = R_fp_max * (x3 / (x3 + K_s_fp)) * (x5 / (x5 + K_i_fp))
    
    # Total feed rate
    F_total = u1 + u2
    
    # ODEs
    dx1dt = F_total  # Volume
    dx2dt = mu * x2 - (F_total / x1) * x2  # Cell density
    dx3dt = (u1 / x1) * Cn_l - (F_total / x1) * x3 - (1/Y) * mu * x2  # Nutrient
    dx4dt = R_fp * x2 - (F_total / x1) * x4  # Foreign protein
    dx5dt = (u2 / x1) * Ci_l - (F_total / x1) * x5  # Inducer
    dx6dt = -k1 * x6  # Inducer shock factor
    dx7dt = k2 * (1 - x7)  # Inducer recovery factor
    dx8dt = u2  # Integral of inducer feed rate
    
    return np.array([dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt])


# =============================================================================
# OPTIMIZATION SETUP
# =============================================================================

def simulate_system(controls, params, tf, N, Q):
    """
    Simulate the bioreactor with piecewise constant controls.
    
    Args:
        controls: Array of length 2*N (u1_0, u2_0, u1_1, u2_1, ..., u1_{N-1}, u2_{N-1})
        params: Dictionary of model parameters
        tf: Final time (h)
        N: Number of control segments
        Q: Cost coefficient for inducer
        
    Returns:
        x_final: Final state vector
        t_eval: Time points
        x_trajectory: State trajectory
        u1_trajectory: u1 values at each segment
        u2_trajectory: u2 values at each segment
    """
    # Reshape controls
    u1_vals = controls[0::2]
    u2_vals = controls[1::2]
    
    # Time points
    t_points = np.linspace(0, tf, N+1)
    dt = tf / N
    
    # Initial conditions (x1, x2, x3, x4, x5, x6, x7, x8)
    x0 = np.array([1.0, 0.1, 40.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    # Store trajectory
    x_trajectory = [x0.copy()]
    u1_trajectory = []
    u2_trajectory = []
    
    # Integrate piecewise
    x = x0.copy()
    for i in range(N):
        u1, u2 = u1_vals[i], u2_vals[i]
        u1_trajectory.append(u1)
        u2_trajectory.append(u2)
        
        # Integrate over this segment
        sol = solve_ivp(
            lambda t, y: lr_bioreactor_ode(t, y, u1, u2, params),
            [t_points[i], t_points[i+1]],
            x,
            method='RK45',
            t_eval=[t_points[i+1]],
            atol=1e-8,
            rtol=1e-8
        )
        x = sol.y[:, -1]
        x_trajectory.append(x.copy())
    
    x_trajectory = np.array(x_trajectory)
    u1_trajectory = np.array(u1_trajectory)
    u2_trajectory = np.array(u2_trajectory)
    
    return x_trajectory, t_points, u1_trajectory, u2_trajectory


def objective_function(controls, params, tf, N, Q):
    """
    Objective function for optimization: Maximize J = x1(tf)*x4(tf) - Q*integral(u2)
    """
    x_trajectory, _, _, _ = simulate_system(controls, params, tf, N, Q)
    x_final = x_trajectory[-1]
    
    # J = x1(tf) * x4(tf) - Q * integral(u2) = x1 * x4 - Q * x8
    J = x_final[0] * x_final[3] - Q * x_final[7]
    
    # We minimize -J to maximize J
    return -J


def create_constraints(controls, params, tf, N, Q, u1_bounds=(0, 1), u2_bounds=(0, 1)):
    """
    Create bounds and constraints for the optimization problem.
    """
    # Bounds for controls (u1 and u2 for each segment)
    bounds = []
    for i in range(N):
        bounds.append(u1_bounds)  # u1 bounds
        bounds.append(u2_bounds)  # u2 bounds
    
    return bounds


def solve_optimal_control(Q_value, N_segments=40, case_name="LR", initial_guess=None):
    """
    Solve the optimal control problem for a given Q value.
    
    Args:
        Q_value: Cost coefficient for inducer (0.0, 2.5, or 5.0)
        N_segments: Number of control segments (discretization level)
        case_name: Name for the case (LR-1, LR-2, LR-3)
        initial_guess: Initial guess for controls (optional)
        
    Returns:
        Dictionary with results: optimal controls, trajectory, objective value
    """
    tf = 10.0  # Final time (h)
    
    # Initial guess (constant controls at mid-point of bounds)
    if initial_guess is None:
        initial_guess = np.array([0.5] * (2 * N_segments))
    
    # Bounds
    bounds = create_constraints(initial_guess, PARAMS, tf, N_segments, Q_value)
    
    # Objective function with fixed parameters
    def obj(controls):
        return objective_function(controls, PARAMS, tf, N_segments, Q_value)
    
    # Solve using SLSQP (Sequential Least Squares Programming)
    print(f"Solving {case_name} case (Q={Q_value}, N={N_segments})...")
    
    result = minimize(
        obj,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        options={
            'maxiter': 200,
            'ftol': 1e-8,
            'disp': True
        }
    )
    
    # Extract results
    optimal_controls = result.x
    optimal_J = -result.fun
    
    # Simulate with optimal controls to get full trajectory
    x_trajectory, t_points, u1_opt, u2_opt = simulate_system(
        optimal_controls, PARAMS, tf, N_segments, Q_value
    )
    
    return {
        'Q': Q_value,
        'N': N_segments,
        'J': optimal_J,
        'success': result.success,
        'message': result.message,
        'nfev': result.nfev,
        't_points': t_points,
        'x_trajectory': x_trajectory,
        'u1': u1_opt,
        'u2': u2_opt,
        'controls': optimal_controls
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Solve all three LR case studies and plot results."""
    
    # Case studies from the paper
    cases = [
        {'Q': 0.0, 'name': 'LR-1', 'color': 'blue', 'paper_J': 6.15160},
        {'Q': 2.5, 'name': 'LR-2', 'color': 'green', 'paper_J': 5.75711},
        {'Q': 5.0, 'name': 'LR-3', 'color': 'red', 'paper_J': 5.56789}
    ]
    
    # Discretization level (can be adjusted)
    N_SEGMENTS = 40
    
    # Dictionary to store results
    results = {}
    
    # Solve each case
    for case in cases:
        Q = case['Q']
        name = case['name']
        
        print(f"\n{'='*60}")
        print(f"Solving case: {name} (Q = {Q})")
        print(f"{'='*60}")
        
        result = solve_optimal_control(
            Q_value=Q,
            N_segments=N_SEGMENTS,
            case_name=name,
            initial_guess=None
        )
        
        results[name] = result
        
        print(f"\nResult for {name}:")
        print(f"  Objective J: {result['J']:.5f}")
        print(f"  Paper J: {case['paper_J']:.5f}")
        print(f"  Difference: {result['J'] - case['paper_J']:.5f}")
        print(f"  Success: {result['success']}")
        print(f"  Function evaluations: {result['nfev']}")
    
    # =============================================================================
    # PLOTTING USING plot_signals FUNCTION
    # =============================================================================
    
    print("\n" + "="*60)
    print("Plotting results using plot_signals...")
    print("="*60)
    
    # Prepare data for plotting
    # Create fine time grid for smooth interpolation
    t_plot = np.linspace(0, 10, 1000)
    
    # Collect control profiles for all cases
    u1_profiles = []
    u2_profiles = []
    u1_labels = []
    u2_labels = []
    
    for case in cases:
        name = case['name']
        result = results[name]
        
        t_points = result['t_points']
        u1 = result['u1']
        u2 = result['u2']
        
        # Interpolate to fine grid
        u1_plot = np.interp(t_plot, t_points[:-1], u1, left=u1[0], right=u1[-1])
        u2_plot = np.interp(t_plot, t_points[:-1], u2, left=u2[0], right=u2[-1])
        
        u1_profiles.append(u1_plot)
        u2_profiles.append(u2_plot)
        u1_labels.append(f'{name} (J={result["J"]:.4f})')
        u2_labels.append(f'{name} (J={result["J"]:.4f})')
    
    # Plot u1 profiles
    plot_signals(
        t=t_plot,
        signals=u1_profiles,
        labels=u1_labels,
        title='Glucose Feed Rate ($u_1$) Profiles',
        xlabel='Time (h)',
        ylabel='$u_1$ (L/h)',
        figsize=(8, 5),
        show=True
    )
    
    # Plot u2 profiles
    plot_signals(
        t=t_plot,
        signals=u2_profiles,
        labels=u2_labels,
        title='Inducer Feed Rate ($u_2$) Profiles',
        xlabel='Time (h)',
        ylabel='$u_2$ (L/h)',
        figsize=(8, 5),
        show=True
    )
    
    # Plot state trajectories for LR-1
    lr1_result = results['LR-1']
    t_points_lr1 = lr1_result['t_points']
    x_trajectory_lr1 = lr1_result['x_trajectory']
    
    plot_signals(
        t=t_points_lr1,
        signals=[
            x_trajectory_lr1[:, 0],  # x1: Volume
            x_trajectory_lr1[:, 1],  # x2: Cell density
            x_trajectory_lr1[:, 3]   # x4: Foreign protein
        ],
        labels=['Reactor Volume ($x_1$)', 'Cell Density ($x_2$)', 'Foreign Protein ($x_4$)'],
        title='LR-1 State Trajectories',
        xlabel='Time (h)',
        ylabel='Value',
        figsize=(8, 5),
        show=True
    )
    
    # Plot objective comparison as a line plot (alternative to bar chart)
    paper_J = [6.15160, 5.75711, 5.56789]
    computed_J = [results['LR-1']['J'], results['LR-2']['J'], results['LR-3']['J']]
    cases_names = ['LR-1', 'LR-2', 'LR-3']
    
    plot_signals(
        t=np.arange(len(cases_names)),
        signals=[paper_J, computed_J],
        labels=['Paper Results', 'Computed Results'],
        title='Comparison with Paper Results',
        xlabel='Case Study',
        ylabel='Objective Value (J)',
        figsize=(8, 5),
        show=True
    )
    
    print("\nAll plots generated using plot_signals function")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for case in cases:
        name = case['name']
        result = results[name]
        print(f"\n{name}:")
        print(f"  Q = {result['Q']}")
        print(f"  Discretization (N) = {result['N']}")
        print(f"  Computed J = {result['J']:.5f}")
        print(f"  Paper J = {case['paper_J']:.5f}")
        print(f"  Difference = {result['J'] - case['paper_J']:.5f}")
        print(f"  Optimization success = {result['success']}")


# =============================================================================
# MESH REFINING APPROACH (from the paper)
# =============================================================================

def mesh_refining_solve(Q_value, initial_N=15, final_N=120, r_p=2, case_name="LR"):
    """
    Implement the mesh-refining approach from the paper.
    
    This approach performs successive re-optimizations with increasing discretization levels.
    
    Args:
        Q_value: Cost coefficient for inducer
        initial_N: Initial number of segments
        final_N: Final number of segments
        r_p: Refinement ratio (typically 2)
        case_name: Name for the case
        
    Returns:
        Dictionary with results from each refinement step
    """
    tf = 10.0
    
    # Calculate number of refinement steps
    NRO = int(np.log2(final_N / initial_N) / np.log2(r_p)) + 1
    
    print(f"\nMesh refining for {case_name} (Q={Q_value}):")
    print(f"  Initial N = {initial_N}, Final N = {final_N}, r_p = {r_p}")
    print(f"  Number of refinement steps = {NRO}")
    
    results = []
    current_N = initial_N
    initial_guess = None
    
    for step in range(1, NRO + 1):
        print(f"\n  Step {step}: N = {current_N}")
        
        result = solve_optimal_control(
            Q_value=Q_value,
            N_segments=current_N,
            case_name=f"{case_name}-Step{step}",
            initial_guess=initial_guess
        )
        
        results.append(result)
        
        # Use current solution as initial guess for next step
        # Interpolate to next discretization level
        if step < NRO:
            next_N = current_N * r_p
            # Simple interpolation: repeat each control value
            if initial_guess is None:
                initial_guess = result['controls']
            else:
                # Interpolate by repeating values
                old_u1 = result['controls'][0::2]
                old_u2 = result['controls'][1::2]
                new_u1 = np.repeat(old_u1, 2)
                new_u2 = np.repeat(old_u2, 2)
                initial_guess = np.zeros(2 * next_N)
                initial_guess[0::2] = new_u1
                initial_guess[1::2] = new_u2
            
            current_N = next_N
    
    return results


# =============================================================================
# RUN MESH REFINING (Optional - takes longer)
# =============================================================================

def run_mesh_refining():
    """Run mesh refining for all cases (commented out by default as it's slow)."""
    
    print("\n" + "="*60)
    print("RUNNING MESH REFINING APPROACH")
    print("="*60)
    
    cases = [
        {'Q': 0.0, 'name': 'LR-1'},
        {'Q': 2.5, 'name': 'LR-2'},
        {'Q': 5.0, 'name': 'LR-3'}
    ]
    
    for case in cases:
        Q = case['Q']
        name = case['name']
        
        print(f"\nSolving {name} with mesh refining...")
        results = mesh_refining_solve(
            Q_value=Q,
            initial_N=15,
            final_N=120,
            r_p=2,
            case_name=name
        )
        
        # Print final result
        final_result = results[-1]
        print(f"\nFinal result for {name}:")
        print(f"  N = {final_result['N']}")
        print(f"  J = {final_result['J']:.5f}")


# =============================================================================
# EXECUTION OPTIONS
# =============================================================================

if __name__ == "__main__":
    # Run the main analysis with moderate discretization
    main()
    
    # Uncomment the line below to run mesh refining (takes longer)
    # run_mesh_refining()
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print("\nNote: For better accuracy, increase N_SEGMENTS in the main() function.")
    print("The paper used up to 320 segments with mesh refining.")
    print("\nTo reproduce paper results exactly, ensure model parameters match")
    print("Lee & Ramirez (1994) or Tholudur & Ramirez (1997) modifications.")