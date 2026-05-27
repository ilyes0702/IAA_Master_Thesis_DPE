# Import standard libraries
import os
import numpy as np
import pandas as pd

# import machine learning modules
from src.sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import torch
from src.sample.classes.MambaInverseController import MambaInverseController, MambaInverseController
import matplotlib.pyplot as plt
import random

plt.style.use("src/sample/style.mplstyle")


import pickle

def load_scaler(scaler_dir):
    """
    Loads the fitted scaler_x and scaler_y for a specific fold.
    
    Args:
        fold_dir (str): Path to the specific fold directory (e.g., 'name_directory/fold_1')
    Returns:
        scaler_x, scaler_y: The deserialized scikit-learn StandardScaler objects
    """
    with open(scaler_dir, "rb") as f:
        scaler = pickle.load(f)
        
        
    print(f"✅ Successfully loaded scalers from {scaler_dir}")
    return scaler

#=== FUNCTION TO COMPUTE AND SAVE TRACKING METRICS ===#
def compute_and_save_tracking_metrics(
    y_np,        # Actual output [steps, batch_size]
    ref_np,      # Reference trajectory [steps, batch_size]
    dt,
    dirname,
    settle_tol=0.05, # Band for "tracking error"
):
    """
    Pure tracking metrics for curve comparison.
    Calculates how well the plant output follows a dynamic reference.

    Parameters:
    - y_np (numpy.ndarray): Actual plant output values, shaped [steps, batch_size].
    - ref_np (numpy.ndarray or float): Reference trajectory targets to match.
    - dt (float): Sampling time increment between steps.
    - dirname (str): Directory pathway where metrics reports will be saved.
    - settle_tol (float): Tolerance threshold scale defining acceptable error bands (default: 0.05).

    Returns:
    - df (pandas.DataFrame): Evaluated performance data broken down for every individual trajectory sequence.

    The function standardizes dimensions between the plant output configurations and the reference target 
    signals. It executes transient step error calculations to extract statistical benchmarks including 
    Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Integral Absolute 
    Error (IAE), and Integral Square Error (ISE). Additionally, it isolates transient maximum errors and 
    determines the overall percentage of processing duration spent within the defined settling tolerance band, 
    writing raw and summary analytical files to disk.
    """
    steps, batch_size = y_np.shape
    
    # Ensure ref_np is the same shape as y_np
    if np.isscalar(ref_np):
        ref_np = np.full_like(y_np, ref_np)
    elif ref_np.ndim == 1:
        ref_np = np.tile(ref_np[:, np.newaxis], (1, batch_size))

    error = y_np - ref_np
    abs_error = np.abs(error)

    # --- Integral Metrics (Total Tracking Performance) ---
    mae = abs_error.mean(axis=0)
    mse = (error ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    iae = abs_error.sum(axis=0) * dt
    ise = (error ** 2).sum(axis=0) * dt

    # --- Dynamic Tracking Metrics ---
    # Max Tracking Error: The single biggest deviation during the run
    max_error = np.max(abs_error, axis=0)
    
    # Time spent within tolerance band (%)
    # How much of the sequence was the error < 5% of the reference?
    denom = np.abs(ref_np) + 1e-8
    within_band = abs_error <= (settle_tol * denom)
    time_in_band_pct = (np.sum(within_band, axis=0) / steps) * 100

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "trajectory": np.arange(batch_size),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "IAE": iae,
        "ISE": ise,
        "Max_Error": max_error,
        "TimeInBand_%": time_in_band_pct
    })

    # Summary Statistics
    summary_df = df.describe().loc[['mean', 'std', 'min', 'max']].T.reset_index()
    summary_df.columns = ['metric', 'mean', 'std', 'min', 'max']

    # --- Save ---
    save_df_to_csv(df, dirname, "tracking_metrics_per_trajectory")
    save_df_to_csv(summary_df, dirname, "tracking_metrics_summary")

    return df



#=== FUNCTION TO GENERATE REFERENCE TRAJECTORY ===#
def generate_reference_trajectory(steps, dt, device, mode="constant", constant_val=0.3, gain=1.0, period=20.0):
    """
    Generates reference target trajectories for physical control system tracking simulations.

    Parameters:
    - steps (int): Total sequence length of the reference trajectory.
    - dt (float): Sampling time increment between sequential steps.
    - device (str/torch.device): Target execution hardware device mapping for the output tensor.
    - mode (str): Style of reference generated; choices are "constant" or "dynamic" (default: "constant").
    - constant_val (float): Fixed tracking setpoint used if mode is "constant" (default: 0.3).
    - gain (float): Tuning scaling modifier to sharpen tracking switches if mode is "dynamic" (default: 1.0).
    - period (float): Cyclical hour sequence timeframe for oscillations if mode is "dynamic" (default: 20.0).

    Returns:
    - r_trajectory (torch.Tensor): Reference target values mapped to the execution device, shaped [steps, 1].

    The function acts as an isolated generator for processing setpoints. Depending on the requested 
    mode, it outputs either a uniform static tensor filled with a base setpoint value or computes 
    a dynamic, bounded time-varying profile utilizing transcendental mathematical operators to yield 
    smooth rectangular transitions.
    """
    if mode == "constant":
        # Create a tensor of shape [steps, 1] filled with a static target value
        r_trajectory = torch.full((steps, 1), constant_val, device=device, dtype=torch.float32)
    
    elif mode == "dynamic":
        # Generate time steps array
        time_axis = np.arange(steps) * dt
        
        # Calculate a smooth time-varying waveform base
        sine_base = np.sin(2 * np.pi * time_axis / period)
        
        # Use tanh to sharpen the curve into a "soft" rectangle with an upward linear drift

        noise = np.random.uniform(-0.005, 0.005, size=time_axis.shape)

        r_trajectory_np = 0.25 + 0.02 * np.tanh(gain * sine_base) - 0.00 * time_axis   # Add small random noise for realism
        
        # Convert the structural numpy baseline into a target PyTorch tensor array
        r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)
    
    else:
        raise ValueError(f"Unknown reference mode selection: '{mode}'. Choose 'constant' or 'dynamic'.")

    return r_trajectory


#=== FUNCTION FOR THE SIMULATION OF CONTROLLED PLANT WITH TRACKING ===#
def simulate_tracking_old(model, plant, r_trajectory, hyperparam_config, dirname):
    """
    Simulates a controlled plant over a specified time horizon while tracking a reference trajectory.

    Parameters:
    - model: The neural network controller model used to generate control inputs.
    - plant: The simulation environment or physical plant object representing the controlled system.
    - r_trajectory (torch.Tensor): Target reference trajectory to follow, shaped [steps, 1].
    - hyperparam_config (dict): Configuration dictionary containing nested parameters for 'train', 'signal', and 'simulate'.
    - dirname (str): Base directory pathway where tracking reports, logs, and plots will be saved.

    Returns:
    - tuple: Returns an empty tuple upon successful execution and disk storage operations.

    The function configures a tracking simulation run across parallel trajectories. It processes an 
    externally supplied reference target trajectory, instantiates physical history trackers, and runs a forward 
    closed-loop control loop utilizing look-ahead prediction windows. At each time step, the 
    controller performs inference to update the plant state. The resulting historical metrics, 
    state variables, and control efforts are isolated per trajectory and written to disk as CSV 
    reports and comprehensive visual summaries.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]

    # Flatten reference tensor for CPU metrics calculations
    r_np = r_trajectory.cpu().numpy().flatten()
    
    # Initialize history trackers and GPU tensor buffers
    history = {"x1": np.zeros(steps), "x2": np.zeros(steps), "y": np.zeros(steps), "r": np.zeros(steps), "u": np.zeros(steps)}
    all_y = torch.zeros((steps, batch_size), device=device)
    all_u = torch.zeros((steps, batch_size), device=device)
    all_x1 = torch.zeros((steps, batch_size), device=device)
    all_x2 = torch.zeros((steps, batch_size), device=device)
    state = plant.get_initial_state(batch_size)

    # Prepare model for evaluation mode and reset hidden state memory
    model.eval()
    model.reset_memory(batch_size=batch_size, device=device)

    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories...")
    delta_steps = hyperparam_config["train"]["delay_steps"]  # For example, look 5 steps ahead (delta = 5 * dt)
    # Execute forward tracking simulation without tracking gradients
    with torch.no_grad():
        for i in range(steps - delta_steps):
            t = i * dt
            y_current = plant.get_y(state, t) # This is y(t)
            
            # 2. Grab the specific target for "Next" time step y(t+dt)
            current_r = r_trajectory[i].expand(batch_size, 1)

            # Grab the target steps into the future matching delta steps
            target_r = r_trajectory[i + delta_steps].expand(batch_size, 1)

            # Now the triplet [y_t, r_{t+delta_steps}] matches the training data configuration exactly
            current_input = torch.cat([y_current, target_r], dim=-1).unsqueeze(1)

            # 4. INFERENCE
            u_out = model(current_input, use_memory=True) 
            
            # Note: If using Mamba, u_out usually returns [batch, 1, output_dim]
            u = u_out[:, -1, :]
            # u = torch.clamp(u_out[:, -1, :], 0.0, plant.U_MAX)

            # 5. STEP PLANT
            # This uses the predicted U to move the REAL plant state forward
            state, _ = plant.step(state, u, t, dt)

            # --- LOGGING ---
            history["y"][i] = y_current[0].item()
            history["r"][i] = current_r[0].item() 
            history["u"][i] = u[0].item()
            history["x1"][i] = state[0, 0].item() # Assuming index 0 is biomass
            history["x2"][i] = state[0, 1].item() # Assuming index 1 is substrate
            
            all_y[i] = y_current.squeeze()
            all_u[i] = u.squeeze()
            all_x1[i] = state[:, 0]               # Assuming index 0 is biomass
            all_x2[i] = state[:, 1]               # Assuming index 1 is substrate

    # --- PLOTTING & EXPORT ---
    time_axis = np.arange(steps) * dt

    # Parse and save individual trajectory records
    for b in range(batch_size):
        # Define a unique sub-directory for this specific batch member
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        
        # Extract specific data for this trajectory (b)
        y_traj = all_y[:, b].cpu().numpy()
        u_traj = all_u[:, b].cpu().numpy()
        x1_traj = all_x1[:, b].cpu().numpy()
        x2_traj = all_x2[:, b].cpu().numpy()
        r_traj = np.full(steps, r_np)

        # SAVE THE CSV (using custom method)
        df_traj = pd.DataFrame({
            "time": time_axis,
            "biomass_x1": x1_traj,
            "substrate_x2": x2_traj,
            "growth_rate_y": y_traj,
            "control_u": u_traj,
            "target_r": r_traj
        })
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")

        # GENERATE THE 3 PLOTS PER BATCH
        # Plot 1: Control Signal (u)
        plot_signals(
            t=time_axis, 
            signals=[u_traj],
            labels=["Control Signal (u)"],
            title=f"Trajectory {b}: Control Action",
            xlabel="Time (h)", ylabel="Action Value",
            dirname=state_dirname, filename="plot_control_signal"
        )

        # Plot 2: Output (y) along with Reference (r)
        plot_signals(
            t=time_axis, 
            signals=[y_traj, r_traj],
            labels=["Growth Rate (y)", "Target (r)"],
            title=f"Trajectory {b}: Tracking Performance",
            xlabel="Time (h)", ylabel="Growth Rate",
            dirname=state_dirname, filename="plot_output_tracking"
        )

        # Plot 3: State Variables (x1, x2)
        plot_signals(
            t=time_axis, 
            signals=[x1_traj, x2_traj],
            labels=["Biomass (x1)", "Substrate (x2)"],
            title=f"Trajectory {b}: Plant States",
            xlabel="Time (h)", ylabel="Concentration",
            dirname=state_dirname, filename="plot_plant_states"
        )

    # Batch Summary Plot (All curves with random initial states aggregated)
    y_np = all_y.cpu().numpy()
    summary_signals = [y_np[:, j] for j in range(batch_size)]
    summary_signals.append(r_np)
    
    plot_signals(
        t=time_axis,
        signals=summary_signals,
        labels=[None]*(batch_size) + ["Target"],
        title=f"Batch Convergence ({batch_size} Trajectories)",
        xlabel="Time (h)",
        ylabel="Growth Rate (mu)",
        dirname=dirname,
        filename="batch_summary"
    )
    
    # Calculate overarching summary tracking stats
    compute_and_save_tracking_metrics(y_np, r_np, dt, dirname)

    return ()



import os
import numpy as np
import pandas as pd
import torch

import os
import numpy as np
import torch
import pandas as pd

import os
import numpy as np
import torch
import pandas as pd

def simulate_tracking(model, plant, r_trajectory, hyperparam_config, dirname):
    """
    Simulates a controlled plant over a specified time horizon while tracking a reference trajectory.
    Gracefully handles look-ahead clamping at boundary limits and leverages Mamba's recurrent cache.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]

    # Flatten reference tensor for CPU metrics calculations
    r_np = r_trajectory.cpu().numpy().flatten()
    
    # Initialize GPU tensor buffers across the COMPLETE step timeline
    all_y = torch.zeros((steps, batch_size), device=device)
    all_u = torch.zeros((steps, batch_size), device=device)
    all_x1 = torch.zeros((steps, batch_size), device=device)
    all_x2 = torch.zeros((steps, batch_size), device=device)
    
    state = plant.get_initial_state(batch_size)

    # Prepare model for evaluation mode and reset hidden state memory
    model.eval()
    model.reset_memory(batch_size=batch_size, device=device)

    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories across {steps} steps...")
    delta_steps = hyperparam_config["train"]["delay_steps"]
    
    # Execute forward tracking simulation without tracking gradients
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant.get_y(state, t) # Shape: (batch_size, 1)
            
            # Prevent index out of bounds using boundary clamping
            look_ahead_idx = min(i + delta_steps, steps - 1)
            target_r = r_trajectory[look_ahead_idx].expand(batch_size, 1) # Shape: (batch_size, 1)

            # Combined Feature Vector: [y(t), y(t + Delta)]
            # Dynamic Step Slicing Shape: (batch_size, 1, 2)
            current_input = torch.stack([y_current, target_r], dim=-1)

            # --- INFERENCE ---
            # Forward step passes through Mamba's active inference parameter states
            u_out = model(current_input, use_memory=True) 
            u = u_out[:, -1, :] # Extract active time-step control effort: (batch_size, 1)

            # --- STEP PLANT ---
            state, _ = plant.step(state, u, t, dt)

            # --- LOGGING (Safe assignment mapping across arbitrary batch sizes) ---
            all_y[i]  = y_current.view(-1)
            all_u[i]  = u.view(-1)
            all_x1[i] = state[:, 0]
            all_x2[i] = state[:, 1]

    # --- PLOTTING & EXPORT ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []

    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True) # Ensure path exists before logging
        
        # Extract specific data arrays for this trajectory (b)
        y_traj = all_y[:, b].cpu().numpy()
        u_traj = all_u[:, b].cpu().numpy()
        x1_traj = all_x1[:, b].cpu().numpy()
        x2_traj = all_x2[:, b].cpu().numpy()
        r_traj = r_np 

        # BUILD THE MATRIX DATAFRAME
        df_traj = pd.DataFrame({
            "time": time_axis,
            "state_x1": x1_traj,
            "state_x2": x2_traj,
            "output_y": y_traj,
            "control_u": u_traj,
            "target_r": r_traj
        })
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        # GENERATE TIME-SERIES PLOTS PER BATCH INDEX
        plot_signals(
            t=time_axis, 
            signals=[u_traj],
            labels=["Control Signal (u)"],
            title=f"Trajectory {b}: Control Action",
            xlabel="Time (h)", ylabel="Action Value",
            dirname=state_dirname, filename="plot_control_signal"
        )

        plot_signals(
            t=time_axis, 
            signals=[y_traj, r_traj],
            labels=["Output (y)", "Target (r)"],
            title=f"Trajectory {b}: Tracking Performance",
            xlabel="Time (h)", ylabel="Signal Value",
            dirname=state_dirname, filename="plot_output_tracking"
        )

        # GENERATE PARITY PLOTS PER BATCH INDEX (Sorted chronologically to prevent line crossings)
        sorted_indices = np.argsort(r_traj)
        plot_signals(
            t=r_traj[sorted_indices],
            signals=[y_traj[sorted_indices], r_traj[sorted_indices]],   
            labels=["Output (y)", "Ideal (y = r)"],
            title=f"Trajectory {b}: Parity Plot",
            xlabel="Reference (r)", ylabel="Output (y)",
            dirname=state_dirname, filename="parity_plot_output_tracking"
        )

        plot_signals(
            t=time_axis, 
            signals=[x1_traj, x2_traj],
            labels=["State x1", "State x2"],
            title=f"Trajectory {b}: Internal Plant States",
            xlabel="Time (h)", ylabel="State Magnitude",
            dirname=state_dirname, filename="plot_plant_states"
        )

    # --- GLOBAL BATCH OVERLAY PLOT GENERATION ---
    y_np = all_y.cpu().numpy()
    summary_signals = [y_np[:, j] for j in range(batch_size)]
    sum_w_ref = summary_signals + [r_np]
    
    plot_signals(
        t=time_axis,
        signals=sum_w_ref,
        labels=[f"Traj {j}" for j in range(batch_size)] + ["Target Reference"],
        title=f"Batch Convergence ({batch_size} Trajectories Overview)",
        xlabel="Time (h)", ylabel="System Output (y)",
        dirname=dirname, filename="batch_summary"
    )

    # GLOBAL BATCH PARITY MAP GENERATION
    sorted_batch_indices = np.argsort(r_np)
    batch_summary_signals = [y_np[sorted_batch_indices, j] for j in range(batch_size)]
    batch_sum_w_ref = batch_summary_signals + [r_np[sorted_batch_indices]]

    plot_signals(
        t=r_np[sorted_batch_indices],
        signals=batch_sum_w_ref,   
        labels=[f"Traj {j}" for j in range(batch_size)] + ["Ideal Line"],
        title=f"Batch Convergence ({batch_size} Trajectories) Parity Map",
        xlabel="Reference Target (r)", ylabel="System Output (y)",
        dirname=dirname, filename="batch_summary_parity_plot"
    )
    
    # --- METRICS COMPILATION ---
    # Calculate, print, and save overarching tracking stats (MSE, IAE, rise time, etc.)
    tracking_metrics = compute_and_save_tracking_metrics(y_np, r_np, dt, dirname)

    # Clean return structure instead of references to non-existent variables
    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }


def simulate_tracking_exp(model, plant, r_trajectory, hyperparam_config, normalization_stats_path, dirname):
    """
    Simulates a controlled plant over a specified time horizon while tracking a reference trajectory.
    Gracefully handles look-ahead clamping at boundary limits and leverages Mamba's recurrent cache.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]

    # Flatten reference tensor for CPU metrics calculations
    r_np = r_trajectory.cpu().numpy().flatten()
    
    # Initialize GPU tensor buffers across the COMPLETE step timeline
    all_y = torch.zeros((steps, batch_size), device=device)
    all_u = torch.zeros((steps, batch_size), device=device)
    all_x1 = torch.zeros((steps, batch_size), device=device)
    all_x2 = torch.zeros((steps, batch_size), device=device)
    
    state = plant.get_initial_state(batch_size)

    # Prepare model for evaluation mode and reset hidden state memory
    model.eval()
    model.reset_memory(batch_size=batch_size, device=device)

    # Load normalization stats (saved during training)
    import json
    with open(normalization_stats_path, "r") as f:
        norm_stats = json.load(f)


    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories across {steps} steps...")
    delta_steps = hyperparam_config["train"]["delay_steps"]
    
    # Execute forward tracking simulation without tracking gradients
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant.get_y(state, t)  # Shape: (batch_size, 1)

            # Prevent index out of bounds
            look_ahead_idx = min(i + delta_steps, steps - 1)
            target_r = r_trajectory[look_ahead_idx].expand(batch_size, 1)  # Shape: (batch_size, 1)

            # 🔥 NORMALIZE INPUTS (do this ONCE per iteration)
            y_current_norm = (y_current - norm_stats['y_mean']) / (norm_stats['y_std'] + 1e-8)
            target_r_norm = (target_r - norm_stats['y_mean']) / (norm_stats['y_std'] + 1e-8)

            # Prepare model inputs
            y_t = y_current_norm.unsqueeze(1)          # Shape: (batch_size, 1, 1)
            y_t_delta = target_r_norm.unsqueeze(1)     # Shape: (batch_size, 1, 1)

            # Model inference
            u_out = model(y_t=y_t, y_t_delta=y_t_delta, use_memory=True)
            u = u_out[:, -1, :] * norm_stats['u_std'] + norm_stats['u_mean']  # 🔥 DENORMALIZE

            # Step the plant
            state, _ = plant.step(state, u, t, dt)

            # Logging
            all_y[i] = y_current.view(-1)
            all_u[i] = u.view(-1)
            all_x1[i] = state[:, 0]
            all_x2[i] = state[:, 1]

    # --- PLOTTING & EXPORT ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []

    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True) # Ensure path exists before logging
        
        # Extract specific data arrays for this trajectory (b)
        y_traj = all_y[:, b].cpu().numpy()
        u_traj = all_u[:, b].cpu().numpy()
        x1_traj = all_x1[:, b].cpu().numpy()
        x2_traj = all_x2[:, b].cpu().numpy()
        r_traj = r_np 

        # BUILD THE MATRIX DATAFRAME
        df_traj = pd.DataFrame({
            "time": time_axis,
            "state_x1": x1_traj,
            "state_x2": x2_traj,
            "output_y": y_traj,
            "control_u": u_traj,
            "target_r": r_traj
        })
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        # GENERATE TIME-SERIES PLOTS PER BATCH INDEX
        plot_signals(
            t=time_axis, 
            signals=[u_traj],
            labels=["Control Signal (u)"],
            title=f"Trajectory {b}: Control Action",
            xlabel="Time (h)", ylabel="Action Value",
            dirname=state_dirname, filename="plot_control_signal"
        )

        plot_signals(
            t=time_axis, 
            signals=[y_traj, r_traj],
            labels=["Output (y)", "Target (r)"],
            title=f"Trajectory {b}: Tracking Performance",
            xlabel="Time (h)", ylabel="Signal Value",
            dirname=state_dirname, filename="plot_output_tracking"
        )

        # GENERATE PARITY PLOTS PER BATCH INDEX (Sorted chronologically to prevent line crossings)
        sorted_indices = np.argsort(r_traj)
        plot_signals(
            t=r_traj[sorted_indices],
            signals=[y_traj[sorted_indices], r_traj[sorted_indices]],   
            labels=["Output (y)", "Ideal (y = r)"],
            title=f"Trajectory {b}: Parity Plot",
            xlabel="Reference (r)", ylabel="Output (y)",
            dirname=state_dirname, filename="parity_plot_output_tracking"
        )

        plot_signals(
            t=time_axis, 
            signals=[x1_traj, x2_traj],
            labels=["State x1", "State x2"],
            title=f"Trajectory {b}: Internal Plant States",
            xlabel="Time (h)", ylabel="State Magnitude",
            dirname=state_dirname, filename="plot_plant_states"
        )

    # --- GLOBAL BATCH OVERLAY PLOT GENERATION ---
    y_np = all_y.cpu().numpy()
    summary_signals = [y_np[:, j] for j in range(batch_size)]
    sum_w_ref = summary_signals + [r_np]
    
    plot_signals(
        t=time_axis,
        signals=sum_w_ref,
        labels=[f"Traj {j}" for j in range(batch_size)] + ["Target Reference"],
        title=f"Batch Convergence ({batch_size} Trajectories Overview)",
        xlabel="Time (h)", ylabel="System Output (y)",
        dirname=dirname, filename="batch_summary"
    )

    # GLOBAL BATCH PARITY MAP GENERATION
    sorted_batch_indices = np.argsort(r_np)
    batch_summary_signals = [y_np[sorted_batch_indices, j] for j in range(batch_size)]
    batch_sum_w_ref = batch_summary_signals + [r_np[sorted_batch_indices]]

    plot_signals(
        t=r_np[sorted_batch_indices],
        signals=batch_sum_w_ref,   
        labels=[f"Traj {j}" for j in range(batch_size)] + ["Ideal Line"],
        title=f"Batch Convergence ({batch_size} Trajectories) Parity Map",
        xlabel="Reference Target (r)", ylabel="System Output (y)",
        dirname=dirname, filename="batch_summary_parity_plot"
    )
    
    # --- METRICS COMPILATION ---
    # Calculate, print, and save overarching tracking stats (MSE, IAE, rise time, etc.)
    tracking_metrics = compute_and_save_tracking_metrics(y_np, r_np, dt, dirname)

    # Clean return structure instead of references to non-existent variables
    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }

def simulate_tracking_sakura(model, plant, r_trajectory, hyperparam_config, x_scaler, y_scaler, dirname):
    """
    Simulates a controlled plant over a specified time horizon while tracking a reference trajectory.
    Aligns perfectly with the mict processing pipeline, using scikit-learn scalers for normalizations.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]

    # Flatten reference tensor for CPU metrics calculations
    r_np = r_trajectory.cpu().numpy().flatten()
    
    # Initialize GPU tensor buffers across the COMPLETE step timeline
    all_y = torch.zeros((steps, batch_size), device=device)
    all_u = torch.zeros((steps, batch_size), device=device)
    all_x1 = torch.zeros((steps, batch_size), device=device)
    all_x2 = torch.zeros((steps, batch_size), device=device)
    
    state = plant.get_initial_state(batch_size)

    # Prepare model for evaluation mode
    model.eval()

    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories across {steps} steps...")
    delta_steps = hyperparam_config["train"]["delay_steps"]
    
    # Keep track of the historical pairs over time for Mamba's memory tracking per batch trajectory
    # List of lists to hold history for each sequence batch item cleanly
    history_pairs = [[] for _ in range(batch_size)]

    # Execute forward tracking simulation without tracking gradients
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_current = plant.get_y(state, t)  # Shape: (batch_size, 1)

            # Prevent index out of bounds
            look_ahead_idx = min(i + delta_steps, steps - 1)
            target_r = r_trajectory[look_ahead_idx].expand(batch_size, 1)  # Shape: (batch_size, 1)

            # Convert current steps to CPU NumPy arrays for Scikit-Learn transformations
            y_curr_np = y_current.cpu().numpy()
            tgt_r_np = target_r.cpu().numpy()

            # Process normalization matching mict's structure step-by-step per batch trajectory
            batch_y_t_norm = []
            batch_y_next_norm = []

            for b_idx in range(batch_size):
                input_pair = np.array([[y_curr_np[b_idx, 0], tgt_r_np[b_idx, 0]]])
                # Transform using the exact x_scaler fitted during training
                input_normalized = x_scaler.transform(input_pair) if x_scaler else input_pair
                history_pairs[b_idx].append(input_normalized[0])

                # Extract the history vectors up to this step idx
                curr_history = np.array(history_pairs[b_idx]) # Shape: (i + 1, 2)
                batch_y_t_norm.append(curr_history[:, 0])
                batch_y_next_norm.append(curr_history[:, 1])

            # Convert compiled histories into 3D Tensors: [Batch, Seq_Len, 1]
            y_t_tensor = torch.tensor(np.array(batch_y_t_norm), dtype=torch.float32).unsqueeze(-1).to(device)
            y_next_tensor = torch.tensor(np.array(batch_y_next_norm), dtype=torch.float32).unsqueeze(-1).to(device)

            # Model inference: passes entire tracking history down to the recurrent block
            u_seq_norm = model(y_t=y_t_tensor, y_next=y_next_tensor)
            
            # Extract the normalized control signal exclusively for the CURRENT step (the last item)
            u_norm_np = u_seq_norm[:, -1, :].cpu().numpy() # Shape: (batch_size, 1)

            # Inverse transform via y_scaler to obtain raw control actions for the physical system
            if y_scaler:
                u_unscaled = y_scaler.inverse_transform(u_norm_np)
            else:
                u_unscaled = u_norm_np
            
            u = torch.tensor(u_unscaled, dtype=torch.float32, device=device)

            # Step the physical plant forward
            state, _ = plant.step(state, u, t, dt)

            # Logging
            all_y[i] = y_current.view(-1)
            all_u[i] = u.view(-1)
            all_x1[i] = state[:, 0]
            all_x2[i] = state[:, 1]

    # --- PLOTTING & EXPORT ---
    time_axis = np.arange(steps) * dt
    trajectory_reports = []

    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        os.makedirs(state_dirname, exist_ok=True) 
        
        y_traj = all_y[:, b].cpu().numpy()
        u_traj = all_u[:, b].cpu().numpy()
        x1_traj = all_x1[:, b].cpu().numpy()
        x2_traj = all_x2[:, b].cpu().numpy()
        r_traj = r_np 

        df_traj = pd.DataFrame({
            "time": time_axis,
            "state_x1": x1_traj,
            "state_x2": x2_traj,
            "output_y": y_traj,
            "control_u": u_traj,
            "target_r": r_traj
        })
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")
        trajectory_reports.append(df_traj)

        # GENERATE TIME-SERIES PLOTS PER BATCH INDEX
        plot_signals(
            t=time_axis, 
            signals=[u_traj],
            labels=["Control Signal (u)"],
            title=f"Trajectory {b}: Control Action",
            xlabel="Time (h)", ylabel="Action Value",
            dirname=state_dirname, filename="plot_control_signal"
        )

        plot_signals(
            t=time_axis, 
            signals=[y_traj, r_traj],
            labels=["Output (y)", "Target (r)"],
            title=f"Trajectory {b}: Tracking Performance",
            xlabel="Time (h)", ylabel="Signal Value",
            dirname=state_dirname, filename="plot_output_tracking"
        )

        # GENERATE PARITY PLOTS PER BATCH INDEX
        sorted_indices = np.argsort(r_traj)
        plot_signals(
            t=r_traj[sorted_indices],
            signals=[y_traj[sorted_indices], r_traj[sorted_indices]],   
            labels=["Output (y)", "Ideal (y = r)"],
            title=f"Trajectory {b}: Parity Plot",
            xlabel="Reference (r)", ylabel="Output (y)",
            dirname=state_dirname, filename="parity_plot_output_tracking"
        )

        plot_signals(
            t=time_axis, 
            signals=[x1_traj, x2_traj],
            labels=["State x1", "State x2"],
            title=f"Trajectory {b}: Internal Plant States",
            xlabel="Time (h)", ylabel="State Magnitude",
            dirname=state_dirname, filename="plot_plant_states"
        )

    # --- GLOBAL BATCH OVERLAY PLOT GENERATION ---
    y_np = all_y.cpu().numpy()
    summary_signals = [y_np[:, j] for j in range(batch_size)]
    sum_w_ref = summary_signals + [r_np]
    
    plot_signals(
        t=time_axis,
        signals=sum_w_ref,
        labels=[f"Traj {j}" for j in range(batch_size)] + ["Target Reference"],
        title=f"Batch Convergence ({batch_size} Trajectories Overview)",
        xlabel="Time (h)", ylabel="System Output (y)",
        dirname=dirname, filename="batch_summary"
    )

    # GLOBAL BATCH PARITY MAP GENERATION
    sorted_batch_indices = np.argsort(r_np)
    batch_summary_signals = [y_np[sorted_batch_indices, j] for j in range(batch_size)]
    batch_sum_w_ref = batch_summary_signals + [r_np[sorted_batch_indices]]

    plot_signals(
        t=r_np[sorted_batch_indices],
        signals=batch_sum_w_ref,   
        labels=[f"Traj {j}" for j in range(batch_size)] + ["Ideal Line"],
        title=f"Batch Convergence ({batch_size} Trajectories) Parity Map",
        xlabel="Reference Target (r)", ylabel="System Output (y)",
        dirname=dirname, filename="batch_summary_parity_plot"
    )
    
    # --- METRICS COMPILATION ---
    tracking_metrics = compute_and_save_tracking_metrics(y_np, r_np, dt, dirname)

    return {
        "trajectory_dataframes": trajectory_reports,
        "metrics": tracking_metrics,
        "simulated_outputs": y_np,
        "simulated_controls": all_u.cpu().numpy()
    }

def simulate_tracking_old(model, plant, r_trajectory, hyperparam_config, dirname):
    """
    Simulates a controlled plant over a specified time horizon while tracking a reference trajectory.
    Handles look-ahead clamping gracefully at bounds limits.
    """
    # Extract configuration sub-dictionaries
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    # Unpack specific parameters
    steps = sim_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]

    # Flatten reference tensor for CPU metrics calculations
    r_np = r_trajectory.cpu().numpy().flatten()
    
    # Initialize GPU tensor buffers across the COMPLETE step timeline
    all_y = torch.zeros((steps, batch_size), device=device)
    all_u = torch.zeros((steps, batch_size), device=device)
    all_x1 = torch.zeros((steps, batch_size), device=device)
    all_x2 = torch.zeros((steps, batch_size), device=device)
    
    state = plant.get_initial_state(batch_size)

    # Prepare model for evaluation mode and reset hidden state memory
    model.eval()
    model.reset_memory(batch_size=batch_size, device=device)

    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories across {steps} steps...")
    delta_steps = hyperparam_config["train"]["delay_steps"]
    
    # Execute forward tracking simulation without tracking gradients
    with torch.no_grad():
        # FIX: Loop now runs for ALL steps to prevent early simulation freezing
        for i in range(steps):
            t = i * dt
            y_current = plant.get_y(state, t) # This is y(t)
            
            # Grab current reference profile
            current_r = r_trajectory[i].expand(batch_size, 1)

            # FIX: Prevent index out of bounds using boundary clamping
            look_ahead_idx = min(i + delta_steps, steps - 1)
            target_r = r_trajectory[look_ahead_idx].expand(batch_size, 1)

            # Concatenate current observed state and future look-ahead target
            current_input = torch.cat([y_current, target_r], dim=-1).unsqueeze(1)

            # --- INFERENCE ---
            u_out = model(current_input, use_memory=True) 
            u = u_out[:, -1, :]

            # --- STEP PLANT ---
            state, _ = plant.step(state, u, t, dt)

            # --- LOGGING (Captured safely on every index) ---
            all_y[i]  = y_current.squeeze()
            all_u[i]  = u.squeeze()
            all_x1[i] = state[:, 0]
            all_x2[i] = state[:, 1]

    # --- PLOTTING & EXPORT ---
    time_axis = np.arange(steps) * dt

    # Parse and save individual trajectory records
    for b in range(batch_size):
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        
        # Extract specific data for this trajectory (b)
        y_traj = all_y[:, b].cpu().numpy()
        u_traj = all_u[:, b].cpu().numpy()
        x1_traj = all_x1[:, b].cpu().numpy()
        x2_traj = all_x2[:, b].cpu().numpy()
        
        # FIX: Directly assign the flat reference array instead of using np.full
        r_traj = r_np 

        # SAVE THE CSV
        df_traj = pd.DataFrame({
            "time": time_axis,
            "state_x1": x1_traj,
            "state_x2": x2_traj,
            "output_y": y_traj,
            "control_u": u_traj,
            "target_r": r_traj
        })
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")

        # GENERATE PLOTS PER BATCH
        plot_signals(
            t=time_axis, 
            signals=[u_traj],
            labels=["Control Signal (u)"],
            title=f"Trajectory {b}: Control Action",
            xlabel="Time (h)", ylabel="Action Value",
            dirname=state_dirname, filename="plot_control_signal"
        )

        plot_signals(
            t=time_axis, 
            signals=[y_traj, r_traj],
            labels=["Output (y)", "Target (r)"],
            title=f"Trajectory {b}: Tracking Performance",
            xlabel="Time (h)", ylabel="Signal Value",
            dirname=state_dirname, filename="plot_output_tracking"
        )

        plot_signals(
            t=r_traj,
            signals=[y_traj, r_traj],   # ✅ add diagonal line
            labels=["Output (y)", "Ideal (y = r)"],
            title=f"Trajectory {b}: Parity Plot",
            xlabel="Reference (r)",
            ylabel="Output (y)",
            dirname=state_dirname,
            filename="parity_plot_output_tracking"
        )
        

        plot_signals(
            t=time_axis, 
            signals=[x1_traj, x2_traj],
            labels=["State x1", "State x2"],
            title=f"Trajectory {b}: Internal Plant States",
            xlabel="Time (h)", ylabel="State Magnitude",
            dirname=state_dirname, filename="plot_plant_states"
        )

    # Batch Summary Plot
    y_np = all_y.cpu().numpy()
    summary_signals = [y_np[:, j] for j in range(batch_size)]
    sum_w_ref = summary_signals + [r_np]
    
    plot_signals(
        t=time_axis,
        signals=sum_w_ref,
        labels=[None] * batch_size + ["Target"],
        title=f"Batch Convergence ({batch_size} Trajectories)",
        xlabel="Time (h)",
        ylabel="System Output (y)",
        dirname=dirname,
        filename="batch_summary"
    )

    plot_signals(
        t=r_traj,
        signals=sum_w_ref,   # ✅ add diagonal line
        labels=[None] * batch_size + ["Target"],
        title=f"Batch Convergence ({batch_size} Trajectories)",
        xlabel="Time (h)",
        ylabel="System Output (y)",
        dirname=dirname,
        filename="batch_summary_parity_plot"
    )
    
    # Calculate overarching summary tracking stats
    compute_and_save_tracking_metrics(y_np, r_np, dt, dirname)

    return ()



#=== FUNCTION TO SEED EVERYTHING FOR REPRODUCIBILITY ===#
def seed_everything(seed=42):
    """
    Seeds all relevant libraries to ensure reproducible results.

    Parameters:
    - seed (int): The numerical seed value used to initialize all random number generators (default: 42).

    Returns:
    - None: The function configures global runtime states and does not return a value.

    The function enforces absolute determinism across various execution contexts. It explicitly binds 
    the seed to Python's core `random` module, environment variables, NumPy's matrix operations, and 
    both CPU and GPU tensor variants in PyTorch. Finally, it overrides standard CUDA Deep Neural Network 
    (cuDNN) configurations to deactivate dynamic kernel auto-tuning algorithm selection, completely 
    eliminating stochastic variance across identical processing runs.
    """
    # Seed native Python behaviors and system environments
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed third-party matrix execution engines
    np.random.seed(seed)
    
    # Seed deep learning core frameworks across standard processing hardware
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Force synchronization across multi-GPU environments
    
    # Critical for CUDA reproducibility: override cuDNN runtime optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


#=== FUNCTION TO LOAD TRAINED MODEL ===#
def load_model(model_class, checkpoint_path, device=None):
    """
    Loads a trained PyTorch model state dictionary and configuration from a checkpoint file.

    Parameters:
    - model_class (class): The class of the model to be instantiated.
    - checkpoint_path (str): The system file path pointing to the saved checkpoint file (.pt).
    - device (str or torch.device, optional): The target computing device (e.g., 'cpu', 'cuda') 
      where the model parameters will be mapped. If None, it automatically falls back to 
      GPU if available, otherwise CPU.

    Returns:
    - model (MambaInverseController): The instantiated model restored to its trained weights 
      and set to evaluation mode.

    The function unpacks a saved PyTorch checkpoint dictionary, extracts the embedded 
    hyperparameter configuration dictionary, and uses it to dynamically instantiate the 
    `MambaInverseController` architecture. It handles safe tensor device reassignment, 
    maps the recovered parameters to the model structure, and locks the model layer states 
    into evaluation mode (`.eval()`) for reliable forward-pass inference.
    """
    # Load the serialized checkpoint dictionary from disk
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract structural configurations and instantiate the network
    hyperparam_config = checkpoint['config']
    model = model_class(hyperparam_config)

    # Determine default execution hardware if none was provided
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Transfer the model parameters to the target processing device
    model = model.to(device)
    
    # Restore the historical weight state configurations
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze layers into evaluation mode for tracking inference
    model.eval()
    
    return model

