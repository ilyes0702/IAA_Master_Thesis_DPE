# Import standard libraries
from logging import config
import os
from networkx import config
import sys
import numpy as np
import pandas as pd

# import machine learning modules
from src.sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import torch
import torch.nn as nn
import pandas as pd
import os
import torch
from src.sample.classes.MambaInverseController import MambaInverseController

plt.style.use("src/sample/style.mplstyle")

import numpy as np
import pandas as pd

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



#=== FUNCTION FOR THE SIMULATION OF CONTROLLED PLANT WITH TRACKING ===#
def GPUSimulateTracking(model, plant, hyperparam_config, dirname):
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    steps = sim_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = sim_cfg["batch_size"]
    device = train_cfg["device"]

    # 1. GENERATE A TIME-VARYING REFERENCE TRAJECTORY
    # A sine wave base
    # time_axis = np.arange(steps) * dt
    # period = 20 # Total hours for one full cycle
    # sine_base = np.sin(2 * np.pi * time_axis / period)

    # # Use tanh to sharpen the curve into a "soft" rectangle
    # # Higher gain = more rectangular; Lower gain = more like a pure sine
    # gain = 1.0 
    # r_trajectory_np = 0.26 + 0.03 * np.tanh(gain * sine_base) + 0.001*time_axis

    # # (Note: 0.375 is the midpoint between 0.3 and 0.45)
    # r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)

    
    
    # # # 1. GENERATE A CONSTANT REFERENCE
    constant_val = 0.3
    # Create a tensor of shape [steps, 1] filled with the constant value
    r_trajectory = torch.full((steps, 1), constant_val, device=device, dtype=torch.float32)

    r_np = r_trajectory.cpu().numpy().flatten()
    # Initialize history and state as before
    history = {"x1": np.zeros(steps), "x2": np.zeros(steps), "y": np.zeros(steps), "r": np.zeros(steps), "u": np.zeros(steps)}
    all_y = torch.zeros((steps, batch_size), device=device)
    all_u = torch.zeros((steps, batch_size), device=device)
    all_x1 = torch.zeros((steps, batch_size), device=device)
    all_x2 = torch.zeros((steps, batch_size), device=device)
    state = plant.get_initial_state(batch_size)

    model.eval()
    model.reset_memory(batch_size=batch_size, device=device)

    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories...")
    delta_steps =hyperparam_config["train"]["delay_steps"]  # For example, look 5 steps ahead (delta = 5 * dt)
    
    with torch.no_grad():
        for i in range(steps-delta_steps):
            t = i * dt
            y_current = plant.get_y(state, t) # This is y(t)
            
            # 2. Grab the specific target for "Next" time step y(t+dt)
            current_r = r_trajectory[i].expand(batch_size, 1)

            # Grab the target 20 steps into the future
            target_r = r_trajectory[i + delta_steps].expand(batch_size, 1)

            # Now the triplet [y_t, r_{t+20}] matches the training data exactly
            current_input = torch.cat([y_current, target_r], dim=-1).unsqueeze(1)

            # 3. CONSTRUCT INPUT: [y(t), y_target]
            # This is exactly the (y_t, y_next) triplet the model was trained on
            # current_input = torch.cat([y, current_r], dim=-1).unsqueeze(1) 

            # 4. INFERENCE
            u_out = model(current_input, use_memory=True) 
            
            # Note: If using Mamba, u_out usually returns [batch, 1, output_dim]
            u = u_out[:, -1, :]
            #u = torch.clamp(u_out[:, -1, :], 0.0, plant.U_MAX)

            # 5. STEP PLANT
            # This uses the predicted U to move the REAL plant state
            state, _ = plant.step(state, u, t, dt)

            # --- LOGGING ---
            history["y"][i] = y_current[0].item()
            history["r"][i] = current_r[0].item() 
            history["u"][i] = u[0].item()
            history["x1"][i] = state[0, 0].item() # Assuming index 0 is biomass
            history["x2"][i] = state[0, 1].item() # Assuming index 1 is substrate
            all_y[i] = y_current.squeeze()
            all_u[i] = u.squeeze()
            all_x1[i] = state[:, 0] # Assuming index 0 is biomass
            all_x2[i] = state[:, 1] # Assuming index 1 is substrate

    # --- PLOTTING ---
    time_axis = np.arange(steps) * dt

    for b in range(batch_size):
        # Create subfolder path for this specific initial state
        # 1. Define a unique sub-directory for this specific batch member
        state_dirname = os.path.join(dirname, f"initial_state_{b}")
        
        # 2. Extract specific data for this trajectory (b)
        y_traj = all_y[:, b].cpu().numpy()
        u_traj = all_u[:, b].cpu().numpy()
        x1_traj = all_x1[:, b].cpu().numpy()
        x2_traj = all_x2[:, b].cpu().numpy()
        r_traj = np.full(steps, r_np)

        # 3. SAVE THE CSV (using your custom method)
        df_traj = pd.DataFrame({
            "time": time_axis,
            "biomass_x1": x1_traj,
            "substrate_x2": x2_traj,
            "growth_rate_y": y_traj,
            "control_u": u_traj,
            "target_r": r_traj
        })
        save_df_to_csv(df_traj, dirname=state_dirname, filename="state_report")

        # 4. GENERATE THE 3 PLOTS PER BATCH

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


    # 2. Batch Summary Plot (All curves with random initial states)
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
    compute_and_save_tracking_metrics(y_np, r_np, dt, dirname)

    # # Create a dictionary of the logged history
    # data_dict = {
    #     "time": time_axis,
    #     "y_growth_rate": history["y"],
    #     "r_reference": history["r"],
    #     "u_control": history["u"],
    #     # If your plant records internal states x1, x2 in history:
    #     "x1_biomass": history["x1"],
    #     "x2_substrate": history["x2"]
    # }

    # # Convert to DataFrame and save
    # df = pd.DataFrame(data_dict)
    # save_df_to_csv(df, dirname=dirname, filename="tracking_results")

    return()

import torch
import numpy as np
import random
import os

#=== FUNCTION TO SEED EVERYTHING FOR REPRODUCIBILITY ===#
def seed_everything(seed=42):
    """
    Seeds all relevant libraries to ensure reproducible results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    
    # Critical for CUDA reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def load_model(checkpoint_path, device=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparam_config = checkpoint['config']
    model = MambaInverseController(hyperparam_config)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model