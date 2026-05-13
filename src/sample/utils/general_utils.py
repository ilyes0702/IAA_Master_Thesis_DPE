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

@track_resources
def GPUtrain_controllerFFT(model, plant, hyperparam_config, dirname="name_directory"):
    # --- EXTRACT HYPERPARAMETERS FROM CONFIG ---
    # Training params
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    device = train_cfg["device"]
    
    # Signal params
    sig_cfg =   hyperparam_config["signal"]
    seq_len = sig_cfg["seq_len"]
    dt = sig_cfg["dt"]
    lambd = sig_cfg.get("lambd", 5.0)
    p = sig_cfg.get("p", 0.4)

    # --- INITIALIZE TRAINING ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta = 0.1)
    loss_history = []
    all_data_frames = []
    all_D_centers = []

    model.to(device)
    print(f"🚀 Training Mamba {plant.__class__.__name__} on {device}")

    for epoch in range(epochs):
        
        if hasattr(plant, 'reset_trajectory'):
            try:
                D_center = plant.reset_trajectory(seq_len=seq_len, dt=dt, lambd=lambd, p=p)
            except TypeError:
                D_center = plant.reset_trajectory()
            all_D_centers.append(D_center) 


        state = plant.get_initial_state(batch_size)
        all_y_t, all_y_next, all_u = [], [], []
        sequence_D_centers = []

        # --- SIMULATION PHASE ---
        # Inside GPUtrain_controllerFFT
        delta_steps = hyperparam_config["train"]["delay_steps"]  # For example, look 5 steps ahead (delta = 5 * dt)
        delta_steps = 20
        for t_idx in range(seq_len - delta_steps):
            t = t_idx * dt
            u_signal = plant.get_u_at_step(t_idx)
            
            # Current state
            y_t = plant.get_y(state)
            
            # Forward simulate k steps using the SAME u_signal
            # (Canaday logic: the input v_train is held constant over the interval delta)
            temp_state = state
            for _ in range(delta_steps):
                temp_state, _ = plant.step(temp_state, u_signal, t, dt)
            
            # The state after delta
            y_delta = plant.get_y(temp_state)
            
            all_y_t.append(y_t)
            all_y_next.append(y_delta) # This is now y(t + delta)
            all_u.append(u_signal)
            
            # IMPORTANT: Move the actual simulation forward only 1 step 
            # to keep the trajectory continuous, or skip delta_steps.
            state_next, _ = plant.step(state, u_signal, t, dt)
            state = state_next.detach()
            # If D_center is updated per sequence, store it here
            if hasattr(plant, 'current_D_center'):
                sequence_D_centers.append(plant.current_D_center)

        # --- DATA PREPARATION FOR CSV ---
        y_t_stack = torch.stack(all_y_t, dim=1).cpu().numpy()
        y_next_stack = torch.stack(all_y_next, dim=1).cpu().numpy()
        u_stack = torch.stack(all_u, dim=1).cpu().numpy()

        num_samples = y_t_stack.shape[1] 

        epoch_df = pd.DataFrame({
            "t": [i * dt for i in range(num_samples)],
            "y_t": y_t_stack[0, :, 0],
            "y_next": y_next_stack[0, :, 0],
            "u_control": u_stack[0, :, 0],
            "D_center": D_center  # Add D_center for this epoch
        })
        
        save_df_to_csv(epoch_df, 
                       dirname=dirname+"/sequences",
                       filename=f"sample_trajectory_epoch_{epoch}_seq_0.csv")

        # Save a signal plot for this sample sequence, including D_center as a horizontal line
        time_axis = epoch_df["t"].values
        u_signal_plot = epoch_df["u_control"].values
        y_t_plot = epoch_df["y_t"].values
        y_next_plot = epoch_df["y_next"].values
        d_center_plot = np.full_like(time_axis, D_center, dtype=float)

        plot_signals(
            time_axis,
            [u_signal_plot, y_t_plot, y_next_plot, d_center_plot],
            labels=["u_control", "y_t", "y_next", "D_center"],
            xlabel="Time",
            ylabel="Signal",
            title=f"Sample Sequence Signals Epoch {epoch}",
            dirname=dirname+"/sequences",
            filename=f"sample_trajectory_epoch_{epoch}_seq_0_plot"
        )

        all_data_frames.append(epoch_df)

        # --- TRAINING PHASE ---
        y_t_seq = torch.stack(all_y_t, dim=1)
        y_next_seq = torch.stack(all_y_next, dim=1)
        y_target = torch.stack(all_u, dim=1)
        x_tensor = torch.cat([y_t_seq, y_next_seq], dim=-1)

        model.train()
        optimizer.zero_grad()
        u_pred = model(x_tensor)
        loss = criterion(u_pred, y_target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} | Loss: {loss.item():.6f}")
            # We take index 0 of the batch
            u_truth_sample = y_target[0].detach().cpu().numpy().flatten()
            u_pred_sample = u_pred[0].detach().cpu().numpy().flatten()
            
            plot_signals(
                time_axis[:len(u_truth_sample)], 
                [u_truth_sample, u_pred_sample],
                labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                xlabel="Time", ylabel="Control Signal",
                title=f"Mamba Prediction Accuracy - Epoch {epoch}",
                dirname=dirname+"/sequences",
                filename=f"prediction_accuracy_epoch_{epoch}"
            )

    # --- FINAL AGGREGATION & SAVING ---
    master_df = pd.concat(all_data_frames, ignore_index=True)
    save_df_to_csv(master_df, dirname=dirname, filename="all_training_data_summary")

    df_loss = pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_history})
    save_df_to_csv(df_loss, dirname=dirname, filename="training_loss_history")
    
    # Passing the whole config for tracking
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller")

    plot_signals(
        df_loss["epoch"].values, [df_loss["loss"].values],
        labels=["MSE Loss"], xlabel="Epoch", ylabel="Loss",
        title=f"Convergence ({plant.__class__.__name__})",
        dirname=dirname, filename="training_loss_plot"
    )

    # --- SAVE D_CENTER HISTORY ---
    # Create a DataFrame for D_center history (per epoch)
    d_center_df = pd.DataFrame({
        "Sequence": range(1, epochs*batch_size + 1),
        "D_center": sequence_D_centers
    })
    save_df_to_csv(
        d_center_df,
        dirname=dirname,
        filename="D_center_history.csv"
    )

    return loss_history




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
#=== FUNCTION FOR THE SIMULATION OF CONTROLLED PLANT ===#
def GPUSimulateControl_new_ma(model, plant, hyperparam_config, dirname):
    train_cfg = hyperparam_config["train"]
    sig_cfg   = hyperparam_config["signal"]
    sim_cfg   = hyperparam_config["simulate"]

    device = train_cfg["device"]
    dt = sig_cfg["dt"]
    steps = sim_cfg["seq_len"]
    batch_size = sim_cfg["batch_size"]

    # --- Buffers for logging ---
    # history stores the full state details for the first trajectory only
    history = {
        "x1": np.zeros(steps), "x2": np.zeros(steps),
        "y":  np.zeros(steps), "r":  np.zeros(steps),
        "u":  np.zeros(steps)
    }
    # all_y captures the plant output (y) for EVERY trajectory in the batch
    all_y = torch.zeros((steps, batch_size), device=device)

    # Initial state (randomized by the plant internally for the batch)
    state = plant.get_initial_state(batch_size)
    r_tensor = plant.ref_value.expand(batch_size, 1)

    model.eval()
    print(f"🚀 Running Raw Inference: {batch_size} random initial states...")

    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y = plant.get_y(state, t)

            # --- MATCH TRAINING INPUT CONSTRUCTION ---
            current_input = torch.cat([y, r_tensor], dim=-1).unsqueeze(1) # [Batch, 1, 2]

            # --- INFERENCE ---
            u_out = model(current_input) 
            u = u_out[:, -1, 0:1] 
            
            # Physical safety clamp
            u = torch.clamp(u, 0.0, plant.U_MAX)

            # --- LOGGING ---
            # Detail log for Trajectory 0
            history["x1"][i] = state[0, 0].item()
            history["x2"][i] = state[0, 1].item()
            history["y"][i]  = y[0].item()
            history["r"][i]  = plant.ref_value.item()
            history["u"][i]  = u[0].item()
            
            # Summary log for all batch members
            all_y[i] = y.squeeze()

            # --- STEP ---
            state, _ = plant.step(state, u, t, dt)

    # --- PLOTTING ---
    time_axis = np.arange(steps) * dt
    
    # 1. Detailed plots (Biomass, Substrate, etc.) for the first trajectory
    plot_config = plant.get_plot_config()
    for idx, cfg in enumerate(plot_config):
        signals = [history[col] for col in cfg["cols"]]
        plot_signals(
            t=time_axis, signals=signals, labels=cfg["labels"],
            title=cfg["title"], xlabel="Time (h)", ylabel=cfg["ylabel"],
            dirname=dirname, filename=f"detailed_plot_{idx}"
        )

    # 2. NEW: Batch Summary Plot (All curves with random initial states)
    y_np = all_y.cpu().numpy()
    summary_signals = [y_np[:, j] for j in range(batch_size)]
    summary_signals.append(np.full(steps, plant.ref_value.item())) # Add target line
    
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

    return


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
    time_axis = np.arange(steps) * dt
    period = 20 # Total hours for one full cycle
    sine_base = np.sin(2 * np.pi * time_axis / period)

    # Use tanh to sharpen the curve into a "soft" rectangle
    # Higher gain = more rectangular; Lower gain = more like a pure sine
    gain = 1.0 
    r_trajectory_np = 0.26 + 0.03 * np.tanh(gain * sine_base) + 0.001*time_axis

    # (Note: 0.375 is the midpoint between 0.3 and 0.45)
    r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)

    
    
    # # # 1. GENERATE A CONSTANT REFERENCE
    # constant_val = 0.3
    # # Create a tensor of shape [steps, 1] filled with the constant value
    # r_trajectory = torch.full((steps, 1), constant_val, device=device, dtype=torch.float32)

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


def load_model_complete(model_class, filepath, device='cuda'):
    """
    Loads a model without needing to manually provide params.
    """
    # 1. Load the checkpoint
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    
    # 2. Extract the config and reconstruct the architecture
    model_config = checkpoint['model_config']
    model = model_class(**model_config)
    
    # 3. Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded with config: {model_config}")
    return model, model_config


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