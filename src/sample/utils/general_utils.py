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
    criterion = nn.MSELoss()
    loss_history = []
    all_data_frames = []

    # Create a directory for individual sequence CSVs
    seq_dir = os.path.join("results", dirname, "sequences")
    os.makedirs(seq_dir, exist_ok=True)

    model.to(device)
    print(f"🚀 Training Mamba {plant.__class__.__name__} on {device}")

    for epoch in range(epochs):
        # Reset trajectory using parameters from config
        if hasattr(plant, 'reset_trajectory'):
            try:
                plant.reset_trajectory(seq_len=seq_len, dt=dt, lambd=lambd, p=p)
            except TypeError:
                plant.reset_trajectory()

        state = plant.get_initial_state(batch_size)
        all_y_t, all_y_next, all_u = [], [], []

        # --- SIMULATION PHASE ---
        # Inside GPUtrain_controllerFFT
        delta_steps = 5  # For example, look 5 steps ahead (delta = 5 * dt)

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

        # --- DATA PREPARATION FOR CSV ---
        y_t_stack = torch.stack(all_y_t, dim=1).cpu().numpy()
        y_next_stack = torch.stack(all_y_next, dim=1).cpu().numpy()
        u_stack = torch.stack(all_u, dim=1).cpu().numpy()

        num_samples = y_t_stack.shape[1] 

        epoch_df = pd.DataFrame({
            "t": [i * dt for i in range(num_samples)],
            "y_t": y_t_stack[0, :, 0],
            "y_next": y_next_stack[0, :, 0],
            "u_control": u_stack[0, :, 0]
        })
        
        save_df_to_csv(epoch_df, 
                       dirname=dirname+"/sequences",
                       filename=f"sample_trajectory_epoch_{epoch}_seq_0.csv")

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

    return loss_history




def compute_and_save_stabilization_metrics(
    y_np,
    ref_val,
    dt,
    dirname,
    settle_tol=0.05,
    steady_frac=0.2,
):
    """
    Compute stabilization metrics and save them to CSV.
    """

    steps, batch_size = y_np.shape
    ref = ref_val

    error = y_np - ref
    abs_error = np.abs(error)

    # --- Error-based metrics ---
    mae = abs_error.mean(axis=0)
    mse = (error ** 2).mean(axis=0)
    iae = abs_error.sum(axis=0) * dt
    ise = (error ** 2).sum(axis=0) * dt

    # --- Overshoot ---
    overshoot = np.max(y_np, axis=0) - ref

    # --- Settling time ---
    settle_band = settle_tol * abs(ref)
    settling_time = np.full(batch_size, np.nan)

    for b in range(batch_size):
        for t in range(steps):
            if np.all(abs_error[t:, b] <= settle_band):
                settling_time[b] = t * dt
                break

    # --- Steady-state error ---
    n_ss = int(steps * steady_frac)
    steady_state_error = abs_error[-n_ss:, :].mean(axis=0)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "trajectory": np.arange(batch_size),
        "MAE": mae,
        "MSE": mse,
        "IAE": iae,
        "ISE": ise,
        "Overshoot": overshoot,
        "SettlingTime": settling_time,
        "SteadyStateError": steady_state_error,
    })

    # --- Also save aggregate statistics ---
    summary_df = pd.DataFrame({
        "metric": [
            "MAE", "MSE", "IAE", "ISE",
            "Overshoot", "SettlingTime", "SteadyStateError"
        ],
        "mean": [
            np.nanmean(mae),
            np.nanmean(mse),
            np.nanmean(iae),
            np.nanmean(ise),
            np.nanmean(overshoot),
            np.nanmean(settling_time),
            np.nanmean(steady_state_error),
        ],
        "std": [
            np.nanstd(mae),
            np.nanstd(mse),
            np.nanstd(iae),
            np.nanstd(ise),
            np.nanstd(overshoot),
            np.nanstd(settling_time),
            np.nanstd(steady_state_error),
        ],
        "worst": [
            np.nanmax(mae),
            np.nanmax(mse),
            np.nanmax(iae),
            np.nanmax(ise),
            np.nanmax(overshoot),
            np.nanmax(settling_time),
            np.nanmax(steady_state_error),
        ],
    })

    # --- Save using your existing utility ---
    save_df_to_csv(
        df=df,
        dirname=dirname,
        filename="stabilization_metrics_per_trajectory",
    )

    save_df_to_csv(
        df=summary_df,
        dirname=dirname,
        filename="stabilization_metrics_summary",
    )
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
    compute_and_save_stabilization_metrics(y_np, plant.ref_value, dt, dirname)

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
    # # A sine wave base
    # time_axis = np.arange(steps) * dt
    # period = 50 # Total hours for one full cycle
    # sine_base = np.sin(2 * np.pi * time_axis / period)

    # # Use tanh to sharpen the curve into a "soft" rectangle
    # # Higher gain = more rectangular; Lower gain = more like a pure sine
    # gain = 1.0 
    # r_trajectory_np = 0.25 + 0.025 * np.tanh(gain * sine_base) 

    # # (Note: 0.375 is the midpoint between 0.3 and 0.45)
    # r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)

    
    
    # # 1. GENERATE A CONSTANT REFERENCE
    constant_val = 0.24
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

    print(f"📈 Testing Trajectory Tracking: {batch_size} trajectories...")

    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y = plant.get_y(state, t) # This is y(t)
            
            # 2. Grab the specific target for "Next" time step y(t+dt)
            current_r = r_trajectory[i].expand(batch_size, 1)

            # 3. CONSTRUCT INPUT: [y(t), y_target]
            # This is exactly the (y_t, y_next) triplet the model was trained on
            current_input = torch.cat([y, current_r], dim=-1).unsqueeze(1) 

            # 4. INFERENCE
            u_out = model(current_input) 
            
            # Note: If using Mamba, u_out usually returns [batch, 1, output_dim]
            u = u_out[:, -1, :]
            #u = torch.clamp(u_out[:, -1, :], 0.0, plant.U_MAX)

            # 5. STEP PLANT
            # This uses the predicted U to move the REAL plant state
            state, _ = plant.step(state, u, t, dt)

            # --- LOGGING ---
            history["y"][i] = y[0].item()
            history["r"][i] = current_r[0].item() 
            history["u"][i] = u[0].item()
            history["x1"][i] = state[0, 0].item() # Assuming index 0 is biomass
            history["x2"][i] = state[0, 1].item() # Assuming index 1 is substrate
            all_y[i] = y.squeeze()
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
    compute_and_save_stabilization_metrics(y_np, r_trajectory, dt, dirname)

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