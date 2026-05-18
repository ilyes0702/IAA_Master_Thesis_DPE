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
from src.sample.classes.MambaInverseController import MambaInverseController
import matplotlib.pyplot as plt
import random

plt.style.use("src/sample/style.mplstyle")

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
        r_trajectory_np = 0.23 + 0.05 * np.tanh(gain * sine_base) - 0.001 * time_axis
        
        # Convert the structural numpy baseline into a target PyTorch tensor array
        r_trajectory = torch.tensor(r_trajectory_np, device=device, dtype=torch.float32).unsqueeze(1)
    
    else:
        raise ValueError(f"Unknown reference mode selection: '{mode}'. Choose 'constant' or 'dynamic'.")

    return r_trajectory


#=== FUNCTION FOR THE SIMULATION OF CONTROLLED PLANT WITH TRACKING ===#
def simulate_tracking(model, plant, r_trajectory, hyperparam_config, dirname):
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
def load_model(checkpoint_path, device=None):
    """
    Loads a trained PyTorch model state dictionary and configuration from a checkpoint file.

    Parameters:
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
    model = MambaInverseController(hyperparam_config)

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