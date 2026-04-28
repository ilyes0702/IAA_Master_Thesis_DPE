# Import standard libraries
import os
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

plt.style.use("src/sample/style.mplstyle")

@track_resources
def GPUtrain_controllerFFT(model, plant, epochs, seq_len, dt, model_config, batch_size, device='cuda', dirname="name_directory"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_history = []

    # List to hold all data for the final master CSV
    all_data_frames = []

    # Create a directory for individual sequence CSVs
    seq_dir = os.path.join("results", dirname, "sequences")
    os.makedirs(seq_dir, exist_ok=True)

    model.to(device)
    print(f"🚀 Training Mamba {plant.__class__.__name__}")

    for epoch in range(epochs):
        if hasattr(plant, 'reset_trajectory'):
            try:
                plant.reset_trajectory(seq_len=seq_len, dt=dt, lambd=5.0, p=0.4)
            except TypeError:
                plant.reset_trajectory()

        state = plant.get_initial_state()
        all_y_t, all_y_next, all_u = [], [], []

        # --- SIMULATION PHASE ---
        with torch.no_grad():
            for t_idx in range(seq_len):
                t = t_idx * dt
                u_signal = plant.get_u_at_step(t_idx) if hasattr(plant, 'get_u_at_step') else \
                           torch.rand((batch_size, 1), device=device) * plant.U_MAX

                y_t = plant.get_y(state, t)
                state_next, y_next = plant.step(state, u_signal, t, dt)

                all_y_t.append(y_t)
                all_y_next.append(y_next)
                all_u.append(u_signal)
                state = state_next.detach()

        # --- DATA PREPARATION FOR CSV ---
        # We take the first batch [0] from this epoch to save as an example sequence CSV
        y_t_stack = torch.stack(all_y_t, dim=1).cpu().numpy()      # [Batch, Seq, Feat]
        y_next_stack = torch.stack(all_y_next, dim=1).cpu().numpy()
        u_stack = torch.stack(all_u, dim=1).cpu().numpy()

        # Create a DataFrame for this specific epoch's first batch sequence
        epoch_df = pd.DataFrame({
            "t": [i*dt for i in range(seq_len)],
            "y_t": y_t_stack[0, :, 0],
            "y_next": y_next_stack[0, :, 0],
            "u_control": u_stack[0, :, 0]
        })
        # Save individual sequence (one per epoch to avoid file bloat)
        epoch_df.to_csv(f"{seq_dir}/epoch_{epoch}_seq_0.csv", index=False)

        # Append to our master list (flattening the batch dimension for describe())
        # We sample only the first batch item to keep the master CSV size manageable
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

    # Save master CSV
    from src.sample.utils.general_utils import save_model, save_df_to_csv, plot_signals
    save_df_to_csv(master_df, dirname=dirname, filename="all_training_data_summary")

    # Save Loss
    df_loss = pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_history})
    save_df_to_csv(df_loss, dirname=dirname, filename="training_loss_history")
    save_model(model, dirname=dirname, model_config=model_config, filename="trained_controller")

    plot_signals(
        df_loss["epoch"].values, [df_loss["loss"].values],
        labels=["MSE Loss"], xlabel="Epoch", ylabel="Loss",
        title=f"Convergence ({plant.__class__.__name__})",
        dirname=dirname, filename="training_loss_plot"
    )

    return()

### 00
def train_controller(model, plant, epochs, seq_len, dt, model_config,device='cuda', dirname="name_directory"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    loss_history = []
    all_training_pairs = []

    print(f"Training Mamba on {plant.__class__.__name__}...")

    for epoch in range(epochs):
        # 1. Randomize parameters if the plant supports it
        if hasattr(plant, 'reset_trajectory'):
            plant.reset_trajectory()
            
        state = plant.get_initial_state()
        epoch_raw_history = []
        
        # --- Data Collection Phase ---
        for t_idx in range(seq_len):
            t = t_idx * dt
            u_signal = plant.generate_random_u(t)
            
            # Pass 't' to get_y to account for time-dependent Volume (V)
            y_t = plant.get_y(state, t) 
            
            # Step plant (ensure plant.step also uses/returns correct y_next)
            state_next, y_next = plant.step(state, u_signal, t, dt)
            
            record = {
                "t": t,
                "y_t": y_t,
                "y_t+dt": y_next,
                "u_t": u_signal,
                **plant.parse_state(state)
            }
            epoch_raw_history.append(record)
            state = state_next

        # Create DataFrame for this epoch
        df_epoch = pd.DataFrame(epoch_raw_history)
        
        # Optional: Save every X epochs to save disk space if training is long
        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_df_to_csv(
                df_epoch, 
                dirname=f"{dirname}/training_episodes", 
                filename=f"train_series_epoch_{epoch+1:04d}"
            )

        # --- Training Phase ---
        # 2. Prepare Inverse Mapping [y_t, y_t+dt] -> u_t
        inputs = df_epoch[["y_t", "y_t+dt"]].values
        targets = df_epoch["u_t"].values
        
        # Use torch.from_numpy and np.array to avoid the "slow tensor creation" warning
        x_tensor = torch.from_numpy(np.array([inputs])).float().to(device)
        y_target = torch.from_numpy(np.array([targets])).float().unsqueeze(-1).to(device)

        model.train()
        optimizer.zero_grad()
        
        # Forward pass through Mamba
        u_pred = model(x_tensor)
        
        loss = criterion(u_pred, y_target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        
     

        # Summary logging
        summary_slice = df_epoch[["y_t", "y_t+dt", "u_t"]].copy()
        summary_slice["epoch"] = epoch + 1
        all_training_pairs.append(summary_slice)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # --- Final Post-Training Steps ---
    df_loss = pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_history})
    df_train_full = pd.concat(all_training_pairs, ignore_index=True)
    
    save_df_to_csv(df_loss, dirname=dirname, filename="training_loss_history")
    save_df_to_csv(df_train_full, dirname=dirname, filename="training_data_full")
    save_df_to_csv(df_train_full.describe().reset_index(), dirname=dirname, filename="training_data_stats")

    plot_signals(
        df_loss["epoch"].values, [df_loss["loss"].values],
        labels=["MSE Loss"], xlabel="Epoch", ylabel="Loss",
        title=f"Convergence ({plant.__class__.__name__})",
        dirname=dirname, filename="training_loss_plot"
    )

    # --- Save the Model Weights ---
    save_model(model, dirname=dirname, model_config=model_config, filename="trained_controller")

    return()




import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

def GPUtrain_controller(model, plant, epochs, seq_len, dt, model_config, batch_size, device='cuda', dirname="name_directory"):
    """
    Vectorized training: 128 parallel simulations are performed per epoch.
    The model learns the inverse mapping: (y_t, y_t+1) -> u_t
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_history = []
    model.to(device)

    print(f"🚀 Training Mamba on {plant.__class__.__name__}")
    print(f"   Batch Size: {batch_size} | Device: {device}")

    start_time = time.time()

    for epoch in range(epochs):
        # 1. Diversity: randomize trajectories if the plant supports it
        if hasattr(plant, 'reset_trajectory'):
            plant.reset_trajectory()
            
        # 2. Reset starting state for the entire batch [batch_size, state_dim]
        state = plant.get_initial_state() 
        
        all_y_t = []
        all_y_next = []
        all_u = []

        # --- SIMULATION PHASE (Data Generation) ---
        # We use torch.no_grad() because we are just collecting data from the physics engine
        with torch.no_grad():
            for t_idx in range(seq_len):
                t = t_idx * dt
                
                # Generate random control inputs (Glucose feed) for exploration
                u_signal = torch.rand((batch_size, 1), device=device) * plant.U_MAX
                
                # Current output (Growth rate)
                y_t = plant.get_y(state, t) 
                
                # Physics Step
                state_next, y_next = plant.step(state, u_signal, t, dt)
                
                # Store
                all_y_t.append(y_t)
                all_y_next.append(y_next)
                all_u.append(u_signal)
                
                # Transition - DETACH to stop gradient flow through simulation time
                state = state_next.detach()

        # --- TRAINING PHASE (Gradient Descent) ---
        # Stack lists into [Batch, Seq, Dim]
        y_t_seq = torch.stack(all_y_t, dim=1)      
        y_next_seq = torch.stack(all_y_next, dim=1) 
        y_target = torch.stack(all_u, dim=1) 
        
        # Input to model: pair of current and resulting growth rate
        x_tensor = torch.cat([y_t_seq, y_next_seq], dim=-1) 

        model.train()
        optimizer.zero_grad()
        
        # Mamba predicts the 'u' that caused the change from y_t to y_next
        u_pred = model(x_tensor) 
        
        loss = criterion(u_pred, y_target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:04d}/{epochs} | Loss: {loss.item():.6f} | Time: {elapsed:.2f}s")

    # --- SAVE & PLOT ---
    df_loss = pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_history})
    
    # Import locally to avoid circular dependencies
    from src.sample.utils.general_utils import save_model, save_df_to_csv, plot_signals
    
    save_model(model, dirname=dirname, model_config=model_config, filename="trained_controller")
    save_df_to_csv(df_loss, dirname=dirname, filename="training_loss_history")

    plot_signals(
        df_loss["epoch"].values, [df_loss["loss"].values],
        labels=["MSE Loss"], xlabel="Epoch", ylabel="Loss",
        title=f"Convergence ({plant.__class__.__name__})",
        dirname=dirname, filename="training_loss_plot"
    )

    return loss_history

def load_controller(model_class, filepath, device='cuda'):
    """
    Loads a MambaInverseController and its configuration from a saved checkpoint.
    
    Args:
        model_class: The class name (MambaInverseController).
        filepath: Full path to the .pt file.
        device: 'cuda' or 'cpu'.
    """
    # 1. Load the checkpoint dictionary from disk
    # map_location ensures it works even if you move from GPU to CPU
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    
    # 2. Extract the saved config
    model_config = checkpoint['model_config']
    print(f"Loading model with config: {model_config}")
    
    # 3. Reconstruct the architecture using the saved hyperparams
    model = model_class(**model_config)
    
    # 4. Load the learned weights (state_dict) into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 5. Move to device and set to evaluation mode
    model.to(device)
    model.eval() 
    
    return model, model_config

def simulate_control(model, plant, reference_signal, duration, dt, device, dirname):
    model.eval()
    state = plant.get_initial_state()
    history = []
    current_context = []
    steps = int(duration / dt)

    print(f"Starting generic simulation for {duration}h...")

    for i in range(steps):
        t = i * dt
        y_meas = plant.get_y(state, t)
        
        # 1. Reference Logic
        r_t = reference_signal[i] if isinstance(reference_signal, np.ndarray) else reference_signal

        # 2. Normalization & Inference
        y_norm = y_meas / plant.Y_MAX
        r_norm = r_t / plant.Y_MAX

        step_input = np.array([[y_norm, r_norm]])
        current_context.append(step_input)
        
        # Maintain a sliding window or full history for Mamba context
        input_tensor = torch.tensor(np.array(current_context), dtype=torch.float32).transpose(0, 1).to(device)

        with torch.no_grad():
            u_norm = float(model(input_tensor)[0, -1, 0])
        
        u_phys = np.clip(u_norm * plant.U_MAX, 0, plant.U_MAX)

        # 3. Physics Step
        next_state, _ = plant.step(state, u_phys, t, dt)
        
        # 4. Data Collection
        # Combines general control signals with plant-specific internal states
        record = {
            "t": t, 
            "y": y_meas, 
            "r": r_t, 
            "u": u_phys, 
            "error": r_t - y_meas,
            **plant.parse_state(state) 
        }
        history.append(record)
        state = next_state

    # --- Data Processing & Saving ---
    df_sim = pd.DataFrame(history)
    save_df_to_csv(df_sim, dirname=dirname, filename=f"{plant.__class__.__name__}_sim_data")

    # --- Automated Plotting ---
    t_data = df_sim["t"].values
    
    # Use the plant's own config to decide what to plot
    if hasattr(plant, 'get_plot_config'):
        plot_configs = plant.get_plot_config()
    else:
        raise NotImplementedError("Plant class must implement get_plot_config() to specify plotting configuration.")
    
    for idx, config in enumerate(plot_configs):
        signals = [df_sim[col].values for col in config["cols"]]
        
        plot_signals(
            t_data, 
            signals,
            labels=config["labels"],
            xlabel="Time (h)",
            ylabel=config["ylabel"],
            dirname=dirname,
            filename=f"plot_{idx}_{config['title'].lower().replace(' ', '_')}"
        )

    print(f"Simulation finished. Data and {len(plot_configs)} plots saved to {dirname}.")

    def calculate_metrics(df):
        error = df["r"] - df["y"]
        u_diff = df["u"].diff().abs().sum() # Total Variation
        
        metrics = {
            "RMSE": np.sqrt((error**2).mean()),
            "MAE": error.abs().mean(),
            "Max_Error": error.abs().max(),
            "Control_Total_Variation": u_diff,
            "Mean_Control_Signal": df["u"].mean(),
            "Standard_Deviation_Error": error.std()
        }
        return metrics

    # In your simulate_control function:
    perf_metrics = calculate_metrics(df_sim)
    df_perf = pd.DataFrame([perf_metrics])
    save_df_to_csv(df_perf, dirname=dirname, filename="performance_metrics")
    return()

import torch
import numpy as np
import pandas as pd
from src.sample.utils.general_utils import save_df_to_csv, plot_signals

def GPUSimulateControl(model, plant, reference_signal, duration, dt, device, dirname):
    model.eval()
    # Ensure plant is on the correct device and get initial state [1, 2]
    state = plant.get_initial_state() 
    if state.shape[0] != 1:
        # If the plant was initialized with a large batch, we force it to 1 for simulation
        state = state[0:1] 
        
    steps = int(duration / dt)
    history = []
    
    # Pre-allocate a tensor for context: [Batch=1, Steps, Features=2]
    # This avoids the slow process of appending lists and converting to tensors in the loop
    context_tensor = torch.zeros((1, steps, 2), device=device)

    print(f"🚀 Starting GPU simulation for {duration}h...")

    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            y_meas = plant.get_y(state, t) # Returns [1, 1]
            
            # 1. Reference Logic
            # --- Replace the old Reference Logic ---
            # Check if reference_signal is indexable (array/tensor) or just a single value
            if isinstance(reference_signal, (np.ndarray, torch.Tensor)) and reference_signal.ndim > 0:
                r_t = reference_signal[i]
            else:
                # If it's a scalar tensor, float, or int, just take the value
                r_t = reference_signal

            # Ensure r_t is a float for calculations later
            if torch.is_tensor(r_t):
                r_t = r_t.item()
            
            # 2. Normalization
            y_norm = y_meas / plant.Y_MAX
            r_norm = r_t / plant.Y_MAX

            # 3. Update Context & Inference
            context_tensor[0, i, 0] = y_norm
            context_tensor[0, i, 1] = r_norm
            
            # Mamba processes the sequence up to current step 'i'
            # slice is [Batch, current_history, Features]
            u_out = model(context_tensor[:, :i+1, :]) 
            u_norm = u_out[0, -1, 0] # Get latest prediction
            
            # Scale and Clamp to physical pump limits
            u_phys = torch.clamp(u_norm * plant.U_MAX, 0, plant.U_MAX)

            # 4. Physics Step (returns Tensors)
            state_next, _ = plant.step(state, u_phys.unsqueeze(0), t, dt)
            
            # 5. Data Collection (Move to CPU only for history/logging)
            record = {
                "t": t, 
                "y": y_meas.item(), 
                "r": float(r_t), 
                "u": u_phys.item(), 
                "error": float(r_t) - y_meas.item()
            }
            
            # Add plant specific states (e.g. Biomass, Substrate) if available
            if hasattr(plant, 'parse_state'):
                record.update(plant.parse_state(state.cpu().numpy()[0]))
            
            history.append(record)
            state = state_next

    # --- Data Processing ---
    df_sim = pd.DataFrame(history)
    save_df_to_csv(df_sim, dirname=dirname, filename=f"{plant.__class__.__name__}_sim_data")

    # --- Metrics & Plotting (Logic remains similar to your original) ---
    def calculate_metrics(df):
        error = df["r"] - df["y"]
        return {
            "RMSE": np.sqrt((error**2).mean()),
            "MAE": error.abs().mean(),
            "Control_TV": df["u"].diff().abs().sum()
        }

    perf_metrics = calculate_metrics(df_sim)
    save_df_to_csv(pd.DataFrame([perf_metrics]), dirname=dirname, filename="performance_metrics")

    # Plot using plant config
    if hasattr(plant, 'get_plot_config'):
        for idx, config in enumerate(plant.get_plot_config()):
            signals = [df_sim[col].values for col in config["cols"]]
            plot_signals(df_sim["t"].values, signals, labels=config["labels"], 
                         xlabel="Time (h)", ylabel=config["ylabel"], dirname=dirname,
                         filename=f"plot_{idx}_{config['title'].lower().replace(' ', '_')}")

    print(f"✅ Simulation complete. Metrics: RMSE={perf_metrics['RMSE']:.5f}")
    return df_sim


import torch
import numpy as np
import random
import os

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


