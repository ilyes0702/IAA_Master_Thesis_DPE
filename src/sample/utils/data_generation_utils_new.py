import os
import numpy as np
import torch
import pandas as pd

from src.sample.utils.saving_utils import save_df_to_csv, save_training_dataset
from src.sample.utils.plotting_utils import plot_signals
import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")

#=== FUNCTION TO GENERATE TRAINING DATA ===#
def generate_training_batch(plant, hyperparam_config):
    """
    Simulates parallel trajectories sequentially and processes time slices to 
    extract aligned historical tracking output pairs [y(t), y(t + Delta)].
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    
    seq_len = sig_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = train_cfg["batch_size"]
    delta_steps = train_cfg["delay_steps"]
    device = train_cfg["device"]

    # Reset plant state history tracking
    if hasattr(plant, 'reset_trajectory'):
        plant.reset_trajectory()
    
    state = plant.get_initial_state(batch_size)
    
    # Pre-allocate trajectory storage buffers for the true sequence length
    raw_y_history = []
    raw_u_history = []

    # 1. RUN THE FULL PLANT TRAJECTORY LINEARLY (No nested look-ahead loops)
    for t_idx in range(seq_len):
        t = t_idx * dt
        u_signal = plant.get_u_at_step(t_idx) # Shape matches your plant design requirements
        y_t = plant.get_y(state, t)
        
        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)
        
        # Advance true plant state
        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # 2. ALIGN TIME HORIZONS VIA INDEX SLICING
    # We truncate the sequence length down to (seq_len - delta_steps)
    valid_steps = seq_len - delta_steps
    
    all_y_t = raw_y_history[:valid_steps]
    all_y_next = raw_y_history[delta_steps:] # Aligned exactly to t + delta_steps
    all_u = raw_u_history[:valid_steps]      # The control input applied at time t to cause y_next

    # 3. CONSTRUCT MODEL TENSORS [Batch, Seq, Dim]
    x_tensor = torch.cat([
        torch.stack(all_y_t, dim=1), 
        torch.stack(all_y_next, dim=1)
    ], dim=-1).to(device)
    
    y_target = torch.stack(all_u, dim=1).to(device)
    
    return x_tensor, y_target


def generate_and_save_dataset(plant, hyperparam_config, num_batches, dirname, show_plots=False):
    """
    Simulates physics, saves CSVs for EVERY individual trajectory sequentially, 
    and returns the final tensors for training.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    
    dt = sig_cfg["dt"]
    batch_size = train_cfg["batch_size"]
    
    all_batches_x = []
    all_batches_y = []

    # Safe cross-platform folder mapping
    logs_dir = os.path.join(dirname, "dataset_logs")
    os.makedirs(logs_dir, exist_ok=True)

    print(f"📂 Generating {num_batches} batches ({num_batches * batch_size} total sequences)...")

    for b_idx in range(num_batches):
        # 1. RUN CORRECTED SIMULATION (Fixed time-alignment)
        x_tensor, y_target = generate_training_batch(plant, hyperparam_config)
        print(f"Batch {b_idx+1}/{num_batches} generated. Tensor shapes:")
        print("x_tensor.shape:", x_tensor.shape)
        print("y_target.shape:", y_target.shape)
        
        # Extract D_centers for this batch (Shape: [batch_size, 1])
        batch_d_centers = plant.current_D_center.cpu().numpy()

        # Move tensors to CPU memory for exporting operations
        x_np = x_tensor.cpu().numpy() 
        y_np = y_target.cpu().numpy() 
        
        # 2. EXPORT COMPONENT TRAJECTORIES LOGS INDIVIDUALLY
        for s_idx in range(batch_size):
            y_t_plot = x_np[s_idx, :, 0]
            y_next_plot = x_np[s_idx, :, 1]
            u_signal_plot = y_np[s_idx, :, 0]
            d_center_val = float(batch_d_centers[s_idx])
            
            num_samples = len(u_signal_plot)
            time_axis = np.arange(num_samples) * dt

            # Build and commit the dataframe to disk
            seq_df = pd.DataFrame({
                "t": time_axis,
                "y_t": y_t_plot,
                "y_next": y_next_plot,
                "u_control": u_signal_plot,
                "D_center": d_center_val
            })
            
            filename_base = f"batch_{b_idx}_seq_{s_idx}.csv"
            save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)
            
            if show_plots:
                if s_idx % 1 == 0:
                    plot_signals(
                        time_axis,
                        [u_signal_plot, y_t_plot, y_next_plot],
                        labels=["u_control", "y_t", "y_next"],
                        xlabel="Time",
                        ylabel="Signal",
                        title=f"Dataset Sample: {filename_base}",
                        dirname=dirname + "/dataset_plots",
                        filename=f"{filename_base}_plot_{s_idx}.png"
                    )

        all_batches_x.append(x_tensor.cpu())
        all_batches_y.append(y_target.cpu())

    # =========================================================================
    # --- COMPUTE COMPREHENSIVE DATASET POPULATION STATISTICS ---
    # =========================================================================
    print("📊 Compiling macro-level dataset statistics...")
    
    compiled_x = torch.cat(all_batches_x, dim=0).numpy()
    compiled_y = torch.cat(all_batches_y, dim=0).numpy()
    
    all_y_t = compiled_x[:, :, 0].flatten()
    all_y_next = compiled_x[:, :, 1].flatten()
    all_u_control = compiled_y[:, :, 0].flatten()

    stats_data = {
        "Metric": ["Mean", "Std_Dev", "Min", "25%", "50%_Median", "75%", "Max"],
        
        "y_t (Plant Out)": [
            np.mean(all_y_t), np.std(all_y_t), np.min(all_y_t),
            np.percentile(all_y_t, 25), np.percentile(all_y_t, 50), np.percentile(all_y_t, 75),
            np.max(all_y_t)
        ],
        
        "y_next (Plant Out+1)": [
            np.mean(all_y_next), np.std(all_y_next), np.min(all_y_next),
            np.percentile(all_y_next, 25), np.percentile(all_y_next, 50), np.percentile(all_y_next, 75),
            np.max(all_y_next)
        ],
        
        "u_control (Targets)": [
            np.mean(all_u_control), np.std(all_u_control), np.min(all_u_control),
            np.percentile(all_u_control, 25), np.percentile(all_u_control, 50), np.percentile(all_u_control, 75),
            np.max(all_u_control)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    save_df_to_csv(stats_df, dirname=dirname, filename="dataset_global_statistics.csv")
    print("📝 Global dataset statistical breakdown saved successfully.")
    
    # Save the aggregated structured raw tensors dictionary to disk
    # WITH THIS:
    data_to_save = {
    "x": torch.cat(all_batches_x, dim=0),  # Shape: (total_sequences, seq_len, 2)
    "y": torch.cat(all_batches_y, dim=0)   # Shape: (total_sequences, seq_len, 1)
    }
    save_training_dataset(data_to_save, dirname=dirname)

    