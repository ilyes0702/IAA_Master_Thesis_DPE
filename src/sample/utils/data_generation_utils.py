import torch
import numpy as np  
import pandas as pd

from src.sample.utils.plotting_utils import plot_signals
from src.sample.utils.saving_utils import save_df_to_csv, save_training_dataset

def generate_training_batch(plant, hyperparam_config):
    """Simulates parallel trajectories and returns training tensors."""
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    
    seq_len = sig_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = train_cfg["batch_size"]
    delta_steps = train_cfg["delay_steps"]
    device = train_cfg["device"]

    # Reset plant and get initial states
    if hasattr(plant, 'reset_trajectory'):
        plant.reset_trajectory()
    
    state = plant.get_initial_state(batch_size)
    all_y_t, all_y_next, all_u = [], [], []

    # Simulation loop
    for t_idx in range(seq_len - delta_steps):
        t = t_idx * dt
        u_signal = plant.get_u_at_step(t_idx)
        y_t = plant.get_y(state, t)
        
        # Forward simulate to find the target state
        temp_state = state
        for _ in range(delta_steps):
            temp_state, _ = plant.step(temp_state, u_signal, t, dt)
        
        y_delta = plant.get_y(temp_state, t + delta_steps * dt)
        
        all_y_t.append(y_t)
        all_y_next.append(y_delta)
        all_u.append(u_signal)
        
        # Advance the actual plant
        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # Construct Tensors [Batch, Seq, Dim]
    x_tensor = torch.cat([torch.stack(all_y_t, dim=1), 
                          torch.stack(all_y_next, dim=1)], dim=-1).to(device)
    y_target = torch.stack(all_u, dim=1).to(device)
    
    return x_tensor, y_target


def generate_and_save_dataset(plant, hyperparam_config, num_batches, dirname):
    """
    Simulates physics, saves CSVs/Plots for EVERY individual trajectory, 
    and returns the final tensors for training.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    
    dt = sig_cfg["dt"]
    batch_size = train_cfg["batch_size"]
    delta_steps = train_cfg["delay_steps"]
    
    all_batches_x = []
    all_batches_y = []

    print(f"📂 Generating {num_batches} batches ({num_batches * batch_size} total sequences)...")

    for b_idx in range(num_batches):
        # 1. RUN PARALLEL SIMULATION
        # (This uses the generator logic we discussed to get parallel trajectories)
        x_tensor, y_target = generate_training_batch(plant, hyperparam_config)
        
        

        # Extract D_centers for this batch (Shape: [batch_size, 1])
        # We move to CPU immediately for logging
        batch_d_centers = plant.current_D_center.cpu().numpy()

        # Move to CPU for logging and saving
        x_np = x_tensor.cpu().numpy() # Shape: [batch, seq_len, 2]
        y_np = y_target.cpu().numpy() # Shape: [batch, seq_len, 1]
        
        # 2. LOOP THROUGH EACH SEQUENCE IN THE BATCH
        for s_idx in range(batch_size):
            # Extract individual signals
            y_t_plot = x_np[s_idx, :, 0]
            y_next_plot = x_np[s_idx, :, 1]
            u_signal_plot = y_np[s_idx, :, 0]

            # Get the specific D_center for THIS sequence
            d_center_val = float(batch_d_centers[s_idx])
            
            num_samples = len(u_signal_plot)
            time_axis = np.arange(num_samples) * dt
            d_center_line = np.full_like(time_axis, d_center_val)
            # Identify current D_center for this specific plant 
            # (Assuming plant stores them or they are constant across batch)
            

            # --- SAVE INDIVIDUAL CSV ---
            seq_df = pd.DataFrame({
                "t": time_axis,
                "y_t": y_t_plot,
                "y_next": y_next_plot,
                "u_control": u_signal_plot,
                "D_center": d_center_val
            })
            
            # unique filename: e.g., batch_0_seq_5.csv
            filename_base = f"batch_{b_idx}_seq_{s_idx}"
            save_df_to_csv(seq_df, 
                           dirname=dirname + "/dataset_logs", 
                           filename=f"{filename_base}.csv")

            # --- SAVE INDIVIDUAL PLOT ---
            # plot_signals(
            #     time_axis,
            #     [u_signal_plot, y_t_plot, y_next_plot, d_center_line],
            #     labels=["u_control", "y_t", "y_next", "D_center"],
            #     xlabel="Time",
            #     ylabel="Signal",
            #     title=f"Dataset Sample: {filename_base}",
            #     dirname=dirname + "/dataset_plots",
            #     filename=f"{filename_base}_plot"
            # )

        all_batches_x.append(x_tensor.cpu())
        all_batches_y.append(y_target.cpu())

    # Final Save of the raw tensors for the Trainer to load later
    data_to_save = {"x": all_batches_x, "y": all_batches_y}
    save_training_dataset(data_to_save, dirname=dirname)


