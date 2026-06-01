import os
import copy
import numpy as np
import torch
import pandas as pd
from src.sample.utils.saving_utils import save_df_to_csv, save_training_dataset
from src.sample.utils.plotting_utils import plot_signals
from Archive.data_generation_utils_siso_old import generate_canaday_signals

# =====================================================================
# 1. MIMO Canaday Signal Generation
# =====================================================================
def generate_canaday_signals(hyperparam_config):
    """
    Generate MIMO control signals (u1, u2, ...) using Canaday's method.
    Each control input is generated independently.
    """
    train_cfg = hyperparam_config["train"]
    mamba_cfg = hyperparam_config["mamba"]

    batch_size = int(train_cfg["batch_size"])
    output_dim = mamba_cfg.get("output_dim", 2)  # Number of control inputs (u1, u2, ...)

    u_buffer = []
    D_center_list = []

    for _ in range(output_dim):
        # Generate a single control signal (SISO case)
        u_single, D_center = generate_canaday_signals(hyperparam_config)
        # Ensure u_single is 3D: [batch_size, seq_len, 1]
        u_single = u_single.unsqueeze(-1)  
        u_buffer.append(u_single)
        D_center_list.append(D_center)

    # Stack to get [batch_size, seq_len, output_dim]
    u_buffer = torch.cat(u_buffer, dim=-1)  
    D_center = torch.stack(D_center_list, dim=-1)  # Shape: [batch_size, output_dim]

    return u_buffer, D_center


# =====================================================================
# 2. FIXED: Fully Clean MIMO Training Batch (No Delays)
# =====================================================================
def generate_training_batch(plant, hyperparam_config):
    """
    Generate a clean MIMO training batch without any look-back delay tensors.
    Slices perfectly to match only y_t and y_next.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]

    seq_len = int(sig_cfg["seq_len"])
    dt = sig_cfg["dt"]
    batch_size = int(train_cfg["batch_size"])
    delta_steps = int(train_cfg.get("delay_steps", 1)) # Step lookahead window length
    device = train_cfg["device"]

    # Initialize plant state
    state = plant.get_initial_state(batch_size)

    # Extend seq_len long enough to pull clean future states (y_next)
    extended_config = copy.deepcopy(hyperparam_config)
    extended_config["signal"]["seq_len"] = seq_len + delta_steps
    u_buffer, D_center = generate_canaday_signals(extended_config)

    raw_y_history = []
    raw_u_history = []

    total_simulation_steps = seq_len + delta_steps
    for t_idx in range(total_simulation_steps):
        t = t_idx * dt
        u_signal = u_buffer[:, t_idx, :]  # Shape: [batch_size, output_dim]
        y_t = plant.get_y(state, t)       # Shape: [batch_size, input_dim]

        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)

        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # Slice out exact pairs without historical delay padding
    all_y_t = raw_y_history[:seq_len]  
    all_y_next = raw_y_history[delta_steps : seq_len + delta_steps]  
    all_u = raw_u_history[:seq_len]  

    # Construct the training matrix tensors
    # x_tensor shape: [batch_size, seq_len, input_dim * 2] (holding [y_t, y_next])
    x_tensor = torch.cat([
        torch.stack(all_y_t, dim=1),  
        torch.stack(all_y_next, dim=1)
    ], dim=-1).to(device)

    # y_target shape: [batch_size, seq_len, output_dim]
    y_target = torch.stack(all_u, dim=1).to(device)

    return x_tensor, y_target, D_center


# =====================================================================
# 3. Clean Dataset Compilation and Disk Exporter
# =====================================================================
def generate_and_save_dataset(
    plant,
    hyperparam_config,
    dirname,
    show_plots=False,
    save_logs=False
):
    """
    Generate and save a clean MIMO dataset tracking only current and future plant outputs.
    Removes historical delayed state inputs entirely.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    mamba_cfg = hyperparam_config["mamba"]
    dt = sig_cfg["dt"]
    
    # Automatically infer dimensions from config setup
    input_dim = mamba_cfg.get("input_dim", 2)   # (y1, y2)
    output_dim = mamba_cfg.get("output_dim", 2) # (u1, u2)

    total_sequences = int(train_cfg["batch_size"])
    logs_dir = os.path.join(dirname, "dataset_logs")
    plots_dir = os.path.join(dirname, "dataset_plots")

    print(f"🚀 Simulating {total_sequences} parallel MIMO sequences...")
    # Fits flawlessly now that generate_training_batch matches
    x_tensor, y_target, batch_d_centers = generate_training_batch(plant, hyperparam_config)

    print(f"✅ Simulation complete. Shapes: x={x_tensor.shape}, y={y_target.shape}")

    x_np = x_tensor.cpu().numpy()  
    y_np = y_target.cpu().numpy()  
    batch_d_centers_np = batch_d_centers.cpu().numpy()  

    # Save individual sequences
    for s_idx in range(total_sequences):
        y_t = x_np[s_idx, :, :input_dim]          
        y_next = x_np[s_idx, :, input_dim:]       
        u = y_np[s_idx]                           

        time_axis = np.arange(len(u)) * dt

        columns = ["t"]
        values = [time_axis]

        # Dynamically append columns without hardcoded loops
        for i in range(input_dim):
            columns.append(f"y_{i+1}_t")
            values.append(y_t[:, i])
        for i in range(input_dim):
            columns.append(f"y_{i+1}_next")
            values.append(y_next[:, i])

        for i in range(output_dim):
            columns.append(f"u_{i+1}")
            values.append(u[:, i])

        d_center = np.squeeze(batch_d_centers_np[s_idx])
        for i in range(output_dim):
            columns.append(f"D_center_u_{i+1}")
            d_center_value = float(d_center[i]) if output_dim > 1 else float(d_center)
            values.append(np.full(len(time_axis), d_center_value))

        seq_df = pd.DataFrame({col: val for col, val in zip(columns, values)})
        filename_base = f"sequence_{s_idx}.csv"
        
        if save_logs:
            save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)

        if show_plots:
            signals_to_plot = []
            labels_to_plot = []

            for i in range(input_dim):
                signals_to_plot.append(y_t[:, i])
                labels_to_plot.append(f"y_{i+1}_t")
            for i in range(input_dim):
                signals_to_plot.append(y_next[:, i])
                labels_to_plot.append(f"y_{i+1}_next")
            for i in range(output_dim):
                signals_to_plot.append(u[:, i])
                labels_to_plot.append(f"u_{i+1}")

            plot_signals(
                t=time_axis,
                signals=signals_to_plot,
                labels=labels_to_plot,
                xlabel="Time [h]",
                ylabel="Signal Profile",
                title=f"MIMO Sequence {s_idx} Profile",
                dirname=plots_dir,
                filename=f"{filename_base}_plot.png",
                show=True
            )

    # 📊 Compile Clean Global Statistics
    print("📊 Compiling MIMO macro-dataset statistics...")
    stats_data = {"Metric": ["Mean", "Std_Dev", "Min", "25%", "50%", "75%", "Max"]}

    for i in range(input_dim):
        slice_y_t = x_np[:, :, i]
        stats_data[f"y_{i+1}_t"] = [
            np.mean(slice_y_t), np.std(slice_y_t), np.min(slice_y_t),
            np.percentile(slice_y_t, 25), np.percentile(slice_y_t, 50),
            np.percentile(slice_y_t, 75), np.max(slice_y_t)
        ]
        
    for i in range(input_dim):
        slice_y_next = x_np[:, :, input_dim + i]
        stats_data[f"y_{i+1}_next"] = [
            np.mean(slice_y_next), np.std(slice_y_next), np.min(slice_y_next),
            np.percentile(slice_y_next, 25), np.percentile(slice_y_next, 50),
            np.percentile(slice_y_next, 75), np.max(slice_y_next)
        ]

    for i in range(output_dim):
        slice_u = y_np[:, :, i]
        stats_data[f"u_{i+1}"] = [
            np.mean(slice_u), np.std(slice_u), np.min(slice_u),
            np.percentile(slice_u, 25), np.percentile(slice_u, 50),
            np.percentile(slice_u, 75), np.max(slice_u)
        ]

    stats_df = pd.DataFrame(stats_data)
    save_df_to_csv(stats_df, dirname=dirname, filename="mimo_dataset_global_statistics.csv")

    # Package unified training dataset structures
    data_to_save = {
        "x": x_tensor.cpu(),
        "y": y_target.cpu()
    }
    save_training_dataset(data_to_save, dirname=dirname)
    print(f"🎉 Clean tracking MIMO dataset generated successfully without historical delays.")