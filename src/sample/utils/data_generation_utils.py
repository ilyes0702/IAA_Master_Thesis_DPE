import os
import numpy as np
import torch
import pandas as pd
from src.sample.utils.saving_utils import save_df_to_csv, save_training_dataset
from src.sample.utils.plotting_utils import plot_signals
import copy

#=== FUNCTION TO GENERATE TRAINING SIGNALS ACCORDING TO CANADAY ===#
def generate_canaday_signals(hyperparam_config):
    """
    Generate smooth, band-limited control signals using the Canaday/RC methodology.

    This function creates physically-bounded control signals by applying a low-pass
    filter in the frequency domain to smooth noise, then scaling and shifting the
    result to the plant's operating bounds.

    The process follows five steps:
    1. Uniform sampling: Generate random values in [-1, 1]
    2. Fourier Transform: Convert to frequency domain using FFT
    3. Frequency cutoff: Zero out frequencies above 1/lambda (low-pass filter)
    4. Inverse Fourier Transform: Convert back to time domain using iFFT
    5. Scaling & shifting: Normalize to [-p, p] then shift to [D_center_min, D_center_max]

    Args:
        hyperparam_config: Dictionary with nested 'signal', 'train', and 'plant' configs:
            - signal['seq_len']: Sequence length (number of timesteps)
            - signal['dt']: Sampling interval
            - signal['lambd']: Bandwidth parameter (lambda); cutoff = 1/lambda
            - signal['p']: Half-width of the normalized signal range
            - train['batch_size']: Number of signals to generate
            - train['device']: torch device (cpu or cuda)
            - plant['D_center_min']: Minimum center offset
            - plant['D_center_max']: Maximum center offset

    Returns:
        u_buffer: torch.Tensor of shape [batch_size, seq_len] with control signals
        D_center: torch.Tensor of shape [batch_size, 1] with randomly sampled center offsets
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    
    batch_size = train_cfg["batch_size"]
    seq_len = sig_cfg["seq_len"]
    device = train_cfg["device"]
    
    # Step 1: Sample values from a uniform distribution [-1, 1]
    raw = torch.rand((batch_size, seq_len), device=device) * 2 - 1
    
    # Step 2: Fourier-transform to frequency domain
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=sig_cfg["dt"])
    
    # Step 3: Drop frequencies above 1/lambda
    cutoff = 1.0 / sig_cfg["lambd"]
    fft_sig[:, freqs > cutoff] = 0
    
    # Step 4: Inverse-Fourier-transform
    v_train = torch.fft.irfft(fft_sig, n=seq_len, dim=1)
    
    # Step 5: Normalize and Scale to [-p, p]
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1
    
    # Read configurable target limits instead of using hardcoded numbers
    c_min = plant_cfg["D_center_min"]
    c_max = plant_cfg["D_center_max"]
    
    D_center = torch.rand((batch_size, 1), device=device) * (c_max - c_min) + c_min
    u_buffer = D_center + (v_norm * sig_cfg["p"])
    
    return u_buffer, D_center


#=== FUNCTION TO GENERATE TRAINING BATCH ===#
def generate_training_batch(plant, hyperparam_config):
    """
    Generate a single training batch by simulating the plant with control signals.

    The function creates control signals using the Canaday generator long enough to
    support a slicing window that may include a lookahead (delay_steps). It then
    simulates the plant for the extended horizon and slices out sequences of
    length `seq_len` returning model inputs and targets.

    Args:
        plant: Plant object implementing get_initial_state(batch_size), get_y(state, t),
               and step(state, u, t, dt) methods.
        hyperparam_config: Dictionary containing nested 'signal' and 'train' configs
                           with keys used below (seq_len, dt, batch_size, delay_steps, device).

    Returns:
        x_tensor: torch.Tensor of shape [batch_size, seq_len, features] containing
                  stacked [y_t, y_next] for each timestep.
        y_target: torch.Tensor of shape [batch_size, seq_len, 1] containing the control
                  signals (u) corresponding to each timestep.
        D_center: torch.Tensor of shape [batch_size, 1] containing the center offsets
                  used when generating the control signals.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    
    seq_len = sig_cfg["seq_len"]
    dt = sig_cfg["dt"]
    batch_size = train_cfg["batch_size"]
    delta_steps = train_cfg["delay_steps"]
    device = train_cfg["device"]

    state = plant.get_initial_state(batch_size)
    
    # 1. 🔥 FIX: Generate signals long enough to support your slicing window
    # Extend control signal generation by delta_steps
    extended_config = copy.deepcopy(hyperparam_config)
    extended_config["signal"]["seq_len"] = seq_len + delta_steps
    u_buffer, D_center = generate_canaday_signals(extended_config)
    
    raw_y_history = []
    raw_u_history = []

    # 2. 🔥 FIX: Run the simulation loop for the full extended horizon
    total_simulation_steps = seq_len + delta_steps
    for t_idx in range(total_simulation_steps):
        t = t_idx * dt
        u_signal = u_buffer[:, t_idx].unsqueeze(1)
        y_t = plant.get_y(state, t)
        
        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)
        
        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    # 3. 🔥 FIX: Slice out clean, full-length sequences of length (seq_len)
    all_y_t = raw_y_history[:seq_len]
    all_y_next = raw_y_history[delta_steps : seq_len + delta_steps] 
    all_u = raw_u_history[:seq_len]     

    # Construct model tensors: Shapes will now be exactly [Batch, seq_len, Feature]
    x_tensor = torch.cat([
        torch.stack(all_y_t, dim=1), 
        torch.stack(all_y_next, dim=1)
    ], dim=-1).to(device)
    
    y_target = torch.stack(all_u, dim=1).to(device)
    
    return x_tensor, y_target, D_center


import os
import torch
import numpy as np
import pandas as pd

def generate_and_save_dataset(plant, hyperparam_config, dirname, show_plots=False):
    """Generate and save training dataset with all sequences in parallel.
    
    Generates a batch of training sequences simultaneously by simulating the plant
    with multiple input signals in parallel. Each sequence consists of states,
    control inputs, and outputs saved to individual CSV files with visualization support.
    
    Args:
        plant: Plant dynamics model with step() method for state evolution.
        hyperparam_config (dict): Configuration dict with keys:
            - "signal": Signal parameters (dt, freq_min, freq_max, etc.)
            - "train": Training parameters (batch_size, seq_len, delta_steps, etc.)
        dirname (str): Root directory where dataset logs will be saved.
        show_plots (bool, optional): If True, generate and display plots for each sequence.
            Defaults to False.
    
    Returns:
        None. Saves CSV files to dirname/dataset_logs/ with one file per sequence.
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    dt = sig_cfg["dt"]
    
    # This now represents the TOTAL number of sequences generated simultaneously
    total_sequences = train_cfg["batch_size"] 
    
    logs_dir = os.path.join(dirname, "dataset_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1. 🔥 SIMULTANEOUS GENERATION: No loop. One shot.
    print(f"🚀 Simulating {total_sequences} sequences simultaneously in parallel...")
    x_tensor, y_target, batch_d_centers = generate_training_batch(plant, hyperparam_config)
    
    print(f"✅ Simulation complete. Matrix shapes: x={x_tensor.shape}, y={y_target.shape}")
    
    # Move entire blocks to CPU/NumPy at once for disk writing
    batch_d_centers_np = batch_d_centers.cpu().numpy()
    x_np = x_tensor.cpu().numpy() 
    y_np = y_target.cpu().numpy() 
    
    # 2. File Saving Loop (Only needed because saving to individual CSVs is an I/O operation)
    print(f"💾 Saving {total_sequences} sequence files to disk...")
    for s_idx in range(total_sequences):
        y_t_plot = x_np[s_idx, :, 0]
        y_next_plot = x_np[s_idx, :, 1]
        u_signal_plot = y_np[s_idx, :, 0]
        d_center_val = float(batch_d_centers_np[s_idx])
        
        num_samples = len(u_signal_plot)
        time_axis = np.arange(num_samples) * dt

        seq_df = pd.DataFrame({
            "t": time_axis,
            "y_t": y_t_plot,
            "y_next": y_next_plot,
            "u_control": u_signal_plot,
            "D_center": d_center_val
        })
        
        filename_base = f"sequence_{s_idx}.csv"
        save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)
        
        if show_plots:
            plot_signals(
                time_axis,
                [u_signal_plot, y_t_plot, y_next_plot],
                labels=["u_control", "y_t", "y_next"],
                xlabel="Time", ylabel="Signal",
                title=f"Dataset Sample: {filename_base}",
                dirname=os.path.join(dirname, "dataset_plots"),
                filename=f"{filename_base}_plot.png"
            )

    # 3. 📊 Macro-level dataset statistics computed instantaneously via NumPy flattening
    print("📊 Compiling macro-level dataset statistics...")
    all_y_t = x_np[:, :, 0].flatten()
    all_y_next = x_np[:, :, 1].flatten()
    all_u_control = y_np[:, :, 0].flatten()

    stats_data = {
        "Metric": ["Mean", "Std_Dev", "Min", "25%", "50%_Median", "75%", "Max"],
        "y_t (Plant Out)": [
            np.mean(all_y_t), np.std(all_y_t), np.min(all_y_t),
            np.percentile(all_y_t, 25), np.percentile(all_y_t, 50), np.percentile(all_y_t, 75), np.max(all_y_t)
        ],
        "y_next (Plant Out+1)": [
            np.mean(all_y_next), np.std(all_y_next), np.min(all_y_next),
            np.percentile(all_y_next, 25), np.percentile(all_y_next, 50), np.percentile(all_y_next, 75), np.max(all_y_next)
        ],
        "u_control (Targets)": [
            np.mean(all_u_control), np.std(all_u_control), np.min(all_u_control),
            np.percentile(all_u_control, 25), np.percentile(all_u_control, 50), np.percentile(all_u_control, 75), np.max(all_u_control)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    save_df_to_csv(stats_df, dirname=dirname, filename="dataset_global_statistics.csv")
    
    # 4. Save the full, aggregated training dataset
    data_to_save = {
        "x": x_tensor.cpu(),  
        "y": y_target.cpu()   
    }
    save_training_dataset(data_to_save, dirname=dirname)
    print(f"🎉 Done! Dataset successfully generated in one single batch pass.")



def generate_exact_sequence_dataset(plant, hyperparam_config, dirname, total_sequences, show_plots=False):
    """Generates and saves an exact number of total sequences without nested batch loops."""
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    dt = sig_cfg["dt"]
    total_sequences = train_cfg["num_sequences"]  # Total number of sequences to generate across all batches
    
    
    
    all_x = []
    all_y = []
    
    logs_dir = os.path.join(dirname, "dataset_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    sequences_saved = 0
    print(f"📂 Target: Generating exactly {total_sequences} sequences.")
    
    while sequences_saved < total_sequences:
        # 1. Generate a batch of parallel sequences
        x_tensor, y_target, batch_d_centers = generate_training_batch(plant, hyperparam_config)
        
        # Move to CPU / NumPy for disk-writing operations
        x_np = x_tensor.cpu().numpy()
        y_np = y_target.cpu().numpy()
        d_centers_np = batch_d_centers.cpu().numpy()
        
        # 2. Iterate through the generated batch linearly
        num_generated = x_np.shape[0]
        for idx in range(num_generated):
            if sequences_saved >= total_sequences:
                break  # Stop immediately if we reached our exact target limit
                
            y_t_plot = x_np[idx, :, 0]
            y_next_plot = x_np[idx, :, 1]
            u_signal_plot = y_np[idx, :, 0]
            d_center_val = float(d_centers_np[idx])
            
            time_axis = np.arange(len(u_signal_plot)) * dt
            
            # Save the individual sequence CSV
            seq_df = pd.DataFrame({
                "t": time_axis, "y_t": y_t_plot, "y_next": y_next_plot, 
                "u_control": u_signal_plot, "D_center": d_center_val
            })
            
            filename_base = f"sequence_{sequences_saved}.csv"
            save_df_to_csv(seq_df, dirname=logs_dir, filename=filename_base)
            
            if show_plots:
                plot_signals(
                    time_axis, [u_signal_plot, y_t_plot, y_next_plot],
                    labels=["u_control", "y_t", "y_next"],
                    xlabel="Time", ylabel="Signal",
                    title=f"Dataset Sample: {filename_base}",
                    dirname=os.path.join(dirname, "dataset_plots"),
                    filename=f"{filename_base}_plot.png"
                )
                
            # Keep track of individual sequences to slice the global tensors later
            all_x.append(x_tensor[idx].cpu())
            all_y.append(y_target[idx].cpu())
            
            sequences_saved += 1
            
        print(f"📈 Progress: {sequences_saved}/{total_sequences} sequences saved.")

    # 3. Compile precise global statistics and save final dataset
    print("📊 Compiling macro-level dataset statistics...")
    compiled_x = torch.stack(all_x, dim=0).numpy()
    compiled_y = torch.stack(all_y, dim=0).numpy()
    
    all_y_t = compiled_x[:, :, 0].flatten()
    all_y_next = compiled_x[:, :, 1].flatten()
    all_u_control = compiled_y[:, :, 0].flatten()

    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Std_Dev", "Min", "25%", "50%_Median", "75%", "Max"],
        "y_t (Plant Out)": [np.mean(all_y_t), np.std(all_y_t), np.min(all_y_t), np.percentile(all_y_t, 25), np.percentile(all_y_t, 50), np.percentile(all_y_t, 75), np.max(all_y_t)],
        "y_next (Plant Out+1)": [np.mean(all_y_next), np.std(all_y_next), np.min(all_y_next), np.percentile(all_y_next, 25), np.percentile(all_y_next, 50), np.percentile(all_y_next, 75), np.max(all_y_next)],
        "u_control (Targets)": [np.mean(all_u_control), np.std(all_u_control), np.min(all_u_control), np.percentile(all_u_control, 25), np.percentile(all_u_control, 50), np.percentile(all_u_control, 75), np.max(all_u_control)]
    })
    save_df_to_csv(stats_df, dirname=dirname, filename="dataset_global_statistics.csv")
    
    data_to_save = {
        "x": torch.stack(all_x, dim=0),  
        "y": torch.stack(all_y, dim=0)   
    }
    save_training_dataset(data_to_save, dirname=dirname)
    print(f"✅ Success! Dataset saved with exactly {sequences_saved} entries.")